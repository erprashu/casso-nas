"""CASSO on top of the official NAS-Bench-201 search code (D-X-Y/AutoDL-Projects,
xautodl), for NAS-Bench-201.

Everything a baseline depends on comes from the official code and protocol:
the TinyNetworkGDAS supernet (non-affine BatchNorm, NAS-Bench-201 cell), the
official train/valid split (configs/nas-benchmark/cifar-split.txt), the GDAS
optimizer config (SGD-Nesterov 0.025, cosine, wd 5e-4, batch 64), the linear
Gumbel temperature schedule 10 -> 0.1, the arch optimizer (Adam 3e-4, wd 1e-3),
and gradient clipping at 5.

Modes:
  --sampler gdas     official GDAS: Gumbel-softmax single path, phi trained on
                     the valid half.
  --sampler uniform  uniform random single path per step (random-NAS / SPOS),
                     phi never used.
  --method vanilla   plain cross-entropy on the sampled path (the baselines).
  --method casso     CASSO: sensitivity scores (Eq. 7-8), streaming SFL archive
                     (Alg. 1), and MMLF (Eq. 10) with replay, sensitivity-weighted
                     EMA stability, and KL consistency. As in Zhang et al.'s
                     official code, archived architectures are replayed on the
                     CURRENT training batch.

After search, inherited-weight ranking fidelity is scored on the same 200
NAS-Bench-201 architectures (seeded) using fixed batches from the held-out
valid half.
"""

import argparse
import json
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
from scipy.stats import kendalltau

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from xautodl.config_utils import dict2config, load_config  # noqa: E402
from xautodl.datasets import get_datasets, get_nas_search_loaders  # noqa: E402
from xautodl.models import get_cell_based_tiny_net, get_search_spaces  # noqa: E402
from xautodl.models.cell_searchs.search_cells import NAS201SearchCell  # noqa: E402
from xautodl.procedures import get_optim_scheduler, prepare_seed  # noqa: E402

from casso.archive import StreamingFacilityLocationArchive  # noqa: E402
from casso.losses import EMATeacher, MMLFLoss  # noqa: E402
from casso.nb201.api_wrapper import NB201Oracle  # noqa: E402
from casso.nb201.genotype_utils import parse_arch_string  # noqa: E402
from casso.sensitivity import (  # noqa: E402
    compute_depth_sharing_weight,
    compute_snip_saliency,
    layerwise_sensitivity_weight,
    sensitivity_distance,
)


# ----------------------------------------------------------------------------
# Supernet helpers (official TinyNetworkGDAS)
# ----------------------------------------------------------------------------

def forward_path(net, x, hardwts, index):
    """TinyNetworkGDAS.forward with an externally supplied path."""
    feature = net.stem(x)
    for cell in net.cells:
        if isinstance(cell, NAS201SearchCell):
            feature = cell.forward_gdas(feature, hardwts, index)
        else:
            feature = cell(feature)
    out = net.global_pooling(net.lastact(feature)).view(x.size(0), -1)
    return net.classifier(out)


def sample_gdas(net, tau):
    """Official GDAS Gumbel-softmax sampling (search_model_gdas.forward)."""
    while True:
        gumbels = -torch.empty_like(net.arch_parameters).exponential_().log()
        logits = (net.arch_parameters.log_softmax(dim=1) + gumbels) / tau
        probs = F.softmax(logits, dim=1)
        index = probs.max(-1, keepdim=True)[1]
        one_h = torch.zeros_like(logits).scatter_(-1, index, 1.0)
        hardwts = one_h - probs.detach() + probs
        if not (torch.isinf(gumbels).any() or torch.isinf(probs).any() or torch.isnan(probs).any()):
            return hardwts, index


def fixed_path(net, indices):
    """One-hot path (no gradient to phi) for a given per-edge op index vector."""
    e, k = net.arch_parameters.shape
    index = indices.view(e, 1).to(net.arch_parameters.device)
    hardwts = torch.zeros(e, k, device=index.device).scatter_(-1, index, 1.0)
    return hardwts, index


def sample_uniform(net):
    e, k = net.arch_parameters.shape
    return fixed_path(net, torch.randint(0, k, (e,)))


def search_cells(net):
    return [c for c in net.cells if isinstance(c, NAS201SearchCell)]


def node_param_map(net):
    """(edge_index, op_name, cell_position) -> params of that candidate op."""
    out = {}
    for pos, cell in enumerate(search_cells(net), start=1):
        for key, e_idx in cell.edge2index.items():
            for op_i, op in enumerate(cell.edges[key]):
                out[(e_idx, cell.op_names[op_i], pos)] = list(op.parameters())
    return out


def active_node_keys(net, index):
    ops = net.op_names
    n_pos = len(search_cells(net))
    idx = index.view(-1).tolist()
    return [(e, ops[o], p) for p in range(1, n_pos + 1) for e, o in enumerate(idx)]


def active_params(net, index):
    idx = index.view(-1).tolist()
    params = []
    for cell in search_cells(net):
        for key, e_idx in cell.edge2index.items():
            params.extend(cell.edges[key][idx[e_idx]].parameters())
    return params


# ----------------------------------------------------------------------------
# CASSO state (sensitivity, archive, MMLF)
# ----------------------------------------------------------------------------

class CASSOState:
    def __init__(self, net, criterion, args):
        self.net, self.criterion, self.args = net, criterion, args
        self.s_bar, self.variance, self.omega, self.sharing = {}, {}, {}, {}
        self.nodes = {}  # archive key -> node keys
        self.index_of = {}  # archive key -> index tensor (cpu)
        self.next_id = 0
        self.archive = StreamingFacilityLocationArchive(
            budget=args.archive_size, distance_fn=self._dist, tau=1.0,
            stream_window=args.stream_window)
        self.mmlf = MMLFLoss(args.beta, args.gamma, args.eta, weight_decay=0.0)
        self.ema = EMATeacher(net, args.ema_decay)
        self.layer_of_param = {}
        id2layer = {id(p): pos for (_, _, pos), ps in node_param_map(net).items() for p in ps}
        for name, p in net.named_parameters():
            if id(p) in id2layer:
                self.layer_of_param[name] = id2layer[id(p)]
        self.rng = random.Random(args.rand_seed)

    def _dist(self, a, b):
        return sensitivity_distance(self.nodes[a], self.nodes[b], self.s_bar, self.variance,
                                    self.omega, 1e-8)

    def kappa(self, nodes):
        return sum(self.omega.get(u, 0.0) * self.s_bar.get(u, 0.0) for u in nodes)

    def refresh(self, batches, sampler):
        def fwd(x):
            hw, idx = sampler()
            return forward_path(self.net, x, hw, idx)
        self.s_bar, self.variance = compute_snip_saliency(
            self.net, batches, lambda: node_param_map(self.net), self.criterion, forward_fn=fwd)
        n_pos = len(search_cells(self.net))
        self.omega = compute_depth_sharing_weight(self.sharing, n_pos, 1, self.args.rho)

    def record(self, index):
        nodes = active_node_keys(self.net, index)
        for u in nodes:
            self.sharing[u] = self.sharing.get(u, 0) + 1
        return nodes

    def offer(self, nodes, index):
        key = self.next_id
        self.next_id += 1
        self.nodes[key] = nodes
        self.index_of[key] = index.view(-1).detach().cpu()
        _, evicted = self.archive.offer(key, self.kappa(nodes) if self.omega else 0.0, None)
        for k in evicted:
            self.nodes.pop(k, None)
            self.index_of.pop(k, None)

    def loss(self, logits, x, y, index):
        """MMLF (Eq. 10); archived paths replayed on the current batch."""
        arch_logits, arch_targets, arch_params, f_bar = [], [], [], {}
        members = list(self.archive.members)
        if members and self.omega:
            f_bar = layerwise_sensitivity_weight([self.nodes[k] for k in members], self.omega,
                                                 self.s_bar, lambda u: u[2])
            for k in self.rng.sample(members, min(self.args.replay_k, len(members))):
                hw, idx = fixed_path(self.net, self.index_of[k])
                arch_logits.append(forward_path(self.net, x, hw, idx))
                arch_targets.append(y)
                arch_params.append(active_params(self.net, idx))
        layer_w, ema_w = {}, {}
        if f_bar:
            for name, p in self.net.named_parameters():
                j = self.layer_of_param.get(name)
                if j is not None and j in f_bar:
                    layer_w.setdefault(j, []).append(p)
                    ema_w.setdefault(j, []).append(self.ema.get(name))
        return self.mmlf(logits, y, active_params(self.net, index), arch_logits, arch_targets,
                         arch_params, layer_w, ema_w, f_bar,
                         active_on_replay_logits=[logits] * len(arch_logits))["total"]


# ----------------------------------------------------------------------------
# Inherited-weight ranking fidelity
# ----------------------------------------------------------------------------

@torch.no_grad()
def ranking_fidelity(net, oracle, dataset, eval_batches, n_arch, seed, device):
    # Batch statistics in EVERY BatchNorm layer. The cells' BN layers never keep
    # running statistics, but the official stem, residual-block, and final BN
    # layers do, and those were averaged over thousands of different sampled
    # paths during search, so they are invalid for any single architecture
    # (using them gave near-chance accuracy even for the architecture GDAS
    # itself selected). Running-stat updates made here are harmless: the
    # searched weights are saved before evaluation.
    net.train()
    rows = []
    for arch in oracle.random_arch_strings(n_arch, seed=seed):
        gt = oracle.test_accuracy(arch, dataset)
        if gt is None:
            continue
        hw, idx = fixed_path(net, parse_arch_string(arch))
        correct = total = 0
        for x, y in eval_batches:
            pred = forward_path(net, x.to(device), hw, idx).argmax(1)
            correct += (pred == y.to(device)).sum().item()
            total += y.numel()
        rows.append((arch, gt, 100.0 * correct / total))
    net.train()
    tau, p = kendalltau([r[2] for r in rows], [r[1] for r in rows])
    return tau, p, rows


# ----------------------------------------------------------------------------
# Checkpoint / resume (saved every epoch; a run resumes from its .ckpt)
# ----------------------------------------------------------------------------

def save_ckpt(path, epoch, step, net, w_opt, w_sched, a_opt, casso):
    state = {"epoch": epoch, "step": step, "net": net.state_dict(), "w_opt": w_opt.state_dict(),
             "w_sched": w_sched.state_dict(), "a_opt": a_opt.state_dict(),
             "rng": (random.getstate(), torch.get_rng_state(), torch.cuda.get_rng_state_all())}
    if casso is not None:
        ar = casso.archive
        state["casso"] = {
            "s_bar": casso.s_bar, "variance": casso.variance, "omega": casso.omega,
            "sharing": casso.sharing, "nodes": casso.nodes, "index_of": casso.index_of,
            "next_id": casso.next_id, "ema": casso.ema.shadow, "rng": casso.rng.getstate(),
            "members": {k: m.kappa for k, m in ar.members.items()}, "g": ar._g,
            "beta_star": ar._beta_star, "stream_kappa": ar._stream_kappa,
            "arrival": list(ar._arrival_order)}
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)


def load_ckpt(path, net, w_opt, w_sched, a_opt, casso):
    if not os.path.exists(path):
        return 0, 0
    st = torch.load(path, map_location="cuda", weights_only=False)
    net.load_state_dict(st["net"]); w_opt.load_state_dict(st["w_opt"])
    w_sched.load_state_dict(st["w_sched"]); a_opt.load_state_dict(st["a_opt"])
    random.setstate(st["rng"][0]); torch.set_rng_state(st["rng"][1].cpu())
    torch.cuda.set_rng_state_all([r.cpu() for r in st["rng"][2]])
    if casso is not None:
        from collections import deque
        from casso.archive import ArchiveMember
        c = st["casso"]
        casso.s_bar, casso.variance, casso.omega = c["s_bar"], c["variance"], c["omega"]
        casso.sharing, casso.nodes, casso.index_of = c["sharing"], c["nodes"], c["index_of"]
        casso.next_id, casso.ema.shadow = c["next_id"], c["ema"]
        casso.rng.setstate(c["rng"])
        ar = casso.archive
        ar.members = {k: ArchiveMember(k, kap, None) for k, kap in c["members"].items()}
        ar._g, ar._beta_star, ar._stream_kappa = c["g"], c["beta_star"], c["stream_kappa"]
        ar._arrival_order = deque(c["arrival"])
    return st["epoch"], st["step"]


def main(args):
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    prepare_seed(args.rand_seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    cfg_root = os.path.join(args.autodl_root, "configs/nas-benchmark/")
    train_data, valid_data, xshape, class_num = get_datasets(args.dataset, args.data_path, -1)
    config = load_config(os.path.join(args.autodl_root, "configs/nas-benchmark/algos/GDAS.config"),
                         {"class_num": class_num, "xshape": xshape}, None)
    if args.epochs is not None:
        config = config._replace(epochs=args.epochs)
    search_loader, _, valid_loader = get_nas_search_loaders(
        train_data, valid_data, args.dataset, cfg_root, config.batch_size, args.workers)

    search_space = get_search_spaces("cell", "nas-bench-201")
    net = get_cell_based_tiny_net(dict2config({
        "name": "GDAS", "C": 16, "N": 5, "max_nodes": 4, "num_classes": class_num,
        "space": search_space, "affine": False, "track_running_stats": False}, None)).to(device)
    w_optimizer, w_scheduler, criterion = get_optim_scheduler(net.get_weights(), config)
    criterion = criterion.to(device)
    a_optimizer = torch.optim.Adam(net.get_alphas(), lr=3e-4, betas=(0.5, 0.999), weight_decay=1e-3)

    casso = CASSOState(net, criterion, args) if args.method == "casso" else None
    it = iter(search_loader)
    sens_batches = [tuple(t for t in next(it)[:2]) for _ in range(5)]  # K=5 fixed batches

    ckpt_path = args.out.replace(".json", ".ckpt")
    start_epoch, step0 = load_ckpt(ckpt_path, net, w_optimizer, w_scheduler, a_optimizer, casso)
    total_epoch = config.epochs + config.warmup
    steps_per_epoch = len(search_loader)
    warmup_steps = args.warmup_epochs * steps_per_epoch
    log = open(args.out.replace(".json", ".log"), "a")
    t0, step = time.time(), step0
    if start_epoch:
        print(f"resumed from {ckpt_path} at epoch {start_epoch}", flush=True)
        log.write(f"resumed at epoch {start_epoch}\n")
    for epoch in range(start_epoch, total_epoch):
        w_scheduler.update(epoch, 0.0)
        tau = args.tau_max - (args.tau_max - args.tau_min) * epoch / (total_epoch - 1)
        loss_sum = acc_sum = n = 0
        for i, (bx, by, ax, ay) in enumerate(search_loader):
            w_scheduler.update(None, 1.0 * i / steps_per_epoch)
            step += 1
            bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
            if args.sampler == "gdas":
                hw, idx = sample_gdas(net, tau)
                sampler = lambda: sample_gdas(net, tau)  # noqa: E731
            else:
                hw, idx = sample_uniform(net)
                sampler = lambda: sample_uniform(net)  # noqa: E731

            nodes = None
            if casso:
                nodes = casso.record(idx)
                if step > warmup_steps and step % args.refresh_interval == 0:
                    casso.refresh(sens_batches, sampler)

            w_optimizer.zero_grad()
            logits = forward_path(net, bx, hw, idx)
            loss = casso.loss(logits, bx, by, idx) if casso else criterion(logits, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
            w_optimizer.step()
            if casso:
                casso.ema.update(net)
                casso.offer(nodes, idx)

            if args.sampler == "gdas":
                a_optimizer.zero_grad()
                ahw, aidx = sample_gdas(net, tau)
                arch_loss = criterion(forward_path(net, ax.to(device), ahw, aidx), ay.to(device))
                arch_loss.backward()
                a_optimizer.step()

            loss_sum += loss.item()
            acc_sum += (logits.argmax(1) == by).float().mean().item()
            n += 1
        msg = (f"[{time.strftime('%X')}] epoch {epoch + 1}/{total_epoch} tau={tau:.3f} "
               f"loss={loss_sum / n:.3f} acc={100 * acc_sum / n:.2f}% "
               f"archive={len(casso.archive) if casso else 0} elapsed={time.time() - t0:.0f}s")
        print(msg, flush=True)
        log.write(msg + "\n"); log.flush()
        save_ckpt(ckpt_path, epoch + 1, step, net, w_optimizer, w_scheduler, a_optimizer, casso)

    search_time = time.time() - t0
    torch.save({"net": net.state_dict(), "args": vars(args)}, args.out.replace(".json", ".pth"))

    # Fixed held-out batches from the official valid half (test-time transforms).
    g = torch.Generator().manual_seed(0)
    vidx = valid_loader.sampler.indices
    vset = torch.utils.data.Subset(valid_loader.dataset, [vidx[i] for i in torch.randperm(len(vidx), generator=g)[:1280]])
    eval_batches = list(torch.utils.data.DataLoader(vset, batch_size=64, shuffle=False))
    oracle = NB201Oracle(args.oracle)
    tau_k, p_k, rows = ranking_fidelity(net, oracle, args.dataset, eval_batches, 200, args.rand_seed, device)

    result = {"args": vars(args), "search_time_s": search_time, "kendall_tau_inherited": tau_k,
              "kendall_p": p_k, "n": len(rows), "rows": rows}
    json.dump(result, open(args.out, "w"), indent=1)
    msg = f"DONE inherited-weight Kendall tau={tau_k:.4f} (p={p_k:.3g}, n={len(rows)}) time={search_time:.0f}s"
    print(msg, flush=True)
    log.write(msg + "\n"); log.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100", "ImageNet16-120"])
    ap.add_argument("--data_path", default=os.environ.get("CASSO_DATA", "data/cifar.python"),
                    help="CIFAR python-format directory (TORCH_HOME layout used by xautodl)")
    ap.add_argument("--autodl_root", default=os.environ.get("AUTODL_ROOT", "AutoDL-Projects"),
                    help="clone of D-X-Y/AutoDL-Projects (for configs/nas-benchmark)")
    ap.add_argument("--oracle", default=os.environ.get("NB201_CACHE", "data/nb201_test_acc_cache.json"),
                    help="ground-truth cache from scripts/build_oracle_cache.py")
    ap.add_argument("--sampler", default="gdas", choices=["gdas", "uniform"])
    ap.add_argument("--method", default="vanilla", choices=["vanilla", "casso"])
    ap.add_argument("--epochs", type=int, default=None, help="default: official 250")
    ap.add_argument("--warmup_epochs", type=int, default=15, help="CASSO sensitivity warmup")
    ap.add_argument("--tau_max", type=float, default=10.0)
    ap.add_argument("--tau_min", type=float, default=0.1)
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--rho", type=float, default=0.5)
    ap.add_argument("--archive_size", type=int, default=10)
    ap.add_argument("--replay_k", type=int, default=3)
    ap.add_argument("--refresh_interval", type=int, default=5)
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--stream_window", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rand_seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
