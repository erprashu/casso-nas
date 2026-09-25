"""Ranking fidelity of a saved CASSO supernet checkpoint (R1-8 / R3-1).

For the same 200 NAS-Bench-201 architectures used by search_nb201.py
(oracle.random_arch_strings(200, seed)), computes Kendall tau against
NAS-Bench-201 ground-truth test accuracy for two signals:

1. inherited-weight accuracy: each architecture is activated as a one-hot
   path through the shared supernet weights, BatchNorm statistics are
   re-estimated for that path on training batches (shared BN running stats
   are a mixture over all sampled paths), and accuracy is measured on
   held-out evaluation images.
2. logit strength: mean architecture logit of the candidate's ops (signed),
   plus the |logit| variant used by the original evaluation, as diagnostics.

Usage:
    python scripts/eval_checkpoint_ranking.py --ckpt runs/x.ckpt --dataset cifar10 --seed 0
"""

import argparse
import json
import os
import sys

import torch
import torch.nn as nn
from scipy.stats import kendalltau

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from casso.nb201.api_wrapper import NB201Oracle  # noqa: E402
from casso.nb201.cell import NUM_EDGES  # noqa: E402
from casso.nb201.genotype_utils import parse_arch_string  # noqa: E402
from casso.nb201.supernet import NB201Supernet  # noqa: E402
from casso.utils import get_cifar_loaders, set_seed  # noqa: E402

HF_DIRS = {
    "cifar10": os.path.expanduser("~/CASSO/data/cifar10_hf/plain_text"),
    "cifar100": os.path.expanduser("~/CASSO/data/cifar100_hf/cifar100"),
}


def one_hot(indices: torch.Tensor, num_ops: int, device) -> torch.Tensor:
    hw = torch.zeros(NUM_EDGES, num_ops, device=device)
    hw[torch.arange(NUM_EDGES), indices.to(device)] = 1.0
    return hw


@torch.no_grad()
def inherited_weight_accuracy(net, indices, bn_batches, eval_batches, device) -> float:
    hw = one_hot(indices, net.arch_logits.shape[1], device)
    idx = indices.to(device)
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None  # cumulative average over the recalibration batches
    net.train()
    for x, _ in bn_batches:
        net(x.to(device), hw, idx)
    net.eval()
    correct = total = 0
    for x, y in eval_batches:
        pred = net(x.to(device), hw, idx).argmax(1)
        correct += (pred == y.to(device)).sum().item()
        total += y.numel()
    return 100.0 * correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True, choices=["cifar10", "cifar100"])
    ap.add_argument("--seed", type=int, required=True,
                    help="seed of the search run; selects the same 200 candidates it was scored on")
    ap.add_argument("--n_arch", type=int, default=200)
    ap.add_argument("--bn_batches", type=int, default=10)
    ap.add_argument("--eval_batches", type=int, default=20)
    ap.add_argument("--oracle", default=os.path.expanduser(
        "~/CASSO/data/nasbench201/nb201_test_acc_cache.json"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    set_seed(1234)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    net = NB201Supernet(num_classes=10 if args.dataset == "cifar10" else 100).to(device)
    net.load_state_dict(state["net_state"])
    logits = net.arch_logits.detach().cpu()

    train_loader, val_loader = get_cifar_loaders(args.dataset, None, 64,
                                                 hf_parquet_dir=HF_DIRS[args.dataset])
    it = iter(train_loader)
    bn_batches = [next(it) for _ in range(args.bn_batches)]
    it = iter(val_loader)
    eval_batches = [next(it) for _ in range(args.eval_batches)]

    oracle = NB201Oracle(args.oracle)
    rows = []
    for arch in oracle.random_arch_strings(args.n_arch, seed=args.seed):
        gt = oracle.test_accuracy(arch, args.dataset)
        if gt is None:
            continue
        indices = parse_arch_string(arch)
        vals = logits[torch.arange(NUM_EDGES), indices]
        rows.append({
            "arch": arch,
            "gt": gt,
            "inherited_acc": inherited_weight_accuracy(net, indices, bn_batches, eval_batches, device),
            "logit_signed": vals.mean().item(),
            "logit_abs": vals.abs().mean().item(),
        })

    gt = [r["gt"] for r in rows]
    result = {"ckpt": args.ckpt, "step": state["step"], "dataset": args.dataset,
              "seed": args.seed, "n": len(rows),
              "eval_images": sum(y.numel() for _, y in eval_batches)}
    for key in ("inherited_acc", "logit_signed", "logit_abs"):
        tau, p = kendalltau([r[key] for r in rows], gt)
        result[f"tau_{key}"], result[f"p_{key}"] = tau, p
    result["rows"] = rows
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))


if __name__ == "__main__":
    main()
