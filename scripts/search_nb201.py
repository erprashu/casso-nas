"""Real CASSO NAS search on NAS-Bench-201 (Sec. 4.4, 4.5.2).

Usage:
    python scripts/search_nb201.py --epochs 2 --dataset cifar10 --seed 0
"""

import argparse
import json
import os
import sys
import time

import torch
from scipy.stats import kendalltau

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from casso.config import CASSOConfig  # noqa: E402
from casso.nb201.api_wrapper import NB201Oracle  # noqa: E402
from casso.nb201.genotype_utils import operation_strength, parse_arch_string  # noqa: E402
from casso.nb201.supernet import NB201Supernet  # noqa: E402
from casso.train_search import CASSOSearcher  # noqa: E402
from casso.utils import AverageMeter, get_cifar_loaders, infinite_loader, set_seed  # noqa: E402


def evaluate_kendall_tau(net: NB201Supernet, oracle: NB201Oracle, dataset: str,
                          n_samples: int, seed: int):
    arch_strings = oracle.random_arch_strings(n_samples, seed=seed)
    strengths, ground_truths = [], []
    for arch_str in arch_strings:
        indices = parse_arch_string(arch_str)
        strength = operation_strength(net.arch_logits.detach().cpu(), indices)
        acc = oracle.test_accuracy(arch_str, dataset)
        if acc is None:
            continue
        strengths.append(strength)
        ground_truths.append(acc)
    tau, p_value = kendalltau(strengths, ground_truths)
    return tau, p_value, len(strengths)


def main():
    parser = argparse.ArgumentParser()
    # Paper-faithful defaults (Sec. 3.7 / 4.2.2): batch_size=64, warmup=15
    # epochs, SGD lr 0.025->0.001 cosine, wd=0.0005, momentum=0.9 (already
    # the CASSOSearcher/CASSOConfig defaults). The paper states search cost
    # as ~34560s (0.4 GPU-days) but does NOT state an explicit total epoch
    # count for this search space (only DARTS-space search is explicitly
    # "50 epochs"). epochs=193 below is THIS SCRIPT'S calibration to hit
    # that same wall-clock budget, measured empirically on this RTX 5090
    # from the validated 1-epoch run (781 steps / 179s with archive+MMLF
    # overhead already included): 34560s / (179s/781 steps) / 781
    # steps-per-epoch ~= 193 epochs. This is an assumption, not a value
    # taken verbatim from the paper -- flagged here rather than silently
    # presented as if it were.
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--epochs", type=int, default=193,
                         help="calibrated to match the paper's ~34560s search cost on this "
                              "GPU; the paper itself does not state an explicit epoch count "
                              "for NAS-Bench-201 search (see comment above)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_dir", default=os.path.expanduser("~/CASSO/data/cifar"))
    parser.add_argument("--hf_parquet_dir", default=None,
        help="fast HF mirror path; defaults to the right cifar10/cifar100 mirror dir "
             "for --dataset unless overridden; pass '' to force the (slow) torchvision "
             "download instead")
    parser.add_argument("--nb201_pkl", default=os.path.expanduser(
        "~/CASSO/data/nasbench201/nasbench201_v1_0-e61699.pkl"))
    parser.add_argument("--warmup_epochs", type=int, default=15,
                         help="paper value (Sec. 3.7): 'a warmup period of 15 epochs'")
    parser.add_argument("--eval_every_steps", type=int, default=500)
    parser.add_argument("--kendall_samples", type=int, default=200)
    parser.add_argument("--out", default="run_output.json")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    print("Loading CIFAR data...", flush=True)
    if args.hf_parquet_dir is not None:
        hf_dir = args.hf_parquet_dir or None  # '' means "force torchvision path"
    else:
        default_hf_dirs = {
            "cifar10": os.path.expanduser("~/CASSO/data/cifar10_hf/plain_text"),
            "cifar100": os.path.expanduser("~/CASSO/data/cifar100_hf/cifar100"),
        }
        hf_dir = default_hf_dirs[args.dataset]
    train_loader, val_loader = get_cifar_loaders(args.dataset, args.data_dir, args.batch_size,
                                                  hf_parquet_dir=hf_dir)
    train_iter = infinite_loader(train_loader)
    val_iter = infinite_loader(val_loader)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs
    print(f"{steps_per_epoch} steps/epoch, {total_steps} total steps, "
          f"{warmup_steps} warmup steps", flush=True)

    print("Loading NAS-Bench-201 oracle...", flush=True)
    oracle = NB201Oracle(args.nb201_pkl)

    cfg = CASSOConfig()
    net = NB201Supernet(num_classes=10 if args.dataset == "cifar10" else 100,
                         base_channels=args.base_channels)
    searcher = CASSOSearcher(net, cfg, device, total_steps=total_steps, warmup_steps=warmup_steps)
    print(f"Supernet params: {sum(p.numel() for p in net.parameters()):,}", flush=True)

    sens_batches = [next(train_iter) for _ in range(cfg.num_minibatches)]

    loss_meter, acc_meter = AverageMeter(), AverageMeter()
    history = []
    t_start = time.time()

    for t in range(1, total_steps + 1):
        out = searcher.step(t, train_iter, val_iter, sensitivity_batches=sens_batches)
        loss_meter.update(out["loss"])
        acc_meter.update(out["acc"])

        if t % args.eval_every_steps == 0 or t == total_steps:
            elapsed = time.time() - t_start
            print(f"[step {t}/{total_steps}] loss={loss_meter.avg:.3f} "
                  f"acc={acc_meter.avg:.2f}% archive_size={len(searcher.archive)} "
                  f"tau_gumbel={out['tau']:.3f} elapsed={elapsed:.0f}s", flush=True)
            history.append({"step": t, "loss": loss_meter.avg, "acc": acc_meter.avg,
                             "archive_size": len(searcher.archive), "elapsed_s": elapsed})
            loss_meter.reset()
            acc_meter.reset()

    print("Evaluating Kendall-tau against NAS-Bench-201 ground truth...", flush=True)
    tau, p_value, n_valid = evaluate_kendall_tau(
        net, oracle, args.dataset, args.kendall_samples, seed=args.seed
    )
    print(f"Kendall-tau = {tau:.4f} (p={p_value:.4g}, n={n_valid})", flush=True)

    result = {
        "args": vars(args),
        "history": history,
        "kendall_tau": tau,
        "kendall_p_value": p_value,
        "kendall_n": n_valid,
        "total_wall_time_s": time.time() - t_start,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved results to {args.out}", flush=True)


if __name__ == "__main__":
    main()
