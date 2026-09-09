"""Real CASSO NAS search on the standard DARTS space (Sec. 4.2.1).

Usage:
    python scripts/search_darts.py --seed 0 --dataset cifar10
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from casso.config import CASSOConfig  # noqa: E402
from casso.darts.cell import derive_genotype  # noqa: E402
from casso.darts.supernet import DARTSSupernet  # noqa: E402
from casso.train_search_darts import DARTSCASSOSearcher  # noqa: E402
from casso.utils import AverageMeter, get_cifar_loaders, infinite_loader, set_seed  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    # Paper-exact defaults (Sec. 4.2.1): 50 epochs, 16 initial channels,
    # batch size 256, SGD lr=0.025 cosine / momentum 0.9 / wd 0.0003,
    # Adam arch lr=0.0006 / betas(0.5,0.999) / wd 0.001 (all set inside
    # DARTSCASSOSearcher itself, not overridable here to stay paper-exact).
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--epochs", type=int, default=50, help="paper value (Sec. 4.2.1)")
    parser.add_argument("--batch_size", type=int, default=256, help="paper value (Sec. 4.2.1)")
    parser.add_argument("--init_channels", type=int, default=16, help="paper value (Sec. 4.2.1)")
    parser.add_argument("--layers", type=int, default=8,
                         help="standard DARTS search-phase cell count (not explicitly "
                              "restated in the excerpt of Sec. 4.2.1 we have, but this is "
                              "the standard DARTS-lineage convention for the SEARCH "
                              "supernet, as distinct from the 20-cell EVALUATION network)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup_epochs", type=int, default=15,
                         help="paper value (Sec. 3.7), same warmup used for NAS-Bench-201")
    parser.add_argument("--data_dir", default=os.path.expanduser("~/CASSO/data/cifar"))
    parser.add_argument("--hf_parquet_dir", default=None)
    parser.add_argument("--eval_every_steps", type=int, default=200)
    parser.add_argument("--out", default="run_output_darts.json")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    print("Loading CIFAR data...", flush=True)
    if args.hf_parquet_dir is not None:
        hf_dir = args.hf_parquet_dir or None
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

    cfg = CASSOConfig()
    num_classes = 10 if args.dataset == "cifar10" else 100
    net = DARTSSupernet(num_classes=num_classes, init_channels=args.init_channels,
                         layers=args.layers)
    searcher = DARTSCASSOSearcher(net, cfg, device, total_steps=total_steps,
                                   warmup_steps=warmup_steps)
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

    genotype = derive_genotype(net.normal_logits.detach().cpu(), net.reduce_logits.detach().cpu())
    print(f"Final genotype: {genotype}", flush=True)

    result = {
        "args": vars(args),
        "history": history,
        "genotype": {
            "normal": genotype.normal, "normal_concat": genotype.normal_concat,
            "reduce": genotype.reduce, "reduce_concat": genotype.reduce_concat,
        },
        "total_wall_time_s": time.time() - t_start,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved results to {args.out}", flush=True)


if __name__ == "__main__":
    main()
