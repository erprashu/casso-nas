"""Retrain a CASSO-discovered DARTS genotype from scratch (Sec. 4.3.1:
"Discovered architectures are retrained from scratch following [DARTS]").
Standard DARTS-lineage retrain protocol: 600 epochs, batch 96, SGD lr=0.025
cosine, momentum 0.9, weight_decay 3e-4, auxiliary head (weight 0.4),
drop-path (linearly annealed to 0.2), gradient clip 5, optional Cutout
(Sec. 4.3.1 "Augmentation fairness": CutOut/AutoAugment for the CASSO
NAS(Aug) variant only).
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torchvision.transforms as transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from casso.darts.model_eval import NetworkCIFAR  # noqa: E402
from casso.genotypes import Genotype  # noqa: E402
from casso.utils import AverageMeter, accuracy, get_cifar_loaders, set_seed  # noqa: E402


class Cutout:
    """Standard Cutout augmentation (DeVries & Taylor, 2017), used only for
    the CASSO NAS(Aug) retraining variant (Sec. 4.3.1)."""

    def __init__(self, length: int):
        self.length = length

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        h, w = img.size(1), img.size(2)
        mask = torch.ones(h, w, dtype=img.dtype)
        y, x = torch.randint(h, (1,)).item(), torch.randint(w, (1,)).item()
        y1, y2 = max(0, y - self.length // 2), min(h, y + self.length // 2)
        x1, x2 = max(0, x - self.length // 2), min(w, x + self.length // 2)
        mask[y1:y2, x1:x2] = 0.0
        return img * mask.unsqueeze(0)


def load_genotype(path: str) -> Genotype:
    with open(path) as f:
        data = json.load(f)
    g = data["genotype"] if "genotype" in data else data
    return Genotype(
        normal=[tuple(e) for e in g["normal"]], normal_concat=g["normal_concat"],
        reduce=[tuple(e) for e in g["reduce"]], reduce_concat=g["reduce_concat"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--genotype_json", required=True,
                         help="a search_darts.py output JSON (reads its 'genotype' field)")
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--epochs", type=int, default=600, help="standard DARTS retrain value")
    parser.add_argument("--batch_size", type=int, default=96, help="standard DARTS retrain value")
    parser.add_argument("--init_channels", type=int, default=36, help="paper Table 2 param count target")
    parser.add_argument("--layers", type=int, default=20, help="standard DARTS eval-network depth")
    parser.add_argument("--auxiliary_weight", type=float, default=0.4)
    parser.add_argument("--drop_path_prob", type=float, default=0.2, help="final value, linearly annealed from 0")
    parser.add_argument("--cutout", action="store_true", help="CASSO NAS(Aug) variant only")
    parser.add_argument("--cutout_length", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_dir", default=os.path.expanduser("~/CASSO/data/cifar"))
    parser.add_argument("--hf_parquet_dir", default=None)
    parser.add_argument("--out", default="retrain_output.json")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    genotype = load_genotype(args.genotype_json)
    print(f"Genotype: {genotype}", flush=True)

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
    if args.cutout:
        train_loader.dataset.transform.transforms.append(Cutout(args.cutout_length))

    num_classes = 10 if args.dataset == "cifar10" else 100
    net = NetworkCIFAR(num_classes=num_classes, genotype=genotype,
                        init_channels=args.init_channels, layers=args.layers,
                        auxiliary=True).to(device)
    print(f"Params: {sum(p.numel() for p in net.parameters()):,}", flush=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.025, momentum=0.9, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    t_start = time.time()
    best_val_acc = 0.0

    for epoch in range(args.epochs):
        net.drop_path_prob = args.drop_path_prob * epoch / max(args.epochs - 1, 1)

        net.train()
        train_loss, train_acc = AverageMeter(), AverageMeter()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, logits_aux = net(x)
            loss = criterion(logits, y)
            if logits_aux is not None:
                loss = loss + args.auxiliary_weight * criterion(logits_aux, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss.update(loss.item(), x.size(0))
            train_acc.update(accuracy(logits, y), x.size(0))
        scheduler.step()

        net.eval()
        val_acc = AverageMeter()
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits, _ = net(x)
                val_acc.update(accuracy(logits, y), x.size(0))
        best_val_acc = max(best_val_acc, val_acc.avg)

        elapsed = time.time() - t_start
        print(f"[epoch {epoch+1}/{args.epochs}] train_loss={train_loss.avg:.3f} "
              f"train_acc={train_acc.avg:.2f}% val_acc={val_acc.avg:.2f}% "
              f"best={best_val_acc:.2f}% drop_path={net.drop_path_prob:.3f} "
              f"elapsed={elapsed:.0f}s", flush=True)
        history.append({"epoch": epoch + 1, "train_loss": train_loss.avg,
                         "train_acc": train_acc.avg, "val_acc": val_acc.avg,
                         "elapsed_s": elapsed})

    result = {"args": vars(args), "history": history, "best_val_acc": best_val_acc,
              "final_val_acc": history[-1]["val_acc"], "total_wall_time_s": time.time() - t_start}
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved results to {args.out}", flush=True)


if __name__ == "__main__":
    main()
