"""End-to-end smoke test: tiny CASSO supernet + tiny archive on synthetic
data. This does NOT validate the paper's numbers -- it validates that every
module (sensitivity, archive, MMLF, GDAS sampling, NAS-Bench-201 lookup)
wires together without crashing and produces sane, finite values."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from casso.config import CASSOConfig  # noqa: E402
from casso.nb201.supernet import NB201Supernet  # noqa: E402
from casso.train_search import CASSOSearcher  # noqa: E402


def synthetic_loader(n_batches, batch_size=8, num_classes=10):
    def gen():
        while True:
            x = torch.randn(batch_size, 3, 8, 8)  # tiny spatial size for speed
            y = torch.randint(0, num_classes, (batch_size,))
            yield x, y
    it = gen()
    return it


def test_smoke():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = CASSOConfig(archive_size=3, num_minibatches=2, refresh_interval=2)
    net = NB201Supernet(num_classes=10, base_channels=4)

    total_steps = 20
    warmup_steps = 4
    searcher = CASSOSearcher(net, cfg, device, total_steps=total_steps, warmup_steps=warmup_steps)

    train_iter = synthetic_loader(total_steps)
    val_iter = synthetic_loader(total_steps)
    sens_batches = [(torch.randn(4, 3, 8, 8), torch.randint(0, 10, (4,))) for _ in range(cfg.num_minibatches)]

    losses = []
    for t in range(1, total_steps + 1):
        out = searcher.step(t, train_iter, val_iter, sensitivity_batches=sens_batches)
        assert torch.isfinite(torch.tensor(out["loss"])), f"non-finite loss at step {t}"
        losses.append(out["loss"])

    assert len(searcher.archive) > 0, "archive should have accepted at least one architecture"
    assert len(searcher.archive) <= cfg.archive_size, "archive must respect its budget"
    assert len(searcher.s_bar) > 0, "sensitivity scores should have been computed after warmup"

    best_indices = searcher.best_architecture(None)
    geno_str = net.genotype_string(best_indices)
    print("Final genotype string:", geno_str)
    assert geno_str.count("|") == 3 * 2 + 3  # 3 node-groups; sanity check on format

    print("Losses over training:", [round(l, 3) for l in losses])
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    test_smoke()
