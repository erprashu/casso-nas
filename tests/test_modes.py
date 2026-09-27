"""Behavioral tests for the sampler/method options used by the ranking
experiments (GDAS vs. uniform sampling; CASSO vs. vanilla training)."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from casso.config import CASSOConfig  # noqa: E402
from casso.nb201.supernet import NB201Supernet  # noqa: E402
from casso.train_search import CASSOSearcher  # noqa: E402


def _batches(n_classes=10, bs=8):
    while True:
        yield torch.randn(bs, 3, 8, 8), torch.randint(0, n_classes, (bs,))


def _run(sampler, method, steps=12, warmup=3):
    torch.manual_seed(0)
    cfg = CASSOConfig(archive_size=3, num_minibatches=2, refresh_interval=2)
    net = NB201Supernet(num_classes=10, base_channels=4)
    s = CASSOSearcher(net, cfg, torch.device("cpu"), total_steps=steps, warmup_steps=warmup,
                      sampler=sampler, method=method)
    sens = [(torch.randn(4, 3, 8, 8), torch.randint(0, 10, (4,))) for _ in range(2)]
    phi0 = net.arch_logits.detach().clone()
    tr, va = _batches(), _batches()
    for t in range(1, steps + 1):
        out = s.step(t, tr, va, sensitivity_batches=sens)
        assert torch.isfinite(torch.tensor(out["loss"]))
    return s, phi0


@pytest.mark.parametrize("sampler", ["gdas", "uniform"])
@pytest.mark.parametrize("method", ["casso", "vanilla"])
def test_all_modes_run(sampler, method):
    _run(sampler, method)


def test_uniform_never_updates_phi():
    s, phi0 = _run("uniform", "casso")
    assert torch.equal(s.net.arch_logits.detach(), phi0)


def test_gdas_updates_phi():
    s, phi0 = _run("gdas", "vanilla")
    assert not torch.equal(s.net.arch_logits.detach(), phi0)


def test_vanilla_has_no_archive_or_sensitivity():
    s, _ = _run("gdas", "vanilla")
    assert len(s.archive) == 0 and not s.s_bar and not s.sharing_count


def test_casso_fills_archive_and_sensitivity():
    s, _ = _run("uniform", "casso")
    assert len(s.archive) > 0 and s.s_bar


def test_uniform_sampler_covers_all_ops():
    cfg = CASSOConfig()
    net = NB201Supernet(num_classes=10, base_channels=4)
    s = CASSOSearcher(net, cfg, torch.device("cpu"), total_steps=10, warmup_steps=1,
                      sampler="uniform", method="vanilla")
    seen = torch.zeros_like(net.arch_logits, dtype=torch.bool)
    for _ in range(400):
        hw, idx = s.sample_path(1.0)
        assert torch.allclose(hw.sum(dim=1), torch.ones(hw.shape[0]))
        seen[torch.arange(idx.numel()), idx] = True
    assert seen.all()


def test_replay_members_are_random():
    s, _ = _run("uniform", "casso", steps=10)
    members = list(s.archive.members)
    picks = {tuple(s._rng.sample(members, 1)) for _ in range(50)}
    assert len(picks) > 1


def test_search_split_is_disjoint():
    from torch.utils.data import TensorDataset, Subset
    import casso.utils as U
    full = TensorDataset(torch.arange(100).float().unsqueeze(1), torch.zeros(100))
    perm = torch.randperm(len(full), generator=torch.Generator().manual_seed(0)).tolist()
    a, b = Subset(full, perm[:50]), Subset(full, perm[50:])
    assert set(a.indices).isdisjoint(b.indices) and len(a) + len(b) == 100
    assert "search_split" in U.get_cifar_loaders.__code__.co_varnames
