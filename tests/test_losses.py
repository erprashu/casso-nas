"""Unit tests for casso/losses.py (Eq. 10-11, MMLF and EMA teacher)."""

import os
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from casso.losses import EMATeacher, MMLFLoss  # noqa: E402


def test_supervised_term_matches_manual_ce_plus_l2():
    torch.manual_seed(0)
    mmlf = MMLFLoss(beta=0.3, gamma=0.1, eta=0.05, weight_decay=0.01)
    logits = torch.randn(4, 5, requires_grad=True)
    targets = torch.randint(0, 5, (4,))
    params = [torch.randn(3, 3), torch.randn(2)]

    term = mmlf.supervised_term(logits, targets, params)
    expected = F.cross_entropy(logits, targets) + 0.01 * sum((p ** 2).sum() for p in params)
    assert term.item() == pytest.approx(expected.item(), rel=1e-5)


def test_supervised_term_with_no_params_is_pure_ce():
    mmlf = MMLFLoss(beta=0.3, gamma=0.1, eta=0.05, weight_decay=0.5)
    logits = torch.randn(4, 5)
    targets = torch.randint(0, 5, (4,))
    term = mmlf.supervised_term(logits, targets, [])
    assert term.item() == pytest.approx(F.cross_entropy(logits, targets).item(), rel=1e-5)


def test_ema_stability_term_matches_manual_weighted_l2():
    mmlf = MMLFLoss(beta=0.3, gamma=0.1, eta=0.05, weight_decay=0.0)
    w1 = torch.tensor([1.0, 2.0])
    w1_ema = torch.tensor([1.5, 1.5])
    w2 = torch.tensor([0.0])
    w2_ema = torch.tensor([1.0])

    layer_weights = {1: [w1], 2: [w2]}
    ema_weights = {1: [w1_ema], 2: [w2_ema]}
    f_bar = {1: 2.0, 2: 0.5}

    term = mmlf.ema_stability_term(layer_weights, ema_weights, f_bar)

    layer1_sq = ((w1 - w1_ema) ** 2).sum().item()  # (1-1.5)^2 + (2-1.5)^2 = 0.5
    layer2_sq = ((w2 - w2_ema) ** 2).sum().item()  # (0-1)^2 = 1.0
    expected = mmlf.gamma * (f_bar[1] * layer1_sq + f_bar[2] * layer2_sq)
    assert term.item() == pytest.approx(expected, rel=1e-5)


def test_ema_stability_term_empty_returns_zero():
    mmlf = MMLFLoss(beta=0.3, gamma=0.1, eta=0.05, weight_decay=0.0)
    term = mmlf.ema_stability_term({}, {}, {})
    assert term.item() == pytest.approx(0.0)


def test_ema_stability_term_skips_layers_missing_from_f_bar_implicitly():
    """A layer with weights but no f_bar entry should contribute according to
    f_bar.get(j, 0.0) == 0, i.e. no penalty -- concentrating regularization
    only where the archive says forgetting risk exists (Eq. 11)."""
    mmlf = MMLFLoss(beta=0.3, gamma=1.0, eta=0.0, weight_decay=0.0)
    layer_weights = {1: [torch.tensor([5.0])]}
    ema_weights = {1: [torch.tensor([0.0])]}
    term = mmlf.ema_stability_term(layer_weights, ema_weights, f_bar={})  # no entry for layer 1
    assert term.item() == pytest.approx(0.0)


def test_kl_consistency_term_matches_manual_kl():
    """Each pair (active_on_replay_i, archived_i) comes from the SAME
    replayed batch of inputs for architecture i -- but different archived
    architectures can be replayed on different batches, so the two active
    logits in the two pairs are deliberately different tensors here."""
    mmlf = MMLFLoss(beta=0.3, gamma=0.1, eta=1.0, weight_decay=0.0)  # eta=1 to simplify
    torch.manual_seed(0)
    pairs = [
        (torch.randn(2, 4), torch.randn(2, 4)),
        (torch.randn(2, 4), torch.randn(2, 4)),
    ]

    term = mmlf.kl_consistency_term(pairs)

    total = 0.0
    for active_i, archived_i in pairs:
        log_p_active = F.log_softmax(active_i, dim=-1)
        log_p_i = F.log_softmax(archived_i, dim=-1)
        kl = F.kl_div(log_p_i, log_p_active, log_target=True, reduction="none").sum(-1).mean()
        total += kl.item()
    expected = total / len(pairs)
    assert term.item() == pytest.approx(expected, rel=1e-4)


def test_kl_consistency_term_zero_when_identical_distributions():
    mmlf = MMLFLoss(beta=0.3, gamma=0.1, eta=1.0, weight_decay=0.0)
    logits = torch.randn(3, 4)
    pairs = [(logits, logits.clone()), (logits, logits.clone())]
    term = mmlf.kl_consistency_term(pairs)
    assert term.item() == pytest.approx(0.0, abs=1e-6)


def test_kl_consistency_term_empty_archive_returns_zero():
    mmlf = MMLFLoss(beta=0.3, gamma=0.1, eta=0.05, weight_decay=0.0)
    term = mmlf.kl_consistency_term([])
    assert term.item() == pytest.approx(0.0)


def test_forward_combines_terms_with_correct_beta_weighting():
    torch.manual_seed(0)
    beta, gamma, eta, wd = 0.3, 0.0, 0.0, 0.0  # isolate the beta-weighted CE terms
    mmlf = MMLFLoss(beta, gamma, eta, wd)

    active_logits = torch.randn(4, 3, requires_grad=True)
    active_targets = torch.randint(0, 3, (4,))
    archived_logits = [torch.randn(4, 3), torch.randn(4, 3)]
    archived_targets = [torch.randint(0, 3, (4,)), torch.randint(0, 3, (4,))]

    out = mmlf(
        active_logits, active_targets, [],
        archived_logits, archived_targets, [[], []],
        layer_weights={}, ema_weights={}, f_bar={},
    )

    active_ce = F.cross_entropy(active_logits, active_targets)
    replay_ce = sum(F.cross_entropy(lg, tg) for lg, tg in zip(archived_logits, archived_targets)) / 2
    expected_total = (1 - beta) * active_ce + beta * replay_ce
    assert out["total"].item() == pytest.approx(expected_total.item(), rel=1e-4)


def test_forward_with_no_archive_falls_back_to_active_only_when_beta_zero():
    torch.manual_seed(0)
    mmlf = MMLFLoss(beta=0.0, gamma=0.0, eta=0.0, weight_decay=0.0)
    active_logits = torch.randn(4, 3)
    active_targets = torch.randint(0, 3, (4,))

    out = mmlf(active_logits, active_targets, [], [], [], [], {}, {}, {})
    expected = F.cross_entropy(active_logits, active_targets)
    assert out["total"].item() == pytest.approx(expected.item(), rel=1e-5)


def test_ema_teacher_update_matches_manual_exponential_average():
    torch.manual_seed(0)
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    decay = 0.9
    ema = EMATeacher(model, decay)

    initial_shadow = ema.get("weight").clone()
    assert torch.allclose(initial_shadow, model.weight)

    with torch.no_grad():
        model.weight.copy_(torch.tensor([[2.0, 0.0], [0.0, 2.0]]))
    ema.update(model)

    expected = decay * initial_shadow + (1 - decay) * model.weight
    assert torch.allclose(ema.get("weight"), expected, atol=1e-6)


def test_ema_teacher_shadow_is_detached_copy_not_a_reference():
    model = nn.Linear(2, 2, bias=False)
    ema = EMATeacher(model, decay=0.99)
    with torch.no_grad():
        model.weight.fill_(99.0)
    # The shadow must NOT have changed just because the live model did.
    assert not torch.allclose(ema.get("weight"), model.weight)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
