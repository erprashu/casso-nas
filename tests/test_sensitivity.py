"""Unit tests for casso/sensitivity.py (Eqs. 7-9, 6, 11)."""

import math
import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from casso.sensitivity import (  # noqa: E402
    architecture_kappa,
    compute_depth_sharing_weight,
    compute_snip_saliency,
    layerwise_sensitivity_weight,
    sensitivity_distance,
)


class TinyTwoNodeNet(nn.Module):
    """A 2-output model made of two INDEPENDENT nn.Linear(2, 1) heads, each
    treated as its own 'node' with its own genuine leaf nn.Parameter (unlike
    a slice of a shared weight matrix, e.g. w[0:1, :], which is a non-leaf
    view whose .grad is never populated by autograd -- an easy footgun for
    any node_param_map implementation to fall into, caught by an earlier
    version of this test that sliced a single Linear's weight instead)."""

    def __init__(self):
        super().__init__()
        self.head_a = nn.Linear(2, 1, bias=False)
        self.head_b = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.head_a.weight.copy_(torch.tensor([[1.0, 2.0]]))
            self.head_b.weight.copy_(torch.tensor([[3.0, -1.0]]))

    def forward(self, x):
        return torch.cat([self.head_a(x), self.head_b(x)], dim=1)

    def node_param_map(self):
        return {"A": list(self.head_a.parameters()), "B": list(self.head_b.parameters())}


def _manual_snip_scores(model, x, y, criterion):
    """Ground truth: run one forward/backward and compute |grad * theta|
    summed per head, exactly matching what compute_snip_saliency should
    produce for a single mini-batch."""
    model.zero_grad(set_to_none=True)
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    scores = {}
    for name, head in (("A", model.head_a), ("B", model.head_b)):
        w = head.weight
        scores[name] = (w.grad.detach() * w.detach()).abs().sum().item()
    model.zero_grad(set_to_none=True)
    return scores


def test_snip_saliency_matches_manual_autograd():
    torch.manual_seed(0)
    model = TinyTwoNodeNet()
    criterion = nn.CrossEntropyLoss()

    batches = [
        (torch.tensor([[1.0, 0.5]]), torch.tensor([0])),
        (torch.tensor([[-0.3, 0.8]]), torch.tensor([1])),
        (torch.tensor([[0.2, 0.2]]), torch.tensor([0])),
    ]

    expected_per_batch = [_manual_snip_scores(model, x, y, criterion) for x, y in batches]

    s_bar, variance = compute_snip_saliency(
        model, batches, model.node_param_map, criterion
    )

    for node in ("A", "B"):
        vals = [b[node] for b in expected_per_batch]
        expected_mean = sum(vals) / len(vals)
        assert s_bar[node] == pytest.approx(expected_mean, rel=1e-4), (
            f"node {node}: s_bar mismatch"
        )
        expected_var = torch.tensor(vals).var(unbiased=False).item()
        assert variance[node] == pytest.approx(expected_var, rel=1e-4), (
            f"node {node}: variance mismatch"
        )


def test_snip_saliency_untouched_node_scores_zero():
    """If a node's parameters never receive a gradient (e.g. an op never
    selected on the sampled path), its score must be exactly 0, not NaN or
    a stale value -- Sec. 3.4's implicit assumption that chi(u) only tracks
    visited nodes. This directly mirrors the real single-path supernet,
    where an unselected candidate op's parameters are never touched by the
    forward pass at all."""
    torch.manual_seed(0)
    model = TinyTwoNodeNet()
    criterion = nn.CrossEntropyLoss()

    def forward_only_head_a(x):
        # Only run head_a; head_b's parameters never enter the graph at
        # all, exactly as an unselected op is never called in a real
        # single-path forward (Sec. 3.7).
        out_a = model.head_a(x)
        return torch.cat([out_a, torch.zeros_like(out_a)], dim=1)

    batches = [(torch.tensor([[1.0, 0.5]]), torch.tensor([0]))]
    s_bar, variance = compute_snip_saliency(
        model, batches, model.node_param_map, criterion, forward_fn=forward_only_head_a
    )
    assert s_bar["B"] == 0.0
    assert variance["B"] == 0.0
    assert s_bar["A"] != 0.0


def test_depth_sharing_weight_monotonic_in_depth_and_sharing():
    sharing_count = {
        ("e", "op", 1): 1,   # shallow, unshared
        ("e", "op", 15): 1,  # deep, unshared
        ("e", "op2", 1): 5,  # shallow, heavily shared
    }
    omega = compute_depth_sharing_weight(sharing_count, cell_positions_per_stage=5,
                                          num_stages=3, rho=1.0)
    assert max(omega.values()) == pytest.approx(1.0), "must normalize so max omega == 1"
    # Deeper node should outweigh an equally-shared shallow node (rho > 0).
    assert omega[("e", "op", 15)] > omega[("e", "op", 1)]
    # A heavily-shared shallow node can still outweigh a barely-shared one.
    assert omega[("e", "op2", 1)] > omega[("e", "op", 1)]


def test_depth_sharing_weight_rho_zero_still_reflects_sharing():
    """Sec. 4.5.4 correction (Reviewer #3): rho=0 must NOT collapse omega to
    a fully uniform weight -- chi(u) remains active."""
    sharing_count = {("e", "op", 1): 1, ("e", "op2", 1): 4}
    omega = compute_depth_sharing_weight(sharing_count, cell_positions_per_stage=5,
                                          num_stages=3, rho=0.0)
    assert omega[("e", "op2", 1)] > omega[("e", "op", 1)], (
        "sharing-frequency signal must survive even at rho=0"
    )


def test_sensitivity_distance_self_is_zero():
    nodes = [("e0", "opA", 1), ("e1", "opB", 1)]
    s_bar = {("e0", "opA", 1): 0.5, ("e1", "opB", 1): 0.2}
    variance = {("e0", "opA", 1): 0.01, ("e1", "opB", 1): 0.01}
    omega = {("e0", "opA", 1): 1.0, ("e1", "opB", 1): 0.8}
    d = sensitivity_distance(nodes, nodes, s_bar, variance, omega)
    assert d == pytest.approx(0.0)


def test_sensitivity_distance_matches_hand_computation():
    # Two architectures differing in the op chosen at edge 0.
    nodes_a = [("e0", "opA", 1)]
    nodes_b = [("e0", "opB", 1)]
    s_bar = {("e0", "opA", 1): 0.5, ("e0", "opB", 1): 0.2}
    variance = {("e0", "opA", 1): 0.01, ("e0", "opB", 1): 0.04}
    omega = {("e0", "opA", 1): 1.0, ("e0", "opB", 1): 0.5}
    eps = 1e-8

    d = sensitivity_distance(nodes_a, nodes_b, s_bar, variance, omega, eps=eps)

    w_uv = (1.0 + 0.5) / 2.0
    num = abs(0.5 - 0.2)
    denom = math.sqrt(0.01 + 0.04 + eps)
    expected = w_uv * num / denom
    assert d == pytest.approx(expected, rel=1e-6)


def test_architecture_kappa_and_layerwise_weight():
    omega = {("e0", "op", 1): 0.5, ("e1", "op", 2): 1.0}
    s_bar = {("e0", "op", 1): 0.4, ("e1", "op", 2): 0.3}
    nodes = [("e0", "op", 1), ("e1", "op", 2)]

    kappa = architecture_kappa(nodes, omega, s_bar)
    assert kappa == pytest.approx(0.5 * 0.4 + 1.0 * 0.3)

    # Two archived architectures both using the same two nodes.
    f_bar = layerwise_sensitivity_weight([nodes, nodes], omega, s_bar, layer_of_node=lambda u: u[2])
    # layer 1 (cell_position=1): only node ("e0", "op", 1) contributes, from both archs
    assert f_bar[1] == pytest.approx((0.5 * 0.4 * 2) / 2)
    assert f_bar[2] == pytest.approx((1.0 * 0.3 * 2) / 2)


if __name__ == "__main__":
    import sys as _sys
    raise SystemExit(pytest.main([__file__, "-v"]))
