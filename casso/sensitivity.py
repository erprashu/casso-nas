"""Depth- and sharing-aware sensitivity scoring (paper Sec. 3.4, Eqs. 7-9).

Design note on what a "node" u is (the paper leaves this implementation
detail informal): we treat each *candidate-operation module instance* in the
supernet as a node -- i.e., the tuple (edge_index, op_name, cell_position),
where cell_position in {1, ..., L} is the position of that cell in the
network's sequential stack of L cells. This is what makes depth (Eq. 8)
meaningful: the same op/edge choice re-appears at every cell_position, and
each occurrence is its own node with its own parameters (since supernet
weights are NOT tied across cell instances, only the discrete architecture
choice is), so a node's depth is simply its cell_position. This directly
supports the layer-wise forgetting breakdown of Table 11 (Stage 1 = early
cell_positions, Stage 3 = late cell_positions).

An architecture alpha is characterized by one op choice per edge (shared
across all cell instances of a given type); U(alpha) is the set of nodes
(edge, op, cell_position) it actually uses, for op = alpha's chosen op at
that edge, ranging over every cell_position.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

NodeKey = Tuple[int, str, int]  # (edge_index, op_name, cell_position)


@dataclass
class SensitivityState:
    """Holds the current sensitivity statistics for every node in the supernet."""

    s_bar: Dict[NodeKey, float] = field(default_factory=dict)   # mean SNIP saliency, Eq. 7
    variance: Dict[NodeKey, float] = field(default_factory=dict)  # across-minibatch variance, Eq. 7
    omega: Dict[NodeKey, float] = field(default_factory=dict)   # depth-sharing weight, Eq. 8
    sharing_count: Dict[NodeKey, int] = field(default_factory=dict)  # chi(u) raw count


def compute_snip_saliency(
    supernet: nn.Module,
    minibatches: List[Tuple[torch.Tensor, torch.Tensor]],
    node_param_getter,
    criterion: nn.Module,
    forward_fn=None,
) -> Tuple[Dict[NodeKey, float], Dict[NodeKey, float]]:
    """Eq. 7: mean and across-mini-batch-variance SNIP saliency per node.

    Args:
        supernet: the weight-sharing supernet (all candidate op modules present).
        minibatches: K balanced mini-batches (x, y), drawn either at init (t=0)
            or at the current weights theta_t (refresh at step t > 0; Sec. 3.4).
        node_param_getter: callable() -> Dict[NodeKey, List[nn.Parameter]],
            mapping each node to the parameters that belong to it.
        criterion: loss function, e.g. nn.CrossEntropyLoss().
        forward_fn: callable(x) -> logits. Since the supernet is single-path
            (Sec. 3.7) its forward requires a sampled (hardwts, indices), not
            just x; the caller closes over those (or samples fresh per call)
            and passes a plain callable here so this function stays agnostic
            to the search space's specific forward signature. Defaults to
            `supernet(x)` for search spaces whose forward only needs x.

    Returns:
        (s_bar, variance) dictionaries keyed by NodeKey. A node's score only
        reflects mini-batches in which it was actually on the sampled path
        (its parameters receive no gradient otherwise); this is intentional
        and consistent with chi(u) tracking only visited nodes (Sec. 3.4).
    """
    node_params = node_param_getter()
    per_batch_scores: Dict[NodeKey, List[float]] = {k: [] for k in node_params}
    forward_fn = forward_fn or supernet

    device = next(supernet.parameters()).device
    for x, y in minibatches:
        x, y = x.to(device), y.to(device)
        supernet.zero_grad(set_to_none=True)
        out = forward_fn(x)
        loss = criterion(out, y)
        loss.backward()
        with torch.no_grad():
            for node, params in node_params.items():
                total = 0.0
                for p in params:
                    if p.grad is None:
                        continue
                    total += (p.grad * p).abs().sum().item()
                per_batch_scores[node].append(total)
    supernet.zero_grad(set_to_none=True)

    s_bar: Dict[NodeKey, float] = {}
    variance: Dict[NodeKey, float] = {}
    for node, scores in per_batch_scores.items():
        t = torch.tensor(scores) if scores else torch.zeros(1)
        s_bar[node] = t.mean().item()
        variance[node] = t.var(unbiased=False).item() if t.numel() > 1 else 0.0
    return s_bar, variance


def compute_depth_sharing_weight(
    sharing_count: Dict[NodeKey, int],
    cell_positions_per_stage: int,
    num_stages: int,
    rho: float,
) -> Dict[NodeKey, float]:
    """Eq. 8: omega(u) = exp(rho * dep(u)) * chi(u), normalized per cell so
    max_u omega(u) = 1 (normalization is applied globally here since the
    network's cells share one search space; this matches the paper's
    "normalize omega(u) across all nodes within each cell").
    """
    total_positions = cell_positions_per_stage * num_stages
    omega: Dict[NodeKey, float] = {}
    for node, chi in sharing_count.items():
        _, _, cell_position = node
        # Normalize depth to [0, 1] so exp(rho * dep) has a bounded, comparable
        # scale across search spaces with different total depths.
        dep = cell_position / max(total_positions, 1)
        chi_val = max(chi, 1)  # chi(u) >= 1 by definition (Sec. 3.4)
        omega[node] = float(torch.exp(torch.tensor(rho * dep)).item() * chi_val)
    if omega:
        max_omega = max(omega.values())
        if max_omega > 0:
            omega = {k: v / max_omega for k, v in omega.items()}
    return omega


def sensitivity_distance(
    nodes_alpha: List[NodeKey],
    nodes_beta: List[NodeKey],
    s_bar: Dict[NodeKey, float],
    variance: Dict[NodeKey, float],
    omega: Dict[NodeKey, float],
    eps: float = 1e-8,
) -> float:
    """Eq. 9: sensitivity-aware semi-metric distance between two architectures.

    For cell-based spaces (DARTS, NAS-Bench-201) node alignment is
    deterministic by edge index (Sec. 3.4), so we pair nodes_alpha[i] with
    nodes_beta[i] directly (both lists are ordered by (edge_index,
    cell_position) and have one entry per edge/position, differing only in
    which op was selected there) -- this is the "simple one-to-one pairing
    computed in O(|U|) time" the paper describes.
    """
    assert len(nodes_alpha) == len(nodes_beta), (
        "cell-based one-to-one pairing requires matching (edge, cell_position) "
        "structure between the two architectures"
    )
    total = 0.0
    for u, v in zip(nodes_alpha, nodes_beta):
        su = s_bar.get(u, 0.0)
        sv = s_bar.get(v, 0.0)
        vu = variance.get(u, 0.0)
        vv = variance.get(v, 0.0)
        wu = omega.get(u, 0.0)
        wv = omega.get(v, 0.0)
        w_uv = (wu + wv) / 2.0
        num = abs(su - sv)
        denom = (vu + vv + eps) ** 0.5
        total += w_uv * num / denom
    return total


def architecture_kappa(nodes_alpha: List[NodeKey], omega: Dict[NodeKey, float],
                        s_bar: Dict[NodeKey, float]) -> float:
    """Eq. 6: kappa(alpha) = sum_{u in U(alpha)} omega(u) * s_bar(u)."""
    return sum(omega.get(u, 0.0) * s_bar.get(u, 0.0) for u in nodes_alpha)


def layerwise_sensitivity_weight(
    archive_nodes: List[List[NodeKey]],
    omega: Dict[NodeKey, float],
    s_bar: Dict[NodeKey, float],
    layer_of_node,
) -> Dict[int, float]:
    """Eq. 11: F_bar_j(M_t) = (1/m) * sum_i sum_{u in U_j(alpha^(i))} omega(u) s_bar(u).

    Args:
        archive_nodes: list of U(alpha^(i)) for each archived architecture.
        layer_of_node: callable(NodeKey) -> int, mapping a node to its layer
            index j (here: its cell_position, so each depth position is its
            own "layer" for regularization purposes).
    """
    m = max(len(archive_nodes), 1)
    f_bar: Dict[int, float] = {}
    for nodes in archive_nodes:
        for u in nodes:
            j = layer_of_node(u)
            f_bar[j] = f_bar.get(j, 0.0) + omega.get(u, 0.0) * s_bar.get(u, 0.0)
    return {j: v / m for j, v in f_bar.items()}
