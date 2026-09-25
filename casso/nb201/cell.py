"""NAS-Bench-201 searchable cell: 4 nodes, 6 edges, 5 candidate operations
per edge (Sec. 4.2.2), trained single-path via Gumbel-softmax (GDAS-style,
Sec. 3.7): at each optimization step, exactly one operation per edge is
executed in the forward pass (saving compute and avoiding gradient
interference across candidates), while gradients still flow to the
architecture logits through the Gumbel-softmax relaxation in the backward
pass (the standard straight-through Gumbel-softmax trick of Jang et al.,
2017, applied here to architecture search as in GDAS).
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..genotypes import NB201_PRIMITIVES
from ..ops import build_op

NUM_NODES = 4  # 4 nodes -> 6 edges (all pairs i<j among {0,1,2,3})
EDGE_LIST: List[Tuple[int, int]] = [(j, i) for j in range(NUM_NODES) for i in range(j)]
NUM_EDGES = len(EDGE_LIST)  # 6


def gumbel_softmax_sample(logits: torch.Tensor, tau: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Straight-through Gumbel-softmax: returns (hard-one-hot-with-soft-grad,
    selected index) for a single edge's logits over primitives."""
    for _ in range(8):  # a few retries guard against the rare inf/nan draw
        gumbels = -torch.empty_like(logits).exponential_().log()
        scores = (logits + gumbels) / tau
        probs = F.softmax(scores, dim=-1)
        index = probs.argmax(dim=-1)
        one_hot = F.one_hot(index, logits.shape[-1]).to(logits.dtype)
        hard = one_hot - probs.detach() + probs
        if not (torch.isinf(gumbels).any() or torch.isnan(hard).any()):
            return hard, index
    return hard, index  # fall through with the last (possibly imperfect) draw


class NB201SearchCell(nn.Module):
    """One instance of the shared NAS-Bench-201 cell at a specific network
    depth position. Architecture logits (phi) are passed in from the
    supernet and SHARED across all cell instances (Sec. 4.2.2: "each stage
    repeating a ... cell five times" with one underlying architecture);
    only the per-instance conv/BN weights are independent."""

    def __init__(self, c_in: int, c_out: int, stride: int, cell_position: int):
        super().__init__()
        self.cell_position = cell_position
        self.stride = stride
        self.edges = nn.ModuleDict()
        for e_idx, (j, i) in enumerate(EDGE_LIST):
            edge_stride = stride if i == 0 else 1
            c_edge_in = c_in if i == 0 else c_out
            for op_name in NB201_PRIMITIVES:
                key = f"{e_idx}::{op_name}"
                self.edges[key] = build_op(op_name, c_edge_in, c_out, edge_stride)

    def forward(self, x: torch.Tensor, hardwts: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """hardwts: (NUM_EDGES, |PRIMITIVES|) straight-through weights.
        indices: (NUM_EDGES,) selected primitive index per edge.

        GDAS straight-through forward (Dong & Yang 2019; verified against
        the official NAS-Bench-201/GDAS reference, search_cells.py::
        forward_gdas): the SELECTED op at each edge is actually executed and
        scaled by its (numerically ~1) straight-through weight, but every
        OTHER candidate op at that edge must also contribute its own
        straight-through weight (numerically ~0, since hard = one_hot -
        probs.detach() + probs) DIRECTLY to the node sum -- without
        executing its forward pass. This adds exactly zero to the forward
        VALUE (each unselected weight is ~0 in the forward pass) but is
        required for the backward pass: it is what routes gradient to the
        *non-selected* architecture logits at all. Without this term (as in
        an earlier version of this function), only the selected logit's
        softmax-Jacobian row receives gradient, which is a materially
        different and, empirically, degenerate training signal -- it
        reproducibly collapsed the learned architecture logits toward the
        trivial 'none' op on 5/6 edges instead of differentiating them.
        """
        nodes = [x]
        num_ops = hardwts.shape[1]
        for node_idx in range(1, NUM_NODES):
            incoming = []
            for e_idx, (j, i) in enumerate(EDGE_LIST):
                if j != node_idx:
                    continue
                sel = indices[e_idx].item()
                op_name = NB201_PRIMITIVES[sel]
                key = f"{e_idx}::{op_name}"
                out = self.edges[key](nodes[i])
                edge_term = hardwts[e_idx, sel] * out
                other = [k for k in range(num_ops) if k != sel]
                edge_term = edge_term + hardwts[e_idx, other].sum()
                incoming.append(edge_term)
            nodes.append(sum(incoming))
        return nodes[-1]

    def active_node_keys(self, indices: torch.Tensor):
        """Return the (edge_index, op_name, cell_position) node keys actually
        used by the currently-sampled discrete architecture (for sensitivity
        bookkeeping, Sec. 3.4)."""
        keys = []
        for e_idx in range(NUM_EDGES):
            op_name = NB201_PRIMITIVES[indices[e_idx].item()]
            keys.append((e_idx, op_name, self.cell_position))
        return keys

    def active_params(self, indices: torch.Tensor) -> List[nn.Parameter]:
        params = []
        for e_idx in range(NUM_EDGES):
            op_name = NB201_PRIMITIVES[indices[e_idx].item()]
            key = f"{e_idx}::{op_name}"
            params.extend(self.edges[key].parameters())
        return params

    def all_node_param_map(self) -> Dict[Tuple[int, str, int], List[nn.Parameter]]:
        """Every (edge, op, this cell's position) -> its parameters, used to
        build the global node-parameter map for SNIP saliency (Eq. 7)."""
        out = {}
        for e_idx in range(NUM_EDGES):
            for op_name in NB201_PRIMITIVES:
                key = f"{e_idx}::{op_name}"
                out[(e_idx, op_name, self.cell_position)] = list(self.edges[key].parameters())
        return out
