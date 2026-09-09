"""Standard DARTS searchable cell (Sec. 4.2.1): 4 intermediate nodes, 14
directed edges (2+3+4+5, since node i receives an edge from every earlier
node including the cell's 2 input states), 8 candidate operations per edge.

Trained single-path via Gumbel-softmax as in the NAS-Bench-201 cell
(casso/nb201/cell.py): each edge independently samples ONE op per step.
Unlike the NAS-Bench-201 cell, a node here sums contributions from *all* of
its (multiple) incoming edges during search; the standard DARTS convention
of keeping only the top-2 incoming edges per node (by architecture-weight
magnitude, Table 2's "Search Selection Method: Magnitude") is applied only
at final discretization time (derive_genotype), not during search.
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from ..genotypes import DARTS_PRIMITIVES, Genotype
from ..ops import FactorizedReduce, ReLUConvBN, build_op
from ..nb201.cell import gumbel_softmax_sample  # shared Gumbel-softmax utility

STEPS = 4       # number of intermediate nodes
MULTIPLIER = 4  # output = concat of all STEPS intermediate node outputs


def edge_list(steps: int = STEPS) -> List[Tuple[int, int]]:
    """(node_index, predecessor_index) pairs; node indices 0,1 are the two
    cell inputs (s0, s1); intermediate nodes are 2..steps+1."""
    edges = []
    for node in range(2, steps + 2):
        for pred in range(node):
            edges.append((node, pred))
    return edges


EDGE_LIST = edge_list()
NUM_EDGES = len(EDGE_LIST)  # 2+3+4+5 = 14


class DARTSSearchCell(nn.Module):
    def __init__(self, steps: int, multiplier: int, c_prev_prev: int, c_prev: int,
                 c: int, reduction: bool, reduction_prev: bool, cell_position: int):
        super().__init__()
        self.reduction = reduction
        self.steps = steps
        self.multiplier = multiplier
        self.cell_position = cell_position

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(c_prev_prev, c)
        else:
            self.preprocess0 = ReLUConvBN(c_prev_prev, c, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(c_prev, c, 1, 1, 0)

        self.edges = nn.ModuleDict()
        for e_idx, (node, pred) in enumerate(EDGE_LIST):
            stride = 2 if (reduction and pred < 2) else 1
            for op_name in DARTS_PRIMITIVES:
                key = f"{e_idx}::{op_name}"
                self.edges[key] = build_op(op_name, c, c, stride)

    def forward(self, s0: torch.Tensor, s1: torch.Tensor,
                hardwts: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        for node in range(2, self.steps + 2):
            incoming = []
            for e_idx, (n, pred) in enumerate(EDGE_LIST):
                if n != node:
                    continue
                op_name = DARTS_PRIMITIVES[indices[e_idx].item()]
                key = f"{e_idx}::{op_name}"
                out = self.edges[key](states[pred])
                incoming.append(hardwts[e_idx, indices[e_idx]] * out)
            states.append(sum(incoming))
        return torch.cat(states[2:], dim=1)

    def active_node_keys(self, indices: torch.Tensor):
        keys = []
        for e_idx in range(NUM_EDGES):
            op_name = DARTS_PRIMITIVES[indices[e_idx].item()]
            keys.append((e_idx, op_name, self.cell_position))
        return keys

    def active_params(self, indices: torch.Tensor) -> List[nn.Parameter]:
        params = []
        for e_idx in range(NUM_EDGES):
            op_name = DARTS_PRIMITIVES[indices[e_idx].item()]
            key = f"{e_idx}::{op_name}"
            params.extend(self.edges[key].parameters())
        return params

    def all_node_param_map(self) -> Dict[Tuple[int, str, int], List[nn.Parameter]]:
        out = {}
        for e_idx in range(NUM_EDGES):
            for op_name in DARTS_PRIMITIVES:
                key = f"{e_idx}::{op_name}"
                out[(e_idx, op_name, self.cell_position)] = list(self.edges[key].parameters())
        return out


def derive_genotype(normal_logits: torch.Tensor, reduce_logits: torch.Tensor) -> Genotype:
    """Final discretization (Table 2, "Search Selection Method: Magnitude"):
    for each node, rank its incoming edges by the magnitude of their
    strongest non-'none' op weight, keep the top-2, and use each kept
    edge's argmax op -- the standard DARTS derivation procedure."""
    none_idx = DARTS_PRIMITIVES.index("none")

    def _parse(logits: torch.Tensor) -> List[Tuple[str, int]]:
        weights = torch.softmax(logits, dim=-1)
        gene = []
        for node in range(2, STEPS + 2):
            edge_ids = [e for e, (n, _) in enumerate(EDGE_LIST) if n == node]
            # strongest non-'none' weight per candidate edge into this node
            def edge_strength(e):
                w = weights[e].clone()
                w[none_idx] = -1.0
                return w.max().item()
            best_edges = sorted(edge_ids, key=edge_strength, reverse=True)[:2]
            for e in best_edges:
                w = weights[e].clone()
                w[none_idx] = -1.0
                op_idx = w.argmax().item()
                pred = EDGE_LIST[e][1]
                gene.append((DARTS_PRIMITIVES[op_idx], pred))
        return gene

    normal_gene = _parse(normal_logits)
    reduce_gene = _parse(reduce_logits)
    concat = list(range(2, STEPS + 2))
    return Genotype(normal=normal_gene, normal_concat=concat,
                     reduce=reduce_gene, reduce_concat=concat)
