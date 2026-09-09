"""Parse NAS-Bench-201's arch-string format into per-edge op indices, the
inverse of NB201Supernet.genotype_string(). Format verified against the
real downloaded benchmark: api.arch(0) ==
'|avg_pool_3x3~0|+|nor_conv_1x1~0|skip_connect~1|+|nor_conv_1x1~0|skip_connect~1|skip_connect~2|'
"""

import torch

from ..genotypes import NB201_PRIMITIVES
from .cell import EDGE_LIST, NUM_EDGES


def parse_arch_string(arch_str: str) -> torch.Tensor:
    groups = [g for g in arch_str.split("+") if g.strip()]
    # Each group corresponds to one target node j (in increasing order),
    # and lists "op~i" for each predecessor i in increasing order --
    # exactly matching EDGE_LIST's (j, i) ordering.
    ops_by_node = {}
    for j, group in enumerate(groups, start=1):
        entries = [e for e in group.split("|") if e.strip()]
        ops_by_node[j] = [entry.split("~")[0] for entry in entries]

    indices = torch.zeros(NUM_EDGES, dtype=torch.long)
    counters = {j: 0 for j in ops_by_node}
    for e_idx, (j, i) in enumerate(EDGE_LIST):
        op_name = ops_by_node[j][counters[j]]
        counters[j] += 1
        indices[e_idx] = NB201_PRIMITIVES.index(op_name)
    return indices


def operation_strength(arch_logits: torch.Tensor, indices: torch.Tensor) -> float:
    """Mean magnitude of the architecture logits corresponding to this
    candidate's chosen op at each edge (Sec. 4.5.2, Fig. 7)."""
    with torch.no_grad():
        vals = arch_logits[torch.arange(NUM_EDGES), indices]
        return vals.abs().mean().item()
