"""Candidate-operation lists for each search space (Sec. 4.2.1 / 4.2.2)."""

from collections import namedtuple

# NAS-Bench-201: 5 operations, 4 nodes, 6 edges per cell (Sec. 4.2.2).
NB201_PRIMITIVES = [
    "none",
    "skip_connect",
    "nor_conv_1x1",
    "nor_conv_3x3",
    "avg_pool_3x3",
]

# Standard DARTS search space: 8 operations, 4 intermediate nodes,
# 14 directed edges per cell (Sec. 4.2.1).
DARTS_PRIMITIVES = [
    "none",
    "max_pool_3x3",
    "avg_pool_3x3",
    "skip_connect",
    "sep_conv_3x3",
    "sep_conv_5x5",
    "dil_conv_3x3",
    "dil_conv_5x5",
]

Genotype = namedtuple("Genotype", "normal normal_concat reduce reduce_concat")
