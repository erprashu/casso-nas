"""NAS-Bench-201 macro supernet (Sec. 4.2.2): stem -> 3 stages x 5 cells,
connected by fixed stride-2 residual blocks, matching the published
NAS-Bench-201 macro skeleton (Dong & Yang, ICLR 2020)."""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .cell import NB201SearchCell, NUM_EDGES, gumbel_softmax_sample
from ..genotypes import NB201_PRIMITIVES

CELLS_PER_STAGE = 5
NUM_STAGES = 3
STAGE_CHANNELS = (16, 32, 64)


class ResNetBasicBlock(nn.Module):
    """Fixed (non-searched) stride-2 downsampling block between stages."""

    def __init__(self, c_in: int, c_out: int, stride: int = 2):
        super().__init__()
        self.conv_a = nn.Sequential(
            nn.BatchNorm2d(c_in), nn.ReLU(inplace=False),
            nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False),
        )
        self.conv_b = nn.Sequential(
            nn.BatchNorm2d(c_out), nn.ReLU(inplace=False),
            nn.Conv2d(c_out, c_out, 3, stride=1, padding=1, bias=False),
        )
        self.downsample = nn.Sequential(
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(c_in, c_out, 1, stride=1, padding=0, bias=False),
        )

    def forward(self, x):
        out = self.conv_b(self.conv_a(x))
        return out + self.downsample(x)


class NB201Supernet(nn.Module):
    def __init__(self, num_classes: int, base_channels: int = 16):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
        )

        self.cells = nn.ModuleList()
        self.cell_kind: List[str] = []  # "search" or "reduce", for bookkeeping
        cell_position = 1
        c_prev = base_channels
        for stage_idx, c_stage in enumerate(STAGE_CHANNELS):
            if stage_idx > 0:
                self.cells.append(ResNetBasicBlock(c_prev, c_stage, stride=2))
                self.cell_kind.append("reduce")
                c_prev = c_stage
            for _ in range(CELLS_PER_STAGE):
                self.cells.append(NB201SearchCell(c_prev, c_stage, stride=1, cell_position=cell_position))
                self.cell_kind.append("search")
                cell_position += 1
                c_prev = c_stage

        self.total_positions = cell_position - 1  # = CELLS_PER_STAGE * NUM_STAGES = 15
        self.lastact = nn.Sequential(nn.BatchNorm2d(c_prev), nn.ReLU(inplace=True))
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c_prev, num_classes)

        num_search_cells = sum(1 for k in self.cell_kind if k == "search")
        self.arch_logits = nn.Parameter(1e-3 * torch.randn(NUM_EDGES, len(NB201_PRIMITIVES)))
        self._num_search_cells = num_search_cells

    def sample_architecture(self, tau: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample ONE discrete architecture (shared across all search-cell
        instances, per Sec. 4.2.2) via Gumbel-softmax; returns
        (hardwts, indices), each of shape (NUM_EDGES, ...)."""
        return gumbel_softmax_sample(self.arch_logits, tau)

    def forward(self, x: torch.Tensor, hardwts: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        feat = self.stem(x)
        for cell, kind in zip(self.cells, self.cell_kind):
            if kind == "reduce":
                feat = cell(feat)
            else:
                feat = cell(feat, hardwts, indices)
        feat = self.lastact(feat)
        feat = self.global_pool(feat).flatten(1)
        return self.classifier(feat)

    # ---- sensitivity / archive plumbing -----------------------------------

    def all_node_param_map(self) -> Dict[Tuple[int, str, int], List[nn.Parameter]]:
        out: Dict[Tuple[int, str, int], List[nn.Parameter]] = {}
        for cell, kind in zip(self.cells, self.cell_kind):
            if kind == "search":
                out.update(cell.all_node_param_map())
        return out

    def active_node_keys(self, indices: torch.Tensor):
        keys = []
        for cell, kind in zip(self.cells, self.cell_kind):
            if kind == "search":
                keys.extend(cell.active_node_keys(indices))
        return keys

    def active_params(self, indices: torch.Tensor) -> List[nn.Parameter]:
        params = []
        for cell, kind in zip(self.cells, self.cell_kind):
            if kind == "search":
                params.extend(cell.active_params(indices))
        return params

    def genotype_string(self, indices: torch.Tensor) -> str:
        """NAS-Bench-201 arch-string format: |op~i|+|op~i|op~i|+|op~i|op~i|op~i|,
        matching the format used by the official benchmark API (verified
        against api.arch(0) during setup)."""
        from .cell import EDGE_LIST
        # group edges by target node j, in increasing i order, per official format
        by_node: Dict[int, List[str]] = {}
        for e_idx, (j, i) in enumerate(EDGE_LIST):
            op_name = NB201_PRIMITIVES[indices[e_idx].item()]
            by_node.setdefault(j, []).append(f"{op_name}~{i}")
        parts = []
        for j in range(1, 4):
            parts.append("|" + "|".join(by_node[j]) + "|")
        return "+".join(parts)
