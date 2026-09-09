"""Standard DARTS macro supernet (Sec. 4.2.1): stem -> stack of cells with
reduction cells at 1/3 and 2/3 depth, doubling channels at each reduction --
the canonical DARTS search-space skeleton (Liu et al., ICLR 2019),
independently implemented from its published architecture description."""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .cell import DARTSSearchCell, MULTIPLIER, NUM_EDGES, STEPS
from ..genotypes import DARTS_PRIMITIVES
from ..nb201.cell import gumbel_softmax_sample


class DARTSSupernet(nn.Module):
    def __init__(self, num_classes: int, init_channels: int = 16, layers: int = 8,
                 steps: int = STEPS, multiplier: int = MULTIPLIER, stem_multiplier: int = 3):
        super().__init__()
        self.layers = layers
        c_curr = stem_multiplier * init_channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, c_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_curr),
        )

        c_prev_prev, c_prev, c_curr = c_curr, c_curr, init_channels
        self.cells = nn.ModuleList()
        reduction_prev = False
        self.reduction_flags: List[bool] = []
        for i in range(layers):
            if i in (layers // 3, 2 * layers // 3):
                c_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = DARTSSearchCell(steps, multiplier, c_prev_prev, c_prev, c_curr,
                                    reduction, reduction_prev, cell_position=i + 1)
            self.cells.append(cell)
            self.reduction_flags.append(reduction)
            reduction_prev = reduction
            c_prev_prev, c_prev = c_prev, multiplier * c_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c_prev, num_classes)
        self.total_positions = layers

        self.normal_logits = nn.Parameter(1e-3 * torch.randn(NUM_EDGES, len(DARTS_PRIMITIVES)))
        self.reduce_logits = nn.Parameter(1e-3 * torch.randn(NUM_EDGES, len(DARTS_PRIMITIVES)))

    def sample_architecture(self, tau: float):
        n_hardwts, n_idx = gumbel_softmax_sample(self.normal_logits, tau)
        r_hardwts, r_idx = gumbel_softmax_sample(self.reduce_logits, tau)
        return n_hardwts, n_idx, r_hardwts, r_idx

    def forward(self, x: torch.Tensor, n_hardwts, n_idx, r_hardwts, r_idx) -> torch.Tensor:
        s0 = s1 = self.stem(x)
        for cell in self.cells:
            if cell.reduction:
                s0, s1 = s1, cell(s0, s1, r_hardwts, r_idx)
            else:
                s0, s1 = s1, cell(s0, s1, n_hardwts, n_idx)
        out = self.global_pooling(s1).flatten(1)
        return self.classifier(out)

    # -- sensitivity / archive plumbing (mirrors nb201/supernet.py) --------

    def all_node_param_map(self) -> Dict[Tuple[int, str, int], List[nn.Parameter]]:
        out: Dict[Tuple[int, str, int], List[nn.Parameter]] = {}
        for cell in self.cells:
            out.update(cell.all_node_param_map())
        return out

    def active_node_keys(self, n_idx, r_idx):
        keys = []
        for cell in self.cells:
            idx = r_idx if cell.reduction else n_idx
            keys.extend(cell.active_node_keys(idx))
        return keys

    def active_params(self, n_idx, r_idx) -> List[nn.Parameter]:
        params = []
        for cell in self.cells:
            idx = r_idx if cell.reduction else n_idx
            params.extend(cell.active_params(idx))
        return params
