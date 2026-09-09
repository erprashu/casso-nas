"""Fixed discrete evaluation network built from a discovered Genotype
(Sec. 4.3.1: "Discovered architectures are retrained from scratch
following [DARTS]"). Unlike DARTSSearchCell (all 8 candidate ops per edge,
Gumbel-softmax selection), each edge here has exactly ONE fixed operation
as specified by the genotype -- there is no more searching at this stage.

Includes the two standard DARTS-lineage retrain-time regularizers this
protocol relies on: drop-path (stochastically zeroing an op's output with
increasing probability over training) and an auxiliary classifier head
attached at 2/3 depth (weighted into the loss, standard Inception-style
deep supervision), both well-documented techniques from the original DARTS
paper (Liu et al., ICLR 2019), independently implemented here.
"""

from typing import List

import torch
import torch.nn as nn

from ..genotypes import Genotype
from ..ops import FactorizedReduce, ReLUConvBN, build_op


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob <= 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        mask = torch.empty(x.size(0), 1, 1, 1, device=x.device).bernoulli_(keep_prob)
        return x / keep_prob * mask


class FixedCell(nn.Module):
    def __init__(self, genotype: Genotype, c_prev_prev: int, c_prev: int, c: int,
                 reduction: bool, reduction_prev: bool):
        super().__init__()
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(c_prev_prev, c)
        else:
            self.preprocess0 = ReLUConvBN(c_prev_prev, c, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(c_prev, c, 1, 1, 0)

        gene, concat = (genotype.reduce, genotype.reduce_concat) if reduction \
            else (genotype.normal, genotype.normal_concat)
        self.concat = concat
        self.reduction = reduction

        self.ops = nn.ModuleList()
        self.indices: List[int] = []
        for op_name, pred_idx in gene:
            stride = 2 if reduction and pred_idx < 2 else 1
            self.ops.append(build_op(op_name, c, c, stride))
            self.indices.append(pred_idx)

    def forward(self, s0: torch.Tensor, s1: torch.Tensor, drop_prob: float) -> torch.Tensor:
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        # gene lists edges 2-per-node in order; consume them 2 at a time.
        for i in range(0, len(self.ops), 2):
            h1 = states[self.indices[i]]
            h2 = states[self.indices[i + 1]]
            out1 = self.ops[i](h1)
            out2 = self.ops[i + 1](h2)
            if self.training and drop_prob > 0.0:
                if not isinstance(self.ops[i], nn.Identity):
                    out1 = DropPath(drop_prob)(out1)
                if not isinstance(self.ops[i + 1], nn.Identity):
                    out2 = DropPath(drop_prob)(out2)
            states.append(out1 + out2)
        return torch.cat([states[i] for i in self.concat], dim=1)


class AuxiliaryHeadCIFAR(nn.Module):
    """Standard DARTS-lineage auxiliary classifier, attached at 2/3 depth
    during training only, weighted 0.4 into the total loss (Sec. 4.3.1
    follows the standard DARTS retrain protocol, which uses this)."""

    def __init__(self, c: int, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),
            nn.Conv2d(c, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.flatten(1))


class NetworkCIFAR(nn.Module):
    def __init__(self, num_classes: int, genotype: Genotype, init_channels: int = 36,
                 layers: int = 20, auxiliary: bool = True, stem_multiplier: int = 3):
        super().__init__()
        self.auxiliary = auxiliary
        self.drop_path_prob = 0.0  # set externally per epoch during training

        c_curr = stem_multiplier * init_channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, c_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_curr),
        )

        c_prev_prev, c_prev, c_curr = c_curr, c_curr, init_channels
        self.cells = nn.ModuleList()
        reduction_prev = False
        self.aux_position = 2 * layers // 3
        c_to_auxiliary = c_prev
        for i in range(layers):
            if i in (layers // 3, 2 * layers // 3):
                c_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = FixedCell(genotype, c_prev_prev, c_prev, c_curr, reduction, reduction_prev)
            self.cells.append(cell)
            reduction_prev = reduction
            multiplier = len(genotype.normal_concat if not reduction else genotype.reduce_concat)
            c_prev_prev, c_prev = c_prev, multiplier * c_curr
            if i == self.aux_position:
                c_to_auxiliary = c_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(c_to_auxiliary, num_classes)

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c_prev, num_classes)

    def forward(self, x: torch.Tensor):
        logits_aux = None
        s0 = s1 = self.stem(x)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == self.aux_position and self.auxiliary and self.training:
                logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1).flatten(1)
        logits = self.classifier(out)
        return logits, logits_aux
