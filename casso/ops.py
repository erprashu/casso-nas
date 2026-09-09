"""Candidate operation primitives shared by the DARTS and NAS-Bench-201 cell
search spaces (Sec. 4.2.1 / 4.2.2). These are standard building blocks used
throughout the differentiable-NAS literature (ReLU-Conv-BN stacks, separable
and dilated convolutions, pooling, skip-connect, and a structural zero op);
implemented here from their well-known mathematical definitions rather than
copied from any single codebase.
"""

import torch
import torch.nn as nn


class ReLUConvBN(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(c_in, c_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(c_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):
    """Depthwise-separable conv, applied twice (as in DARTS)."""

    def __init__(self, c_in, c_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(c_in, c_in, kernel_size, stride=stride, padding=padding, groups=c_in, bias=False),
            nn.Conv2d(c_in, c_in, 1, padding=0, bias=False),
            nn.BatchNorm2d(c_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(c_in, c_in, kernel_size, stride=1, padding=padding, groups=c_in, bias=False),
            nn.Conv2d(c_in, c_out, 1, padding=0, bias=False),
            nn.BatchNorm2d(c_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation, affine=True):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(c_in, c_in, kernel_size, stride=stride, padding=padding,
                      dilation=dilation, groups=c_in, bias=False),
            nn.Conv2d(c_in, c_out, 1, padding=0, bias=False),
            nn.BatchNorm2d(c_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):
    def forward(self, x):
        return x


class Zero(nn.Module):
    """Structural zero: outputs an all-zero tensor with the *target* (c_out)
    channel count and correctly strided spatial size, regardless of the
    input's own channel count. This matters whenever c_in != c_out on an
    edge (e.g. the very first search cell of a stage, whose input comes
    from the stem/previous stage and may not match that stage's channel
    width) -- naively zeroing the input in place (`x.mul(0)`) would silently
    produce the wrong channel count and break summation with sibling edges."""

    def __init__(self, c_in: int, c_out: int, stride: int):
        super().__init__()
        self.c_out = c_out
        self.stride = stride

    def forward(self, x):
        n, _, h, w = x.shape
        h_out = (h + self.stride - 1) // self.stride
        w_out = (w + self.stride - 1) // self.stride
        return x.new_zeros(n, self.c_out, h_out, w_out)


class FactorizedReduce(nn.Module):
    """Channel-and-spatial-dim reduction used for skip-connect under stride=2."""

    def __init__(self, c_in, c_out, affine=True):
        super().__init__()
        assert c_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(c_in, c_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(c_in, c_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(c_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        return self.bn(out)


class Pooling(nn.Module):
    def __init__(self, mode, c_in, c_out, stride, affine=True):
        super().__init__()
        self.preprocess = None
        if c_in != c_out:
            self.preprocess = ReLUConvBN(c_in, c_out, 1, 1, 0, affine=affine)
        if mode == "avg":
            self.op = nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)
        elif mode == "max":
            self.op = nn.MaxPool2d(3, stride=stride, padding=1)
        else:
            raise ValueError(f"Unknown pooling mode {mode}")

    def forward(self, x):
        if self.preprocess is not None:
            x = self.preprocess(x)
        return self.op(x)


def build_op(name: str, c_in: int, c_out: int, stride: int, affine: bool = True) -> nn.Module:
    """Instantiate a candidate operation by its PRIMITIVE name (Sec. 4.2.1/4.2.2)."""
    if name == "none":
        return Zero(c_in, c_out, stride)
    if name == "skip_connect":
        if stride == 1 and c_in == c_out:
            return Identity()
        if stride == 1:
            # Channel change only (no spatial downsampling): a 1x1 conv
            # projection, NOT FactorizedReduce, which always halves spatial
            # size internally regardless of the `stride` argument passed to
            # it -- using it here would silently downsample when the edge's
            # actual stride is 1 (this exact bug was caught by the smoke test).
            return ReLUConvBN(c_in, c_out, 1, 1, 0, affine=affine)
        return FactorizedReduce(c_in, c_out, affine=affine)
    if name == "avg_pool_3x3":
        return Pooling("avg", c_in, c_out, stride, affine=affine)
    if name == "max_pool_3x3":
        return Pooling("max", c_in, c_out, stride, affine=affine)
    if name == "nor_conv_1x1":
        return ReLUConvBN(c_in, c_out, 1, stride, 0, affine=affine)
    if name == "nor_conv_3x3":
        return ReLUConvBN(c_in, c_out, 3, stride, 1, affine=affine)
    if name == "sep_conv_3x3":
        return SepConv(c_in, c_out, 3, stride, 1, affine=affine)
    if name == "sep_conv_5x5":
        return SepConv(c_in, c_out, 5, stride, 2, affine=affine)
    if name == "dil_conv_3x3":
        return DilConv(c_in, c_out, 3, stride, 2, dilation=2, affine=affine)
    if name == "dil_conv_5x5":
        return DilConv(c_in, c_out, 5, stride, 4, dilation=2, affine=affine)
    raise ValueError(f"Unknown operation {name}")
