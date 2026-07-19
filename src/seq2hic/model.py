"""Pure dilated CNN: 1D sequence trunk -> outer concat -> 2D dilated head.

No attention. The whole point of the Akita-style design is that dilation alone
buys a receptive field wider than the window, and the 2D stage sees the full
pairwise structure directly rather than having to route information through a
1D bottleneck at inference time.

The sequence axis is downsampled by an explicit list of pool factors whose
product must equal bp-per-bin. At the default 2^21 bp / 256 bins that product is
8,192 = 2^13, i.e. thirteen stride-2 stages. The factors are configured rather
than assumed, so a non-power-of-two grid (e.g. 10 kb bins = 2^4 * 5^4) is a
config change and not a code change.

Shapes, end to end, at the default config:
    (B, 4|5, 2_097_152)   input
    (B, 224, 256)         after stem + 13 pool-2 stages (product 8_192)
    (B, 96,  256)         after dilated 1D tower + bottleneck
    (B, 193, 256, 256)    after outer concat (96 + 96 + 1 distance channel)
    (B, 256, 256)         output, symmetric
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock1d(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel: int, pool: int, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, kernel, padding=kernel // 2, bias=False)
        self.norm = nn.BatchNorm1d(c_out)
        # Non-power-of-2 pools are the norm here (10_000 = 2^4 * 5^4), so the
        # kernel and stride are set explicitly rather than left to default.
        self.pool = nn.MaxPool1d(kernel_size=pool, stride=pool) if pool > 1 else nn.Identity()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.pool(F.gelu(self.norm(self.conv(x)))))


class DilatedResidual1d(nn.Module):
    """Bottlenecked dilated residual block (Basenji/Akita style)."""

    def __init__(self, channels: int, dilation: int, kernel: int = 3, dropout: float = 0.0):
        super().__init__()
        mid = max(8, channels // 2)
        pad = dilation * (kernel - 1) // 2
        self.conv1 = nn.Conv1d(channels, mid, kernel, dilation=dilation, padding=pad, bias=False)
        self.norm1 = nn.BatchNorm1d(mid)
        self.conv2 = nn.Conv1d(mid, channels, 1, bias=False)
        self.norm2 = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.norm1(self.conv1(x)))
        h = self.drop(self.norm2(self.conv2(h)))
        return F.gelu(x + h)


class DilatedResidual2d(nn.Module):
    """Dilated 2D residual block with an explicit symmetrisation.

    Symmetrising inside every block, rather than only at the output, keeps the
    representation on the symmetric manifold throughout. A contact map is
    symmetric by construction; letting the network drift off that and correcting
    once at the end wastes capacity learning a constraint we already know.
    """

    def __init__(self, channels: int, dilation: int, kernel: int = 3, dropout: float = 0.0):
        super().__init__()
        mid = max(8, channels // 2)
        pad = dilation * (kernel - 1) // 2
        self.conv1 = nn.Conv2d(channels, mid, kernel, dilation=dilation, padding=pad, bias=False)
        self.norm1 = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid, channels, 1, bias=False)
        self.norm2 = nn.BatchNorm2d(channels)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.norm1(self.conv1(x)))
        h = self.drop(self.norm2(self.conv2(h)))
        out = F.gelu(x + h)
        return 0.5 * (out + out.transpose(-1, -2))


def outer_concat(x: torch.Tensor, distance_channel: bool) -> torch.Tensor:
    """(B, C, N) -> (B, 2C[+1], N, N).

    Element (i, j) carries the representation of bin i and of bin j, plus
    optionally log(1 + |i-j|) so the 2D stack does not have to rediscover
    genomic distance -- which it otherwise spends most of its early capacity on.
    """
    b, c, n = x.shape
    xi = x.unsqueeze(-1).expand(b, c, n, n)
    xj = x.unsqueeze(-2).expand(b, c, n, n)
    parts = [xi, xj]
    if distance_channel:
        idx = torch.arange(n, device=x.device, dtype=x.dtype)
        d = (idx[:, None] - idx[None, :]).abs()
        d = torch.log1p(d) / math.log(1.0 + n)
        parts.append(d.expand(b, 1, n, n))
    return torch.cat(parts, dim=1)


class Seq2HiC(nn.Module):
    def __init__(self, in_channels: int, n_bins: int, mcfg: dict):
        super().__init__()
        self.in_channels = in_channels
        self.n_bins = n_bins
        self.use_distance = bool(mcfg.get("use_distance_channel", True))
        dropout = float(mcfg.get("dropout", 0.0))

        chans = list(mcfg["trunk_channels"])
        pools = [int(p) for p in mcfg["trunk_pools"]]
        if len(pools) != len(chans):
            raise ValueError(
                f"trunk_pools has {len(pools)} entries but trunk_channels has {len(chans)}; "
                "they describe the same stages and must be the same length"
            )
        self.trunk_pools = pools
        self.downsample_factor = math.prod(pools)

        stem_k = int(mcfg["stem_kernel"])
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, chans[0], stem_k, padding=stem_k // 2, bias=False),
            nn.BatchNorm1d(chans[0]),
            nn.GELU(),
        )

        kernel = int(mcfg["trunk_kernel"])
        blocks = []
        c_prev = chans[0]
        for c, p in zip(chans, pools):
            blocks.append(ConvBlock1d(c_prev, c, kernel, pool=p, dropout=dropout))
            c_prev = c
        self.trunk = nn.Sequential(*blocks)

        self.dilated1d = nn.Sequential(
            *[
                DilatedResidual1d(c_prev, dilation=2**i, dropout=dropout)
                for i in range(int(mcfg["dilated1d_blocks"]))
            ]
        )

        c1 = int(mcfg["bottleneck_1d"])
        self.bottleneck = nn.Sequential(
            nn.Conv1d(c_prev, c1, 1, bias=False), nn.BatchNorm1d(c1), nn.GELU()
        )

        c2 = int(mcfg["channels_2d"])
        c_pair = 2 * c1 + (1 if self.use_distance else 0)
        self.to2d = nn.Sequential(
            nn.Conv2d(c_pair, c2, 1, bias=False), nn.BatchNorm2d(c2), nn.GELU()
        )
        base = int(mcfg.get("dilation_base", 2))
        self.dilation_base = base
        self.head = nn.Sequential(
            *[
                DilatedResidual2d(c2, dilation=base**i, dropout=dropout)
                for i in range(int(mcfg["dilated2d_blocks"]))
            ]
        )
        self.out = nn.Conv2d(c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"model built for {self.in_channels} input channels, got {x.shape[1]}. "
                "Did you point a seq_only checkpoint at a seq+ATAC dataset?"
            )
        h = self.trunk(self.stem(x))
        if h.shape[-1] != self.n_bins:
            raise RuntimeError(
                f"trunk produced {h.shape[-1]} bins, expected {self.n_bins}; "
                f"window.bp / prod(trunk_pools)={self.downsample_factor} "
                "must equal window.n_bins"
            )
        h = self.bottleneck(self.dilated1d(h))
        p = self.to2d(outer_concat(h, self.use_distance))
        y = self.out(self.head(p)).squeeze(1)
        return 0.5 * (y + y.transpose(-1, -2))

    @property
    def receptive_field_2d_bins(self) -> int:
        """Bins of 2D context each output element can see (excluding the trunk)."""
        rf = 1
        for i in range(len(self.head)):
            rf += 2 * (self.dilation_base**i)
        return rf

    def covers_window(self) -> bool:
        """Whether the 2D stack can see the full window from any output bin.

        If False, output bins near the corners are structurally unable to see
        their own anchors -- the model is not merely under-trained, it is
        under-specified. Increase model.dilated2d_blocks.
        """
        return self.receptive_field_2d_bins >= self.n_bins


def build_model(cfg) -> Seq2HiC:
    in_ch = 4 + (1 if cfg["inputs"]["atac"] else 0)
    return Seq2HiC(in_ch, int(cfg["window"]["n_bins"]), dict(cfg["model"]))


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE over valid upper-triangle elements only.

    Returns a zero that still carries grad when a batch has no valid elements,
    so a pathological window cannot produce a NaN loss and poison the optimiser.
    """
    n = pred.shape[-1]
    tri = torch.triu(torch.ones(n, n, dtype=torch.bool, device=pred.device), diagonal=1)
    m = mask & tri
    denom = m.sum()
    if denom == 0:
        return (pred * 0.0).sum()
    diff = (pred - target)[m]
    return (diff * diff).mean()
