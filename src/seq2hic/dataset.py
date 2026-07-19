"""Window index, chromosome splits, and the torch Dataset.

Targets are materialised to disk by `prepare.py` as a single float16 array plus
a packed-bit validity mask, so the training loop never touches cooler. That
matters more than it sounds: cooler handles are not fork-safe, and a DataLoader
with workers > 0 reading .mcool directly deadlocks in ways that look like a
hung GPU.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import Cfg, resolve
from .genome import load_genome, one_hot_encode, reverse_complement
from .tracks import TrackReader


@dataclass(frozen=True)
class Window:
    chrom: str
    start: int
    end: int
    index: int  # row in the materialised target array


def build_windows(
    chrom_sizes: dict[str, int], chroms: list[str], window_bp: int, stride: int
) -> list[tuple[str, int, int]]:
    """Tiling windows fully contained within each chromosome.

    Windows never straddle chromosomes, and the final partial window is dropped
    rather than padded -- a half-N window contributes nothing but teaches the
    model that the map goes blank at chromosome ends.
    """
    out = []
    for c in chroms:
        size = chrom_sizes.get(c)
        if not size or size < window_bp:
            continue
        for start in range(0, size - window_bp + 1, stride):
            out.append((c, start, start + window_bp))
    return out


def split_windows(
    windows: list[Window], split: dict[str, list[str]], which: str
) -> list[Window]:
    keep = set(split[which])
    return [w for w in windows if w.chrom in keep]


class WindowIndex:
    """The on-disk artefact produced by prepare: window list + target store."""

    def __init__(self, proc_dir: str | Path):
        self.dir = Path(proc_dir)
        meta_path = self.dir / "index.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{meta_path} missing. Run `python -m seq2hic.prepare --config <cfg>`."
            )
        self.meta = json.loads(meta_path.read_text())
        self.windows = [Window(**w) for w in self.meta["windows"]]
        self.n_bins = int(self.meta["n_bins"])
        self.window_bp = int(self.meta["window_bp"])
        self.expected = np.load(self.dir / "expected.npy")

        self._targets = np.load(self.dir / "targets.npy", mmap_mode="r")
        self._masks = np.load(self.dir / "masks.npy", mmap_mode="r")

    def target(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        t = np.asarray(self._targets[i], dtype=np.float32)
        m = np.unpackbits(
            np.asarray(self._masks[i]), count=self.n_bins * self.n_bins
        ).reshape(self.n_bins, self.n_bins).astype(bool)
        return t, m


class HiCWindowDataset(Dataset):
    """Yields (inputs (C, L), target (n, n), mask (n, n))."""

    def __init__(self, cfg: Cfg, index: WindowIndex, which: str, training: bool):
        self.cfg = cfg
        self.index = index
        self.training = training
        self.windows = split_windows(index.windows, cfg["split"], which)
        if not self.windows:
            raise ValueError(f"no windows in split '{which}' -- check split config")

        self.window_bp = index.window_bp
        self.n_bins = index.n_bins
        self.use_atac = bool(cfg["inputs"]["atac"])
        self.synthetic = bool(cfg.get("synthetic", False))

        aug = cfg["train"]["augment"]
        self.rc = bool(aug["reverse_complement"]) and training
        self.shift_bp = int(aug["shift_bp"]) if training else 0

        if not self.synthetic:
            self.genome = load_genome(str(resolve(cfg, "genome.fasta")))
            self.chrom_sizes = self.genome.chrom_sizes
        else:
            self.genome = None
            self.chrom_sizes = {w.chrom: 10_000_000 for w in self.windows}

        self.atac: TrackReader | None = None
        if self.use_atac and not self.synthetic:
            self.atac = TrackReader(
                Path(cfg["paths"]["proc"]) / "atac",
                transform=cfg["atac"]["transform"],
                clip=float(cfg["atac"]["clip"]),
            )

        self._rng = np.random.default_rng(cfg["train"]["seed"])

    def __len__(self) -> int:
        return len(self.windows)

    def _inputs(self, chrom: str, start: int, end: int) -> np.ndarray:
        if self.synthetic:
            return _synthetic_inputs(chrom, start, end, self.use_atac)

        x = self.genome.one_hot(chrom, start, end)
        if self.use_atac:
            if self.atac is None:
                raise RuntimeError("inputs.atac is true but no ATAC track was prepared")
            a = self.atac.values(chrom, start, end)[None, :]
            x = np.concatenate([x, a.astype(np.float32)], axis=0)
        return x

    def __getitem__(self, i: int):
        w = self.windows[i]
        tgt, mask = self.index.target(w.index)

        # Sub-bin jitter only. Anything larger would shift the target grid and
        # silently misalign labels; config validation enforces shift < bin size.
        shift = int(self._rng.integers(-self.shift_bp, self.shift_bp + 1)) if self.shift_bp else 0
        start, end = w.start + shift, w.end + shift

        x = self._inputs(w.chrom, start, end)

        if self.rc and self._rng.random() < 0.5:
            x = reverse_complement(x)
            tgt = tgt[::-1, ::-1].copy()
            mask = mask[::-1, ::-1].copy()

        return (
            torch.from_numpy(np.ascontiguousarray(x)),
            torch.from_numpy(np.ascontiguousarray(tgt)),
            torch.from_numpy(np.ascontiguousarray(mask)),
        )

    def window_at(self, i: int) -> Window:
        return self.windows[i]


def _synthetic_inputs(chrom: str, start: int, end: int, use_atac: bool) -> np.ndarray:
    """Deterministic pseudo-sequence for the smoke config.

    Seeded on (chrom, start) so the same window always produces the same input
    -- otherwise the smoke test's loss curve is pure noise and tells you nothing.
    """
    rng = np.random.default_rng(abs(hash((chrom, start))) % (2**32))
    L = end - start
    idx = rng.integers(0, 4, size=L)
    x = np.zeros((4, L), dtype=np.float32)
    x[idx, np.arange(L)] = 1.0
    if use_atac:
        a = np.abs(rng.normal(0, 1, size=L)).astype(np.float32)
        x = np.concatenate([x, a[None, :]], axis=0)
    return x
