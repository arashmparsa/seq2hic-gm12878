"""FASTA access and one-hot encoding.

One-hot uses a 4-row encoding with N (and any other ambiguity code) as an
all-zero column rather than 0.25 everywhere. All-zero is the honest encoding:
0.25 tells the first conv layer "equally likely A/C/G/T", which is a claim about
the sequence that the assembly gap does not support.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np

_BASE_TO_IDX = np.full(256, 255, dtype=np.uint8)
for _i, _b in enumerate("ACGT"):
    _BASE_TO_IDX[ord(_b)] = _i
    _BASE_TO_IDX[ord(_b.lower())] = _i

# A<->T, C<->G under one-hot row order ACGT: reverse the row axis.
_COMPLEMENT_ROW_ORDER = np.array([3, 2, 1, 0])


class Genome:
    """Thin pyfaidx wrapper that returns uppercase byte sequences."""

    def __init__(self, fasta_path: str | Path):
        self.path = Path(fasta_path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} not found. Run `python -m seq2hic.encode genome`."
            )
        self._pid = None
        self._handle = None
        self.chrom_sizes = {name: len(rec) for name, rec in self._fa.records.items()}

    @property
    def _fa(self):
        # pyfaidx holds one open file handle. DataLoader workers are forked, so a
        # handle opened in the parent is shared by every worker and they race on
        # its seek offset -- reads come back the wrong length or, worse, the
        # wrong bases. Reopen once per process.
        if self._pid != os.getpid():
            from pyfaidx import Fasta

            self._handle = Fasta(str(self.path), as_raw=True, sequence_always_upper=True)
            self._pid = os.getpid()
        return self._handle

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle"] = None
        state["_pid"] = None
        return state

    def sequence(self, chrom: str, start: int, end: int) -> bytes:
        """Half-open [start, end). Out-of-bounds is N-padded rather than clipped,
        so the caller always gets exactly end-start bases."""
        size = self.chrom_sizes[chrom]
        lpad = max(0, -start)
        rpad = max(0, end - size)
        s = self._fa[chrom][max(0, start) : min(size, end)]
        seq = s.encode("ascii") if isinstance(s, str) else bytes(s)
        if lpad or rpad:
            seq = b"N" * lpad + seq + b"N" * rpad
        want = end - start
        if len(seq) != want:
            raise ValueError(f"{chrom}:{start}-{end} returned {len(seq)} bp, expected {want}")
        return seq

    def one_hot(self, chrom: str, start: int, end: int) -> np.ndarray:
        return one_hot_encode(self.sequence(chrom, start, end))


def one_hot_encode(seq: bytes) -> np.ndarray:
    """(4, L) float32. Ambiguity codes become all-zero columns."""
    idx = _BASE_TO_IDX[np.frombuffer(seq, dtype=np.uint8)]
    out = np.zeros((4, idx.size), dtype=np.float32)
    known = idx != 255
    out[idx[known], np.nonzero(known)[0]] = 1.0
    return out


def reverse_complement(x: np.ndarray) -> np.ndarray:
    """Reverse-complement a (4, L) or (C, L) one-hot-leading array.

    Only the first four rows are complemented; any extra channels (e.g. ATAC)
    are reversed but not row-permuted, which is what you want -- accessibility
    is strand-agnostic.
    """
    out = x[:, ::-1].copy()
    out[:4] = out[_COMPLEMENT_ROW_ORDER]
    return out


def n_fraction(seq: bytes) -> float:
    arr = np.frombuffer(seq, dtype=np.uint8)
    return float(np.mean(_BASE_TO_IDX[arr] == 255)) if arr.size else 1.0


@lru_cache(maxsize=4)
def load_genome(fasta_path: str) -> Genome:
    return Genome(fasta_path)


def read_tss_bed(path: str | Path, chroms: set[str] | None = None) -> dict[str, np.ndarray]:
    """Read the BED6 produced by `encode.py genome` into {chrom: sorted positions}."""
    from collections import defaultdict

    acc: dict[str, list[int]] = defaultdict(list)
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip() or line.startswith(("#", "track")):
                continue
            parts = line.split("\t")
            chrom, start = parts[0], int(parts[1])
            if chroms and chrom not in chroms:
                continue
            acc[chrom].append(start)
    return {c: np.sort(np.asarray(v, dtype=np.int64)) for c, v in acc.items()}


def read_narrowpeak(path: str | Path, chroms: set[str] | None = None) -> dict[str, np.ndarray]:
    """Read a narrowPeak/BED into {chrom: (n, 2) int64 [start, end)} sorted by start."""
    from collections import defaultdict

    acc: dict[str, list[tuple[int, int]]] = defaultdict(list)
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            parts = line.split("\t")
            chrom = parts[0]
            if chroms and chrom not in chroms:
                continue
            acc[chrom].append((int(parts[1]), int(parts[2])))
    out = {}
    for c, v in acc.items():
        arr = np.asarray(v, dtype=np.int64)
        out[c] = arr[np.argsort(arr[:, 0])]
    return out
