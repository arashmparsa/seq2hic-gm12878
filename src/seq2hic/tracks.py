"""bigWig -> per-chromosome float16 memmap.

Motivation: `pyBigWig` ships no Windows wheel, and hauling a bigWig reader into
the training loop makes the DataLoader workers fight over a single file handle
anyway. So the conversion happens once during `prepare`, and training reads
plain numpy memmaps -- fast, fork-safe, and dependency-free.

Three backends, tried in this order under `backend: auto`:
  pybigwig  -- native, best (Linux/macOS/WSL)
  ucsc      -- UCSC `bigWigToBedGraph` binary, set atac.ucsc_bigwig_to_bedgraph
  npy       -- you already converted it; expects <proc>/atac/<chrom>.npy
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from tqdm import tqdm


def _have_pybigwig() -> bool:
    try:
        import pyBigWig  # noqa: F401
    except ImportError:
        return False
    return True


def resolve_backend(requested: str, ucsc_tool: str | None) -> str:
    if requested != "auto":
        return requested
    if _have_pybigwig():
        return "pybigwig"
    if ucsc_tool and Path(ucsc_tool).exists():
        return "ucsc"
    if shutil.which("bigWigToBedGraph"):
        return "ucsc"
    raise RuntimeError(
        "No bigWig backend available.\n"
        "  Linux/macOS/WSL: pip install pyBigWig\n"
        "  Windows:         download bigWigToBedGraph.exe from\n"
        "                   https://hgdownload.soe.ucsc.edu/admin/exe/win32.x86_64/\n"
        "                   and set atac.ucsc_bigwig_to_bedgraph in your config\n"
        "  Or pre-convert to <proc>/atac/<chrom>.npy and set atac.backend: npy"
    )


def bigwig_to_memmaps(
    bigwig: str | Path,
    out_dir: str | Path,
    chrom_sizes: dict[str, int],
    backend: str = "auto",
    ucsc_tool: str | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Write one float16 .npy per chromosome. Returns {chrom: path}.

    float16 is intentional: the values are log1p-transformed fold-change in
    roughly [0, 6], where fp16's ~3 decimal digits are far more precision than
    the underlying signal has, and it halves 3 GB of memmap to 1.5 GB.
    """
    bigwig, out_dir = Path(bigwig), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    backend = resolve_backend(backend, ucsc_tool)

    paths: dict[str, Path] = {}
    todo = []
    for chrom, size in chrom_sizes.items():
        dest = out_dir / f"{chrom}.npy"
        paths[chrom] = dest
        if dest.exists() and not overwrite and _npy_len(dest) == size:
            continue
        todo.append((chrom, size, dest))

    if not todo:
        return paths

    print(f"converting {bigwig.name} -> {len(todo)} chromosome memmaps [{backend}]")
    if backend == "pybigwig":
        _convert_pybigwig(bigwig, todo)
    elif backend == "ucsc":
        _convert_ucsc(bigwig, todo, ucsc_tool)
    elif backend == "npy":
        missing = [c for c, _, d in todo if not d.exists()]
        raise FileNotFoundError(
            f"backend 'npy' but these are missing or wrong length: {missing[:5]}"
        )
    else:
        raise ValueError(f"unknown bigWig backend: {backend}")
    return paths


def _npy_len(path: Path) -> int:
    try:
        return int(np.load(path, mmap_mode="r").shape[0])
    except Exception:
        return -1


def _convert_pybigwig(bigwig: Path, todo: list[tuple[str, int, Path]]) -> None:
    import pyBigWig

    bw = pyBigWig.open(str(bigwig))
    try:
        available = bw.chroms()
        for chrom, size, dest in tqdm(todo, desc="  bigWig", unit="chrom"):
            if chrom not in available:
                raise KeyError(f"{chrom} absent from {bigwig.name}; assembly mismatch?")
            # Chunked so a 250 Mb chromosome does not need a float64 temp array.
            arr = np.zeros(size, dtype=np.float16)
            step = 1 << 24
            for s in range(0, size, step):
                e = min(size, s + step)
                vals = bw.values(chrom, s, e, numpy=True)
                np.nan_to_num(vals, copy=False)
                arr[s:e] = vals.astype(np.float16)
            _atomic_save(dest, arr)
    finally:
        bw.close()


def _convert_ucsc(bigwig: Path, todo: list[tuple[str, int, Path]], tool: str | None) -> None:
    exe = tool or shutil.which("bigWigToBedGraph")
    if not exe:
        raise RuntimeError("bigWigToBedGraph not found")
    for chrom, size, dest in tqdm(todo, desc="  bigWig", unit="chrom"):
        with tempfile.TemporaryDirectory() as td:
            bg = Path(td) / f"{chrom}.bedGraph"
            subprocess.run(
                [str(exe), f"-chrom={chrom}", str(bigwig), str(bg)],
                check=True,
                capture_output=True,
            )
            arr = np.zeros(size, dtype=np.float16)
            with open(bg, "r", encoding="utf-8") as fh:
                for line in fh:
                    parts = line.split("\t")
                    if len(parts) < 4:
                        continue
                    s, e, v = int(parts[1]), int(parts[2]), float(parts[3])
                    if s >= size:
                        continue
                    arr[s : min(e, size)] = np.float16(v)
            _atomic_save(dest, arr)


def _atomic_save(dest: Path, arr: np.ndarray) -> None:
    """Write via .tmp then rename, so an interrupted prepare never leaves a
    truncated .npy that looks complete on the next run."""
    tmp = dest.with_suffix(".tmp.npy")
    np.save(tmp, arr)
    tmp.replace(dest)


class TrackReader:
    """Lazy per-chromosome memmap access with transform applied on read."""

    def __init__(self, dir_path: str | Path, transform: str = "log1p", clip: float = 32.0):
        self.dir = Path(dir_path)
        self.transform = transform
        self.clip = clip
        self._cache: dict[str, np.ndarray] = {}

    def _mm(self, chrom: str) -> np.ndarray:
        if chrom not in self._cache:
            path = self.dir / f"{chrom}.npy"
            if not path.exists():
                raise FileNotFoundError(f"{path} missing; run `seq2hic.prepare` first")
            self._cache[chrom] = np.load(path, mmap_mode="r")
        return self._cache[chrom]

    def values(self, chrom: str, start: int, end: int) -> np.ndarray:
        """(L,) float32, zero-padded out of bounds, transformed and clipped."""
        mm = self._mm(chrom)
        size = mm.shape[0]
        lpad, rpad = max(0, -start), max(0, end - size)
        core = np.asarray(mm[max(0, start) : min(size, end)], dtype=np.float32)
        if lpad or rpad:
            core = np.concatenate(
                [np.zeros(lpad, np.float32), core, np.zeros(rpad, np.float32)]
            )
        np.nan_to_num(core, copy=False)
        np.clip(core, 0.0, self.clip, out=core)
        if self.transform == "log1p":
            np.log1p(core, out=core)
        elif self.transform not in ("none", None):
            raise ValueError(f"unknown atac.transform: {self.transform}")
        return core
