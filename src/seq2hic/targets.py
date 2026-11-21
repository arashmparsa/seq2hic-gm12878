"""Hi-C contact matrices -> model targets, following Akita's transform.

    balanced counts
      -> adaptive coarse-grain   (pool sparse pixels until they carry signal)
      -> project onto the 8,192 bp model grid
      -> per-diagonal detrend    (divide out distance decay)
      -> log
      -> clip to [-2, 2]
      -> zero the near-diagonal band, and mask it out of the loss to match

Each step exists for a specific failure it prevents:

* **Adaptive coarse-graining.** Far from the diagonal, a 5-10 kb pixel holds a
  handful of reads; its balanced value is mostly Poisson noise. Coarse-graining
  adaptively -- pooling a pixel with its neighbours only until the pooled *raw*
  count clears a threshold -- denoises the sparse periphery while leaving the
  dense near-diagonal region at full resolution. This needs the raw counts, not
  just the balanced matrix, which is why the readers fetch both.

* **Per-diagonal detrend.** Contact frequency is dominated by genomic distance.
  Skip this and a model reaches very low MSE by predicting the decay curve and
  nothing else -- it will look like it works and will have learned nothing.

* **Zeroing + masking the near-diagonal band.** The first couple of diagonals are
  dominated by ligation artefacts and self-circles. They are zeroed in the target
  *and* excluded from the loss; doing only the first would train the model to
  reproduce an artefact, and doing only the second leaves garbage in the array
  that every downstream metric would have to remember to skip.

Where the expected curve comes from is a real choice. Default `train_only`
estimates it on training chromosomes and applies it everywhere, so no
test-chromosome statistics reach the test targets. `per_chrom` is the more common
published choice and is available, but opt-in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np


class MatrixSource(Protocol):
    binsize: int

    def chroms(self) -> dict[str, int]: ...
    def fetch(self, chrom: str, start: int, end: int) -> tuple[np.ndarray, np.ndarray, int]: ...


class CoolerSource:
    """cooler-backed reader (.cool / .mcool). Pure Python, works on Windows."""

    def __init__(self, path: str | Path, prefer_resolution: int = 5000, balance: bool = True):
        import cooler

        self.path = str(path)
        self.balance = balance
        uri = self.path
        if self.path.endswith(".mcool"):
            avail = sorted(int(r.split("/")[-1]) for r in cooler.fileops.list_coolers(self.path))
            chosen = min((r for r in avail if r <= prefer_resolution), default=min(avail))
            uri = f"{self.path}::resolutions/{chosen}"
            self.available_resolutions = avail
        self._clr = cooler.Cooler(uri)
        self.binsize = int(self._clr.binsize)
        self._bal = self._clr.matrix(balance=balance, sparse=False)
        self._raw = self._clr.matrix(balance=False, sparse=False)

    def chroms(self) -> dict[str, int]:
        df = self._clr.chroms()[:]
        return dict(zip(df["name"].astype(str), df["length"].astype(int)))

    def fetch(self, chrom: str, start: int, end: int) -> tuple[np.ndarray, np.ndarray, int]:
        """(balanced, raw_counts, genomic_start_of_first_bin), snapped outward
        to native bin edges."""
        b = self.binsize
        s = (start // b) * b
        e = -(-end // b) * b
        size = self.chroms()[chrom]
        e = min(e, -(-size // b) * b)
        region = f"{chrom}:{s}-{e}"
        bal = np.asarray(self._bal.fetch(region), dtype=np.float64)
        raw = np.asarray(self._raw.fetch(region), dtype=np.float64)
        return bal, np.nan_to_num(raw, nan=0.0), s


class StrawSource:
    """hicstraw-backed reader (.hic), used when ENCODE released no .mcool."""

    def __init__(self, path: str | Path, prefer_resolution: int = 5000, balance: bool = True):
        import hicstraw

        self.path = str(path)
        self._hic = hicstraw.HiCFile(self.path)
        avail = sorted(self._hic.getResolutions())
        self.available_resolutions = avail
        self.binsize = min((r for r in avail if r <= prefer_resolution), default=min(avail))
        self._norm = "KR" if balance else "NONE"
        self._sizes = {c.name: c.length for c in self._hic.getChromosomes() if c.name.lower() != "all"}
        self._prefixed = any(k.startswith("chr") for k in self._sizes)

    def chroms(self) -> dict[str, int]:
        if self._prefixed:
            return dict(self._sizes)
        return {f"chr{k}": v for k, v in self._sizes.items()}

    def _name(self, chrom: str) -> str:
        return chrom if self._prefixed else chrom.removeprefix("chr")

    def _matrix(self, chrom: str, s: int, e: int, norm: str) -> np.ndarray:
        name = self._name(chrom)
        mzd = self._hic.getMatrixZoomData(name, name, "observed", norm, "BP", self.binsize)
        mat = np.asarray(mzd.getRecordsAsMatrix(s, e - 1, s, e - 1), dtype=np.float64)
        n = (e - s) // self.binsize
        if mat.shape != (n, n):  # straw trims trailing all-empty bins
            padded = np.zeros((n, n), dtype=np.float64)
            r, c = mat.shape[0], mat.shape[1]
            padded[:r, :c] = mat
            mat = padded
        return mat

    def fetch(self, chrom: str, start: int, end: int) -> tuple[np.ndarray, np.ndarray, int]:
        b = self.binsize
        s = (start // b) * b
        e = min(-(-end // b) * b, -(-self.chroms()[chrom] // b) * b)
        raw = self._matrix(chrom, s, e, "NONE")
        bal = self._matrix(chrom, s, e, self._norm)
        # straw returns 0 where cooler returns NaN; distinguishing "unmappable"
        # from "genuinely zero contacts" matters for the coarse-grain mask.
        bal[bal == 0] = np.nan
        return bal, raw, s


def open_matrix(path: str | Path, prefer_resolution: int = 5000, balance: bool = True) -> MatrixSource:
    if str(path).startswith(("http://", "https://")):
        if not str(path).endswith(".hic"):
            raise ValueError("only remote .hic is supported (hicstraw reads it by HTTP range)")
        return StrawSource(str(path), prefer_resolution, balance)
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Download it with `seq2hic.encode fetch`.")
    if p.suffix in (".cool", ".mcool"):
        return CoolerSource(p, prefer_resolution, balance)
    if p.suffix == ".hic":
        return StrawSource(p, prefer_resolution, balance)
    raise ValueError(f"unrecognised contact matrix format: {p.suffix}")


# --------------------------------------------------------------------------
# adaptive coarse-graining
# --------------------------------------------------------------------------


def adaptive_coarsegrain(
    balanced: np.ndarray, counts: np.ndarray, cutoff: float = 5.0, max_levels: int = 8
) -> np.ndarray:
    """Akita/cooltools adaptive coarse-graining.

    For each pixel, walk a 2x2-pooling pyramid outward and take the value from
    the finest level whose pooled *raw* count reaches `cutoff`. Dense
    near-diagonal pixels keep full resolution; sparse distal pixels get pooled
    until they mean something.

    Delegates to cooltools when installed -- it is the reference implementation
    and matching it exactly matters more than owning the code.
    """
    try:
        from cooltools.lib.numutils import adaptive_coarsegrain as _cg

        return np.asarray(_cg(balanced, counts, cutoff=cutoff, max_levels=max_levels))
    except ImportError:
        pass

    n = balanced.shape[0]
    mask = np.isfinite(balanced)
    vals = np.where(mask, np.nan_to_num(balanced, nan=0.0), 0.0)
    wts = mask.astype(np.float64)
    cnts = np.nan_to_num(counts, nan=0.0)

    # Level 0 is the original grid; each subsequent level pools 2x2.
    val_levels, wt_levels, cnt_levels = [vals], [wts], [cnts]
    cur_v, cur_w, cur_c = vals, wts, cnts
    for _ in range(max_levels):
        if cur_v.shape[0] < 2:
            break
        cur_v, cur_w, cur_c = _pool2(cur_v), _pool2(cur_w), _pool2(cur_c)
        val_levels.append(cur_v)
        wt_levels.append(cur_w)
        cnt_levels.append(cur_c)

    out = np.full((n, n), np.nan, dtype=np.float64)
    resolved = np.zeros((n, n), dtype=bool)
    for lv in range(len(val_levels)):
        v = _upsample(val_levels[lv], n, lv)
        w = _upsample(wt_levels[lv], n, lv)
        c = _upsample(cnt_levels[lv], n, lv)
        ok = (~resolved) & mask & (w > 0) & (c >= cutoff)
        out[ok] = v[ok] / w[ok]
        resolved |= ok
        if resolved[mask].all():
            break

    # Pixels that never reached the cutoff even fully pooled: take the coarsest
    # available estimate rather than dropping them, so the map stays dense.
    left = (~resolved) & mask
    if left.any():
        v = _upsample(val_levels[-1], n, len(val_levels) - 1)
        w = _upsample(wt_levels[-1], n, len(val_levels) - 1)
        good = left & (w > 0)
        out[good] = v[good] / w[good]
    return out


def _pool2(a: np.ndarray) -> np.ndarray:
    """2x2 block sum, odd dimensions edge-padded so nothing is discarded."""
    n = a.shape[0]
    if n % 2:
        a = np.pad(a, ((0, 1), (0, 1)), mode="constant")
        n += 1
    return a.reshape(n // 2, 2, n // 2, 2).sum(axis=(1, 3))


def _upsample(a: np.ndarray, n: int, level: int) -> np.ndarray:
    f = 2**level
    if f == 1:
        return a[:n, :n]
    big = np.repeat(np.repeat(a, f, axis=0), f, axis=1)
    return big[:n, :n]


# --------------------------------------------------------------------------
# resolution projection
# --------------------------------------------------------------------------


def overlap_weights(
    out_start: int, out_end: int, n_out: int, in_start: int, in_binsize: int, n_in: int
) -> np.ndarray:
    """(n_out, n_in) fraction of each output bin covered by each input bin.

    Rows sum to 1 wherever the output bin is fully covered, so `W @ v` is exact
    area-weighted averaging -- no interpolation, no smoothing beyond what the
    native resolution already imposes.
    """
    out_bin = (out_end - out_start) / n_out
    o_lo = out_start + np.arange(n_out, dtype=np.float64) * out_bin
    o_hi = o_lo + out_bin
    i_lo = in_start + np.arange(n_in, dtype=np.float64) * in_binsize
    i_hi = i_lo + in_binsize

    ov = np.minimum(o_hi[:, None], i_hi[None, :]) - np.maximum(o_lo[:, None], i_lo[None, :])
    np.clip(ov, 0.0, None, out=ov)
    return ov / out_bin


def area_resample(mat: np.ndarray, w: np.ndarray) -> np.ndarray:
    """W @ M @ W.T, propagating NaN as missing rather than poisoning the bin."""
    mask = np.isfinite(mat).astype(np.float64)
    filled = np.where(mask > 0, np.nan_to_num(mat, nan=0.0), 0.0)
    num = w @ filled @ w.T
    den = w @ mask @ w.T
    out = np.full(num.shape, np.nan, dtype=np.float64)
    good = den > 1e-9
    out[good] = num[good] / den[good]
    return out


def fetch_window(
    src: MatrixSource,
    chrom: str,
    start: int,
    end: int,
    n_bins: int,
    coarsegrain_cutoff: float = 5.0,
    coarsegrain_levels: int = 8,
) -> np.ndarray:
    """Coarse-grained balanced contacts for [start, end) on the model grid.

    Coarse-graining happens at native resolution, *before* projection -- pooling
    is a denoising operation over the real pixel lattice, and doing it after
    resampling would pool bins that have already been mixed.
    """
    bal, raw, in_start = src.fetch(chrom, start, end)
    if coarsegrain_cutoff > 0:
        bal = adaptive_coarsegrain(bal, raw, cutoff=coarsegrain_cutoff, max_levels=coarsegrain_levels)
    w = overlap_weights(start, end, n_bins, in_start, src.binsize, bal.shape[0])
    out = area_resample(bal, w)
    return 0.5 * (out + out.T)  # float error otherwise breaks exact symmetry


# --------------------------------------------------------------------------
# detrend / log / clip
# --------------------------------------------------------------------------


def accumulate_expected(src, windows, n_bins: int, cutoff: float = 5.0, levels: int = 8,
                        progress: bool = True) -> np.ndarray:
    """Mean coarse-grained contact per diagonal offset, streamed over windows.

    Computed on exactly the same transform the targets use -- estimating the
    decay curve on un-coarse-grained data and applying it to coarse-grained
    windows would bias every distal pixel.
    """
    from tqdm import tqdm

    sums = np.zeros(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.float64)
    it = tqdm(windows, desc="  expected", unit="win") if progress else windows

    for chrom, start, end in it:
        m = fetch_window(src, chrom, start, end, n_bins, cutoff, levels)
        finite = np.isfinite(m)
        for d in range(n_bins):
            diag = np.diagonal(m, offset=d)
            dmask = np.diagonal(finite, offset=d)
            if dmask.any():
                sums[d] += diag[dmask].sum()
                counts[d] += dmask.sum()

    exp = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    last = np.nan
    for d in range(n_bins):
        if np.isfinite(exp[d]):
            last = exp[d]
        elif np.isfinite(last):
            exp[d] = last
    if np.isfinite(exp).any():
        exp[~np.isfinite(exp)] = np.nanmin(exp[np.isfinite(exp)])
    else:
        exp[:] = 1.0
    return np.maximum(exp, 1e-12)


def to_target(
    raw: np.ndarray, expected: np.ndarray, clip: float = 2.0, diagonal_offset: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Detrend -> log -> clip -> zero and mask the near-diagonal band.

    Returns (target, valid_mask). The band `|i-j| < diagonal_offset` is zero in
    the target and False in the mask; `masked_mse` consumes the same mask, so
    the zeroed band contributes no gradient. Unmappable pixels are likewise
    zeroed and masked -- never imputed, because 0 in log-O/E space asserts
    "exactly as expected", which is a strong and usually wrong claim.
    """
    n = raw.shape[0]
    ii, jj = np.indices((n, n))
    dist = np.abs(ii - jj)
    exp_mat = expected[np.minimum(dist, expected.shape[0] - 1)]

    with np.errstate(divide="ignore", invalid="ignore"):
        tgt = np.log(raw / exp_mat)

    valid = np.isfinite(tgt) & (dist >= diagonal_offset)
    tgt = np.where(valid, tgt, 0.0)
    np.clip(tgt, -clip, clip, out=tgt)
    tgt = 0.5 * (tgt + tgt.T)
    valid = valid & valid.T
    return tgt.astype(np.float32), valid


def coverage_fraction(raw: np.ndarray) -> float:
    """Fraction of bins with any mappable support -- drops windows sitting on
    centromeres and assembly gaps."""
    n = raw.shape[0]
    return float((np.isfinite(raw).sum(axis=1) > 0).sum() / n)
