"""Map-level metrics: correlation, insulation, boundary agreement.

Every metric here respects the validity mask and the near-diagonal exclusion.
That is not fussiness -- the near-diagonal band is the highest-variance part of
the map, and including it inflates correlation enough to hide a model that has
learned nothing about long-range structure.

`map_correlation` is deliberately computed *per window* and then averaged,
rather than pooling all pixels genome-wide. Pooled correlation is dominated by
between-window variance in overall contact intensity, which any model gets right
trivially; per-window correlation asks the harder and more interesting question
of whether the internal structure of each locus is right.
"""

from __future__ import annotations

import numpy as np


def _upper_valid(n: int, mask: np.ndarray, diagonal_offset: int = 2) -> np.ndarray:
    ii, jj = np.indices((n, n))
    return mask & (jj - ii >= max(1, diagonal_offset))


def map_correlation(
    pred: np.ndarray, target: np.ndarray, mask: np.ndarray, diagonal_offset: int = 2,
    method: str = "pearson",
) -> list[float]:
    """Per-window correlation over valid upper-triangle pixels.

    Accepts (n, n) or (B, n, n). Returns one value per window; NaN where a
    window has too few valid pixels or zero variance (a constant prediction has
    undefined correlation, and returning 0 would be a silent lie).
    """
    if pred.ndim == 2:
        pred, target, mask = pred[None], target[None], mask[None]
    out = []
    n = pred.shape[-1]
    for p, t, m in zip(pred, target, mask):
        sel = _upper_valid(n, m.astype(bool), diagonal_offset)
        pv, tv = p[sel], t[sel]
        if pv.size < 10 or pv.std() < 1e-8 or tv.std() < 1e-8:
            out.append(float("nan"))
            continue
        if method == "spearman":
            pv, tv = _rank(pv), _rank(tv)
        out.append(float(np.corrcoef(pv, tv)[0, 1]))
    return out


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(x.size, dtype=np.float64)
    return ranks


def insulation_score(mat: np.ndarray, window_bins: int = 10, mask: np.ndarray | None = None) -> np.ndarray:
    """Mean contact inside a diamond centred on each bin's boundary.

    Low score = strong insulation = candidate TAD boundary. Bins closer than
    `window_bins` to the window edge get NaN rather than a truncated diamond,
    since a partial diamond is systematically biased and would plant a fake
    boundary at every window edge.
    """
    n = mat.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    valid = np.isfinite(mat) if mask is None else (mask.astype(bool) & np.isfinite(mat))
    m = np.where(valid, mat, np.nan)

    for i in range(window_bins, n - window_bins):
        block = m[i - window_bins : i, i + 1 : i + window_bins + 1]
        finite = np.isfinite(block)
        if finite.sum() >= max(4, window_bins):
            out[i] = float(np.nanmean(block))
    return out


def insulation_correlation(
    pred: np.ndarray, target: np.ndarray, mask: np.ndarray, window_bins: int = 10
) -> list[float]:
    if pred.ndim == 2:
        pred, target, mask = pred[None], target[None], mask[None]
    out = []
    for p, t, m in zip(pred, target, mask):
        ip = insulation_score(p, window_bins, m)
        it = insulation_score(t, window_bins, m)
        ok = np.isfinite(ip) & np.isfinite(it)
        if ok.sum() < 10 or ip[ok].std() < 1e-8 or it[ok].std() < 1e-8:
            out.append(float("nan"))
        else:
            out.append(float(np.corrcoef(ip[ok], it[ok])[0, 1]))
    return out


def call_boundaries(insulation: np.ndarray, prominence: float = 0.7) -> np.ndarray:
    """Local minima of the insulation profile deeper than `prominence` below the
    mean of their neighbourhood. Returns bin indices.

    The profile is z-scored first, so `prominence` is in standard deviations of
    that window's own profile. An absolute threshold silently fails on model
    output: MSE-trained predictions are shrunk toward the mean (on GM12878 test
    windows the predicted insulation SD was ~0.07 vs ~0.22 observed), so a cut
    tuned to observed maps called zero boundaries on every prediction and
    boundary F1 read 0 for a model whose insulation tracks the truth at r~0.35.
    0.7 SD reproduces the old 0.15 absolute cut on observed maps."""
    insulation = np.asarray(insulation, dtype=np.float64)
    fin = np.isfinite(insulation)
    if fin.sum() < 3:
        return np.asarray([], dtype=int)
    sd = insulation[fin].std()
    if sd < 1e-12:
        return np.asarray([], dtype=int)
    insulation = (insulation - insulation[fin].mean()) / sd
    n = insulation.size
    out = []
    for i in range(1, n - 1):
        a, b, c = insulation[i - 1], insulation[i], insulation[i + 1]
        if not np.isfinite([a, b, c]).all():
            continue
        if b < a and b < c:
            lo, hi = max(0, i - 5), min(n, i + 6)
            nb = insulation[lo:hi]
            nb = nb[np.isfinite(nb)]
            if nb.size and (np.mean(nb) - b) >= prominence:
                out.append(i)
    return np.asarray(out, dtype=int)


def boundary_f1(
    pred: np.ndarray, target: np.ndarray, mask: np.ndarray, window_bins: int = 10,
    prominence: float = 0.7, tolerance: int = 1,
) -> dict[str, float]:
    """Boundary-calling agreement, matching within +/- `tolerance` bins.

    A tolerance of 1 bin (8 kb) is not generosity -- boundary calls from Hi-C
    are themselves only accurate to about a bin, so demanding exact agreement
    would measure binning noise rather than biology.
    """
    if pred.ndim == 2:
        pred, target, mask = pred[None], target[None], mask[None]
    tp = fp = fn = 0
    for p, t, m in zip(pred, target, mask):
        bp = call_boundaries(insulation_score(p, window_bins, m), prominence)
        bt = call_boundaries(insulation_score(t, window_bins, m), prominence)
        used = np.zeros(bt.size, dtype=bool)
        for x in bp:
            hit = np.where((np.abs(bt - x) <= tolerance) & ~used)[0]
            if hit.size:
                used[hit[0]] = True
                tp += 1
            else:
                fp += 1
        fn += int((~used).sum())

    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"boundary_precision": prec, "boundary_recall": rec, "boundary_f1": f1}


def bootstrap_ci(
    values: list[float] | np.ndarray, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """(mean, lo, hi) percentile bootstrap over windows."""
    v = np.asarray([x for x in np.asarray(values, dtype=float) if np.isfinite(x)])
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(v, v.size, replace=True).mean() for _ in range(n_boot)])
    return float(v.mean()), float(np.percentile(means, 100 * alpha / 2)), float(
        np.percentile(means, 100 * (1 - alpha / 2))
    )


def paired_bootstrap_delta(
    a: list[float], b: list[float], n_boot: int = 1000, alpha: float = 0.05, seed: int = 0
) -> dict[str, float]:
    """Bootstrap of (b - a) over paired windows.

    Paired, because both arms are evaluated on exactly the same windows. An
    unpaired interval would be far wider and would understate a real effect --
    the between-window variance is shared and should cancel.
    """
    av = np.asarray(a, dtype=float)
    bv = np.asarray(b, dtype=float)
    if av.shape != bv.shape:
        raise ValueError(f"paired bootstrap needs equal lengths, got {av.shape} and {bv.shape}")
    ok = np.isfinite(av) & np.isfinite(bv)
    d = bv[ok] - av[ok]
    if d.size == 0:
        return {"delta": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(d, d.size, replace=True).mean() for _ in range(n_boot)])
    return {
        "delta": float(d.mean()),
        "lo": float(np.percentile(boots, 100 * alpha / 2)),
        "hi": float(np.percentile(boots, 100 * (1 - alpha / 2))),
        "n": int(d.size),
    }
