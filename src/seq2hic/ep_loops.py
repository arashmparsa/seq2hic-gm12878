"""Enhancer-promoter loop scoring on held-out chromosomes.

The question this answers: on a chromosome the model has never seen, does its
predicted contact map rank genuinely-looping enhancer-promoter pairs above
non-looping ones?

Construction:

* **Enhancers** = ATAC IDR peaks (ENCSR637XSC) that are *distal* -- at least
  `enhancer_tss_exclusion_bp` from any annotated TSS. Without that filter most
  "enhancers" are promoter-proximal peaks, and the task collapses into
  promoter-promoter contact, which is a different and much easier question.

* **Promoters** = GENCODE TSS +/- `tss_flank_bp`.

* **Labels** come from the *observed* map: a pair is positive if its observed
  log(O/E) sits above `contact_quantile` of all candidate pairs. Alternatively
  supply a HiCCUPS loop call set with `positives: loops_bedpe`.

* **Negatives are distance-matched.** This is the part that makes the number
  mean anything. Predicted contact falls off steeply with genomic distance, so
  unmatched negatives let a model score well by knowing distance decay and
  nothing else -- it would post a high AUPRC having learned no biology at all.
  Negatives are therefore drawn from the same distance stratum as the positive
  they are matched to.

Note the model never sees a label: enhancer positions come from ATAC, promoter
positions from annotation, and the contact labels come from a chromosome held
out of training entirely.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PairSet:
    """Flat arrays of candidate E-P pairs, one entry per pair."""

    window: list[int] = field(default_factory=list)   # index into the window list
    bin_e: list[int] = field(default_factory=list)
    bin_p: list[int] = field(default_factory=list)
    distance_bp: list[int] = field(default_factory=list)
    observed: list[float] = field(default_factory=list)
    predicted: list[float] = field(default_factory=list)

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {
            "window": np.asarray(self.window, dtype=np.int64),
            "bin_e": np.asarray(self.bin_e, dtype=np.int64),
            "bin_p": np.asarray(self.bin_p, dtype=np.int64),
            "distance_bp": np.asarray(self.distance_bp, dtype=np.int64),
            "observed": np.asarray(self.observed, dtype=np.float64),
            "predicted": np.asarray(self.predicted, dtype=np.float64),
        }

    def __len__(self) -> int:
        return len(self.bin_e)


def peak_centers(peaks: np.ndarray) -> np.ndarray:
    return (peaks[:, 0] + peaks[:, 1]) // 2


def filter_distal(centers: np.ndarray, tss: np.ndarray, exclusion_bp: int) -> np.ndarray:
    """Keep peak centres at least `exclusion_bp` from every TSS."""
    if tss.size == 0 or centers.size == 0:
        return centers
    idx = np.searchsorted(tss, centers)
    left = np.where(idx > 0, tss[np.clip(idx - 1, 0, tss.size - 1)], -10**12)
    right = np.where(idx < tss.size, tss[np.clip(idx, 0, tss.size - 1)], 10**12)
    nearest = np.minimum(np.abs(centers - left), np.abs(centers - right))
    return centers[nearest >= exclusion_bp]


def build_pairs_for_window(
    pairs: PairSet,
    w_index: int,
    chrom: str,
    start: int,
    end: int,
    n_bins: int,
    observed: np.ndarray,
    predicted: np.ndarray,
    mask: np.ndarray,
    enhancer_centers: dict[str, np.ndarray],
    tss: dict[str, np.ndarray],
    cfg_ep: dict,
) -> None:
    """Append every valid E-P candidate inside this window to `pairs`."""
    bin_bp = (end - start) // n_bins
    min_d = int(cfg_ep["min_distance_bp"])
    max_d = int(cfg_ep["max_distance_bp"])

    e_all = enhancer_centers.get(chrom, np.empty(0, dtype=np.int64))
    p_all = tss.get(chrom, np.empty(0, dtype=np.int64))
    e_in = e_all[(e_all >= start) & (e_all < end)]
    p_in = p_all[(p_all >= start) & (p_all < end)]
    if e_in.size == 0 or p_in.size == 0:
        return

    e_bins = np.clip((e_in - start) // bin_bp, 0, n_bins - 1)
    p_bins = np.clip((p_in - start) // bin_bp, 0, n_bins - 1)

    # Collapse duplicates: several peaks in one 8 kb bin are one observation,
    # not several. Leaving them in would weight dense loci by peak density.
    e_bins = np.unique(e_bins)
    p_bins = np.unique(p_bins)

    for eb in e_bins:
        for pb in p_bins:
            if eb == pb:
                continue
            d = abs(int(eb) - int(pb)) * bin_bp
            if d < min_d or d > max_d:
                continue
            if not mask[eb, pb]:
                continue
            pairs.window.append(w_index)
            pairs.bin_e.append(int(eb))
            pairs.bin_p.append(int(pb))
            pairs.distance_bp.append(int(d))
            pairs.observed.append(float(observed[eb, pb]))
            pairs.predicted.append(float(predicted[eb, pb]))


def label_by_quantile(observed: np.ndarray, quantile: float) -> np.ndarray:
    """Positive = observed contact above the given quantile of all candidates."""
    if observed.size == 0:
        return np.zeros(0, dtype=bool)
    thresh = np.quantile(observed, quantile)
    return observed > thresh


def label_by_loops(
    pairs: dict[str, np.ndarray],
    windows,
    bin_bp: int,
    bedpe_path: str,
    tolerance_bins: int = 1,
) -> np.ndarray:
    """Positive = pair coincides with a called loop anchor pair (from HiCCUPS)."""
    loops: dict[str, list[tuple[int, int]]] = {}
    with open(bedpe_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip() or line.startswith(("#", "chr1\t", "track")):
                continue
            f = line.split("\t")
            if len(f) < 6 or f[0] != f[3]:
                continue
            c = f[0] if f[0].startswith("chr") else f"chr{f[0]}"
            loops.setdefault(c, []).append(
                ((int(f[1]) + int(f[2])) // 2, (int(f[4]) + int(f[5])) // 2)
            )

    labels = np.zeros(pairs["bin_e"].size, dtype=bool)
    for k in range(labels.size):
        w = windows[pairs["window"][k]]
        anchors = loops.get(w.chrom)
        if not anchors:
            continue
        pe = w.start + int(pairs["bin_e"][k]) * bin_bp
        pp = w.start + int(pairs["bin_p"][k]) * bin_bp
        tol = tolerance_bins * bin_bp
        for a, b in anchors:
            if (abs(a - pe) <= tol and abs(b - pp) <= tol) or (
                abs(a - pp) <= tol and abs(b - pe) <= tol
            ):
                labels[k] = True
                break
    return labels


def distance_matched_subset(
    labels: np.ndarray,
    distance_bp: np.ndarray,
    negatives_per_positive: int = 10,
    n_strata: int = 10,
    seed: int = 0,
) -> np.ndarray:
    """Indices of all positives plus distance-matched negatives.

    Strata are quantile-based on the distance distribution of positives, so the
    matching is tight where the positives actually live rather than spread over
    a uniform grid that mostly covers empty distance ranges.
    """
    rng = np.random.default_rng(seed)
    pos_idx = np.flatnonzero(labels)
    neg_idx = np.flatnonzero(~labels)
    if pos_idx.size == 0 or neg_idx.size == 0:
        return np.arange(labels.size)

    edges = np.unique(np.quantile(distance_bp[pos_idx], np.linspace(0, 1, n_strata + 1)))
    if edges.size < 2:
        edges = np.array([distance_bp.min(), distance_bp.max() + 1], dtype=float)
    pos_stratum = np.clip(np.searchsorted(edges, distance_bp[pos_idx], side="right") - 1, 0, edges.size - 2)
    neg_stratum = np.clip(np.searchsorted(edges, distance_bp[neg_idx], side="right") - 1, 0, edges.size - 2)

    chosen = [pos_idx]
    shortfall = 0
    for s in range(edges.size - 1):
        n_pos = int((pos_stratum == s).sum())
        if n_pos == 0:
            continue
        pool = neg_idx[neg_stratum == s]
        want = n_pos * negatives_per_positive
        if pool.size == 0:
            shortfall += want
            continue
        take = min(want, pool.size)
        shortfall += want - take
        chosen.append(rng.choice(pool, size=take, replace=False))

    if shortfall:
        # Reported rather than silently absorbed: a large shortfall means the
        # matching failed and the AUPRC is no longer distance-controlled.
        print(
            f"      note: {shortfall} distance-matched negatives unavailable "
            f"({shortfall / max(1, pos_idx.size * negatives_per_positive):.0%} of target)"
        )
    return np.sort(np.concatenate(chosen))


def score(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """AUPRC + AUROC, with the positive rate reported alongside.

    The positive rate is not decoration: AUPRC's baseline *is* the positive
    rate, so an AUPRC of 0.30 is excellent at 3% positives and useless at 40%.
    Reporting one without the other is close to meaningless.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    if labels.size == 0 or labels.sum() == 0 or labels.all():
        return {"auprc": float("nan"), "auroc": float("nan"),
                "positive_rate": float(labels.mean()) if labels.size else float("nan"),
                "n_pairs": int(labels.size), "n_positive": int(labels.sum())}
    return {
        "auprc": float(average_precision_score(labels, scores)),
        "auroc": float(roc_auc_score(labels, scores)),
        "positive_rate": float(labels.mean()),
        "n_pairs": int(labels.size),
        "n_positive": int(labels.sum()),
    }


def distance_only_baseline(distance_bp: np.ndarray) -> np.ndarray:
    """Score = -distance. The floor any real model must clear.

    If a model's E-P AUPRC does not beat this, it has learned the distance
    decay and nothing more, whatever its map correlation says.
    """
    return -distance_bp.astype(np.float64)
