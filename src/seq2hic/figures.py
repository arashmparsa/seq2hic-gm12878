"""Figure generation from evaluation artefacts.

Produces, into <run>/figures/ (and a combined set for the ablation):

    fig1_contact_maps.png     observed | predicted | difference, best/median/worst
    fig2_insulation.png       observed vs predicted insulation profile
    fig3_training_curves.png  loss + val correlation per epoch, both arms
    fig4_ep_pr_curve.png      E-P precision-recall, model vs distance baseline
    fig5_ablation.png         per-metric delta with paired bootstrap CIs
    fig6_distance_decay.png   correlation as a function of genomic distance

Design choices worth stating: contact maps use a diverging colormap centred on
zero because the target is log(O/E) -- zero means "as expected", and a sequential
map would hide the sign of the deviation, which is the entire signal. Difference
panels share the colour scale with the data panels so the eye can compare
magnitudes across panels rather than being fooled by autoscaling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import metrics as M

CMAP = "RdBu_r"
VLIM = 2.0


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
    })


MIN_VALID_FRAC = 0.5   # windows mostly masked (gaps, low coverage) give meaningless r


def fig_contact_maps(pred, targ, mask, chrom, start, bin_bp, out: Path, note: str = "") -> None:
    corrs = np.asarray(M.map_correlation(pred, targ, mask), dtype=float)
    n = mask.shape[-1]
    band = np.abs(np.arange(n)[None, :] - np.arange(n)[:, None]) > 2
    valid = (mask.astype(bool) & band).sum(axis=(1, 2)) / band.sum()
    ok = np.flatnonzero(np.isfinite(corrs) & (valid >= MIN_VALID_FRAC))
    if ok.size == 0:
        return
    order = ok[np.argsort(corrs[ok])]
    picks = [("best", order[-1]), ("median", order[order.size // 2]), ("worst", order[0])]

    # Predicted panels get their own symmetric scale: an MSE-trained model shrinks
    # toward 0, and on the observed scale its structure is invisible. The scale is
    # printed in the panel so nobody mistakes it for the observed magnitude.
    sel = np.array([i for _, i in picks])
    pv = np.abs(pred[sel][mask[sel].astype(bool)])
    plim = float(np.percentile(pv, 99)) if pv.size else VLIM
    collapsed = plim < 0.05
    plim = max(plim, 0.05)

    fig, axes = plt.subplots(3, 3, figsize=(9.5, 9.6))
    for r, (label, i) in enumerate(picks):
        m = mask[i].astype(bool)
        o = np.where(m, targ[i], np.nan)
        p = np.where(m, pred[i], np.nan)
        d = np.where(m, pred[i] - targ[i], np.nan)
        span_mb = (o.shape[0] * bin_bp) / 1e6
        ext = [0, span_mb, span_mb, 0]

        for c, (arr, title, lim) in enumerate(
            ((o, "observed", VLIM), (p, f"predicted (own scale ±{plim:.2f})", plim),
             (d, "predicted − observed", VLIM))
        ):
            ax = axes[r, c]
            h = ax.imshow(arr, cmap=CMAP, vmin=-lim, vmax=lim, extent=ext,
                          interpolation="nearest")
            if c != 1:
                im = h
            if r == 0:
                ax.set_title(title)
            if c == 1 and collapsed:
                ax.text(0.5, 0.5, "prediction ≈ constant\n(model collapsed)",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=9, color="#b3261e")
            if c == 0:
                ax.set_ylabel(
                    f"{label}\n{chrom[i]}:{start[i]/1e6:.1f} Mb\nr = {corrs[i]:.3f}",
                    fontsize=8,
                )
            # Ticks are shared across the row and drawn only on the outer
            # panels; three autoscaled axes per row collide at any window size.
            ticks = [0, span_mb / 2, span_mb]
            fmt = "%.2f" if span_mb < 1 else "%.1f"
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.tick_params(labelsize=7)
            ax.set_xticklabels([fmt % t for t in ticks] if r == 2 else [])
            ax.set_yticklabels([fmt % t for t in ticks] if c == 0 else [])
            if r == 2:
                ax.set_xlabel("Mb")

    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("log(observed / expected), clipped ±2  (observed + difference panels)")
    fig.suptitle("Held-out contact maps: best / median / worst window" + note
                 + f"\n(ranked among {ok.size}/{len(corrs)} windows with ≥{MIN_VALID_FRAC:.0%} valid pixels)",
                 y=0.99)
    fig.savefig(out)
    plt.close(fig)


def fig_insulation(pred, targ, mask, chrom, start, bin_bp, out: Path, window_bins: int = 10) -> None:
    corrs = np.asarray(M.insulation_correlation(pred, targ, mask, window_bins), dtype=float)
    ok = np.flatnonzero(np.isfinite(corrs))
    if ok.size == 0:
        return
    i = ok[np.argsort(corrs[ok])][ok.size // 2]

    io = M.insulation_score(targ[i], window_bins, mask[i].astype(bool))
    ip = M.insulation_score(pred[i], window_bins, mask[i].astype(bool))
    x = np.arange(io.size) * bin_bp / 1e6

    # Rotated-triangle view: bin pair (a, b) sits at x = midpoint, y = separation, so the
    # map shares the insulation track's x axis. Observed above zero, predicted mirrored below.
    n = targ[i].shape[0]
    scale = bin_bp / 1e6
    depth = min(n, max(4 * window_bins, n // 4))          # show up to ~1/4 window of separation
    edges = np.arange(n + 1)
    A, B = np.meshgrid(edges, edges, indexing="ij")
    X = (A + B) / 2 * scale
    Y = (B - A) / 2 * scale
    keep = mask[i].astype(bool)
    upper = np.triu(np.ones((n, n), bool))
    far = (np.arange(n)[None, :] - np.arange(n)[:, None]) > depth

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 5.4), height_ratios=[1.2, 1], sharex=True)
    for arr, sign in ((targ[i], 1), (pred[i], -1)):
        shown = np.ma.masked_where(~(keep & upper) | far, arr)
        ax0.pcolormesh(X, sign * Y, shown, cmap=CMAP, vmin=-VLIM, vmax=VLIM, shading="flat",
                       rasterized=True)
    ymax = depth / 2 * scale
    ax0.set_ylim(-ymax, ymax)
    ax0.axhline(0, color="k", lw=0.6)
    ax0.set_yticks([-ymax, 0, ymax], [f"{2*ymax:.2f}", "0", f"{2*ymax:.2f}"])
    ax0.set_ylabel("separation (Mb)")
    ax0.text(0.005, 0.97, "observed", transform=ax0.transAxes, va="top", fontsize=8)
    ax0.text(0.005, 0.03, "predicted", transform=ax0.transAxes, va="bottom", fontsize=8)
    ax0.set_title(f"Held-out map and insulation profile — {chrom[i]}:{start[i]/1e6:.1f} Mb "
                  f"(median-r window)")
    ax0.grid(False)

    ax1.plot(x, io, lw=1.4, label="observed", color="#1f4e79")
    ax1.plot(x, ip, lw=1.4, label="predicted", color="#c1440e")
    for b in M.call_boundaries(io):
        ax1.axvline(b * bin_bp / 1e6, color="#1f4e79", alpha=0.2, lw=1)
    ax1.set_xlabel("Mb")
    ax1.set_ylabel("insulation score")
    ax1.legend(frameon=False, fontsize=8)
    ax1.set_title(f"insulation r = {corrs[i]:.3f}  (lower = stronger boundary)", fontsize=9)
    fig.savefig(out)
    plt.close(fig)


def fig_training_curves(runs: list[Path], out: Path) -> None:
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9, 3.4))
    colors = ["#1f4e79", "#c1440e", "#2e7d32"]
    any_data = False
    for k, run in enumerate(runs):
        hp = run / "history.json"
        if not hp.exists():
            continue
        hist = json.loads(hp.read_text())
        if not hist:
            continue
        any_data = True
        ep = [h["epoch"] + 1 for h in hist]
        c = colors[k % len(colors)]
        ax0.plot(ep, [h["train_loss"] for h in hist], color=c, lw=1.5, label=f"{run.name} train")
        ax0.plot(ep, [h["val_loss"] for h in hist], color=c, lw=1.5, ls="--", label=f"{run.name} val")
        ax1.plot(ep, [h["val_pearson"] for h in hist], color=c, lw=1.5, label=run.name)
    if not any_data:
        plt.close(fig)
        return
    ax0.set_yscale("log")   # one diverging run would otherwise flatten every other curve
    ax0.set_xlabel("epoch"); ax0.set_ylabel("masked MSE (log scale)"); ax0.set_title("loss")
    ax0.legend(frameon=False, fontsize=7)
    ax1.set_xlabel("epoch"); ax1.set_ylabel("val map Pearson"); ax1.set_title("validation correlation")
    ax1.legend(frameon=False, fontsize=8)
    fig.savefig(out)
    plt.close(fig)


def fig_ep_pr(pairs_files: dict[str, Path], out: Path) -> None:
    from sklearn.metrics import average_precision_score, precision_recall_curve

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    colors = ["#1f4e79", "#c1440e"]
    baseline_done = False
    plotted = False

    for k, (name, path) in enumerate(pairs_files.items()):
        if not path.exists():
            continue
        d = np.load(path, allow_pickle=False)
        if "labels" not in d or "kept" not in d:
            continue
        keep = d["kept"]
        y = d["labels"][keep]
        if y.sum() == 0 or y.all():
            continue
        s = d["predicted"][keep]
        pr, rc, _ = precision_recall_curve(y, s)
        ap = average_precision_score(y, s)
        ax.plot(rc, pr, lw=1.6, color=colors[k % 2], label=f"{name}  AP={ap:.3f}")
        plotted = True

        if not baseline_done:
            db = -d["distance_bp"][keep].astype(float)
            pr_b, rc_b, _ = precision_recall_curve(y, db)
            ax.plot(rc_b, pr_b, lw=1.2, ls="--", color="#777777",
                    label=f"distance only  AP={average_precision_score(y, db):.3f}")
            ax.axhline(y.mean(), color="#bbbbbb", lw=1, ls=":",
                       label=f"random  AP={y.mean():.3f}")
            baseline_done = True

    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_title("Enhancer–promoter loop retrieval\n(held-out chromosomes, distance-matched)")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.savefig(out)
    plt.close(fig)


def fig_ablation(ablation_json: Path, out: Path) -> None:
    if not ablation_json.exists():
        return
    data = json.loads(ablation_json.read_text())
    rows = [r for r in data["rows"] if np.isfinite(r.get("delta", np.nan))]
    if not rows:
        return

    labels = [r["metric"] for r in rows]
    deltas = [r["delta"] for r in rows]
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(6.4, 0.55 * len(rows) + 1.8))
    colors = ["#2e7d32" if d > 0 else "#b3261e" for d in deltas]
    ax.barh(y, deltas, color=colors, alpha=0.85, height=0.6)

    for i, r in enumerate(rows):
        if r.get("ci_lo") is not None and np.isfinite(r["ci_lo"]):
            ax.plot([r["ci_lo"], r["ci_hi"]], [i, i], color="black", lw=1.4)
            ax.plot([r["ci_lo"], r["ci_lo"]], [i - 0.12, i + 0.12], color="black", lw=1.4)
            ax.plot([r["ci_hi"], r["ci_hi"]], [i - 0.12, i + 0.12], color="black", lw=1.4)

    for i, r in enumerate(rows):
        if r.get("ci_lo") is None or not np.isfinite(r["ci_lo"]):
            labels[i] += "  †"
        if isinstance(r.get("baseline"), (int, float)) and isinstance(r.get("treatment"), (int, float)):
            labels[i] += f"\n{r['baseline']:.3f} → {r['treatment']:.3f}"

    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(f"Δ  ({data['treatment']} − {data['baseline']})")
    ax.set_title("Effect of adding the ATAC channel\n(per-window paired bootstrap, 95% CI)")
    if any(r.get("ci_lo") is None for r in rows):
        fig.text(0.01, -0.02, "† pooled over all test pairs/boundaries: single number, no CI",
                 fontsize=7.5, color="#555555")
    fig.savefig(out)
    plt.close(fig)


def fig_distance_decay(pred, targ, mask, bin_bp, out: Path, diagonal_offset: int = 2,
                       note: str = "") -> None:
    """Correlation as a function of genomic separation.

    This is the figure that catches the classic failure: strong overall
    correlation carried entirely by short-range contacts, with the long-range
    structure -- the part anyone actually cares about -- no better than chance.
    """
    n = pred.shape[-1]
    offsets, corrs, counts = [], [], []
    for d in range(diagonal_offset, n):
        pv, tv = [], []
        for i in range(pred.shape[0]):
            dm = np.diagonal(mask[i].astype(bool), offset=d)
            if not dm.any():
                continue
            pv.append(np.diagonal(pred[i], offset=d)[dm])
            tv.append(np.diagonal(targ[i], offset=d)[dm])
        if not pv:
            continue
        pv, tv = np.concatenate(pv), np.concatenate(tv)
        if pv.size < 30 or pv.std() < 1e-8 or tv.std() < 1e-8:
            continue
        offsets.append(d * bin_bp / 1e6)
        corrs.append(float(np.corrcoef(pv, tv)[0, 1]))
        counts.append(pv.size)

    if not offsets:
        return
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(offsets, corrs, lw=1.6, color="#1f4e79")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("genomic separation (Mb)")
    ax.set_ylabel("Pearson r")
    ax.set_title("Prediction accuracy vs genomic distance" + note)
    ax.set_ylim(min(-0.05, min(corrs) - 0.05), max(1.0, max(corrs) + 0.05))
    fig.savefig(out)
    plt.close(fig)


def generate_for_run(
    run: Path, split: str, bin_bp: int, note: str = "",
    out_dir: Path | None = None, prefix: str = "",
) -> list[Path]:
    """Write this run's figures. `out_dir`/`prefix` let the same figures also be
    collected into a combined directory, so a reader does not have to go
    spelunking through runs/*/figures/ to find the contact maps."""
    _style()
    figs_dir = out_dir if out_dir is not None else run / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)
    made = []

    npz = run / f"preds_{split}.npz"
    if npz.exists():
        d = np.load(npz, allow_pickle=False)
        pred = d["pred"].astype(np.float32)
        targ = d["target"].astype(np.float32)
        mask = d["mask"]
        chrom, start = d["chrom"], d["start"]

        n = f" — {run.name}{note}"
        p = figs_dir / f"{prefix}fig1_contact_maps.png"
        fig_contact_maps(pred, targ, mask, chrom, start, bin_bp, p, n); made.append(p)
        p = figs_dir / f"{prefix}fig2_insulation.png"
        fig_insulation(pred, targ, mask, chrom, start, bin_bp, p); made.append(p)
        p = figs_dir / f"{prefix}fig6_distance_decay.png"
        fig_distance_decay(pred, targ, mask, bin_bp, p, note=n); made.append(p)

    if out_dir is None:
        p = figs_dir / "fig3_training_curves.png"
        fig_training_curves([run], p); made.append(p)
    return [m for m in made if m.exists()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="seq2hic.figures", description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--bin-bp", type=int, default=8192)
    ap.add_argument("--out", default="figures", help="directory for combined figures")
    ap.add_argument("--config", default=None,
                    help="also emit the config-derived design figures into --out")
    ap.add_argument("--note", default="")
    args = ap.parse_args(argv)

    _style()
    runs = [Path(r) for r in args.runs]
    combined = Path(args.out)
    combined.mkdir(parents=True, exist_ok=True)

    made = []
    for run in runs:
        made += generate_for_run(run, args.split, args.bin_bp, args.note)
        # Same figures again into the combined directory, arm-prefixed, so all
        # of a run's output is readable from one place.
        made += generate_for_run(run, args.split, args.bin_bp, args.note,
                                 out_dir=combined, prefix=f"{run.name}_")

    # Config-derived design figures belong with the results they describe.
    if args.config:
        from .architecture_fig import fig_architecture, fig_receptive_field, fig_target_transform
        from .config import load_config

        cfg = load_config(args.config)
        for fn, name in ((fig_architecture, "fig0_architecture.png"),
                         (fig_receptive_field, "fig7_receptive_field.png"),
                         (fig_target_transform, "fig8_target_transform.png")):
            p = combined / name
            fn(cfg, p)
            made.append(p)

    p = combined / "fig3_training_curves.png"
    fig_training_curves(runs, p)
    if p.exists():
        made.append(p)

    p = combined / "fig4_ep_pr_curve.png"
    fig_ep_pr({r.name: r / f"pairs_{args.split}.npz" for r in runs}, p)
    if p.exists():
        made.append(p)

    p = combined / "fig5_ablation.png"
    fig_ablation(runs[0].parent / "ablation.json", p)
    if p.exists():
        made.append(p)

    for m in made:
        print(f"  {m}")
    print(f"\n{len(made)} figures written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
