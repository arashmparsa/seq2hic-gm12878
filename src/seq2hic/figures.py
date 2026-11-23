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


MIN_VALID_FRAC = 0.5    # windows mostly masked (gaps, low coverage) give meaningless r
COLLAPSE_RATIO = 0.05   # pred varies across windows < 5% as much as the data -> input ignored
SMOOTH_SIGMA = 1.0      # display-only Gaussian (bins) on observed maps; raw 8 kb pixels are shot noise


def _smooth(arr, valid, sigma=SMOOTH_SIGMA):
    """NaN-aware Gaussian smoothing (normalised convolution) for display only."""
    from scipy.ndimage import gaussian_filter
    num = gaussian_filter(np.where(valid, arr, 0.0), sigma)
    den = gaussian_filter(valid.astype(float), sigma)
    # Isolated missing pixels are filled from their neighbours; larger holes stay blank.
    return np.where(den > 0.3, num / np.maximum(den, 1e-9), np.nan)


def _display_maps(pred_i, targ_i, mask_i):
    """Smoothed observed + raw predicted, blanking only empty rows/cols and the masked band.
    The prediction exists at sparse pixels too, so it is not punched full of holes."""
    m = mask_i.astype(bool)
    n = m.shape[0]
    bad = m.mean(axis=1) < 0.1
    near = np.abs(np.arange(n)[None, :] - np.arange(n)[:, None]) <= 1
    drawable = ~(bad[:, None] | bad[None, :]) & ~near
    obs = np.where(drawable, _smooth(targ_i, m), np.nan)
    prd = np.where(drawable, pred_i, np.nan)
    return obs, prd


def collapse_ratio(pred, targ, mask) -> float:
    """Mean across-window std of the prediction / same for the target. Near 0 means the
    model emits the same map whatever the input."""
    import warnings
    m = mask.astype(bool)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sp = np.nanmean(np.nanstd(np.where(m, pred, np.nan), axis=0))
        st = np.nanmean(np.nanstd(np.where(m, targ, np.nan), axis=0))
    return float(sp / st) if st > 0 else float("nan")


def _collapse_banner(ax, ratio):
    ax.text(0.5, 0.5, "PREDICTION IGNORES INPUT\n(same map for every window;\n"
            f"between-window spread {ratio:.1%} of data's)",
            transform=ax.transAxes, ha="center", va="center", fontsize=8.5,
            color="#b3261e", bbox=dict(fc="white", ec="#b3261e", alpha=0.9))


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
    ratio = collapse_ratio(pred, targ, mask)
    collapsed = np.isfinite(ratio) and ratio < COLLAPSE_RATIO

    shown = [_display_maps(pred[i], targ[i], mask[i]) for _, i in picks]
    olim = float(np.nanpercentile(np.abs(np.concatenate([o.ravel() for o, _ in shown])), 99))
    plim = max(float(np.nanpercentile(np.abs(np.concatenate([p.ravel() for _, p in shown])), 99)), 1e-3)
    span_mb = n * bin_bp / 1e6
    up = np.triu(np.ones((n, n), bool), 1)
    ext = [0, span_mb, span_mb, 0]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.6))
    fig.subplots_adjust(left=0.05, right=0.98, top=0.86, bottom=0.24, wspace=0.12)
    for ax, (label, i), (obs, prd) in zip(axes, picks, shown):
        h_o = ax.imshow(np.where(up, obs, np.nan), cmap=CMAP, vmin=-olim, vmax=olim,
                        extent=ext, interpolation="nearest")
        h_p = ax.imshow(np.where(up.T, prd, np.nan), cmap=CMAP, vmin=-plim, vmax=plim,
                        extent=ext, interpolation="nearest")
        ax.plot([0, span_mb], [0, span_mb], color="k", lw=0.6)
        ax.text(0.97, 0.97, "observed", transform=ax.transAxes, ha="right", va="top", fontsize=8)
        ax.text(0.03, 0.03, "predicted", transform=ax.transAxes, ha="left", va="bottom", fontsize=8)
        ax.set_title(f"{label}: {chrom[i]}:{start[i]/1e6:.1f} Mb   r = {corrs[i]:.2f}", fontsize=9)
        ticks = [0, span_mb / 2, span_mb]
        ax.set_xticks(ticks, [f"{t:.1f}" for t in ticks])
        ax.set_yticks(ticks, [f"{t:.1f}" for t in ticks])
        ax.tick_params(labelsize=7)
        ax.set_xlabel("Mb")
        ax.grid(False)
        if collapsed:
            _collapse_banner(ax, ratio)
    axes[0].set_ylabel("Mb")
    # Two horizontal colour bars under the panels, side by side, so they never overlap.
    for h, x0, lab in ((h_o, 0.12, f"observed log(O/E), smoothed σ={SMOOTH_SIGMA:g} bin"),
                       (h_p, 0.56, "predicted log(O/E), own scale")):
        cax = fig.add_axes([x0, 0.09, 0.32, 0.025])
        cb = fig.colorbar(h, cax=cax, orientation="horizontal")
        cb.set_label(lab, fontsize=8)
        cb.ax.tick_params(labelsize=7)
    fig.suptitle("Held-out windows: observed (upper-right) vs predicted (lower-left)" + note
                 + f"\nbest / median / worst by map Pearson, among {ok.size}/{len(corrs)} windows "
                 f"with ≥{MIN_VALID_FRAC:.0%} valid pixels", fontsize=10, y=0.985)
    fig.savefig(out, bbox_inches=None)
    plt.close(fig)


def fig_insulation(pred, targ, mask, chrom, start, bin_bp, out: Path, window_bins: int = 10) -> None:
    corrs = np.asarray(M.insulation_correlation(pred, targ, mask, window_bins), dtype=float)
    ok = np.flatnonzero(np.isfinite(corrs))
    if ok.size == 0:
        return
    i = ok[np.argsort(corrs[ok])][ok.size // 2]
    ratio = collapse_ratio(pred, targ, mask)

    io = M.insulation_score(targ[i], window_bins, mask[i].astype(bool))
    ip = M.insulation_score(pred[i], window_bins, mask[i].astype(bool))
    x = np.arange(io.size) * bin_bp / 1e6

    obs, prd = _display_maps(pred[i], targ[i], mask[i])
    olim = float(np.nanpercentile(np.abs(obs), 99))
    plim = max(float(np.nanpercentile(np.abs(prd), 99)), 1e-3)

    # Rotated-triangle view: bin pair (a, b) sits at x = midpoint, y = separation, so the
    # map shares the insulation track's x axis. Observed above zero, predicted mirrored below.
    n = targ[i].shape[0]
    scale = bin_bp / 1e6
    depth = min(n, max(4 * window_bins, n // 4))
    edges = np.arange(n + 1)
    A, B = np.meshgrid(edges, edges, indexing="ij")
    X = (A + B) / 2 * scale
    Y = (B - A) / 2 * scale
    upper = np.triu(np.ones((n, n), bool))
    far = (np.arange(n)[None, :] - np.arange(n)[:, None]) > depth

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 5.6), height_ratios=[1.2, 1], sharex=True)
    for arr, sign, lim in ((obs, 1, olim), (prd, -1, plim)):
        shown = np.ma.masked_invalid(np.where(upper & ~far, arr, np.nan))
        ax0.pcolormesh(X, sign * Y, shown, cmap=CMAP, vmin=-lim, vmax=lim, shading="flat",
                       rasterized=True)
    ymax = depth / 2 * scale
    ax0.set_ylim(-ymax, ymax)
    ax0.axhline(0, color="k", lw=0.6)
    ax0.set_yticks([-ymax, 0, ymax], [f"{2*ymax:.2f}", "0", f"{2*ymax:.2f}"])
    ax0.set_ylabel("separation (Mb)")
    ax0.text(0.005, 0.97, f"observed (smoothed, ±{olim:.2f})", transform=ax0.transAxes,
             va="top", fontsize=8)
    ax0.text(0.005, 0.03, f"predicted (own scale, ±{plim:.2f})", transform=ax0.transAxes,
             va="bottom", fontsize=8)
    ax0.set_title(f"Held-out map and insulation — {chrom[i]}:{start[i]/1e6:.1f} Mb (median window)")
    ax0.grid(False)
    if np.isfinite(ratio) and ratio < COLLAPSE_RATIO:
        _collapse_banner(ax0, ratio)

    ax1.plot(x, io, lw=1.4, label="observed", color="#1f4e79")
    ax1.plot(x, ip, lw=1.4, label="predicted", color="#c1440e")
    for b in M.call_boundaries(io):
        ax1.axvline(b * bin_bp / 1e6, color="#1f4e79", alpha=0.2, lw=1)
    ax1.set_xlabel("Mb")
    ax1.set_ylabel("insulation score")
    ax1.legend(frameon=False, fontsize=8)
    ax1.set_title(f"insulation r = {corrs[i]:.2f}  (dips = boundaries; shaded lines = observed calls)",
                  fontsize=9)
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
