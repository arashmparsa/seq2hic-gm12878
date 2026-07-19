"""Architecture schematic + receptive-field figures.

These describe the *design*, not any result, so they are generated directly from
a config without training anything -- and they stay correct when the config
changes, because every number drawn is read from the config rather than typed
into the figure.

    fig0_architecture.png    the full seq(+ATAC) -> contact map data flow
    fig7_receptive_field.png dilation schedule and 2D receptive-field growth
    fig8_target_transform.png the Akita target transform, stage by stage
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .config import load_config

SEQ_C = "#2c6e9b"
ATAC_C = "#c1440e"
TRUNK_C = "#4a7ba7"
D1_C = "#6a9fc0"
PAIR_C = "#8e6c9b"
D2_C = "#7a5a8c"
OUT_C = "#b3261e"


def _box(ax, x, y, w, h, label, color, sub="", fontsize=8, alpha=0.85):
    ax.add_patch(
        FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                       facecolor=color, edgecolor="none", alpha=alpha)
    )
    ax.text(x + w / 2, y + h / 2 + (0.012 if sub else 0), label, ha="center", va="center",
            fontsize=fontsize, color="white", fontweight="bold")
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.028, sub, ha="center", va="center",
                fontsize=fontsize - 1.5, color="white", alpha=0.95)


def _arrow(ax, x0, y0, x1, y1, color="#444444", lw=1.3):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11,
                                 color=color, lw=lw, shrinkA=0, shrinkB=0))


def fig_architecture(cfg, out: Path) -> None:
    m = cfg["model"]
    w = cfg["window"]
    bp, n_bins = int(w["bp"]), int(w["n_bins"])
    bin_bp = bp // n_bins
    pools = [int(p) for p in m["trunk_pools"]]
    chans = list(m["trunk_channels"])
    c1, c2 = int(m["bottleneck_1d"]), int(m["channels_2d"])
    n_d1, n_d2 = int(m["dilated1d_blocks"]), int(m["dilated2d_blocks"])
    base = int(m.get("dilation_base", 2))
    use_dist = bool(m.get("use_distance_channel", True))
    c_pair = 2 * c1 + (1 if use_dist else 0)

    fig, ax = plt.subplots(figsize=(12.5, 6.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.text(0.5, 0.965, "seq2hic: dilated CNN, sequence (+ATAC) → GM12878 contact map",
            ha="center", fontsize=13, fontweight="bold")
    ax.text(0.5, 0.925,
            f"{bp:,} bp window   →   {n_bins}×{n_bins} map at {bin_bp:,} bp resolution",
            ha="center", fontsize=9, color="#555555")

    # ---- inputs
    y_seq, y_atac, bh = 0.72, 0.60, 0.075
    _box(ax, 0.015, y_seq, 0.145, bh, "DNA one-hot", SEQ_C, f"4 × {bp:,}")
    _box(ax, 0.015, y_atac, 0.145, bh, "ATAC signal", ATAC_C, f"1 × {bp:,}  (arm B only)")
    ax.text(0.0875, y_atac - 0.045, "ENCSR637XSC", ha="center", fontsize=7, color=ATAC_C)
    ax.text(0.0875, y_seq + bh + 0.018, "GRCh38", ha="center", fontsize=7, color=SEQ_C)

    _arrow(ax, 0.162, y_seq + bh / 2, 0.196, 0.70)
    _arrow(ax, 0.162, y_atac + bh / 2, 0.196, 0.685)
    ax.text(0.179, 0.638, "concat", ha="center", fontsize=7, color="#444444", rotation=90)

    # ---- 1D trunk: stacked shrinking bars
    # Layout runs left-to-right in a single normalised row; the constants below
    # are tuned so the final contact map ends just inside x = 1.0.
    x0, x1 = 0.19, 0.375
    ax.text((x0 + x1) / 2, 0.885, f"1D convolution trunk — {len(pools)} stages",
            ha="center", fontsize=9, fontweight="bold", color=TRUNK_C)
    ax.text((x0 + x1) / 2, 0.855,
            f"pools {'×'.join(str(p) for p in pools[:6])}"
            + ("…" if len(pools) > 6 else "")
            + f"  =  {math.prod(pools):,}",
            ha="center", fontsize=7.5, color="#555555")

    n = len(pools)
    bw = (x1 - x0) / n * 0.82
    length = bp
    for i, (p, c) in enumerate(zip(pools, chans)):
        length //= p
        xx = x0 + i * (x1 - x0) / n
        hh = 0.30 * (1 - 0.78 * i / max(1, n - 1)) + 0.045
        ax.add_patch(FancyBboxPatch((xx, 0.665 - hh / 2 + 0.05), bw, hh,
                                    boxstyle="round,pad=0.004,rounding_size=0.01",
                                    facecolor=TRUNK_C, edgecolor="none",
                                    alpha=0.45 + 0.5 * i / max(1, n - 1)))
        if i in (0, n - 1):
            ax.text(xx + bw / 2, 0.665 - hh / 2 + 0.032, f"{c}ch", ha="center",
                    fontsize=6.5, color="#333333")
    ax.text((x0 + x1) / 2, 0.492, f"→ {n_bins} bins × {chans[-1]} ch", ha="center",
            fontsize=8, color="#333333")

    _arrow(ax, x1 + 0.005, 0.665, x1 + 0.033, 0.665)

    # ---- dilated 1D
    xd = 0.412
    _box(ax, xd, 0.60, 0.095, 0.13,
         f"dilated 1D\n× {n_d1}", D1_C,
         f"d = {', '.join(str(base**i) for i in range(n_d1))}", fontsize=8)
    _arrow(ax, xd + 0.098, 0.665, xd + 0.126, 0.665)

    # ---- outer concat
    xo = 0.542
    _box(ax, xo, 0.615, 0.095, 0.10, "outer\nconcat", PAIR_C, f"→ {c_pair} ch", fontsize=8)
    ax.text(xo + 0.0475, 0.585,
            f"bin i ⊕ bin j{' ⊕ log dist' if use_dist else ''}",
            ha="center", fontsize=7, color=PAIR_C)

    # little 1D->2D glyph
    gx, gy = xo + 0.008, 0.775
    for k in range(8):
        ax.add_patch(plt.Rectangle((gx + k * 0.0095, gy), 0.008, 0.016,
                                   facecolor=D1_C, edgecolor="none"))
    ax.text(gx + 0.038, gy + 0.008, "→", fontsize=9, va="center", color="#444444")
    grid = np.add.outer(np.arange(6), np.arange(6)) % 2
    ax.imshow(grid, cmap="Purples", alpha=0.75,
              extent=[gx + 0.055, gx + 0.105, gy - 0.012, gy + 0.038], aspect="auto")

    _arrow(ax, xo + 0.098, 0.665, xo + 0.126, 0.665)

    # ---- dilated 2D stack (drawn as receding cards)
    xs = 0.672
    ax.text(xs + 0.075, 0.885, f"2D dilated residual stack — {n_d2} blocks",
            ha="center", fontsize=9, fontweight="bold", color=D2_C)
    ax.text(xs + 0.075, 0.855,
            f"d = {', '.join(str(base**i) for i in range(n_d2))}   ({c2} ch, symmetrised)",
            ha="center", fontsize=7.5, color="#555555")
    for k in range(min(n_d2, 5)):
        off = k * 0.013
        ax.add_patch(FancyBboxPatch((xs + off, 0.60 - off), 0.10, 0.13,
                                    boxstyle="round,pad=0.005,rounding_size=0.015",
                                    facecolor=D2_C, edgecolor="white", lw=1.1,
                                    alpha=0.55 + 0.09 * k))
    ax.text(xs + 0.076, 0.655, f"×{n_d2}", ha="center", va="center", fontsize=10,
            color="white", fontweight="bold")

    _arrow(ax, xs + 0.160, 0.665, xs + 0.188, 0.665)

    # ---- output map
    xm = 0.865
    rng = np.random.default_rng(0)
    ii, jj = np.indices((48, 48))
    # Built directly in log(O/E) space: background ~0 (white), diagonal and
    # loops positive (red). Taking a log of a positive-valued map here would
    # invert the palette and read as anti-correlation.
    demo = 1.8 * np.exp(-np.abs(ii - jj) / 6.0)
    for (a, b) in [(10, 30), (18, 40), (5, 22), (26, 43)]:
        demo += 1.5 * np.exp(-(((ii - a) ** 2 + (jj - b) ** 2) / 5.0))
        demo += 1.5 * np.exp(-(((ii - b) ** 2 + (jj - a) ** 2) / 5.0))
    demo += rng.normal(0, 0.12, demo.shape)
    demo = np.clip(0.5 * (demo + demo.T), -2, 2)
    mw = 0.115
    ax.imshow(demo, cmap="RdBu_r", vmin=-2, vmax=2,
              extent=[xm, xm + mw, 0.585, 0.745], aspect="auto")
    ax.add_patch(plt.Rectangle((xm, 0.585), mw, 0.16, fill=False,
                               edgecolor=OUT_C, lw=1.6))
    ax.text(xm + mw / 2, 0.762, f"{n_bins} × {n_bins}", ha="center", fontsize=8.5,
            fontweight="bold", color=OUT_C)
    ax.text(xm + mw / 2, 0.558, "log(obs/exp), ±2", ha="center", fontsize=7, color="#555555")
    ax.text(xm + mw / 2, 0.532, "ENCSR830NVY", ha="center", fontsize=7, color=OUT_C)

    # ---- ablation banner
    ax.add_patch(FancyBboxPatch((0.015, 0.30), 0.965, 0.155,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                facecolor="#f2f2f4", edgecolor="#dddddd"))
    ax.text(0.035, 0.408, "The experiment", fontsize=9.5, fontweight="bold", va="center")
    ax.text(0.035, 0.372,
            "Arm A (seq only, 4 ch)  and  Arm B (seq + ATAC, 5 ch) are trained on identical windows, "
            "splits, targets, seeds and schedules.",
            fontsize=8, va="center", color="#333333")
    ax.text(0.035, 0.340,
            "The only difference is the first conv layer's input width — so the measured Δ is "
            "attributable to accessibility and nothing else.",
            fontsize=8, va="center", color="#333333")

    # ---- data / split strip
    tr = ", ".join(cfg["split"]["train"][:6]) + ", …"
    ax.text(0.035, 0.245, "Split (chromosome holdout)", fontsize=9, fontweight="bold")
    ax.text(0.035, 0.205, f"train  {tr}", fontsize=8, color="#333333", family="monospace")
    ax.text(0.035, 0.170, f"val    {', '.join(cfg['split']['val'])}", fontsize=8,
            color="#333333", family="monospace")
    ax.text(0.035, 0.135, f"test   {', '.join(cfg['split']['test'])}   "
                          f"← E–P loop scoring happens only here",
            fontsize=8, color=OUT_C, family="monospace")
    ax.text(0.035, 0.075,
            "No window from a test chromosome is seen in training. Enhancers come from ATAC peaks, "
            "promoters from GENCODE TSS;\ncontact labels come from the held-out observed map, and "
            "negatives are distance-matched so distance decay alone cannot score.",
            fontsize=7.5, color="#555555", va="center")

    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_receptive_field(cfg, out: Path) -> None:
    m = cfg["model"]
    w = cfg["window"]
    bin_bp = int(w["bp"]) // int(w["n_bins"])
    n_bins = int(w["n_bins"])
    base = int(m.get("dilation_base", 2))
    n_d2 = int(m["dilated2d_blocks"])
    n_d1 = int(m["dilated1d_blocks"])

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.5, 3.9))

    # 2D receptive field growth
    rf, xs, ys = 1, [], []
    for i in range(n_d2):
        rf += 2 * (base**i)
        xs.append(i + 1)
        ys.append(rf * bin_bp / 1e6)
    ax0.plot(xs, ys, "o-", color=D2_C, lw=1.8, ms=5)
    ax0.axhline(int(w["bp"]) / 1e6, color=OUT_C, ls="--", lw=1.3,
                label=f"window = {int(w['bp'])/1e6:.2f} Mb")
    for i, (x, y) in enumerate(zip(xs, ys)):
        ax0.annotate(f"d={base**i}", (x, y), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=7, color="#555555")
    ax0.set_xlabel("2D dilated residual block")
    ax0.set_ylabel("receptive field (Mb)")
    ax0.set_title("2D receptive field vs depth")
    ax0.legend(frameon=False, fontsize=8)
    ax0.set_ylim(0, max(max(ys), int(w["bp"]) / 1e6) * 1.25)
    ax0.grid(alpha=0.25)

    covered = ys[-1] >= int(w["bp"]) / 1e6
    ax0.text(0.5, 0.06,
             "covers the full window ✓" if covered else "does NOT cover the window ✗",
             transform=ax0.transAxes, ha="center", fontsize=8,
             color="#2e7d32" if covered else OUT_C, fontweight="bold")

    # dilation schedule as a stem plot
    d1 = [base**i for i in range(n_d1)]
    d2 = [base**i for i in range(n_d2)]
    ax1.stem(np.arange(len(d1)), d1, linefmt="-", markerfmt="o", basefmt=" ",
             label=f"1D tower ({n_d1})")
    ax1.stem(np.arange(len(d2)) + 0.35, d2, linefmt="-", markerfmt="s", basefmt=" ",
             label=f"2D stack ({n_d2})")
    ax1.set_yscale("log", base=2)
    ax1.set_xlabel("block index")
    ax1.set_ylabel("dilation")
    ax1.set_title(f"Dilation schedule (bin = {bin_bp:,} bp, {n_bins} bins)")
    ax1.legend(frameon=False, fontsize=8)
    ax1.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out, dpi=165, facecolor="white")
    plt.close(fig)


def fig_target_transform(cfg, out: Path) -> None:
    """Illustrate the Akita target transform stage by stage on a toy matrix."""
    from . import targets as T

    n = 96
    rng = np.random.default_rng(3)
    ii, jj = np.indices((n, n))
    d = np.abs(ii - jj)

    expected = np.exp(-np.arange(n) / 12.0) + 0.01
    truth = expected[d].copy()
    for (a, b, s) in [(15, 55, 2.2), (30, 78, 1.9), (60, 88, 1.6)]:
        bump = np.exp(-(((ii - a) ** 2 + (jj - b) ** 2) / 14.0))
        truth += s * expected[d] * (bump + bump.T)
    truth += np.abs(np.add.outer(np.arange(n), np.arange(n)) % 17 - 8) * 1e-4

    depth = 40000 * truth / truth.max()
    counts = rng.poisson(np.maximum(depth * (1.0 / (1 + d / 6.0)), 0.02))
    balanced = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    balanced = 0.5 * (balanced + balanced.T)
    balanced[counts == 0] = np.nan

    cg = T.adaptive_coarsegrain(balanced, counts.astype(float), cutoff=5.0, max_levels=6)
    with np.errstate(invalid="ignore"):
        exp_from_cg = np.array([np.nanmean(np.diagonal(cg, offset=k)) for k in range(n)])
    exp_from_cg = np.maximum(np.nan_to_num(exp_from_cg, nan=1e-12), 1e-12)
    tgt, valid = T.to_target(cg, exp_from_cg, clip=2.0,
                             diagonal_offset=int(cfg["target"]["diagonal_offset"]))

    panels = [
        (np.log10(np.where(counts > 0, counts, np.nan)), "1. raw counts", "viridis", None),
        (np.log10(balanced), "2. balanced", "viridis", None),
        (np.log10(cg), "3. adaptive coarse-grain", "viridis", None),
        (np.where(valid, tgt, np.nan), "4. detrend → log → clip ±2", "RdBu_r", 2.0),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.6))
    for ax, (arr, title, cmap, vl) in zip(axes, panels):
        kw = dict(vmin=-vl, vmax=vl) if vl else {}
        im = ax.imshow(arr, cmap=cmap, interpolation="nearest", **kw)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    axes[0].set_ylabel("toy 96×96 locus", fontsize=8)
    fig.suptitle("Target transform: what each stage removes", fontsize=11, y=1.03)
    fig.tight_layout()
    fig.savefig(out, dpi=165, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="seq2hic.architecture_fig", description=__doc__)
    ap.add_argument("--config", default="configs/seq_atac.yaml")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    made = []
    for fn, name in (
        (fig_architecture, "fig0_architecture.png"),
        (fig_receptive_field, "fig7_receptive_field.png"),
        (fig_target_transform, "fig8_target_transform.png"),
    ):
        p = out / name
        fn(cfg, p)
        made.append(p)
        print(f"  {p}")
    print(f"\n{len(made)} figures written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
