"""One-shot dataset materialisation.

Run once per config family; both arms share the output, which is the whole point
-- if the two arms saw different windows or different targets, the ablation delta
would be measuring the pipeline instead of the ATAC channel.

Produces under paths.proc:
    index.json     window list (chrom, start, end, row index) + provenance
    targets.npy    (n_windows, n_bins, n_bins) float16 log(O/E)
    masks.npy      (n_windows, ceil(n_bins^2/8)) packed-bit validity
    expected.npy   (n_bins,) distance-decay curve actually used
    atac/*.npy     per-chromosome float16 signal memmaps
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from . import targets as T
from .config import load_config, resolve
from .dataset import build_windows
from .genome import Genome, n_fraction


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="seq2hic.prepare", description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    proc = Path(cfg["paths"]["proc"])
    if not proc.is_absolute():
        proc = Path(cfg["paths"]["root"]).resolve() / proc
    proc.mkdir(parents=True, exist_ok=True)

    if cfg.get("synthetic"):
        return _prepare_synthetic(cfg, proc, args.overwrite)

    window_bp = int(cfg["window"]["bp"])
    n_bins = int(cfg["window"]["n_bins"])
    chroms = list(cfg["genome"]["chroms"])

    print("[1/5] genome")
    genome = Genome(resolve(cfg, "genome.fasta"))
    chrom_sizes = {c: s for c, s in genome.chrom_sizes.items() if c in chroms}
    missing = [c for c in chroms if c not in chrom_sizes]
    if missing:
        raise ValueError(f"chromosomes absent from FASTA: {missing}")

    print("[2/5] contact matrix")
    src = T.open_matrix(
        resolve(cfg, "target.source"),
        prefer_resolution=int(cfg["target"]["prefer_resolution"]),
        balance=bool(cfg["target"]["balance"]),
    )
    bin_bp = window_bp // n_bins
    print(f"      native resolution {src.binsize} bp -> model grid {bin_bp} bp")
    if src.binsize > bin_bp:
        print(
            f"      NOTE: native resolution is coarser than the model grid. Targets are\n"
            f"      area-upsampled; the model cannot learn structure finer than "
            f"{src.binsize} bp."
        )
    hic_chroms = src.chroms()
    absent = [c for c in chroms if c not in hic_chroms]
    if absent:
        raise ValueError(f"chromosomes absent from contact matrix: {absent}")

    print("[3/5] windows")
    train_stride = int(cfg["window"]["train_stride"])
    eval_stride = int(cfg["window"]["eval_stride"])
    split = cfg["split"]

    raw_windows: list[tuple[str, int, int]] = []
    for name, stride in (("train", train_stride), ("val", eval_stride), ("test", eval_stride)):
        raw_windows += build_windows(chrom_sizes, list(split[name]), window_bp, stride)
    raw_windows.sort()
    print(f"      {len(raw_windows)} candidate windows")

    max_n = float(cfg["window"]["max_n_frac"])
    min_cov = float(cfg["window"]["min_cov_frac"])

    print("[4/5] targets")
    kept: list[dict] = []
    tgt_store: list[np.ndarray] = []
    mask_store: list[np.ndarray] = []

    # Expected curve first, on training chromosomes only (see targets.py docstring).
    exp_mode = cfg["target"]["expected"]
    cg = cfg["target"]["coarsegrain"]
    cg_cutoff, cg_levels = float(cg["cutoff"]), int(cg["max_levels"])
    train_set = set(split["train"])
    exp_windows = [w for w in raw_windows if w[0] in train_set]
    if exp_mode == "train_only":
        expected = T.accumulate_expected(src, exp_windows[::4], n_bins, cg_cutoff, cg_levels)
    elif exp_mode == "per_chrom":
        expected = T.accumulate_expected(src, raw_windows[::4], n_bins, cg_cutoff, cg_levels)
    else:
        raise ValueError(f"unknown target.expected: {exp_mode}")
    np.save(proc / "expected.npy", expected)

    clip = float(cfg["target"]["clip"])
    diag = int(cfg["target"]["diagonal_offset"])
    n_dropped_n, n_dropped_cov = 0, 0

    for chrom, start, end in tqdm(raw_windows, desc="  windows", unit="win"):
        if n_fraction(genome.sequence(chrom, start, end)) > max_n:
            n_dropped_n += 1
            continue
        raw = T.fetch_window(src, chrom, start, end, n_bins, cg_cutoff, cg_levels)
        if T.coverage_fraction(raw) < min_cov:
            n_dropped_cov += 1
            continue
        tgt, valid = T.to_target(raw, expected, clip=clip, diagonal_offset=diag)
        kept.append({"chrom": chrom, "start": start, "end": end, "index": len(kept)})
        tgt_store.append(tgt.astype(np.float16))
        mask_store.append(np.packbits(valid.ravel()))

    if not kept:
        raise RuntimeError("every window was filtered out -- check min_cov_frac/max_n_frac")

    print(
        f"      kept {len(kept)}  (dropped {n_dropped_n} N-rich, "
        f"{n_dropped_cov} low-coverage)"
    )
    np.save(proc / "targets.npy", np.stack(tgt_store))
    np.save(proc / "masks.npy", np.stack(mask_store))

    print("[5/5] ATAC track")
    if any(Path(p).exists() for p in [resolve(cfg, "atac.signal")]):
        from .tracks import bigwig_to_memmaps

        bigwig_to_memmaps(
            resolve(cfg, "atac.signal"),
            proc / "atac",
            chrom_sizes,
            backend=cfg["atac"]["backend"],
            ucsc_tool=cfg["atac"].get("ucsc_bigwig_to_bedgraph"),
            overwrite=args.overwrite,
        )
    else:
        print("      ATAC bigWig not found -- skipping (arm B will fail until you fetch it)")

    (proc / "index.json").write_text(
        json.dumps(
            {
                "window_bp": window_bp,
                "n_bins": n_bins,
                "bin_bp": bin_bp,
                "native_binsize": src.binsize,
                "expected_mode": exp_mode,
                "coarsegrain": {"cutoff": cg_cutoff, "max_levels": cg_levels},
                "clip": clip,
                "diagonal_offset": diag,
                "split": dict(split),
                "windows": kept,
            },
            indent=1,
        )
    )
    print(f"\ndone -> {proc}")
    return 0


def _prepare_synthetic(cfg, proc: Path, overwrite: bool) -> int:
    """Build a fake dataset with real structure so the smoke run is meaningful.

    Targets are a distance-decaying background plus a handful of off-diagonal
    'loops' whose positions are a deterministic function of the window -- so a
    working model can actually reduce the loss, and a broken one cannot.
    """
    n_bins = int(cfg["window"]["n_bins"])
    window_bp = int(cfg["window"]["bp"])
    stride = int(cfg["window"]["train_stride"])

    # Contigs come from the split itself, capped for speed. Deriving them from
    # genome.chroms instead silently produced a dataset with no test
    # chromosomes -- which is exactly the failure a smoke test should catch.
    per_split = {"train": 4, "val": 2, "test": 2}
    sizes: dict[str, int] = {}
    windows = []
    for name, cap in per_split.items():
        chosen = list(cfg["split"][name])[:cap]
        if not chosen:
            raise ValueError(f"split '{name}' is empty; the smoke run needs all three")
        for c in chosen:
            sizes[c] = 200 * window_bp
            for s in range(0, sizes[c] - window_bp + 1, stride):
                windows.append((c, s, s + window_bp))
    windows = sorted(set(windows))

    ii, jj = np.indices((n_bins, n_bins))
    dist = np.abs(ii - jj).astype(np.float32)

    kept, tgts, masks = [], [], []
    for chrom, start, end in windows:
        rng = np.random.default_rng(abs(hash((chrom, start))) % (2**32))
        m = np.zeros((n_bins, n_bins), dtype=np.float32)
        for _ in range(3):
            a, b = sorted(rng.integers(0, n_bins, size=2))
            if b - a < 3:
                continue
            m += 1.5 * np.exp(-(((ii - a) ** 2 + (jj - b) ** 2) / 8.0))
            m += 1.5 * np.exp(-(((ii - b) ** 2 + (jj - a) ** 2) / 8.0))
        m += rng.normal(0, 0.05, m.shape).astype(np.float32)
        m = 0.5 * (m + m.T)
        np.clip(m, -2, 2, out=m)
        valid = dist >= int(cfg["target"]["diagonal_offset"])
        m = np.where(valid, m, 0.0).astype(np.float32)

        kept.append({"chrom": chrom, "start": start, "end": end, "index": len(kept)})
        tgts.append(m.astype(np.float16))
        masks.append(np.packbits(valid.ravel()))

    proc.mkdir(parents=True, exist_ok=True)
    np.save(proc / "targets.npy", np.stack(tgts))
    np.save(proc / "masks.npy", np.stack(masks))
    np.save(proc / "expected.npy", np.ones(n_bins, dtype=np.float64))
    (proc / "index.json").write_text(
        json.dumps(
            {
                "window_bp": window_bp,
                "n_bins": n_bins,
                "bin_bp": window_bp // n_bins,
                "native_binsize": window_bp // n_bins,
                "expected_mode": "synthetic",
                "clip": 2.0,
                "diagonal_offset": int(cfg["target"]["diagonal_offset"]),
                "split": dict(cfg["split"]),
                "windows": kept,
                "synthetic": True,
            },
            indent=1,
        )
    )
    print(f"synthetic dataset: {len(kept)} windows -> {proc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
