"""Evaluate one arm on a held-out split: map metrics + E-P loop retrieval.

Writes to the run directory:
    eval_<split>.json      all metrics with bootstrap CIs
    preds_<split>.npz      predictions, targets, masks (for figures + ablation)
    pairs_<split>.npz      E-P pair table with labels and scores
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import ep_loops as EP
from . import metrics as M
from .config import load_config, resolve, run_dir
from .dataset import HiCWindowDataset, WindowIndex
from .model import build_model


@torch.no_grad()
def predict_split(cfg, index: WindowIndex, split: str, device, ckpt: Path):
    ds = HiCWindowDataset(cfg, index, split, training=False)
    dl = DataLoader(ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=False, num_workers=0)

    model = build_model(cfg).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    preds, targs, masks = [], [], []
    for x, y, m in tqdm(dl, desc=f"  predict[{split}]", unit="win"):
        p = model(x.to(device)).float().cpu().numpy()
        preds.append(p)
        targs.append(y.numpy())
        masks.append(m.numpy())
    return ds, np.concatenate(preds), np.concatenate(targs), np.concatenate(masks)


def map_metrics(pred, targ, mask, cfg) -> dict:
    diag = int(cfg["target"]["diagonal_offset"])
    ecfg = cfg["eval"]
    n_boot = int(ecfg["bootstrap"])

    pear = M.map_correlation(pred, targ, mask, diag, "pearson")
    spear = M.map_correlation(pred, targ, mask, diag, "spearman")
    insul = M.insulation_correlation(pred, targ, mask, int(ecfg["insulation_window_bins"]))
    bf1 = M.boundary_f1(pred, targ, mask, int(ecfg["insulation_window_bins"]),
                        float(ecfg["boundary_prominence"]))

    out = {"per_window": {"pearson": pear, "spearman": spear, "insulation": insul}}
    for name, vals in (("map_pearson", pear), ("map_spearman", spear),
                       ("insulation_pearson", insul)):
        mean, lo, hi = M.bootstrap_ci(vals, n_boot)
        out[name] = {"mean": mean, "ci_lo": lo, "ci_hi": hi,
                     "n": int(np.isfinite(vals).sum())}
    out.update(bf1)
    return out


def ep_metrics(ds, pred, targ, mask, cfg, index) -> tuple[dict, dict]:
    from .genome import read_narrowpeak, read_tss_bed

    ecfg = dict(cfg["eval"]["ep"])
    chroms = {w.chrom for w in ds.windows}

    peaks_path = resolve(cfg, "atac.peaks")
    tss_path = resolve(cfg, "genome.tss")
    if not peaks_path.exists() or not tss_path.exists():
        print("      skipping E-P scoring: ATAC peaks or TSS annotation missing")
        return {}, {}

    peaks = read_narrowpeak(peaks_path, chroms)
    tss = read_tss_bed(tss_path, chroms)
    enh = {
        c: EP.filter_distal(EP.peak_centers(p), tss.get(c, np.empty(0, np.int64)),
                            int(ecfg["enhancer_tss_exclusion_bp"]))
        for c, p in peaks.items()
    }
    n_enh = sum(len(v) for v in enh.values())
    print(f"      {n_enh} distal ATAC peaks, {sum(len(v) for v in tss.values())} TSS "
          f"on {sorted(chroms)}")

    pairs = EP.PairSet()
    n_bins = index.n_bins
    for i, w in enumerate(ds.windows):
        EP.build_pairs_for_window(pairs, i, w.chrom, w.start, w.end, n_bins,
                                  targ[i], pred[i], mask[i].astype(bool), enh, tss, ecfg)

    arr = pairs.as_arrays()
    if arr["observed"].size == 0:
        print("      no E-P candidate pairs found")
        return {}, arr

    bin_bp = index.window_bp // n_bins
    if ecfg["positives"] == "loops_bedpe" and ecfg.get("loops_bedpe"):
        labels = EP.label_by_loops(arr, ds.windows, bin_bp, str(ecfg["loops_bedpe"]))
    else:
        labels = EP.label_by_quantile(arr["observed"], float(ecfg["contact_quantile"]))

    keep = EP.distance_matched_subset(labels, arr["distance_bp"],
                                      int(ecfg["negatives_per_positive"]))
    lab_k = labels[keep]
    res = EP.score(lab_k, arr["predicted"][keep])
    base = EP.score(lab_k, EP.distance_only_baseline(arr["distance_bp"][keep]))
    obs_ceiling = EP.score(lab_k, arr["observed"][keep])

    arr = {**arr, "labels": labels, "kept": keep}
    return {
        "ep_auprc": res["auprc"],
        "ep_auroc": res["auroc"],
        "ep_positive_rate": res["positive_rate"],
        "ep_n_pairs": res["n_pairs"],
        "ep_n_positive": res["n_positive"],
        "ep_auprc_distance_baseline": base["auprc"],
        "ep_auroc_distance_baseline": base["auroc"],
        "ep_auprc_observed_ceiling": obs_ceiling["auprc"],
    }, arr


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="seq2hic.evaluate", description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out = run_dir(cfg)
    ckpt = Path(args.checkpoint) if args.checkpoint else out / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"{ckpt} not found -- train this arm first")

    proc = Path(cfg["paths"]["proc"])
    if not proc.is_absolute():
        proc = Path(cfg["paths"]["root"]).resolve() / proc
    index = WindowIndex(proc)

    print(f"\nevaluating '{cfg['name']}' on split '{args.split}'  ({device})")
    ds, pred, targ, mask = predict_split(cfg, index, args.split, device, ckpt)
    print(f"      {len(ds)} windows on {sorted({w.chrom for w in ds.windows})}")

    report = {"arm": cfg["name"], "split": args.split,
              "atac": bool(cfg["inputs"]["atac"]),
              "n_windows": len(ds), "checkpoint": str(ckpt)}
    report.update(map_metrics(pred, targ, mask, cfg))
    ep, pair_arr = ep_metrics(ds, pred, targ, mask, cfg, index)
    report.update(ep)

    np.savez_compressed(out / f"preds_{args.split}.npz", pred=pred.astype(np.float16),
                        target=targ.astype(np.float16), mask=mask,
                        chrom=np.array([w.chrom for w in ds.windows]),
                        start=np.array([w.start for w in ds.windows]))
    if pair_arr:
        np.savez_compressed(out / f"pairs_{args.split}.npz", **pair_arr)
    (out / f"eval_{args.split}.json").write_text(json.dumps(report, indent=1))

    print(f"\n  map Pearson        {report['map_pearson']['mean']:.4f}  "
          f"[{report['map_pearson']['ci_lo']:.4f}, {report['map_pearson']['ci_hi']:.4f}]")
    print(f"  insulation Pearson {report['insulation_pearson']['mean']:.4f}")
    print(f"  boundary F1        {report['boundary_f1']:.4f}")
    if ep:
        print(f"  E-P AUPRC          {ep['ep_auprc']:.4f}   "
              f"(distance baseline {ep['ep_auprc_distance_baseline']:.4f}, "
              f"observed ceiling {ep['ep_auprc_observed_ceiling']:.4f})")
        print(f"  E-P positive rate  {ep['ep_positive_rate']:.4f}  "
              f"over {ep['ep_n_pairs']} pairs")
    print(f"\n  -> {out/f'eval_{args.split}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
