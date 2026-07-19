"""Arm A vs arm B: the ablation table.

Deltas use a **paired** bootstrap over windows -- both arms are evaluated on the
identical window list, so pairing removes the shared between-window variance and
gives a much tighter, honest interval than comparing two independent CIs.

Emits ablation.json and ablation.md into the first run's parent directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import metrics as M


def _load(run: Path, split: str) -> dict:
    p = run / f"eval_{split}.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing -- run `seq2hic.evaluate` for this arm first")
    return json.loads(p.read_text())


def _fmt(x: float, nd: int = 4) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="seq2hic.ablation", description=__doc__)
    ap.add_argument("--runs", nargs=2, required=True, metavar=("BASELINE", "TREATMENT"),
                    help="two run directories, baseline first (e.g. runs/seq_only runs/seq_atac)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    a_dir, b_dir = Path(args.runs[0]), Path(args.runs[1])
    a, b = _load(a_dir, args.split), _load(b_dir, args.split)

    if a["n_windows"] != b["n_windows"]:
        raise ValueError(
            f"arms evaluated on different window counts ({a['n_windows']} vs "
            f"{b['n_windows']}) -- they must share a prepared dataset for the "
            "delta to mean anything"
        )

    # summary key in eval json -> per-window array key
    PAIRED = (
        ("map_pearson", "pearson", "map Pearson"),
        ("map_spearman", "spearman", "map Spearman"),
        ("insulation_pearson", "insulation", "insulation Pearson"),
    )

    rows = []
    for key, pw_key, label in PAIRED:
        pa = a["per_window"][pw_key]
        pb = b["per_window"][pw_key]
        d = M.paired_bootstrap_delta(pa, pb, args.bootstrap)
        rows.append({
            "metric": label,
            "baseline": a[key]["mean"],
            "treatment": b[key]["mean"],
            "delta": d["delta"], "ci_lo": d["lo"], "ci_hi": d["hi"],
            "significant": bool(d["lo"] > 0 or d["hi"] < 0),
        })

    for key, label in (
        ("boundary_f1", "boundary F1"),
        ("ep_auprc", "E-P loop AUPRC"),
        ("ep_auroc", "E-P loop AUROC"),
    ):
        if key not in a or key not in b:
            continue
        rows.append({
            "metric": label, "baseline": a[key], "treatment": b[key],
            "delta": b[key] - a[key], "ci_lo": None, "ci_hi": None,
            "significant": None,
        })

    md = [
        f"# Ablation: `{a['arm']}` vs `{b['arm']}` (split: {args.split})",
        "",
        f"{a['n_windows']} held-out windows. Deltas for per-window metrics use a "
        f"paired bootstrap ({args.bootstrap} resamples, 95% CI); pooled metrics "
        "(boundary F1, E-P AUPRC/AUROC) are single numbers over all pairs and "
        "carry no interval.",
        "",
        f"| metric | {a['arm']} | {b['arm']} | Δ | 95% CI |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        ci = ("—" if r["ci_lo"] is None
              else f"[{_fmt(r['ci_lo'])}, {_fmt(r['ci_hi'])}]"
                   + ("**" if r["significant"] else ""))
        md.append(
            f"| {r['metric']} | {_fmt(r['baseline'])} | {_fmt(r['treatment'])} | "
            f"{_fmt(r['delta'])} | {ci} |"
        )

    if "ep_auprc_distance_baseline" in a:
        md += [
            "",
            "### Reference points for E–P AUPRC",
            "",
            f"- distance-only baseline: **{_fmt(a['ep_auprc_distance_baseline'])}** — "
            "a model scoring at or below this has learned distance decay, not looping.",
            f"- observed-map ceiling: **{_fmt(a['ep_auprc_observed_ceiling'])}** — "
            "scoring pairs with the *true* map, i.e. the best achievable given how "
            "labels were defined.",
            f"- positive rate: **{_fmt(a.get('ep_positive_rate'))}** — the AUPRC of "
            "random guessing.",
        ]

    md += ["", "*Rows marked ** have a 95% CI excluding zero.*", ""]

    out_dir = Path(args.out) if args.out else a_dir.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ablation.md").write_text("\n".join(md), encoding="utf-8")
    (out_dir / "ablation.json").write_text(
        json.dumps({"baseline": a["arm"], "treatment": b["arm"],
                    "split": args.split, "rows": rows}, indent=1)
    )
    print("\n".join(md))
    print(f"\n-> {out_dir/'ablation.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
