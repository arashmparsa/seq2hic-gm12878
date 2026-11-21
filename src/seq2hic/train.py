"""Training loop: AMP, gradient accumulation, cosine schedule, early stopping.

Both arms run this same loop with the same seed and schedule. The only thing the
config changes is `inputs.atac`, which changes the first conv layer's input
channel count and nothing else -- that is what makes the delta interpretable.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import load_config, run_dir
from .dataset import HiCWindowDataset, WindowIndex
from .metrics import map_correlation
from .model import build_model, masked_mse


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_at(step: int, total: int, base_lr: float, warmup: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * min(1.0, prog)))


def make_loaders(cfg, index: WindowIndex):
    tcfg = cfg["train"]
    train_ds = HiCWindowDataset(cfg, index, "train", training=True)
    val_ds = HiCWindowDataset(cfg, index, "val", training=False)
    common = dict(num_workers=int(tcfg["num_workers"]), pin_memory=torch.cuda.is_available())
    train_dl = DataLoader(
        train_ds, batch_size=int(tcfg["batch_size"]), shuffle=True, drop_last=True, **common
    )
    val_dl = DataLoader(val_ds, batch_size=int(tcfg["batch_size"]), shuffle=False, **common)
    return train_ds, val_ds, train_dl, val_dl


@torch.no_grad()
def validate(model, loader, device, amp: bool) -> dict[str, float]:
    model.eval()
    losses, corrs = [], []
    for x, y, m in loader:
        x, y, m = x.to(device), y.to(device), m.to(device)
        with torch.autocast(device_type=device.type, enabled=amp):
            pred = model(x)
        losses.append(masked_mse(pred.float(), y, m).item())
        corrs.extend(map_correlation(pred.float().cpu().numpy(), y.cpu().numpy(), m.cpu().numpy()))
    model.train()
    finite = [c for c in corrs if np.isfinite(c)]
    return {
        "val_loss": float(np.mean(losses)) if losses else float("nan"),
        "val_pearson": float(np.mean(finite)) if finite else float("nan"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="seq2hic.train", description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--max-hours", type=float, default=None,
        help="stop cleanly after the epoch that would push past this wall-clock budget "
        "(for free GPU sessions with a hard time limit; rerun with --resume)",
    )
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    tcfg = cfg["train"]
    set_seed(int(tcfg["seed"]))

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = bool(tcfg["amp"]) and device.type == "cuda"
    if bool(tcfg["amp"]) and device.type != "cuda":
        print("note: amp requested but no CUDA device; running fp32")

    proc = Path(cfg["paths"]["proc"])
    if not proc.is_absolute():
        proc = Path(cfg["paths"]["root"]).resolve() / proc
    index = WindowIndex(proc)

    train_ds, val_ds, train_dl, val_dl = make_loaders(cfg, index)
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    out = run_dir(cfg)
    print(f"\narm '{cfg['name']}'  device={device}  params={n_params/1e6:.2f}M")
    print(f"  train windows {len(train_ds)}   val windows {len(val_ds)}")
    print(f"  input channels {model.in_channels}  (ATAC {'on' if cfg['inputs']['atac'] else 'off'})")
    print(f"  2D receptive field {model.receptive_field_2d_bins} bins")
    print(f"  run dir {out}\n")

    opt = torch.optim.AdamW(
        model.parameters(), lr=float(tcfg["lr"]), weight_decay=float(tcfg["weight_decay"])
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    accum = int(tcfg["grad_accum"])
    epochs = int(tcfg["epochs"])
    steps_per_epoch = max(1, len(train_dl) // accum)
    total_steps = steps_per_epoch * epochs

    start_epoch, best, bad_epochs, gstep = 0, float("inf"), 0, 0
    history: list[dict] = []
    ckpt_path = out / "last.pt"
    if args.resume and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        if "scaler" in state:
            scaler.load_state_dict(state["scaler"])
        start_epoch, best, gstep = state["epoch"] + 1, state["best"], state["gstep"]
        bad_epochs = int(state.get("bad_epochs", 0))
        history = list(state.get("history", []))
        if "torch_rng" in state:
            torch.set_rng_state(state["torch_rng"].cpu())
        print(f"resumed from epoch {start_epoch}")
        if bad_epochs >= int(tcfg["early_stop_patience"]):
            print("  run had already early-stopped; nothing to do")
            return 0

    t_start = time.time()
    out_of_time = False
    for epoch in range(start_epoch, epochs):
        model.train()
        t0 = time.time()
        running, seen = 0.0, 0
        opt.zero_grad(set_to_none=True)

        bar = tqdm(train_dl, desc=f"epoch {epoch+1}/{epochs}", unit="win")
        for i, (x, y, m) in enumerate(bar):
            x, y, m = x.to(device, non_blocking=True), y.to(device), m.to(device)
            with torch.autocast(device_type=device.type, enabled=amp):
                loss = masked_mse(model(x).float(), y, m)
            scaler.scale(loss / accum).backward()

            if (i + 1) % accum == 0:
                for g in opt.param_groups:
                    g["lr"] = lr_at(gstep, total_steps, float(tcfg["lr"]), int(tcfg["warmup_steps"]))
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg["clip_grad_norm"]))
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                gstep += 1

            running += loss.item()
            seen += 1
            bar.set_postfix(loss=f"{running/seen:.4f}")

        stats = validate(model, val_dl, device, amp)
        row = {
            "epoch": epoch,
            "train_loss": running / max(1, seen),
            **stats,
            "seconds": round(time.time() - t0, 1),
        }
        history.append(row)
        print(
            f"  epoch {epoch+1}: train {row['train_loss']:.4f}  "
            f"val {row['val_loss']:.4f}  r={row['val_pearson']:.4f}  ({row['seconds']}s)"
        )

        improved = stats["val_loss"] < best
        if improved:
            best, bad_epochs = stats["val_loss"], 0
            torch.save({"model": model.state_dict(), "config": dict(cfg), "epoch": epoch}, out / "best.pt")
        else:
            bad_epochs += 1

        # last.pt is written after best/bad_epochs are updated so a resume sees
        # exactly the state the next epoch would have seen.
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best": best,
                "bad_epochs": bad_epochs,
                "gstep": gstep,
                "history": history,
                "torch_rng": torch.get_rng_state(),
                "config": dict(cfg),
            },
            ckpt_path,
        )
        (out / "history.json").write_text(json.dumps(history, indent=1))

        if bad_epochs >= int(tcfg["early_stop_patience"]):
            print(f"  early stop: no val improvement for {bad_epochs} epochs")
            break
        if args.max_hours is not None:
            elapsed = time.time() - t_start
            per_epoch = elapsed / (epoch - start_epoch + 1)
            if elapsed + per_epoch > args.max_hours * 3600:
                print(f"  time budget: stopping after epoch {epoch+1}; rerun with --resume")
                out_of_time = True
                break

    (out / "history.json").write_text(json.dumps(history, indent=1))
    if not out_of_time:
        (out / "DONE").write_text("training finished\n")  # lets scripts skip a completed arm
    print(f"\nbest val loss {best:.4f} -> {out/'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
