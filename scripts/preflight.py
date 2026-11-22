"""Pre-launch check for a Kaggle GPU run, meant for a free CPU session (minutes,
not hours). Pushes every train and val window through the real DataLoader with
the configured worker count on Linux (fork), checks every tensor's shape and
finiteness, then takes one forward/backward step and one eval-mode forward.
Exit code 0 = safe to spend GPU hours."""
from __future__ import annotations

import sys
import time

import torch

from seq2hic.config import load_config
from seq2hic.dataset import WindowIndex
from seq2hic.train import make_loaders
from seq2hic.model import build_model


def check(config: str) -> None:
    cfg = load_config(config)
    from pathlib import Path
    proc = Path(cfg["paths"]["proc"])
    if not proc.is_absolute():
        proc = Path(cfg["paths"]["root"]).resolve() / proc
    _, _, train_dl, val_dl = make_loaders(cfg, WindowIndex(proc))
    shape = None
    for name, dl in (("train", train_dl), ("val", val_dl)):
        t0, n = time.time(), 0
        for x, y, m in dl:
            if shape is None:
                shape = (tuple(x.shape[1:]), tuple(y.shape[1:]))
            assert (tuple(x.shape[1:]), tuple(y.shape[1:])) == shape, f"{name}: shape {x.shape} {y.shape}"
            assert torch.isfinite(x).all() and torch.isfinite(y).all(), f"{name}: non-finite batch {n}"
            n += x.shape[0]
        print(f"  {config} {name}: {n} windows ok ({time.time() - t0:.0f}s)", flush=True)
    model = build_model(cfg)
    x, y, m = next(iter(train_dl))
    model.train()
    loss = ((model(x) - y) ** 2 * m).sum() / m.sum().clamp(min=1)
    loss.backward()
    assert torch.isfinite(loss), "train-mode loss not finite"
    model.eval()
    with torch.no_grad():
        out = model(x)
    assert torch.isfinite(out).all(), "eval-mode output not finite"
    print(f"  {config}: train step + eval forward ok (loss {loss.item():.4f})", flush=True)


if __name__ == "__main__":
    for c in sys.argv[1:]:
        check(c)
    print("PREFLIGHT OK")
