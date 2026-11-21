"""Config loading with single-level `inherit:` merging.

Deliberately plain dicts wrapped in attribute access rather than a dataclass tree:
the config is written by hand and read in a dozen places, and a schema would cost
more than it catches for a repo this size.
"""

from __future__ import annotations

import copy
import math
import os
from pathlib import Path
from typing import Any

import yaml


class Cfg(dict):
    """dict with attribute access, recursively applied."""

    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return Cfg(val) if isinstance(val, dict) else val

    def __setattr__(self, key: str, val: Any) -> None:
        self[key] = val

    def get_cfg(self, key: str, default: Any = None) -> Any:
        val = self.get(key, default)
        return Cfg(val) if isinstance(val, dict) else val


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | os.PathLike) -> Cfg:
    """Load a YAML config, resolving a single `inherit:` parent relative to it."""
    path = Path(path).resolve()
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    parent_name = raw.pop("inherit", None)
    if parent_name:
        parent = load_config(path.parent / parent_name)
        raw = _deep_merge(dict(parent), raw)

    cfg = Cfg(raw)
    cfg["_config_path"] = str(path)
    cfg.setdefault("name", path.stem)
    _validate(cfg)
    return cfg


def _validate(cfg: Cfg) -> None:
    """Catch the config mistakes that otherwise surface as a cryptic shape error
    several hundred GPU-seconds into a run."""
    w = cfg["window"]
    m = cfg["model"]
    pools = [int(p) for p in m["trunk_pools"]]
    if len(pools) != len(m["trunk_channels"]):
        raise ValueError(
            f"model.trunk_pools has {len(pools)} entries but model.trunk_channels has "
            f"{len(m['trunk_channels'])}; they describe the same stages"
        )
    factor = math.prod(pools)
    if w["bp"] // w["n_bins"] != factor or w["bp"] % w["n_bins"] != 0:
        raise ValueError(
            f"window.bp ({w['bp']}) / window.n_bins ({w['n_bins']}) = "
            f"{w['bp'] / w['n_bins']:g} bp per bin, but prod(trunk_pools) = {factor}. "
            "The trunk must downsample the sequence axis by exactly the bin size."
        )
    if w["bp"] % w["n_bins"] != 0:
        raise ValueError("window.bp must be divisible by window.n_bins")

    shift = cfg["train"]["augment"]["shift_bp"]
    bin_bp = w["bp"] // w["n_bins"]
    if shift >= bin_bp:
        raise ValueError(
            f"augment.shift_bp ({shift}) must be smaller than one bin ({bin_bp} bp), "
            "otherwise the jitter moves the target grid and the labels go stale."
        )

    split = cfg["split"]
    seen: dict[str, str] = {}
    for name in ("train", "val", "test"):
        for c in split[name]:
            if c in seen:
                raise ValueError(f"{c} appears in both {seen[c]} and {name} splits")
            seen[c] = name


def is_url(s: str) -> bool:
    return str(s).startswith(("http://", "https://"))


def resolve(cfg: Cfg, key_path: str) -> Path | str:
    """Resolve a path-valued config entry against paths.root.

    http(s) URLs are returned unchanged (as str): a remote .hic is read with
    HTTP range requests, so the 95 GB file never has to be downloaded.
    """
    node: Any = cfg
    for part in key_path.split("."):
        node = node[part]
    if isinstance(node, str) and is_url(node):
        return node
    root = Path(cfg["paths"]["root"]).resolve()
    p = Path(node)
    return p if p.is_absolute() else (root / p)


def run_dir(cfg: Cfg) -> Path:
    d = Path(cfg["paths"]["runs"]) / cfg["name"]
    if not d.is_absolute():
        d = Path(cfg["paths"]["root"]).resolve() / d
    d.mkdir(parents=True, exist_ok=True)
    return d
