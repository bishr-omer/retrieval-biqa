"""Read the manifests and split files from the multimodal-biqa project (biqa-revision),
so that retrieval results are on exactly the same splits as the LoRA results.

Manifest: <project_root>/manifests/<dataset>.csv with columns image, ref, mos
Splits:   <project_root>/splits/<dataset>_reference.json  (a list, one entry per split)

MOS is normalised to [0, 1] with each dataset's fixed scale, so datasets can share one index.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SCALES = {
    "koniq10k": (1.0, 5.0),
    "liveitw": (0.0, 100.0),
    "kadid10k": (1.0, 5.0),
    "koniq": (1.0, 5.0), "livec": (0.0, 100.0), "kadid": (1.0, 5.0),
}

_VAL_KEYS = ("val", "valid", "validation", "dev")


def load_manifest(project_root: str | Path, dataset: str) -> pd.DataFrame:
    d = pd.read_csv(Path(project_root) / "manifests" / f"{dataset}.csv")
    lo, hi = SCALES[dataset]
    out = pd.DataFrame({"name": d["image"].astype(str), "ref": d["ref"].astype(str),
                        "mos_raw": d["mos"].astype(float)})
    out["mos"] = (out["mos_raw"] - lo) / (hi - lo)
    if dataset.startswith("kadid"):
        parts = out["name"].str.rsplit(".", n=1).str[0].str.split("_", expand=True)
        out["dist_type"] = parts[1].astype(int)
        out["dist_level"] = parts[2].astype(int)
    return out


def _to_idx(values, df: pd.DataFrame) -> np.ndarray:
    values = list(values)
    if len(values) == 0:
        return np.array([], dtype=int)
    if all(isinstance(v, (int, np.integer)) for v in values):
        return np.sort(np.asarray(values, dtype=int))
    values = [str(v) for v in values]
    vs = set(values)
    if vs <= set(df["name"]):
        return np.flatnonzero(df["name"].isin(vs).values)
    if vs <= set(df["ref"]):  # split given by reference scene: take every image of that scene
        return np.flatnonzero(df["ref"].isin(vs).values)
    miss = [v for v in values if v not in set(df["name"]) and v not in set(df["ref"])]
    raise KeyError(f"{len(miss)} split entries match neither image nor ref names, e.g. {miss[:3]}")


def _carve_val(train: np.ndarray, df: pd.DataFrame, seed: int, frac: float = 0.1):
    """No validation set in the file: hold out `frac` of train *scenes* (seeded)."""
    refs = df["ref"].values[train]
    uniq = np.unique(refs)
    rng = np.random.default_rng(seed)
    val_refs = rng.choice(uniq, size=max(1, int(round(frac * len(uniq)))), replace=False)
    is_val = np.isin(refs, val_refs)
    return train[~is_val], train[is_val]


def parse_entry(entry, df: pd.DataFrame, seed: int) -> dict:
    if isinstance(entry, dict):
        keys = {k.lower(): k for k in entry}
        if "train" not in keys or "test" not in keys:
            raise ValueError(f"Split entry has keys {list(entry)}; expected train/test")
        tr = _to_idx(entry[keys["train"]], df)
        te = _to_idx(entry[keys["test"]], df)
        vk = next((keys[k] for k in _VAL_KEYS if k in keys), None)
        va = _to_idx(entry[vk], df) if vk else None
    elif isinstance(entry, (list, tuple)) and len(entry) in (2, 3):
        tr, te = _to_idx(entry[0], df), _to_idx(entry[-1], df)
        va = _to_idx(entry[1], df) if len(entry) == 3 else None
    else:
        raise ValueError(f"Unrecognised split entry of type {type(entry).__name__}")
    carved = va is None
    if carved:
        tr, va = _carve_val(tr, df, seed)
    return {"train": tr, "val": va, "test": te, "val_carved_from_train": carved}


def load_splits(project_root: str | Path, dataset: str, df: pd.DataFrame, seed: int = 0,
                kind: str = "reference") -> list[dict]:
    raw = json.loads((Path(project_root) / "splits" / f"{dataset}_{kind}.json").read_text())
    if isinstance(raw, dict):
        raw = raw.get("splits", list(raw.values()))
    return [parse_entry(e, df, seed + i) for i, e in enumerate(raw)]
