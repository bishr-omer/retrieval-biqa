"""Shared setup for every experiment: data, splits, cached features, and aggregation."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from data.datasets import load
from data.manifest import load_manifest, load_splits
from data.splits import assert_disjoint, official_or_random
from features.load_cached import load_features


def get_data(cfg: dict):
    """Return (df, splits). cfg['source'] == 'manifest' reads multimodal-biqa manifests/splits."""
    if cfg.get("source") == "manifest":
        root = cfg["project_root"]
        df = load_manifest(root, cfg["dataset"])
        splits = load_splits(root, cfg["dataset"], df, seed=cfg["seed"], kind=cfg.get("split_kind", "reference"))
    else:
        df = load(cfg["dataset"], cfg["data_root"])
        splits = [official_or_random(df, seed=cfg["seed"])]
    n_max = cfg.get("max_splits")
    splits = splits[:n_max] if n_max else splits
    for s in splits:
        assert_disjoint({k: v for k, v in s.items() if k in ("train", "val", "test")})
        # Scene-level leakage check: no reference scene may appear in both train and test.
        shared = set(df["ref"].values[s["train"]]) & set(df["ref"].values[s["test"]]) if "ref" in df else set()
        assert not shared, f"{len(shared)} reference scenes appear in both train and test"
    return df, splits


def feature_getter(cfg: dict, df: pd.DataFrame, feats_by_emb: dict | None = None):
    names = df["name"].tolist()

    @lru_cache(maxsize=None)
    def get(emb: str) -> np.ndarray:
        if feats_by_emb is not None:
            return feats_by_emb[emb]
        return load_features(cfg["feat_root"], cfg["dataset"], emb, names)

    return get


def summarise(per_split: list[dict]) -> dict:
    """Mean and std across splits for every numeric leaf, e.g. test.srcc."""
    flat = [pd.json_normalize(r, sep=".").iloc[0].to_dict() for r in per_split]
    df = pd.DataFrame(flat)
    num = df.select_dtypes("number")
    return {c: {"mean": float(num[c].mean()), "std": float(num[c].std(ddof=0))} for c in num.columns}
