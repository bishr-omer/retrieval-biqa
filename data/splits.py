"""Train / val / test splits with fixed seeds.

KonIQ: use the official 'set' column when available (7058 / 1000 / 2010).
Otherwise: seeded random 70 / 10 / 20 split. For KADID, split by *reference image*
so that no scene appears in both train and test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def official_or_random(df: pd.DataFrame, seed: int = 0, ratios=(0.7, 0.1, 0.2)) -> dict:
    if "set" in df.columns:
        return {
            "train": np.flatnonzero(df["set"].eq("training").values),
            "val": np.flatnonzero(df["set"].eq("validation").values),
            "test": np.flatnonzero(df["set"].eq("test").values),
        }
    return random_split(len(df), seed, ratios)


def random_split(n: int, seed: int = 0, ratios=(0.7, 0.1, 0.2)) -> dict:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    a = int(round(ratios[0] * n))
    b = a + int(round(ratios[1] * n))
    return {"train": np.sort(perm[:a]), "val": np.sort(perm[a:b]), "test": np.sort(perm[b:])}


def group_split(groups: np.ndarray, seed: int = 0, ratios=(0.7, 0.1, 0.2)) -> dict:
    """Split so that all items sharing a group (e.g. KADID reference image) land together."""
    uniq = np.unique(groups)
    g = random_split(len(uniq), seed, ratios)
    return {k: np.flatnonzero(np.isin(groups, uniq[v])) for k, v in g.items()}


def assert_disjoint(split: dict) -> None:
    keys = list(split)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            inter = np.intersect1d(split[keys[i]], split[keys[j]])
            assert inter.size == 0, f"{keys[i]} and {keys[j]} overlap on {inter.size} items"
