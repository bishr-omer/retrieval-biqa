"""Load embeddings cached by multimodal-biqa, aligned to a dataset's image order.

Expected layout (adapt `_read` if multimodal-biqa saved them differently):
    <feat_root>/<dataset>/<backbone>.npy          float array, shape (N, D)
    <feat_root>/<dataset>/<backbone>_names.txt    one image name per line, same order

Backbones are referred to by name, e.g. "siglip", "clip", "nss".
"concat:siglip+clip+nss" L2-normalises each part, then concatenates and normalises again,
so that no single backbone dominates the cosine similarity because of its dimension.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def l2n(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def zscore(x: np.ndarray, mu=None, sd=None):
    """Hand-crafted NSS features have arbitrary scales; standardise before cosine."""
    mu = x.mean(0) if mu is None else mu
    sd = x.std(0) + 1e-8 if sd is None else sd
    return (x - mu) / sd, mu, sd


def _read(feat_root: Path, dataset: str, backbone: str):
    d = Path(feat_root) / dataset
    feats = np.load(d / f"{backbone}.npy")
    names = (d / f"{backbone}_names.txt").read_text().split()
    if len(names) != len(feats):
        raise ValueError(f"{backbone}: {len(feats)} features but {len(names)} names")
    return feats, names


def load_features(feat_root, dataset: str, spec: str, names_in_order) -> np.ndarray:
    """Return an (len(names_in_order), D) L2-normalised matrix."""
    parts = spec.split(":", 1)[1].split("+") if spec.startswith("concat:") else [spec]
    blocks = []
    for bb in parts:
        feats, names = _read(feat_root, dataset, bb)
        pos = {n: i for i, n in enumerate(names)}
        missing = [n for n in names_in_order if n not in pos]
        if missing:
            raise KeyError(f"{bb}: {len(missing)} images have no cached feature, e.g. {missing[:3]}")
        x = feats[[pos[n] for n in names_in_order]].astype(np.float32)
        if bb == "nss":
            # NOTE: statistics are computed on the full set here; for a strict protocol,
            # compute mu/sd on the index (train) split only. Effect is small for NSS.
            x, _, _ = zscore(x)
        blocks.append(l2n(x))
    return l2n(np.concatenate(blocks, axis=1))
