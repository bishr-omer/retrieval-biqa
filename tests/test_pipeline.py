"""Synthetic-data tests. They check plumbing and leakage guards, not real performance.

Run: python -m pytest -q tests/
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.splits import assert_disjoint, group_split, official_or_random  # noqa: E402
from eval.metrics import distortion_precision_at_k, report  # noqa: E402
from features.load_cached import l2n, load_features  # noqa: E402
from index.faiss_index import Index  # noqa: E402
from models.knn_regressor import KNNRegressor  # noqa: E402
from scripts import run_e0, run_e1  # noqa: E402


def synth(n=1500, d=64, seed=0):
    """Features where quality is a smooth function of one direction plus noise."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = 1 / (1 + np.exp(-X[:, 0])) + 0.05 * rng.normal(size=n)
    sets = np.array(["training"] * int(0.7 * n) + ["validation"] * int(0.1 * n))
    sets = np.concatenate([sets, ["test"] * (n - len(sets))])
    df = pd.DataFrame({"name": [f"img{i}.jpg" for i in range(n)], "mos": y, "set": sets})
    return df, l2n(X)


def test_splits_disjoint():
    df, _ = synth()
    s = official_or_random(df)
    assert_disjoint(s)
    assert sum(len(v) for v in s.values()) == len(df)


def test_group_split_keeps_scenes_together():
    groups = np.repeat(np.arange(81), 125)  # KADID layout
    s = group_split(groups, seed=1)
    assert_disjoint(s)
    for a in ("train", "val"):
        assert not set(groups[s[a]]) & set(groups[s["test"]])


def test_leave_one_out_removes_self():
    df, X = synth()
    idx = Index(X, df["mos"].values)
    _, _, ids_no_ex = idx.search(X[:50], k=5)
    assert (ids_no_ex[:, 0] == np.arange(50)).all()  # self is top-1 without the guard
    _, _, ids = idx.search(X[:50], k=5, exclude=np.arange(50))
    assert not (ids == np.arange(50)[:, None]).any()


def test_same_reference_exclusion():
    rng = np.random.default_rng(0)
    refs = np.repeat(np.arange(20), 125)
    centres = rng.normal(size=(20, 32))
    X = l2n(centres[refs] + 0.1 * rng.normal(size=(len(refs), 32)))
    idx = Index(X, np.zeros(len(X)))
    q = np.arange(0, len(X), 125)
    _, _, ids = idx.search(X[q], k=10, exclude=lambda r, c: refs[c] == refs[q[r]])
    assert not (refs[ids] == refs[q][:, None]).any()


def test_knn_beats_chance_and_shuffle_hurts():
    df, X = synth()
    s = official_or_random(df)
    y = df["mos"].values
    m = KNNRegressor(Index(X[s["train"]], y[s["train"]]), k=10, tau=0.05)
    good = report(m.predict(X[s["test"]]), y[s["test"]])["srcc"]
    bad = report(m.predict(X[s["test"]], shuffle_rng=np.random.default_rng(0)), y[s["test"]])["srcc"]
    assert good > 0.3 and abs(bad) < 0.15


def test_database_update():
    df, X = synth()
    y = df["mos"].values
    idx = Index(X[:1000], y[:1000], ids=np.arange(1000))
    idx.add(X[1000:1100], y[1000:1100], ids=np.arange(1000, 1100))
    assert len(idx) == 1100
    _, ys, ids = idx.search(X[1000:1010], k=1)
    assert (ids[:, 0] == np.arange(1000, 1010)).all()


def test_distortion_precision():
    assert distortion_precision_at_k(np.array([1, 2]), np.array([[1, 1], [3, 2]])) == 0.75


def test_feature_loader_alignment(tmp_path):
    names = [f"a{i}.png" for i in range(10)]
    feats = np.eye(10, dtype=np.float32)
    (tmp_path / "ds").mkdir()
    np.save(tmp_path / "ds" / "siglip.npy", feats)
    (tmp_path / "ds" / "siglip_names.txt").write_text("\n".join(names))
    X = load_features(tmp_path, "ds", "siglip", names[::-1])
    assert np.allclose(X, feats[::-1])
    with pytest.raises(KeyError):
        load_features(tmp_path, "ds", "siglip", ["missing.png"])


def test_e0_e1_end_to_end(tmp_path):
    df, X = synth()
    base = {"dataset": "koniq", "seed": 0}
    feats = {"siglip": X, "clip": X}
    r1 = run_e1.run({**base, "embeddings": ["siglip", "clip"], "k": [5, 10], "tau": [0.05, 1.0],
                     "out_dir": str(tmp_path / "e1")}, df=df, feats_by_emb=feats)
    assert r1["test"]["srcc"] > 0.3
    r0 = run_e0.run({**base, "embeddings": ["siglip"], "alphas": [0.1, 1.0],
                     "out_dir": str(tmp_path / "e0"), "lora_reference": {"srcc": 0.9, "plcc": 0.92}},
                    df=df, feats_by_emb=feats)
    assert "siglip" in r0
    cost = pd.read_csv(tmp_path / "cost.csv")
    assert {"kNN (siglip, k=10)", "linear probe (siglip)", "LoRA (reference)"} & set(cost["method"])
