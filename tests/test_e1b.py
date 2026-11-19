"""E1b sanity checks on synthetic data."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.cost import log_row  # noqa: E402
from features.load_cached import l2n  # noqa: E402
from scripts import run_e1b  # noqa: E402

CFG = {"seed": 0, "embeddings": ["x"], "pls_dims": [4, 8], "k": [5, 20], "tau": [0.1, 1.0],
       "lam": [0.0, 0.5, 1.0], "alphas": [0.1, 1.0]}


def data(cluster_bias: bool, n=1500, d=32, seed=0):
    rng = np.random.default_rng(seed)
    c = rng.integers(0, 30, n)
    centres = rng.normal(size=(30, d)) * 2
    X = centres[c] + rng.normal(size=(n, d))
    y = 0.3 * X[:, 0] / 3 + 0.02 * rng.normal(size=n)
    if cluster_bias:  # nonlinear per-scene offset: a linear probe cannot fit this, neighbours can
        y = y + 0.4 * np.sign(np.sin(3 * c))
    df = pd.DataFrame({"name": [f"{i}.jpg" for i in range(n)], "ref": [f"{i}.jpg" for i in range(n)], "mos": y})
    return df, {"x": l2n(X)}


def test_residual_helps_when_neighbours_carry_signal(tmp_path):
    df, F = data(True)
    r = run_e1b.run({**CFG, "out_dir": str(tmp_path / "e1b")}, df=df, feats_by_emb=F)["x"]["residual_knn"]
    s = r["summary"]
    assert s["test.srcc"]["mean"] > s["test_probe_only.srcc"]["mean"] + 0.05
    assert s["test_shuffled_residuals.srcc"]["mean"] < s["test.srcc"]["mean"]
    assert all(p["config"]["lam"] > 0 for p in r["per_split"])


def test_residual_does_not_hurt_when_no_signal(tmp_path):
    df, F = data(False)
    s = run_e1b.run({**CFG, "out_dir": str(tmp_path / "e1b")}, df=df, feats_by_emb=F)["x"]["residual_knn"]["summary"]
    assert s["test.srcc"]["mean"] > s["test_probe_only.srcc"]["mean"] - 0.02


def test_cost_rows_replaced_not_duplicated(tmp_path):
    p = tmp_path / "cost.csv"
    log_row(p, method="a", srcc=1); log_row(p, method="b", srcc=2); log_row(p, method="a", srcc=3)
    df = pd.read_csv(p)
    assert df.method.tolist() == ["b", "a"] and df.srcc.tolist() == [2, 3]
