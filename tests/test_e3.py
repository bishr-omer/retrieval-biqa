"""E3 checks on synthetic data and on a mock biqa-revision layout."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.splits import random_split  # noqa: E402
from features.load_cached import l2n  # noqa: E402
from scripts import run_e3  # noqa: E402

BASE = {"seed": 0, "N": [10, 50], "draws": 3, "alphas": [0.1, 1.0], "refit_target_weight": 0.1,
        "retrieval": [{"index": "source", "k": 20, "tau": 0.2, "lam": 1.0},
                      {"index": "mixed", "k": 20, "tau": 0.2, "lam": 1.0},
                      {"index": "target", "k": 5, "tau": 0.2, "lam": 1.0}]}


def shifted(seed=0, d=24):
    """Target scenes carry a nonlinear offset that the source never shows."""
    rng = np.random.default_rng(seed)
    cs = rng.normal(size=(20, d)) * 2
    ns, nt = 1500, 400
    c_s, c_t = rng.integers(0, 20, ns), rng.integers(0, 20, nt)
    Xs = cs[c_s] + rng.normal(size=(ns, d)); Xt = cs[c_t] + rng.normal(size=(nt, d))
    ys = 0.1 * Xs[:, 0] + 0.02 * rng.normal(size=ns)
    yt = 0.1 * Xt[:, 0] + 0.3 * np.sign(np.sin(5 * c_t)) + 0.02 * rng.normal(size=nt)
    mk = lambda y, p: pd.DataFrame({"name": [f"{p}{i}" for i in range(len(y))],  # noqa: E731
                                    "ref": [f"{p}{i}" for i in range(len(y))], "mos": y})
    sdf, tdf = mk(ys, "s"), mk(yt, "t")
    splits = [random_split(ns, seed=i) for i in range(2)]
    return (sdf, splits, l2n(Xs)), (tdf, l2n(Xt))


def test_supports_reproducible_and_disjoint():
    a = run_e3.support_draws(300, [10, 50], 4, seed=0)
    b = run_e3.support_draws(300, [10, 50], 4, seed=0)
    assert all((a[k] == b[k]).all() for k in a)
    assert len({tuple(v) for v in a.values()}) == len(a)  # draws differ
    assert all(len(np.unique(v)) == len(v) for v in a.values())


def test_target_retrieval_recovers_shift(tmp_path):
    src, tgt = shifted()
    r = run_e3.run({**BASE, "embeddings": ["x"], "out_dir": str(tmp_path / "e3")}, src=src, tgt=tgt)
    s = r["summary"].set_index(["condition", "N"])
    zero = s.loc[("zero-shot", 50), "srcc_mean"]
    assert s.loc[("residual kNN [target] k=5", 50), "srcc_mean"] > zero + 0.1
    assert s.loc[("residual kNN [target] k=5", 50), "frac_improved"] == 1.0
    assert abs(s.loc[("zero-shot", 50), "delta_mean"]) < 1e-12
    # Each support file lists exactly N images and none is in the corresponding test set.
    per = pd.read_csv(tmp_path / "e3/per_run.csv")
    assert per.groupby(["source_split", "N", "draw"]).condition.nunique().eq(6).all()
    names = (tmp_path / "e3/supports/N10_d0.txt").read_text().split()
    assert len(names) == 10


def test_e3_on_mock_project(tmp_path):
    rng = np.random.default_rng(1)
    for d in ("manifests", "splits", "cache"):
        (tmp_path / d).mkdir()
    for ds, n, lo, hi in (("koniq10k", 800, 1, 5), ("liveitw", 300, 0, 100)):
        names = [f"{ds}_{i}.jpg" for i in range(n)]
        X = rng.normal(size=(n, 16)).astype(np.float32)
        q = 1 / (1 + np.exp(-X[:, 0]))
        pd.DataFrame({"image": names, "ref": names, "mos": lo + (hi - lo) * q}).to_csv(
            tmp_path / f"manifests/{ds}.csv", index=False)
        for bb, F in (("siglip", X), ("cliph", X[:, :12]), ("nss", X[:, :6] * 30 + 5)):
            np.savez(tmp_path / f"cache/{ds}_{bb}.npz", names=np.array(names), feats=F)
        sp = []
        for s in range(3):
            p = np.random.default_rng(s).permutation(n)
            a, b = int(0.6 * n), int(0.8 * n)
            sp.append({"dataset": ds, "fold": s, "train": [names[i] for i in p[:a]],
                       "val": [names[i] for i in p[a:b]], "test": [names[i] for i in p[b:]]})
        (tmp_path / f"splits/{ds}_reference.json").write_text(json.dumps(sp))
    cfg = {**BASE, "source": "manifest", "project_root": str(tmp_path), "feat_root": str(tmp_path / "cache"),
           "source_dataset": "koniq10k", "target": "liveitw", "split_kind": "reference",
           "embeddings": ["concat:siglip+cliph+nss"], "source_splits": 2, "out_dir": str(tmp_path / "res/e3")}
    r = run_e3.run(cfg)
    assert r["ceiling"]["srcc_mean"] > 0.5
    assert r["summary"].query("condition == 'zero-shot'").srcc_mean.min() > 0.5
    assert (tmp_path / "res/cost.csv").exists()
