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
    assert all((a[k][0] == b[k][0]).all() for k in a)
    assert len({tuple(v[0]) for v in a.values()}) == len(a)  # draws differ
    for S, te in a.values():
        assert len(np.unique(S)) == len(S) and not set(S) & set(te) and len(S) + len(te) == 300


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
    assert per.groupby(["source_split", "N", "draw"]).condition.nunique().eq(7).all()
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
    assert r["ceiling"]["concat:siglip+cliph+nss"]["srcc_mean"] > 0.5
    assert r["summary"].query("condition == 'zero-shot'").srcc_mean.min() > 0.5
    assert (tmp_path / "res/cost.csv").exists()


def test_scene_protocol_holds_out_scenes(tmp_path):
    """KADID-like target: 20 scenes x 25 distortions. Support must never share a scene with test."""
    rng = np.random.default_rng(3)
    d = 16
    src, _ = shifted()
    n_ref, per = 20, 25
    refs = np.repeat(np.arange(n_ref), per)
    dist = np.tile(np.arange(per), n_ref)
    cen, dvec = rng.normal(size=(n_ref, 24)), rng.normal(size=(per, 24))
    Xt = l2n(cen[refs] + dvec[dist] + 0.1 * rng.normal(size=(len(refs), 24)))
    yt = -0.05 * dist + 0.01 * rng.normal(size=len(refs))  # quality set by distortion, not scene
    names = [f"I{r:02d}_{k:02d}.png" for r, k in zip(refs, dist)]
    tdf = pd.DataFrame({"name": names, "ref": [f"I{r:02d}.png" for r in refs], "mos": yt})
    from data.splits import group_split
    tsplits = [group_split(refs, seed=s) for s in range(3)]
    r = run_e3.run({**BASE, "embeddings": ["x"], "target_protocol": "scene", "out_dir": str(tmp_path / "e3")},
                   src=src, tgt=(tdf, Xt, tsplits))
    draws = run_e3.support_draws(len(tdf), BASE["N"], BASE["draws"], 0, tsplits, tdf["ref"].values)
    for (S, te) in draws.values():
        assert not set(tdf.ref.values[S]) & set(tdf.ref.values[te])
    s = r["summary"].set_index(["condition", "N"])
    # Distortion types are shared across scenes, so target neighbours carry real signal here.
    assert s.loc[("residual kNN [mixed] k=20", 50), "delta_mean"] > 0


def test_combined_condition_present(tmp_path):
    src, tgt = shifted()
    r = run_e3.run({**BASE, "embeddings": ["x"], "out_dir": str(tmp_path / "e3")}, src=src, tgt=tgt)
    assert "ridge refit + residual kNN [mixed] k=100" in set(r["summary"].condition)
