"""E1: training-free kNN regression on KonIQ.

Protocol (no test leakage):
  index    = KonIQ train split
  tune     = choose (embedding, k, tau) by SRCC on the validation split
  report   = the chosen config once on the test split, plus the full sweep on val
Also runs the shuffled-score check on the chosen config (a cheap preview of E4).

Usage: python scripts/run_e1.py --config configs/e1_koniq.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.datasets import load  # noqa: E402
from data.splits import assert_disjoint, official_or_random  # noqa: E402
from eval.cost import log_row  # noqa: E402
from eval.metrics import report  # noqa: E402
from features.load_cached import load_features  # noqa: E402
from index.faiss_index import Index  # noqa: E402
from models.knn_regressor import KNNRegressor  # noqa: E402


def run(cfg: dict, df: pd.DataFrame | None = None, feats_by_emb: dict | None = None) -> dict:
    np.random.seed(cfg["seed"])
    df = load(cfg["dataset"], cfg["data_root"]) if df is None else df
    split = official_or_random(df, seed=cfg["seed"])
    assert_disjoint(split)
    y = df["mos"].values
    tr, va, te = split["train"], split["val"], split["test"]

    rows = []
    for emb in cfg["embeddings"]:
        X = feats_by_emb[emb] if feats_by_emb else load_features(
            cfg["feat_root"], cfg["dataset"], emb, df["name"].tolist())
        idx = Index(X[tr], y[tr], ids=tr)
        for k in cfg["k"]:
            for tau in cfg["tau"]:
                m = KNNRegressor(idx, k=k, tau=tau)
                r = report(m.predict(X[va]), y[va])
                rows.append({"embedding": emb, "k": k, "tau": tau, **{f"val_{a}": b for a, b in r.items()}})

    sweep = pd.DataFrame(rows).sort_values("val_srcc", ascending=False)
    best = sweep.iloc[0]
    X = feats_by_emb[best.embedding] if feats_by_emb else load_features(
        cfg["feat_root"], cfg["dataset"], best.embedding, df["name"].tolist())
    idx = Index(X[tr], y[tr], ids=tr)
    m = KNNRegressor(idx, k=int(best.k), tau=float(best.tau))

    t0 = time.perf_counter()
    pred = m.predict(X[te])
    latency_ms = 1000 * (time.perf_counter() - t0) / len(te)
    test = report(pred, y[te])

    rng = np.random.default_rng(cfg["seed"])
    shuffled = report(m.predict(X[te], shuffle_rng=rng), y[te])

    result = {
        "best_config": {"embedding": best.embedding, "k": int(best.k), "tau": float(best.tau)},
        "val": {"srcc": float(best.val_srcc), "plcc": float(best.val_plcc)},
        "test": test,
        "test_shuffled_scores": shuffled,
        "n": {"train_index": int(len(tr)), "val": int(len(va)), "test": int(len(te))},
    }

    out = Path(cfg["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(out / "sweep_val.csv", index=False)
    (out / "result.json").write_text(json.dumps(result, indent=2))
    log_row(Path(cfg["out_dir"]).parent / "cost.csv", method=f"kNN ({best.embedding}, k={int(best.k)})",
            trainable_params=0, train_gpu_hours=0, latency_ms_per_img=round(latency_ms, 4),
            index_mb=round(idx.feats.nbytes / 2**20, 2), srcc=round(test["srcc"], 4),
            plcc=round(test["plcc"], 4), notes="search time only; features precomputed")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = yaml.safe_load(open(ap.parse_args().config))
    print(json.dumps(run(cfg), indent=2))
