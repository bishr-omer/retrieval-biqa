"""E1: training-free kNN regression.

Protocol, repeated on every split (5 for your KonIQ split file), reported as mean ± std:
  index    = train split
  tune     = choose (embedding, k, tau) by SRCC on the validation split
  report   = the chosen config once on the test split
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
from eval.cost import log_row  # noqa: E402
from eval.metrics import report  # noqa: E402
from data.splits import official_or_random  # noqa: E402
from eval.protocol import feature_getter, get_data, summarise  # noqa: E402
from index.faiss_index import Index  # noqa: E402
from models.knn_regressor import KNNRegressor  # noqa: E402


def run_split(cfg, df, split, get_X, seed):
    y = df["mos"].values
    tr, va, te = split["train"], split["val"], split["test"]
    rows = []
    for emb in cfg["embeddings"]:
        X = get_X(emb)
        idx = Index(X[tr], y[tr], ids=tr)
        for k in cfg["k"]:
            for tau in cfg["tau"]:
                r = report(KNNRegressor(idx, k=k, tau=tau).predict(X[va]), y[va])
                rows.append({"embedding": emb, "k": k, "tau": tau, **{f"val_{a}": b for a, b in r.items()}})
    sweep = pd.DataFrame(rows).sort_values("val_srcc", ascending=False)
    best = sweep.iloc[0]

    X = get_X(best.embedding)
    idx = Index(X[tr], y[tr], ids=tr)
    m = KNNRegressor(idx, k=int(best.k), tau=float(best.tau))
    t0 = time.perf_counter()
    pred = m.predict(X[te])
    latency_ms = 1000 * (time.perf_counter() - t0) / len(te)
    res = {
        "best_config": {"embedding": best.embedding, "k": int(best.k), "tau": float(best.tau)},
        "val": {"srcc": float(best.val_srcc), "plcc": float(best.val_plcc)},
        "test": report(pred, y[te]),
        "test_shuffled_scores": report(m.predict(X[te], shuffle_rng=np.random.default_rng(seed)), y[te]),
        "latency_ms_per_img": latency_ms,
        "index_mb": idx.feats.nbytes / 2**20,
        "n": {"train_index": int(len(tr)), "val": int(len(va)), "test": int(len(te))},
    }
    return res, sweep


def run(cfg: dict, df=None, feats_by_emb=None, splits=None) -> dict:
    np.random.seed(cfg["seed"])
    if df is None:
        df, splits = get_data(cfg)
    elif splits is None:
        splits = [official_or_random(df, seed=cfg["seed"])]
    get_X = feature_getter(cfg, df, feats_by_emb)
    out = Path(cfg["out_dir"]); out.mkdir(parents=True, exist_ok=True)

    per_split = []
    for i, split in enumerate(splits):
        res, sweep = run_split(cfg, df, split, get_X, cfg["seed"] + i)
        sweep.to_csv(out / f"sweep_val_split{i}.csv", index=False)
        per_split.append(res)
        print(f"split {i}: {res['best_config']}  test SRCC {res['test']['srcc']:.4f}  "
              f"PLCC {res['test']['plcc']:.4f}  shuffled SRCC {res['test_shuffled_scores']['srcc']:.4f}",
              flush=True)

    summary = summarise(per_split)
    result = {"n_splits": len(splits), "summary": summary, "per_split": per_split}
    (out / "result.json").write_text(json.dumps(result, indent=2))

    configs = {json.dumps(r["best_config"]) for r in per_split}
    log_row(out.parent / "cost.csv", method="kNN (E1, tuned per split)", trainable_params=0, train_gpu_hours=0,
            latency_ms_per_img=round(summary["latency_ms_per_img"]["mean"], 4),
            index_mb=round(summary["index_mb"]["mean"], 2),
            srcc=f"{summary['test.srcc']['mean']:.4f} ± {summary['test.srcc']['std']:.4f}",
            plcc=f"{summary['test.plcc']['mean']:.4f} ± {summary['test.plcc']['std']:.4f}",
            notes=f"configs chosen: {'; '.join(sorted(configs))}; search time only")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = yaml.safe_load(open(ap.parse_args().config))
    r = run(cfg)
    s = r["summary"]
    print(f"\nE1 over {r['n_splits']} splits: test SRCC {s['test.srcc']['mean']:.4f} ± {s['test.srcc']['std']:.4f}, "
          f"PLCC {s['test.plcc']['mean']:.4f} ± {s['test.plcc']['std']:.4f}, "
          f"shuffled SRCC {s['test_shuffled_scores.srcc']['mean']:.4f}")
