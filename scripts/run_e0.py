"""E0: linear probe (ridge regression) on frozen embeddings.

alpha is chosen on the validation split, then reported once on test. The LoRA reference
numbers from the config are written into the same cost table for comparison.

Usage: python scripts/run_e0.py --config configs/e0_koniq.yaml
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
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.datasets import load  # noqa: E402
from data.splits import assert_disjoint, official_or_random  # noqa: E402
from eval.cost import log_row  # noqa: E402
from eval.metrics import report  # noqa: E402
from features.load_cached import load_features  # noqa: E402


def run(cfg: dict, df: pd.DataFrame | None = None, feats_by_emb: dict | None = None) -> dict:
    np.random.seed(cfg["seed"])
    df = load(cfg["dataset"], cfg["data_root"]) if df is None else df
    split = official_or_random(df, seed=cfg["seed"])
    assert_disjoint(split)
    y = df["mos"].values
    tr, va, te = split["train"], split["val"], split["test"]
    cost_path = Path(cfg["out_dir"]).parent / "cost.csv"

    results = {}
    for emb in cfg["embeddings"]:
        X = feats_by_emb[emb] if feats_by_emb else load_features(
            cfg["feat_root"], cfg["dataset"], emb, df["name"].tolist())
        best = max(cfg["alphas"], key=lambda a: report(
            Ridge(alpha=a).fit(X[tr], y[tr]).predict(X[va]), y[va])["srcc"])
        t0 = time.perf_counter()
        model = Ridge(alpha=best).fit(X[tr], y[tr])
        train_s = time.perf_counter() - t0
        test = report(model.predict(X[te]), y[te])
        results[emb] = {"alpha": best, "test": test}
        log_row(cost_path, method=f"linear probe ({emb})", trainable_params=X.shape[1] + 1,
                train_gpu_hours=0, srcc=round(test["srcc"], 4), plcc=round(test["plcc"], 4),
                notes=f"CPU ridge fit {train_s:.2f}s; features precomputed")

    lora = cfg.get("lora_reference") or {}
    if lora.get("srcc") is not None:
        log_row(cost_path, method="LoRA (reference)", **{k: lora.get(k) for k in
                ("srcc", "plcc", "trainable_params", "train_gpu_hours")})

    out = Path(cfg["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = yaml.safe_load(open(ap.parse_args().config))
    print(json.dumps(run(cfg), indent=2))
