"""E0: linear probe (ridge regression) on frozen embeddings, on every split.

alpha is chosen on validation, then reported once on test; results are mean ± std over splits.
The LoRA reference numbers from the config go into the same cost table.

Usage: python scripts/run_e0.py --config configs/e0_koniq.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.cost import log_row  # noqa: E402
from eval.metrics import report  # noqa: E402
from data.splits import official_or_random  # noqa: E402
from eval.protocol import feature_getter, get_data, summarise  # noqa: E402


def run(cfg: dict, df=None, feats_by_emb=None, splits=None) -> dict:
    np.random.seed(cfg["seed"])
    if df is None:
        df, splits = get_data(cfg)
    elif splits is None:
        splits = [official_or_random(df, seed=cfg["seed"])]
    get_X = feature_getter(cfg, df, feats_by_emb)
    y = df["mos"].values
    out = Path(cfg["out_dir"]); out.mkdir(parents=True, exist_ok=True)

    results = {}
    for emb in cfg["embeddings"]:
        X = get_X(emb)
        per_split = []
        for s in splits:
            tr, va, te = s["train"], s["val"], s["test"]
            best = max(cfg["alphas"], key=lambda a: report(
                Ridge(alpha=a).fit(X[tr], y[tr]).predict(X[va]), y[va])["srcc"])
            t0 = time.perf_counter()
            model = Ridge(alpha=best).fit(X[tr], y[tr])
            fit_s = time.perf_counter() - t0
            per_split.append({"alpha": best, "test": report(model.predict(X[te]), y[te]), "fit_s": fit_s})
        summ = summarise(per_split)
        results[emb] = {"summary": summ, "per_split": per_split}
        print(f"{emb}: test SRCC {summ['test.srcc']['mean']:.4f} ± {summ['test.srcc']['std']:.4f}, "
              f"PLCC {summ['test.plcc']['mean']:.4f} ± {summ['test.plcc']['std']:.4f}", flush=True)
        log_row(out.parent / "cost.csv", method=f"linear probe ({emb})", trainable_params=X.shape[1] + 1,
                train_gpu_hours=0,
                srcc=f"{summ['test.srcc']['mean']:.4f} ± {summ['test.srcc']['std']:.4f}",
                plcc=f"{summ['test.plcc']['mean']:.4f} ± {summ['test.plcc']['std']:.4f}",
                notes=f"CPU ridge fit {summ['fit_s']['mean']:.2f}s; features precomputed")

    lora = cfg.get("lora_reference") or {}
    if lora.get("srcc") is not None:
        log_row(out.parent / "cost.csv", method="LoRA (reference)",
                **{k: lora.get(k) for k in ("srcc", "plcc", "trainable_params", "train_gpu_hours")})

    (out / "result.json").write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    run(yaml.safe_load(open(ap.parse_args().config)))
