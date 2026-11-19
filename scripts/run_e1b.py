"""E1b: does retrieval add anything a linear probe does not already have?

E1 showed kNN on frozen generic embeddings (SRCC ~0.77) loses clearly to a linear probe on the
same features (~0.89). Two diagnostics, on every split, tuned on val, reported once on test:

  (a) PLS-kNN: learn a supervised low-dimensional projection (PLS, fitted on train MOS only),
      then kNN with Euclidean distance in that space. If this nears the probe, E1's problem was
      the metric (content-dominated cosine), not retrieval as such.

  (b) Residual kNN: pred = probe(q) + lam * weighted mean of the neighbours' probe residuals.
      Neighbours are found by cosine in the original embedding space. Train residuals are
      out-of-fold (5-fold cross-fitting inside train), so the index never holds residuals of a
      model that saw those images. lam = 0 is in the grid, so if retrieval adds nothing the
      tuning will choose lam = 0 and the result equals the probe. Shuffled-residual check included.

Usage: python scripts/run_e1b.py --config configs/e1b_koniq.yaml
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
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.splits import official_or_random  # noqa: E402
from eval.cost import log_row  # noqa: E402
from eval.metrics import report, srcc  # noqa: E402
from eval.protocol import feature_getter, get_data, summarise  # noqa: E402
from index.faiss_index import Index  # noqa: E402
from models.knn_regressor import weights  # noqa: E402


# ---------- helpers ----------
def topk_euclid(Q: np.ndarray, D: np.ndarray, k: int):
    """Exact Euclidean top-k (small dims, so brute force is fine). Returns (dist, idx)."""
    d2 = (Q**2).sum(1, keepdims=True) - 2 * Q @ D.T + (D**2).sum(1)[None]
    d2 = np.maximum(d2, 0)
    idx = np.argpartition(d2, kth=min(k, D.shape[0] - 1), axis=1)[:, :k]
    part = np.take_along_axis(d2, idx, axis=1)
    order = np.argsort(part, axis=1)
    return np.sqrt(np.take_along_axis(part, order, 1)), np.take_along_axis(idx, order, 1)


def knn_from(dist_or_sims, ys, k, tau, is_dist):
    s = dist_or_sims[:, :k]
    if is_dist:  # turn distances into similarities on a data-adaptive scale
        s = -s / (np.median(dist_or_sims[:, :k]) + 1e-8)
    return (weights(s, tau) * ys[:, :k]).sum(1)


def best_alpha(X, y, tr, va, alphas):
    return max(alphas, key=lambda a: srcc(Ridge(alpha=a).fit(X[tr], y[tr]).predict(X[va]), y[va]))


def oof_residuals(X, y, tr, alpha, seed, folds=5):
    res = np.empty(len(tr))
    for a, b in KFold(folds, shuffle=True, random_state=seed).split(tr):
        m = Ridge(alpha=alpha).fit(X[tr[a]], y[tr[a]])
        res[b] = y[tr[b]] - m.predict(X[tr[b]])
    return res


# ---------- (a) PLS-kNN ----------
def pls_knn(cfg, X, y, tr, va, te):
    ks, kmax = cfg["k"], max(cfg["k"])
    best = None
    for d in cfg["pls_dims"]:
        pls = PLSRegression(n_components=d, scale=False).fit(X[tr], y[tr])
        Ztr, Zva = pls.transform(X[tr]), pls.transform(X[va])
        dist, idx = topk_euclid(Zva, Ztr, kmax)
        ys = y[tr][idx]
        for k in ks:
            for tau in cfg["tau"]:
                s = srcc(knn_from(dist, ys, k, tau, True), y[va])
                if best is None or s > best[0]:
                    best = (s, d, k, tau)
    _, d, k, tau = best
    pls = PLSRegression(n_components=d, scale=False).fit(X[tr], y[tr])
    Ztr, Zte = pls.transform(X[tr]), pls.transform(X[te])
    t0 = time.perf_counter()
    dist, idx = topk_euclid(Zte, Ztr, k)
    pred = knn_from(dist, y[tr][idx], k, tau, True)
    lat = 1000 * (time.perf_counter() - t0) / len(te)
    return {"config": {"pls_dim": d, "k": k, "tau": tau}, "val_srcc": best[0],
            "test": report(pred, y[te]), "latency_ms_per_img": lat,
            "index_mb": Ztr.astype(np.float32).nbytes / 2**20}


# ---------- (b) residual kNN ----------
def residual_knn(cfg, X, y, tr, va, te, seed):
    alpha = best_alpha(X, y, tr, va, cfg["alphas"])
    probe = Ridge(alpha=alpha).fit(X[tr], y[tr])
    res_tr = oof_residuals(X, y, tr, alpha, seed)
    idx = Index(X[tr], res_tr, ids=tr)
    kmax = max(cfg["k"])

    p_va = probe.predict(X[va])
    sims, rs, _ = idx.search(X[va], kmax)
    best = None
    for k in cfg["k"]:
        for tau in cfg["tau"]:
            corr = knn_from(sims, rs, k, tau, False)
            for lam in cfg["lam"]:
                s = srcc(p_va + lam * corr, y[va])
                if best is None or s > best[0]:
                    best = (s, k, tau, lam)
    _, k, tau, lam = best

    p_te = probe.predict(X[te])
    t0 = time.perf_counter()
    sims, rs, _ = idx.search(X[te], k)
    corr = knn_from(sims, rs, k, tau, False)
    lat = 1000 * (time.perf_counter() - t0) / len(te)
    rng = np.random.default_rng(seed)
    corr_shuf = knn_from(sims, rng.choice(res_tr, size=rs.shape), k, tau, False)
    return {"config": {"alpha": alpha, "k": k, "tau": tau, "lam": lam}, "val_srcc": best[0],
            "test_probe_only": report(p_te, y[te]),
            "test": report(p_te + lam * corr, y[te]),
            "test_forced_lam1": report(p_te + corr, y[te]),
            "test_shuffled_residuals": report(p_te + lam * corr_shuf, y[te]),
            "latency_ms_per_img": lat, "index_mb": idx.feats.nbytes / 2**20}


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
        per = {"pls_knn": [], "residual_knn": []}
        for i, s in enumerate(splits):
            tr, va, te = s["train"], s["val"], s["test"]
            a = pls_knn(cfg, X, y, tr, va, te)
            b = residual_knn(cfg, X, y, tr, va, te, cfg["seed"] + i)
            per["pls_knn"].append(a); per["residual_knn"].append(b)
            print(f"[{emb}] split {i}: PLS-kNN {a['test']['srcc']:.4f} {a['config']} | probe "
                  f"{b['test_probe_only']['srcc']:.4f} -> +residual {b['test']['srcc']:.4f} "
                  f"(lam={b['config']['lam']}, k={b['config']['k']}), forced lam=1 "
                  f"{b['test_forced_lam1']['srcc']:.4f}, shuffled {b['test_shuffled_residuals']['srcc']:.4f}",
                  flush=True)
        results[emb] = {m: {"summary": summarise(v), "per_split": v} for m, v in per.items()}

        sa, sb = results[emb]["pls_knn"]["summary"], results[emb]["residual_knn"]["summary"]
        fmt = lambda s, key: f"{s[key]['mean']:.4f} ± {s[key]['std']:.4f}"  # noqa: E731
        log_row(out.parent / "cost.csv", method=f"PLS-kNN ({emb})", trainable_params="PLS projection",
                train_gpu_hours=0, latency_ms_per_img=round(sa["latency_ms_per_img"]["mean"], 4),
                index_mb=round(sa["index_mb"]["mean"], 2), srcc=fmt(sa, "test.srcc"), plcc=fmt(sa, "test.plcc"),
                notes="E1b(a); supervised projection fitted on train MOS")
        log_row(out.parent / "cost.csv", method=f"probe + residual kNN ({emb})", trainable_params=X.shape[1] + 1,
                train_gpu_hours=0, latency_ms_per_img=round(sb["latency_ms_per_img"]["mean"], 4),
                index_mb=round(sb["index_mb"]["mean"], 2), srcc=fmt(sb, "test.srcc"), plcc=fmt(sb, "test.plcc"),
                notes=(f"E1b(b); probe alone {fmt(sb, 'test_probe_only.srcc')}; lam per split "
                       f"{[r['config']['lam'] for r in results[emb]['residual_knn']['per_split']]}"))
        print(f"[{emb}] MEAN: PLS-kNN {fmt(sa, 'test.srcc')} | probe {fmt(sb, 'test_probe_only.srcc')} | "
              f"probe+residual {fmt(sb, 'test.srcc')} | shuffled {fmt(sb, 'test_shuffled_residuals.srcc')}\n",
              flush=True)

    (out / "result.json").write_text(json.dumps(results, indent=2, default=float))
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    run(yaml.safe_load(open(ap.parse_args().config)))
