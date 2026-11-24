"""E3: cross-dataset test and database update (the key RAG test).

Source KonIQ-10k, target LIVE in the Wild. For each KonIQ split (probe fitted on train,
alpha on KonIQ val, out-of-fold train residuals in the index) and for each support size N:
draw N labelled LIVE-itW images (10 seeded draws; the same draws for every source split and
every method), test on the remaining LIVE-itW images.

Conditions, all on the identical test images of each draw:
  zero-shot                  KonIQ probe, nothing from the target
  residual kNN [source]      E1b correction with a KonIQ-only index (does in-domain correction transfer?)
  residual kNN [mixed]       index = KonIQ residuals + the N target residuals.  No training.
  residual kNN [target]      index = the N target residuals only.               No training.
  ridge refit                ridge retrained on KonIQ train + the N target images
  ridge refit (upweighted)   same, target images carry `refit_target_weight` of the total weight
Few-shot LoRA on the same supports is run separately with your LoRA code: the support lists are
written to results/e3/supports/ so every method sees exactly the same images.

Retrieval settings (k, tau, lam) are FIXED in the config before looking at target results,
because there is no target validation set. All listed settings are reported; none is picked on test.
Note: an affine recalibration of the probe on the support set cannot change SRCC, so a
correction only counts if it changes the ranking. That is why the main metric is paired
delta-SRCC against zero-shot on the same test images.

Usage: python scripts/run_e3.py --config configs/e3_koniq_to_liveitw.yaml
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
from data.manifest import load_manifest, load_splits  # noqa: E402
from eval.cost import log_row  # noqa: E402
from eval.metrics import report  # noqa: E402
from eval.protocol import get_data  # noqa: E402
from features.load_cached import load_features, nss_stats  # noqa: E402
from index.faiss_index import Index  # noqa: E402
from scripts.run_e1b import best_alpha, knn_from, oof_residuals  # noqa: E402


def support_draws(n_target: int, Ns, n_draws: int, seed: int):
    """Same supports for every source split and every method (seeded by N and draw only)."""
    return {(N, d): np.sort(np.random.default_rng([seed, N, d]).choice(n_target, N, replace=False))
            for N in Ns for d in range(n_draws)}


def residual_pred(p, Xq, index_feats, index_res, k, tau, lam):
    k = min(k, len(index_res))
    sims, rs, _ = Index(index_feats, index_res).search(Xq, k)
    return p + lam * knn_from(sims, rs, k, tau, False)


def run(cfg: dict, src=None, tgt=None) -> dict:
    """src/tgt: optional (df, splits, X) and (df, X) for tests; otherwise loaded from cfg."""
    seed = cfg["seed"]
    if src is None:
        scfg = {**cfg, "dataset": cfg["source_dataset"]}
        sdf, ssplits = get_data(scfg)
        tdf = load_manifest(cfg["project_root"], cfg["target"])
        X_by_emb = {}
        for emb in cfg["embeddings"]:
            ref = nss_stats(cfg["feat_root"], cfg["source_dataset"]) if "nss" in emb else None
            X_by_emb[emb] = (load_features(cfg["feat_root"], cfg["source_dataset"], emb, sdf["name"].tolist(), nss_ref=ref),
                             load_features(cfg["feat_root"], cfg["target"], emb, tdf["name"].tolist(), nss_ref=ref))
    else:
        sdf, ssplits, Xs = src
        tdf, Xt = tgt
        X_by_emb = {cfg["embeddings"][0]: (Xs, Xt)}
    ssplits = ssplits[: cfg.get("source_splits", len(ssplits))]
    ys, yt = sdf["mos"].values, tdf["mos"].values
    nt = len(tdf)

    out = Path(cfg["out_dir"]); (out / "supports").mkdir(parents=True, exist_ok=True)
    draws = support_draws(nt, cfg["N"], cfg["draws"], seed)
    for (N, d), S in draws.items():
        (out / "supports" / f"N{N}_d{d}.txt").write_text("\n".join(tdf["name"].values[S]))

    rows, timing = [], {}
    for emb, (Xs, Xt) in X_by_emb.items():
        for si, sp in enumerate(ssplits):
            tr, va = sp["train"], sp["val"]
            alpha = best_alpha(Xs, ys, tr, va, cfg["alphas"])
            probe = Ridge(alpha=alpha).fit(Xs[tr], ys[tr])
            res_tr = oof_residuals(Xs, ys, tr, alpha, seed + si)
            p_t = probe.predict(Xt)
            r_t = yt - p_t
            for (N, d), S in draws.items():
                te = np.setdiff1d(np.arange(nt), S)
                base = report(p_t[te], yt[te])
                preds = {"zero-shot": p_t[te]}
                for r in cfg["retrieval"]:
                    name = f"residual kNN [{r['index']}] k={r['k']}"
                    t0 = time.perf_counter()
                    if r["index"] == "source":
                        F, R = Xs[tr], res_tr
                    elif r["index"] == "mixed":
                        F, R = np.vstack([Xs[tr], Xt[S]]), np.concatenate([res_tr, r_t[S]])
                    else:  # target only
                        F, R = Xt[S], r_t[S]
                    preds[name] = residual_pred(p_t[te], Xt[te], F, R, r["k"], r["tau"], r["lam"])
                    timing.setdefault(name, []).append(1000 * (time.perf_counter() - t0) / len(te))
                Xa, ya = np.vstack([Xs[tr], Xt[S]]), np.concatenate([ys[tr], yt[S]])
                t0 = time.perf_counter()
                preds["ridge refit"] = Ridge(alpha=alpha).fit(Xa, ya).predict(Xt[te])
                timing.setdefault("ridge refit", []).append(time.perf_counter() - t0)
                w = np.ones(len(ya))
                w[len(tr):] = cfg["refit_target_weight"] * len(tr) / ((1 - cfg["refit_target_weight"]) * N)
                preds["ridge refit (upweighted)"] = Ridge(alpha=alpha).fit(Xa, ya, sample_weight=w).predict(Xt[te])

                for cond, pr in preds.items():
                    m = base if cond == "zero-shot" else report(pr, yt[te])
                    rows.append({"embedding": emb, "source_split": si, "N": N, "draw": d, "condition": cond,
                                 "srcc": m["srcc"], "plcc": m["plcc"], "delta_srcc": m["srcc"] - base["srcc"]})
            print(f"[{emb}] source split {si} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "per_run.csv", index=False)
    summ = (df.groupby(["embedding", "condition", "N"], sort=False)
              .agg(srcc_mean=("srcc", "mean"), srcc_std=("srcc", "std"), plcc_mean=("plcc", "mean"),
                   delta_mean=("delta_srcc", "mean"), delta_std=("delta_srcc", "std"),
                   frac_improved=("delta_srcc", lambda x: float((x > 0).mean())), runs=("srcc", "size"))
              .reset_index())
    summ.to_csv(out / "summary.csv", index=False)

    # In-domain ceiling on the target, with your own LIVE-itW splits (reference only).
    ceiling = None
    if src is None and cfg.get("target_ceiling", True):
        tsplits = load_splits(cfg["project_root"], cfg["target"], tdf, seed=seed)
        emb = cfg["embeddings"][0]
        Xt = X_by_emb[emb][1]
        vals = [report(Ridge(alpha=best_alpha(Xt, yt, s["train"], s["val"], cfg["alphas"]))
                       .fit(Xt[s["train"]], yt[s["train"]]).predict(Xt[s["test"]]), yt[s["test"]])["srcc"]
                for s in tsplits]
        ceiling = {"embedding": emb, "srcc_mean": float(np.mean(vals)), "srcc_std": float(np.std(vals))}
        (out / "ceiling.json").write_text(json.dumps(ceiling, indent=2))

    for name, t in timing.items():
        log_row(out.parent / "cost.csv", method=f"E3 {name}", trainable_params=0 if "kNN" in name else "",
                train_gpu_hours=0, latency_ms_per_img=round(float(np.mean(t)), 4) if "kNN" in name else "",
                notes="no training; database update only" if "kNN" in name else f"CPU refit {np.mean(t):.2f}s")

    pd.set_option("display.width", 200)
    for emb in summ.embedding.unique():
        t = summ[summ.embedding == emb].pivot(index="condition", columns="N", values="srcc_mean")
        dlt = summ[summ.embedding == emb].pivot(index="condition", columns="N", values="delta_mean")
        fr = summ[summ.embedding == emb].pivot(index="condition", columns="N", values="frac_improved")
        order = list(dict.fromkeys(summ[summ.embedding == emb].condition))
        print(f"\n[{emb}] test SRCC on LIVE-itW (mean over source splits x draws)")
        print(t.loc[order].round(4).to_string())
        print(f"\n[{emb}] paired delta SRCC vs zero-shot")
        print(dlt.loc[order].round(4).to_string())
        print(f"\n[{emb}] fraction of runs improved")
        print(fr.loc[order].round(2).to_string())
    if ceiling:
        print(f"\nIn-domain LIVE-itW probe (ceiling reference, own splits): "
              f"{ceiling['srcc_mean']:.4f} ± {ceiling['srcc_std']:.4f}")
    return {"summary": summ, "ceiling": ceiling}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    run(yaml.safe_load(open(ap.parse_args().config)))
