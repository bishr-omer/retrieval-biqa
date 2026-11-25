"""E3: cross-dataset test and database update (the key RAG test).

Source KonIQ-10k, target configurable (LIVE-itW, KADID-10k, ...). For each KonIQ split (probe
fitted on train, alpha on KonIQ val, out-of-fold train residuals in the index) and for each
support size N: draw N labelled target images (10 seeded draws; the same draws for every
source split and every method) and test on held-out target images.

Two target protocols (cfg target_protocol):
  random  support = N random target images, test = all the others. Fine when every image is
          its own scene (LIVE-itW).
  scene   uses the target's reference split file. Draw d uses target split d % n_splits:
          support from its train+val scenes, test = its test scenes. Needed for KADID, where
          each scene appears with 125 distortions: otherwise retrieval just looks up the same
          scene with a known score. Checked: no scene is in both support and test.

Conditions, all on the identical test images of each draw:
  zero-shot                       KonIQ probe, nothing from the target
  residual kNN [source]           E1b correction, KonIQ-only index (does in-domain correction transfer?)
  residual kNN [mixed]            index = KonIQ residuals + the N target residuals.  No training.
  residual kNN [target]           index = the N target residuals only.               No training.
  ridge refit                     ridge retrained on KonIQ train + the N target images
  ridge refit (upweighted)        same, target images carry `refit_target_weight` of total weight
  ridge refit + residual kNN      the refit, plus a residual correction whose target residuals are
                                  out-of-fold for the refit model (cross-fitted within the support).
                                  If gains add up, retrieval brings something refitting does not.
Few-shot LoRA on the same supports is run separately: support lists go to <out_dir>/supports/.

Retrieval settings (k, tau, lam) are FIXED in the config before looking at target results,
because there is no target validation set. All listed settings are reported; none is picked on
test. An affine recalibration cannot change SRCC, so the main metric is paired delta-SRCC
against zero-shot on the same test images.

Refits use exact ridge from sufficient statistics (models/fast_ridge.py, identical to sklearn
Ridge to ~1e-14): the KonIQ part of the normal equations is computed once per source split.

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
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.manifest import load_manifest, load_splits  # noqa: E402
from eval.cost import log_row  # noqa: E402
from eval.metrics import report  # noqa: E402
from eval.protocol import get_data  # noqa: E402
from features.load_cached import load_features, nss_stats  # noqa: E402
from index.faiss_index import Index  # noqa: E402
from models.fast_ridge import SuffStats  # noqa: E402
from scripts.run_e1b import best_alpha, knn_from, oof_residuals  # noqa: E402


def support_draws(n_target: int, Ns, n_draws: int, seed: int, tsplits=None, refs=None):
    """Return {(N, d): (support_idx, test_idx)}. Same draws for every source split and method."""
    out = {}
    for N in Ns:
        for d in range(n_draws):
            rng = np.random.default_rng([seed, N, d])
            if tsplits is None:
                S = np.sort(rng.choice(n_target, N, replace=False))
                te = np.setdiff1d(np.arange(n_target), S)
            else:
                sp = tsplits[d % len(tsplits)]
                pool = np.concatenate([sp["train"], sp["val"]])
                S = np.sort(rng.choice(pool, N, replace=False))
                te = np.asarray(sp["test"])
                if refs is not None:
                    shared = set(refs[S]) & set(refs[te])
                    assert not shared, f"{len(shared)} scenes in both support and test"
            out[(N, d)] = (S, te)
    return out


def residual_pred(p, Xq, index_feats, index_res, k, tau, lam):
    k = min(k, len(index_res))
    sims, rs, _ = Index(index_feats, index_res).search(Xq, k)
    return p + lam * knn_from(sims, rs, k, tau, False)


def load_inputs(cfg):
    sdf, ssplits = get_data({**cfg, "dataset": cfg["source_dataset"]})
    tdf = load_manifest(cfg["project_root"], cfg["target"])
    X_by_emb = {}
    for emb in cfg["embeddings"]:
        ref = nss_stats(cfg["feat_root"], cfg["source_dataset"]) if "nss" in emb else None
        X_by_emb[emb] = (load_features(cfg["feat_root"], cfg["source_dataset"], emb, sdf["name"].tolist(), nss_ref=ref),
                         load_features(cfg["feat_root"], cfg["target"], emb, tdf["name"].tolist(), nss_ref=ref))
    return sdf, ssplits, tdf, X_by_emb


def run(cfg: dict, src=None, tgt=None) -> dict:
    """src/tgt: optional (df, splits, X) and (df, X[, target_splits]) for tests; else loaded from cfg."""
    seed = cfg["seed"]
    protocol = cfg.get("target_protocol", "random")
    if src is None:
        sdf, ssplits, tdf, X_by_emb = load_inputs(cfg)
        tsplits = load_splits(cfg["project_root"], cfg["target"], tdf, seed=seed) if protocol == "scene" else None
    else:
        sdf, ssplits, Xs = src
        tdf, Xt = tgt[0], tgt[1]
        tsplits = tgt[2] if len(tgt) > 2 else None
        X_by_emb = {cfg["embeddings"][0]: (Xs, Xt)}
    ssplits = ssplits[: cfg.get("source_splits", len(ssplits))]
    ys, yt = sdf["mos"].values, tdf["mos"].values
    refs = tdf["ref"].values if "ref" in tdf else None

    out = Path(cfg["out_dir"]); (out / "supports").mkdir(parents=True, exist_ok=True)
    draws = support_draws(len(tdf), cfg["N"], cfg["draws"], seed,
                          tsplits if protocol == "scene" else None, refs)
    for (N, d), (S, _) in draws.items():
        (out / "supports" / f"N{N}_d{d}.txt").write_text("\n".join(tdf["name"].values[S]))

    combo = cfg.get("combined", {"k": 100, "tau": 0.2, "lam": 1.0})
    w_frac = cfg["refit_target_weight"]
    rows, timing = [], {}
    t_start = time.perf_counter()
    for emb, (Xs, Xt) in X_by_emb.items():
        for si, sp in enumerate(ssplits):
            tr, va = sp["train"], sp["val"]
            alpha = best_alpha(Xs, ys, tr, va, cfg["alphas"])
            probe = Ridge(alpha=alpha).fit(Xs[tr], ys[tr])
            res_tr = oof_residuals(Xs, ys, tr, alpha, seed + si)
            base_stats = SuffStats(Xs[tr], ys[tr])
            p_t = probe.predict(Xt)
            r_t = yt - p_t
            for (N, d), (S, te) in draws.items():
                base = report(p_t[te], yt[te])
                preds = {"zero-shot": p_t[te]}
                for r in cfg["retrieval"]:
                    name = f"residual kNN [{r['index']}] k={r['k']}"
                    t0 = time.perf_counter()
                    if r["index"] == "source":
                        F, R = Xs[tr], res_tr
                    elif r["index"] == "mixed":
                        F, R = np.vstack([Xs[tr], Xt[S]]), np.concatenate([res_tr, r_t[S]])
                    else:
                        F, R = Xt[S], r_t[S]
                    preds[name] = residual_pred(p_t[te], Xt[te], F, R, r["k"], r["tau"], r["lam"])
                    timing.setdefault(name, []).append(1000 * (time.perf_counter() - t0) / len(te))

                t0 = time.perf_counter()
                refit = (base_stats + SuffStats(Xt[S], yt[S])).solve(alpha)
                timing.setdefault("ridge refit", []).append(time.perf_counter() - t0)
                p_refit = refit.predict(Xt[te])
                preds["ridge refit"] = p_refit
                wt = w_frac * len(tr) / ((1 - w_frac) * N)
                preds["ridge refit (upweighted)"] = (base_stats + SuffStats(Xt[S], yt[S], np.full(N, wt))).solve(alpha).predict(Xt[te])

                # Refit + residual kNN: target residuals out-of-fold for the refit model.
                r_S = np.empty(N)
                for a, b in KFold(min(5, N), shuffle=True, random_state=seed + d).split(S):
                    m = (base_stats + SuffStats(Xt[S[a]], yt[S[a]])).solve(alpha)
                    r_S[b] = yt[S[b]] - m.predict(Xt[S[b]])
                preds[f"ridge refit + residual kNN [mixed] k={combo['k']}"] = residual_pred(
                    p_refit, Xt[te], np.vstack([Xs[tr], Xt[S]]), np.concatenate([res_tr, r_S]),
                    combo["k"], combo["tau"], combo["lam"])

                for cond, pr in preds.items():
                    m = base if cond == "zero-shot" else report(pr, yt[te])
                    rows.append({"embedding": emb, "source_split": si, "N": N, "draw": d, "condition": cond,
                                 "srcc": m["srcc"], "plcc": m["plcc"], "delta_srcc": m["srcc"] - base["srcc"]})
            print(f"[{emb}] source split {si} done ({time.perf_counter() - t_start:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "per_run.csv", index=False)
    summ = (df.groupby(["embedding", "condition", "N"], sort=False)
              .agg(srcc_mean=("srcc", "mean"), srcc_std=("srcc", "std"), plcc_mean=("plcc", "mean"),
                   delta_mean=("delta_srcc", "mean"), delta_std=("delta_srcc", "std"),
                   frac_improved=("delta_srcc", lambda x: float((x > 0).mean())), runs=("srcc", "size"))
              .reset_index())
    summ.to_csv(out / "summary.csv", index=False)

    # In-domain ceiling on the target, with the target's own splits (reference only).
    ceiling = None
    if src is None and cfg.get("target_ceiling", True):
        ts = tsplits or load_splits(cfg["project_root"], cfg["target"], tdf, seed=seed)
        ceiling = {}
        for emb in cfg["embeddings"]:
            Xt = X_by_emb[emb][1]
            vals = [report(Ridge(alpha=best_alpha(Xt, yt, s["train"], s["val"], cfg["alphas"]))
                           .fit(Xt[s["train"]], yt[s["train"]]).predict(Xt[s["test"]]), yt[s["test"]])["srcc"]
                    for s in ts]
            ceiling[emb] = {"srcc_mean": float(np.mean(vals)), "srcc_std": float(np.std(vals))}
        (out / "ceiling.json").write_text(json.dumps(ceiling, indent=2))

    tag = f"E3 {cfg.get('source_dataset', 'src')}->{cfg.get('target', 'tgt')}"
    for name, t in timing.items():
        log_row(out.parent / "cost.csv", method=f"{tag} {name}", trainable_params=0 if "kNN" in name else "",
                train_gpu_hours=0, latency_ms_per_img=round(float(np.mean(t)), 4) if "kNN" in name else "",
                notes="no training; database update only" if "kNN" in name else f"CPU refit {np.mean(t):.3f}s")

    pd.set_option("display.width", 200)
    for emb in summ.embedding.unique():
        sub = summ[summ.embedding == emb]
        order = list(dict.fromkeys(sub.condition))
        for col, title, nd in (("srcc_mean", "test SRCC", 4), ("delta_mean", "paired delta SRCC vs zero-shot", 4),
                               ("frac_improved", "fraction of runs improved", 2)):
            print(f"\n[{emb}] {title}")
            print(sub.pivot(index="condition", columns="N", values=col).loc[order].round(nd).to_string())
    if ceiling:
        for emb, c in ceiling.items():
            print(f"\nIn-domain {cfg['target']} probe, {emb} (ceiling reference): {c['srcc_mean']:.4f} ± {c['srcc_std']:.4f}")
    return {"summary": summ, "ceiling": ceiling}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    run(yaml.safe_load(open(ap.parse_args().config)))
