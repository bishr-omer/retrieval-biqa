"""E4: what do the retrieved neighbours actually match: the scene, or the distortion?

All on KADID-10k (81 clean scenes x 25 synthetic distortion types x 5 levels), cosine
retrieval on frozen embeddings. Three measurements:

  A. Held-out scenes (the E3 setting). For each KADID reference split: index = train+val
     scenes, queries = test scenes, so a neighbour can never be the same scene.
     Reports distortion-type precision@k, type-and-level precision@k, the chance level for
     each, and kNN-MOS SRCC (E1 on KADID). Also per distortion type, at k = 10.

  B. All scenes (content check). Index = every KADID image except the query itself.
     Reports how many neighbours are the SAME SCENE versus the SAME DISTORTION. If the
     embedding is organised by content, same-scene dominates.

  C. Which distortions does the KonIQ probe fail on? KonIQ probe (one per KonIQ split) scored
     on KADID; SRCC within each distortion type (81 scenes x 5 levels per type). Explains
     where E3 had a gap to recover.

Also writes neighbour lists (split 0, first embedding) for scripts/make_figures.py.

Usage: python scripts/run_e4.py --config configs/e4_kadid.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.manifest import load_manifest, load_splits  # noqa: E402
from eval.metrics import srcc  # noqa: E402
from eval.protocol import get_data  # noqa: E402
from features.load_cached import load_features, nss_stats  # noqa: E402
from models.knn_regressor import weights  # noqa: E402
from scripts.run_e1b import best_alpha  # noqa: E402

# Distortion types in KADID-10k file-name order (I01_<type>_<level>.png), from the dataset docs.
KADID_TYPES = {
    1: "gaussian blur", 2: "lens blur", 3: "motion blur", 4: "color diffusion", 5: "color shift",
    6: "color quantization", 7: "color saturation 1", 8: "color saturation 2", 9: "jpeg2000",
    10: "jpeg", 11: "white noise", 12: "white noise (color)", 13: "impulse noise",
    14: "multiplicative noise", 15: "denoise", 16: "brighten", 17: "darken", 18: "mean shift",
    19: "jitter", 20: "non-eccentricity patch", 21: "pixelate", 22: "quantization",
    23: "color block", 24: "high sharpen", 25: "contrast change",
}


# The 7 distortion families of KADID-10k. Credits perceptually equivalent matches
# (e.g. lens blur vs Gaussian blur) that exact-type precision counts as misses.
KADID_FAMILY_OF = {**{t: "blur" for t in (1, 2, 3)}, **{t: "colour" for t in (4, 5, 6, 7, 8)},
                   **{t: "compression" for t in (9, 10)}, **{t: "noise" for t in (11, 12, 13, 14, 15)},
                   **{t: "brightness" for t in (16, 17, 18)}, **{t: "spatial" for t in (19, 20, 21, 22, 23)},
                   **{t: "sharpness/contrast" for t in (24, 25)}}


def topk_cosine(Q: np.ndarray, D: np.ndarray, k: int, self_offset: int | None = None, batch: int = 1024):
    """Top-k by inner product (rows are L2-normalised), batched to bound memory.
    self_offset: if Q is D[self_offset:...], drop each query's own row."""
    idx_out = np.empty((len(Q), k), dtype=np.int64)
    sim_out = np.empty((len(Q), k), dtype=np.float32)
    for s in range(0, len(Q), batch):
        S = Q[s:s + batch] @ D.T
        if self_offset is not None:
            rows = np.arange(S.shape[0])
            S[rows, self_offset + s + rows] = -np.inf
        part = np.argpartition(-S, kth=k - 1, axis=1)[:, :k]
        ps = np.take_along_axis(S, part, 1)
        order = np.argsort(-ps, axis=1)
        idx_out[s:s + batch] = np.take_along_axis(part, order, 1)
        sim_out[s:s + batch] = np.take_along_axis(ps, order, 1)
    return sim_out, idx_out


def chance(q_labels: np.ndarray, idx_labels: np.ndarray) -> float:
    """Probability that a random index item shares the query's label."""
    vals, cnt = np.unique(idx_labels, return_counts=True)
    p = dict(zip(vals, cnt / cnt.sum()))
    return float(np.mean([p.get(v, 0.0) for v in q_labels]))


def run(cfg: dict, tdf=None, X_by_emb=None, tsplits=None, koniq=None) -> dict:
    seed = cfg["seed"]
    if tdf is None:
        tdf = load_manifest(cfg["project_root"], cfg["target"])
        tsplits = load_splits(cfg["project_root"], cfg["target"], tdf, seed=seed)
        X_by_emb = {}
        for emb in cfg["embeddings"]:
            ref = nss_stats(cfg["feat_root"], cfg["source_dataset"]) if "nss" in emb else None
            X_by_emb[emb] = load_features(cfg["feat_root"], cfg["target"], emb, tdf["name"].tolist(), nss_ref=ref)
    y = tdf["mos"].values
    dtype, dlev = tdf["dist_type"].values, tdf["dist_level"].values
    tl = dtype * 10 + dlev
    fam = np.asarray([KADID_FAMILY_OF.get(int(t), "other") for t in dtype])
    refs = np.asarray(tdf["ref"].astype(str).tolist())
    ks = cfg["k"]; kmax = max(ks)
    out = Path(cfg["out_dir"]); out.mkdir(parents=True, exist_ok=True)

    rows_a, rows_type, rows_b = [], [], []
    for emb, X in X_by_emb.items():
        # ---- A: held-out scenes ----
        for si, sp in enumerate(tsplits):
            pool = np.concatenate([sp["train"], sp["val"]]); te = np.asarray(sp["test"])
            assert not set(refs[pool]) & set(refs[te]), "scene leak between index and queries"
            sims, nb = topk_cosine(X[te], X[pool], kmax)
            nb_g = pool[nb]
            same_t = dtype[nb_g] == dtype[te][:, None]
            same_tl = tl[nb_g] == tl[te][:, None]
            same_f = fam[nb_g] == fam[te][:, None]
            ch_t, ch_tl = chance(dtype[te], dtype[pool]), chance(tl[te], tl[pool])
            ch_f = chance(fam[te], fam[pool])
            for k in ks:
                pred = (weights(sims[:, :k], cfg["tau"]) * y[nb_g[:, :k]]).sum(1)
                rows_a.append({"embedding": emb, "split": si, "k": k,
                               "type_precision": float(same_t[:, :k].mean()), "type_chance": ch_t,
                               "type_level_precision": float(same_tl[:, :k].mean()), "type_level_chance": ch_tl,
                               "family_precision": float(same_f[:, :k].mean()), "family_chance": ch_f,
                               "knn_mos_srcc": srcc(pred, y[te])})
            k10 = min(10, kmax)
            for t in np.unique(dtype[te]):
                m = dtype[te] == t
                rows_type.append({"embedding": emb, "split": si, "dist_type": int(t),
                                  "name": KADID_TYPES.get(int(t), str(t)),
                                  "family": KADID_FAMILY_OF.get(int(t), "other"),
                                  "type_precision_at10": float(same_t[m, :k10].mean()),
                                  "family_precision_at10": float(same_f[m, :k10].mean())})
            if si == 0 and emb == cfg["embeddings"][0]:
                rng = np.random.default_rng(seed)
                pick = rng.choice(len(te), size=min(cfg.get("n_figure_queries", 12), len(te)), replace=False)
                kf = cfg.get("figure_k", 5)
                w5 = weights(sims[:, :kf], cfg["tau"])
                knn_raw = (w5 * tdf["mos_raw"].values[nb_g[:, :kf]]).sum(1)
                nbr = [{"query": tdf["name"].values[te[i]], "query_mos": float(tdf["mos_raw"].values[te[i]]),
                        "query_type": KADID_TYPES.get(int(dtype[te[i]])), "query_level": int(dlev[te[i]]),
                        "query_family": KADID_FAMILY_OF.get(int(dtype[te[i]])),
                        "knn_pred": float(knn_raw[i]), "knn_k": kf,
                        "neighbours": [{"name": tdf["name"].values[j], "mos": float(tdf["mos_raw"].values[j]),
                                        "type": KADID_TYPES.get(int(dtype[j])), "level": int(dlev[j]),
                                        "family": KADID_FAMILY_OF.get(int(dtype[j])),
                                        "sim": float(s)} for j, s in zip(nb_g[i, :5], sims[i, :5])]}
                       for i in pick]
                (out / "neighbours_heldout_scenes.json").write_text(json.dumps({"embedding": emb, "items": nbr}, indent=2))

        # ---- B: all scenes, excluding self ----
        sims, nb = topk_cosine(X, X, kmax, self_offset=0)
        for k in ks:
            rows_b.append({"embedding": emb, "k": k,
                           "same_scene": float((refs[nb[:, :k]] == refs[:, None]).mean()),
                           "same_scene_chance": chance(refs, refs) - 1 / len(refs),
                           "same_type": float((dtype[nb[:, :k]] == dtype[:, None]).mean()),
                           "same_type_chance": chance(dtype, dtype),
                           "same_family": float((fam[nb[:, :k]] == fam[:, None]).mean()),
                           "same_family_chance": chance(fam, fam),
                           "same_scene_and_type": float(((refs[nb[:, :k]] == refs[:, None]) &
                                                         (dtype[nb[:, :k]] == dtype[:, None])).mean())})

    a = pd.DataFrame(rows_a); b = pd.DataFrame(rows_b); pt = pd.DataFrame(rows_type)
    a_s = a.groupby(["embedding", "k"], sort=False).agg(
        type_precision=("type_precision", "mean"), type_precision_std=("type_precision", "std"),
        type_chance=("type_chance", "mean"), type_level_precision=("type_level_precision", "mean"),
        type_level_chance=("type_level_chance", "mean"), family_precision=("family_precision", "mean"),
        family_chance=("family_chance", "mean"), knn_mos_srcc=("knn_mos_srcc", "mean")).reset_index()
    pt_s = pt.groupby(["embedding", "dist_type", "name", "family"], sort=False)[
        ["type_precision_at10", "family_precision_at10"]].mean().reset_index()
    a.to_csv(out / "heldout_per_split.csv", index=False); a_s.to_csv(out / "heldout_summary.csv", index=False)
    b.to_csv(out / "all_scenes.csv", index=False); pt_s.to_csv(out / "per_type_precision.csv", index=False)

    # ---- C: KonIQ probe, SRCC within each distortion type ----
    c_s = None
    if koniq is not None or cfg.get("probe_check", True) and "source_dataset" in cfg:
        if koniq is None:
            sdf, ssplits = get_data({**cfg, "dataset": cfg["source_dataset"]})
            emb = cfg["probe_embedding"]
            ref = nss_stats(cfg["feat_root"], cfg["source_dataset"]) if "nss" in emb else None
            Xs = load_features(cfg["feat_root"], cfg["source_dataset"], emb, sdf["name"].tolist(), nss_ref=ref)
            Xt = X_by_emb[emb] if emb in X_by_emb else load_features(
                cfg["feat_root"], cfg["target"], emb, tdf["name"].tolist(), nss_ref=ref)
        else:
            sdf, ssplits, Xs, Xt, emb = koniq
        ys = sdf["mos"].values
        rows_c = []
        for si, sp in enumerate(ssplits):
            alpha = best_alpha(Xs, ys, sp["train"], sp["val"], cfg["alphas"])
            p = Ridge(alpha=alpha).fit(Xs[sp["train"]], ys[sp["train"]]).predict(Xt)
            rows_c.append({"dist_type": 0, "name": "ALL (overall)", "srcc": srcc(p, y), "split": si})
            for t in np.unique(dtype):
                m = dtype == t
                rows_c.append({"dist_type": int(t), "name": KADID_TYPES.get(int(t), str(t)),
                               "srcc": srcc(p[m], y[m]), "split": si})
        c = pd.DataFrame(rows_c)
        c_s = c.groupby(["dist_type", "name"]).srcc.agg(["mean", "std"]).reset_index().sort_values("mean")
        c_s.insert(0, "embedding", emb)
        c_s.to_csv(out / "koniq_probe_within_type_srcc.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n== A. Held-out scenes: do neighbours share the query's distortion? (mean over KADID splits) ==")
    print(a_s.round(4).to_string(index=False))
    print("\n== B. All scenes (self excluded): same scene vs same distortion among the top-k ==")
    print(b.round(4).to_string(index=False))
    for emb in pt_s.embedding.unique():
        print(f"\n== A per distortion type, {emb}: type precision@10 (chance ~0.04) ==")
        print(pt_s[pt_s.embedding == emb].sort_values("type_precision_at10")
              [["dist_type", "name", "family", "type_precision_at10", "family_precision_at10"]].round(3).to_string(index=False))
    if c_s is not None:
        print(f"\n== C. KonIQ probe ({c_s.embedding.iloc[0]}) on KADID: SRCC within each distortion type ==")
        print(c_s[["dist_type", "name", "mean", "std"]].round(3).to_string(index=False))
    return {"heldout": a_s, "all_scenes": b, "per_type": pt_s, "probe_within_type": c_s}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    run(yaml.safe_load(open(ap.parse_args().config)))
