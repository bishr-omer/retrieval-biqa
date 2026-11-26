"""E4 checks on a synthetic KADID-like set (scenes x distortion types x levels)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.splits import group_split, random_split  # noqa: E402
from features.load_cached import l2n  # noqa: E402
from scripts import run_e4  # noqa: E402

N_REF, N_TYPE, N_LEV = 20, 25, 5


def kadid_like(seed=0, d=32):
    rng = np.random.default_rng(seed)
    r, t, l = np.meshgrid(np.arange(N_REF), np.arange(1, N_TYPE + 1), np.arange(1, N_LEV + 1), indexing="ij")
    r, t, l = r.ravel(), t.ravel(), l.ravel()
    scene, dist = rng.normal(size=(N_REF, d)), rng.normal(size=(N_TYPE + 1, d))
    noise = rng.normal(size=(len(r), d))
    X_content = l2n(3 * scene[r] + 0.5 * dist[t] * l[:, None] / 5 + 0.3 * noise)
    X_distort = l2n(0.3 * scene[r] + 3 * dist[t] + 0.3 * noise)
    names = [f"I{a + 1:02d}_{b:02d}_{c:02d}.png" for a, b, c in zip(r, t, l)]
    mos = 5 - 0.6 * l - 0.05 * t + 0.05 * rng.normal(size=len(r))
    tdf = pd.DataFrame({"name": names, "ref": [f"I{a + 1:02d}.png" for a in r], "mos_raw": mos,
                        "mos": (mos - 1) / 4, "dist_type": t, "dist_level": l})
    return tdf, {"content": X_content, "distortion": X_distort}, [group_split(r, seed=s) for s in range(2)]


CFG = {"seed": 0, "embeddings": ["content", "distortion"], "k": [1, 5, 10], "tau": 0.1,
       "alphas": [0.1, 1.0], "probe_check": False, "n_figure_queries": 4}


def test_content_vs_distortion_embeddings(tmp_path):
    tdf, X, ts = kadid_like()
    r = run_e4.run({**CFG, "out_dir": str(tmp_path / "e4")}, tdf=tdf, X_by_emb=X, tsplits=ts)
    a = r["heldout"].set_index(["embedding", "k"]); b = r["all_scenes"].set_index(["embedding", "k"])
    assert a.loc[("distortion", 10), "type_precision"] > 0.8
    assert a.loc[("content", 10), "type_precision"] < a.loc[("distortion", 10), "type_precision"]
    assert abs(a.loc[("content", 10), "type_chance"] - 1 / N_TYPE) < 0.01
    assert b.loc[("content", 10), "same_scene"] > 0.9            # content embedding: neighbours = same scene
    assert b.loc[("distortion", 10), "same_type"] > 0.8
    nb = json.loads((tmp_path / "e4/neighbours_heldout_scenes.json").read_text())
    q_ref = {it["query"][:3] for it in nb["items"]}
    assert all(n["name"][:3] not in {it["query"][:3]} for it in nb["items"] for n in it["neighbours"])
    assert len(nb["items"]) == 4 and q_ref


def test_probe_within_type(tmp_path):
    tdf, X, ts = kadid_like()
    rng = np.random.default_rng(1)
    Xs = l2n(rng.normal(size=(600, 32)))
    sdf = pd.DataFrame({"name": [f"s{i}" for i in range(600)], "ref": [f"s{i}" for i in range(600)],
                        "mos": Xs @ rng.normal(size=32)})
    r = run_e4.run({**CFG, "embeddings": ["distortion"], "out_dir": str(tmp_path / "e4")}, tdf=tdf,
                   X_by_emb={"distortion": X["distortion"]}, tsplits=ts,
                   koniq=(sdf, [random_split(600, seed=0)], Xs, X["distortion"], "distortion"))
    c = r["probe_within_type"]
    assert len(c) == N_TYPE + 1 and "ALL (overall)" in set(c.name)


def test_make_figures(tmp_path):
    tdf, X, ts = kadid_like()
    run_e4.run({**CFG, "embeddings": ["distortion"], "out_dir": str(tmp_path / "e4")},
               tdf=tdf, X_by_emb={"distortion": X["distortion"]}, tsplits=ts)
    img_dir = tmp_path / "imgs" / "sub"; img_dir.mkdir(parents=True)
    nb = json.loads((tmp_path / "e4/neighbours_heldout_scenes.json").read_text())
    for it in nb["items"]:
        for n in [it["query"]] + [x["name"] for x in it["neighbours"]]:
            Image.new("RGB", (64, 48), (120, 80, 40)).save(img_dir / n)
    out = subprocess.run([sys.executable, str(ROOT / "scripts/make_figures.py"), "--neighbours",
                          str(tmp_path / "e4/neighbours_heldout_scenes.json"), "--img_root", str(tmp_path / "imgs"),
                          "--out", str(tmp_path / "figs"), "--rows", "2"], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert len(list((tmp_path / "figs").glob("*.png"))) == 2
