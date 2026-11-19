"""Tests against a mock of the biqa-revision layout: manifests/, splits/, cache/*.npz."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.manifest import load_manifest, load_splits  # noqa: E402
from eval.protocol import get_data  # noqa: E402
from scripts import run_e0, run_e1  # noqa: E402


def make_project(tmp, split_format, n=600, n_splits=5):
    rng = np.random.default_rng(0)
    names = [f"{1000 + i}.jpg" for i in range(n)]
    X = rng.normal(size=(n, 48)).astype(np.float32)
    mos = 1 + 4 / (1 + np.exp(-X[:, 0])) + 0.1 * rng.normal(size=n)  # 1..5 like KonIQ
    for d in ("manifests", "splits", "cache"):
        (tmp / d).mkdir()
    pd.DataFrame({"image": names, "ref": names, "mos": mos}).to_csv(tmp / "manifests/koniq10k.csv", index=False)
    order = rng.permutation(n)  # cache stored in a different order than the manifest
    for bb, dim in (("siglip", 48), ("cliph", 32), ("nss", 10)):
        F = X[:, :dim] if bb != "nss" else X[:, :10] * 50 + 3
        np.savez(tmp / f"cache/koniq10k_{bb}.npz", names=np.array(names)[order], feats=F[order])
    splits = []
    for s in range(n_splits):
        p = np.random.default_rng(s).permutation(n)
        tr, va, te = p[:420], p[420:480], p[480:]
        if split_format == "dict_names":
            splits.append({"train": [names[i] for i in tr], "test": [names[i] for i in te]})
        elif split_format == "dict_idx_val":
            splits.append({"train": tr.tolist(), "val": va.tolist(), "test": te.tolist()})
        elif split_format == "list_pair":
            splits.append([[names[i] for i in tr], [names[i] for i in te]])
    (tmp / "splits/koniq10k_reference.json").write_text(json.dumps(splits))
    return tmp


@pytest.mark.parametrize("fmt", ["dict_names", "dict_idx_val", "list_pair"])
def test_split_formats(tmp_path, fmt):
    root = make_project(tmp_path, fmt)
    df = load_manifest(root, "koniq10k")
    assert df.mos.between(-0.1, 1.1).all()
    sp = load_splits(root, "koniq10k", df)
    assert len(sp) == 5
    for s in sp:
        assert len(s["test"]) == 120
        assert not set(s["train"]) & set(s["val"]) and not set(s["train"]) & set(s["test"])
        assert s["val_carved_from_train"] == (fmt != "dict_idx_val")


def test_kadid_manifest_parses_distortions(tmp_path):
    (tmp_path / "manifests").mkdir()
    pd.DataFrame({"image": ["I01_01_01.png", "I02_24_5.png"], "ref": ["I01.png", "I02.png"],
                  "mos": [4.57, 1.2]}).to_csv(tmp_path / "manifests/kadid10k.csv", index=False)
    df = load_manifest(tmp_path, "kadid10k")
    assert df.dist_type.tolist() == [1, 24] and df.dist_level.tolist() == [1, 5]


def test_ref_leak_is_caught(tmp_path):
    root = make_project(tmp_path, "dict_idx_val", n_splits=1)
    m = pd.read_csv(root / "manifests/koniq10k.csv")
    m["ref"] = "same_scene.jpg"  # every image is the same scene -> train/test share it
    m.to_csv(root / "manifests/koniq10k.csv", index=False)
    with pytest.raises(AssertionError, match="reference scenes"):
        get_data({"source": "manifest", "project_root": root, "dataset": "koniq10k", "seed": 0})


def test_e0_e1_on_mock_project(tmp_path):
    root = make_project(tmp_path, "dict_names")
    base = {"source": "manifest", "project_root": str(root), "feat_root": str(root / "cache"),
            "dataset": "koniq10k", "seed": 0}
    r1 = run_e1.run({**base, "embeddings": ["siglip", "concat:siglip+cliph+nss"], "k": [5, 10],
                     "tau": [0.05, 1.0], "out_dir": str(tmp_path / "res/e1")})
    assert r1["n_splits"] == 5
    assert r1["summary"]["test.srcc"]["mean"] > 0.4
    assert abs(r1["summary"]["test_shuffled_scores.srcc"]["mean"]) < 0.2
    r0 = run_e0.run({**base, "embeddings": ["siglip"], "alphas": [0.1, 1.0], "out_dir": str(tmp_path / "res/e0")})
    assert r0["siglip"]["summary"]["test.srcc"]["mean"] > 0.4
    cost = pd.read_csv(tmp_path / "res/cost.csv")
    assert {"kNN (E1, tuned per split)", "linear probe (siglip)"} <= set(cost.method)


def test_check_setup_script(tmp_path):
    root = make_project(tmp_path, "dict_names")
    cfg = {"source": "manifest", "project_root": str(root), "feat_root": str(root / "cache"),
           "dataset": "koniq10k", "seed": 0, "embeddings": ["siglip", "nss"]}
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg))
    out = subprocess.run([sys.executable, str(ROOT / "scripts/check_setup.py"), "--config", str(tmp_path / "c.yaml")],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "OK: everything lines up." in out.stdout
