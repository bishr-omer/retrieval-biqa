"""Run this first. Checks that manifests, splits and cached features line up, and prints
what was found. Nothing is trained.

Usage: python scripts/check_setup.py --config configs/e1_koniq.yaml
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.protocol import feature_getter, get_data  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
cfg = yaml.safe_load(open(ap.parse_args().config))

if cfg.get("source") == "manifest":
    raw = json.loads((Path(cfg["project_root"]) / "splits" / f"{cfg['dataset']}_{cfg.get('split_kind', 'reference')}.json").read_text())
    e = raw[0] if isinstance(raw, list) else raw
    print("split file: first entry is", type(e).__name__,
          "with keys", list(e)[:6] if isinstance(e, dict) else f"length {len(e)}")

df, splits = get_data(cfg)
print(f"manifest: {len(df)} images, mos_raw {df.mos_raw.min():.2f}..{df.mos_raw.max():.2f}, "
      f"normalised {df.mos.min():.3f}..{df.mos.max():.3f}")
for i, s in enumerate(splits):
    print(f"split {i}: train {len(s['train'])}  val {len(s['val'])}  test {len(s['test'])}"
          + ("  (val carved from train)" if s.get("val_carved_from_train") else ""))

get = feature_getter(cfg, df)
for emb in cfg["embeddings"]:
    X = get(emb)
    print(f"features {emb}: {X.shape}, finite={np.isfinite(X).all()}, norm≈{np.linalg.norm(X, axis=1).mean():.3f}")
print("OK: everything lines up.")
