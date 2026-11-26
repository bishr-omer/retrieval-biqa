"""Figures for E4: each row is a query image (left, outlined) and its top-5 retrieved neighbours.

Labels show MOS, distortion type and level, and cosine similarity. Neighbours come from
held-out scenes (results/e4/neighbours_heldout_scenes.json), so none is the query's own scene:
any match you see is a distortion match, not a scene match.

Usage:
  python scripts/make_figures.py --neighbours results/e4/neighbours_heldout_scenes.json \
      --img_root ../datasets/kadid10k --out notebooks/figures --rows 4
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402


def index_images(root: str | Path, names: set[str]) -> dict[str, str]:
    found = {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f in names and f not in found:
                found[f] = os.path.join(dirpath, f)
        if len(found) == len(names):
            break
    return found


def draw(items, paths, out_file, title):
    n = len(items)
    fig, axes = plt.subplots(n, 6, figsize=(15, 2.9 * n), squeeze=False)
    for r, it in enumerate(items):
        cells = [("QUERY", it["query"], it["query_mos"], it["query_type"], it["query_level"], None)] + [
            (f"#{i + 1}", nb["name"], nb["mos"], nb["type"], nb["level"], nb["sim"]) for i, nb in enumerate(it["neighbours"])]
        for c, (tag, name, mos, typ, lev, sim) in enumerate(cells):
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([])
            if name in paths:
                img = Image.open(paths[name]).convert("RGB")
                img.thumbnail((384, 384))
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "image not found", ha="center", va="center", fontsize=8)
            match = "" if c == 0 else (" ✓" if typ == it["query_type"] else "")
            label = f"{tag}  MOS {mos:.2f}\n{typ} L{lev}{match}"
            if sim is not None:
                label += f"\ncos {sim:.3f}"
            ax.set_title(label, fontsize=8)
            for s in ax.spines.values():
                s.set_linewidth(3 if c == 0 else 0.5)
                s.set_edgecolor("black" if c == 0 else "0.7")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_file, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neighbours", default="results/e4/neighbours_heldout_scenes.json")
    ap.add_argument("--img_root", required=True)
    ap.add_argument("--out", default="notebooks/figures")
    ap.add_argument("--rows", type=int, default=4)
    a = ap.parse_args()

    data = json.loads(Path(a.neighbours).read_text())
    items = data["items"]
    names = {it["query"] for it in items} | {nb["name"] for it in items for nb in it["neighbours"]}
    paths = index_images(a.img_root, names)
    print(f"found {len(paths)} of {len(names)} images under {a.img_root}")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(items), a.rows):
        f = out / f"e4_kadid_neighbours_{i // a.rows + 1}.png"
        draw(items[i:i + a.rows], paths, f,
             f"KADID-10k, held-out scenes, {data['embedding']}: query and top-5 neighbours (✓ = same distortion type)")
        print("wrote", f)


if __name__ == "__main__":
    main()
