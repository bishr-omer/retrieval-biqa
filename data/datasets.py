"""Dataset metadata loaders: image names, MOS normalised to [0, 1], and extra labels.

Each loader returns a pandas DataFrame with at least:
    name      image file name (the key used to look up cached features)
    mos       quality score normalised to [0, 1], higher = better
    mos_raw   original score
KADID also returns: ref (reference image name), dist_type (1..25), dist_level (1..5).
KonIQ also returns: set ('training' | 'validation' | 'test') when the official file has it.

Normalisation matters for E3: KonIQ MOS is 1-5, LIVEC MOS is 0-100, KADID DMOS is 1-5.
Mixing them in one index without normalising would make retrieved scores meaningless.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Fixed score ranges per dataset (theoretical scale, not data min/max, so that
# normalisation does not depend on which split you load).
SCALES = {
    "koniq": (1.0, 5.0),
    "livec": (0.0, 100.0),
    "kadid": (1.0, 5.0),
}


def _norm(x: np.ndarray, dataset: str) -> np.ndarray:
    lo, hi = SCALES[dataset]
    return (np.asarray(x, dtype=np.float64) - lo) / (hi - lo)


def load_koniq(root: str | Path) -> pd.DataFrame:
    """KonIQ-10k. Expects koniq10k_distributions_sets.csv (has the official 'set' column)
    or koniq10k_scores_and_distributions.csv in `root`."""
    root = Path(root)
    for fname in ("koniq10k_distributions_sets.csv", "koniq10k_scores_and_distributions.csv"):
        f = root / fname
        if f.exists():
            df = pd.read_csv(f)
            break
    else:
        raise FileNotFoundError(f"No KonIQ csv found in {root}")
    mos_col = "MOS" if "MOS" in df.columns else "mos"
    out = pd.DataFrame({"name": df["image_name"], "mos_raw": df[mos_col]})
    out["mos"] = _norm(out["mos_raw"], "koniq")
    if "set" in df.columns:
        out["set"] = df["set"].str.lower()
    return out.reset_index(drop=True)


def load_livec(root: str | Path) -> pd.DataFrame:
    """LIVE In the Wild Challenge. Expects Data/AllImages_release.mat and Data/AllMOS_release.mat.
    The first 7 images are the training images used in the original study; standard practice
    is to drop them, which we do."""
    from scipy.io import loadmat

    root = Path(root)
    data = root / "Data" if (root / "Data").exists() else root
    names = loadmat(data / "AllImages_release.mat")["AllImages_release"]
    mos = loadmat(data / "AllMOS_release.mat")["AllMOS_release"].ravel()
    names = [str(n[0][0]) for n in names]
    out = pd.DataFrame({"name": names, "mos_raw": mos}).iloc[7:]
    out["mos"] = _norm(out["mos_raw"], "livec")
    return out.reset_index(drop=True)


def load_kadid(root: str | Path) -> pd.DataFrame:
    """KADID-10k. Expects dmos.csv with columns dist_img, ref_img, dmos.
    File names look like I01_05_03.png = reference 01, distortion type 05, level 3."""
    df = pd.read_csv(Path(root) / "dmos.csv")
    out = pd.DataFrame({"name": df["dist_img"], "ref": df["ref_img"], "mos_raw": df["dmos"]})
    parts = out["name"].str.replace(".png", "", regex=False).str.split("_", expand=True)
    out["dist_type"] = parts[1].astype(int)
    out["dist_level"] = parts[2].astype(int)
    out["mos"] = _norm(out["mos_raw"], "kadid")
    return out.reset_index(drop=True)


LOADERS = {"koniq": load_koniq, "livec": load_livec, "kadid": load_kadid}


def load(dataset: str, root: str | Path) -> pd.DataFrame:
    return LOADERS[dataset](root)
