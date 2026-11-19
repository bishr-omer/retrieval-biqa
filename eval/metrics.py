"""SRCC, PLCC (after the standard 4-parameter logistic mapping), and evidence metrics."""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr


def srcc(pred, gt) -> float:
    return float(spearmanr(pred, gt).correlation)


def _logistic4(x, b1, b2, b3, b4):
    z = np.clip(-(x - b3) / (abs(b4) + 1e-8), -50, 50)
    return (b1 - b2) / (1 + np.exp(z)) + b2


def plcc(pred, gt, fit: bool = True) -> float:
    pred, gt = np.asarray(pred, float), np.asarray(gt, float)
    if fit:
        try:
            p0 = [gt.max(), gt.min(), np.median(pred), pred.std() + 1e-6]
            popt, _ = curve_fit(_logistic4, pred, gt, p0=p0, maxfev=20000)
            pred = _logistic4(pred, *popt)
        except RuntimeError:
            pass  # fall back to raw PLCC
    return float(pearsonr(pred, gt)[0])


def report(pred, gt) -> dict:
    return {"srcc": srcc(pred, gt), "plcc": plcc(pred, gt)}


def distortion_precision_at_k(query_types: np.ndarray, neighbour_types: np.ndarray) -> float:
    """Fraction of retrieved neighbours whose KADID distortion type matches the query's.
    Random baseline with 25 balanced types = 0.04."""
    return float((neighbour_types == query_types[:, None]).mean())
