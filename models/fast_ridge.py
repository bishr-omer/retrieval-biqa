"""Exact ridge regression from sufficient statistics, for fast refits.

Minimises  sum_i w_i (y_i - x_i.b - c)^2 + alpha * ||b||^2   (intercept c not penalised),
the same objective as sklearn.linear_model.Ridge(fit_intercept=True) with sample_weight.

E3 refits the KonIQ probe after adding N target images, 40 times per source split. Refitting
from scratch costs a full pass over ~6,000 x 2,314 features each time. Here the KonIQ part of
the normal equations is computed once; adding N images only adds their (small) contribution,
then one D x D solve.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve


class SuffStats:
    def __init__(self, X: np.ndarray | None = None, y: np.ndarray | None = None, w=None, d: int | None = None):
        if X is None:
            self.n = 0.0
            self.sx = np.zeros(d); self.sy = 0.0
            self.xx = np.zeros((d, d)); self.xy = np.zeros(d)
            return
        X = np.asarray(X, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
        w = np.ones(len(y)) if w is None else np.asarray(w, dtype=np.float64)
        Xw = X * w[:, None]
        self.n = float(w.sum())
        self.sx = Xw.sum(0); self.sy = float(w @ y)
        self.xx = X.T @ Xw; self.xy = Xw.T @ y

    def __add__(self, other: "SuffStats") -> "SuffStats":
        out = SuffStats.__new__(SuffStats)
        out.n = self.n + other.n
        out.sx = self.sx + other.sx; out.sy = self.sy + other.sy
        out.xx = self.xx + other.xx; out.xy = self.xy + other.xy
        return out

    def solve(self, alpha: float):
        mu, my = self.sx / self.n, self.sy / self.n
        Sxx = self.xx - self.n * np.outer(mu, mu)
        Sxy = self.xy - self.n * mu * my
        coef = solve(Sxx + alpha * np.eye(len(mu)), Sxy, assume_a="pos")
        return FittedRidge(coef, my - mu @ coef)


class FittedRidge:
    def __init__(self, coef: np.ndarray, intercept: float):
        self.coef_, self.intercept_ = coef, intercept

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64) @ self.coef_ + self.intercept_
