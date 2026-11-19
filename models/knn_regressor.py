"""E1: training-free kNN quality regressor.

score(q) = sum_i w_i * y_i,   w = softmax(sim_i / tau) over the top-k neighbours.
tau -> 0 approaches 1-NN; large tau approaches the plain mean of the k neighbours.
"""
from __future__ import annotations

import numpy as np

from index.faiss_index import Index


def weights(sims: np.ndarray, tau: float) -> np.ndarray:
    z = sims / max(tau, 1e-8)
    z = z - z.max(axis=1, keepdims=True)
    w = np.exp(z)
    return w / w.sum(axis=1, keepdims=True)


class KNNRegressor:
    def __init__(self, index: Index, k: int = 10, tau: float = 0.05):
        self.index, self.k, self.tau = index, k, tau

    def predict(self, q: np.ndarray, exclude=None, return_evidence: bool = False, shuffle_rng=None):
        sims, ys, ids = self.index.search(q, self.k, exclude=exclude)
        if shuffle_rng is not None:
            # E4 evidence check: replace each neighbour's score with a random database score.
            ys = shuffle_rng.choice(self.index.scores, size=ys.shape, replace=True)
        w = weights(sims, self.tau)
        pred = (w * ys).sum(axis=1)
        if return_evidence:
            return pred, {"sims": sims, "scores": ys, "ids": ids, "weights": w}
        return pred
