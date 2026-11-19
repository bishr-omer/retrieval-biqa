"""Exact inner-product index (cosine on L2-normalised vectors).

KonIQ is ~10k images, so exact search (IndexFlatIP) is fast; no need for approximate search.
Falls back to numpy if faiss is not installed.

Leakage guard: `search(..., exclude=...)` drops forbidden neighbours per query, e.g.
  * the query itself (leave-one-out when the query set is also the index set),
  * KADID images that share the query's reference scene.
"""
from __future__ import annotations

import numpy as np

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None


class Index:
    def __init__(self, feats: np.ndarray, scores: np.ndarray, ids: np.ndarray | None = None):
        self.feats = np.ascontiguousarray(feats, dtype=np.float32)
        self.scores = np.asarray(scores, dtype=np.float64)
        self.ids = np.arange(len(feats)) if ids is None else np.asarray(ids)
        self._build()

    def _build(self):
        if faiss is not None:
            self._fi = faiss.IndexFlatIP(self.feats.shape[1])
            self._fi.add(self.feats)
        else:
            self._fi = None

    def add(self, feats: np.ndarray, scores: np.ndarray, ids: np.ndarray) -> None:
        """Database update without retraining (E3)."""
        self.feats = np.vstack([self.feats, np.asarray(feats, dtype=np.float32)])
        self.scores = np.concatenate([self.scores, scores])
        self.ids = np.concatenate([self.ids, ids])
        self._build()

    def __len__(self):
        return len(self.feats)

    def search(self, q: np.ndarray, k: int, exclude=None):
        """Return (sims, scores, ids), each shape (n_query, k).

        exclude: optional callable(query_row, candidate_ids) -> bool mask of candidates to drop,
                 or an array of ids (one per query) to drop (leave-one-out).
        """
        q = np.ascontiguousarray(q, dtype=np.float32)
        # KADID has 125 images per reference scene, so excluding the same scene can drop up to
        # 124 candidates; a pad of 256 covers that.
        pad = 0 if exclude is None else 256
        kk = min(len(self), k + pad)
        if self._fi is not None:
            sims, idx = self._fi.search(q, kk)
        else:
            s = q @ self.feats.T
            idx = np.argsort(-s, axis=1)[:, :kk]
            sims = np.take_along_axis(s, idx, axis=1)

        out_s = np.empty((len(q), k)); out_y = np.empty((len(q), k)); out_i = np.empty((len(q), k), dtype=self.ids.dtype)
        for r in range(len(q)):
            cand_ids = self.ids[idx[r]]
            keep = np.ones(kk, dtype=bool)
            if exclude is not None:
                if callable(exclude):
                    keep &= ~exclude(r, cand_ids)
                else:
                    keep &= cand_ids != exclude[r]
            sel = np.flatnonzero(keep)[:k]
            if sel.size < k:
                raise RuntimeError("Not enough neighbours after exclusion; increase the pad.")
            out_s[r] = sims[r, sel]; out_y[r] = self.scores[idx[r, sel]]; out_i[r] = cand_ids[sel]
        return out_s, out_y, out_i
