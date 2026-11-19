"""Cost table: every method reports what it costs next to how well it does.

This is the link to the vacancy's "adapt within a compute budget" direction. One row per
method in results/cost.csv. Re-running a method replaces its row instead of adding a duplicate.
"""
from __future__ import annotations

import csv
import time
from contextlib import contextmanager
from pathlib import Path

FIELDS = ["method", "trainable_params", "train_gpu_hours", "latency_ms_per_img",
          "index_mb", "srcc", "plcc", "notes"]


@contextmanager
def timer():
    t = {"s": time.perf_counter()}
    yield t
    t["elapsed"] = time.perf_counter() - t["s"]


def log_row(path: str | Path, **row) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if path.exists():
        with path.open(newline="") as f:
            rows = [r for r in csv.DictReader(f) if r.get("method") != row.get("method")]
    rows.append({k: row.get(k, "") for k in FIELDS})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows({k: r.get(k, "") for k in FIELDS} for r in rows)
