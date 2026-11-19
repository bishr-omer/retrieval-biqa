"""Cost table: every method reports what it costs next to how well it does.

This is the link to the vacancy's "adapt within a compute budget" direction. Fill one row
per method (linear probe, LoRA, kNN, retrieval head, database update) into results/cost.csv.
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
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})
