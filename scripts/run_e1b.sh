#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/run_e1b.py --config configs/e1b_koniq.yaml
