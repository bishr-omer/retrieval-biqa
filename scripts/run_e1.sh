#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/run_e1.py --config configs/e1_koniq.yaml
