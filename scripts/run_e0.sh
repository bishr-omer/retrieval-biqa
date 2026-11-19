#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/run_e0.py --config configs/e0_koniq.yaml
