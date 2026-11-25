#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/run_e3.py --config configs/e3_koniq_to_kadid.yaml
