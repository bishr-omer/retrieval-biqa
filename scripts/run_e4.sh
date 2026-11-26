#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/run_e4.py --config configs/e4_kadid.yaml
python scripts/make_figures.py --neighbours results/e4/neighbours_heldout_scenes.json --img_root ../datasets/kadid10k --out notebooks/figures --rows 4
