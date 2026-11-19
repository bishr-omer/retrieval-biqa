# retrieval-biqa

Does retrieving similar images with known quality scores improve blind image quality
prediction, and does the retrieved evidence actually explain the score?

This is a skills project. Retrieval for IQA has been done before (see Related work).
What this repo contributes is a careful protocol: leakage-free retrieval, a cost-versus-accuracy
table, a fair test of database updates against few-shot fine-tuning, and checks of whether
predictions depend on the evidence.

## Status
| Exp | What | Status |
|---|---|---|
| E0 | Linear probe on frozen SigLIP / CLIP; LoRA reference | code ready |
| E1 | Training-free kNN (k, embedding, temperature sweep) | code ready |
| E2 | Retrieval + cross-attention head | todo |
| E3 | KonIQ → LIVEC, database update vs few-shot LoRA | todo |
| E4 | Shuffled-score test, KADID distortion precision@k, figures | partial (shuffle test in E1) |
| E5 | VLM given retrieved images + scores (stretch) | todo |

## Results
Filled in as experiments run. Negative results are reported as they are.

## Protocol
- **Scores:** MOS normalised to [0, 1] per dataset using its fixed scale (KonIQ 1-5,
  LIVEC 0-100, KADID 1-5), so datasets can share one index.
- **Splits:** the 5 reference-level splits used for the LoRA experiments (`splits/*_reference.json`),
  reported as mean ± std. If a split has no validation set, 10% of the train scenes are held out
  (seeded) for tuning. No reference scene appears in both train and test (checked at load time).
- **No test leakage:** the index holds the train split only. Hyperparameters (k, embedding,
  temperature, ridge alpha) are picked on validation and reported once on test.
- **Leave-one-out:** when the query set is also in the index (E2 training), each query's own
  entry is excluded from its neighbours.
- **KADID evidence check:** distortion-type precision@k is reported with same-scene
  neighbours excluded and included. Random baseline: 1/25 = 4%.
- **Cost table:** every method logs trainable parameters, training GPU-hours, latency and
  index size to `results/cost.csv`.

## Setup
```bash
pip install -r requirements.txt          # FAISS optional; numpy fallback is fine at this scale
python -m pytest -q tests/               # synthetic-data checks of plumbing and leakage guards
```

This repo lives inside the `multimodal-biqa` project folder (`biqa-revision/`) and reads
from it directly, so results use **the same 5 splits as the LoRA experiments**:
```
biqa-revision/
├── cache/<dataset>_<backbone>.npz     names, feats (siglip, siglip2, cliph, nss, onealign)
├── manifests/<dataset>.csv            image, ref, mos
├── splits/<dataset>_reference.json    list of 5 splits
└── retrieval-biqa/                    this repo
```

```bash
python scripts/check_setup.py --config configs/e1_koniq.yaml   # checks alignment, trains nothing
bash scripts/run_e0.sh
bash scripts/run_e1.sh
```
Outputs: `results/e0/result.json`, `results/e1/result.json` (mean ± std over splits plus
per-split detail), `results/e1/sweep_val_split*.csv`, `results/cost.csv`.

## Layout
```
configs/    yaml per experiment
data/       dataset loaders (KonIQ, LIVEC, KADID) and splits
features/   load cached embeddings, aligned to dataset order
index/      exact inner-product index with exclusion guards and database update
models/     knn_regressor.py (E1); retrieval_head.py (E2, todo)
eval/       SRCC, PLCC (4-parameter logistic), distortion precision@k, cost table
scripts/    run_e0 / run_e1
tests/      synthetic-data tests
```

## Related work
- X. Wang et al., *Regression-free Blind Image Quality Assessment with Content-Distortion
  Consistency*, arXiv:2307.09279. Hierarchical kNN retrieval for BIQA.
- K. Fu et al., *Enhancing Image Quality Assessment Ability of LMMs via Retrieval-Augmented
  Generation* (IQARAG), arXiv:2601.08311. Retrieved images and MOS as LMM context; KonIQ,
  LIVEC, KADID, SPAQ.
- K. Fan et al., *ME-IQA: Memory-Enhanced Image Quality Assessment via Re-Ranking*,
  arXiv:2603.20785.
- *QualiRAG: Retrieval-Augmented Generation for Visual Quality Understanding*, arXiv:2601.18195.
