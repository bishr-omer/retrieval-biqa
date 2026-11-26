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
| E0 | Linear probe on frozen SigLIP / CLIP; LoRA reference | done (LoRA numbers pending) |
| E1 | Training-free kNN (k, embedding, temperature sweep) | done |
| E1b | Does retrieval add anything beyond a probe? PLS-kNN and probe + residual kNN | done |
| E2 | Retrieval + cross-attention head | dropped (see Findings 3 and 6) |
| E3 | KonIQ → LIVE-itW and KonIQ → KADID-10k (scenes held out): done. Few-shot LoRA on the same supports: optional | done |
| E4 | What do neighbours match on KADID: scene or distortion? Per-type breakdown, KonIQ probe failures, figures | code ready |
| E5 | VLM given retrieved images + scores (stretch) | todo |

## Results
KonIQ-10k, 5 reference-level splits, mean ± std on test. Negative results are reported as they are.

| Method | Trainable params | SRCC | PLCC |
|---|---|---|---|
| kNN, frozen features (E1, concat SigLIP+CLIP-H+NSS, k 10-20) | 0 | 0.771 ± 0.007 | 0.791 ± 0.005 |
| Linear probe, SigLIP (E0) | 1,153 | 0.881 ± 0.006 | 0.901 ± 0.002 |
| Linear probe, SigLIP2 (E0) | 1,153 | 0.878 ± 0.008 | 0.902 ± 0.005 |
| Linear probe, CLIP-H (E0) | 1,025 | 0.894 ± 0.006 | 0.912 ± 0.003 |
| Linear probe, SigLIP + CLIP-H + NSS (E1b) | 2,315 | 0.910 ± 0.004 | |
| Probe + residual kNN, SigLIP + CLIP-H + NSS (E1b) | 2,315 | **0.913 ± 0.004** | |
| LoRA r8 (q,v), SigLIP, multimodal-biqa (reference) | 1.72M (0.23% of backbone) | **0.950 ± 0.002** | 0.960 |

**Finding 1.** Training-free retrieval on frozen generic embeddings is about 0.12 SRCC below a
linear probe on the same features. The quality signal is present (the probe finds it) but
cosine similarity is dominated by content, so nearest neighbours share scenes, not quality.
Shuffling neighbour scores drops kNN to ~0 SRCC, as expected for a method with no other input.

**Finding 2.** In a supervised projection (PLS, 8-16 dims, fitted on train MOS) kNN recovers most
of the gap (CLIP-H 0.887, SigLIP 0.872), so E1 failed because of the similarity measure. It still
does not beat the probe, and on the concatenated features it falls further behind (0.884 vs 0.910).
Retrieval as the predictor itself is not competitive here.

**Finding 3.** Retrieval as a correction on top of the probe adds a small but consistent gain:
the probe plus a similarity-weighted mean of the neighbours' out-of-fold residuals.

| Embedding | Probe | Probe + residual kNN | Gain | Splits improved | Shuffled residuals |
|---|---|---|---|---|---|
| CLIP-H | 0.8941 ± 0.0058 | 0.8972 ± 0.0053 | +0.0031 | 5/5 | 0.8910 |
| SigLIP | 0.8813 ± 0.0060 | 0.8834 ± 0.0058 | +0.0021 | 5/5 | 0.8781 |
| SigLIP + CLIP-H + NSS | 0.9095 ± 0.0039 | **0.9132 ± 0.0036** | +0.0037 | 5/5 | 0.9070 |

The gain is real (positive on 15/15 paired comparisons; shuffling the residuals drops below the
probe) but small: in-domain on KonIQ the neighbours correct a slight scene-level bias of the probe.
The best k is usually 50-100, which points to broad bias correction rather than close matches.
A learned cross-attention head (E2) can at best capture this ~0.003, so E3, where retrieval's
real case lies (adding target data without retraining), comes first.

**Finding 4. Retrieval does not beat LoRA in-domain.** On the same 5 splits, LoRA r8 on SigLIP
reaches 0.950 SRCC against 0.913 for the best retrieval method, a gap of 0.037 (over 10× the split
std). Retrieval's case is cost: no GPU and ~2.3k trainable parameters, against 1.72M parameters
and ~32 GPU-minutes per LoRA run. The ridge probes here match the multimodal-biqa frozen heads to
within 0.001 SRCC on the same splits (SigLIP 0.8813 vs 0.8825; SigLIP+CLIP-H+NSS 0.9095 vs
0.9088), which cross-checks splits and features between the two codebases. Every E1 number
reproduces exactly on rerun.

**Finding 5. KonIQ → LIVE-itW is too small a shift to separate the methods (E3).** Paired
delta-SRCC against the zero-shot KonIQ probe, mean over 5 source splits × 10 support draws:

| N target images added | Concat: database update (mixed index, k=100) | Concat: ridge refit | CLIP-H: database update | CLIP-H: ridge refit |
|---|---|---|---|---|
| 10 | +0.0004 | +0.0008 | +0.0040 | +0.0070 |
| 25 | +0.0015 | +0.0020 | +0.0047 | +0.0109 |
| 50 | +0.0033 | +0.0026 | +0.0066 | +0.0119 |
| 100 | +0.0051 (100% of runs improved) | +0.0038 | +0.0080 | +0.0147 |

Zero-shot is 0.823 (concat) against an in-domain LIVE-itW ceiling of 0.836, so there is only
~0.013 SRCC to recover. On concat features the database update matches or slightly beats
refitting from N = 50; on CLIP-H, refitting wins at every N. A target-only index hurts
(k = 5: −0.03 to −0.05): a few noisy residuals are not reliable evidence. With a frozen backbone
and a linear head, refitting the head is practically free, so "update without retraining" only
pays off when retraining is expensive (LoRA). The next test uses a large shift: KonIQ → KADID-10k.

**Finding 6. On a large shift, refitting the head beats the database update by 4-5×
(E3, KonIQ → KADID-10k).** Support images come only from KADID training scenes and the test
images from held-out scenes. Zero-shot SRCC is about 0.730 (concat) and 0.666 (CLIP-H), against
in-domain ceilings of 0.888 and 0.848. Paired delta-SRCC against zero-shot, concat features,
mean of 50 runs:

| N | Database update (mixed, k=100) | Ridge refit | Refit, target at 10% weight | Refit + residual kNN |
|---|---|---|---|---|
| 10 | +0.001 | +0.009 | **+0.022** | +0.009 |
| 50 | +0.005 | +0.032 | **+0.072** | +0.034 |
| 100 | +0.014 | +0.059 | **+0.101** | +0.063 |
| 250 | +0.021 | +0.086 | **+0.113** | +0.089 |

The weighted refit recovers ~72% of the gap to the ceiling at N = 250 (0.113 of 0.158) using
~4% of KADID's training data; the database update recovers ~13%. The weight was fixed before
seeing results. Retrieval on top of the refit adds a small, consistent +0.002 to +0.004 (up to
+0.007 on CLIP-H). One exception: on CLIP-H, a target-only index with k = 10 beats the plain
refit at small N (+0.018 vs +0.009 at N = 10) but not the weighted refit (+0.027).
Interpretation: the shift is a global change in how features map to quality for distortion
types KonIQ never shows. Refitting changes the head's direction for every image at once; a
residual correction only shifts each image by its neighbours' mean error.

**Overall.** Across in-domain KonIQ, a small shift (LIVE-itW) and a large shift (KADID),
retrieval as a correction on frozen features never beats a properly weighted refit of a linear
head. Its consistent contribution is a small gain on top of the refit. E4 tests why.

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
