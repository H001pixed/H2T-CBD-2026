# H2T-CBD (2026) — Reproducible Materials

This repository provides the code, figures, and aggregated analysis results behind the paper
*Preserving Accuracy and Improving Cross-Prompt Representation Alignment with Selective Anchoring:
Interpretable Concept Bottlenecks for Automated Holistic Essay Scoring* (submitted to PeerJ Computer Science).

The project builds **H2T-CBD (Holistic-to-Trait Concept-Bottleneck Distillation)**, an interpretable
automated essay scoring (AES) model whose 6-dimensional concept bottleneck is anchored to 6 groups of
19 linguistic features, and proposes the **Anchor Robustness Score (ARS = μ − σ)** to decide which anchor
groups should be *deactivated* for cross-prompt generalization. The paper shows that selectively dropping
the cross-prompt-unstable group (rhetoric/engagement) improves cross-prompt QWK and aligns bottleneck
representations across prompts.

This file explains what each part of the repository is, what it does, what software it uses, where to get
the data, and how to reproduce every number in the paper.

---

## 1. Repository Contents

```text
H2T-CBD-2026
|-- README.md        This file
|-- LICENSE          MIT license for the code
|-- .gitignore       Excludes raw data, NLTK data, model weights, and caches
|-- code/            All Python code (described module by module below)
|-- results/         Aggregated analysis outputs (CSV/JSON) for checking the reported numbers
|-- figures/         Paper figures (PNG) and the scripts that generate them
```

### 1.1 code/00_core — shared foundation (used by every experiment)

| File | What it is | What it does | Uses |
|---|---|---|---|
| `common.py` | Shared foundation | Loads the datasets, defines the anchor-group structure, the score-scaling map (ScaleMap), and the quadratic weighted kappa (QWK) metric | numpy, pandas |
| `models.py` | Model zoo | Implements the three model variants: BlackBox (MLP 768→256→256→1 with sigmoid), FeatConcat (linguistic features concatenated as input), and H2TCBD (6-D bottleneck with one anchor head per group) | PyTorch |
| `engine.py` | Training engine | Loads cached embeddings and trains/evaluates the variants under P-in (80/10/10 stratified split), P-LOPO (leave-one-prompt-out, 8 folds), and P-cross (train on source, zero-shot evaluate on target) | PyTorch |
| `metrics_logger.py` | Logging helper | Streaming per-epoch metric logger for training runs | json, time |
| `extract_embeddings.py` | Embedding extraction | The only GPU-heavy step: extracts 768-d embeddings from the frozen DeBERTa-v3-base encoder for all three datasets and caches them as `.npz` under `results/runs/_emb/` | transformers, torch |
| `m0_sanity.py` | Sanity gate | Verifies QWK correctness, the ScaleMap round-trip, and data-pipeline connectivity before experiments | numpy |
| `get_data.py` | Dataset downloader | Optional automatic download of the three official Kaggle datasets (requires a Kaggle API token) | kagglehub |
| `requirements.txt` | Dependency list | Minimal packages needed to run the pipeline | — |

### 1.2 code/01_features — the 19 linguistic features

| File | What it does |
|---|---|
| `prepare_feat.py` | Converts the Kaggle ASAP-1 file `training_set_rel3.tsv` into a clean CSV (decoded as cp1252; undefined bytes replaced) |
| `fix_feat.py` | Computes the 19 linguistic features for ASAP-1 (takes about 1–1.5 h) |
| `fix_feat2.py` | Normalizes each prompt's score to the 0–10 range |
| `features.py` | **FeatAlign**: the actual 19-feature computation (pure data processing; the M0 sanity gate requires per-column correlation r > 0.95 with the provided values) — uses numpy/pandas/nltk/textstat/textblob/pyspellchecker |

### 1.3 code/02–12 — experiments and analyses

| Folder / script | What it does |
|---|---|
| `02_exp1_non_inferiority/run_block1.py` | **Main experiments**: BlackBox + H2T-CBD across P-in, P-LOPO, and P-cross (90 training units in total). `result.json` marks completed units; results accumulate into `results.tsv` |
| `03_exp2_ablation_pin/run_ablation_pin.py` | **P-in ablation** on ASAP-1: no-anchor and feature-concatenation variants vs black-box and full anchoring; decomposes bottleneck cost, anchor cost, and feature value |
| `04_exp3_lopo/README.md` | Notes only; the P-LOPO (24-fold) results are produced by `run_block1.py` |
| `05_exp4_ablation_lopo/run_ablation_lopo.py` | **LOPO ablation**: no-anchor × 3 seeds × 8 folds = 24 units; tests whether removing the anchor loss shrinks cross-prompt variance and whether the hurt folds recover |
| `06_exp5_feature_corr/analyze_feature_score_corr.py` | **Feature–score correlations**: 8 prompts × 19 features × score (Spearman); tests whether hurt prompts have weaker associations and identifies globally weak groups |
| `07_exp6_gradient_conflict/analyze_gradient_conflict.py` | **Gradient-conflict hypothesis**: cosine between the MSE gradient and the anchor-loss gradient on shared parameters per LOPO fold; compares benefit vs hurt folds |
| `08_exp7_auxiliary_task/run_pin_per_prompt.py` | **Prompt-level P-in**: per-prompt QWK evaluation (3 seeds); combined with the 24 LOPO folds it produces the 48 paired samples behind the baseline-predictability correlation |
| `08_exp7_auxiliary_task/bootstrap_validation.py` | **Prompt-level cluster bootstrap** (10,000 draws, prompt as the sampling unit) for the correlation between black-box QWK and ΔQWK |
| `09_exp8_ars/recompute_ars.py` | **ARS recomputation**: the six ARS values (μ − σ, population σ) directly from the feature table |
| `09_exp8_ars/ars_loo_robustness.py` | **Leave-one-out robustness** of ARS: recomputes μ/σ/ARS per fold using only the 7 training prompts |
| `10_exp9_selective_anchoring/run_lopo_5groups.py` | **Selective anchoring**: drops the rhetoric/engagement group (ARS = 0.070 < θ = 0.1) and trains the 5-group configuration under P-LOPO, 24 folds |
| `11_exp10_mmd/analyze_representation_alignment.py` | **Mechanism validation**: extracts bottleneck representations from the LOPO checkpoints and measures cross-prompt cosine similarity and MMD² |
| `12_analysis/CI_Bootstrap.py` | **Non-inferiority CI**: bootstrap confidence intervals for the 24 paired conditions read from `results/runs/` |
| `12_analysis/rbf_kernel.py` | **RBF kernel** with median-heuristic bandwidth, used by the MMD computation (Gretton et al., 2012) |

### 1.4 results/ — aggregated outputs

- `results/feature_score_correlation.csv` — the 8 × 19 absolute Spearman correlation table used by the ARS computation.
- `results/runs/` — per-condition analysis outputs (JSON/CSV), including the main P-in/P-LOPO/P-cross per-fold results, ablations, ARS bootstrap, θ sensitivity, K/λ control, gradient-conflict and MMD analyses. These allow the numbers reported in the paper to be checked **directly**, without re-running the pipeline. Model weights and embedding caches are intentionally **not** included.

### 1.5 figures/

- `paper_figures/` — the paper's figures (PNG, ≥300 dpi).
- `figure_scripts/` — the scripts that regenerate them (`generate_fig1_accuracy.py` … `generate_fig5_validation.py`).

---

## 2. Datasets (third-party, download required)

All three datasets are public Kaggle competition data. **The raw files are not redistributed with this
repository** (their licenses do not permit redistribution); download them from the official pages below
and place them in the indicated directories.

| Dataset | Official source | File to download | Put it in | Content |
|---|---|---|---|---|
| ASAP-1 | [Hewlett Foundation — Automated Essay Scoring](https://www.kaggle.com/competitions/asap-aes/data) | `training_set_rel3.tsv` | `data/01_raw_datasets/feat/` | 12,976 essays, 8 prompts, scores 0–10 |
| ASAP-2.0 | [Learning Agency Lab — AES 2.0](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/data) | `train.csv` | `data/01_raw_datasets/asap2/` | 17,307 essays, scores 1–6, no prompt labels |
| Feedback | [Feedback Prize — English Language Learning](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/data) | `train.csv` | `data/01_raw_datasets/dysf/` | 3,911 essays, six analytic scores (averaged and rounded to a 1–5 holistic target) |

Notes:
- The directories `data/01_raw_datasets/feat/`, `asap2/`, and `dysf/` are **not** tracked in the archive;
  create them (or let `get_data.py` do it) and drop the files in.
- ASAP-1 is the only dataset that activates the anchor loss; ASAP-2.0 and Feedback are used for the
  in-set and cross-dataset evaluations without the 19 linguistic features.
- The ASAP-1 TSV is decoded as Windows-1252 (cp1252) with the few undefined bytes replaced;
  see `code/01_features/prepare_feat.py`.
- To use `get_data.py`, place your `kaggle.json` (from https://www.kaggle.com/settings/account) in
  `~/.kaggle/`, or set `KAGGLE_USERNAME` and `KAGGLE_KEY`.

---

## 3. Environment

- Tested with **Python 3.10.20** on **Ubuntu 22.04.3 LTS** (CUDA 12.1, NVIDIA GeForce RTX 4090, 24 GB).
- Install dependencies:

  ```bash
  pip install -r code/00_core/requirements.txt
  ```

- Main libraries: PyTorch ≥ 2.2, Transformers ≥ 4.46 (DeBERTa-v3-base), NumPy, SciPy, pandas, NLTK,
  textstat, TextBlob, pyspellchecker (offline features only), matplotlib, openpyxl, tqdm,
  huggingface_hub, kagglehub.
- **GPU** is needed only for embedding extraction (`extract_embeddings.py`). Model training runs on CPU
  (slower) or GPU; all analysis and figure steps are CPU-only.
- NLTK tokenizer and POS-tagging data are not bundled; install them with:

  ```bash
  python -c "import nltk; [nltk.download(x) for x in ['punkt','punkt_tab','averaged_perceptron_tagger','averaged_perceptron_tagger_eng']]"
  ```

- If the Hugging Face hub is unreachable, set `export HF_ENDPOINT=https://hf-mirror.com` before embedding extraction.

---

## 4. Reproduction (step by step)

All paths below are relative to the repository root. The pipeline writes its outputs
(embeddings, `result.json`, logs, and model weights) under `results/runs/` as it runs.

### Step 0 — Prepare the official data

Download the three files from Section 2 and place them at:

```text
data/01_raw_datasets/feat/training_set_rel3.tsv
data/01_raw_datasets/asap2/train.csv
data/01_raw_datasets/dysf/train.csv
```

or run the automatic downloader:

```bash
cd code/00_core
python get_data.py
```

### Step 1 — Data preprocessing

```bash
cd code/01_features
python prepare_feat.py
python fix_feat.py     # computes the 19 features (≈1–1.5 h)
python fix_feat2.py    # normalizes each prompt's score to 0–10
```

### Step 2 — Embedding extraction (GPU)

```bash
export HF_ENDPOINT=https://hf-mirror.com
cd code/00_core
python extract_embeddings.py
```

Caches the DeBERTa-v3-base embeddings under `results/runs/_emb/`.

### Step 3 — M0 sanity check

```bash
python m0_sanity.py
```

Prints `M0 SANITY PASSED` when QWK, ScaleMap, and the data pipeline check out.

### Step 4 — Main experiments (P-in / P-LOPO / P-cross)

```bash
cd code/02_exp1_non_inferiority
python run_block1.py
```

Runs BlackBox + H2T-CBD across all protocols (90 training units). Each unit's `result.json` is a
completion marker; re-running skips completed units.

### Step 5 — Ablations

```bash
cd code/03_exp2_ablation_pin
python run_ablation_pin.py

cd ../05_exp4_ablation_lopo
python run_ablation_lopo.py
```

### Step 6 — Auxiliary-task effect and its bootstrap

```bash
cd code/08_exp7_auxiliary_task
python run_pin_per_prompt.py
python bootstrap_validation.py
```

### Step 7 — The two mechanism hypotheses

```bash
cd code/06_exp5_feature_corr
python analyze_feature_score_corr.py

cd ../07_exp6_gradient_conflict
python analyze_gradient_conflict.py
```

### Step 8 — ARS and leave-one-out robustness

```bash
cd code/09_exp8_ars
python recompute_ars.py
python ars_loo_robustness.py
```

### Step 9 — Selective anchoring (5-group)

```bash
cd code/10_exp9_selective_anchoring
python run_lopo_5groups.py
```

### Step 10 — Representation alignment (MMD² / cosine)

```bash
cd code/11_exp10_mmd
python analyze_representation_alignment.py
```

Requires the `model.pt` checkpoints saved by Steps 4 and 9.

### Step 11 — Non-inferiority CI

```bash
cd code/12_analysis
python CI_Bootstrap.py
```

### Step 12 — Regenerate the paper figures

```bash
cd figures/figure_scripts
python generate_fig1_accuracy.py
python generate_fig2_lopo.py
python generate_fig3_bb_delta.py
python generate_fig4_ars.py
python generate_fig5_validation.py
```

The outputs overwrite the PNGs in `figures/paper_figures/`.

---

## 5. Notes and precautions

- **Data and licensing**: the raw datasets are third-party Kaggle data and are not redistributed here;
  the code is released under the MIT license. This study uses the data under their public competition
  terms for research purposes.
- **Reproducibility**: random seeds (13, 42, 123) and hyperparameters (e.g., anchor weight λ = 0.5;
  rescaled 5-group λ = 0.4167) are fixed inside the scripts, so re-runs reproduce the reported numbers.
- **Completeness marker**: a run unit is complete when its `result.json` exists. For a full re-run,
  delete the experiment directories under `results/runs/` first (the analysis outputs shipped here are
  then overwritten by fresh ones).
- **What is not included**: raw datasets, embedding caches, model weights, and the paper's
  summary/Excel tables are not distributed; the pipeline regenerates them (see Section 4), and the
  aggregated analysis outputs under `results/runs/` let readers verify the reported values directly.
