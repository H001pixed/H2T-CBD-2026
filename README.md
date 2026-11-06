# H2T-CBD and ARS

Code for Holistic-to-Trait Concept-Bottleneck Distillation (H2T-CBD), the Anchor Robustness Score (ARS), and the English figures in the current manuscript.

H2T-CBD predicts a holistic essay score through six candidate scalar coordinates supervised by 19 automatically computed features in six groups. ARS selects feature groups using fitting essays before model training. Coordinates not retained by ARS leave both scoring and their corresponding feature supervision.

## Repository structure

```text
H2T-CBD-2026/
|-- README.md             Installation, experiment commands, and manuscript mapping.
|-- LICENSE               MIT license for the project code.
|-- STATEMENTS.md         Code availability, data availability, and reproduction scope.
|-- requirements.txt      Training, feature-computation, and plotting dependencies.
|-- run_experiments.py    Entry point for the manuscript experiment plan.
|-- configs/              Main and independent-validation configurations.
|-- Code/                 Data preparation, model, training, ARS, and analysis scripts.
|-- data/                 Original-source locations and preparation instructions.
`-- figures/              English figure scripts and local-input schema documentation.
```

## Installation

Use Python 3.12 and install the dependencies in an environment suitable for your GPU:

```bash
python -m venv .venv
# Activate the environment using the command for your operating system.
python -m pip install -r requirements.txt
```

Figure reproduction uses NumPy, SciPy, and Matplotlib and does not require training or downloading the Qwen model. Numerical inputs must be supplied locally; this repository contains code only. See [figures/README.md](figures/README.md).

## Prepare data and embeddings

Follow [data/README.md](data/README.md) to prepare the four corpora and compute features. Runtime data and results are stored under `workspace/`, outside the tracked source files.

```bash
python run_experiments.py --root workspace --stage embeddings --list
python run_experiments.py --root workspace --stage embeddings
```

The encoder is `Qwen/Qwen3-Embedding-8B` at revision `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`. It encodes the full text of each essay, pools the last non-padding token, and L2-normalizes the 4,096-dimensional representation. ASAP-1 and ASAP-2.0 use the main encoding configuration; PERSUADE and CELA use the independent-validation FP16 configuration.

## Run the manuscript experiments

```bash
# Inspect the full command plan without running experiments.
python run_experiments.py --root workspace --stage all --list

# Main model comparisons, ablations, and ARS evaluation.
python run_experiments.py --root workspace --stage main

# Independent validation on PERSUADE and CELA.
python run_experiments.py --root workspace --stage supplement

# Aggregate completed runs without fitting models.
python run_experiments.py --root workspace --stage analysis
```

`--device cpu` overrides the device for the main scoring-model stage. Encoding and independent-validation device settings are read from their JSON configurations. Use a fresh runtime directory for a new code or data version.

| Manuscript component | Code and evaluation |
|---|---|
| Model comparisons and anchoring ablations | `Code/02_Experiments/`: within-corpus and leave-one-prompt-out comparisons on ASAP-1 and ASAP-2.0. |
| ARS selection and paired scoring | `Ars.py`, `Experiments.py`: full versus selective anchoring, including the ASAP-2.0 official split. |
| Independent ARS validation | `Code/14_ARS_Supplement/`: eight-prompt PERSUADE leave-one-prompt-out evaluation and single-prompt CELA evaluation. |
| Model and paired summary tables | `Code/13_Result_Organization/` and `Analyze_Supplement.py`. |
| Coordinate-contribution cases | `Find_Illustrative_Essays.py` and the Figure 5 script. |
| English Figures 1-5 | Four scripts in `figures/`, reading locally supplied numerical inputs. |

The default plan contains 700 main scoring-model runs and 180 independent-validation runs, each identified by dataset, protocol, variant, seed, and held-out prompt where applicable. Original datasets, model weights, and per-essay artifacts are obtained or generated locally.

## Model and training settings

| Setting | Value |
|---|---|
| Perceptron hidden layers | 256, 256 with GELU |
| Dropout | 0.2 |
| Optimizer | AdamW |
| Learning rate / weight decay | 0.001 / 0.0001 |
| Scoring-model batch size | 256 |
| Maximum epochs / early-stopping patience | 120 / 12 |
| Seeds | 13, 42, 123, 2026-2032 |
| Feature supervision | Each retained group's reconstruction MSE has coefficient 0.5/6. |
| ARS | Oriented feature correlations, group mean minus population SD, strict threshold greater than 0.10. |
| Score target | Fitting-sample empirical cumulative-probability midpoints. |
| Prediction | Rating level with the nearest mapped midpoint. |

The unanchored model retains the six-coordinate scoring path without feature supervision. The black-box baseline directly predicts the score from the perceptron. The feature-concatenation model adds all 19 standardized features to its input. Full anchoring supervises and uses all six coordinates; selective anchoring uses the retained coordinates only.

## Features

| Group | Automatically computed features |
|---|---|
| Length and fluency | Character count, word count, sentence count. |
| Lexical complexity | Mean word length, type-token ratio, adjective count, adverb count. |
| Syntactic complexity | Complex sentence ratio, clause density, verb count, noun count. |
| Mechanics | Spelling error count, punctuation density. |
| Cohesion and readability | Flesch Reading Ease, adjacent-sentence similarity. |
| Rhetoric and engagement | Subjectivity, transitional phrase frequency, figurative language frequency, question mark density. |

Dataset access and licensing information are in [STATEMENTS.md](STATEMENTS.md). Coordinate contributions are the signed terms before the sigmoid and rating-level mapping.
