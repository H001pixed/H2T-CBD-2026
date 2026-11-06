# Source-code map

```text
Code/
|-- 00_Core/                 ARS, feature groups, scoring models, training, and embeddings.
|   |-- Ars.py               Oriented prompt-wise and single-prompt ARS selection.
|   |-- Common.py            Feature definitions, score mapping, metrics, and split helpers.
|   |-- Engine.py            Feature scaling, training, checkpoints, and coordinate contributions.
|   |-- Experiments.py       Paired data splits and experiment execution.
|   |-- Models.py            Black-box, unanchored, full, selective, and feature-concatenation models.
|   |-- Extract_Embeddings.py Full-text Qwen encoding and embedding-cache validation.
|   `-- Get_Data.py           Official source-file retrieval helpers.
|-- 01_Features/              Source preparation, overlap removal, and 19 feature computations.
|-- 02_Experiments/           Within-corpus, leave-one-prompt-out, and selective-anchoring entry points.
|-- 13_Result_Organization/   Expected experiment coverage and paired manuscript summaries.
`-- 14_ARS_Supplement/        PERSUADE/CELA encoding, independent validation, analysis, and cases.
```

The numerical scoring and ARS implementations are extracted from the completed experiment project. Release changes make paths portable, restrict the default experiment plan to the current manuscript, and use English output names and descriptions. The full-text encoder, fitting-only scaling, score mapping, fixed per-group reconstruction weight, and ARS threshold remain as used in the experiments.
