# English manuscript figure scripts

These four scripts generate five English figures. The repository contains no experimental data or generated images. Provide the local CSV inputs described in [data/README.md](data/README.md) to reproduce data-based figures.

```bash
python figures/fig1_architecture.py
python figures/fig2_raincloud.py --data-dir /path/to/local/figure_inputs
python figures/fig3_4_results.py --data-dir /path/to/local/figure_inputs
python figures/fig5_contributions.py --data-dir /path/to/local/figure_inputs
```

All scripts accept `--output-dir PATH`; the default is `figures/output/`. Figures 2-5 additionally accept `--data-dir PATH`, defaulting to `figures/data/`. Outputs are PDF and 600-dpi PNG files, which are ignored by Git.

| Figure | Script | Required local input |
|---|---|---|
| 1. H2T-CBD training and scoring process | `fig1_architecture.py` | None; the diagram and mask are illustrative. |
| 2. Within-corpus model comparisons | `fig2_raincloud.py` | `within_corpus_seed_pairs.csv`. |
| 3. ARS feature-group retention | `fig3_4_results.py` | `ars_retention.csv`. |
| 4. Selective versus full anchoring | `fig3_4_results.py` | `paired_scoring.csv`. |
| 5. Signed coordinate contributions | `fig5_contributions.py` | `case_contributions.csv`. |

The original English labels, layouts, and signed-contribution convention are preserved. The scripts read numerical values from local files. Figure 2 checks seed coverage. Figure 5 verifies that the intercept and coordinate contributions reconstruct the continuous prediction after the sigmoid. Arial is used when available, with DejaVu Sans as the fallback.
