# Experiment 3: P-LOPO leave-one-out (24 folds)

This experiment has no separate script of its own; its data is produced by the main experiment entry:

- Script: `protocol_lopo()` inside `code/02_exp1_non_inferiority/run_block1.py`
- What it does: on ASAP-1, hold out one prompt for testing and train on the other 7; 8 prompts × 3
  seeds = 24 folds. Each fold outputs the BlackBox / H2T-CBD QWK and the paired difference ΔQWK.
- Output: `results/runs/B1_LOPO_feat_{blackbox,h2tcbd}_s{seed}_fold{fold}/result.json`
