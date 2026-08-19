"""Effect sizes for the prompt-level and selective-anchoring analyses.

Reads runs/result.json and writes results/runs/prompt_level_analysis/
effect_sizes.json with:
  - correlation effect size for the 8-prompt analysis (|rho|, r^2);
  - Cohen's d and Hedges' g between benefited (1/2/8) and harmed (3/4/5/6)
    prompts on prompt-level mean dQWK;
  - paired d_z for 5-group vs 6-group P-LOPO QWK over 24 folds and for the
    12 harmed-fold pairs.

Usage: python compute_effect_sizes.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

for core in ("00_core",):
    p = Path(__file__).resolve().parents[1] / core
    if p.is_dir():
        sys.path.insert(0, str(p))
        break

import common as C  # noqa: E402

SEEDS = [13, 42, 123]
HARMED_FOLDS = [3, 4, 5, 6]
BENEFITED_PROMPTS = [1, 2, 8]
HARMED_PROMPTS = [3, 4, 5, 6]
OUT = C.RUNS / "prompt_level_analysis" / "effect_sizes.json"


def load(exp):
    return json.loads((C.RUNS / exp / "result.json").read_text(encoding="utf-8"))


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2)
                 / (na + nb - 2))
    return (a.mean() - b.mean()) / sp


def hedges_g(d, na, nb):
    return d * (1 - 3 / (4 * (na + nb) - 9))


def dz_paired(diffs):
    diffs = np.asarray(diffs, float)
    return diffs.mean() / diffs.std(ddof=1)


def main():
    pl = json.loads((C.RUNS / "prompt_level_analysis" / "result.json").read_text(
        encoding="utf-8"))
    rho8 = pl["rho_8_prompts"]
    means = {m["prompt"]: m["dQWK"] for m in pl["prompt_means"]}

    ben = [means[p] for p in BENEFITED_PROMPTS]
    har = [means[p] for p in HARMED_PROMPTS]
    d = cohens_d(ben, har)
    g = hedges_g(d, len(ben), len(har))

    diffs = []
    for seed in SEEDS:
        for fold in range(1, 9):
            q6 = load(f"B1_LOPO_feat_h2tcbd_s{seed}_fold{fold}")["test_qwk"]
            q5 = load(f"B1_5grp_rescaled_LOPO_feat_s{seed}_fold{fold}")["test_qwk"]
            diffs.append(q5 - q6)
    diffs = np.array(diffs)

    hd = []
    for seed in SEEDS:
        for fold in HARMED_FOLDS:
            q6 = load(f"B1_LOPO_feat_h2tcbd_s{seed}_fold{fold}")["test_qwk"]
            q5 = load(f"B1_5grp_rescaled_LOPO_feat_s{seed}_fold{fold}")["test_qwk"]
            hd.append(q5 - q6)
    hd = np.array(hd)

    result = {
        "note": ("Effect sizes (Cohen 1988; Lakens 2013): correlation effect "
                 "size uses |rho|; two-group comparisons use Cohen's d with "
                 "Hedges' g small-sample correction; paired comparisons use "
                 "paired d_z."),
        "correlation_effect_size": {
            "abs_rho": round(abs(rho8), 4),
            "r2": round(rho8 ** 2, 4),
        },
        "benefited_vs_harmed": {
            "benefited_prompts": BENEFITED_PROMPTS,
            "harmed_prompts": HARMED_PROMPTS,
            "benefited_mean_dqwk": round(float(np.mean(ben)), 4),
            "harmed_mean_dqwk": round(float(np.mean(har)), 4),
            "cohens_d": round(float(d), 4),
            "hedges_g": round(float(g), 4),
        },
        "five_vs_six_groups": {
            "n_folds": len(diffs),
            "mean_diff_qwk": round(float(diffs.mean()), 4),
            "sd_diff": round(float(diffs.std(ddof=1)), 4),
            "paired_dz": round(float(dz_paired(diffs)), 4),
            "hedges_gz": round(float(hedges_g(dz_paired(diffs),
                                              len(diffs), len(diffs))), 4),
        },
        "harmed_fold_recovery": {
            "n_pairs": len(hd),
            "mean_diff_qwk": round(float(hd.mean()), 4),
            "sd_diff": round(float(hd.std(ddof=1)), 4),
            "paired_dz": round(float(dz_paired(hd)), 4),
            "hedges_gz": round(float(hedges_g(dz_paired(hd),
                                              len(hd), len(hd))), 4),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"saved -> {OUT}")
    print("corr:", result["correlation_effect_size"])
    print("ben vs harmed:", result["benefited_vs_harmed"]["cohens_d"],
          result["benefited_vs_harmed"]["hedges_g"])
    print("5vs6 dz:", result["five_vs_six_groups"]["paired_dz"])
    print("harmed dz:", result["harmed_fold_recovery"]["paired_dz"])


if __name__ == "__main__":
    main()
