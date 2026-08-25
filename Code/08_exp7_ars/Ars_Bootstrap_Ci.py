"""Prompt-level bootstrap confidence intervals for the six ARS values.

ARS_k = mu_k - sigma_k, where mu_k and sigma_k are the mean and standard
deviation of the per-prompt group means rho_bar_k(p) =
mean_f |Spearman(feature, score; prompt p)| across the 8 ASAP-1 prompts.
Because sigma is defined across prompts, the natural resampling unit for
uncertainty estimation is the prompt: we resample the 8 prompts with
replacement, recompute mu/sigma/ARS for every group, and take the 2.5%/97.5%
percentiles of 10,000 draws as the 95% CI (same clustered-bootstrap
convention as Bootstrap_Validation.py).

The per-prompt correlations are computed once; the bootstrap resamples the
precomputed per-prompt group means, so the script runs in seconds.

Usage: python Ars_Bootstrap_Ci.py
Outputs: results/runs/ars_bootstrap/result.json
         results/runs/ars_bootstrap/ars_bootstrap_ci.csv
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats

for core in ("00_core",):
    p = Path(__file__).resolve().parents[1] / core
    if p.is_dir():
        sys.path.insert(0, str(p))
        break

import Common as C  # noqa: E402

THETA = 0.1
N_BOOT = 10_000
RNG_SEED = 42
OUT_DIR = C.RUNS / "ars_bootstrap"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = C.load_dataset("feat")
    prompts = sorted(df["essay_set"].unique())

    # Per-prompt per-feature |Spearman| (identical to Recompute_Ars.py).
    per_feat = {}
    for p in prompts:
        sub = df[df["essay_set"] == p]
        scores = sub["score"].values
        d = {}
        for f in C.FEAT_COLS:
            vals = sub[f].values
            mask = np.isfinite(vals)
            if mask.sum() < 10 or np.std(vals[mask]) == 0:
                d[f] = np.nan
                continue
            r, _ = stats.spearmanr(vals[mask], scores[mask])
            d[f] = abs(r)
        per_feat[p] = d

    # Per-prompt group means rho_bar_k(p).
    rho_bar = {}
    for g, feats in C.ANCHOR_GROUPS.items():
        rho_bar[g] = {}
        for p in prompts:
            rs = [per_feat[p][f] for f in feats if not np.isnan(per_feat[p][f])]
            rho_bar[g][p] = float(np.mean(rs)) if rs else np.nan

    # Cluster bootstrap over prompts.
    rng = np.random.default_rng(RNG_SEED)
    boot = {g: np.empty(N_BOOT) for g in C.ANCHOR_GROUPS}
    n_prompt = len(prompts)
    for b in range(N_BOOT):
        sel = rng.integers(0, n_prompt, size=n_prompt)
        for g in C.ANCHOR_GROUPS:
            vals = np.array([rho_bar[g][prompts[i]] for i in sel])
            vals = vals[np.isfinite(vals)]
            mu = float(vals.mean())
            sd = float(vals.std(ddof=0))
            boot[g][b] = mu - sd

    rows = []
    for g in C.ANCHOR_GROUPS:
        draws = boot[g]
        ci_lo, ci_hi = np.percentile(draws, [2.5, 97.5])
        rows.append({
            "group": g,
            "ars": round(float(draws.mean()), 4),
            "boot_mean": round(float(draws.mean()), 4),
            "boot_std": round(float(draws.std(ddof=0)), 4),
            "ci_lo": round(float(ci_lo), 4),
            "ci_hi": round(float(ci_hi), 4),
            "prob_ars_gt_theta": round(float((draws > THETA).mean()), 4),
        })
        print(f"{g:>24s} ARS={draws.mean():.3f} "
              f"95% CI=[{ci_lo:.3f}, {ci_hi:.3f}] "
              f"P(ARS>{THETA})={ (draws > THETA).mean():.3f}")

    # Decision consistency: fraction of draws where exactly rhetoric is below
    # the threshold (same screening decision as the full data).
    below = np.stack([boot[g] <= THETA for g in C.ANCHOR_GROUPS], axis=1)
    same_decision = ((below.sum(axis=1) == 1) &
                     (boot["rhetoric_engagement"] <= THETA)).mean()
    print(f"\ndecision identical to full data (only rhetoric <= {THETA}): "
          f"{same_decision:.3f}")

    result = {
        "note": ("Prompt-level cluster bootstrap over the 8 ASAP-1 prompts, "
                 "B=10,000, seed=42; percentile 95% CI of ARS per anchor "
                 "group (script: Ars_Bootstrap_Ci.py)."),
        "theta": THETA,
        "n_prompts": n_prompt,
        "n_boot": N_BOOT,
        "seed": RNG_SEED,
        "groups": rows,
        "decision_identical_fraction": round(float(same_decision), 4),
    }
    (OUT_DIR / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUT_DIR / "ars_bootstrap_ci.csv", index=False)
    print(f"\nsaved -> {OUT_DIR}")


if __name__ == "__main__":
    main()
