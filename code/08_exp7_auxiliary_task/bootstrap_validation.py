"""Prompt-level (clustered) bootstrap for the BB-delta negative correlation.

Data: the real 48 paired BB-delta measurements read from the packaged runs
(8 prompts x 6 measurements per prompt: P-in 3 seeds + LOPO 3 folds), not
synthetic or mean-replicated data.

Method: cluster bootstrap (Efron & Tibshirani, 1993) with the prompt as the
sampling unit - resample 8 prompts with replacement, include all 6 real
measurements of each sampled prompt (48 points), recompute Spearman rho,
repeat 10,000 times, and take the 2.5%/97.5% percentiles as the 95% CI.

Expected output with the current data (seed=42):
  original rho = -0.4070, p = 0.0041
  bootstrap mean = -0.3761, 95% CI = [-0.6468, -0.0183],
  97.88% of samples negative (212 positive out of 10,000).

Pure numpy/scipy; runnable on a local machine.
"""
import json
from pathlib import Path
import numpy as np
from scipy import stats

SEEDS = [13, 42, 123]
N_PROMPTS = 8
N_BOOT = 10_000
RNG_SEED = 42  # Matches the paper's reproducible run.


def load_48_pairs():
    """Load the 48 real (BB_QWK, delta_QWK) pairs grouped by prompt (6 each)."""
    runs = Path(__file__).resolve().parents[2] / "results" / "runs"
    by_prompt = {p: [] for p in range(1, N_PROMPTS + 1)}
    for seed in SEEDS:
        # P-in: 3 seeds x 8 prompts from the per-prompt results.
        bb = json.loads((runs / f"B1_PinPerPrompt_feat_blackbox_s{seed}" / "result.json").read_text(encoding="utf-8"))
        hc = json.loads((runs / f"B1_PinPerPrompt_feat_h2tcbd_s{seed}" / "result.json").read_text(encoding="utf-8"))
        for pr in sorted(bb["per_prompt"], key=int):
            qb = bb["per_prompt"][pr]["qwk"]
            qh = hc["per_prompt"][pr]["qwk"]
            by_prompt[int(pr)].append((qb, qh - qb))
        # LOPO: 3 seeds x 8 folds.
        for fold in range(1, N_PROMPTS + 1):
            qb = json.loads((runs / f"B1_LOPO_feat_blackbox_s{seed}_fold{fold}" / "result.json").read_text(encoding="utf-8"))["test_qwk"]
            qh = json.loads((runs / f"B1_LOPO_feat_h2tcbd_s{seed}_fold{fold}" / "result.json").read_text(encoding="utf-8"))["test_qwk"]
            by_prompt[fold].append((qb, qh - qb))
    return by_prompt


def main():
    by_prompt = load_48_pairs()
    xs_all = np.array([x for p in range(1, 9) for x, _ in by_prompt[p]])
    ys_all = np.array([y for p in range(1, 9) for _, y in by_prompt[p]])

    orig_rho, orig_p = stats.spearmanr(xs_all, ys_all)
    print(f"original Spearman rho = {orig_rho:.4f}, p = {orig_p:.2e}")

    # Per-prompt within-group rho (diagnostic: within-prompt consistency).
    print("\nper-prompt rho (6 pairs within each prompt):")
    for p in range(1, 9):
        xs = np.array([x for x, _ in by_prompt[p]])
        ys = np.array([y for _, y in by_prompt[p]])
        r, _ = stats.spearmanr(xs, ys)
        print(f"  prompt {p}: rho = {r:+.4f}")

    rng = np.random.default_rng(RNG_SEED)
    boot_rhos = np.empty(N_BOOT)
    for i in range(N_BOOT):
        sel = rng.integers(1, N_PROMPTS + 1, size=N_PROMPTS)
        xs, ys = [], []
        for pid in sel:
            for x, y in by_prompt[pid]:
                xs.append(x)
                ys.append(y)
        boot_rhos[i] = stats.spearmanr(xs, ys).statistic

    ci_lower, ci_upper = np.percentile(boot_rhos, [2.5, 97.5])
    frac_neg = 100.0 * (boot_rhos < 0).mean()
    print(f"\nbootstrap n={N_BOOT} (prompt-level cluster resampling, seed={RNG_SEED}):")
    print(f"  mean rho = {np.mean(boot_rhos):.4f}")
    print(f"  std = {np.std(boot_rhos):.4f}")
    print(f"  95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"  negative fraction = {frac_neg:.2f}%  ({N_BOOT:,} draws: {(boot_rhos < 0).sum()} negative / {(boot_rhos > 0).sum()} positive)")

    # Persist the bootstrap distribution for traceability.
    out = Path(__file__).resolve().parent / "bootstrap_dist_n{}.npy".format(N_BOOT)
    np.save(out, boot_rhos)
    print(f"\nDistribution saved -> {out}")


if __name__ == "__main__":
    main()
