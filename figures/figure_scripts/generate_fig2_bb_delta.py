"""Figure 2 - BB vs delta-QWK scatter with prompt-level means (48 vs 8).

The 48 paired measurements (8 prompts x 6: P-in 3 seeds + LOPO 3 folds) are
read from runs/result.json. The 8 prompt-level means (averaging the 6
measurements per prompt) are added as larger markers, and the two analysis
units are annotated separately: the descriptive 48-pair Spearman rho and the
8-prompt rho with its permutation p. All inference statistics are saved to
results/runs/prompt_level_analysis/result.json.

Usage: python generate_fig2_bb_delta.py
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import stats

RUNS = Path(__file__).resolve().parents[2] / "results" / "runs"
SEEDS = [13, 42, 123]


def load(exp_id):
    p = RUNS / exp_id / "result.json"
    if not p.exists():
        raise FileNotFoundError(f"missing run: {exp_id}")
    return json.loads(p.read_text(encoding="utf-8"))


def collect_pairs():
    """48 real (BB_QWK, delta_QWK, protocol, prompt) measurements."""
    pairs = []
    for seed in SEEDS:
        bb = load(f"B1_PinPerPrompt_feat_blackbox_s{seed}")
        hc = load(f"B1_PinPerPrompt_feat_h2tcbd_s{seed}")
        for pr in sorted(bb["per_prompt"], key=int):
            pairs.append((bb["per_prompt"][pr]["qwk"],
                          hc["per_prompt"][pr]["qwk"] - bb["per_prompt"][pr]["qwk"],
                          "P-in", int(pr)))
    for seed in SEEDS:
        for fold in range(1, 9):
            q_bb = load(f"B1_LOPO_feat_blackbox_s{seed}_fold{fold}")["test_qwk"]
            q_hc = load(f"B1_LOPO_feat_h2tcbd_s{seed}_fold{fold}")["test_qwk"]
            pairs.append((q_bb, q_hc - q_bb, "LOPO", fold))
    assert len(pairs) == 48, len(pairs)
    return pairs


def main():
    pairs = collect_pairs()
    xs = np.array([p[0] for p in pairs])
    ys = np.array([p[1] for p in pairs])
    protocols = np.array([p[2] for p in pairs])
    prompts = np.array([p[3] for p in pairs])

    rho48, _ = stats.spearmanr(xs, ys)

    # Prompt-level means (8 points).
    mx, my = [], []
    for pr in range(1, 9):
        m = prompts == pr
        mx.append(float(xs[m].mean()))
        my.append(float(ys[m].mean()))
    mx, my = np.array(mx), np.array(my)
    rho8, _ = stats.spearmanr(mx, my)

    # Permutation test on the 8 prompt-level points.
    rng = np.random.default_rng(42)
    perm = np.empty(100_000)
    for i in range(100_000):
        perm[i] = stats.spearmanr(mx, rng.permutation(my)).statistic
    p_two = float((np.abs(perm) >= abs(rho8)).mean())
    p_one = float((perm <= rho8).mean())

    # Bootstrap CI over the 8 prompts.
    boot8 = []
    for _ in range(10_000):
        idx = rng.integers(0, 8, size=8)
        xb, yb = mx[idx], my[idx]
        if np.all(xb == xb[0]) or np.all(yb == yb[0]):
            continue
        boot8.append(stats.spearmanr(xb, yb).statistic)
    boot8 = np.array(boot8)
    ci8 = (float(np.percentile(boot8, 2.5)), float(np.percentile(boot8, 97.5)))

    # Cluster bootstrap over the 8 prompts on the 48 pairs (fine-grained).
    by_prompt = {p: [(x, y) for x, y, _, pr in pairs if pr == p]
                 for p in range(1, 9)}
    boot48 = []
    for _ in range(10_000):
        sel = rng.integers(1, 9, size=8)
        cx, cy = [], []
        for p in sel:
            for x, y in by_prompt[p]:
                cx.append(x)
                cy.append(y)
        boot48.append(stats.spearmanr(cx, cy).statistic)
    boot48 = np.array(boot48)
    ci48 = (float(np.percentile(boot48, 2.5)), float(np.percentile(boot48, 97.5)))
    neg_frac48 = float((boot48 < 0).mean())

    # Mixed-effects model (prompt random intercept, protocol fixed effect).
    mixed = None
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
        df = pd.DataFrame({"dQWK": ys, "BB_QWK": xs, "protocol": protocols,
                           "prompt": prompts})
        mf = smf.mixedlm("dQWK ~ BB_QWK + C(protocol)", df,
                         groups=df["prompt"]).fit(reml=True)
        mixed = {"beta_bb": float(mf.params["BB_QWK"]),
                 "se_bb": float(mf.bse["BB_QWK"]),
                 "p_bb": float(mf.pvalues["BB_QWK"])}
    except Exception as e:  # statsmodels optional for the figure itself
        print("mixed model skipped:", e)

    ols_cl = None
    try:
        import pandas as pd
        import statsmodels.api as sm
        df2 = pd.DataFrame({"dQWK": ys, "BB_QWK": xs, "protocol": protocols})
        m2 = sm.OLS.from_formula("dQWK ~ BB_QWK + C(protocol)", df2).fit(
            cov_type="cluster", cov_kwds={"groups": prompts})
        ols_cl = {"beta_bb": float(m2.params["BB_QWK"]),
                  "se_bb": float(m2.bse["BB_QWK"]),
                  "p_bb": float(m2.pvalues["BB_QWK"])}
    except Exception as e:
        print("ols clustered skipped:", e)

    stats_out = {
        "note": ("BB-QWK vs delta-QWK: 8-prompt-level primary analysis plus "
                 "48-pair fine-grained description; permutation and bootstrap "
                 "seed 42."),
        "n_prompts": 8,
        "n_pairs": 48,
        "rho_48_pairs": round(float(rho48), 4),
        "cluster_ci_48": [round(ci48[0], 4), round(ci48[1], 4)],
        "cluster_mean_48": round(float(np.mean(boot48)), 4),
        "negative_fraction_48": round(neg_frac48, 4),
        "rho_8_prompts": round(float(rho8), 4),
        "permutation_p_two_sided": round(p_two, 4),
        "permutation_p_one_sided": round(p_one, 4),
        "bootstrap_ci_8": [round(ci8[0], 4), round(ci8[1], 4)],
        "mixed_model": mixed,
        "ols_clustered": ols_cl,
        "prompt_means": [{"prompt": int(pr), "BB_QWK": round(float(mx[i]), 4),
                          "dQWK": round(float(my[i]), 4)}
                         for i, pr in enumerate(range(1, 9))],
    }
    out_dir = RUNS / "prompt_level_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(
        json.dumps(stats_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved -> {out_dir / 'result.json'}")

    # ---- Figure: 48 pairs + 8 prompt-level means ----
    RED = '#F28E2B'
    BLUE = '#4E79A7'
    GRAY = '#444444'
    INK = '#444444'
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica'],
        'font.size': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    fig.subplots_adjust(left=0.11, right=0.97, top=0.96, bottom=0.13)

    pin = protocols == 'P-in'
    ax.scatter(xs[pin], ys[pin], c=RED, marker='o', s=42, alpha=0.65,
               edgecolors='white', linewidth=1.2, zorder=3, label='P-in (n=24)')
    ax.scatter(xs[~pin], ys[~pin], c=BLUE, marker='o', s=42, alpha=0.65,
               edgecolors='white', linewidth=1.2, zorder=3, label='P-LOPO (n=24)')
    ax.scatter(mx, my, c=GRAY, marker='D', s=80, edgecolors='white', linewidth=1.4,
               zorder=5, label='Prompt mean (n=8)')
    ax.axhline(0, color=GRAY, linewidth=0.8, linestyle='--', zorder=1)
    ax.set_xlabel('BlackBox QWK', fontsize=12)
    ax.set_ylabel('ΔQWK', fontsize=12)
    ax.set_xlim(0.12, 0.88)
    ax.set_ylim(-0.15, 0.15)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.text(0.02, 0.96,
            f'ρ(48 pairs) = {rho48:.2f}\n'
            f'ρ(8 prompts) = {rho8:.2f} (perm p = {p_two:.3f})',
            transform=ax.transAxes, ha='left', va='top', fontsize=11,
            fontweight='bold', color=INK)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(0.5)
    ax.tick_params(width=0.5)

    figdir = Path(__file__).resolve().parents[1]
    figdir = figdir / "paper_figures"
    out_png = figdir / "Figure 2.png"
    fig.savefig(str(out_png), dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"saved -> {out_png}")
    print(f"rho48={rho48:.3f}  rho8={rho8:.3f}  perm p2={p_two:.4f}  "
          f"ci8=[{ci8[0]:.3f},{ci8[1]:.3f}]")


if __name__ == "__main__":
    main()
