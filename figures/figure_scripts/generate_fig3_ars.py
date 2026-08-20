"""Figure 3 - ARS (Anchor Robustness Score) bar chart with bootstrap CI bars.

ARS values are computed from data.csv with the same procedure as
recompute_ars.py (no hard-coding), so the figure always matches the data.
The error bars are the prompt-level bootstrap 95% CIs from
ars_bootstrap_ci.py (results/runs/ars_bootstrap/ars_bootstrap_ci.csv).
Usage: python generate_fig3_ars.py
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

# ---- Compute ARS from data.csv using the recompute_ars.py procedure ----
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "01_raw_datasets" / "feat" / "data.csv"
FEATURES = [
    "char_count", "word_count", "sent_count", "avg_word_len", "spell_err_count",
    "noun_count", "adj_count", "verb_count", "adv_count", "readability_score",
    "punctuation_score", "vocabulary_richness", "complex_sentence_ratio",
    "clause_density", "semantic_coherence", "sentiment_subjectivity",
    "transitional_phrase_use", "figurative_language_use", "question_usage",
]
ANCHOR_GROUPS = {
    "length_fluency": ["char_count", "word_count", "sent_count"],
    "lexical_sophistication": ["avg_word_len", "vocabulary_richness", "adj_count", "adv_count"],
    "syntactic_complexity": ["complex_sentence_ratio", "clause_density", "verb_count", "noun_count"],
    "mechanics": ["spell_err_count", "punctuation_score"],
    "coherence_readability": ["readability_score", "semantic_coherence"],
    "rhetoric_engagement": ["sentiment_subjectivity", "transitional_phrase_use",
                            "figurative_language_use", "question_usage"],
}
df = pd.read_csv(DATA_PATH)
per_feat = {}
for fold in sorted(df["essay_set"].unique()):
    sub = df[df["essay_set"] == fold]
    scores = sub["final_score"].values
    d = {}
    for feat in FEATURES:
        vals = sub[feat].values
        mask = np.isfinite(vals)
        if mask.sum() < 10 or np.std(vals[mask]) == 0:
            d[feat] = np.nan
            continue
        r, _ = stats.spearmanr(vals[mask], scores[mask])
        d[feat] = abs(r)
    per_feat[fold] = d
prompts = sorted(df["essay_set"].unique())

labels = [
    'Length &\nFluency',
    'Lexical\nSophistication',
    'Syntactic\nComplexity',
    'Mechanics',
    'Coherence &\nReadability',
    'Rhetorical\nEngagement',
]
ars = []
for gname, feats in ANCHOR_GROUPS.items():
    rho_bar = []
    for p in prompts:
        rs = [per_feat[p][f] for f in feats if not np.isnan(per_feat[p][f])]
        rho_bar.append(float(np.mean(rs)))
    ars.append(float(np.mean(rho_bar)) - float(np.std(rho_bar)))
ars = [round(a, 3) for a in ars]
THRESHOLD = 0.1

# 95% CI from the prompt-level bootstrap (ars_bootstrap_ci.py).
BOOT_CSV = Path(__file__).resolve().parents[2] / "results" / "runs" / \
    "ars_bootstrap" / "ars_bootstrap_ci.csv"
ci_df = pd.read_csv(BOOT_CSV)
ci_map = dict(zip(ci_df["group"], zip(ci_df["ci_lo"], ci_df["ci_hi"])))
ci_lo = np.array([ci_map[g][0] for g in ANCHOR_GROUPS])
ci_hi = np.array([ci_map[g][1] for g in ANCHOR_GROUPS])

BLUE      = '#4E79A7'
RED       = '#CC3311'
GRAY      = '#444444'
INK       = '#444444'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 16,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
})

fig, ax = plt.subplots(figsize=(7.5, 4.2))
fig.subplots_adjust(left=0.10, right=0.96, top=0.94, bottom=0.32)
# No figure title (PeerJ: moved to the caption).

x = np.arange(len(labels))
bar_colors = [BLUE if a > THRESHOLD else RED for a in ars]

bars = ax.bar(x, ars, 0.55, color=bar_colors, edgecolor='white',
              linewidth=1.2, zorder=3,
              yerr=np.vstack([ars - ci_lo, ci_hi - ars]),
              error_kw=dict(ecolor=INK, elinewidth=1.0, capsize=3.0))

# ARS values above the upper error-bar caps (breathing room for the CI ticks)
for i, a in enumerate(ars):
    ax.text(i, ci_hi[i] + 0.022, f'{a:.3f}', ha='center', va='bottom',
            fontsize=10, fontweight='bold', color=INK)

# Threshold line
ax.axhline(THRESHOLD, color=GRAY, linewidth=1.0, linestyle='--', zorder=2)
ax.text(len(labels) - 0.6, THRESHOLD + 0.012, f'θ = {THRESHOLD}',
        fontsize=12, color=INK, ha='left', va='bottom', fontweight='bold')

# Legend at the upper right (empty area: the tallest bar is on the left and the
# right-hand bars are only 0.110/0.070, so the legend does not overlap).
legend_handles = [
    mpatches.Patch(color=BLUE, label='Retain (ARS > 0.1)'),
    mpatches.Patch(color=RED, label='Drop (ARS ≤ 0.1)'),
]
ax.legend(handles=legend_handles, loc='upper right', fontsize=10, framealpha=0.8)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel('ARS', fontsize=12)
ax.set_ylim(0, 0.78)

for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
for spine in ['left', 'bottom']:
    ax.spines[spine].set_linewidth(0.5)
ax.tick_params(width=0.5)

fbase = Path(__file__).resolve().parents[1] / 'paper_figures' / 'Figure 3'
fig.savefig(str(fbase) + '.png', dpi=600, bbox_inches='tight')
plt.close()
print(f'Saved {fbase}.png  ARS = {ars}')
