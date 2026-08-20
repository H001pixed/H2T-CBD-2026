"""Figure 1 - In-set accuracy + ablation (95% CI error bars).

Usage: python generate_fig1.py
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from scipy import stats

RUNS = Path(__file__).resolve().parents[2] / "results" / "runs"
SEEDS = [13, 42, 123]


def _round3(x):
    return int(x * 1000 + (0.5 if x >= 0 else -0.5)) / 1000.0


def _qwk(exp_id):
    p = RUNS / exp_id / "result.json"
    if not p.exists():
        raise FileNotFoundError(f"missing run: {exp_id}")
    return json.loads(p.read_text(encoding="utf-8"))["test_qwk"]


def _mean_ci(exp_ids):
    vals = np.array([_qwk(e) for e in exp_ids], dtype=float)
    m = float(vals.mean())
    ci = float(stats.t.ppf(0.975, df=len(vals) - 1) * vals.std(ddof=1) / np.sqrt(len(vals)))
    return m, ci


# Data read from runs (mean over 3 seeds; 95% CI from t_{0.975,df=2}).
datasets = ['ASAP-2.0', 'Feedback (dysf)', 'ASAP-1 (feat)']
bb_qwk, h2t_qwk, deltas, bb_ci, h2t_ci = [], [], [], [], []
for name in ("asap2", "dysf", "feat"):
    bm, bci = _mean_ci([f"B1_Pin_{name}_blackbox_s{s}" for s in SEEDS])
    hm, hci = _mean_ci([f"B1_Pin_{name}_h2tcbd_s{s}" for s in SEEDS])
    bb_qwk.append(bm); bb_ci.append(bci)
    h2t_qwk.append(hm); h2t_ci.append(hci)
    deltas.append(hm - bm)

ablation_labels = ['BlackBox', 'FeatConcat', 'No Anchor', 'H2T-CBD']
ablation_qwk, ablation_ci = [], []
for exp_ids in ([f"B1_Pin_feat_blackbox_s{s}" for s in SEEDS],
                [f"B1_Abl_Pin_feat_featconcat_s{s}" for s in SEEDS],
                [f"B1_Abl_Pin_feat_no_anchor_s{s}" for s in SEEDS],
                [f"B1_Pin_feat_h2tcbd_s{s}" for s in SEEDS]):
    m, c = _mean_ci(exp_ids)
    ablation_qwk.append(m); ablation_ci.append(c)

# Colours.
BLUE     = '#4E79A7'
RED      = '#F28E2B'
DARKGRAY = '#76B7B2'
GRAY     = '#76B7B2'
INK      = '#111111'
WHITE    = '#FFFFFF'

# Global style.
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,
})

# Canvas (PeerJ: no titles/legends inside the figure; tightened top margin).
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 5.5),
                                gridspec_kw={'width_ratios': [0.95, 1.05]})
fig.subplots_adjust(left=0.06, right=0.86, top=0.90, bottom=0.11, wspace=0.25)

for ax in [ax1, ax2]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.tick_params(labelsize=10)

ylim_lo, ylim_hi = 0.0, 1.0
bar_w = 0.50

# Left panel.
x = np.arange(3)
w = 0.42

bars1 = ax1.bar(x - w/2, bb_qwk,  w, color=BLUE, edgecolor='white', linewidth=0.5, zorder=3)
bars2 = ax1.bar(x + w/2, h2t_qwk, w, color=RED,  edgecolor='white', linewidth=0.5, zorder=3)

for i in range(3):
    # Error bars (no cap text).
    ax1.errorbar(x[i] - w/2, bb_qwk[i],  yerr=bb_ci[i],
                 fmt='none', ecolor=INK, capsize=3, capthick=0.8, linewidth=0.8, zorder=6)
    ax1.errorbar(x[i] + w/2, h2t_qwk[i], yerr=h2t_ci[i],
                 fmt='none', ecolor=INK, capsize=3, capthick=0.8, linewidth=0.8, zorder=6)

    # White bold mean labels inside the bottom of the bars.
    ax1.text(x[i] - w/2, ylim_lo + 0.025, f'{bb_qwk[i]:.3f}',
             fontsize=10, ha='center', va='bottom', color=WHITE, fontweight='heavy')
    ax1.text(x[i] + w/2, ylim_lo + 0.025, f'{h2t_qwk[i]:.3f}',
             fontsize=10, ha='center', va='bottom', color=WHITE, fontweight='heavy')

    # Delta in red.
    taller_top = max(bb_qwk[i] + bb_ci[i], h2t_qwk[i] + h2t_ci[i])
    ax1.text(x[i], taller_top + 0.012, f'Δ = {_round3(deltas[i]):+.3f}',
             fontsize=10, ha='center', va='bottom', color=INK, fontweight='bold')

ax1.set_xticks(x)
ax1.set_xticklabels(datasets)
ax1.set_ylabel('In-set QWK (mean over 3 seeds)', fontsize=10)
ax1.set_ylim(ylim_lo, ylim_hi)
ax1.yaxis.set_major_locator(plt.MultipleLocator(0.20))
# No subplot title or legend (PeerJ: moved to the caption).

# Right panel (bar width close to the left panel).
x2 = np.arange(4)
colors_abl = [BLUE, DARKGRAY, DARKGRAY, RED]

ax2.bar(x2, ablation_qwk, bar_w, color=colors_abl, edgecolor='white', linewidth=0.5, zorder=3)

for i in range(4):
    ax2.errorbar(x2[i], ablation_qwk[i], yerr=ablation_ci[i],
                 fmt='none', ecolor=INK, capsize=3, capthick=0.8, linewidth=0.8, zorder=6)

# White bold value labels inside the bottom of the bars.
    ax2.text(x2[i], ylim_lo + 0.025, f'{ablation_qwk[i]:.3f}',
             fontsize=10, ha='center', va='bottom', color=WHITE, fontweight='heavy')

    # Δ
    if i > 0:
        d = ablation_qwk[i] - ablation_qwk[0]
        ax2.text(x2[i], ablation_qwk[i] + ablation_ci[i] + 0.012, f'Δ = {_round3(d):+.3f}',
                 fontsize=10, ha='center', va='bottom', color=INK, fontweight='bold')

ax2.set_xticks(x2)
ax2.set_xticklabels(ablation_labels)
ax2.set_ylim(ylim_lo, ylim_hi)
ax2.yaxis.set_major_locator(plt.MultipleLocator(0.20))
ax2.set_ylabel('In-set QWK (mean over 3 seeds)', fontsize=10)
# No subplot title, figure title, or footer note (PeerJ: moved to the caption).

# Save.
outdir = str(Path(__file__).resolve().parents[1] / "paper_figures")
for ext in ['png']:
    path = os.path.join(outdir, f'Figure 1.png')
    fig.savefig(path, format=ext, dpi=300,
                bbox_inches='tight', pad_inches=0.25,
                facecolor='white', edgecolor='none')
    print(f'Saved → {path}')

plt.close()
print('Done.')
