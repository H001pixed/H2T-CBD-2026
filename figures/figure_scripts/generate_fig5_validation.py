"""Figure 5 - Selective-anchoring validation: fold-level QWK change + MMD.

Usage: python generate_fig5.py
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RUNS = Path(__file__).resolve().parents[2] / "results" / "runs"
SEEDS = [13, 42, 123]


def _qwk(exp_id):
    p = RUNS / exp_id / "result.json"
    if not p.exists():
        raise FileNotFoundError(f"missing run: {exp_id}")
    return json.loads(p.read_text(encoding="utf-8"))["test_qwk"]


# Panel A: per-fold mean delta-QWK (5-group minus 6-group) read from runs.
folds = [1, 2, 3, 4, 5, 6, 7, 8]
deltas = []
for fold in folds:
    q6 = np.mean([_qwk(f"B1_LOPO_feat_h2tcbd_s{s}_fold{fold}") for s in SEEDS])
    q5 = np.mean([_qwk(f"B1_5grp_rescaled_LOPO_feat_s{s}_fold{fold}") for s in SEEDS])
    deltas.append(q5 - q6)
def _effect(delta: float) -> str:
    if delta > 0:
        return 'Benefit'
    if delta < 0:
        return 'Hurt'
    return 'Neutral'


effects = [_effect(d) for d in deltas]
fold_labels = [f'F{f}\n({e[:4]})' for f, e in zip(folds, effects)]

# Panel B: per-seed mean MMD read from the Exp10 result.json.
MMD_RESULT = RUNS / "Exp10_MMD" / "result.json"
if not MMD_RESULT.exists():
    raise FileNotFoundError(
        "run analyze_representation_alignment.py first to produce "
        "runs/Exp10_MMD/result.json"
    )
_mmd = json.loads(MMD_RESULT.read_text(encoding="utf-8"))["summary"]
mmd_6 = np.array([_mmd["6groups"][str(s)]["mmd"] for s in SEEDS], dtype=float)
mmd_5 = np.array([_mmd["5groups"][str(s)]["mmd"] for s in SEEDS], dtype=float)
mmd_mean_6 = float(np.mean(mmd_6))
mmd_mean_5 = float(np.mean(mmd_5))
delta_mmd = mmd_mean_5 - mmd_mean_6

BLUE   = '#1A6DB5'
RED    = '#D64545'
ORANGE = '#E8923F'
GRAY   = '#3A3A3A'
PURPLE = '#6B3FA0'
DKGRY  = '#808080'
INK    = '#111111'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 12,
    'axes.labelsize': 11,
    'axes.titlesize': 14,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 5.5),
                                gridspec_kw={'width_ratios': [0.60, 0.40]})
fig.subplots_adjust(left=0.06, right=0.97, top=0.90, bottom=0.18, wspace=0.30)
# No figure title (PeerJ: moved to the caption).

# ---- Panel A ----
x = np.arange(len(folds))
bar_colors = [BLUE if e == 'Benefit' else RED if e == 'Hurt' else ORANGE for e in effects]

ax1.bar(x, deltas, 0.55, color=bar_colors, edgecolor='white', linewidth=1.2, zorder=3)

for i, d in enumerate(deltas):
    offset = 0.004 if d >= 0 else -0.004
    va = 'bottom' if d >= 0 else 'top'
    ax1.text(i + 0.008, d + offset, f'{d:+.3f}', ha='center', va=va,
             fontsize=10, fontweight='bold', color=INK)

ax1.axhline(0, color=INK, linewidth=1.0, linestyle='-', zorder=4)
# No legend (bar colours are described in the caption: blue=benefit,
# red=harmed, orange=neutral).

ax1.set_xticks(x)
ax1.set_xticklabels(fold_labels, fontsize=10)
ax1.set_ylabel('ΔQWK', fontsize=12)
# No subplot title.
ax1.set_ylim(-0.05, 0.08)

# ---- Panel B ----
bar_w = 0.32
p0, p1 = 0, 0.85

ax2.bar([p0], [mmd_mean_6], bar_w, color=DKGRY, edgecolor='white',
        linewidth=1.2, zorder=3)
ax2.bar([p1], [mmd_mean_5], bar_w, color=PURPLE, edgecolor='white',
        linewidth=1.2, zorder=3)

# Horizontal dashed lines from y-axis to x=1.0
ax2.hlines(mmd_mean_6, -0.4, 1.0, color=INK, linewidth=0.8, linestyle='--', zorder=5)
ax2.hlines(mmd_mean_5, -0.4, 1.0, color=INK, linewidth=0.8, linestyle='--', zorder=5)

# Vertical double-arrow + delta annotation
hi = max(mmd_mean_6, mmd_mean_5)
lo = min(mmd_mean_6, mmd_mean_5)
mid_x = (p0 + p1) / 2

ax2.annotate('', xy=(mid_x, lo), xytext=(mid_x, hi),
             arrowprops=dict(arrowstyle='<->', color=INK, lw=1.2), zorder=6)
ax2.text(mid_x + 0.06, (hi + lo) / 2, f'{delta_mmd:+.3f}', ha='left', va='center',
         fontsize=12, fontweight='bold', color=INK)

ax2.set_xticks([p0, p1])
ax2.set_xticklabels(['6 Groups', '5 Groups\n(Selective)'], fontsize=10)
ax2.set_ylabel('Cross-Prompt MMD²', fontsize=12)
# No subplot title or legend (the x-axis labels identify the two groups).
ax2.set_ylim(0, 0.75)
ax2.yaxis.set_major_locator(plt.MultipleLocator(0.10))
ax2.set_xlim(-0.4, 1.25)

for ax in [ax1, ax2]:
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(0.5)
    ax.tick_params(width=0.5)

fbase = Path(__file__).resolve().parents[1] / 'paper_figures' / 'Figure 4'
fig.savefig(str(fbase) + '.png', dpi=300, bbox_inches='tight')
plt.close()
print(f'Saved {fbase}.png')
