"""Figure 2 - LOPO per-fold audit.

Usage: python generate_fig2.py
"""

import json
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy as np
import os
from pathlib import Path

# Data: 8 folds x 3 seeds = 24 delta-QWK values read from runs (same source as
# Figure 3; nothing is hard-coded).
RUNS = Path(__file__).resolve().parents[2] / "results" / "runs"
SEEDS = [13, 42, 123]
deltas = []
for fold in range(1, 9):
    row = []
    for s in SEEDS:
        bb = json.loads((RUNS / f"B1_LOPO_feat_blackbox_s{s}_fold{fold}" / "result.json").read_text())["test_qwk"]
        hc = json.loads((RUNS / f"B1_LOPO_feat_h2tcbd_s{s}_fold{fold}" / "result.json").read_text())["test_qwk"]
        row.append(hc - bb)
    deltas.append(row)

all_d = [v for f in deltas for v in f]
overall_mean = np.mean(all_d)
overall_n = len(all_d)
print(f"loaded 24 fold delta-QWK, fold mean = {overall_mean:+.4f} (n = {overall_n})")

flat = [(v, fi+1, ['s13','s42','s123'][j]) for fi, f in enumerate(deltas) for j, v in enumerate(f)]

MARGIN = -0.02

# Colours.
BLUE  = '#0F4D92'
GRAY  = '#767676'
RED   = '#B64342'
INK   = '#000000'

# Global style.
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,
})

# Canvas.
fig, ax = plt.subplots(figsize=(6.2, 3.6))
fig.subplots_adjust(left=0.06, right=0.82, top=0.92, bottom=0.16)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.8)
ax.spines['bottom'].set_linewidth(0.8)

# Reference lines (both dashed).
ax.axvline(x=0, color=GRAY, linewidth=0.8, linestyle='--', zorder=1)
ax.axvline(x=MARGIN, color=RED, linewidth=1.0, linestyle='--', zorder=1)

# 24 points (red left of the margin, grey otherwise).
y_folds = [7, 6, 5, 4, 3, 2, 1, 0]
for fi in range(8):
    y = y_folds[fi]
    vals = sorted(enumerate(deltas[fi]), key=lambda x: x[1])
    offsets = [-0.30, 0, 0.30]
    for rank, (orig_j, d) in enumerate(vals):
        c = RED if d < MARGIN else GRAY
        ax.plot(d, y + offsets[rank],
                'o', color=c, markersize=4.5,
                markeredgecolor='white', markeredgewidth=0.3, zorder=5)

# Y axis: keep the spine, label folds manually.
ax.set_yticks([])
ax.spines['left'].set_visible(True)
ax.spines['left'].set_linewidth(0.8)
ax.spines['left'].set_color(INK)


# X axis.
ax.set_xlabel('Per-fold ΔQWK  (H2T-CBD − black-box), 8 prompts × 3 seeds',
              fontsize=12, labelpad=4)
ax.set_xlim(-0.22, 0.16)
ax.set_ylim(-0.8, 7.8)
ax.xaxis.set_major_locator(tck.MultipleLocator(0.05))
ax.tick_params(axis='x', labelsize=10, length=4, width=0.6)

# Annotations.
ax.text(MARGIN - 0.008, 7.65, '−0.02 margin',
        fontsize=10, color=RED, ha='right', va='top', fontstyle='italic')

# Dark-blue triangle marking the overall mean.
tri_y = 3.5
ax.plot(overall_mean, tri_y, 'v', color='#08306B', markersize=9,
        markeredgecolor='white', markeredgewidth=0.3, zorder=6)
ax.text(overall_mean + 0.012, tri_y, f'fold mean {overall_mean:+.4f}\n(n={overall_n})',
        fontsize=10, color=INK, ha='left', va='center',
        linespacing=1.6)

# Annotate the four worst points with generic, data-driven offsets.
worst_pts = sorted(flat, key=lambda x: x[0])[:4]          # [(d, fold, seed)]
worst_lines = ["failing:"]
for d, fi, seed in worst_pts:
    worst_lines.append(f"fold{fi} {seed}: {d:+.3f}")
ax.text(0.115, 7.25, "\n".join(worst_lines),
        fontsize=9, color=RED, ha='left', va='top',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor=RED, linewidth=0.8, alpha=0.92),
        zorder=11)

# No title (PeerJ: moved to the caption; single-panel scatter, no legend).

# Save.
outdir = str(Path(__file__).resolve().parents[1] / "paper_figures")
for ext in ['png']:
    path = os.path.join(outdir, f'fig2.png')
    fig.savefig(path, format=ext, dpi=400,
                bbox_inches='tight', pad_inches=0.25,
                facecolor='white', edgecolor='none')
    print(f'Saved → {path}')

plt.close()
print('Done.')
