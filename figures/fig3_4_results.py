from pathlib import Path
import argparse
import csv
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
INK = '#222222'
BLUE = '#3B6EA5'
ORANGE = '#E08A3C'
GREY = '#9AA0A6'
BAND = '#F4F5F7'
FS = 7.0
NAMES = {f.name for f in fm.fontManager.ttflist}
EN_FONT = 'Arial' if 'Arial' in NAMES else 'DejaVu Sans'
ZH_FONT = 'SimHei' if 'SimHei' in NAMES else 'Microsoft YaHei'
MPL_FALLBACK = tuple((int(x) for x in matplotlib.__version__.split('.')[:2])) >= (3, 6)

def fam(s):
    return EN_FONT

def setup(lang):
    family = [EN_FONT]
    plt.rcParams.update({'font.family': family, 'font.size': FS, 'axes.unicode_minus': True, 'axes.linewidth': 0.7, 'xtick.major.width': 0.7, 'xtick.major.size': 2.5, 'ytick.major.size': 0, 'pdf.fonttype': 42, 'savefig.facecolor': 'white'})

def save(fig, folder, stem, lang):
    target = folder / (stem + '')
    fig.savefig(target.with_suffix('.png'), dpi=600)
    fig.savefig(target.with_suffix('.pdf'))
    plt.close(fig)
    print(target)
GROUPS_EN = ['Length and fluency', 'Lexical complexity', 'Syntactic complexity', 'Mechanics', 'Cohesion and readability', 'Rhetoric and engagement']
COLS_EN = ['ASAP-1\nleave-one-\nprompt-out', 'ASAP-2.0\nleave-one-\nprompt-out', 'ASAP-2.0\nofficial split', 'PERSUADE 2.0\nnon-overlapping subset\nleave-one-prompt-out', 'CELA\nsingle-prompt']

def fig3(lang, folder):
    setup(lang)
    prop = KEPT / np.array(TOTAL)[None, :]
    cmap = LinearSegmentedColormap.from_list('ret', ['#F7F9FC', '#A9C3DF', BLUE])
    fig = plt.figure(figsize=(180 / 25.4, 74 / 25.4))
    ax = fig.add_axes([0.23, 0.05, 0.61, 0.66])
    ax.imshow(prop, cmap=cmap, vmin=0, vmax=1, aspect='auto')
    for i in range(prop.shape[0]):
        for j in range(prop.shape[1]):
            ax.text(j, i, f'{KEPT[i, j]}/{TOTAL[j]}', ha='center', va='center', fontsize=FS, color='white' if prop[i, j] > 0.6 else INK, fontfamily=EN_FONT)
    ax.set_xticks(np.arange(prop.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(prop.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='white', lw=1.6)
    ax.tick_params(which='both', length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cols = COLS_EN
    rows = GROUPS_EN
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels([''] * len(cols))
    for j, c in enumerate(cols):
        ax.text(j, -0.75, c, ha='center', va='bottom', fontsize=FS, color=INK, fontfamily=fam(c), linespacing=1.15)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows, fontsize=FS, color=INK)
    for t, r in zip(ax.get_yticklabels(), rows):
        t.set_fontfamily(fam(r))
    cax = fig.add_axes([0.865, 0.05, 0.015, 0.66])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1)), cax=cax)
    cb.outline.set_visible(False)
    cb.set_ticks([0, 0.5, 1])
    cb.ax.tick_params(length=2, width=0.6, labelsize=FS)
    lab = 'Retention proportion'
    cb.set_label(lab, fontsize=FS, color=INK, fontfamily=fam(lab))
    save(fig, folder, 'fig3_ars_retention', lang)

def fig4(lang, folder):
    setup(lang)
    zh = lang == 'zh'
    fig = plt.figure(figsize=(180 / 25.4, 66 / 25.4))
    ax = fig.add_axes([0.3, 0.19, 0.66, 0.66])
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis='y', length=0)
    n = len(ROWS4)
    ys = np.arange(n)[::-1].astype(float)
    for k, y in enumerate(ys):
        if k % 2 == 0:
            ax.axhspan(y - 0.45, y + 0.45, color=BAND, lw=0, zorder=0)
    ax.axvline(0, color=GREY, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.axvline(-0.02, color=ORANGE, lw=1.0, ls=(0, (4, 2)), zorder=1)
    for y, row in zip(ys, ROWS4):
        _, m, lo, hi = row
        ax.plot([lo, hi], [y, y], color=BLUE, lw=1.3, solid_capstyle='butt', zorder=2)
        ax.plot(m, y, 'o', ms=4.6, mfc=BLUE, mec=BLUE, mew=1.1, zorder=3)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks(ys)
    labels = [r[0] for r in ROWS4]
    ax.set_yticklabels(labels, fontsize=FS, color=INK, linespacing=1.1)
    for t, s in zip(ax.get_yticklabels(), labels):
        t.set_fontfamily(fam(s))
    ax.set_xlim(-0.05, 0.05)
    ax.set_xticks([-0.04, -0.02, 0, 0.02, 0.04])
    ax.set_xticklabels(['−0.04', '−0.02', '0', '0.02', '0.04'])
    xl = 'QWK difference (selective minus full anchoring)'
    ax.set_xlabel(xl, fontsize=FS, color=INK, fontfamily=fam(xl), labelpad=3)
    leg = ['Mean difference and 95% interval', 'Bound of the non-inferiority criterion (−0.02)']
    handles = [Line2D([], [], color=BLUE, marker='o', ms=4.6, mfc=BLUE, mec=BLUE, lw=1.3), Line2D([], [], color=ORANGE, lw=1.0, ls=(0, (4, 2)))]
    lg = fig.legend(handles, leg, loc='upper center', bbox_to_anchor=(0.63, 1.0), ncol=2, frameon=False, fontsize=FS, handlelength=2.2, columnspacing=2.2)
    for t, s in zip(lg.get_texts(), leg):
        t.set_fontfamily([EN_FONT, ZH_FONT] if MPL_FALLBACK and fam(s) == ZH_FONT else fam(s))
    save(fig, folder, 'fig4_selective_vs_full', lang)


EVALUATION_LABELS = ['ASAP-1, leave-one-prompt-out', 'ASAP-2.0, leave-one-prompt-out', 'ASAP-2.0, official split', 'PERSUADE 2.0 non-overlapping subset,\nleave-one-prompt-out', 'CELA, single-prompt']

def load_data(data_dir):
    global KEPT, TOTAL, ROWS4
    with (data_dir / 'ars_retention.csv').open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 30
    KEPT = np.zeros((6, 5), dtype=int)
    TOTAL = np.zeros(5, dtype=int)
    seen = set()
    for row in rows:
        i, j = int(row['group_index']), int(row['evaluation_index'])
        assert (i, j) not in seen and 0 <= i < 6 and 0 <= j < 5
        seen.add((i, j))
        KEPT[i, j] = int(row['retained'])
        total = int(row['total'])
        assert TOTAL[j] in (0, total) and 0 <= KEPT[i, j] <= total
        TOTAL[j] = total
    with (data_dir / 'paired_scoring.csv').open(encoding='utf-8-sig', newline='') as handle:
        scores = sorted(csv.DictReader(handle), key=lambda row: int(row['evaluation_index']))
    assert [int(row['evaluation_index']) for row in scores] == list(range(5))
    ROWS4 = [(EVALUATION_LABELS[i], float(row['mean_difference']), float(row['ci_lower']), float(row['ci_upper']))
             for i, row in enumerate(scores)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', choices=['en'], default='en')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'output')
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parent / 'data')
    args = parser.parse_args()
    load_data(args.data_dir)
    folder = args.output_dir
    folder.mkdir(parents=True, exist_ok=True)
    for lang in ['en', 'zh'] if args.lang == 'both' else [args.lang]:
        fig3(lang, folder)
        fig4(lang, folder)
if __name__ == '__main__':
    main()
