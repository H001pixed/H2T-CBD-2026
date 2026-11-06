from pathlib import Path
import argparse
import csv
import re
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from scipy.stats import gaussian_kde
SRC = Path(__file__).resolve().parent / 'data/within_corpus_seed_pairs.csv'
NAMES = {f.name for f in fm.fontManager.ttflist}
EN_FONT = 'Arial' if 'Arial' in NAMES else 'DejaVu Sans'
ZH_FONT = 'SimHei' if 'SimHei' in NAMES else 'Microsoft YaHei'
INK, GRID = ('#222222', '#E6E6E6')
MEAN_FACE, MEAN_EDGE = ('#D4C6EC', '#8A73B5')
FS = 7.0
CORPORA = [('feat', 'ASAP-1'), ('asap2', 'ASAP-2.0')]
KEYS = ['blackbox', 'no_anchor', 'h2tcbd', 'featconcat']
LABELS = {'en': ['Black-box baseline', 'Unanchored model', 'H2T-CBD (full anchoring)', 'Feature-concatenation model']}
COLORS = ['#8C8C8C', '#8FB3D9', '#3B6EA5', '#3F9A86']

def fam(s):
    return EN_FONT

def load():
    rows = [r for r in csv.DictReader(open(SRC, encoding='utf-8-sig')) if r['protocol'] == 'P-in' and r['dataset'] in {c for c, _ in CORPORA} and (r['complete'] == 'True')]
    vals = defaultdict(dict)
    for r in rows:
        for side in ('baseline', 'comparison'):
            key = (r['dataset'], r[side])
            seed, q = (r['seed'], float(r['qwk_' + side]))
            if seed in vals[key]:
                assert abs(vals[key][seed] - q) < 1e-12
            vals[key][seed] = q
    for code, name in CORPORA:
        for i, k in enumerate(KEYS):
            v = list(vals[code, k].values())
            assert len(v) == 10, (name, k, len(v))
    return vals

def draw(lang, vals, folder):
    plt.rcParams.update({'font.family': [EN_FONT], 'font.size': FS, 'axes.linewidth': 0.7, 'xtick.major.width': 0.7, 'xtick.major.size': 2.5, 'ytick.major.size': 0, 'pdf.fonttype': 42, 'axes.unicode_minus': True})
    labels = LABELS[lang]
    fig, axes = plt.subplots(1, 2, figsize=(180 / 25.4, 70 / 25.4), sharey=True)
    fig.subplots_adjust(left=0.205, right=0.985, bottom=0.17, top=0.9, wspace=0.08)
    rng = np.random.default_rng(7)
    for ax, (code, name) in zip(axes, CORPORA):
        for s in ('top', 'right', 'left'):
            ax.spines[s].set_visible(False)
        allv = np.concatenate([list(vals[code, k].values()) for k in KEYS])
        lo, hi = (allv.min(), allv.max())
        pad = (hi - lo) * 0.22
        grid = np.linspace(lo - pad, hi + pad, 500)
        for i, (k, c) in enumerate(zip(KEYS, COLORS)):
            y = len(KEYS) - 1 - i
            v = np.array(sorted(vals[code, k].values()))
            dens = gaussian_kde(v, bw_method=0.6)(grid)
            dens = dens / dens.max() * 0.42
            keep = dens > 0.012
            ax.fill_between(grid, y + 0.06, y + 0.06 + dens, where=keep, color=c, alpha=0.32, lw=0)
            ax.plot(np.where(keep, grid, np.nan), y + 0.06 + dens, color=c, lw=0.9)
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            ax.plot([v.min(), v.max()], [y, y], color=c, lw=0.9, zorder=2)
            ax.add_patch(plt.Rectangle((q1, y - 0.05), q3 - q1, 0.1, fc='white', ec=c, lw=0.9, zorder=3))
            ax.plot([med, med], [y - 0.05, y + 0.05], color=c, lw=1.4, zorder=6)
            jit = rng.uniform(-0.3, -0.14, size=len(v))
            ax.scatter(v, y + jit, s=9, color=c, edgecolors='white', linewidths=0.4, zorder=4)
            ax.plot(np.mean(v), y, marker='o', ms=4.2, mfc=MEAN_FACE, mec=MEAN_EDGE, mew=0.8, zorder=4)
        ax.set_xlim(lo - pad, hi + pad)
        ax.xaxis.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.set_title(name, fontsize=FS + 0.8, color=INK, loc='center', pad=4)
        ax.set_xlabel('QWK', fontsize=FS, color=INK)
    axes[0].set_yticks(range(len(KEYS)))
    axes[0].set_yticklabels(labels[::-1], fontsize=FS, color=INK)
    for t, s in zip(axes[0].get_yticklabels(), labels[::-1]):
        t.set_fontfamily(fam(s))
    axes[0].set_ylim(-0.45, len(KEYS) - 0.45)
    stem = folder / ('fig2_raincloud' + '')
    fig.savefig(stem.with_suffix('.png'), dpi=600)
    fig.savefig(stem.with_suffix('.pdf'))
    plt.close(fig)
    print(stem)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', choices=['en'], default='en')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'output')
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parent / 'data')
    args = parser.parse_args()
    global SRC
    SRC = args.data_dir / 'within_corpus_seed_pairs.csv'
    folder = args.output_dir
    folder.mkdir(parents=True, exist_ok=True)
    vals = load()
    for lang in ['en', 'zh'] if args.lang == 'both' else [args.lang]:
        draw(lang, vals, folder)
if __name__ == '__main__':
    main()
