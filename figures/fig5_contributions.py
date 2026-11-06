from pathlib import Path
import argparse
import os, sys, math, csv, warnings
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D
names = {f.name for f in fm.fontManager.ttflist}
EN_FONT = 'Arial' if 'Arial' in names else 'DejaVu Sans'
ZH_FONT = 'SimHei' if 'SimHei' in names else 'Microsoft YaHei' if 'Microsoft YaHei' in names else None
MPL_FALLBACK = tuple((int(x) for x in matplotlib.__version__.split('.')[:2])) >= (3, 6)
parser = argparse.ArgumentParser(description='Figure 5: signed coordinate contributions.')
parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'output')
parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parent / 'data')
args = parser.parse_args()
SRC = args.data_dir / 'case_contributions.csv'
OUT = str(args.output_dir)
os.makedirs(OUT, exist_ok=True)
rows = list(csv.DictReader(open(SRC, encoding='utf-8-sig')))
order = {'high': 0, 'mid': 1, 'low': 2}
rows.sort(key=lambda r: order[r['role']])
src_groups = ['length_fluency', 'lexical_complexity', 'syntactic_complexity', 'cohesion_readability']
data = []
for r in rows:
    c = [float(r[g]) for g in src_groups]
    b = float(r['intercept'])
    t = b + sum(c)
    u = 1.0 / (1.0 + math.exp(-t))
    assert abs(u - float(r['pred_unit'])) < 0.0001
    data.append(dict(h=int(float(r['human_score'])), m=int(float(r['model_score'])), w=int(r['word_count']), c=c, t=t, u=u))
LINE = '#3D3D3D'
COLORS = ['#3B6EA5', '#3F9A86', '#E08A3C']
GRID = '#DADADA'
ZERO = '#3C3C3C'
INK, GREY = ('#1A1A1A', '#707070')
RMIN, RMAX = (-1.0, 1.75)
RTICKS = [-0.5, 0.0, 0.5, 1.0, 1.5]
THETA = np.deg2rad([45, 315, 225, 135])
W_MM, H_MM = (183.0, 60.0)
R_MM = 17.0
CX_MM = [34.5, 92.5, 150.5]
CY_MM = 27.0

def fmt(v, nd=2):
    s = f'{v:.{nd}f}'
    if float(s) == 0:
        s = '0'
    return s.replace('-', '−')

def draw(lang):
    zh = lang == 'zh'
    main_font = EN_FONT
    plt.rcParams.update({'font.family': [EN_FONT], 'font.size': 6.5, 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'mathtext.fontset': 'custom', 'mathtext.rm': EN_FONT, 'mathtext.it': EN_FONT + ':italic', 'mathtext.default': 'regular', 'axes.unicode_minus': False})
    labels = ['Length and\nfluency', 'Lexical\ncomplexity', 'Syntactic\ncomplexity', 'Cohesion and\nreadability']
    title = lambda d: f"Human {d['h']} / model {d['m']} · {d['w']} words"
    leg = ['Positive contribution', 'Negative contribution', 'Zero contribution']
    fig = plt.figure(figsize=(W_MM / 25.4, H_MM / 25.4))
    full = np.linspace(0, 2 * np.pi, 361)
    for k, d in enumerate(data):
        rect = [(CX_MM[k] - R_MM) / W_MM, (CY_MM - R_MM) / H_MM, 2 * R_MM / W_MM, 2 * R_MM / H_MM]
        ax = fig.add_axes(rect, projection='polar')
        ax.set_theta_zero_location('E')
        ax.set_theta_direction(1)
        ax.set_ylim(RMIN, RMAX)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.grid(False)
        ax.spines['polar'].set_visible(False)
        ax.patch.set_alpha(0)
        for rr in RTICKS + [RMAX]:
            if rr != 0.0:
                ax.plot(full, np.full_like(full, rr), color=GRID, lw=0.45, zorder=1)
        for th in THETA:
            ax.plot([th, th], [RMIN, RMAX], color=GRID, lw=0.45, zorder=1)
        ax.plot(full, np.zeros_like(full), color=ZERO, lw=0.8, ls=(0, (3.2, 1.8)), zorder=2)
        th = np.append(THETA, THETA[0])
        cv = np.append(d['c'], d['c'][0])
        ax.fill(th, cv, color=COLORS[k], alpha=0.16, zorder=3, lw=0)
        ax.plot(th, cv, color=COLORS[k], lw=1.15, zorder=4, solid_joinstyle='round')
        for ang, v in zip(THETA, d['c']):
            face = COLORS[k] if v >= 0 else 'white'
            ax.plot(ang, v, 'o', ms=3.6, mfc=face, mec=COLORS[k], mew=1.0, zorder=5)
        for rr in RTICKS if k == 0 else []:
            ax.text(np.deg2rad(90), rr, fmt(rr, 1), fontsize=5.2, color=ZERO if rr == 0 else GREY, fontweight='bold' if rr == 0 else 'normal', ha='center', va='center', zorder=6, bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=0.85))
        for ang, lab in zip(THETA, labels):
            deg = round(np.rad2deg(ang)) % 360
            ha = {45: 'left', 315: 'left', 225: 'right', 135: 'right'}[deg]
            va = {45: 'bottom', 135: 'bottom', 225: 'top', 315: 'top'}[deg]
            ax.text(ang, RMAX + 0.12, lab, ha=ha, va=va, fontsize=6.3, color=INK, linespacing=1.1)
        x0 = (CX_MM[k] - R_MM - 9.0) / W_MM
        fig.text(x0, 0.94, 'abc'[k], fontsize=8, fontweight='bold', color=INK, ha='left', va='center', family=EN_FONT)
        fig.text(x0 + 3.6 / W_MM, 0.94, title(d), fontsize=6.8, color=COLORS[k], ha='left', va='center')
        fig.text(x0 + 3.6 / W_MM, 0.875, '$\\mathit{t}$ = ' + fmt(d['t']) + ', $\\mathit{u}$ = ' + fmt(d['u']), fontsize=6.2, color=GREY, ha='left', va='center')
    handles = [Line2D([], [], marker='o', ls='none', ms=3.3, mfc=LINE, mec=LINE, mew=0.9), Line2D([], [], marker='o', ls='none', ms=3.3, mfc='white', mec=LINE, mew=0.9), Line2D([], [], color=ZERO, lw=0.8, ls=(0, (3.2, 1.8)))]
    fig.legend(handles, leg, loc='lower center', ncol=3, frameon=False, fontsize=6.3, handlelength=2.0, columnspacing=2.4, bbox_to_anchor=(0.5, 0.0))
    base = os.path.join(OUT, 'fig5_contributions' + '')
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        fig.savefig(base + '.pdf')
        fig.savefig(base + '.png', dpi=600)
        missing = [str(x.message) for x in w if 'glyph' in str(x.message)]
    plt.close(fig)
    print('saved', lang, 'glyph warnings:', len(missing))
draw('en')
