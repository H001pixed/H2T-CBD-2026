"""Figure 1: training and scoring process of H2T-CBD (English and Chinese versions).

Icon-based layout: essay sheet, frozen encoder, representation vector, perceptron,
candidate coordinates, scoring layer, score badge, features, ARS filter and decoders.
"""
from pathlib import Path
import argparse
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Circle, Rectangle, Arc
from matplotlib.lines import Line2D
W, H = (180, 84)
INK = '#222222'
SOFT = '#8C9297'
BLUE, BLUE_FILL, BLUE_MID = ('#3B6EA5', '#E7EEF6', '#9DB8D8')
TEAL, TEAL_FILL = ('#3F9A86', '#E4F2EE')
ORANGE, ORANGE_FILL = ('#E08A3C', '#FBEEDF')
GREY, GREY_FILL = ('#A3A9AE', '#F3F3F3')
FS = 6.6
MASKED = {4, 6}
NAMES = {f.name for f in fm.fontManager.ttflist}
EN_FONT = 'Arial' if 'Arial' in NAMES else 'DejaVu Sans'
ZH_FONT = 'SimHei' if 'SimHei' in NAMES else 'Microsoft YaHei'
MPL_FALLBACK = tuple((int(x) for x in matplotlib.__version__.split('.')[:2])) >= (3, 6)

def fam(s):
    return EN_FONT
FT = '\\tilde{\\mathbf{f}}'
TEXT = {'en': dict(essay=['Essay'], encoder=['Qwen3-Embedding-8B', '(parameters fixed)'], repr=['Neural', 'representation'], mlp=['Two-layer', 'perceptron'], coord_head='Candidate coordinates', coords=['$z_1$  Length and fluency', '$z_2$  Lexical complexity', '$z_3$  Syntactic complexity', '$z_4$  Mechanics', '$z_5$  Cohesion and readability', '$z_6$  Rhetoric and engagement'], scoring=['Scoring layer', 'Eq. (1)'], output=['Holistic score'], decoder=['Linear decoders', '$D_g$, Eq. (2)'], features=['19 features', 'six feature groups'], ars=['ARS', 'selects $S$'], targets='$' + FT + '_g$', legend=['Scoring path', 'Feature supervision (training only)', 'ARS selection (before training)', 'Not retained by ARS'])}
LS = {'solid': '-', 'dashed': (0, (2.6, 1.5))}
LH = FS * 0.3528 * 1.45

def label(ax, x, ytop, lines, color=INK, bold_first=False, size=FS):
    for i, s in enumerate(lines):
        ax.text(x, ytop - i * LH, s, ha='center', va='top', fontsize=size, color=color, fontfamily=fam(s), fontweight='bold' if bold_first and i == 0 else 'normal', zorder=4)

def rbox(ax, x0, y0, x1, y1, fc, ec, lw=0.8, ls='-', r=1.2, z=2):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f'round,pad=0,rounding_size={r}', fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))

def arrow(ax, pts, color, kind='solid', head=True, lw=0.9):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    if head:
        (xa, ya), (xb, yb) = (pts[-2], pts[-1])
        length = ((xb - xa) ** 2 + (yb - ya) ** 2) ** 0.5
        hl = min(1.5, length * 0.7)
        xc, yc = (xb - (xb - xa) / length * hl, yb - (yb - ya) / length * hl)
        xs[-1], ys[-1] = (xc, yc)
        ax.add_patch(FancyArrowPatch((xc, yc), (xb, yb), arrowstyle='-|>,head_length=0.62,head_width=0.3', mutation_scale=7, color=color, lw=lw, shrinkA=0, shrinkB=0, zorder=1.6))
    ax.add_line(Line2D(xs, ys, color=color, lw=lw, ls=LS[kind], solid_capstyle='butt', dash_capstyle='butt', zorder=1.5))

def essay_icon(ax, x0, y0, w, h):
    """Ruled sheet with margin, centred title, indented paragraphs and a pencil."""
    f = 3.2
    ax.add_patch(Polygon([(x0, y0), (x0 + w, y0), (x0 + w, y0 + h - f), (x0 + w - f, y0 + h), (x0, y0 + h)], closed=True, fc='white', ec=INK, lw=0.9, zorder=3, joinstyle='round'))
    ax.add_patch(Polygon([(x0 + w - f, y0 + h), (x0 + w - f, y0 + h - f), (x0 + w, y0 + h - f)], closed=True, fc=BLUE_FILL, ec=INK, lw=0.7, zorder=3.1, joinstyle='round'))
    step = 1.75
    rules = [y0 + h - 5.4 - k * step for k in range(9)]
    for y in rules:
        ax.add_line(Line2D([x0 + 0.7, x0 + w - 0.7], [y, y], color='#D3E0EE', lw=0.45, zorder=3.15))
    ax.add_line(Line2D([x0 + 2.3, x0 + 2.3], [y0 + 0.6, y0 + h - 0.6], color='#E5A0A0', lw=0.5, zorder=3.15))
    ax.add_line(Line2D([x0 + w * 0.3, x0 + w * 0.7], [y0 + h - 3.0] * 2, color=BLUE, lw=1.5, solid_capstyle='round', zorder=3.2))
    body = [(1.6, 0.92), (0, 1.0), (0, 0.55), (1.6, 0.92), (0, 1.0), (0, 0.72), (1.6, 0.92), (0, 0.4)]
    span = w - 4.0
    for (indent, L), y in zip(body, rules):
        xa = x0 + 3.0 + indent
        xb = x0 + 3.0 + (span - 0.9) * L
        ax.add_line(Line2D([xa, xb], [y + 0.42, y + 0.42], color='#5B636B', lw=0.75, solid_capstyle='round', zorder=3.2))
    d = np.array([np.cos(np.radians(52)), np.sin(np.radians(52))])
    n = np.array([-d[1], d[0]])
    tip = np.array([x0 + w - 4.2, y0 + 2.0])
    wd = 0.85
    cone = tip + d * 2.0
    body_end = tip + d * 8.6
    ferrule = tip + d * 9.4
    eraser = tip + d * 10.6
    ax.add_patch(Polygon([tip, cone + n * wd, cone - n * wd], closed=True, fc='#F2D7AE', ec=INK, lw=0.5, zorder=3.6, joinstyle='round'))
    ax.add_patch(Polygon([tip, tip + d * 0.75 + n * 0.32, tip + d * 0.75 - n * 0.32], closed=True, fc=INK, ec=INK, lw=0.3, zorder=3.7))
    ax.add_patch(Polygon([cone + n * wd, body_end + n * wd, body_end - n * wd, cone - n * wd], closed=True, fc=ORANGE, ec=INK, lw=0.5, zorder=3.6, joinstyle='round'))
    ax.add_patch(Polygon([body_end + n * wd, ferrule + n * wd, ferrule - n * wd, body_end - n * wd], closed=True, fc=GREY, ec=INK, lw=0.5, zorder=3.6))
    ax.add_patch(Polygon([ferrule + n * wd, eraser + n * wd, eraser - n * wd, ferrule - n * wd], closed=True, fc='#E5A0A0', ec=INK, lw=0.5, zorder=3.6, joinstyle='round'))

def encoder_icon(ax, x0, y0, w, h):
    rbox(ax, x0, y0, x0 + w, y0 + h, BLUE_FILL, BLUE, z=2)
    n, pad = (4, 2.2)
    sh = (h - 2 * pad - (n - 1) * 1.0) / n
    for k in range(n):
        y = y0 + pad + k * (sh + 1.0)
        rbox(ax, x0 + 2.4, y, x0 + w - 6.2, y + sh, BLUE_MID if k % 2 else BLUE, BLUE, lw=0.0, r=0.6, z=2.5)
    cx, cy = (x0 + w - 3.4, y0 + h / 2 - 1.2)
    ax.add_patch(Arc((cx, cy + 1.6), 2.4, 3.0, theta1=0, theta2=180, ec=INK, lw=0.8, zorder=3))
    ax.add_patch(FancyBboxPatch((cx - 1.6, cy - 1.4), 3.2, 3.0, boxstyle='round,pad=0,rounding_size=0.4', fc='white', ec=INK, lw=0.8, zorder=3.1))
    ax.add_patch(Circle((cx, cy + 0.15), 0.35, fc=INK, ec='none', zorder=3.2))

def vector_icon(ax, cx, y0, y1, n=7):
    hcell = (y1 - y0 - (n - 1) * 0.35) / n
    shades = [0.95, 0.55, 0.8, 0.35, 0.7, 0.45, 0.9]
    for k in range(n):
        y = y0 + k * (hcell + 0.35)
        ax.add_patch(Rectangle((cx - 1.6, y), 3.2, hcell, fc=BLUE, alpha=shades[k % len(shades)], ec='none', zorder=3))
    ax.add_patch(Rectangle((cx - 1.6, y0), 3.2, y1 - y0, fc='none', ec=BLUE, lw=0.7, zorder=3.1))

def mlp_icon(ax, x0, y0, w, h):
    rbox(ax, x0, y0, x0 + w, y0 + h, BLUE_FILL, BLUE, z=2)
    layers = [3, 4, 4, 3]
    xs = np.linspace(x0 + 2.6, x0 + w - 2.6, len(layers))
    pts = []
    for x, n in zip(xs, layers):
        ys = np.linspace(y0 + h / 2 - (n - 1) * 1.55, y0 + h / 2 + (n - 1) * 1.55, n)
        pts.append([(x, y) for y in ys])
    for a, b in zip(pts[:-1], pts[1:]):
        for p in a:
            for q in b:
                ax.add_line(Line2D([p[0], q[0]], [p[1], q[1]], color=BLUE, lw=0.35, alpha=0.55, zorder=2.6))
    for layer in pts:
        for x, y in layer:
            ax.add_patch(Circle((x, y), 0.85, fc='white', ec=BLUE, lw=0.7, zorder=3))

def sigma_icon(ax, cx, cy, r):
    ax.add_patch(Circle((cx, cy), r, fc=BLUE_FILL, ec=BLUE, lw=0.9, zorder=3))
    ax.text(cx, cy + 0.1, '$\\Sigma$', ha='center', va='center', fontsize=FS + 4.5, color=BLUE, zorder=4)

def badge_icon(ax, cx, cy, r):
    for sgn in (-1, 1):
        x = cx + sgn * r * 0.42
        ax.add_patch(Polygon([(x - 1.4, cy - r * 0.55), (x + 1.4, cy - r * 0.55), (x + 1.4, cy - r - 3.6), (x, cy - r - 2.6), (x - 1.4, cy - r - 3.6)], closed=True, fc=BLUE_MID, ec=BLUE, lw=0.6, zorder=2.8, joinstyle='round'))
    t = np.linspace(0, 2 * np.pi, 400)
    rr = r + 0.45 * np.cos(12 * t)
    ax.add_patch(Polygon(np.c_[cx + rr * np.cos(t), cy + rr * np.sin(t)], closed=True, fc=BLUE, ec=BLUE, lw=0.6, zorder=3))
    ax.add_patch(Circle((cx, cy), r * 0.74, fc='white', ec=BLUE, lw=0.6, zorder=3.1))
    ax.text(cx, cy, '$\\hat{y}$', ha='center', va='center', fontsize=FS + 4, color=BLUE, zorder=4)

def bars_icon(ax, x0, y0, w, h, color, dashed=False):
    heights = [0.55, 0.85, 0.7, 0.4, 0.95, 0.6]
    bw = w / (len(heights) * 1.5)
    for k, hh in enumerate(heights):
        x = x0 + k * bw * 1.5
        ax.add_patch(Rectangle((x, y0), bw, h * hh, fc='white' if dashed else color, ec=color, lw=0.7, ls=(0, (1.6, 0.9)) if dashed else '-', zorder=3))
    ax.add_line(Line2D([x0 - 0.6, x0 + w], [y0, y0], color=color, lw=0.7, zorder=3))

def funnel_icon(ax, cx, cy, s):
    ax.add_patch(Polygon([(cx - s, cy + s * 0.8), (cx + s, cy + s * 0.8), (cx + s * 0.22, cy - s * 0.1), (cx + s * 0.22, cy - s * 0.95), (cx - s * 0.22, cy - s * 0.7), (cx - s * 0.22, cy - s * 0.1)], closed=True, fc=ORANGE, ec=ORANGE, lw=0.6, zorder=3, joinstyle='round'))

def pill(ax, x0, y0, x1, y1, text, off):
    rbox(ax, x0, y0, x1, y1, GREY_FILL if off else BLUE_FILL, GREY if off else BLUE, ls=(0, (2.2, 1.2)) if off else '-', r=(y1 - y0) / 2.2)
    ax.text(x0 + 2.2, (y0 + y1) / 2, text, ha='left', va='center', fontsize=FS, color=SOFT if off else INK, fontfamily=fam(text), zorder=4)

def draw(lang, folder):
    T = TEXT[lang]
    family = [EN_FONT]
    plt.rcParams.update({'font.family': family, 'mathtext.fontset': 'custom', 'mathtext.rm': EN_FONT, 'mathtext.it': EN_FONT + ':italic', 'mathtext.bf': EN_FONT + ':bold', 'mathtext.fallback': 'stixsans', 'axes.unicode_minus': False, 'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
    fig = plt.figure(figsize=(W / 25.4, H / 25.4), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect('equal')
    ax.axis('off')
    ly = 81
    renderer = fig.canvas.get_renderer()
    mm_per_px = W / fig.bbox.width

    def text_w(txt):
        t = ax.text(0, 0, txt, fontsize=FS, fontfamily=fam(txt))
        w = t.get_window_extent(renderer).width * mm_per_px
        t.remove()
        return w
    sample, pad, sep = (7.0, 2.0, 7.0)
    widths = [sample + pad + text_w(txt) for txt in T['legend']]
    x = (W - (sum(widths) + sep * (len(widths) - 1))) / 2
    kinds = [('solid', BLUE), ('dashed', TEAL), ('solid', ORANGE), None]
    for item, wdt, txt in zip(kinds, widths, T['legend']):
        if item:
            arrow(ax, [(x, ly), (x + sample, ly)], item[1], item[0])
        else:
            rbox(ax, x + 0.5, ly - 1.5, x + sample - 0.5, ly + 1.5, GREY_FILL, GREY, ls=(0, (2.2, 1.2)), r=0.6)
        ax.text(x + sample + pad, ly, txt, ha='left', va='center', fontsize=FS, color=INK, fontfamily=fam(txt))
        x += wdt + sep
    yc = 62
    ex0, ey0, ew, eh = (2.5, 32, 15, 21)
    essay_icon(ax, ex0, ey0, ew, eh)
    label(ax, ex0 + ew / 2 + 2.6, ey0 - 1.2, T['essay'])
    arrow(ax, [(ex0 + ew / 2, ey0 + eh), (ex0 + ew / 2, yc), (23, yc)], BLUE)
    encoder_icon(ax, 23, yc - 7, 17, 14)
    label(ax, 31.5, yc - 8.6, T['encoder'])
    arrow(ax, [(40, yc), (51.4, yc)], BLUE)
    vector_icon(ax, 53.5, yc - 7, yc + 7)
    label(ax, 53.5, yc - 8.6, T['repr'])
    arrow(ax, [(55.6, yc), (63, yc)], BLUE)
    mlp_icon(ax, 63, yc - 7, 16, 14)
    label(ax, 71, yc - 8.6, T['mlp'])
    cx0, cx1, ctop, cbot, gap = (87, 122, 71, 19, 1.7)
    h = (ctop - cbot - 5 * gap) / 6
    centers = []
    for i in range(6):
        y1 = ctop - i * (h + gap)
        y0 = y1 - h
        pill(ax, cx0, y0, cx1, y1, T['coords'][i], i + 1 in MASKED)
        centers.append((y0 + y1) / 2)
    ax.text((cx0 + cx1) / 2, ctop + 2.4, T['coord_head'], ha='center', va='center', fontsize=FS, color=INK, fontweight='bold', fontfamily=fam(T['coord_head']))
    bus_l = 83
    ax.add_line(Line2D([79, bus_l], [yc, yc], color=BLUE, lw=0.9))
    ax.add_line(Line2D([bus_l, bus_l], [centers[0], centers[-1]], color=BLUE, lw=0.9))
    for c in centers:
        arrow(ax, [(bus_l, c), (cx0, c)], BLUE)
    bus_r = 125
    kept = [c for i, c in enumerate(centers) if i + 1 not in MASKED]
    for c in kept:
        ax.add_line(Line2D([cx1, bus_r], [c, c], color=BLUE, lw=0.9))
    ax.add_line(Line2D([bus_r, bus_r], [kept[-1], kept[0]], color=BLUE, lw=0.9))
    sx = 136
    arrow(ax, [(bus_r, yc), (sx - 5.6, yc)], BLUE)
    sigma_icon(ax, sx, yc, 5.4)
    label(ax, sx, yc - 6.6, T['scoring'])
    bx = 160
    arrow(ax, [(sx + 5.4, yc), (bx - 7.4, yc)], BLUE)
    badge_icon(ax, bx, yc + 1.2, 6.4)
    label(ax, bx, yc - 10.0, T['output'])
    dec_c = kept[-1]
    dx0, dx1 = (130, 163.5)
    rbox(ax, dx0, dec_c - 6.5, dx1, dec_c + 6.5, TEAL_FILL, TEAL)
    bars_icon(ax, dx0 + 2.3, dec_c - 3.8, 6.6, 7.0, TEAL, dashed=True)
    for i, s in enumerate(T['decoder']):
        ax.text(dx0 + 10.6, dec_c + LH / 2 - i * LH, s, ha='left', va='center', fontsize=FS, color=INK, fontfamily=fam(s), zorder=4)
    arrow(ax, [(bus_r, dec_c), (dx0, dec_c)], TEAL, 'dashed')
    fx0, fx1, fy0, fy1 = (25, 66, 7, 21)
    rbox(ax, fx0, fy0, fx1, fy1, TEAL_FILL, TEAL)
    bars_icon(ax, fx0 + 3, fy0 + 3, 9, 8, TEAL)
    ax.text(fx0 + 16, (fy0 + fy1) / 2 + LH / 2, T['features'][0], ha='left', va='center', fontsize=FS, color=INK, fontweight='bold', fontfamily=fam(T['features'][0]), zorder=4)
    ax.text(fx0 + 16, (fy0 + fy1) / 2 - LH / 2, T['features'][1], ha='left', va='center', fontsize=FS, color=INK, fontfamily=fam(T['features'][1]), zorder=4)
    arrow(ax, [(ex0 + 3.5, ey0), (ex0 + 3.5, (fy0 + fy1) / 2), (fx0, (fy0 + fy1) / 2)], TEAL, 'dashed')
    tx = 146
    arrow(ax, [(45.5, fy0), (45.5, 2.0), (tx, 2.0), (tx, dec_c - 6.5)], TEAL, 'dashed')
    ax.text(tx + 1.3, (dec_c - 6.5 + 2.0) / 2 + 1.5, T['targets'], ha='left', va='center', fontsize=FS, color=TEAL)
    ax0, ax1, ay0, ay1 = (89, 117, 4.5, 14)
    rbox(ax, ax0, ay0, ax1, ay1, ORANGE_FILL, ORANGE)
    funnel_icon(ax, ax0 + 5.2, (ay0 + ay1) / 2, 2.8)
    ax.text(ax0 + 10.5, (ay0 + ay1) / 2 + LH / 2, T['ars'][0], ha='left', va='center', fontsize=FS, color=INK, fontweight='bold', fontfamily=EN_FONT, zorder=4)
    ax.text(ax0 + 10.5, (ay0 + ay1) / 2 - LH / 2, T['ars'][1], ha='left', va='center', fontsize=FS, color=INK, fontfamily=fam(T['ars'][1]), zorder=4)
    arrow(ax, [(fx1, 10), (ax0, 10)], ORANGE)
    mx = 103
    arrow(ax, [(mx, ay1), (mx, cbot)], ORANGE)
    ax.text(mx + 1.3, (ay1 + cbot) / 2, '$m_g$', ha='left', va='center', fontsize=FS, color=ORANGE)
    stem = 'fig1_architecture' + ''
    target = folder / stem
    fig.savefig(target.with_suffix('.png'), dpi=600)
    fig.savefig(target.with_suffix('.pdf'))
    plt.close(fig)
    print(target, family)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', choices=['en'], default='en')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'output')
    args = parser.parse_args()
    folder = args.output_dir
    folder.mkdir(parents=True, exist_ok=True)
    for lang in ['en', 'zh'] if args.lang == 'both' else [args.lang]:
        draw(lang, folder)
if __name__ == '__main__':
    main()
