"""Fig. 1: study design (a) and the agreement-gated system as a modular schematic (b).

Layout is drawn in millimetre coordinates on a 174 mm wide canvas. Semantic colours: blue = expert L, orange = expert G61,
purple = agreement system, green = healthy, grey = referral; neutral grey for design scaffolding. Panel b traces one real
external image (Roboflow new-wheat-disease v2, CC BY 4.0; the same image as Fig. 4e) through the system using its cached
expert outputs, and shows the distribution of the 860 external images over the four outputs (ms_revision.json).
Exports PDF/SVG (editable vector) and 600-dpi PNG/TIFF.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle, Circle  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REV = json.loads((HERE.parent / 'analysis' / 'ms_revision.json').read_text())
EXAMPLE = ROOT.parent / 'shared/datasets/extra_staging_c/targeted_rust_20260927/new-wheat-disease-v2/train/stripe145_jpg.rf.eee2edbfdf2344d04fdb64d6269b7684.jpg'
MM = 1 / 25.4
W, H = 174, 156
plt.rcParams.update({'font.family': 'Arial', 'font.size': 7, 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'mathtext.default': 'regular'})

INK, MUTED, FAINT = '#1d1c1a', '#5d5c58', '#9a9993'
L_C, L_BG = '#2a78d6', '#eaf2fc'
G_C, G_BG = '#e0602a', '#fdefe7'
S_C, S_BG, S_MID = '#4a3aa7', '#efedf9', '#8577e0'
H_C = '#1baf7a'
R_C, R_BG = '#9a9993', '#f2f1ee'
ZONE = '#f7f7f5'


def card(ax, x, y, w, h, fc='white', ec=FAINT, lw=0.8, ls='-', r=1.6, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))


def header(ax, x, y, w, text, color, h=5.2):
    ax.add_patch(FancyBboxPatch((x, y - h), w, h, boxstyle='round,pad=0,rounding_size=1.6', fc=color, ec='none', zorder=3))
    ax.add_patch(Rectangle((x, y - h), w, h / 2, fc=color, ec='none', zorder=3))
    ax.text(x + 2, y - h / 2, text, color='white', fontsize=7.4, fontweight='bold', va='center', ha='left', zorder=4)


def arrow(ax, a, b, color=MUTED, lw=0.9, ls='-', rad=0.0, z=5, ms=7):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=ms, lw=lw, color=color, ls=ls,
                                 connectionstyle=f'arc3,rad={rad}', shrinkA=0, shrinkB=0, zorder=z))


def chip(ax, x, y, text, color, fs=6.6):
    ax.text(x, y, text, fontsize=fs, color='white', fontweight='bold', ha='left', va='center', zorder=6,
            bbox=dict(boxstyle='round,pad=0.25,rounding_size=0.6', fc=color, ec='none'))


def zone(ax, x, y, w, h, title):
    card(ax, x, y, w, h, fc=ZONE, ec='#e3e2dd', lw=0.6, r=2.2, z=0)
    ax.text(x + 2.2, y + h - 2.6, title, fontsize=6.8, color=MUTED, fontweight='bold', va='center', ha='left')


def panel_a(ax):
    ax.text(1, H - 2, 'a', fontweight='bold', fontsize=9, va='top')
    ax.text(5, H - 2.1, 'Study design', fontweight='bold', fontsize=8.5, va='top', color=INK)
    y0, h = 125, 19
    phases = [('1', 'Train expert L', '2,616 public images per fold\n(WFD2020 other folds, CerealConv,\nMendeley leaf, iNaturalist)', L_C),
              ('2', 'Develop and freeze', 'EVAL450: 450 images, 8 sources\nsource-held nested CV; rule,\nthresholds and prompt frozen', S_C),
              ('3', 'External tests', 'iNat 68 · ETS 552 · ETS2 240\nde-duplicated, rules time-\nstamped, each evaluated once', INK),
              ('4', 'Retrospective', '1,310 images, 16 sources\nleave-one-source-out fusion\nand K-shot rule selection', R_C)]
    x, w, d = 2, 41.5, 4.0
    for k, (num, title, body, col) in enumerate(phases):
        pts = [(x, y0), (x + w - d, y0), (x + w, y0 + h / 2), (x + w - d, y0 + h), (x, y0 + h)]
        if k:
            pts.append((x + d, y0 + h / 2))
        dashed = num == '4'
        ax.add_patch(Polygon(pts, closed=True, fc='white' if not dashed else R_BG, ec=col, lw=1.0, ls='--' if dashed else '-', zorder=2))
        cx = x + (d if k else 0) + 4.2
        ax.add_patch(Circle((cx, y0 + h - 4.2), 2.4, fc=col, ec='none', zorder=3))
        ax.text(cx, y0 + h - 4.2, num, color='white', fontsize=7, fontweight='bold', ha='center', va='center', zorder=4)
        ax.text(cx + 3.6, y0 + h - 4.2, title, fontsize=7.6, fontweight='bold', va='center', color=INK)
        ax.text(x + (d if k else 0) + 2.4, y0 + h - 8.6, body, fontsize=6.4, va='top', color=INK, linespacing=1.25)
        x += w + 1.2
    ax.plot([2, 128.6], [121.5, 121.5], color=S_C, lw=1.3, solid_capstyle='butt')
    ax.text(65, 121.5, ' prespecified, in time order ', fontsize=6.6, color=S_C, style='italic', ha='center', va='center',
            bbox=dict(fc='white', ec='none', pad=0.3))
    ax.plot([129.8, 172], [121.5, 121.5], color=R_C, lw=1.3, ls=(0, (3, 1.6)))
    ax.text(151, 121.5, ' after all tests ', fontsize=6.6, color=MUTED, style='italic', ha='center', va='center',
            bbox=dict(fc='white', ec='none', pad=0.3))


def panel_b(ax):
    top = 114
    ax.text(1, top + 1.5, 'b', fontweight='bold', fontsize=9, va='top')
    ax.text(5, top + 1.4, 'Agreement-gated hierarchical diagnosis', fontweight='bold', fontsize=8.5, va='top', color=INK)
    zt, zb = top - 6, 2
    zone(ax, 1, zb, 25, zt - zb, 'INPUT')
    zone(ax, 28, zb, 50, zt - zb, 'HETEROGENEOUS EXPERTS')
    zone(ax, 80, zb, 56, zt - zb, 'AGREEMENT GATES')
    zone(ax, 138, zb, 35, zt - zb, 'OUTPUT (860 IMAGES)')

    # input: real example image
    im = ImageOps.exif_transpose(Image.open(EXAMPLE)).convert('RGB'); w0, h0 = im.size; s = min(w0, h0)
    im = im.crop(((w0 - s) // 2, (h0 - s) // 2, (w0 - s) // 2 + s, (h0 - s) // 2 + s)).resize((500, 500))
    ix, iy, isz = 3.0, 55, 21
    ax.imshow(im, extent=(ix, ix + isz, iy, iy + isz), zorder=3)
    ax.add_patch(Rectangle((ix, iy), isz, isz, fc='none', ec=MUTED, lw=0.6, zorder=4))
    ax.text(ix + isz / 2, iy - 1.6, 'single RGB photo\nno source metadata', fontsize=6.6, ha='center', va='top', color=INK, linespacing=1.2)
    ax.text(ix + isz / 2, iy - 9.4, 'example: ETS image,\nlabel stripe rust', fontsize=6.4, ha='center', va='top', color=MUTED, style='italic', linespacing=1.2)

    # expert L card
    lx, ly, lw_, lh = 30, 57, 46, 45
    card(ax, lx, ly, lw_, lh, ec=L_C, lw=1.0); header(ax, lx, ly + lh, lw_, 'Expert L · vision foundation model', L_C)
    gx, gy = lx + 2.5, ly + lh - 15.5  # architecture glyph: patches -> transformer -> head
    for i in range(4):
        for j in range(4):
            ax.add_patch(Rectangle((gx + i * 1.9, gy + j * 1.9), 1.6, 1.6, fc=L_BG, ec=L_C, lw=0.4, zorder=3))
    arrow(ax, (gx + 8.2, gy + 3.7), (gx + 10.4, gy + 3.7), L_C, lw=0.7, ms=5)
    for k in range(4):
        ax.add_patch(FancyBboxPatch((gx + 11 + k * 1.3, gy + 0.2 + k * 0.5), 9.5, 6.4, boxstyle='round,pad=0,rounding_size=0.6',
                                    fc='white' if k < 3 else L_BG, ec=L_C, lw=0.5, zorder=3 + k))
    ax.text(gx + 20.6, gy + 4.6, 'DINOv3\nViT-B/16', fontsize=6.2, ha='center', va='center', color=INK, zorder=8, linespacing=1.1)
    ax.text(gx + 16.5, gy - 1.3, 'fine-tuned · 9-model ensemble', fontsize=6.2, ha='center', va='top', color=MUTED)
    arrow(ax, (gx + 26.5, gy + 3.7), (gx + 28.7, gy + 3.7), L_C, lw=0.7, ms=5)
    ax.add_patch(FancyBboxPatch((gx + 29.2, gy + 1.2), 9.6, 5, boxstyle='round,pad=0,rounding_size=0.6', fc=L_BG, ec=L_C, lw=0.5, zorder=3))
    ax.text(gx + 34, gy + 3.7, '6 sigmoid', fontsize=6.1, ha='center', va='center', color=INK, zorder=4)
    # probability bars of the example
    probs = [('healthy', 0.0), ('leaf rust', 0.034), ('powdery m.', 0.0), ('septoria', 0.0), ('stem rust', 0.001), ('stripe rust', 0.998)]
    by0, bh, bx0, bwmax = ly + 2.0, 2.3, lx + 15, 24
    for k, (n, p) in enumerate(probs[::-1]):
        yy = by0 + k * (bh + 0.4)
        ax.text(bx0 - 1, yy + bh / 2, n, fontsize=6.3, ha='right', va='center', color=INK)
        ax.add_patch(Rectangle((bx0, yy), bwmax, bh, fc='#f1f0ec', ec='none', zorder=3))
        ax.add_patch(Rectangle((bx0, yy), max(p * bwmax, 0.25), bh, fc=L_C if p > 0.5 else '#9cc0ea', ec='none', zorder=4))
        ax.text(bx0 + bwmax + 0.8, yy + bh / 2, f'{p:.2f}', fontsize=6.2, ha='left', va='center', color=INK if p > 0.5 else MUTED)
    ax.text(lx + 2.5, ly + 20.8, '$P_L$ for the example', fontsize=6.4, color=L_C, fontweight='bold', va='center')

    # expert G61 card
    gx0, gy0, gw, gh = 30, 6, 46, 46
    card(ax, gx0, gy0, gw, gh, ec=G_C, lw=1.0); header(ax, gx0, gy0 + gh, gw, 'Expert G61 · multimodal LLM', G_C)
    ax.add_patch(FancyBboxPatch((gx0 + 2.5, gy0 + gh - 13.2), gw - 5, 5.6, boxstyle='round,pad=0,rounding_size=0.8', fc=G_BG, ec=G_C, lw=0.5, zorder=3))
    ax.text(gx0 + gw / 2, gy0 + gh - 10.4, 'gpt-6.1-sol · frozen prompt · JSON schema', fontsize=6.3, ha='center', va='center', color=INK, zorder=4)
    ax.text(gx0 + gw / 2, gy0 + gh - 15.3, 'no task-specific training; tools disabled', fontsize=6.2, ha='center', va='center', color=MUTED)
    js = ['{"primary": "yellow_rust",', ' "confidence": 98,', ' "pustule_color": "yellow",', ' "necrotic_blotches": false, …}']
    ax.add_patch(FancyBboxPatch((gx0 + 2.5, gy0 + 2.4), gw - 5, 21.5, boxstyle='round,pad=0,rounding_size=0.8', fc='#fbfaf8', ec='#e3e2dd', lw=0.5, zorder=3))
    ax.text(gx0 + 4, gy0 + 21.4, 'JSON output for the example', fontsize=6.4, color=G_C, fontweight='bold', va='center', zorder=4)
    for k, line in enumerate(js):
        ax.text(gx0 + 4, gy0 + 17.2 - k * 3.6, line, fontsize=6.3, family='Courier New', va='center', color=INK, zorder=4)

    # input -> experts
    arrow(ax, (ix + isz, iy + isz * 0.7), (lx, ly + lh * 0.55), L_C, lw=1.0)
    arrow(ax, (ix + isz, iy + isz * 0.3), (gx0, gy0 + gh * 0.6), G_C, lw=1.0)

    # gates
    gtx, gtw = 84, 44
    gates = [('1', 'Health status', 'L top class = healthy, $P_L$ ≥ $\\tau_h$\nand G61 says healthy', 82, ['L', 'G61'], 'not healthy → gate 2'),
             ('2', 'Disease group', 'L disease score ≥ $\\tau_d$ and group\nscore ≥ $\\tau_g$; G61 names same group', 52, ['L', 'G61'], 'both say rust → gate 3'),
             ('3', 'Rust type', 'L and G61 name the same rust\nand G61 confidence ≥ 95', 22, ['L', 'G61'], 'same rust, conf. 98')]
    gh_ = 21
    for num, title, rule, yy, chips, trace in gates:
        card(ax, gtx, yy, gtw, gh_, ec=S_C, lw=1.0)
        ax.add_patch(FancyBboxPatch((gtx, yy), 6.5, gh_, boxstyle='round,pad=0,rounding_size=1.6', fc=S_C, ec='none', zorder=3))
        ax.add_patch(Rectangle((gtx + 3, yy), 3.5, gh_, fc=S_C, ec='none', zorder=3))
        ax.text(gtx + 3.25, yy + gh_ / 2, num, color='white', fontsize=9, fontweight='bold', ha='center', va='center', zorder=4)
        ax.text(gtx + 8.5, yy + gh_ - 3.2, title, fontsize=7.4, fontweight='bold', va='center', color=INK)
        cx = gtx + gtw - 2
        for ch in chips[::-1]:
            wch = 3.2 if ch == 'L' else 6.0
            cx -= wch; chip(ax, cx, yy + gh_ - 3.2, ch, L_C if ch == 'L' else G_C, fs=6.2); cx -= 1.4
        ax.text(gtx + 8.5, yy + gh_ - 6.6, rule, fontsize=6.5, va='top', color=INK, linespacing=1.3)
        ax.text(gtx + 8.5, yy + 2.3, 'example: ' + trace, fontsize=6.3, va='center', color=S_C, fontweight='bold')
    # experts -> gates (two buses)
    arrow(ax, (lx + lw_, ly + lh * 0.75), (gtx, 82 + gh_ * 0.6), L_C, lw=0.9)
    arrow(ax, (lx + lw_, ly + lh * 0.30), (gtx, 52 + gh_ * 0.6), L_C, lw=0.9)
    arrow(ax, (gx0 + gw, gy0 + gh * 0.85), (gtx, 52 + gh_ * 0.35), G_C, lw=0.9)
    arrow(ax, (gx0 + gw, gy0 + gh * 0.45), (gtx, 22 + gh_ * 0.5), G_C, lw=0.9)
    # gate-to-gate flow (example path)
    for ya, yb in ((82, 52 + gh_), (52, 22 + gh_)):
        arrow(ax, (gtx + gtw / 2, ya), (gtx + gtw / 2, yb), S_C, lw=1.6, ms=8)

    # output: stacked column of the 860 external images
    c = REV['composition']['external']; e = c['errors_by_level']; n = c['n']
    segs = [('healthy', 'Healthy', H_C), ('specific', 'Specific disease', S_C), ('group', 'Disease group "rust"', S_MID), ('referred', 'Referred for review', R_C)]
    bx, bw, y_top, y_bot = 141, 7, 99, 6
    scale = (y_top - y_bot - 3 * 0.8) / n; yy = y_top; centers = {}
    for key, lab, col in segs:
        hh = c[key] * scale
        ax.add_patch(Rectangle((bx, yy - hh), bw, hh, fc=col, ec='none', zorder=3))
        centers[key] = yy - hh / 2
        ne = e.get(key, 0); err = '' if key == 'referred' else '\n' + f"{ne} error" + ('s' if ne != 1 else '')
        ax.text(bx + bw + 1.6, yy - hh / 2, f"{lab}\n{c[key]} ({100 * c[key] / n:.1f} %){err}", fontsize=6.4, va='center', ha='left',
                color=INK, linespacing=1.2)
        yy -= hh + 0.8
    # gates -> outputs
    arrow(ax, (gtx + gtw, 82 + gh_ * 0.75), (bx, centers['healthy']), H_C, lw=0.9)
    arrow(ax, (gtx + gtw, 52 + gh_ * 0.6), (bx, centers['specific'] + 6), S_C, lw=0.9)
    ax.text(gtx + gtw + 0.8, 52 + gh_ * 0.6 - 1.2, 'PM,\nseptoria', fontsize=6.0, color=S_C, ha='left', va='top', linespacing=1.05)
    arrow(ax, (gtx + gtw, 22 + gh_ * 0.72), (bx, centers['specific'] - 6), S_C, lw=1.6, ms=8)
    ax.text(gtx + gtw + 1.2, 22 + gh_ * 0.72 + 1.0, 'yes', fontsize=6.2, color=S_C, ha='left', va='bottom', fontweight='bold')
    arrow(ax, (gtx + gtw, 22 + gh_ * 0.35), (bx, centers['group']), S_MID, lw=0.9)
    ax.text(gtx + gtw + 1.2, 22 + gh_ * 0.35 - 1.0, 'no', fontsize=6.2, color=S_MID, ha='left', va='top', fontweight='bold')
    # referral bus: any failed gate
    rbx = gtx + gtw + 4.5
    for yy_ in (82 + gh_ * 0.25, 52 + gh_ * 0.2):
        ax.plot([gtx + gtw, rbx], [yy_, yy_], color=R_C, lw=0.8, ls=(0, (2.5, 1.5)), zorder=4)
    ax.plot([rbx, rbx], [82 + gh_ * 0.25, 12.5], color=R_C, lw=0.8, ls=(0, (2.5, 1.5)), zorder=4)
    arrow(ax, (rbx, 12.5), (bx, 12.5), R_C, lw=0.8, ls=(0, (2.5, 1.5)))
    ax.text(rbx - 1.0, 14.5, 'any gate\nfails', fontsize=6.0, color=MUTED, ha='right', va='bottom', linespacing=1.1)
    ax.text(86, 4.2, 'Frozen: $\\tau_h = \\tau_d = \\tau_g$ = 0.5, selected on EVAL450 only', fontsize=6.3, color=MUTED, style='italic', va='center')


def main() -> None:
    fig = plt.figure(figsize=(W * MM, H * MM))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off'); ax.set_aspect('equal')
    panel_a(ax); panel_b(ax)
    for ext, kw in (('pdf', {}), ('svg', {}), ('png', {'dpi': 600}), ('tif', {'dpi': 600, 'pil_kwargs': {'compression': 'tiff_lzw'}})):
        fig.savefig(HERE / f'Fig1.{ext}', pad_inches=0, **kw)


if __name__ == '__main__':
    main()
