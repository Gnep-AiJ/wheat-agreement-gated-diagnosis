"""Fig. 1: (a) study design and data roles, (b) agreement-gated hierarchical decision rule with explicit branches.

Vector schematic drawn in matplotlib (SVG editable in Inkscape/PowerPoint). Output counts in panel b are the pooled external
results (860 images) from ../analysis/ms_revision.json.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
REV = json.loads((HERE.parent / 'analysis' / 'ms_revision.json').read_text())
MM = 1 / 25.4
plt.rcParams.update({'font.family': 'Arial', 'font.size': 7, 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'mathtext.default': 'regular'})
INK, MUTED = '#0b0b0b', '#52514e'
COL = {'L': '#2a78d6', 'G': '#eb6834', 'S': '#4a3aa7', 'R': '#9a9993', 'H': '#1baf7a'}
FILL = {'L': '#eef4fc', 'G': '#fdf0ea', 'S': '#f1effa', 'R': '#f4f4f2', 'H': '#e8f7f1', 'D': '#ffffff'}


def box(ax, x, y, w, h, title, body='', edge=MUTED, fill='#ffffff', fs=6.8, ls='-', lw=1.0, title_color=INK, align='center', gap=None, top=0.022):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.004,rounding_size=0.010', linewidth=lw,
                                edgecolor=edge, facecolor=fill, linestyle=ls))
    tx = x + w / 2 if align == 'center' else x + 0.008
    if body:
        ax.text(tx, y + h - top, title, ha=align, va='top', fontsize=fs, fontweight='bold', color=title_color)
        ax.text(tx, y + h - top - (gap if gap else 0.052 * fs / 6.8), body, ha=align, va='top', fontsize=fs - 0.6, color=INK, linespacing=1.35)
    else:
        ax.text(tx, y + h / 2, title, ha=align, va='center', fontsize=fs, color=INK, linespacing=1.35)


def arrow(ax, a, b, color=MUTED, ls='-', lw=0.9, label=None, lpos=0.5, loff=(0.0, 0.012), rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=7, linewidth=lw, color=color, linestyle=ls,
                                 connectionstyle=f'arc3,rad={rad}', shrinkA=0, shrinkB=0))
    if label:
        ax.text(a[0] + (b[0] - a[0]) * lpos + loff[0], a[1] + (b[1] - a[1]) * lpos + loff[1], label, fontsize=6.2, color=color,
                ha='center', va='center', fontweight='bold', bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none'))


def chip(ax, x, y, text, color):
    ax.text(x, y, text, fontsize=5.8, color='white', ha='left', va='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.18,rounding_size=0.25', fc=color, ec='none'))


def panel_a(ax):
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(0, 1); ax.axis('off')
    ax.text(-0.005, 0.99, 'a', fontweight='bold', fontsize=9, va='top')
    ax.text(0.025, 0.99, 'Study design and data roles', fontweight='bold', fontsize=7.5, va='top')
    y, h = 0.17, 0.62
    items = [
        (0.000, 0.185, 'Training data (expert L)', 'WFD2020 sources of the other\ntwo folds (600 per fold),\nCerealConv subset (999),\nMendeley leaf (406),\niNaturalist (611)', COL['L'], FILL['L']),
        (0.205, 0.185, 'Development', 'EVAL450: 450 images,\n8 WFD2020 sources,\n3 source-held folds\nRule design, threshold\nselection, nested CV', COL['S'], FILL['S']),
        (0.410, 0.135, 'Freeze', 'Prompt, model\nensemble, rule\nand thresholds\nfixed; protocol\ntime-stamped', MUTED, '#ffffff'),
        (0.565, 0.215, 'External tests (each once)', 'iNat: 68 new observations\nETS: 552 images, 6 sources\nETS2: 240 field images,\nHenan, China\nde-duplicated vs all pools', COL['S'], '#ffffff'),
        (0.800, 0.200, 'Retrospective analyses', 'All 1,310 images, 16 sources\nLeave-one-source-out\narbiters; K labelled images\nper source for\nrule selection', COL['R'], FILL['R']),
    ]
    for x, w, t, b, e, f in items:
        box(ax, x, y, w, h, t, b, edge=e, fill=f, fs=6.8, ls='--' if t.startswith('Retro') else '-', gap=0.15, top=0.05)
    for x0, x1 in ((0.185, 0.205), (0.390, 0.410), (0.545, 0.565), (0.780, 0.800)):
        arrow(ax, (x0, y + h / 2), (x1, y + h / 2))
    ax.plot([0.0, 0.78], [0.045, 0.045], color=COL['S'], lw=1.6, solid_capstyle='butt')
    ax.text(0.39, 0.0, 'prespecified analyses', ha='center', va='bottom', fontsize=6.2, color=COL['S'], style='italic',
            bbox=dict(fc='white', ec='none', pad=0.6))
    ax.plot([0.80, 1.0], [0.045, 0.045], color=COL['R'], lw=1.6, ls=(0, (3, 1.5)))
    ax.text(0.90, 0.0, 'after all external tests', ha='center', va='bottom', fontsize=6.2, color=MUTED, style='italic',
            bbox=dict(fc='white', ec='none', pad=0.6))


def panel_b(ax):
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(0, 1); ax.axis('off')
    ax.text(-0.005, 0.995, 'b', fontweight='bold', fontsize=9, va='top')
    ax.text(0.025, 0.995, 'Agreement-gated hierarchical decision rule', fontweight='bold', fontsize=7.5, va='top')
    c = REV['composition']['external']; e = c['errors_by_level']; n = c['n']
    # inputs and experts
    box(ax, 0.000, 0.42, 0.095, 0.17, 'Input image', 'RGB photo,\nno metadata', MUTED)
    box(ax, 0.125, 0.66, 0.185, 0.21, 'Expert L', 'DINOv3 ViT-B/16 fine-tuned\non public images; 9-model\nensemble (3 seeds x 3 folds)\n→ 6 class probabilities $P_L$',
        COL['L'], FILL['L'], title_color=COL['L'])
    box(ax, 0.125, 0.12, 0.185, 0.24, 'Expert G61', 'gpt-6.1-sol, frozen prompt,\nno task-specific training\n→ primary label, confidence\n0–100, symptom report\n(JSON)',
        COL['G'], FILL['G'], title_color=COL['G'])
    arrow(ax, (0.095, 0.53), (0.125, 0.76)); arrow(ax, (0.095, 0.48), (0.125, 0.25))
    # decision nodes
    dx, dw = 0.345, 0.235
    nodes = {
        'q1': (0.74, 'L: top class is healthy and\n$P_L$(healthy) $\\geq \\tau_h$ ?', ['L']),
        'q2': (0.535, 'L: top disease probability\n$\\geq \\tau_d$ ?', ['L']),
        'q3': (0.305, 'L: top disease group score $\\geq \\tau_g$\nand G61 names the same group ?', ['L', 'G61']),
        'q4': (0.075, 'Rust: L and G61 name the same\nrust type, G61 confidence $\\geq$ 95 ?', ['L', 'G61']),
    }
    hh = 0.135
    for k, (yy, txt, chips) in nodes.items():
        box(ax, dx, yy, dw, hh, txt, edge=COL['S'], fill='#ffffff', fs=6.5)
        cx = dx + 0.006
        for ch in chips:
            chip(ax, cx, yy + hh - 0.014, ch, COL['L'] if ch == 'L' else COL['G']); cx += 0.022 if ch == 'L' else 0.034
    q1b = (0.625, 0.74, 0.165)
    box(ax, q1b[0], q1b[1], q1b[2], hh, 'G61 primary label\nis healthy ?', edge=COL['S'], fill='#ffffff', fs=6.5)
    chip(ax, q1b[0] + 0.006, q1b[1] + hh - 0.014, 'G61', COL['G'])
    arrow(ax, (0.310, 0.80), (dx, 0.80), COL['L']); arrow(ax, (0.310, 0.25), (dx, 0.36), COL['G'])
    # outputs
    ox, ow = 0.835, 0.165
    outs = {
        'H': (0.74, 'Healthy', f"{c['healthy']} of {n} external images\n({e.get('healthy', 0)} wrong)", COL['H'], FILL['H'], '-'),
        'R': (0.47, 'Referred for review', f"{c['referred']} images\nno automatic label", COL['R'], FILL['R'], '--'),
        'G': (0.03, 'Disease group: "rust"', f"{c['group']} images ({e.get('group', 0)} wrong)", COL['S'], FILL['S'], '-'),
        'S': (0.235, 'Specific disease', f"{c['specific']} images ({e.get('specific', 0)} wrong)\nPM, septoria or rust type", COL['S'], '#e2ddf5', '-'),
    }
    oh = {'H': 0.135, 'R': 0.15, 'G': 0.125, 'S': 0.15}
    for k, (yy, t, b, ec, fc, ls) in outs.items():
        box(ax, ox, yy, ow, oh[k], t, b, edge=ec, fill=fc, ls=ls, fs=6.6, title_color=ec if k != 'S' else INK)
    S = COL['S']; Rr = COL['R']
    mid = lambda k: nodes[k][0] + hh / 2  # noqa: E731
    arrow(ax, (dx + dw, mid('q1')), (q1b[0], mid('q1')), S, label='yes')
    arrow(ax, (q1b[0] + q1b[2], mid('q1')), (ox, mid('q1')), COL['H'], label='yes')
    arrow(ax, (q1b[0] + q1b[2] / 2, q1b[1]), (ox, 0.585), Rr, ls='--', label='no', lpos=0.35)
    arrow(ax, (dx + dw / 2, nodes['q1'][0]), (dx + dw / 2, nodes['q2'][0] + hh), S, label='no', loff=(0.014, 0))
    arrow(ax, (dx + dw, mid('q2')), (ox, 0.545), Rr, ls='--', label='no', lpos=0.55)
    arrow(ax, (dx + dw / 2, nodes['q2'][0]), (dx + dw / 2, nodes['q3'][0] + hh), S, label='yes', loff=(0.016, 0))
    arrow(ax, (dx + dw, mid('q3') + 0.03), (ox, 0.50), Rr, ls='--', label='no', lpos=0.55)
    arrow(ax, (dx + dw, mid('q3') - 0.02), (ox, 0.335), S, label='yes: PM or septoria', lpos=0.45, loff=(0.0, 0.0))
    arrow(ax, (dx + dw / 2, nodes['q3'][0]), (dx + dw / 2, nodes['q4'][0] + hh), S, label='yes: rust', loff=(0.03, 0))
    arrow(ax, (dx + dw, mid('q4') + 0.03), (ox, 0.275), S, label='yes', lpos=0.5)
    arrow(ax, (dx + dw, mid('q4') - 0.03), (ox, 0.09), S, label='no', lpos=0.5)
    ax.text(0.47, 0.012, r'$\tau_h = \tau_d = \tau_g$ = 0.5 and the rust-type rule were selected on EVAL450 only (error ≤ 5 % among answered images), then frozen',
            ha='center', va='bottom', fontsize=6.0, color=MUTED, style='italic')


def main() -> None:
    fig = plt.figure(figsize=(174 * MM, 140 * MM))
    ax_a = fig.add_axes([0.0, 0.74, 1.0, 0.26]); ax_b = fig.add_axes([0.0, 0.0, 1.0, 0.72])
    panel_a(ax_a); panel_b(ax_b)
    for ext, kw in (('pdf', {}), ('svg', {}), ('png', {'dpi': 600}), ('tif', {'dpi': 600, 'pil_kwargs': {'compression': 'tiff_lzw'}})):
        fig.savefig(HERE / f'Fig1.{ext}', bbox_inches='tight', pad_inches=0.02, **kw)


if __name__ == '__main__':
    main()
