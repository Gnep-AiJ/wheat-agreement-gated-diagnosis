"""Statistical figures for the manuscript (data: ../analysis/ms_results.json and ms_revision.json; cached outputs only).

Fig2 data audit; Fig3 complementarity (separate denominators); Fig5 risk-coverage; Fig6 output level and reliability per source;
FigS1 forced-choice fusion strategies (supplement).
Journal spec (Precision Agriculture): width 174 mm, height <= 234 mm, Arial 7-9 pt, RGB, vector + 600-dpi TIFF; colour is
always paired with a second encoding (marker, hatch, position or direct label). Palette validated with the dataviz validator.
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import beta  # noqa: E402

HERE = Path(__file__).resolve().parent
R = json.loads((HERE.parent / 'analysis' / 'ms_results.json').read_text())
V = json.loads((HERE.parent / 'analysis' / 'ms_revision.json').read_text())
MM = 1 / 25.4
plt.rcParams.update({'font.family': 'Arial', 'font.size': 7.5, 'axes.titlesize': 7.5, 'axes.labelsize': 7.5, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'legend.fontsize': 7, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
                     'ytick.major.width': 0.6, 'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b', 'xtick.color': '#52514e',
                     'ytick.color': '#52514e', 'axes.titleweight': 'bold'})
C = {'L': '#2a78d6', 'G61': '#eb6834', 'RAG': '#1baf7a', 'SYS': '#4a3aa7', 'SYS2': '#8577e0', 'UB': '#9a9993', 'grid': '#e6e5e0',
     'H': '#1baf7a', 'REF': '#d9d8d2', 'INK': '#0b0b0b', 'MUTED': '#52514e'}
COLL_NAME = {'EVAL450': 'Development\n(EVAL450, n=450)', 'ETS': 'External ETS\n(6 sources, n=552)',
             'ETS2': 'External ETS2\n(Henan field, n=240)', 'P4': 'External iNat\n(n=68)'}
COLLS = ['EVAL450', 'ETS', 'ETS2', 'P4']


def cp(k, n):
    return (beta.ppf(.025, k, n - k + 1) if k > 0 else 0.0, beta.ppf(.975, k + 1, n - k) if k < n else 1.0)


def letter(ax, s, x=-0.02, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontweight='bold', fontsize=9, ha='right', va='bottom')


def save(fig, name, rows=None, header=None):
    for ext, kw in (('pdf', {}), ('svg', {}), ('png', {'dpi': 600}), ('tif', {'dpi': 600, 'pil_kwargs': {'compression': 'tiff_lzw'}})):
        fig.savefig(HERE / f'{name}.{ext}', bbox_inches='tight', pad_inches=0.03, **kw)
    if rows:
        with (HERE / f'{name}_source_data.csv').open('w', newline='', encoding='utf-8') as f:
            w = csv.writer(f); w.writerow(header); w.writerows(rows)
    plt.close(fig)


def fig_audit():
    """Fig. 2: (a) cross-collection near-duplicates, (b) visual check of the threshold, (c) source-familiar vs source-held."""
    a = R['leakage']; names = sorted(a['per_dataset'])
    short = {'WFD2020': 'WFD2020', 'kaggle_wheat_disease_small': 'CerealConv subset', 'kaggle_wheat_leaf_disease_jayaprakash': 'Jayaprakash',
             'kaggle_wheat_plant_diseases_kushagra': 'Kushagra', 'mendeley_wheat_disease_2025_original': 'Mendeley 2025',
             'mendeley_wheat_leaf_dataset': 'Mendeley leaf'}
    fig = plt.figure(figsize=(174 * MM, 92 * MM))
    ax = fig.add_axes([0.13, 0.20, 0.42, 0.72]); rows = []
    M = np.full((len(names), len(names)), np.nan)
    for i, x in enumerate(names):
        for j, y in enumerate(names):
            if x != y:
                v = a['pairs'][f'{x}|{y}']; M[i, j] = 100 * v['ge090'] / v['n_a']; rows.append(['a', x, y, v['ge090'], v['n_a']])
    im = ax.imshow(M, cmap='Blues', vmin=0, vmax=100)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, '–' if i == j else f'{M[i, j]:.1f}', ha='center', va='center', fontsize=6.5,
                    color='white' if (i != j and M[i, j] > 55) else C['INK'])
    anyo = a['any_other']
    lab = [f"{short[n]} (n={a['per_dataset'][n]:,})\nany other: {anyo[n]['pct']:.1f} %" for n in names]
    ax.set_xticks(range(len(names)), [short[n] for n in names], rotation=35, ha='right'); ax.set_yticks(range(len(names)), lab, fontsize=6.5)
    ax.set_xlabel('… has a near-duplicate in this collection'); ax.set_ylabel('Images of this collection …')
    for s in ax.spines.values():
        s.set_visible(False)
    cax = fig.add_axes([0.565, 0.32, 0.012, 0.48]); cb = fig.colorbar(im, cax=cax); cb.outline.set_linewidth(0.4)
    cb.set_label('Share of images (%)', fontsize=6.5); cb.ax.tick_params(labelsize=6.5)
    letter(ax, 'a', x=-0.75, y=1.0)
    # b: visual validation of the threshold
    ax = fig.add_axes([0.72, 0.66, 0.27, 0.27]); vv = V['dedup_visual_validation']
    bins = list(vv); xs = np.arange(len(bins))
    vis = [100 * vv[b]['same'] / vv[b]['n'] for b in bins]; kp = [100 * vv[b]['keypoint_same'] / vv[b]['n'] for b in bins]
    ax.bar(xs - 0.19, vis, width=0.36, color=C['L'], edgecolor='white', linewidth=0.6, label='Visual rating')
    ax.bar(xs + 0.19, kp, width=0.36, color=C['SYS'], edgecolor='white', linewidth=0.6, hatch='////', label='Keypoint matching')
    for x, b in zip(xs, bins):
        ax.text(x + 0.19, 100 * vv[b]['keypoint_same'] / vv[b]['n'] + 2, f"{vv[b]['keypoint_same']}", ha='center', va='bottom', fontsize=5.6)
        ax.text(x - 0.19, 100 * vv[b]['same'] / vv[b]['n'] + 2, f"{vv[b]['same']}", ha='center', va='bottom', fontsize=5.6)
        rows.append(['b', b, vv[b]['same'], vv[b]['keypoint_same'], vv[b]['n']])
    ax.legend(loc='upper left', frameon=False, fontsize=5.8, handlelength=1.0, borderaxespad=0.1)
    ax.axvline(2.5, color=C['MUTED'], lw=0.8, ls='--'); None
    ax.set_xticks(xs, [b.replace('0.', '.').replace('1.00', '1') for b in bins], rotation=35, ha='right', fontsize=6.3)
    ax.set_ylim(0, 125); ax.set_ylabel('Same photograph (% of 20)', fontsize=6.8); ax.set_xlabel('Cosine similarity (dashed: threshold 0.90)', fontsize=6.8)
    letter(ax, 'b', x=-0.28)
    # c: source-familiar vs source-held
    ax = fig.add_axes([0.72, 0.11, 0.27, 0.22]); fh = R['familiar_vs_held']
    for yi, key, lab_ in ((1, 'all', 'All 450 images'), (0, 'rust_single_label', 'Single-label rust\n(312 images)')):
        d = fh[key]; f_ = 100 * d['familiar'] / d['n']; h_ = 100 * d['held'] / d['n']
        ax.plot([h_, f_], [yi, yi], color=C['UB'], lw=2, zorder=1)
        ax.scatter([f_], [yi], s=34, color=C['SYS'], marker='o', zorder=3, label='Source seen in training' if yi else None)
        ax.scatter([h_], [yi], s=34, color='white', edgecolor=C['SYS'], linewidth=1.2, marker='o', zorder=3, label='Source held out' if yi else None)
        ax.text(f_ + 3, yi, f'{f_:.1f}', va='center', fontsize=6.3); ax.text(h_ - 3, yi, f'{h_:.1f}', va='center', ha='right', fontsize=6.3)
        rows.append(['c', key, d['familiar'], d['held'], d['n']])
    ax.set_yticks([1, 0], ['All 450 images', 'Single-label rust\n(312 images)'], fontsize=6.5); ax.set_ylim(-0.6, 1.6)
    ax.set_xlim(25, 105); ax.set_xlabel('Accuracy of the same recipe (%)', fontsize=6.8)
    ax.legend(loc='upper center', bbox_to_anchor=(0.40, 1.30), ncol=2, frameon=False, fontsize=6.2, handletextpad=0.2, columnspacing=0.8)
    letter(ax, 'c', x=-0.28, y=1.18)
    save(fig, 'Fig2', rows, ['panel', 'a', 'b', 'c', 'd'])


def fig_complementarity():
    """Fig. 3: (a) who is correct, (b) full-sample accuracy and the two-expert selection bound, (c) accuracy when L and G61 agree."""
    fig, axes = plt.subplots(1, 3, figsize=(174 * MM, 74 * MM), gridspec_kw={'width_ratios': [1.25, 0.9, 0.9]})
    cpd = R['complementarity']; rows = []; y = np.arange(len(COLLS))[::-1]
    ax = axes[0]
    parts = [('both_correct', 'Both correct', C['SYS'], ''), ('only_L', 'Only L correct', C['L'], '////'),
             ('only_G61', 'Only G61 correct', C['G61'], '\\\\\\\\'), ('both_wrong', 'Both wrong', C['REF'], '')]
    for yi, c in zip(y, COLLS):
        left = 0; n = cpd[c]['n']
        for key, lab, col, hatch in parts:
            v = cpd[c][key] / n * 100
            ax.barh(yi, v, left=left, color=col, edgecolor='white', linewidth=1.0, hatch=hatch, height=0.62, label=lab if c == COLLS[0] else None)
            if v >= 7:
                ax.text(left + v / 2, yi, f'{v:.0f}', ha='center', va='center', fontsize=6.8, color='white' if key != 'both_wrong' else C['INK'],
                        bbox=dict(boxstyle='square,pad=0.08', fc=col, ec='none') if hatch else None)
            left += v; rows.append(['a', c, key, cpd[c][key], n])
    ax.set_yticks(y, [COLL_NAME[c] for c in COLLS]); ax.set_xlim(0, 100); ax.set_xlabel('Share of images (%)')
    ax.set_title('Which expert is correct', loc='left', fontsize=7.5); letter(ax, 'a')
    ax.legend(loc='upper center', bbox_to_anchor=(0.45, -0.17), ncol=2, frameon=False, handlelength=1.4, columnspacing=0.8)
    ax = axes[1]; fr = R['forced']
    keys = [('L', 'L', C['L'], 'o'), ('G61 (raw)', 'G61', C['G61'], 's'), ('Selection upper bound (L or G61 correct)', 'L or G61 correct (oracle)', C['UB'], 'D')]
    for j, (k, lab, col, mk) in enumerate(keys):
        xs = np.array([fr[k][c]['acc'] * 100 for c in COLLS]); lo = xs - [fr[k][c]['ci95'][0] * 100 for c in COLLS]
        hi = np.array([fr[k][c]['ci95'][1] * 100 for c in COLLS]) - xs
        ax.errorbar(xs, y + (1 - j) * 0.2, xerr=[lo, hi], fmt=mk, color=col, ms=4.5, mec='white', mew=0.5, elinewidth=0.8, capsize=0, label=lab)
        rows += [['b', c, k, fr[k][c]['k'], fr[k][c]['n']] for c in COLLS]
    ax.set_yticks(y, ['' for _ in COLLS]); ax.set_xlim(40, 100); ax.set_xlabel('Accuracy on all images (%)')
    ax.grid(axis='x', color=C['grid'], linewidth=0.5); ax.set_axisbelow(True)
    ax.set_title('Forced choice, all images', loc='left', fontsize=7.5); letter(ax, 'b')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=1, frameon=False, handletextpad=0.3)
    ax = axes[2]
    for yi, c in zip(y, COLLS):
        k, n = cpd[c]['agree_correct'], cpd[c]['agree_n']; lo, hi = cp(k, n); acc = 100 * k / n
        ax.errorbar([acc], [yi], xerr=[[acc - 100 * lo], [100 * hi - acc]], fmt='*', color=C['SYS'], ms=8, mec='white', mew=0.4,
                    elinewidth=0.8, capsize=0)
        ax.text(53, yi + 0.30, f"{acc:.1f} % ({k}/{n}); agree on {100 * n / cpd[c]['n']:.0f} % of images", fontsize=6.2, color=C['INK'], va='center')
        rows.append(['c', c, 'agreement_subset', k, n])
    ax.set_yticks(y, ['' for _ in COLLS]); ax.set_xlim(52, 100.5); ax.set_ylim(-0.5, 3.6); ax.set_xlabel('Accuracy when L and G61 agree (%)')
    ax.grid(axis='x', color=C['grid'], linewidth=0.5); ax.set_axisbelow(True)
    ax.set_title('Only images where both agree', loc='left', fontsize=7.5); letter(ax, 'c')
    fig.subplots_adjust(wspace=0.16, bottom=0.30, top=0.90)
    save(fig, 'Fig3', rows, ['panel', 'collection', 'quantity', 'count', 'denominator'])


def fig_risk_coverage():
    """Fig. 5: group-level risk-coverage curves; full range (top) and the low-error region (bottom)."""
    rc = R['risk_coverage_group_level']; sel = R['selective']
    fig, axes = plt.subplots(2, 3, figsize=(174 * MM, 112 * MM), sharex=True, gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.14})
    rows = []
    title = {'ETS': 'ETS (6 sources, n=552)', 'ETS2': 'ETS2 (Henan field, n=240)', 'P4': 'iNat (n=68)'}
    for col, c in enumerate(['ETS', 'ETS2', 'P4']):
        s = sel[c]; k = s['released']; cov = 100 * s['coverage']; ge = int(round(k * (1 - s['group_accuracy'])))
        lo, hi = cp(ge, k)
        for row in (0, 1):
            ax = axes[row, col]
            for key, lab, colr, ls in (('L', 'L, ranked by its probability', C['L'], '-'), ('G61', 'G61, ranked by its confidence', C['G61'], '--')):
                pts = np.array(rc[c][key]); ax.plot(pts[:, 0] * 100, pts[:, 1] * 100, ls, color=colr, lw=1.3, label=lab)
                if row == 0:
                    rows += [[c, key, round(a, 4), round(b, 4)] for a, b in pts]
            pts = np.array(rc[c]['Agreement-gated system'])
            ax.plot(pts[:, 0] * 100, pts[:, 1] * 100, ':', color=C['SYS'], lw=1.2, marker='o', ms=2.2, label='Agreement-gated system, thresholds varied')
            if row == 0:
                rows += [[c, 'system_sweep', round(a, 4), round(b, 4)] for a, b in pts]
            mL = 100 * (1 - s['matched_L_group_accuracy']); mG = 100 * (1 - s['matched_G61_group_accuracy'])
            ax.plot([cov], [mL], 'o', ms=5, mfc='white', mec=C['L'], mew=1.2, label='L at the same coverage')
            ax.plot([cov], [mG], 's', ms=5, mfc='white', mec=C['G61'], mew=1.2, label='G61 at the same coverage')
            ax.errorbar([cov], [100 * ge / k], yerr=[[100 * ge / k - 100 * lo], [100 * hi - 100 * ge / k]], fmt='*', ms=10, color=C['SYS'],
                        mec='white', mew=0.5, elinewidth=1.0, capsize=2, zorder=6, label='Frozen operating point (95 % CI)')
            ax.set_xlim(0, 100); ax.grid(color=C['grid'], linewidth=0.5); ax.set_axisbelow(True)
            if row == 0:
                ax.set_ylim(-1, 45); ax.set_title(title[c], fontsize=7.5); letter(ax, 'abc'[col], x=-0.06)
                ax.axhspan(-1, 10, color='#f1effa', zorder=0)
            else:
                ax.set_ylim(-0.4, 10); ax.set_xlabel('Coverage (% of images answered)')
                ax.annotate(f'{k}/{s["n"]} answered\n{ge} group-level errors', (cov, 100 * ge / k), xytext=(4, 8.2) if c != 'P4' else (55, 8.6),
                            textcoords='data', fontsize=6.2, arrowprops=dict(arrowstyle='-', lw=0.5, color=C['MUTED']))
        if col == 0:
            axes[0, 0].set_ylabel('Error rate (%)\nfull range'); axes[1, 0].set_ylabel('Error rate (%)\n0–10 % region')
        rows.append([c, 'operating_point', round(cov / 100, 4), round(ge / k, 4)])
        rows.append([c, 'matched_L', round(cov / 100, 4), round(mL / 100, 4)]); rows.append([c, 'matched_G61', round(cov / 100, 4), round(mG / 100, 4)])
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.06), handlelength=2.2, columnspacing=1.2)
    fig.subplots_adjust(wspace=0.16, bottom=0.17)
    save(fig, 'Fig5', rows, ['collection', 'curve', 'coverage', 'error_rate'])


SRCN = {'ets2:henan_field_2023': 'Henan field 2023 (ETS2)', 'ets:mswdd2022': 'MSWDD2022 (ETS)', 'ets:plantwild': 'PlantWild v2 web (ETS)',
        'ets:roboflow_newwheat': 'Roboflow new-wheat-disease (ETS)', 'ets:roboflow_ptriticina': 'Roboflow P. triticina (ETS)',
        'ets:roboflow_stemrust': 'Roboflow stem-rust (ETS)', 'ets:zenodo13137587': 'Zenodo powdery mildew (ETS)',
        'p4_rag:inat_pathogen': 'iNaturalist (iNat)'}


def fig_sources():
    """Fig. 6: (a) level of the automatic answer per external source, (b) accuracy among answered images per source."""
    ps = V['per_source']; order = sorted(ps, key=lambda s: -ps[s]['n'])
    pooled = V['composition']['external']
    fig, axes = plt.subplots(1, 2, figsize=(174 * MM, 82 * MM), gridspec_kw={'width_ratios': [1.25, 1]}, sharey=True)
    rows = []; labels = [f"{SRCN[s]}, n={ps[s]['n']}" for s in order] + [f"All external images, n={pooled['n']}"]
    y = np.arange(len(labels))[::-1]
    parts = [('specific', 'Specific disease', C['SYS'], ''), ('group', 'Disease group ("rust")', C['SYS2'], ''),
             ('healthy', 'Healthy', C['H'], ''), ('referred', 'Referred for review', C['REF'], '////')]
    ax = axes[0]
    for yi, s in zip(y, order + ['pooled']):
        d = pooled if s == 'pooled' else dict(ps[s], referred=ps[s]['n'] - ps[s]['answered'])
        left = 0
        for key, lab, col, hatch in parts:
            v = 100 * d[key] / d['n']
            ax.barh(yi, v, left=left, height=0.66, color=col, hatch=hatch, edgecolor='white', linewidth=0.8, label=lab if s == order[0] else None)
            if v >= 9:
                ax.text(left + v / 2, yi, f'{v:.0f}', ha='center', va='center', fontsize=6.4,
                        color='white' if key in ('specific', 'group') else C['INK'])
            left += v; rows.append(['a', s, key, d[key], d['n']])
    ax.axhline(0.5, color=C['MUTED'], lw=0.6)
    ax.set_yticks(y, labels, fontsize=6.6); ax.set_xlim(0, 100); ax.set_xlabel('Share of images (%)')
    ax.set_title('Level of the automatic answer', loc='left', fontsize=7.5); letter(ax, 'a', x=-0.02)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False, handlelength=1.4, columnspacing=0.9)
    ax = axes[1]
    for yi, s in zip(y, order + ['pooled']):
        if s == 'pooled':
            k = sum(ps[q]['answered'] for q in ps); e = sum(ps[q]['errors'] for q in ps)
        else:
            k, e = ps[s]['answered'], ps[s]['errors']
        lo, hi = cp(k - e, k); acc = 100 * (k - e) / k
        ax.errorbar([acc], [yi], xerr=[[acc - 100 * lo], [100 * hi - acc]], fmt='D' if s == 'pooled' else 'o', color=C['SYS'], ms=4.5,
                    mec='white', mew=0.5, elinewidth=0.9, capsize=0)
        ax.text(101.2, yi, f'{k - e}/{k}', va='center', fontsize=6.4, color=C['INK'])
        rows.append(['b', s, k - e, k])
    ax.axhline(0.5, color=C['MUTED'], lw=0.6)
    ax.set_xlim(70, 100); ax.set_xlabel('Accuracy among answered images (%)')
    ax.grid(axis='x', color=C['grid'], linewidth=0.5); ax.set_axisbelow(True)
    ax.text(101.2, len(labels) - 0.35, 'correct/\nanswered', fontsize=6, color=C['MUTED'], va='bottom')
    ax.set_title('Reliability of automatic answers', loc='left', fontsize=7.5); letter(ax, 'b', x=-0.02)
    fig.subplots_adjust(wspace=0.12, bottom=0.25, right=0.9)
    save(fig, 'Fig6', rows, ['panel', 'source', 'quantity', 'count', 'denominator'])


def fig_fusion_supp():
    """Fig. S1: forced-choice accuracy of fusion strategies (retrospective leave-one-source-out, 16 sources)."""
    fu = R['fusion_attempts']; t1 = fu['learned']; tk = fu['knowledge']; ks = fu['k_shot']
    series = [('L', t1['L'], 'L', C['L'], 'o'), ('G61+L', t1['G61+L'], 'G61, L if uncertain', C['G61'], 's'),
              ('vote3', t1['vote3'], 'Majority vote (L, G61, RAG)', C['MUTED'], '^'),
              ('learned', t1['M1[L+G]'], 'Best learned arbiter (LOSO)', '#e87ba4', 'v'),
              ('knowledge', tk['K (knowledge arbiter)'], 'Knowledge-rule arbiter', '#eda100', 'P'),
              ('k10', ks['K=10'], '10 labelled images per source', C['SYS'], '*'),
              ('ub', t1['oracle(L|G61)'], 'L or G61 correct (oracle)', C['UB'], 'D')]
    cols = ['pooled'] + COLLS
    fig, ax = plt.subplots(figsize=(174 * MM, 70 * MM)); rows = []; x = np.arange(len(cols))
    for j, (k, d, lab, col, mk) in enumerate(series):
        off = (j - (len(series) - 1) / 2) * 0.1
        ax.plot(x + off, [d[c] * 100 for c in cols], mk, color=col, ms=6 if mk != '*' else 9, mec='white', mew=0.5, ls='none', label=lab)
        rows += [[k, c, round(d[c], 4)] for c in cols]
    ax.set_xticks(x, ['All 16 sources\n(n=1,310)'] + [COLL_NAME[c] for c in COLLS])
    ax.set_ylabel('Forced-choice accuracy (%)'); ax.set_ylim(45, 100); ax.grid(axis='y', color=C['grid'], linewidth=0.5); ax.set_axisbelow(True)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=4, frameon=False, handletextpad=0.3, columnspacing=1.0)
    save(fig, 'FigS1', rows, ['strategy', 'collection', 'accuracy'])


if __name__ == '__main__':
    fig_audit(); fig_complementarity(); fig_risk_coverage(); fig_sources(); fig_fusion_supp()
    print('figures written to', HERE)
