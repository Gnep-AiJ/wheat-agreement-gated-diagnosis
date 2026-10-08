"""Fig. 4: real cases with zoomed insets (CC BY 4.0 sources only): the four outcome types of the agreement-gated system.

a, b  ETS stem rust (Roboflow stem-rust v1): L says leaf rust, G61 stem rust -> answer at the disease-group level ("rust")
c, d  ETS2 late stripe rust (Zenodo 15621359): G61 says septoria (no pustules, black dots), L stripe rust -> referred for review
e     ETS stripe rust (Roboflow new-wheat-disease v2): both agree, G61 confidence >= 95 -> specific answer, correct
f     ETS2 image annotated as stripe rust: both experts say leaf rust -> specific answer counted as an error
      (scattered round pustules and the source file name "LeafRust" suggest an annotation error; the error is kept in all metrics)
System outputs are recomputed with the frozen configuration. Writes Fig4.* and Fig4_cases.csv (image files, outputs, zoom boxes).
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import ConnectionPatch, Rectangle  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
import paper_p3_hrme as H  # noqa: E402
from paper_arbiter import LAB, load_rows  # noqa: E402

MM = 1 / 25.4
plt.rcParams.update({'font.family': 'Arial', 'font.size': 7, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
NICE = {'healthy': 'healthy', 'leaf_rust': 'leaf rust', 'stem_rust': 'stem rust', 'yellow_rust': 'stripe rust',
        'powdery_mildew': 'powdery mildew', 'septoria': 'septoria', None: 'uncertain'}
SRC = {'roboflow_newwheat': 'Roboflow new-wheat-disease v2', 'roboflow_stemrust': 'Roboflow stem-rust v1',
       'henan_field_2023': 'Henan field 2023, Zenodo 15621359'}
OUTCOL = {'group': '#4a3aa7', 'referred': '#52514e', 'specific': '#4a3aa7', 'error': '#b3261e'}
ZOOM_COLOR = '#ffd400'


def system_output(r):
    H.STRICT[0] = True
    conf = (r['gfull'] or {}).get('confidence') or 0
    lvl, claim = H.hrme_v3(dict(L=r['L'], G=(r['g'] if r['g'] in LAB else None, conf)), 0.5, 0.5, 0.5, 95, True, True)
    if lvl == 'specific':
        return 'specific', NICE[claim]
    if lvl == 'group':
        return 'group', {'R': 'rust', 'PM': 'powdery mildew', 'SEP': 'septoria'}[claim]
    if lvl == 'healthy':
        return 'healthy', 'healthy'
    return 'referred', None


def symptoms(sym):
    p = sym.get('pustules_present')
    col = str(sym.get('pustule_color', 'none')).replace('_', '-')
    arr = {'elongated_on_stem_or_sheath': 'elongated on stem', 'stripes_along_veins': 'in stripes', 'scattered': 'scattered'}.get(
        sym.get('pustule_arrangement'), '')
    t = f'{col} pustules {arr}'.strip() if p else 'no pustules'
    if sym.get('necrotic_blotches'):
        t += '; necrotic blotches' + (' with black dots' if sym.get('black_dots_in_lesions') else '')
    return t


def main() -> None:
    rows = load_rows()
    by = {}
    for coll, d in (('ETS', 'ets'), ('ETS2', 'ets2')):
        rr = [r for r in rows if r['coll'] == coll]; it = json.loads((ROOT / f'outputs/paper_hrme/{d}/items.json').read_text())
        for r, x in zip(rr, it):
            by[Path(x['file']).name] = (coll, r, x)
    # (file name, zoom box as fractions of the square crop: x0, y0, size, outcome tag)
    cases = [('TRB8EQ_jpg.rf.dd4845a8d12308979aad027bf7469aa9.jpg', (0.40, 0.15, 0.22)),
             ('O0DG7Q_jpg.rf.83e47c50db0ec03b8761e01dd014e773.jpg', (0.47, 0.46, 0.20)),
             ('StripeRust 0000242.jpg', (0.58, 0.50, 0.22)),
             ('StripeRust 0000118.jpg', (0.56, 0.44, 0.20)),
             ('stripe145_jpg.rf.eee2edbfdf2344d04fdb64d6269b7684.jpg', (0.35, 0.40, 0.25)),
             ('LeafRust 0000519.jpg', (0.36, 0.18, 0.20))]
    fig = plt.figure(figsize=(174 * MM, 166 * MM)); out = []
    W, Hh, top = 0.31, 0.347, 0.985
    for k, (name, (zx, zy, zs)) in enumerate(cases):
        coll, r, it = by[name]
        col, row = k % 3, k // 3
        ax = fig.add_axes([0.005 + col * (W + 0.027), top - row * 0.515 - Hh, W, Hh])
        im = ImageOps.exif_transpose(Image.open(it['file'])).convert('RGB'); w0, h0 = im.size; side = min(w0, h0)
        im = im.crop(((w0 - side) // 2, (h0 - side) // 2, (w0 - side) // 2 + side, (h0 - side) // 2 + side))
        big = im.resize((1000, 1000), Image.LANCZOS)
        ax.imshow(big); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.4); s.set_color('#52514e')
        ax.add_patch(Rectangle((zx * 1000, zy * 1000), zs * 1000, zs * 1000, fill=False, ec=ZOOM_COLOR, lw=1.2))
        # inset in the corner opposite to the zoom box
        ix = 0.02 if zx + zs / 2 > 0.5 else 0.56; iy = 0.56 if zy + zs / 2 > 0.5 else 0.02
        ins = ax.inset_axes([ix, iy, 0.42, 0.42])
        crop = im.crop((int(zx * side), int(zy * side), int((zx + zs) * side), int((zy + zs) * side))).resize((600, 600), Image.LANCZOS)
        ins.imshow(crop); ins.set_xticks([]); ins.set_yticks([])
        for s in ins.spines.values():
            s.set_linewidth(1.4); s.set_color(ZOOM_COLOR)
        ax.add_artist(ConnectionPatch(((zx + zs / 2) * 1000, (zy + zs / 2) * 1000), (0.5, 0.5), coordsA='data', coordsB='axes fraction',
                                      axesA=ax, axesB=ins, color=ZOOM_COLOR, lw=0.6, zorder=1))
        L = r['L']; li = int(np.argmax(L)); g = r['gfull'] or {}; sym = g.get('symptoms', {})
        truth = next(iter(r['truth'])); lvl, claim = system_output(r)
        if lvl == 'referred':
            sysout, tag = 'referred for review (no automatic label)', 'referred'
        else:
            ok = H.correct(lvl, {'rust': 'R'}.get(claim, claim) if lvl == 'group' else
                           {v: kk for kk, v in NICE.items()}.get(claim, claim), r['truth'])
            sysout = f"{'disease group' if lvl == 'group' else 'specific'}: {claim}" + ('' if ok else '  (counted as error)')
            tag = lvl if ok else 'error'
        ax.text(-0.01, 1.01, 'abcdef'[k], transform=ax.transAxes, fontweight='bold', fontsize=9, ha='right', va='bottom')
        lines = [(f"Label: {NICE[truth]}  ·  {SRC[it['source']]}", '#0b0b0b', 'normal'),
                 (f"L: {NICE[LAB[li]]} (p = {L[li]:.2f})   G61: {NICE.get(r['g'], r['g'])} (conf. {g.get('confidence')})", '#0b0b0b', 'normal'),
                 (f"G61 report: {symptoms(sym)}", '#52514e', 'normal'),
                 (f"System → {sysout}", OUTCOL[tag], 'bold')]
        for j, (t, c, fw) in enumerate(lines):
            ax.text(0.0, -0.035 - j * 0.066, t, transform=ax.transAxes, va='top', ha='left', fontsize=6.2, color=c, fontweight=fw)
        out.append(['abcdef'[k], coll, it['source'], name, truth, LAB[li], round(float(L[li]), 3), r['g'], g.get('confidence'),
                    json.dumps(sym), sysout, f'{zx},{zy},{zs}'])
    for ext, kw in (('pdf', {}), ('svg', {}), ('png', {'dpi': 600}), ('tif', {'dpi': 600, 'pil_kwargs': {'compression': 'tiff_lzw'}})):
        fig.savefig(HERE / f'Fig4.{ext}', bbox_inches='tight', pad_inches=0.03, **kw)
    with (HERE / 'Fig4_cases.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['panel', 'collection', 'source', 'file', 'truth', 'L_pred', 'L_prob', 'G61_pred', 'G61_conf', 'G61_symptoms', 'system_output',
                    'zoom_box_fraction_x0_y0_size'])
        w.writerows(out)
    print('\n'.join(str(o[:9] + o[10:]) for o in out))


if __name__ == '__main__':
    main()
