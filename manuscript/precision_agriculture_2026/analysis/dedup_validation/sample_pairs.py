"""Visual validation of the near-duplicate threshold (cached pretrained DINOv3 features; no model calls).
For random query images, the most similar image in another collection is found; pairs are stratified by similarity bin and
rendered as side-by-side montages for visual rating (same photograph incl. crop/resize/flip/colour change vs different)."""
import csv, json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[3]
BINS = [(0.80, 0.85), (0.85, 0.88), (0.88, 0.90), (0.90, 0.92), (0.92, 0.95), (0.95, 1.01)]
PER_BIN = 20

def main():
    with (ROOT / 'outputs/stage0/data_audit_v2_six_sources/image_manifest.csv').open(encoding='utf-8-sig') as f:
        rows = [r for r in csv.DictReader(f) if r['readable'] in ('True', 'true', '1', 'yes', '')]
    seen, items = set(), []
    for r in rows:
        k = (r['dataset'], r['sha256'])
        if k not in seen:
            seen.add(k); items.append(r)
    X = np.load(ROOT / 'outputs/paper_hrme/leak_audit/feats.npy').astype(np.float32)
    assert len(X) == len(items), (len(X), len(items))
    ds = np.array([r['dataset'] for r in items]); rng = np.random.default_rng(20261008)
    q = rng.permutation(len(items))[:6000]
    best = []
    for i in q:
        other = ds != ds[i]
        s = X[other] @ X[i]; j = np.where(other)[0][int(np.argmax(s))]
        best.append((int(i), int(j), float(s.max())))
    out = []
    for lo, hi in BINS:
        cand = [b for b in best if lo <= b[2] < hi]
        pick = [cand[k] for k in rng.permutation(len(cand))[:PER_BIN]]
        for i, j, s in pick:
            out.append(dict(bin=f'{lo:.2f}-{min(hi,1):.2f}', sim=round(s, 4), a_dataset=ds[i], a_path=items[i]['absolute_path'],
                            b_dataset=ds[j], b_path=items[j]['absolute_path']))
    json.dump(out, open(HERE / 'pairs.json', 'w', encoding='utf-8'), indent=1)
    # montages: 10 pairs per sheet
    for s0 in range(0, len(out), 10):
        sheet = Image.new('RGB', (2 * 260 + 40, 10 * 280), 'white'); d = ImageDraw.Draw(sheet)
        for k, p in enumerate(out[s0:s0 + 10]):
            for c, path in enumerate((p['a_path'], p['b_path'])):
                try:
                    im = Image.open(path).convert('RGB'); im.thumbnail((256, 256))
                except Exception:
                    im = Image.new('RGB', (256, 256), 'grey')
                sheet.paste(im, (c * 290 + 2, k * 280 + 20))
            d.text((2, k * 280 + 2), f"#{s0 + k}  sim {p['sim']:.3f}  {p['a_dataset'][:22]} | {p['b_dataset'][:22]}", fill='black')
        sheet.save(HERE / f'sheet_{s0 // 10:02d}.jpg', quality=85)
    print(len(out), 'pairs;', {b: sum(o['bin'] == b for o in out) for b in sorted({o['bin'] for o in out})})

if __name__ == '__main__':
    main()
