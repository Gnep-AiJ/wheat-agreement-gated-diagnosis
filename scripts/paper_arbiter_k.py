"""Knowledge-rule arbiter K (protocol section 20): no training; symptom-evidence disambiguation between L and G61."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from paper_arbiter import LAB, load_rows  # noqa: E402


def support(c, g):
    s = (g or {}).get('symptoms', {}) or {}; pp = bool(s.get('pustules_present')); col = s.get('pustule_color'); arr = s.get('pustule_arrangement')
    crit = {'yellow_rust': [pp, col == 'yellow', arr == 'stripes_along_veins'],
            'leaf_rust': [pp, col == 'orange_brown', arr == 'scattered'],
            'stem_rust': [pp, col in ('brick_red', 'dark_brown_black'),
                          arr == 'elongated_on_stem_or_sheath' or s.get('affected_organ') in ('stem', 'leaf_sheath')],
            'powdery_mildew': [bool(s.get('white_powdery_growth'))],
            'septoria': [bool(s.get('necrotic_blotches')), bool(s.get('black_dots_in_lesions'))],
            'healthy': [(g or {}).get('assessability') == 'clear_no_lesions', not pp, not s.get('necrotic_blotches'),
                        not s.get('white_powdery_growth')]}[c]
    return sum(crit) / len(crit)


def vote3(r, l):
    v = [r['r'] if r['r'] in LAB else None, r['g'] if r['g'] in LAB else None, l]
    return next((c for c in v if c and v.count(c) >= 2), v[0] or v[1] or l)


def arbiter_k(r):
    l = LAB[int(np.argmax(r['L']))]; g = r['g']; rag = r['r'] if r['r'] in LAB else None
    if g == l:
        return l
    if g not in LAB:
        return rag or l
    sl, sg = support(l, r['gfull']), support(g, r['gfull'])
    if sl != sg:
        return l if sl > sg else g
    return rag if rag in (l, g) else vote3(r, l)


def main() -> None:
    rows = load_rows()
    colls = ['EVAL450', 'ETS', 'ETS2', 'P4']
    preds = {'L': [], 'G61+L': [], 'RAG+L': [], 'vote3': [], 'K (knowledge arbiter)': []}
    for r in rows:
        l = LAB[int(np.argmax(r['L']))]
        preds['L'].append(l); preds['G61+L'].append(r['g'] if r['g'] in LAB else l)
        preds['RAG+L'].append(r['r'] if r['r'] in LAB else l); preds['vote3'].append(vote3(r, l)); preds['K (knowledge arbiter)'].append(arbiter_k(r))
    ok = {n: np.array([q in r['truth'] for q, r in zip(v, rows)]) for n, v in preds.items()}
    table = {}
    for n, v in ok.items():
        table[n] = dict(pooled=float(v.mean()), **{c: float(v[[r['coll'] == c for r in rows]].mean()) for c in colls})
        print(f'{n:24s} pooled {table[n]["pooled"]:.3f} | ' + ' '.join(f'{c} {table[n][c]:.3f}' for c in colls))
    singles = ['L', 'G61+L', 'RAG+L', 'vote3']; best = max(singles, key=lambda n: ok[n].mean())
    d = ok['K (knowledge arbiter)'].astype(float) - ok[best]; rng = np.random.default_rng(20261008)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]; lo, hi = np.percentile(bs, [2.5, 97.5])
    worst = {c: 100 * (table['K (knowledge arbiter)'][c] - max(table[s][c] for s in singles)) for c in colls}
    res = dict(table=table, primary=dict(best_single=best, gain_pp=100 * float(d.mean()), ci95_pp=[100 * lo, 100 * hi],
                                         per_collection_vs_best_pp=worst, pass_=bool(d.mean() >= .05 and lo > 0 and min(worst.values()) >= -2)))
    print('PRIMARY', res['primary'])
    (ROOT / 'outputs/paper_hrme/arbiter/arbiter_k_results.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
