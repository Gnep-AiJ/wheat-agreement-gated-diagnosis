"""Arbiter study (protocol section 19): leave-one-source-out stacking over L + G61 (+symptom attributes) + RAG.

Pools: EVAL450 (8 sources), ETS (6), ETS2 (1), P4 iNat (1). Predictions only from cached expert outputs.
"""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
import paper_p2_codex_g61 as P  # noqa: E402
import paper_p3_hrme as H  # noqa: E402

LAB = H.LABELS
OUT = ROOT / 'outputs/paper_hrme/arbiter'
RESP = ROOT / 'outputs/paper_hrme/p2_g61/responses'
CAT = {'image_type': ['leaf_closeup', 'stem', 'spike', 'whole_plant', 'canopy_distant', 'lab_or_specimen', 'non_wheat', 'unclear'],
       'assessability': ['clear_lesions_visible', 'clear_no_lesions', 'partially_visible', 'cannot_judge'],
       'pustule_color': ['none', 'yellow', 'orange_brown', 'brick_red', 'dark_brown_black', 'other'],
       'pustule_arrangement': ['none', 'stripes_along_veins', 'scattered', 'elongated_on_stem_or_sheath', 'other'],
       'affected_organ': ['none', 'leaf', 'stem', 'leaf_sheath', 'spike', 'multiple', 'unclear']}


def g61_full(hashes):
    k = hashlib.sha256(json.dumps([hashes, P.PROMPT, P.MODEL, P.EFFORT]).encode()).hexdigest()[:32]
    f = RESP / f'{k}.json'
    return json.loads(f.read_text(encoding='utf-8')).get('parsed') if f.exists() else None


def onehot(v, opts):
    return [float(v == o) for o in opts]


def featurize(L, g, r):
    L = np.asarray(L, float)
    f = {'L': list(np.log(np.clip(L, 1e-6, 1)))}
    gp = (g or {}).get('primary', 'uncertain'); sym = (g or {}).get('symptoms', {}) or {}
    f['G'] = onehot(gp, LAB + ('uncertain',)) + [((g or {}).get('confidence') or 0) / 100]
    f['S'] = (onehot((g or {}).get('image_type'), CAT['image_type']) + onehot((g or {}).get('assessability'), CAT['assessability'])
              + [float(bool(sym.get(k))) for k in ('pustules_present', 'white_powdery_growth', 'necrotic_blotches', 'black_dots_in_lesions')]
              + onehot(sym.get('pustule_color'), CAT['pustule_color']) + onehot(sym.get('pustule_arrangement'), CAT['pustule_arrangement'])
              + onehot(sym.get('affected_organ'), CAT['affected_organ']))
    rp = (r or {}).get('primary', 'uncertain') if (r or {}).get('status') == 'ok' else 'uncertain'
    f['R'] = onehot(rp, LAB + ('uncertain',)) + [((r or {}).get('conf') or 0) / 100]
    return f, gp, rp


def load_rows():
    rows = []
    per = json.loads((H.S1 / 'per_record.json').read_text()); ev = json.loads((H.S1 / 'sets.json'))['eval'] if False else json.loads((H.S1 / 'sets.json').read_text())['eval']
    prep = json.loads((H.S1 / 'prepared.json').read_text()); Ld = H.load_L('dino')
    rag = json.loads((ROOT / 'outputs/paper_hrme/rag/rag_full.json').read_text())
    for k in ev:
        gf = g61_full(prep[k]); f, gp, rp = featurize(Ld[k], gf, rag[k])
        rows.append(dict(coll='EVAL450', source='eval:' + per[k]['source'], truth=set(per[k]['truth']), f=f, g=gp, r=rp, L=Ld[k], gfull=gf))
    for coll, d in (('ETS', 'ets'), ('ETS2', 'ets2'), ('P4', 'p4_rag')):
        E = ROOT / 'outputs/paper_hrme' / d
        items = json.loads((E / 'items.json').read_text()); prepd = json.loads((E / 'prepared.json').read_text())
        rj = json.loads((E / 'rag.json').read_text())
        if d == 'p4_rag':
            allit = json.loads((ROOT / 'outputs/paper_hrme/p4_inat/items.json').read_text())
            Lall = np.load(ROOT / 'outputs/paper_hrme/p4_inat/L_dino.npy'); idx = {a['obs']: i for i, a in enumerate(allit)}
            Lmat = [Lall[idx[it['obs']]] for it in items]
        else:
            Lmat = np.load(E / 'L_dino.npy')
        for i, it in enumerate(items):
            gf = g61_full(prepd[it['file']]); f, gp, rp = featurize(Lmat[i], gf, rj[it['file']])
            rows.append(dict(coll=coll, source=f'{d}:' + it.get('source', d), truth={it['label']}, f=f, g=gp, r=rp, L=np.asarray(Lmat[i]), gfull=gf))
    return rows


def X_of(rows, parts):
    return np.array([sum((r['f'][p] for p in parts), []) for r in rows])


def fit_predict(kind, Xtr, rows_tr, Xte):
    from sklearn.preprocessing import StandardScaler
    ys, Xs, ws = [], [], []
    for x, r in zip(Xtr, rows_tr):
        for t in sorted(r['truth']):
            ys.append(t); Xs.append(x); ws.append(1 / len(r['truth']))
    Xs, ys, ws = np.array(Xs), np.array(ys), np.array(ws)
    if kind == 'M1':
        from sklearn.linear_model import LogisticRegression
        sc = StandardScaler().fit(Xs)
        m = LogisticRegression(C=1.0, max_iter=5000, class_weight='balanced').fit(sc.transform(Xs), ys, sample_weight=ws)
        return m.predict(sc.transform(Xte))
    from sklearn.ensemble import HistGradientBoostingClassifier
    m = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(Xs, ys, sample_weight=ws)
    return m.predict(Xte)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    srcs = sorted({r['source'] for r in rows}); colls = ['EVAL450', 'ETS', 'ETS2', 'P4']
    print('rows', len(rows), 'sources', len(srcs), {c: sum(r['coll'] == c for r in rows) for c in colls})
    preds = {}
    l = [LAB[int(np.argmax(r['L']))] for r in rows]
    preds['L'] = l
    preds['G61+L'] = [r['g'] if r['g'] in LAB else li for r, li in zip(rows, l)]
    preds['RAG+L'] = [r['r'] if r['r'] in LAB else li for r, li in zip(rows, l)]
    vote = []
    for r, li in zip(rows, l):
        v = [r['r'] if r['r'] in LAB else None, r['g'] if r['g'] in LAB else None, li]
        vote.append(next((c for c in v if c and v.count(c) >= 2), v[0] or v[1] or li))
    preds['vote3'] = vote
    for kind in ('M1', 'M2'):
        for parts in (('L',), ('L', 'G'), ('L', 'G', 'S'), ('L', 'G', 'S', 'R')):
            name = f'{kind}[{"+".join(parts)}]'; X = X_of(rows, parts); p = [None] * len(rows)
            for s in srcs:
                te = [i for i, r in enumerate(rows) if r['source'] == s]; tr = [i for i, r in enumerate(rows) if r['source'] != s]
                for i, q in zip(te, fit_predict(kind, X[tr], [rows[i] for i in tr], X[te])):
                    p[i] = q
            preds[name] = p
    ok = {n: np.array([q in r['truth'] for q, r in zip(v, rows)]) for n, v in preds.items()}
    ok['oracle(L|G61)'] = np.array([(LAB[int(np.argmax(r['L']))] in r['truth']) or (r['g'] in r['truth']) for r in rows])
    table = {}
    for n, v in ok.items():
        table[n] = dict(pooled=float(v.mean()), **{c: float(v[[r['coll'] == c for r in rows]].mean()) for c in colls})
        print(f'{n:22s} pooled {table[n]["pooled"]:.3f} | ' + ' '.join(f'{c} {table[n][c]:.3f}' for c in colls))
    singles = ['L', 'G61+L', 'RAG+L', 'vote3']
    best = max(singles, key=lambda n: ok[n].mean()); main_ = 'M1[L+G+S+R]'
    d = ok[main_].astype(float) - ok[best]; rng = np.random.default_rng(20261008)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]; lo, hi = np.percentile(bs, [2.5, 97.5])
    worst = {c: table[main_][c] - max(table[s][c] for s in singles) for c in colls}
    res = dict(table=table, primary=dict(model=main_, best_single=best, gain_pp=100 * float(d.mean()), ci95_pp=[100 * lo, 100 * hi],
                                         per_collection_vs_best_single_pp={c: 100 * v for c, v in worst.items()},
                                         pass_=bool(d.mean() >= .05 and lo > 0 and min(worst.values()) >= -.02)))
    print('PRIMARY', res['primary'])
    (OUT / 'arbiter_results.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
