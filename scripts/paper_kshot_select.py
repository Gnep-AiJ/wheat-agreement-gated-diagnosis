"""Source-adaptive expert selection with K labelled target images (protocol section 21). No new model calls."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from paper_arbiter import LAB, load_rows  # noqa: E402
from paper_arbiter_k import vote3  # noqa: E402


def main() -> None:
    rows = load_rows(); colls = ['EVAL450', 'ETS', 'ETS2', 'P4']
    rules = ['L', 'G61+L', 'RAG+L', 'vote3']
    P = {}
    for r in rows:
        l = LAB[int(np.argmax(r['L']))]
        r['pred'] = {'L': l, 'G61+L': r['g'] if r['g'] in LAB else l, 'RAG+L': r['r'] if r['r'] in LAB else l, 'vote3': vote3(r, l)}
    ok = {k: np.array([r['pred'][k] in r['truth'] for r in rows]) for k in rules}
    src = np.array([r['source'] for r in rows]); coll = np.array([r['coll'] for r in rows])
    res = {}
    for K in (5, 10, 20):
        acc_sum = np.zeros(len(rows)); cnt = np.zeros(len(rows)); rng = np.random.default_rng(20261008 + K)
        for s in sorted(set(src)):
            idx = np.where(src == s)[0]
            if len(idx) < 2 * K:
                acc_sum[idx] += ok['vote3'][idx] * 50; cnt[idx] += 50
                continue
            for _ in range(50):
                cal = rng.choice(idx, K, replace=False); rest = np.setdiff1d(idx, cal)
                sc = {k: ok[k][cal].mean() for k in rules}; best = max(rules, key=lambda k: (sc[k], k == 'vote3'))
                acc_sum[rest] += ok[best][rest]; cnt[rest] += 1
        a = np.divide(acc_sum, cnt, out=np.full(len(rows), np.nan), where=cnt > 0); m = cnt > 0
        res[f'K={K}'] = dict(pooled=float(np.nanmean(a[m])), **{c: float(np.nanmean(a[m & (coll == c)])) for c in colls})
    oracle_sel = np.zeros(len(rows))
    for s in sorted(set(src)):
        idx = np.where(src == s)[0]; best = max(rules, key=lambda k: ok[k][idx].mean()); oracle_sel[idx] = ok[best][idx]
    res['oracle_selection'] = dict(pooled=float(oracle_sel.mean()), **{c: float(oracle_sel[coll == c].mean()) for c in colls})
    for k in rules:
        res[k] = dict(pooled=float(ok[k].mean()), **{c: float(ok[k][coll == c].mean()) for c in colls})
    orc = np.array([(r['pred']['L'] in r['truth']) or (r['g'] in r['truth']) for r in rows])
    res['oracle(L|G61)'] = dict(pooled=float(orc.mean()), **{c: float(orc[coll == c].mean()) for c in colls})
    for k, v in res.items():
        print(f'{k:18s} pooled {v["pooled"]:.3f} | ' + ' '.join(f'{c} {v[c]:.3f}' for c in colls))
    k10 = res['K=10']; v = res['vote3']
    res['primary'] = dict(gain_pp=100 * (k10['pooled'] - v['pooled']), per_coll_pp={c: 100 * (k10[c] - v[c]) for c in colls},
                          pass_=bool(k10['pooled'] - v['pooled'] >= .03 and min(k10[c] - v[c] for c in colls) >= -.01))
    print('PRIMARY', res['primary'])
    (ROOT / 'outputs/paper_hrme/arbiter/kshot_results.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
