"""One-shot confirmatory evaluation on ETS2 (protocol section 17): H1 adaptive RAG A3 vs A0, H2 A0 vs L, H3 abstention v3.

Inputs: outputs/paper_hrme/ets2/{items.json, L_dino.npy, g61.json, rag.json}. Run once.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import paper_p3_hrme as H  # noqa: E402
from paper_ets_eval import macro_f1  # noqa: E402

E = ROOT / 'outputs/paper_hrme/ets2'
LAB = H.LABELS


def P(v):
    return v['primary'] if v.get('status') == 'ok' and v.get('primary') not in (None, 'uncertain') else None


def rule(name, L, g, r):
    l = LAB[int(np.argmax(L))]
    if name == 'L':
        return l
    if name == 'A0':
        return g or l
    if name == 'A4':
        return r or l
    if g is None or g != l:  # A3 trigger
        return r or l
    return g


def boot_ci(a, b, seed=20261008):
    d = a.astype(float) - b.astype(float); rng = np.random.default_rng(seed)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
    return 100 * float(d.mean()), [100 * float(x) for x in np.percentile(bs, [2.5, 97.5])]


def main() -> None:
    items = json.loads((E / 'items.json').read_text()); Lp = np.load(E / 'L_dino.npy')
    g = json.loads((E / 'g61.json').read_text()); r = json.loads((E / 'rag.json').read_text())
    y = [it['label'] for it in items]
    pred = {n: [rule(n, Lp[i], P(g[it['file']]), P(r[it['file']])) for i, it in enumerate(items)] for n in ('L', 'A0', 'A3', 'A4')}
    ok = {n: np.array([a == b for a, b in zip(v, y)]) for n, v in pred.items()}
    res = dict(n=len(y), labels={c: y.count(c) for c in LAB if c in y}, systems={})
    for n in pred:
        bc = defaultdict(list)
        for o, c in zip(ok[n], y):
            bc[c].append(o)
        res['systems'][n] = dict(top1=int(ok[n].sum()), acc=float(ok[n].mean()), macro_f1=macro_f1(y, pred[n]),
                                 by_class={c: f'{int(sum(v))}/{len(v)}' for c, v in sorted(bc.items())})
        print(n, res['systems'][n])
    trig = np.mean([(P(g[it['file']]) is None or P(g[it['file']]) != LAB[int(np.argmax(Lp[i]))]) for i, it in enumerate(items)])
    d1, c1 = boot_ci(ok['A3'], ok['A0']); d2, c2 = boot_ci(ok['A0'], ok['L']); d4, c4 = boot_ci(ok['A4'], ok['A0'])
    res['H1_A3_vs_A0'] = dict(gain_pp=d1, ci95=c1, rag_call_rate=float(trig), pass_=bool(d1 >= 3 and c1[0] > 0))
    res['H2_A0_vs_L'] = dict(gain_pp=d2, ci95=c2, pass_=bool(d2 >= 3 and c2[0] > 0))
    res['A4_vs_A0'] = dict(gain_pp=d4, ci95=c4)
    H.STRICT[0] = True
    test = {i: dict(truth={y[i]}, L=Lp[i], S=None, G=(P(g[it['file']]), g[it['file']].get('conf') or 0)) for i, it in enumerate(items)}
    s = H.summarize([H.hrme_v3(test[i], .5, .5, .5, 95, True, True) for i in test], test, list(test))
    res['H3_abstention'] = dict(s, pass_=bool(s['risk'] <= .05 and s['risk_cp95'] <= .08 and s['informative_coverage'] >= .60))
    for k in ('H1_A3_vs_A0', 'H2_A0_vs_L', 'A4_vs_A0', 'H3_abstention'):
        print(k, res[k])
    (E / 'ets2_results.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
