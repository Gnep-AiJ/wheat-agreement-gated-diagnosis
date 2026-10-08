"""Final one-shot ETS evaluation (protocol 14.2): frozen system RAG-VLM + local fallback vs single experts.

Inputs (predictions only): outputs/paper_hrme/ets/{items.json, L_dino.npy, g61.json, rag.json}.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import paper_p3_hrme as H  # noqa: E402
from paper_fusion import expert_tuple, fuse  # noqa: E402

E = ROOT / 'outputs/paper_hrme/ets'
LAB = H.LABELS


def macro_f1(y, p):
    f = []
    for c in sorted(set(y)):
        tp = sum(a == c and b == c for a, b in zip(y, p)); fp = sum(a != c and b == c for a, b in zip(y, p)); fn = sum(a == c and b != c for a, b in zip(y, p))
        f.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f))


def main() -> None:
    items = json.loads((E / 'items.json').read_text())
    L = np.load(E / 'L_dino.npy'); g = json.loads((E / 'g61.json').read_text()); r = json.loads((E / 'rag.json').read_text())
    X = [expert_tuple(L[i], g[it['file']], r[it['file']]) for i, it in enumerate(items)]
    y = [it['label'] for it in items]; src = [it['source'] for it in items]
    systems = {'L (DINOv3)': ('L',), 'G61 + L fallback': ('G',), 'RAG-VLM + L fallback (frozen system)': ('RAG',)}
    pred = {k: [fuse(x, c) for x in X] for k, c in systems.items()}
    correct = {k: np.array([a == b for a, b in zip(v, y)]) for k, v in pred.items()}
    res = dict(n=len(y), labels={c: y.count(c) for c in LAB}, systems={})
    for k in systems:
        by_s, by_c = defaultdict(list), defaultdict(list)
        for ok, s, c in zip(correct[k], src, y):
            by_s[s].append(ok); by_c[c].append(ok)
        res['systems'][k] = dict(top1=int(correct[k].sum()), acc=float(correct[k].mean()), macro_f1=macro_f1(y, pred[k]),
                                 by_source={s: f'{int(sum(v))}/{len(v)}' for s, v in sorted(by_s.items())},
                                 by_class={c: f'{int(sum(v))}/{len(v)}' for c, v in sorted(by_c.items())})
        print(k, res['systems'][k])
    fz = correct['RAG-VLM + L fallback (frozen system)']
    best_single = max(('L (DINOv3)', 'G61 + L fallback'), key=lambda k: correct[k].mean())
    d = fz.astype(float) - correct[best_single].astype(float)
    rng = np.random.default_rng(20261008)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    res['primary'] = dict(best_single=best_single, gain_pp=100 * float(d.mean()), ci95_pp=[100 * float(lo), 100 * float(hi)],
                          pass_=bool(d.mean() >= .05 and lo > 0))
    print('PRIMARY', res['primary'])
    # secondary: hierarchical abstention, frozen v3 alpha=5% configuration, G = plain G61 and (exploratory) G = RAG
    H.STRICT[0] = True
    for name, key in (('G61', 'G'), ('RAG', 'R')):
        test = {i: dict(truth={y[i]}, L=X[i]['L'], S=None, G=(X[i][key][0], X[i][key][1] * 100)) for i in range(len(y))}
        ids = list(test)
        for cfg in ((0.5, 0.5, 0.5, 95, True, True),):
            s = H.summarize([H.hrme_v3(test[i], *cfg) for i in ids], test, ids)
            res[f'abstention_v3_alpha5_G={name}'] = s
            print('abstention v3 a5', name, {q: (round(w, 3) if isinstance(w, float) else w) for q, w in s.items()})
    (E / 'ets_results.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
