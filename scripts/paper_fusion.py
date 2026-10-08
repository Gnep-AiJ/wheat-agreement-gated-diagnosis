"""Fusion of experts for forced diagnosis (protocol 14.2): chosen on EVAL450 only (nested source-held CV estimate + final choice).

Experts: L (DINOv3 9-model mean probs), G61 (plain gpt-6.1-sol), RAG (retrieval-augmented gpt-6.1-sol).
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import paper_p3_hrme as H  # noqa: E402

LAB = H.LABELS


def expert_tuple(L, g, r):
    def gp(v):
        return (v['primary'] if v.get('status') == 'ok' and v['primary'] != 'uncertain' else None, (v.get('conf') or 0) / 100)
    return dict(L=np.asarray(L), G=gp(g), R=gp(r))


def fuse(x, cfg):
    kind = cfg[0]
    if kind == 'L':
        return LAB[int(np.argmax(x['L']))]
    if kind == 'G':
        return x['G'][0] or LAB[int(np.argmax(x['L']))]
    if kind == 'RAG':
        return x['R'][0] or LAB[int(np.argmax(x['L']))]
    if kind == 'vote':
        votes = [x['R'][0], x['G'][0], LAB[int(np.argmax(x['L']))]]
        for c in votes:
            if c and votes.count(c) >= 2:
                return c
        return x['R'][0] or x['G'][0] or votes[2]
    a, b, c = cfg[1:]
    s = c * x['L'] / max(x['L'].sum(), 1e-9)
    for (lab, conf), w in ((x['R'], a), (x['G'], b)):
        if lab:
            s = s + w * conf * (np.arange(6) == LAB.index(lab))
    return LAB[int(np.argmax(s))]


CANDS = [('L',), ('G',), ('RAG',), ('vote',)] + [('w', a, b, c) for a, b, c in itertools.product((0.5, 1, 2), (0, 0.5, 1), (0, 0.5, 1, 2))]


def main() -> None:
    per = json.loads((H.S1 / 'per_record.json').read_text()); ev = json.loads((H.S1 / 'sets.json').read_text())['eval']
    L = H.load_L('dino'); g = json.loads((ROOT / 'outputs/paper_hrme/p2_g61/g61_per_record.json').read_text())
    r = json.loads((ROOT / 'outputs/paper_hrme/rag/rag_full.json').read_text())
    X = {k: expert_tuple(L[k], g[k], r[k]) for k in ev}
    acc = lambda cfg, ids: sum(fuse(X[k], cfg) in per[k]['truth'] for k in ids)  # noqa: E731
    full = {str(c): acc(c, ev) for c in CANDS}
    for c in CANDS[:4]:
        print(c, full[str(c)], '/450')
    best_w = max(CANDS[4:], key=lambda c: acc(c, ev)); print('best weighted', best_w, full[str(best_w)])
    folds = {f: [k for k in ev if per[k]['fold'] == f] for f in range(3)}
    nested, picks = 0, []
    for f in range(3):
        fit = [k for h in range(3) if h != f for k in folds[h]]
        c = max(CANDS, key=lambda c: (acc(c, fit), -CANDS.index(c)))
        picks.append(c); nested += acc(c, folds[f])
    final = max(CANDS, key=lambda c: (acc(c, ev), -CANDS.index(c)))
    print('nested-CV estimate', nested, '/450 picks', picks, '| final choice on all 450:', final, full[str(final)])
    (ROOT / 'outputs/paper_hrme/fusion_dev.json').write_text(json.dumps(dict(full=full, nested=nested, picks=[list(p) for p in picks],
                                                                              final=list(final)), indent=1))


if __name__ == '__main__':
    main()
