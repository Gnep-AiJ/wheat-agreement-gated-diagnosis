"""HRME-v3 under the strict (informative-statement) risk metric, nested source-held selection on EVAL450 (plan section 10)."""
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import paper_p3_hrme as H  # noqa: E402

V3_GRID = [g + (v, gh, gg) for g in itertools.product(H.T_H, H.T_H, H.T_H) for v in (None, 70, 80, 90, 95)
           for gh in (False, True) for gg in (False, True)]


def main(arm='dino'):
    H.STRICT[0] = True
    per = json.loads((H.S1 / 'per_record.json').read_text())
    ev = json.loads((H.S1 / 'sets.json').read_text())['eval']
    L, S, G = H.load_L(arm), H.load_S(), H.load_G('g61', per, ev)
    recs = {r: dict(truth=set(per[r]['truth']), fold=per[r]['fold'], L=L[r], S=S[r], G=G[r]) for r in ev}
    folds = {f: [r for r in ev if recs[r]['fold'] == f] for f in range(3)}
    res = {}
    for alpha in (.05, .10):
        out = {'HRME_v3': H.nested(recs, folds, lambda x, c: H.hrme_v3(x, *c), V3_GRID, alpha),
               'HRME_noG_strict': H.nested(recs, folds, lambda x, c: H.hrme(x, *c, mode='noG'), list(itertools.product(H.T_H, H.T_H, H.T_H)), alpha),
               'B1_flat_L': H.nested(recs, folds, H.flat_L, [(t,) for t in H.T_B], alpha),
               'B2_flat_G': H.nested(recs, folds, H.flat_G, [(t,) for t in H.T_G], alpha)}
        for k, v in out.items():
            print(alpha, k, {q: (round(w, 3) if isinstance(w, float) else w) for q, w in v.items()})
        res[f'alpha_{alpha}'] = out
    (H.OUT / f'p3_v3_L{arm}.json').write_text(json.dumps(res, indent=1, default=str))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'dino')
