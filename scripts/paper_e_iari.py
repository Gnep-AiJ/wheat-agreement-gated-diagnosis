"""Paper HRME external validation on IARI 852 (protocol section 6b). Thresholds frozen from full EVAL450.

Usage: python scripts/paper_e_iari.py g61   (Codex calls for IARI images, cached)
       python scripts/paper_e_iari.py eval --G g61|none
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import paper_p3_hrme as H  # noqa: E402

OUT = ROOT / 'outputs/paper_hrme'


def iari_rows():
    from scripts.mainline_v2_final import test_sets
    rows = test_sets()['iari']
    a = np.load(ROOT / 'outputs/mainline_v2/stage5_posthoc/arrays.npz')
    truth = a['truth'][a['group'] == 'iari']
    assert [r[2] for r in rows] == list(truth)
    return rows, a['F_iari'].mean(0)


def g61_calls() -> None:
    from scripts.mainline_v2_vlm import prepare
    import paper_p2_codex_g61 as P
    P.install()
    rows, _ = iari_rows()
    prepared = {r[0]: prepare(r[1]) for r in rows}
    with ThreadPoolExecutor(4) as ex:
        recs = list(ex.map(lambda r: (r[0], P.call(prepared[r[0]])), rows))
    out = {k: (dict(status='ok', primary=v['parsed']['primary'], conf=v['parsed']['confidence']) if v['status'] == 'ok'
               else dict(status=v['status'])) for k, v in recs}
    (OUT / 'e_iari_g61.json').write_text(json.dumps(out, indent=0))
    print({s: sum(v['status'] == s for v in out.values()) for s in {v['status'] for v in out.values()}})


def evaluate(G: str) -> None:
    per = json.loads((H.S1 / 'per_record.json').read_text())
    ev = json.loads((H.S1 / 'sets.json').read_text())['eval']
    Ld, S = H.load_L('dino'), H.load_S()
    Gd = H.load_G('g61', per, ev) if G == 'g61' else {r: (None, -1) for r in ev}
    dev = {r: dict(truth=set(per[r]['truth']), L=Ld[r], S=S[r], G=Gd[r]) for r in ev}
    rows, Fi = iari_rows()
    gi = json.loads((OUT / 'e_iari_g61.json').read_text()) if G == 'g61' else {}

    def gget(k):
        v = gi.get(k)
        return ((v['primary'] if v['primary'] != 'uncertain' else None), v['conf']) if v and v['status'] == 'ok' else (None, -1)
    test = {r[0]: dict(truth={r[2]}, L=Fi[i], S=None, G=gget(r[0])) for i, r in enumerate(rows)}
    import itertools
    res = {}
    for alpha in (.05, .10):
        out = {}
        variants = {'HRME_noG': (lambda x, c: H.hrme(x, *c, mode='noG'), list(itertools.product(H.T_H, H.T_H, H.T_H))),
                    'B1_flat_L': (H.flat_L, [(t,) for t in H.T_B])}
        if G == 'g61':
            variants['HRME'] = (lambda x, c: H.hrme(x, *c, mode='G'), list(itertools.product(H.T_H, H.T_H, H.T_H)))
        for name, (fn, grid) in variants.items():
            best = None
            for cfg in grid:
                s = H.summarize([fn(dev[r], cfg) for r in ev], dev, ev)
                if s['released'] and s['risk'] <= alpha:
                    key = (s['informative_coverage'], s['specific_coverage'])
                    if best is None or key > best[0]:
                        best = (key, cfg)
            ids = list(test)
            s = H.summarize([fn(test[r], best[1]) if best else ('abstain', None) for r in ids], test, ids)
            s['frozen_cfg'] = best[1] if best else None
            out[name] = s
            print(alpha, name, {q: (round(w, 3) if isinstance(w, float) else w) for q, w in s.items()})
        res[f'alpha_{alpha}'] = out
    (OUT / f'e_iari_results_G{G}.json').write_text(json.dumps(res, indent=1, default=str))


if __name__ == '__main__':
    if sys.argv[1] == 'g61':
        g61_calls()
    else:
        evaluate(sys.argv[3] if len(sys.argv) > 3 else 'none')
