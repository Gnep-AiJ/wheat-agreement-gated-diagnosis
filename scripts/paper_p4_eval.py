"""Paper HRME P4: fresh iNaturalist pathogen-observation test (protocol section 9). Run once.

Usage: python scripts/paper_p4_eval.py infer   (local L on CPU + G61 via Codex, cached)
       python scripts/paper_p4_eval.py eval --L dino
"""
import itertools
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import paper_p3_hrme as H  # noqa: E402

P4 = ROOT / 'outputs/paper_hrme/p4_inat'
FOMO_MODELS = [f'outputs/paper_hrme/p1_fomo/fomo_inat_seed{s}_fold{f}/model.pt' for s in (42, 43, 44) for f in range(3)]
DINO_MODELS = [f'outputs/mainline_v2/stage3/dino_inat_seed{s}_fold{f}/model.pt' for s in (42, 43, 44) for f in range(3)]


def local_probs(arm: str, items):
    import torch
    from PIL import Image
    import scripts.mainline_v2_train_inat  # noqa: F401  (transformers import order)
    from scripts.strong_expert_train import Expert
    x = np.stack([np.array(Image.open(it['file']).convert('RGB').resize((384, 384), Image.Resampling.BILINEAR)) for it in items])
    x = (torch.from_numpy(x).permute(0, 3, 1, 2).float() / 255 - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]
    out = []
    for path in (DINO_MODELS if arm == 'dino' else FOMO_MODELS):
        if arm == 'dino':
            m = Expert('dino')
        else:
            from scripts.paper_p1_fomo_train import FomoExpert
            m = FomoExpert()
        m.load_state_dict(torch.load(ROOT / path, map_location='cpu', weights_only=True)); m.eval()
        with torch.inference_mode():
            out.append(torch.cat([m(x[i:i + 8]).sigmoid() for i in range(0, len(x), 8)]).numpy())
    return np.mean(out, 0)


def infer() -> None:
    items = json.loads((P4 / 'items.json').read_text())
    np.save(P4 / 'L_dino.npy', local_probs('dino', items))
    if all((ROOT / p).exists() for p in FOMO_MODELS):
        np.save(P4 / 'L_fomo.npy', local_probs('fomo', items))
    from scripts.mainline_v2_vlm import prepare
    import paper_p2_codex_g61 as P
    P.install()
    with ThreadPoolExecutor(4) as ex:
        recs = list(ex.map(lambda it: P.call(prepare(it['file'])), items))
    g = {str(it['obs']): (dict(status='ok', primary=v['parsed']['primary'], conf=v['parsed']['confidence']) if v['status'] == 'ok'
                          else dict(status=v['status'])) for it, v in zip(items, recs)}
    (P4 / 'g61.json').write_text(json.dumps(g, indent=0))
    print('g61 status', {s: sum(v['status'] == s for v in g.values()) for s in {v['status'] for v in g.values()}})


def evaluate(arm: str) -> None:
    H.STRICT[0] = True  # protocol section 10: strict (informative-statement) risk metric
    from paper_p3_v3 import V3_GRID
    per = json.loads((H.S1 / 'per_record.json').read_text())
    ev = json.loads((H.S1 / 'sets.json').read_text())['eval']
    Ld, S, Gd = H.load_L(arm), H.load_S(), H.load_G('g61', per, ev)
    dev = {r: dict(truth=set(per[r]['truth']), L=Ld[r], S=S[r], G=Gd[r]) for r in ev}
    items = [it for it in json.loads((P4 / 'items.json').read_text()) if it['keep']]
    allitems = json.loads((P4 / 'items.json').read_text())
    Lt = np.load(P4 / f'L_{arm}.npy' if arm != 'mean' else P4 / 'L_dino.npy')
    if arm == 'mean':
        Lt = (Lt + np.load(P4 / 'L_fomo.npy')) / 2
    idx = {str(it['obs']): i for i, it in enumerate(allitems)}
    g = json.loads((P4 / 'g61.json').read_text())

    def gget(k):
        v = g[k]
        return ((v['primary'] if v['primary'] != 'uncertain' else None), v['conf']) if v['status'] == 'ok' else (None, -1)
    test = {str(it['obs']): dict(truth={it['label']}, L=Lt[idx[str(it['obs'])]], S=None, G=gget(str(it['obs']))) for it in items}
    ids = list(test)
    res = dict(n=len(ids), labels={l: sum(next(iter(test[r]['truth'])) == l for r in ids) for l in H.LABELS},
               forced_L=sum(H.LABELS[int(np.argmax(test[r]['L']))] in test[r]['truth'] for r in ids),
               forced_G61=sum(test[r]['G'][0] in test[r]['truth'] for r in ids))
    grid_h = list(itertools.product(H.T_H, H.T_H, H.T_H))
    variants = {'HRME_v3': (lambda x, c: H.hrme_v3(x, *c), V3_GRID),
                'HRME_v2': (lambda x, c: H.hrme(x, *c[:3], mode='v2', tv=c[3]), [gg + (v,) for gg in grid_h for v in (None, 0, 70, 80, 90, 95)]),
                'HRME_noG': (lambda x, c: H.hrme(x, *c, mode='noG'), grid_h),
                'B1_flat_L': (H.flat_L, [(t,) for t in H.T_B]),
                'B2_flat_G': (H.flat_G, [(t,) for t in H.T_G])}
    for alpha in (.05, .10):
        out = {}
        for name, (fn, grid) in variants.items():
            best = None
            for cfg in grid:
                s = H.summarize([fn(dev[r], cfg) for r in ev], dev, ev)
                if s['released'] and s['risk'] <= alpha:
                    key = (s['informative_coverage'], s['specific_coverage'])
                    if best is None or key > best[0]:
                        best = (key, cfg)
            s = H.summarize([fn(test[r], best[1]) if best else ('abstain', None) for r in ids], test, ids)
            s['frozen_cfg'] = best[1] if best else None
            out[name] = s
            print(alpha, name, {q: (round(w, 3) if isinstance(w, float) else w) for q, w in s.items()})
        res[f'alpha_{alpha}'] = out
    print('n', res['n'], res['labels'], 'forced L', res['forced_L'], 'G61', res['forced_G61'])
    (P4 / f'p4_results_L{arm}.json').write_text(json.dumps(res, indent=1, default=str))


if __name__ == '__main__':
    infer() if sys.argv[1] == 'infer' else evaluate(sys.argv[3] if len(sys.argv) > 3 else 'dino')
