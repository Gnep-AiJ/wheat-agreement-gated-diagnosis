"""Paper HRME P3: hierarchical risk-controlled multi-expert decision vs single-model selective baselines on EVAL450.

Rules: research/PAPER_HRME_PROTOCOL_2026-10-07.md sections 4-5. Thresholds for each outer fold are chosen on the other two
source-held folds only. Usage: python scripts/paper_p3_hrme.py --L dino|fomo|mean --G g6|g61
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
S1 = ROOT / 'outputs/mainline_v2/stage1'
OUT = ROOT / 'outputs/paper_hrme'
LABELS = ('healthy', 'leaf_rust', 'powdery_mildew', 'septoria', 'stem_rust', 'yellow_rust')
RUST = ('leaf_rust', 'stem_rust', 'yellow_rust')
GROUP = {'healthy': 'H', 'leaf_rust': 'R', 'stem_rust': 'R', 'yellow_rust': 'R', 'powdery_mildew': 'PM', 'septoria': 'SEP'}
T_H = (.5, .6, .7, .8, .9, .95, .98)
T_B = (.5, .6, .7, .8, .9, .95, .98, .99, .995, .999)
T_G = (0, 50, 60, 70, 80, 85, 90, 95, 98)
L_TPL = {'dino': 'outputs/mainline_v2/stage3/dino_inat_seed{s}_fold{f}', 'fomo': 'outputs/paper_hrme/p1_fomo/fomo_inat_seed{s}_fold{f}'}


def load_L(arm):
    arms = ('dino', 'fomo') if arm == 'mean' else (arm,)
    out = {}
    for f in range(3):
        ps = [np.load(ROOT / L_TPL[a].format(s=s, f=f) / 'evaluation.npz') for a in arms for s in (42, 43, 44)]
        for r, p in zip(ps[0]['record_ids'], np.mean([x['probabilities'] for x in ps], 0)):
            out[str(r)] = p
    return out


def load_S():
    out = {}
    for f in range(3):
        ss = [np.load(ROOT / f'outputs/mainline_v2/stage6/rust_seed{s}_fold{f}/evaluation.npz') for s in (42, 44)]
        for r, p in zip(ss[0]['record_ids'], np.mean([x['probabilities'] for x in ss], 0)):
            out[str(r)] = RUST[int(np.argmax(p))]
    return out


def load_G(which, per, ev):
    if which == 'g6':
        return {r: (per[r]['gpt']['primary'], per[r]['gpt']['conf'] if per[r]['gpt']['conf'] is not None else -1) for r in ev}
    g = json.loads((OUT / 'p2_g61/g61_per_record.json').read_text())
    return {r: ((g[r]['primary'] if g[r]['primary'] != 'uncertain' else None), g[r]['conf']) if g[r]['status'] == 'ok' else (None, -1)
            for r in ev}


def hrme(x, th, td, tg, mode, tv=None):
    """Return (level, claim): level in {'healthy','diseased','group','specific','abstain'}."""
    L = x['L']
    ph, pd = L[0], L[1:].max()
    if ph >= th and ph >= pd:
        return 'healthy', 'healthy'
    if not (pd >= td and pd > ph):
        return 'abstain', None
    grp = {'R': max(L[1], L[4], L[5]), 'PM': L[2], 'SEP': L[3]}
    g = max(grp, key=grp.get)
    if grp[g] < tg:
        return 'diseased', 'diseased'
    if g == 'PM':
        return 'specific', 'powdery_mildew'
    if g == 'SEP':
        return 'specific', 'septoria'
    if mode == 'noG' or (mode == 'v2' and tv is None):
        return 'group', 'R'
    if mode == 'v2':
        l = RUST[int(np.argmax([L[1], L[4], L[5]]))]
        gp, gc = x['G']
        return ('specific', l) if (gp == l and gc >= tv) else ('group', 'R')
    l = RUST[int(np.argmax([L[1], L[4], L[5]]))]
    other = x['S'] if mode == 'S' else (x['G'][0] if x['G'][0] in RUST else None)
    return ('specific', l) if other == l else ('group', 'R')


def hrme_v3(x, th, td, tg, tv, gate_h, gate_g):
    L = x['L']
    gp = x['G'][0]
    ph, pd = L[0], L[1:].max()
    if ph >= th and ph >= pd:
        if gate_h and gp != 'healthy':
            return 'abstain', None
        return 'healthy', 'healthy'
    if not (pd >= td and pd > ph):
        return 'abstain', None
    grp = {'R': max(L[1], L[4], L[5]), 'PM': L[2], 'SEP': L[3]}
    g = max(grp, key=grp.get)
    if grp[g] < tg or (gate_g and (gp is None or GROUP.get(gp) != g)):
        return 'diseased', 'diseased'
    if g == 'PM':
        return 'specific', 'powdery_mildew'
    if g == 'SEP':
        return 'specific', 'septoria'
    l = RUST[int(np.argmax([L[1], L[4], L[5]]))]
    return ('specific', l) if (tv is not None and gp == l and x['G'][1] >= tv) else ('group', 'R')


def correct(level, claim, truth):
    if level == 'healthy':
        return truth == {'healthy'}
    if level == 'diseased':
        return truth != {'healthy'}
    if level == 'group':
        return claim in {GROUP[t] for t in truth}
    return claim in truth


STRICT = [False]


def summarize(outs, recs, ids):
    n = len(ids)
    rel = [(r, o) for r, o in zip(ids, outs) if o[0] not in (('abstain', 'diseased') if STRICT[0] else ('abstain',))]
    err = sum(not correct(o[0], o[1], recs[r]['truth']) for r, o in rel)
    info = sum(o[0] in ('healthy', 'group', 'specific') for _, o in rel)
    spec = sum(o[0] in ('healthy', 'specific') for _, o in rel)
    k = len(rel)
    return dict(n=n, released=k, errors=err, risk=err / k if k else 0., risk_cp95=float(beta.ppf(.95, err + 1, k - err)) if k else 1.,
                informative_coverage=info / n, specific_coverage=spec / n, diseased_only=sum(o[0] == 'diseased' for _, o in rel) / n,
                abstain=(n - k) / n)


def nested(recs, folds, decide, grid, alpha):
    outs_all, chosen = {}, []
    for f in range(3):
        fit = [r for g in range(3) if g != f for r in folds[g]]
        best = None
        for cfg in grid:
            s = summarize([decide(recs[r], cfg) for r in fit], recs, fit)
            if s['released'] and s['risk'] <= alpha:
                key = (s['informative_coverage'], s['specific_coverage'])
                if best is None or key > best[0]:
                    best = (key, cfg)
        cfg = best[1] if best else None
        chosen.append(cfg)
        for r in folds[f]:
            outs_all[r] = decide(recs[r], cfg) if cfg is not None else ('abstain', None)
    ids = [r for f in range(3) for r in folds[f]]
    s = summarize([outs_all[r] for r in ids], recs, ids)
    s['chosen'] = [list(c) if isinstance(c, tuple) else c for c in chosen]
    return s


def flat_L(x, cfg):
    t = cfg[0]
    i = int(np.argmax(x['L']))
    if x['L'][i] < t:
        return 'abstain', None
    return ('healthy', 'healthy') if i == 0 else ('specific', LABELS[i])


def flat_G(x, cfg):
    t = cfg[0]
    p, c = x['G']
    if p is None or c < t:
        return 'abstain', None
    return ('healthy', 'healthy') if p == 'healthy' else ('specific', p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--L', default='dino')
    ap.add_argument('--G', default='g6')
    a = ap.parse_args()
    per = json.loads((S1 / 'per_record.json').read_text())
    ev = json.loads((S1 / 'sets.json').read_text())['eval']
    L, S, G = load_L(a.L), load_S(), load_G(a.G, per, ev)
    recs = {r: dict(truth=set(per[r]['truth']), fold=per[r]['fold'], L=L[r], S=S[r], G=G[r]) for r in ev}
    folds = {f: [r for r in ev if recs[r]['fold'] == f] for f in range(3)}
    res = dict(L=a.L, G=a.G)
    # forced accuracies
    res['forced_top1_L'] = sum(LABELS[int(np.argmax(recs[r]['L']))] in recs[r]['truth'] for r in ev)
    res['forced_top1_G'] = sum(recs[r]['G'][0] in recs[r]['truth'] for r in ev)

    def forced_hrme(x):
        L_ = x['L']
        grp = {'H': L_[0], 'R': max(L_[1], L_[4], L_[5]), 'PM': L_[2], 'SEP': L_[3]}
        g = max(grp, key=grp.get)
        if g != 'R':
            return {'H': 'healthy', 'PM': 'powdery_mildew', 'SEP': 'septoria'}[g]
        return x['G'][0] if x['G'][0] in RUST else RUST[int(np.argmax([L_[1], L_[4], L_[5]]))]
    res['forced_top1_hrme'] = sum(forced_hrme(recs[r]) in recs[r]['truth'] for r in ev)
    grid_h = list(itertools.product(T_H, T_H, T_H))
    for alpha in (.05, .10):
        k = f'alpha_{alpha}'
        res[k] = {
            'HRME': nested(recs, folds, lambda x, c: hrme(x, *c, mode='G'), grid_h, alpha),
            'HRME_noG': nested(recs, folds, lambda x, c: hrme(x, *c, mode='noG'), grid_h, alpha),
            'HRME_S': nested(recs, folds, lambda x, c: hrme(x, *c, mode='S'), grid_h, alpha),
            'HRME_v2': nested(recs, folds, lambda x, c: hrme(x, *c[:3], mode='v2', tv=c[3]),
                              [g + (v,) for g in grid_h for v in (None, 0, 70, 80, 90, 95)], alpha),
            'B1_flat_L': nested(recs, folds, flat_L, [(t,) for t in T_B], alpha),
            'B2_flat_G': nested(recs, folds, flat_G, [(t,) for t in T_G], alpha),
        }
        best_base = max(res[k]['B1_flat_L']['informative_coverage'], res[k]['B2_flat_G']['informative_coverage'])
        h = res[k]['HRME']
        res[k]['primary_gain_pp'] = 100 * (h['informative_coverage'] - best_base)
        res[k]['primary_pass'] = bool(alpha == .05 and h['informative_coverage'] - best_base >= .10 and h['risk_cp95'] <= .10)
        for m, v in res[k].items():
            if isinstance(v, dict):
                print(k, m, {q: (round(w, 3) if isinstance(w, float) else w) for q, w in v.items() if q != 'chosen'})
        print(k, 'gain_pp', round(res[k]['primary_gain_pp'], 1), 'pass', res[k]['primary_pass'])
    print('forced', res['forced_top1_L'], res['forced_top1_G'], res['forced_top1_hrme'])
    (OUT / f'p3_hrme_L{a.L}_G{a.G}.json').write_text(json.dumps(res, indent=1, default=str))


if __name__ == '__main__':
    main()
