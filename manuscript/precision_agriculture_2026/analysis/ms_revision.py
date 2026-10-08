"""Additional analyses requested in internal review (cached outputs only; no model calls). Writes ms_revision.json.

1 per-source results of the frozen system on the external collections (+ source-macro averages)
2 iNat sensitivity without the 3 images whose observers also contributed iNat training images
3 matched-coverage differences (system - single expert, group granularity) with bootstrap CIs that redo the selection
4 output-level composition per collection and pooled
5 all erroneous automatic answers on the external collections
6 three-expert oracle (L | G61 | RAG) and per-source best fixed rule
7 per-source leave-one-source-out accuracies and K-shot variability over draws
8 one-sided vs two-sided Clopper-Pearson limits used in the text
9 share of images with a near-duplicate at cosine >= 0.95 (all visually confirmed bin)
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import beta

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
import paper_p3_hrme as H  # noqa: E402
from paper_arbiter import LAB, X_of, fit_predict, load_rows  # noqa: E402
from paper_arbiter_k import arbiter_k, vote3  # noqa: E402

GROUP = H.GROUP
V3 = (0.5, 0.5, 0.5, 95, True, True)
SEED = 20261008
EXT = ['ETS', 'ETS2', 'P4']


def cp2(k, n):
    lo = beta.ppf(.025, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(.975, k + 1, n - k) if k < n else 1.0
    return [float(lo), float(hi)]


def cp_upper1(k, n):
    return float(beta.ppf(.95, k + 1, n - k)) if k < n else 1.0


def gok(level, claim, truth):
    tg = {GROUP[t] for t in truth}
    if level == 'healthy':
        return truth == {'healthy'}
    return (claim if level == 'group' else GROUP[claim]) in tg


def main() -> None:
    H.STRICT[0] = True
    rows = load_rows()
    for r in rows:
        r['lab_L'] = LAB[int(np.argmax(r['L']))]
        r['lab_G'] = r['g'] if r['g'] in LAB else None
        r['lab_R'] = r['r'] if r['r'] in LAB else None
        r['conf_G'] = ((r['gfull'] or {}).get('confidence') or 0) if r['lab_G'] else -1
        r['sys'] = H.hrme_v3(dict(L=r['L'], G=(r['lab_G'], r['conf_G'] if r['lab_G'] else -1)), *V3)
        r['ans'] = r['sys'][0] in ('healthy', 'group', 'specific')
        r['ok_level'] = r['ans'] and H.correct(r['sys'][0], r['sys'][1], r['truth'])
        r['ok_group'] = r['ans'] and gok(r['sys'][0], r['sys'][1], r['truth'])
        r['gL'] = r['lab_L'] is not None and GROUP[r['lab_L']] in {GROUP[t] for t in r['truth']}
        r['gG'] = r['lab_G'] is not None and GROUP[r['lab_G']] in {GROUP[t] for t in r['truth']}
    out = {}

    # 1 per-source
    per = {}
    for r in rows:
        if r['coll'] in EXT:
            per.setdefault(r['source'], []).append(r)
    tab = {}
    for s, rr in sorted(per.items()):
        n = len(rr); a = [r for r in rr if r['ans']]; k = len(a); e = sum(not r['ok_level'] for r in a)
        lv = Counter(r['sys'][0] for r in a)
        tab[s] = dict(collection=rr[0]['coll'], n=n, answered=k, coverage=k / n, coverage_ci95=cp2(k, n), errors=e,
                      accuracy=(k - e) / k if k else None, accuracy_ci95=cp2(k - e, k) if k else None,
                      specific=lv['specific'], group=lv['group'], healthy=lv['healthy'],
                      L_forced=sum(r['lab_L'] in r['truth'] for r in rr) / n, G61_forced=sum(r['lab_G'] in r['truth'] for r in rr) / n)
    out['per_source'] = tab
    acc = [v['accuracy'] for v in tab.values() if v['answered']]
    out['source_macro'] = dict(n_sources=len(tab), coverage=float(np.mean([v['coverage'] for v in tab.values()])),
                               accuracy=float(np.mean(acc)), min_accuracy=float(min(acc)), max_errors=max(v['errors'] for v in tab.values()))

    # 2 iNat observer sensitivity
    items = json.loads((ROOT / 'outputs/paper_hrme/p4_rag/items.json').read_text())
    import csv
    with (ROOT / 'outputs/mainline_v2/inat/inat_manifest.csv').open(encoding='utf-8-sig') as f:
        train_obs = {(r['observer'] or '').strip().lower() for r in csv.DictReader(f) if r['partition'] == 'inat_train'}
    p4 = [r for r in rows if r['coll'] == 'P4']
    allit = {a['obs']: a for a in json.loads((ROOT / 'outputs/paper_hrme/p4_inat/items.json').read_text())}
    users = [(allit[it['obs']]['user'] or '').strip().lower() for it in items]
    keep = [r for r, u in zip(p4, users) if u not in train_obs]
    a = [r for r in keep if r['ans']]
    out['inat_observer_sensitivity'] = dict(n_observers=len(set(users)), excluded=len(p4) - len(keep), n=len(keep), answered=len(a),
                                            errors=sum(not r['ok_level'] for r in a),
                                            L_forced=sum(r['lab_L'] in r['truth'] for r in keep) / len(keep),
                                            G61_forced=sum(r['lab_G'] in r['truth'] for r in keep) / len(keep))

    # 3 matched coverage differences with re-selection bootstrap
    rng = np.random.default_rng(SEED)
    md = {}
    for c in EXT:
        rr = [r for r in rows if r['coll'] == c]; n = len(rr)
        ans = np.array([r['ans'] for r in rr]); oks = np.array([r['ok_group'] for r in rr])
        pL = np.array([float(np.max(r['L'])) for r in rr]); gL = np.array([r['gL'] for r in rr])
        pG = np.array([r['conf_G'] for r in rr], float); gG = np.array([r['gG'] for r in rr])

        def stat(idx):
            k = int(ans[idx].sum())
            if k == 0:
                return np.nan, np.nan
            s = oks[idx].sum() / k
            oL = idx[np.argsort(-pL[idx], kind='stable')][:k]; oG = idx[np.argsort(-pG[idx], kind='stable')][:k]
            return s - gL[oL].mean(), s - gG[oG].mean()
        d0 = stat(np.arange(n)); bs = np.array([stat(rng.integers(0, n, n)) for _ in range(10000)])
        md[c] = dict(diff_vs_L_pp=100 * d0[0], ci_vs_L=list(100 * np.nanpercentile(bs[:, 0], [2.5, 97.5])),
                     diff_vs_G61_pp=100 * d0[1], ci_vs_G61=list(100 * np.nanpercentile(bs[:, 1], [2.5, 97.5])))
    out['matched_coverage_differences'] = md

    # 4 composition of outputs
    comp = {}
    for c in EXT + ['external']:
        rr = [r for r in rows if (r['coll'] in EXT if c == 'external' else r['coll'] == c)]
        lv = Counter(r['sys'][0] if r['ans'] else 'referred' for r in rr)
        comp[c] = dict(n=len(rr), specific=lv['specific'], group=lv['group'], healthy=lv['healthy'], referred=lv['referred'],
                       errors_by_level=dict(Counter(r['sys'][0] for r in rr if r['ans'] and not r['ok_level'])))
    out['composition'] = comp

    # 5 erroneous automatic answers
    ets = {c: json.loads((ROOT / f'outputs/paper_hrme/{d}/items.json').read_text()) for c, d in (('ETS', 'ets'), ('ETS2', 'ets2'), ('P4', 'p4_rag'))}
    pos = defaultdict(int); errs = []
    for r in rows:
        if r['coll'] not in EXT:
            continue
        it = ets[r['coll']][pos[r['coll']]]; pos[r['coll']] += 1
        if r['ans'] and not r['ok_level']:
            sym = (r['gfull'] or {}).get('symptoms', {})
            errs.append(dict(collection=r['coll'], source=r['source'], file=it.get('file'), obs=it.get('obs'), truth=sorted(r['truth'])[0],
                             L=r['lab_L'], L_prob=float(np.max(r['L'])), G61=r['lab_G'], G61_conf=r['conf_G'],
                             system=f"{r['sys'][0]}: {r['sys'][1]}", symptoms=sym))
    out['erroneous_answers'] = errs

    # 6 three-expert oracle and per-source best fixed rule
    for r in rows:
        r['pred'] = {'L': r['lab_L'], 'G61+L': r['lab_G'] or r['lab_L'], 'RAG+L': r['lab_R'] or r['lab_L'], 'vote3': vote3(r, r['lab_L'])}
    colls = ['EVAL450', 'ETS', 'ETS2', 'P4']
    o3 = {c: float(np.mean([(r['lab_L'] in r['truth']) or (r['lab_G'] in r['truth']) or (r['lab_R'] in r['truth'])
                            for r in rows if (c == 'pooled' or r['coll'] == c)])) for c in colls + ['pooled']}
    out['oracle_three_experts'] = o3

    # 7 per-source LOSO table
    srcs = sorted({r['source'] for r in rows})
    preds = {k: [r['pred'][k] for r in rows] for k in ('L', 'G61+L', 'RAG+L', 'vote3')}
    preds['knowledge'] = [arbiter_k(r) for r in rows]
    for kind, parts, name in (('M1', ('L', 'G'), 'learned_LR_L+G'), ('M1', ('L', 'G', 'S', 'R'), 'learned_LR_all'), ('M2', ('L', 'G', 'S', 'R'), 'learned_GBT_all')):
        X = X_of(rows, parts); p = [None] * len(rows)
        for s in srcs:
            te = [i for i, r in enumerate(rows) if r['source'] == s]; tr = [i for i, r in enumerate(rows) if r['source'] != s]
            for i, q in zip(te, fit_predict(kind, X[tr], [rows[i] for i in tr], X[te])):
                p[i] = q
        preds[name] = p
    ok = {k: np.array([q in r['truth'] for q, r in zip(v, rows)]) for k, v in preds.items()}
    ok['oracle2'] = np.array([(r['lab_L'] in r['truth']) or (r['lab_G'] in r['truth']) for r in rows])
    src = np.array([r['source'] for r in rows])
    rules = ['L', 'G61+L', 'RAG+L', 'vote3']
    kshot = {}
    for K in (5, 10, 20):
        rng = np.random.default_rng(SEED + K); draw_acc = np.zeros((50, len(rows))); draw_w = np.zeros((50, len(rows)))
        fallback = []
        for s in srcs:
            idx = np.where(src == s)[0]
            if len(idx) < 2 * K:
                draw_acc[:, idx] = ok['vote3'][idx]; draw_w[:, idx] = 1; fallback.append((s, len(idx)))
                continue
            for d in range(50):
                cal = rng.choice(idx, K, replace=False); rest = np.setdiff1d(idx, cal)
                sc = {k: ok[k][cal].mean() for k in rules}; best = max(rules, key=lambda k: (sc[k], k == 'vote3'))
                draw_acc[d, rest] = ok[best][rest]; draw_w[d, rest] = 1
        pooled = (draw_acc * draw_w).sum(1) / draw_w.sum(1)
        per_src = {s: [float(((draw_acc[d] * draw_w[d])[src == s]).sum() / max(draw_w[d][src == s].sum(), 1)) for d in range(50)] for s in srcs}
        kshot[f'K={K}'] = dict(pooled_mean=float(pooled.mean()), pooled_p2_5=float(np.percentile(pooled, 2.5)), pooled_p97_5=float(np.percentile(pooled, 97.5)),
                               fallback_sources=fallback, fallback_images=int(sum(n for _, n in fallback)),
                               per_source_mean={s: float(np.mean(v)) for s, v in per_src.items()},
                               per_source_sd={s: float(np.std(v)) for s, v in per_src.items()})
    persrc = {s: dict(n=int((src == s).sum()), coll=rows[int(np.where(src == s)[0][0])]['coll'],
                      **{k: float(v[src == s].mean()) for k, v in ok.items()}, K10=kshot['K=10']['per_source_mean'][s],
                      K10_sd=kshot['K=10']['per_source_sd'][s], best_fixed=max(float(ok[k][src == s].mean()) for k in rules)) for s in srcs}
    out['loso_per_source'] = persrc
    out['kshot'] = {k: {q: w for q, w in v.items() if q not in ('per_source_mean', 'per_source_sd')} for k, v in kshot.items()}

    # 8 CP limits
    out['cp_limits'] = {'dev_15_301': dict(one_sided_upper95=cp_upper1(15, 301), two_sided=cp2(15, 301)),
                        'ets2_1_164': dict(one_sided_upper95=cp_upper1(1, 164), two_sided=cp2(1, 164)),
                        'inat_3_33': dict(one_sided_upper95=cp_upper1(3, 33), two_sided=cp2(3, 33))}

    # 9 near-duplicate share at >= 0.95
    import csv as _csv
    with (ROOT / 'outputs/stage0/data_audit_v2_six_sources/image_manifest.csv').open(encoding='utf-8-sig') as f:
        mrows = [r for r in _csv.DictReader(f) if r['readable'] in ('True', 'true', '1', 'yes', '')]
    seen, its = set(), []
    for r in mrows:
        kk = (r['dataset'], r['sha256'])
        if kk not in seen:
            seen.add(kk); its.append(r)
    X = np.load(ROOT / 'outputs/paper_hrme/leak_audit/feats.npy').astype(np.float32); ds = np.array([r['dataset'] for r in its])
    share95 = {}
    for d in sorted(set(ds)):
        ia = np.where(ds == d)[0]; ib = np.where(ds != d)[0]; hit = 0
        for s0 in range(0, len(ia), 2048):
            hit += int(((X[ia[s0:s0 + 2048]] @ X[ib].T).max(1) >= .95).sum())
        share95[d] = dict(n=len(ia), any_ge095=hit, pct=100 * hit / len(ia))
    out['dup_share_095'] = share95
    out['dedup_visual_validation'] = json.loads((HERE / 'dedup_validation/ratings_summary.json').read_text())

    (HERE / 'ms_revision.json').write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ('source_macro', 'inat_observer_sensitivity', 'matched_coverage_differences', 'composition',
                                          'oracle_three_experts', 'kshot', 'cp_limits', 'dup_share_095')}, indent=1, default=float))
    for s, v in tab.items():
        print(f"{s:32s} n{v['n']:4d} ans {v['answered']:4d} cov {v['coverage']:.3f} err {v['errors']} acc {v['accuracy'] if v['accuracy'] is None else round(v['accuracy'], 3)} S/G/H {v['specific']}/{v['group']}/{v['healthy']} L {v['L_forced']:.2f} G {v['G61_forced']:.2f}")
    for e in errs:
        print(e['collection'], e['source'], e['file'], e['truth'], '| L', e['L'], round(e['L_prob'], 2), '| G', e['G61'], e['G61_conf'], '|', e['system'])


if __name__ == '__main__':
    main()
