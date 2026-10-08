"""Manuscript analyses for the Precision Agriculture submission (definitions fixed in ../00_PLAN.md section E).

Reads cached expert outputs only (no model calls). Writes JSON/CSV source data next to this script.
Run with: <DATA_ROOT>/.venv/Scripts/python.exe manuscript/precision_agriculture_2026/analysis/ms_analysis.py
"""
import csv
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
from paper_arbiter import load_rows  # noqa: E402
from paper_arbiter_k import vote3  # noqa: E402

LAB = H.LABELS
GROUP = H.GROUP
COLLS = ['EVAL450', 'ETS', 'ETS2', 'P4']
RNG_SEED = 20261008
V3_CFG = (0.5, 0.5, 0.5, 95, True, True)


def cp(k, n):
    if n == 0:
        return [float('nan'), float('nan')]
    lo = beta.ppf(.025, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(.975, k + 1, n - k) if k < n else 1.0
    return [float(lo), float(hi)]


def boot(ok, n_boot=10000, seed=RNG_SEED):
    ok = np.asarray(ok, float); rng = np.random.default_rng(seed)
    b = [ok[rng.integers(0, len(ok), len(ok))].mean() for _ in range(n_boot)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def group_ok(pred, truth):
    return pred is not None and GROUP[pred] in {GROUP[t] for t in truth}


def main() -> None:
    rows = load_rows()
    for r in rows:
        r['lab_L'] = LAB[int(np.argmax(r['L']))]
        r['lab_G'] = r['g'] if r['g'] in LAB else None
        r['lab_R'] = r['r'] if r['r'] in LAB else None
        r['conf_G'] = ((r['gfull'] or {}).get('confidence') or 0) if r['lab_G'] else -1
    out = {}

    # 1. dataset composition
    comp = defaultdict(Counter)
    for r in rows:
        comp[(r['coll'], r['source'])]['+'.join(sorted(r['truth']))] += 1
    out['composition'] = {f'{c}|{s}': dict(v) for (c, s), v in sorted(comp.items())}

    # 2. forced-choice accuracy
    systems = {
        'L': lambda r: r['lab_L'], 'G61 (raw)': lambda r: r['lab_G'], 'RAG (raw)': lambda r: r['lab_R'],
        'G61 + L fallback': lambda r: r['lab_G'] or r['lab_L'], 'RAG + L fallback': lambda r: r['lab_R'] or r['lab_L'],
        'Majority vote (L, G61, RAG)': lambda r: vote3(r, r['lab_L']),
    }
    forced = {}
    for name, f in systems.items():
        forced[name] = {}
        for c in COLLS:
            ok = [f(r) in r['truth'] for r in rows if r['coll'] == c]
            forced[name][c] = dict(k=int(sum(ok)), n=len(ok), acc=float(np.mean(ok)), ci95=boot(ok))
    for c in COLLS:
        ok = [(r['lab_L'] in r['truth']) or (r['lab_G'] in r['truth']) for r in rows if r['coll'] == c]
        forced.setdefault('Selection upper bound (L or G61 correct)', {})[c] = dict(k=int(sum(ok)), n=len(ok), acc=float(np.mean(ok)), ci95=boot(ok))
    out['forced'] = forced

    # 3. complementarity decomposition + agreement-subset accuracy
    compl = {}
    for c in COLLS:
        rr = [r for r in rows if r['coll'] == c]
        cnt = Counter(('L+' if r['lab_L'] in r['truth'] else 'L-') + ('G+' if r['lab_G'] in r['truth'] else 'G-') for r in rr)
        agree = [r for r in rr if r['lab_L'] == r['lab_G']]
        ka = sum(r['lab_L'] in r['truth'] for r in agree)
        compl[c] = dict(n=len(rr), both_correct=cnt['L+G+'], only_L=cnt['L+G-'], only_G61=cnt['L-G+'], both_wrong=cnt['L-G-'],
                        agree_n=len(agree), agree_correct=ka, agree_acc=ka / max(len(agree), 1), agree_ci95=cp(ka, len(agree)))
    out['complementarity'] = compl

    # 4. selective output: frozen HRME-v3 vs matched-coverage single experts at group granularity
    H.STRICT[0] = True
    sel = {}
    for c in COLLS:
        rr = [r for r in rows if r['coll'] == c]
        outs = [H.hrme_v3(dict(L=r['L'], G=(r['lab_G'], r['conf_G'] if r['lab_G'] else -1)), *V3_CFG) for r in rr]
        rel = [(r, o) for r, o in zip(rr, outs) if o[0] in ('healthy', 'group', 'specific')]
        k = len(rel); e_exact = sum(not H.correct(o[0], o[1], r['truth']) for r, o in rel)
        g_ok = sum((o[0] == 'healthy' and r['truth'] == {'healthy'}) or (o[0] == 'group' and o[1] in {GROUP[t] for t in r['truth']})
                   or (o[0] == 'specific' and GROUP[o[1]] in {GROUP[t] for t in r['truth']}) for r, o in rel)
        levels = Counter(o[0] for _, o in rel)
        # single-expert baselines at the same number of released images k, scored at group granularity
        orderL = sorted(rr, key=lambda r: -float(np.max(r['L'])))[:k]
        orderG = sorted(rr, key=lambda r: -r['conf_G'])[:k]
        bL = sum(group_ok(r['lab_L'], r['truth']) for r in orderL); bG = sum(group_ok(r['lab_G'], r['truth']) for r in orderG)
        sel[c] = dict(n=len(rr), released=k, coverage=k / len(rr), coverage_ci95=cp(k, len(rr)), errors=e_exact,
                      accuracy=1 - e_exact / k if k else None, accuracy_ci95=cp(k - e_exact, k), levels=dict(levels),
                      group_accuracy=g_ok / k if k else None,
                      matched_L_group_accuracy=bL / k if k else None, matched_G61_group_accuracy=bG / k if k else None,
                      matched_L_ci95=cp(bL, k), matched_G61_ci95=cp(bG, k))
    ext = ['ETS', 'ETS2', 'P4']
    K = sum(sel[c]['released'] for c in ext); N = sum(sel[c]['n'] for c in ext); E = sum(sel[c]['errors'] for c in ext)
    sel['external_pooled'] = dict(n=N, released=K, coverage=K / N, coverage_ci95=cp(K, N), errors=E, accuracy=1 - E / K,
                                  accuracy_ci95=cp(K - E, K))
    out['selective'] = sel

    # 5. risk-coverage curves (descriptive): L by max prob, G61 by confidence, system by sweeping a common threshold t
    curves = {}
    for c in COLLS:
        rr = [r for r in rows if r['coll'] == c]; n = len(rr); cur = {}
        for name, key, pred in (('L', lambda r: float(np.max(r['L'])), lambda r: r['lab_L']), ('G61', lambda r: r['conf_G'], lambda r: r['lab_G'])):
            srt = sorted(rr, key=lambda r: -key(r)); acc = []; okc = 0
            for i, r in enumerate(srt, 1):
                okc += group_ok(pred(r), r['truth']); acc.append((i / n, 1 - okc / i))
            cur[name] = acc[::max(1, n // 60)] + [acc[-1]]
        pts = []
        for t in np.linspace(.3, .999, 40):
            outs = [H.hrme_v3(dict(L=r['L'], G=(r['lab_G'], r['conf_G'] if r['lab_G'] else -1)), t, t, t, 95, True, True) for r in rr]
            rel = [(r, o) for r, o in zip(rr, outs) if o[0] in ('healthy', 'group', 'specific')]
            if rel:
                e = sum(not ((o[0] == 'healthy' and r['truth'] == {'healthy'}) or (o[0] != 'healthy' and
                         (o[1] if o[0] == 'group' else GROUP[o[1]]) in {GROUP[x] for x in r['truth']})) for r, o in rel)
                pts.append((len(rel) / n, e / len(rel)))
        cur['Agreement-gated system'] = sorted(set(pts))
        curves[c] = cur
    out['risk_coverage_group_level'] = curves

    # 6. source-familiar vs source-held (same 450 images, same recipe, seed 42)
    per = json.loads((H.S1 / 'per_record.json').read_text())

    def load(tpl):
        d = {}
        for f in range(3):
            z = np.load(ROOT / tpl.format(f=f) / 'evaluation.npz')
            for rid, p in zip(z['record_ids'], z['probabilities']):
                d[str(rid)] = LAB[int(np.argmax(p))]
        return d
    fam = load('outputs/strong_expert_confirmation_v1/dino_source_familiar_expanded_seed42_fold{f}')
    held = load('outputs/strong_expert_training_v1/dino_expanded_fold{f}')
    rust = [k for k in fam if len(per[k]['truth']) == 1 and per[k]['truth'][0] in ('leaf_rust', 'stem_rust', 'yellow_rust')]
    out['familiar_vs_held'] = dict(
        all=dict(n=len(fam), familiar=sum(fam[k] in per[k]['truth'] for k in fam), held=sum(held[k] in per[k]['truth'] for k in fam)),
        rust_single_label=dict(n=len(rust), familiar=sum(fam[k] in per[k]['truth'] for k in rust), held=sum(held[k] in per[k]['truth'] for k in rust)))

    # 7. fusion attempts and local-label selection (from protocol sections 19-21 result files)
    out['fusion_attempts'] = dict(
        learned=json.loads((ROOT / 'outputs/paper_hrme/arbiter/arbiter_results.json').read_text())['table'],
        knowledge=json.loads((ROOT / 'outputs/paper_hrme/arbiter/arbiter_k_results.json').read_text())['table'],
        k_shot=json.loads((ROOT / 'outputs/paper_hrme/arbiter/kshot_results.json').read_text()))
    # 8. leakage audit
    out['leakage'] = json.loads((ROOT / 'outputs/paper_hrme/leak_audit/audit.json').read_text())

    (HERE / 'ms_results.json').write_text(json.dumps(out, indent=1, default=float))
    # flat CSV for the forced-accuracy table
    with (HERE / 'table_forced_accuracy.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['system'] + [f'{c}_{x}' for c in COLLS for x in ('k', 'n', 'acc', 'lo', 'hi')])
        for s, v in forced.items():
            w.writerow([s] + [y for c in COLLS for y in (v[c]['k'], v[c]['n'], round(v[c]['acc'], 4), round(v[c]['ci95'][0], 4), round(v[c]['ci95'][1], 4))])
    for c in COLLS:
        print(c, {s: f"{forced[s][c]['acc']:.3f}" for s in forced})
        print('   compl', {k: v for k, v in compl[c].items() if k != 'agree_ci95'})
        print('   sel', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in sel[c].items() if 'ci95' not in k})
    print('external pooled', sel['external_pooled']); print('familiar vs held', out['familiar_vs_held'])


if __name__ == '__main__':
    main()
