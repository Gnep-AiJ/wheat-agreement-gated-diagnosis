"""Main tables generated from ../analysis/ms_results.json and ms_revision.json (no hand-typed result numbers except the
prespecified criteria and hypothesis-test results transcribed from the protocol log, PAPER_HRME_PROTOCOL_2026-10-07.md
sections 12, 15, 18-21). Writes tables.json (rows for the Word builder) and tables_preview.md."""
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = json.loads((HERE.parent / 'analysis' / 'ms_results.json').read_text())
V = json.loads((HERE.parent / 'analysis' / 'ms_revision.json').read_text())
COLLS = ['EVAL450', 'ETS', 'ETS2', 'P4']
CN = {'EVAL450': 'EVAL450', 'ETS': 'ETS', 'ETS2': 'ETS2', 'P4': 'iNat'}
NICE = {'healthy': 'healthy', 'leaf_rust': 'leaf rust', 'stem_rust': 'stem rust', 'yellow_rust': 'stripe rust',
        'powdery_mildew': 'powdery mildew', 'septoria': 'septoria'}


def sgn(t):
    return t.replace('+0.0', '0.0').replace('-', '−')


def pct(x, d=1):
    return f'{100 * x:.{d}f}'


def table1():
    comp = R['composition']; per_coll = {c: Counter() for c in COLLS}; srcs = {c: Counter() for c in COLLS}
    for key, v in comp.items():
        c, s = key.split('|', 1)
        for lab, n in v.items():
            for l in lab.split('+'):
                per_coll[c][l] += n
        srcs[c][s.split(':', 1)[1]] += sum(v.values())
    role = {'EVAL450': 'Development (rule design and threshold selection)', 'P4': 'External test (prespecified, evaluated once)',
            'ETS': 'External test (prespecified, evaluated once)', 'ETS2': 'External confirmatory test (prespecified hypotheses H1–H3)'}
    src_txt = {'EVAL450': 'WFD2020, 8 contributing sub-sources (3 source-held folds)',
               'P4': 'iNaturalist pathogen observations with wheat as host (48 observers)',
               'ETS': 'PlantWild v2 wheat (200), MSWDD2022 (80), wheat powdery mildew camera set (40), Roboflow new-wheat-disease v2 (160), stem-rust v1 (32), Puccinia triticina v7 (40)',
               'ETS2': 'Henan field data set, Yuanyang 2023 (smartphone and DSLR)'}
    rows = []
    for c in COLLS:
        n = sum(srcs[c].values())
        labs = ', '.join(f'{NICE[l]} {k}' for l, k in per_coll[c].most_common())
        rows.append([CN[c], role[c], src_txt[c], str(n), labs])
    return dict(title='Table 1 Image collections and their roles in the study',
                header=['Collection', 'Role', 'Sources', 'Images', 'Labels (images per class)'], rows=rows,
                notes=['EVAL450 is multi-label (32 images carry more than one label); class counts are label occurrences.',
                       'All external images were de-duplicated against every training, development and earlier test pool (DINOv3 embedding cosine similarity ≥ 0.90).',
                       'Training pool of expert L per development fold: 2,616 images (600 WFD2020 images from the sources of the other two folds, 999 CerealConv, 406 Mendeley wheat leaf and 611 iNaturalist wheat images from observations not in the iNat test collection).'])


def table2():
    rows = [
        ['iNat', 'Frozen system vs best single expert, same selection procedure (α = 5 %)', 'Coverage gain ≥ 10 points; one-sided upper 95 % limit of error ≤ 10 %',
         'Coverage 48.5 % vs 0 %; error 3/33 (9.1 %), upper limit 21.9 %', 'Coverage: yes; error limit: no'],
        ['ETS', 'RAG with L fallback vs best single expert (forced choice)', 'Gain ≥ 5 points and 95 % CI lower limit > 0',
         '−0.4 points [−2.4, 1.8] vs G61 with L fallback', 'No'],
        ['ETS2, H1', 'Adaptive RAG vs G61 with L fallback (forced choice)', 'Gain ≥ 3 points and CI lower limit > 0', '+1.3 points [−1.3, 3.8]', 'No'],
        ['ETS2, H2', 'G61 with L fallback vs L (forced choice)', 'Gain ≥ 3 points and CI lower limit > 0', '−18.3 points [−23.8, −12.9]', 'No'],
        ['ETS2, H3', 'Frozen agreement-gated system', 'Error ≤ 5 %, one-sided upper 95 % limit ≤ 8 %, coverage ≥ 60 %',
         'Error 1/164 (0.6 %), upper limit 2.9 %, coverage 68.3 %', 'Yes'],
        ['16 sources (retrospective)', 'Learned arbiter (logistic regression, all features) vs best fixed rule',
         'Gain ≥ 5 points, CI lower limit > 0, no collection > 2 points worse', '−8.7 points [−10.7, −6.7]', 'No'],
        ['16 sources (retrospective)', 'Knowledge-rule arbiter vs majority vote', 'As above', '−1.9 points [−2.9, −1.0]', 'No'],
        ['16 sources (retrospective)', 'Local rule selection with 10 labelled images vs majority vote', 'Gain ≥ 3 points, no collection > 1 point worse',
         '+2.3 points; worst collection −1.4 points (iNat)', 'No'],
    ]
    return dict(title='Table 2 Hypotheses fixed before each evaluation and their outcomes',
                header=['Data', 'Comparison', 'Success criterion', 'Result [95 % CI]', 'Met'], rows=rows,
                notes=['iNat, ETS and ETS2 were each evaluated once after the criteria had been time-stamped; the retrospective criteria were fixed before computation but on data whose results had been examined.',
                       'On ETS the frozen system was a secondary, descriptive analysis (coverage 72.6 %, error 1.0 %). A second configuration selected at α = 10 % was also evaluated on iNat (9 errors among 39 answers) and was not used further.'])


def table3():
    order = [('L', 'Vision model L (DINOv3)'), ('G61 (raw)', 'MLLM G61 alone (uncertain counted as wrong)'), ('RAG (raw)', 'Retrieval-augmented MLLM (RAG) alone'),
             ('G61 + L fallback', 'G61 with L fallback'), ('RAG + L fallback', 'RAG with L fallback'),
             ('Majority vote (L, G61, RAG)', 'Majority vote of L, G61 and RAG'), ('Selection upper bound (L or G61 correct)', 'Two-expert selection bound (L or G61 correct)a')]
    rows = []
    for k, name in order:
        f = R['forced'][k]
        rows.append([name] + [f"{pct(f[c]['acc'])} ({f[c]['k']}/{f[c]['n']}) [{pct(f[c]['ci95'][0])}–{pct(f[c]['ci95'][1])}]" for c in COLLS])
    return dict(title='Table 3 Forced-choice accuracy (%) of single experts, simple combinations and the two-expert selection bound',
                header=['Method', 'EVAL450 (n = 450)', 'ETS (n = 552)', 'ETS2 (n = 240)', 'iNat (n = 68)'], rows=rows,
                notes=['Values: accuracy (correct/total) [95 % bootstrap interval], computed on all images of each collection. EVAL450 values are source-held (each image scored by models that never saw its source).',
                       'a Oracle quantity: share of images on which at least one of L and G61 is correct; not attainable without the true label.',
                       'RAG on iNat was computed after the iNat evaluation and is therefore retrospective for that collection. Accuracy on the subset of images where L and G61 agree is shown in Fig. 3c.'])


def table4():
    sel = R['selective']; md = V['matched_coverage_differences']; rows = []
    for c in ['ETS', 'ETS2', 'P4']:
        s = sel[c]; lv = s['levels']; d = md[c]
        rows.append([CN[c], f"{s['released']}/{s['n']} ({pct(s['coverage'])})", f"{s['errors']}",
                     f"{pct(s['accuracy'])} [{pct(s['accuracy_ci95'][0])}–{pct(s['accuracy_ci95'][1])}]",
                     f"{lv.get('specific', 0)} / {lv.get('group', 0)} / {lv.get('healthy', 0)}",
                     f"{pct(s['group_accuracy'])} / {pct(s['matched_L_group_accuracy'])} / {pct(s['matched_G61_group_accuracy'])}",
                     sgn(f"{d['diff_vs_L_pp']:+.1f} [{d['ci_vs_L'][0]:.1f}, {d['ci_vs_L'][1]:.1f}]"),
                     sgn(f"{d['diff_vs_G61_pp']:+.1f} [{d['ci_vs_G61'][0]:.1f}, {d['ci_vs_G61'][1]:.1f}]")])
    p = sel['external_pooled']; c = V['composition']['external']
    rows.append(['Pooled external (8 sources)', f"{p['released']}/{p['n']} ({pct(p['coverage'])})", f"{p['errors']}",
                 f"{pct(p['accuracy'])} [{pct(p['accuracy_ci95'][0])}–{pct(p['accuracy_ci95'][1])}]",
                 f"{c['specific']} / {c['group']} / {c['healthy']}", '–', '–', '–'])
    return dict(title='Table 4 Automatic answers of the frozen agreement-gated system on the external collections',
                header=['Collection', 'Answered (coverage %)', 'Errors', 'Accuracy % [95 % CI]', 'Specific / group / healthy',
                        'Group level %: system / L / G61', 'System − L, points [95 % CI]', 'System − G61, points [95 % CI]'],
                rows=rows,
                notes=['Columns 2–5: accuracy among answered images at the level at which each answer was given; Clopper–Pearson intervals.',
                       'Columns 6–8: all methods at the same number of answered images, scored at group granularity (healthy, rust, powdery mildew, septoria); single experts answer their most confident images. Difference intervals come from a bootstrap that repeats the selection in each resample.',
                       'Development estimate (source-held nested cross-validation on EVAL450): coverage 66.9 %, error rate 5.0 % (15/301). Per-source results are shown in Fig. 5.'])


def table5():
    fu = R['fusion_attempts']; t = fu['learned']; tk = fu['knowledge']; ks = fu['k_shot']; o3 = V['oracle_three_experts']
    order = [('L', t['L'], 'None'), ('G61 with L fallback', t['G61+L'], 'None'), ('RAG with L fallback', t['RAG+L'], 'None'),
             ('Majority vote (L, G61, RAG)', t['vote3'], 'None'),
             ('Learned arbiter, logistic regression, L and G61 features', t['M1[L+G]'], 'None (trained on 15 other sources)'),
             ('Learned arbiter, logistic regression, all features', t['M1[L+G+S+R]'], 'None (trained on 15 other sources)'),
             ('Learned arbiter, gradient boosting, all features', t['M2[L+G+S+R]'], 'None (trained on 15 other sources)'),
             ('Knowledge-rule arbiter', tk['K (knowledge arbiter)'], 'None'),
             ('Local rule selection, K = 5', ks['K=5'], '5 labelled images per source'),
             ('Local rule selection, K = 10', ks['K=10'], '10 labelled images per source'),
             ('Local rule selection, K = 20', ks['K=20'], '20 labelled images per source'),
             ('Per-source best fixed rule (all labels)b', ks['oracle_selection'], 'Oracle reference'),
             ('Two-expert selection bound (L or G61 correct)b', t['oracle(L|G61)'], 'Oracle reference'),
             ('Three-expert bound (L, G61 or RAG correct)b', o3, 'Oracle reference')]
    rows = [[n, cost] + [pct(d[c]) for c in ['pooled'] + COLLS] for n, d, cost in order]
    k10 = V['kshot']['K=10']
    return dict(title='Table 5 Forced-choice accuracy (%) of fusion strategies in retrospective leave-one-source-out analysis (16 sources, 1,310 images)',
                header=['Strategy', 'Labelling needed at a new source', 'All sources', 'EVAL450', 'ETS', 'ETS2', 'iNat'], rows=rows,
                notes=['Retrospective analysis on collections whose results had already been examined; learned components were always fitted on the other 15 sources, using cached expert outputs.',
                       f"Local rule selection chooses, per source, the most accurate of four fixed rules (L; G61 with L fallback; RAG with L fallback; majority vote) on K random labelled images and is evaluated on the remaining images (mean of 50 draws; for K = 10 the pooled accuracy of the central 95 % of draws was {pct(k10['pooled_p2_5'])}–{pct(k10['pooled_p97_5'])} %). Sources with fewer than 2K images use the majority vote (K = 10: {len(k10['fallback_sources'])} sources, {k10['fallback_images']} images).",
                       'b Oracle references that use the true labels. RAG-based rules can exceed the two-expert bound because they draw on a third output; rules that only select among the answers of L, G61 and RAG cannot exceed the three-expert bound.'])


def main() -> None:
    tables = [table1(), table2(), table3(), table4(), table5()]
    (HERE / 'tables.json').write_text(json.dumps(tables, indent=1, ensure_ascii=False), encoding='utf-8')
    md = []
    for t in tables:
        md.append(f"**{t['title']}**\n\n| " + ' | '.join(t['header']) + ' |\n|' + '---|' * len(t['header']))
        md += ['| ' + ' | '.join(r) + ' |' for r in t['rows']]
        md += [''] + [f'_{n}_' for n in t['notes']] + ['']
    (HERE / 'tables_preview.md').write_text('\n'.join(md), encoding='utf-8')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
