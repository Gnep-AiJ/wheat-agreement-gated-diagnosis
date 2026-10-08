"""Supplementary Information (ESM_1.docx) for the Precision Agriculture manuscript, built from cached outputs only (no model calls).

S1 frozen MLLM prompt, JSON schema and command-line settings; S2 retrieval prefix; S3 collection composition; S4 per-class accuracy and
confusion matrices; S5 threshold grid and nested cross-validation; S6 protocol timeline; S7 near-duplicate audit; S8 case images.
Also writes each table as CSV next to this script.
Run with: <DATA_ROOT>/.venv/Scripts/python.exe manuscript/precision_agriculture_2026/supplementary/make_supplementary.py
"""
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

HERE = Path(__file__).resolve().parent
MS = HERE.parent
ROOT = MS.parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts')); sys.path.insert(0, str(MS))
import paper_p2_codex_g61 as P  # noqa: E402
import paper_p3_hrme as H  # noqa: E402
from paper_arbiter import load_rows  # noqa: E402
from build_docx import add_table as add_three_line_table  # noqa: E402

FONT = 'Times New Roman'
LAB = H.LABELS
NICE = {'healthy': 'Healthy', 'leaf_rust': 'Leaf rust', 'powdery_mildew': 'Powdery mildew', 'septoria': 'Septoria',
        'stem_rust': 'Stem rust', 'yellow_rust': 'Stripe rust', None: 'Uncertain'}
COLL = {'EVAL450': 'EVAL450', 'ETS': 'ETS', 'ETS2': 'ETS2', 'P4': 'iNat'}
SRC = {'ets:mswdd2022': 'MSWDD2022', 'ets:plantwild': 'PlantWild v2 (wheat)', 'ets:roboflow_newwheat': 'Roboflow new-wheat-disease v2',
       'ets:roboflow_ptriticina': 'Roboflow Puccinia triticina v7', 'ets:roboflow_stemrust': 'Roboflow stem-rust v1',
       'ets:zenodo13137587': 'Zenodo 13137587 (powdery mildew)', 'ets2:henan_field_2023': 'Zenodo 15621359 (Henan field 2023)',
       'p4_rag:inat_pathogen': 'iNaturalist'}
V3_CFG = (0.5, 0.5, 0.5, 95, True, True)
TABLES = []


def T(num, title, header, rows, notes=()):
    t = dict(title=f'Table S{num} {title}', header=header, rows=[[str(c) for c in r] for r in rows], notes=list(notes))
    TABLES.append(t)
    with open(HERE / f'TableS{num}.csv', 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.writer(fh); w.writerow(header); w.writerows(t['rows'])
    return t


def pct(k, n):
    return f'{100 * k / n:.1f} ({k}/{n})' if n else '–'


def main() -> None:
    rows = load_rows()
    for r in rows:
        r['lab_L'] = LAB[int(np.argmax(r['L']))]
        r['lab_G'] = r['g'] if r['g'] in LAB else None
        r['lab_R'] = r['r'] if r['r'] in LAB else None
        r['conf_G'] = ((r['gfull'] or {}).get('confidence') or 0) if r['lab_G'] else -1
        r['sys'] = H.hrme_v3(dict(L=r['L'], G=(r['lab_G'], r['conf_G'] if r['lab_G'] else -1)), *V3_CFG)

    doc = Document()
    for side, v in (('left_margin', 1.8), ('right_margin', 1.8), ('top_margin', 2.2), ('bottom_margin', 2.2)):
        setattr(doc.sections[0], side, Cm(v))
    st = doc.styles['Normal']; st.font.name = FONT; st.font.size = Pt(10); st.element.rPr.rFonts.set(qn('w:eastAsia'), FONT)
    for lvl in (1, 2):
        h = doc.styles[f'Heading {lvl}']; h.font.name = FONT; h.font.color.rgb = None; h.font.size = Pt(12 if lvl == 1 else 10.5)
        rf = h.element.rPr.rFonts
        for a in ('w:asciiTheme', 'w:hAnsiTheme', 'w:eastAsiaTheme', 'w:cstheme'):
            if rf.get(qn(a)) is not None:
                del rf.attrib[qn(a)]
        for a in ('w:ascii', 'w:hAnsi', 'w:eastAsia', 'w:cs'):
            rf.set(qn(a), FONT)
    p = doc.add_paragraph(); r = p.add_run('Supplementary Information'); r.bold = True; r.font.size = Pt(14)
    doc.add_paragraph('Agreement-gated fusion of a vision foundation model and a multimodal large language model for reliable wheat '
                      'disease diagnosis across image sources. [Authors — TO BE COMPLETED]. Precision Agriculture (submitted).')

    def mono(text):
        for line in text.rstrip('\n').split('\n'):
            q = doc.add_paragraph(); q.paragraph_format.space_after = Pt(0); q.paragraph_format.line_spacing = 1.0
            rr = q.add_run(line); rr.font.name = 'Courier New'; rr.font.size = Pt(7.5)

    # S1 prompt, schema, settings
    doc.add_heading('S1 Frozen prompt, output schema and interface settings of expert G61', level=1)
    doc.add_paragraph('The prompt was fixed on 29 September 2026, before any image of this study was submitted, and was used '
                      'unchanged for all collections (SHA-256 of the UTF-8 file given below). Each image was submitted in a new '
                      'session through the OpenAI Codex command-line interface (version 0.160.1), model gpt-6.1-sol, reasoning '
                      'effort "medium", with the following features disabled and with the JSON schema passed as --output-schema.')
    import hashlib
    doc.add_paragraph('Prompt SHA-256: ' + hashlib.sha256(P.PROMPT.encode('utf-8')).hexdigest())
    doc.add_paragraph('Disabled features: ' + P.FLAGS.replace('--disable ', '').replace(' ', ', '))
    doc.add_heading('S1.1 Prompt', level=2); mono(P.PROMPT)
    doc.add_heading('S1.2 JSON output schema', level=2); mono(json.dumps(P.SCHEMA, indent=1))
    (HERE / 'S1_prompt.txt').write_text(P.PROMPT, encoding='utf-8')
    (HERE / 'S1_schema.json').write_text(json.dumps(P.SCHEMA, indent=1), encoding='utf-8')

    # S2 retrieval prefix
    src = (ROOT / 'scripts/paper_rag_g61.py').read_text(encoding='utf-8')
    prefix = re.search(r'PREFIX = """(.*?)"""', src, re.S).group(1)
    doc.add_heading('S2 Prefix of the retrieval-augmented variant (RAG)', level=1)
    doc.add_paragraph('The reference photographs were attached before the query image and this text was prepended to the S1 prompt; '
                      '{n} is the number of references (9) and {lst} lists their labels in order.')
    mono(prefix); (HERE / 'S2_rag_prefix.txt').write_text(prefix, encoding='utf-8')

    # S3 composition
    doc.add_heading('S3 Composition of the evaluation collections', level=1)
    comp = defaultdict(Counter)
    for r in rows:
        comp[(r['coll'], r['source'])]['+'.join(sorted(r['truth']))] += 1
    single = [c for c in LAB]
    out = []
    for (c, s), v in sorted(comp.items(), key=lambda x: (list(COLL).index(x[0][0]), x[0][1])):
        multi = sum(n for k, n in v.items() if '+' in k)
        out.append([COLL[c], SRC.get(s, s.split(':', 1)[1])] + [v.get(k, 0) for k in single] + [multi, sum(v.values())])
    T(3, 'Number of images per collection, source and class', ['Collection', 'Source'] + [NICE[k] for k in single] + ['Multi-label', 'Total'], out,
      ['EVAL450 is the development collection; its sources are the contributing sub-sources of WFD2020; multi-label images (EVAL450 only) are counted once, in the Multi-label column.'])
    add_three_line_table(doc, TABLES[-1])

    # S4 per-class accuracy and confusion
    doc.add_heading('S4 Per-class results', level=1)
    out = []
    for c in COLL:
        rr = [r for r in rows if r['coll'] == c and len(r['truth']) == 1]
        for k in LAB:
            sub = [r for r in rr if k in r['truth']]
            if not sub:
                continue
            ans = [r for r in sub if r['sys'][0] in ('healthy', 'group', 'specific')]
            ok = sum(H.correct(r['sys'][0], r['sys'][1], r['truth']) for r in ans)
            out.append([COLL[c], NICE[k], len(sub), pct(sum(r['lab_L'] == k for r in sub), len(sub)),
                        pct(sum(r['lab_G'] == k for r in sub), len(sub)), pct(sum(r['lab_R'] == k for r in sub), len(sub)),
                        pct(len(ans), len(sub)), pct(ok, len(ans))])
    T(4, 'Forced-choice recall per class and agreement-gated system output per class (single-label images)',
      ['Collection', 'True class', 'n', 'L % (k/n)', 'G61 % (k/n)', 'RAG % (k/n)', 'System coverage % (k/n)', 'System accuracy % (k/n)'], out,
      ['G61 and RAG answers "uncertain" are counted as wrong. System accuracy is computed among answered images at the level of each answer.'])
    add_three_line_table(doc, TABLES[-1])
    with open(HERE / 'TableS4_confusion.csv', 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.writer(fh); w.writerow(['collection', 'expert', 'true'] + [NICE[k] for k in LAB] + ['Uncertain'])
        for c in ('ETS', 'ETS2', 'P4'):
            for ex, key in (('L', 'lab_L'), ('G61', 'lab_G')):
                conf = defaultdict(Counter)
                for r in rows:
                    if r['coll'] == c:
                        conf[next(iter(r['truth']))][r[key]] += 1
                for t in LAB:
                    if conf[t]:
                        w.writerow([COLL[c], ex, NICE[t]] + [conf[t].get(k, 0) for k in LAB] + [conf[t].get(None, 0)])
    doc.add_paragraph('Full confusion matrices of L and G61 on the external collections are provided as TableS4_confusion.csv.')

    # S5 threshold grid and nested cross-validation
    doc.add_heading('S5 Threshold selection on the development collection', level=1)
    doc.add_paragraph('Grid: τ_h, τ_d, τ_g ∈ {0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98}; G61 confidence required for a rust type ∈ {none, 70, 80, '
                      '90, 95}; health-gate and group-gate agreement switches on/off (6,860 configurations). Selection criterion: maximise '
                      'the share of answered images subject to an error rate ≤ α among answered images, evaluated with "disease present '
                      'only" outputs counted as unanswered. Single-expert baselines: L released its arg-max when the maximum probability '
                      'exceeded a threshold in {0.5, …, 0.999}; G61 released its label when its confidence exceeded a threshold in {0, 50, …, 98}. '
                      'Frozen configuration (selected on all 450 images at α = 0.05): τ_h = τ_d = τ_g = 0.5, rust-type confidence ≥ 95, both gates on.'.replace('τ_', 'τ'))
    v3 = json.loads((ROOT / 'outputs/paper_hrme/p3_v3_Ldino.json').read_text(encoding='utf-8'))
    name = {'HRME_v3': 'Agreement-gated system', 'HRME_noG_strict': 'L-only hierarchy (no MLLM gate)', 'B1_flat_L': 'L alone (probability threshold)',
            'B2_flat_G': 'G61 alone (confidence threshold)'}
    out = []
    for a in ('alpha_0.05', 'alpha_0.1'):
        for k, v in v3[a].items():
            out.append([a.split('_')[1], name[k], f"{100 * v['informative_coverage']:.1f}", f"{v['errors']}/{v['released']}",
                        f"{100 * v['risk']:.1f}" if v['released'] else '–', f"{100 * v['risk_cp95']:.1f}" if v['released'] else '–',
                        '; '.join('none' if c is None else str(tuple(c)) for c in v['chosen'])])
    T(5, 'Source-held nested cross-validation of threshold selection on EVAL450 (450 images, three source folds)',
      ['α', 'Method', 'Coverage %', 'Errors/answered', 'Error %', 'Upper 95 % limit %', 'Configuration chosen in each outer fold'], out,
      ['Configurations are (τh, τd, τg, rust-type confidence, health gate, group gate); "none" means no configuration met the error constraint '
       'on the inner folds, so the outer fold received no automatic answers.'])
    add_three_line_table(doc, TABLES[-1])

    # S6 protocol timeline
    doc.add_heading('S6 Order of prespecification and evaluation', level=1)
    tl = [
        ['29 Sep 2026', 'Prompt of G61 fixed', 'Development'],
        ['7 Oct 2026', 'Protocol: data roles, three source folds, metrics and primary criteria written', 'Development'],
        ['7 Oct 2026', 'First rule version (rust type released on L–G61 agreement) fails the 5 % error target in nested cross-validation', 'Development'],
        ['7 Oct 2026', 'MLLM-free variant applied to IARI laboratory images (852 detached leaves): no automatic answers; not analysed further', 'Earlier internal collection'],
        ['7 Oct 2026', 'Rule revised (optional rust-type level; then agreement gates); final rule and α = 0.05 selection fixed on EVAL450; configuration frozen', 'Development'],
        ['7 Oct 2026', 'iNat rules fixed before download and inference; iNat evaluated once', 'External (prespecified)'],
        ['7 Oct 2026', 'ETS rules and primary hypothesis (retrieval augmentation) fixed before inference', 'External (prespecified)'],
        ['8 Oct 2026', 'ETS evaluated once', 'External (prespecified)'],
        ['8 Oct 2026', 'ETS2 rules and hypotheses H1–H3 fixed before download and inference; ETS2 evaluated once', 'External (prespecified)'],
        ['8 Oct 2026', 'Learned arbiters, knowledge-rule arbiter and K-shot selection: rules written before computation, on already examined data',
         'Retrospective'],
    ]
    T(6, 'Timeline of the internal, time-stamped protocol', ['Date', 'Step', 'Data role'], tl,
      ['The protocol is an internal, version-controlled document (available with the code); it was not registered in a public registry. '
       'All prespecified hypotheses and their outcomes, including those not met, are listed in Table 2 of the main text.'])
    add_three_line_table(doc, TABLES[-1])

    # S7 near-duplicate audit
    doc.add_heading('S7 Near-duplicate audit of public collections', level=1)
    lk = json.loads((MS / 'analysis/ms_results.json').read_text(encoding='utf-8'))['leakage']
    nm = {'WFD2020': 'WFD2020', 'kaggle_wheat_disease_small': 'CerealConv subset', 'kaggle_wheat_leaf_disease_jayaprakash': 'Jayaprakash',
          'kaggle_wheat_plant_diseases_kushagra': 'Kushagra', 'mendeley_wheat_disease_2025_original': 'Mendeley 2025',
          'mendeley_wheat_leaf_dataset': 'Mendeley leaf'}
    out = []
    for key, v in lk['pairs'].items():
        a, b = key.split('|')
        out.append([nm[a], nm[b], v['n_a'], v['ge090'], f"{100 * v['ge090'] / v['n_a']:.1f}", v['ge095'], f"{100 * v['ge095'] / v['n_a']:.1f}"])
    T(7, 'Images of collection A with at least one near-duplicate in collection B (pretrained DINOv3 cosine similarity)',
      ['Collection A', 'Collection B', 'Images in A', '≥ 0.90 (n)', '≥ 0.90 (%)', '≥ 0.95 (n)', '≥ 0.95 (%)'], out,
      [f"{lk['n_images']:,} unique images in total. The 0.90 threshold was used for all de-duplication in this study."])
    add_three_line_table(doc, TABLES[-1])

    # S8 case images
    doc.add_heading('S8 Images shown in Fig. 4', level=1)
    with open(MS / 'figures/Fig4_cases.csv', encoding='utf-8') as fh:
        cases = list(csv.DictReader(fh))
    T(8, 'Source files of the case images', ['Panel', 'Collection', 'Source', 'File', 'Truth', 'L (p)', 'G61 (confidence)', 'System output'],
      [[c['panel'], COLL.get(c['collection'], c['collection']), SRC.get(f"{c['collection'].lower()}:{c['source']}", c['source']), c['file'],
        NICE[c['truth']], f"{NICE[c['L_pred']]} ({c['L_prob']})", f"{NICE[c['G61_pred']]} ({c['G61_conf']})", c['system_output']] for c in cases],
      ['All six images are distributed under CC BY 4.0 by their publishers.'])
    add_three_line_table(doc, TABLES[-1])

    V = json.loads((MS / 'analysis/ms_revision.json').read_text(encoding='utf-8'))
    SRCN = {'ets2:henan_field_2023': 'Henan field 2023 (ETS2)', 'ets:mswdd2022': 'MSWDD2022 (ETS)', 'ets:plantwild': 'PlantWild v2 web (ETS)',
            'ets:roboflow_newwheat': 'Roboflow new-wheat-disease v2 (ETS)', 'ets:roboflow_ptriticina': 'Roboflow P. triticina v7 (ETS)',
            'ets:roboflow_stemrust': 'Roboflow stem-rust v1 (ETS)', 'ets:zenodo13137587': 'Zenodo 13137587 powdery mildew (ETS)',
            'p4_rag:inat_pathogen': 'iNaturalist (iNat)'}
    # S9 per-source results
    doc.add_heading('S9 Results of the frozen system per external source', level=1)
    ps = V['per_source']; out = []
    for s in sorted(ps, key=lambda q: -ps[q]['n']):
        v = ps[s]
        out.append([SRCN[s], v['n'], f"{v['answered']} ({100 * v['coverage']:.1f})", v['errors'],
                    f"{100 * v['accuracy']:.1f} [{100 * v['accuracy_ci95'][0]:.1f}–{100 * v['accuracy_ci95'][1]:.1f}]",
                    f"{v['specific']} / {v['group']} / {v['healthy']}", f"{100 * v['L_forced']:.1f}", f"{100 * v['G61_forced']:.1f}"])
    sm = V['source_macro']; io = V['inat_observer_sensitivity']
    T(9, 'Automatic answers per external source', ['Source', 'Images', 'Answered (coverage %)', 'Errors', 'Accuracy % [95 % CI]',
                                                   'Specific / group / healthy', 'L forced %', 'G61 forced %'], out,
      [f"Mean over the {sm['n_sources']} sources: coverage {100 * sm['coverage']:.1f} %, accuracy among answered images {100 * sm['accuracy']:.1f} %. "
       f"iNat without the {io['excluded']} images whose observers also contributed training images: {io['answered']} of {io['n']} answered, {io['errors']} errors."])
    add_three_line_table(doc, TABLES[-1])

    # S10 erroneous answers
    doc.add_heading('S10 All erroneous automatic answers on the external collections', level=1)
    note = {0: 'septoria answered as powdery mildew', 1: 'source label questionable (rust pustules visible)', 2: 'rust-type confusion',
            3: 'source label questionable (detached tile without visible lesions inherits the leaf-rust label)',
            4: 'source label questionable (scattered round pustules; file name "LeafRust"; Fig. 4f)', 5: 'rust-type confusion',
            6: 'septoria answered as rust group', 7: 'septoria answered as rust group'}
    out = []
    for k, e in enumerate(V['erroneous_answers']):
        out.append([k + 1, COLL.get(e['collection'], e['collection']), SRCN[e['source']], Path(e['file']).name, NICE[e['truth']],
                    f"{NICE[e['L']]} ({e['L_prob']:.2f})", f"{NICE[e['G61']]} ({e['G61_conf']})", e['system'].replace('R', 'rust') if e['system'].startswith('group') else e['system'],
                    note[k]])
    T(10, 'Erroneous automatic answers (8 of 598)', ['#', 'Collection', 'Source', 'File', 'Label', 'L (p)', 'G61 (conf.)', 'System output', 'Visual note'], out,
      ['All eight are counted as errors in every analysis. Visual notes are descriptive and were not used to change labels or results.'])
    add_three_line_table(doc, TABLES[-1])

    # S11 visual validation of the near-duplicate threshold
    doc.add_heading('S11 Visual validation of the near-duplicate threshold', level=1)
    vv = V['dedup_visual_validation']
    T(11, 'Nearest cross-collection pairs rated as the same photograph, by similarity band',
      ['Cosine similarity band', 'Pairs rated', 'Same photograph, visual', 'Same photograph, keypoint matching', 'Both'],
      [[b, v['n'], v['same'], v['keypoint_same'], v['both']] for b, v in vv.items()],
      ['6,000 random images were paired with their most similar image in another collection; 20 pairs per band were rendered side by side and rated. '
       'Crops, resizing, flips and colour changes count as the same photograph. Keypoint matching: SIFT, Lowe ratio 0.75, RANSAC homography, at least 30 inliers, flipped versions included; it misses strongly cropped low-resolution copies. Pair-level results: analysis/dedup_validation/ratings.csv and keypoint_check.csv.'])
    add_three_line_table(doc, TABLES[-1])

    # Fig. S1
    doc.add_heading('Fig. S1 Forced-choice fusion strategies', level=1)
    doc.add_paragraph().add_run().add_picture(str(MS / 'figures/FigS1.png'), width=Cm(17))
    doc.add_paragraph('Fig. S1 Forced-choice accuracy of fusion strategies across the 16 sources in retrospective leave-one-source-out analysis '
                      '(1,310 images). Learned arbiters were trained on the 15 sources not being evaluated; local rule selection uses 10 labelled '
                      'images per source (mean of 50 draws). The two-expert selection bound is an oracle reference. Exact values are given in Table 5 of the main text.')

    doc.save(HERE / 'ESM_1.docx'); print('saved', HERE / 'ESM_1.docx')


if __name__ == '__main__':
    main()
