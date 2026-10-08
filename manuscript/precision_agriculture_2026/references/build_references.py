"""Build the APA-style reference list (journal rules: alphabetical, full journal names in italics, DOI links, max. six authors
then et al., web references with last-accessed date) from refs_verified.json plus manually checked data/web sources.
Writes references.json (key -> {text, cite}) and references.txt."""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
V = json.loads((HERE / 'refs_verified.json').read_text(encoding='utf-8'))
ACCESSED = '8 October 2026'


def apa_name(n):
    n = n.strip()
    if ',' in n:
        fam, giv = [x.strip() for x in n.split(',', 1)]
    else:
        parts = n.split(); fam, giv = parts[-1], ' '.join(parts[:-1])
    fam = fam.title() if fam.isupper() else fam
    ini = ' '.join(p[0] + '.' for p in re.split(r'[\s\-]+', giv.replace('.', ' ')) if p)
    return f'{fam}, {ini}'.strip(', ')


def authors(lst):
    a = [apa_name(x) for x in lst]
    if len(a) > 6:
        return ', '.join(a[:6]) + ', et al.'
    if len(a) == 1:
        return a[0]
    return ', '.join(a[:-1]) + ', & ' + a[-1]


def cite_key(lst, year):
    fam = [apa_name(x).split(',')[0] for x in lst]
    if len(fam) == 1:
        return f'{fam[0]}, {year}'
    if len(fam) == 2:
        return f'{fam[0]} & {fam[1]}, {year}'
    return f'{fam[0]} et al., {year}'


TITLE_FIX = {'singh2020plantdoc': 'PlantDoc: A dataset for visual plant disease detection',
             'clopper1934': 'The use of confidence or fiducial limits illustrated in the case of the binomial'}
VENUE_FIX = {'singh2020plantdoc': 'Proceedings of the 7th ACM IKDD CoDS and 25th COMAD', 'mohanty2016': 'Frontiers in Plant Science'}
USED = ['savary2019', 'mohanty2016', 'long2023cerealconv', 'genaev2021wfd', 'mao2024daemask', 'singh2020plantdoc', 'barbedo2018',
        'bennin2026', 'noyan2022', 'geirhos2020', 'ovadia2019', 'simeoni2025dinov3', 'han2025fomo4wheat', 'pa2026llm', 'wiswheat2025',
        'selfconsistency2025', 'fusionexperts2026', 'kuncheva2003', 'geifman2017', 'clopper1934', 'wei2024plantwild', 'wei2024plantseg',
        'ma2024zenodo']
V['savary2019'] = dict(status='ok', title='The global burden of pathogens and pests on major food crops',
                       authors=['Savary, S.', 'Willocquet, L.', 'Pethybridge, S. J.', 'Esker, P.', 'McRoberts, N.', 'Nelson, A.'],
                       year=2019, venue='Nature Ecology & Evolution', volume='3', issue='3', pages='430-439', doi='10.1038/s41559-018-0793-y')


def sentence_case(t):
    def part(w, cap):
        keep = (any(ch.isupper() for ch in w[1:])) or (w.isupper() and len(w) > 1)
        return w if keep else (w[:1].upper() + w[1:].lower() if cap else w.lower())
    out, cap_next = [], True
    for w in t.split(' '):
        pieces = w.split('-')
        out.append('-'.join(part(x, cap_next and i == 0) for i, x in enumerate(pieces)))
        cap_next = w.endswith(':') or w.endswith('?')
    return ' '.join(out)


def fmt(k, v):
    title = sentence_case(TITLE_FIX.get(k, v['title']).rstrip('.')).replace('DAE-mask', 'DAE-Mask')
    venue = VENUE_FIX.get(k, v.get('venue', ''))
    au = authors(v['authors'])
    if v.get('venue') == 'arXiv':
        return f"{au} ({v['year']}). {title}. *arXiv* preprint arXiv:{v['arxiv']}. https://doi.org/10.48550/arXiv.{v['arxiv']}"
    if v.get('status') == 'ok-zenodo':
        return f"{au} ({v['year']}). *{title}* [Data set]. Zenodo. https://doi.org/{v['doi']}"
    vol = f", *{v['volume']}*" if v.get('volume') else ''
    iss = f"({v['issue']})" if v.get('issue') else ''
    pg = f", {str(v['pages']).replace('-', '–')}" if v.get('pages') else ''
    return f"{au} ({v['year']}). {title}. *{venue}*{vol}{iss}{pg}. https://doi.org/{v['doi']}"


def main() -> None:
    refs = {}
    for k in USED:
        v = V[k]
        assert v['status'].startswith('ok'), k
        refs[k] = dict(text=fmt(k, v), cite=cite_key(v['authors'], v['year']), doi=v.get('doi') or v.get('arxiv'), supports=v.get('supports'))
    for k, suf in (('wei2024plantwild', 'a'), ('wei2024plantseg', 'b')):
        refs[k]['text'] = refs[k]['text'].replace('(2024).', f'(2024{suf}).', 1); refs[k]['cite'] += suf
    manual = {
        'onler2024': dict(text='Önler, E., & Köycü, N. D. (2024). *Wheat powdery mildew image dataset* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.13137587',
                          cite='Önler & Köycü, 2024'),
        'getachew2021': dict(text='Getachew, H. (2021). *Wheat leaf dataset* (Version 1) [Data set]. Mendeley Data. https://doi.org/10.17632/wgd66f8n6h.1',
                             cite='Getachew, 2021'),
        'roboflowP': dict(text=f'Roboflow Universe (2026). *Puccinia triticina* (Version 7) [Data set, CC BY 4.0]. https://universe.roboflow.com/louisvalentins-workspace/puccinia-triticina. Last accessed {ACCESSED}',
                          cite='Roboflow Universe, 2026'),
        'roboflowS': dict(text=f'Roboflow Universe (2023). *Stem rust* (Version 1) [Data set, CC BY 4.0]. https://universe.roboflow.com/kinza-kamal/stem-rust. Last accessed {ACCESSED}',
                          cite='Roboflow Universe, 2023'),
        'roboflowN': dict(text=f'Roboflow Universe (2022). *New wheat disease* (Version 2) [Data set, CC BY 4.0]. https://universe.roboflow.com/object-detection/new-wheat-disease. Last accessed {ACCESSED}',
                          cite='Roboflow Universe, 2022'),
        'inaturalist': dict(text=f'iNaturalist (2026). *iNaturalist API* (observations of wheat pathogens). https://api.inaturalist.org/v1/docs/. Last accessed {ACCESSED}',
                            cite='iNaturalist, 2026'),
        'openai_codex': dict(text=f'OpenAI (2026). *Codex CLI* (Version 0.160.1) and model gpt-6.1-sol [Software]. https://github.com/openai/codex. Last accessed {ACCESSED}',
                             cite='OpenAI, 2026'),
        'arya2020': dict(text='Arya, S. (2020). *Wheat nitrogen deficiency and leaf rust image dataset* (Version 1) [Data set, CC BY 4.0]. Mendeley Data. https://doi.org/10.17632/th422bg4yd.1',
                         cite='Arya, 2020'),
        'jia2026code': dict(text='Jia, P., & Zhang, P. (2026). *Agreement-gated fusion of a vision foundation model and a multimodal LLM for wheat disease diagnosis: Code, frozen prompt and cached outputs* (Version 1.0.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.23235119',
                            cite='Jia & Zhang, 2026'),
        'radowan2025': dict(text='Radowan, M. I. R., & Ayon, R. (2025). *Disease dataset of wheat: Original, augmented, and balanced for deep learning* (Version 1) [Data set, CC BY 4.0]. Mendeley Data. https://doi.org/10.17632/5gc7hwydwg.1',
                            cite='Radowan & Ayon, 2025'),
        'kushagra2024': dict(text=f'kushagra3204 (2024). *Wheat plant diseases* (Version 6) [Data set, CC0]. Kaggle. https://www.kaggle.com/datasets/kushagra3204/wheat-plant-diseases. Last accessed {ACCESSED}',
                             cite='kushagra3204, 2024'),
        'jayaprakash2023': dict(text=f'jayaprakashpondy (2023). *Wheat leaf disease* (Version 1) [Data set, CC0]. Kaggle. https://www.kaggle.com/datasets/jayaprakashpondy/wheat-leaf-disease. Last accessed {ACCESSED}',
                                cite='jayaprakashpondy, 2023'),
    }
    refs.update(manual)
    order = sorted(refs, key=lambda k: re.sub(r'[^a-z0-9]', '', refs[k]['text'].lower().replace('ö', 'o').replace('é', 'e'))[:40])
    (HERE / 'references.json').write_text(json.dumps({k: refs[k] for k in order}, indent=1, ensure_ascii=False), encoding='utf-8')
    (HERE / 'references.txt').write_text('\n'.join(refs[k]['text'] for k in order), encoding='utf-8')
    for k in order:
        print(f"{refs[k]['cite']:32s} | {refs[k]['text'][:150]}")


if __name__ == '__main__':
    main()
