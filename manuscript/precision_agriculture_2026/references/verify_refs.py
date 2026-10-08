"""Fetch and verify bibliographic metadata from Crossref (DOIs) and arXiv (IDs). Writes refs_verified.json.

Each entry keeps the claim it is meant to support, so the text can be checked against the source.
"""
import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
S = requests.Session(); S.headers['User-Agent'] = 'manuscript-reference-check (mailto:not-provided)'
DOIS = {
    'bennin2026': ('10.1007/s11119-026-10398-7', 'lab-to-field gap; accuracy insufficient; private/unrealistic datasets'),
    'han2026apple': ('10.1007/s11119-026-10451-5', 'recent PA disease diagnosis system in field conditions'),
    'mao2024daemask': ('10.1007/s11119-023-10093-x', 'MSWDD2022 dataset (ETS source); in-field wheat disease detection'),
    'pa2026llm': ('10.1007/s11119-026-10349-2', 'LLMs in smart farming / precision agriculture'),
    'genaev2021wfd': ('10.3390/plants10081500', 'WFD2020 wheat fungal disease dataset (development set)'),
    'mohanty2016': ('10.3389/fpls.2016.01419', 'early deep learning plant disease; lab images; generalisation drop'),
    'barbedo2018': ('10.1016/j.biosystemseng.2018.05.013', 'factors limiting deep learning plant disease recognition (data variability)'),
    'singh2020plantdoc': ('10.1145/3371158.3371196', 'PlantDoc field dataset; lab-to-field gap'),
    'kuncheva2003': ('10.1023/A:1022859003006', 'ensemble diversity / complementary errors'),
    'clopper1934': ('10.1093/biomet/26.4.404', 'exact binomial confidence intervals'),
    'geirhos2020': ('10.1038/s42256-020-00257-z', 'shortcut learning'),
    'long2023cerealconv': ('10.1111/ppa.13684', 'CerealConv wheat disease images; field and glasshouse'),
    'ma2024zenodo': ('10.5281/zenodo.15621359', 'Henan field wheat disease dataset (ETS2)'),
}
ARXIV = {
    'simeoni2025dinov3': ('2508.10104', 'DINOv3 vision foundation model'),
    'han2025fomo4wheat': ('2509.06907', 'FoMo4Wheat wheat foundation model'),
    'wiswheat2025': ('2506.06084', 'WisWheat; GPT-4o accuracy on wheat stress / growth stage'),
    'geifman2017': ('1705.08500', 'selective classification with a reject option'),
    'ovadia2019': ('1906.02530', 'uncertainty under dataset shift; overconfidence'),
    'wei2024plantwild': ('2408.03120', 'PlantWild in-the-wild plant disease benchmark (ETS source)'),
    'wei2024plantseg': ('2409.04038', 'PlantSeg dataset (candidate source removed by dedup)'),
    'noyan2022': ('2206.04374', 'PlantVillage bias: background-only accuracy'),
    'selfconsistency2025': ('2507.08024', 'multi-response consensus for VLM crop disease diagnosis'),
    'fusionexperts2026': ('2608.24934', 'fusing vision experts with MLLMs for plant disease diagnosis'),
}


def crossref(doi):
    r = S.get(f'https://api.crossref.org/works/{doi}', timeout=60)
    if r.status_code != 200:
        if 'zenodo' in doi:
            rid = doi.split('.')[-1]
            z = S.get(f'https://zenodo.org/api/records/{rid}', timeout=60).json()['metadata']
            return dict(status='ok-zenodo', title=z['title'], authors=[c['name'] for c in z.get('creators', [])],
                        year=z.get('publication_date', '')[:4], venue='Zenodo', doi=doi)
        return dict(status=f'http {r.status_code}', doi=doi)
    m = r.json()['message']
    au = [f"{a.get('family', '')}, {' '.join(x[0] + '.' for x in a.get('given', '').replace('-', ' ').split())}".strip(', ')
          for a in m.get('author', [])]
    year = (m.get('published-print') or m.get('published-online') or m.get('issued'))['date-parts'][0][0]
    return dict(status='ok', title=(m.get('title') or [''])[0], authors=au, year=year, venue=(m.get('container-title') or [''])[0],
                volume=m.get('volume'), issue=m.get('issue'), pages=m.get('page') or m.get('article-number'), doi=doi)


def arxiv(aid):
    r = None
    for url in (f'https://export.arxiv.org/api/query?id_list={aid}', f'http://export.arxiv.org/api/query?id_list={aid}'):
        try:
            r = S.get(url, timeout=60); break
        except Exception:
            time.sleep(5)
    if r is None:
        return dict(status='network error', arxiv=aid)
    ns = {'a': 'http://www.w3.org/2005/Atom'}
    e = ET.fromstring(r.text).find('a:entry', ns)
    if e is None or e.find('a:title', ns) is None:
        return dict(status='not found', arxiv=aid)
    title = re.sub(r'\s+', ' ', e.find('a:title', ns).text).strip()
    au = [a.find('a:name', ns).text for a in e.findall('a:author', ns)]
    return dict(status='ok', title=title, authors=au, year=e.find('a:published', ns).text[:4], venue='arXiv', arxiv=aid,
                url=f'https://arxiv.org/abs/{aid}')


def main() -> None:
    out = {}
    for k, (doi, claim) in DOIS.items():
        out[k] = dict(crossref(doi), supports=claim); time.sleep(0.5)
    for k, (aid, claim) in ARXIV.items():
        out[k] = dict(arxiv(aid), supports=claim); time.sleep(3)
    (HERE / 'refs_verified.json').write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
    for k, v in out.items():
        print(f"{k:22s} {v['status']:10s} {v.get('year')} | {', '.join(v.get('authors', [])[:3])} | {v.get('title', '')[:90]} | {v.get('venue', '')}")


if __name__ == '__main__':
    main()
