"""Build the external test set (ETS) per protocol section 13.1. No model predictions are computed here.

Shared resources used (junctions under <DATA_ROOT>/shared/datasets):
  extra_plantwild_v2/wheat_originals, extra_staging_d/mswdd2022_32c56f23/extracted, extra_staging_d/zenodo_13137587/extracted,
  extra_staging_c/targeted_rust_20260927/{new-wheat-disease-v2,stem-rust-v1}, extra_staging_c/roboflow_puccinia_triticina_v7.
Dedup reference: outputs/paper_hrme/p4_inat/local_feats.npy (pretrained DINOv3 CLS of all local train/dev/test pools).
Output: outputs/paper_hrme/ets/items.json
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.vendor/transformers_v2'))
SH = Path('<DATA_ROOT>/shared/datasets')
OUT = ROOT / 'outputs/paper_hrme/ets'
OUT2 = ROOT / 'outputs/paper_hrme/ets2'
IMG = ('.jpg', '.jpeg', '.png')


def coco_items(folder, mapping, source):
    out = []
    for split in ('train', 'valid', 'test'):
        f = folder / split / '_annotations.coco.json'
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        cat = {c['id']: c['name'] for c in d['categories']}
        labs = defaultdict(set)
        for a in d['annotations']:
            labs[a['image_id']].add(mapping.get(cat[a['category_id']]))
        for im in d['images']:
            s = labs.get(im['id'], set())
            if len(s) == 1 and None not in s:
                out.append(dict(file=str(folder / split / im['file_name']), label=next(iter(s)), source=source))
    return out


def collect_ets2():
    """Protocol section 17: Henan field (Zenodo 15621359, YOLO labels) + PlantSeg v7 wheat (web)."""
    items = []
    hn = ROOT / 'data/staging/zenodo_15621359/extracted/xiaomai'
    hmap = {0: 'powdery_mildew', 2: 'leaf_rust', 3: 'yellow_rust', 6: 'healthy'}
    for split in ('train', 'valid', 'test'):
        for lab in sorted((hn / 'labels' / split).glob('*.txt')):
            cls = {int(float(l.split()[0])) for l in lab.read_text().splitlines() if l.strip()}
            if len(cls) == 1 and next(iter(cls)) in hmap:
                im = [p for p in (hn / 'images' / split).glob(lab.stem + '.*') if p.suffix.lower() in IMG]
                if im:
                    items.append(dict(file=str(im[0]), label=hmap[next(iter(cls))], source='henan_field_2023'))
    import csv
    ps = next((SH / 'extra_staging_c/plantseg_v7/extracted').iterdir())
    pmap = {'wheat stripe rust': 'yellow_rust', 'wheat powdery mildew': 'powdery_mildew', 'wheat septoria blotch': 'septoria',
            'wheat stem rust': 'stem_rust', 'wheat leaf rust': 'leaf_rust'}
    split_dir = {'Training': 'train', 'Validation': 'val', 'Test': 'test'}
    for r in csv.DictReader(open(ps / 'Metadata.csv', encoding='utf-8-sig', errors='replace')):
        if r['Disease'] in pmap:
            f = ps / 'images' / split_dir.get(r['Split'], r['Split'].lower()) / r['Name']
            if f.exists():
                items.append(dict(file=str(f), label=pmap[r['Disease']], source='plantseg_web'))
    return items


def collect():
    items = []
    pw = SH / 'extra_plantwild_v2/wheat_originals'
    for cls, lab in (('wheat leaf rust', 'leaf_rust'), ('wheat stem rust', 'stem_rust'), ('wheat stripe rust', 'yellow_rust'),
                     ('wheat powdery mildew', 'powdery_mildew'), ('wheat septoria blotch', 'septoria')):
        items += [dict(file=str(p), label=lab, source='plantwild') for p in sorted((pw / cls).iterdir()) if p.suffix.lower() in IMG]
    ms = SH / 'extra_staging_d/mswdd2022_32c56f23/extracted'
    for cls, lab in (('wheat stripe rust', 'yellow_rust'), ('wheat powdery mildew', 'powdery_mildew')):
        items += [dict(file=str(p), label=lab, source='mswdd2022') for p in sorted((ms / cls / 'img').iterdir()) if p.suffix.lower() in IMG]
    zn = SH / 'extra_staging_d/zenodo_13137587/extracted/images'
    items += [dict(file=str(p), label='powdery_mildew', source='zenodo13137587') for p in sorted(zn.rglob('*'))
              if p.suffix.lower() in IMG and not p.stem.endswith(('_0', '_1', '_2', '_3'))]
    rt = SH / 'extra_staging_c/targeted_rust_20260927'
    items += coco_items(rt / 'new-wheat-disease-v2', {'Wheat Brown-rust': 'leaf_rust', 'Wheat Healthy': 'healthy',
                                                       'Wheat-Yellow-rust': 'yellow_rust', 'wheat Stem Rust': 'stem_rust'}, 'roboflow_newwheat')
    items += coco_items(rt / 'stem-rust-v1', {'stemrust': 'stem_rust', 'leafrust': 'leaf_rust', 'healthywheat': 'healthy'}, 'roboflow_stemrust')
    items += coco_items(SH / 'extra_staging_c/roboflow_puccinia_triticina_v7', {'Puccinia-triticina': 'leaf_rust'}, 'roboflow_ptriticina')
    return items


class DS:
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        import torch
        from PIL import Image
        im = Image.open(self.paths[i]).convert('RGB').resize((384, 384), Image.Resampling.BILINEAR)
        x = torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255
        return (x - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]


def main() -> None:
    import torch
    from transformers import AutoModel
    global OUT
    ets2 = len(sys.argv) > 1 and sys.argv[1] == 'ets2'
    if ets2:
        OUT = OUT2
    OUT.mkdir(parents=True, exist_ok=True)
    items = collect_ets2() if ets2 else collect()
    print('candidates', len(items), flush=True)
    m = AutoModel.from_pretrained(ROOT / 'outputs/pretrained/dinov3_vitb16', local_files_only=True).eval().cuda()
    feats = []
    with torch.inference_mode():
        for x in torch.utils.data.DataLoader(DS([it['file'] for it in items]), batch_size=32, num_workers=4):
            feats.append(torch.nn.functional.normalize(m(pixel_values=x.cuda()).pooler_output.float(), dim=1).cpu().numpy())
    X = np.concatenate(feats); np.save(OUT / 'candidate_feats.npy', X)
    local = np.load(ROOT / 'outputs/paper_hrme/p4_inat/local_feats.npy').astype(np.float32)
    if ets2:  # also exclude anything near any ETS candidate (protocol 17)
        local = np.concatenate([local, np.load(ROOT / 'outputs/paper_hrme/ets/candidate_feats.npy').astype(np.float32)])
    local = torch.from_numpy(local).cuda()
    Xt = torch.from_numpy(X).cuda()
    near_local = torch.cat([(Xt[i:i + 512] @ local.T).max(1).values for i in range(0, len(Xt), 512)]).cpu().numpy()
    S = (Xt @ Xt.T).cpu().numpy()
    n = len(items); parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for a, b in zip(*np.where(np.triu(S >= .90, 1))):
        parent[find(a)] = find(b)
    comp_labels = defaultdict(set)
    for i in range(n):
        comp_labels[find(i)].add(items[i]['label'])
    seen, keep = set(), []
    for i in range(n):
        r = find(i)
        if near_local[i] >= .90 or r in seen or len(comp_labels[r]) > 1:
            continue
        seen.add(r); keep.append(i)
    by = defaultdict(list)
    for i in keep:
        by[(items[i]['source'], items[i]['label'])].append(i)
    rng = random.Random(20261008 if ets2 else 20261007); final = []
    cap = 60 if ets2 else 40
    for k in sorted(by):
        idx = by[k][:]; rng.shuffle(idx); final += idx[:cap]
    out = [dict(items[i], max_local_cos=float(near_local[i])) for i in final]
    (OUT / 'items.json').write_text(json.dumps(out, indent=0))
    print('near-local removed', int((near_local >= .90).sum()), 'unique kept', len(keep), 'sampled', len(out))
    print(json.dumps({f'{s}|{l}': [len(v), min(len(v), cap)] for (s, l), v in sorted(by.items())}, indent=0))


if __name__ == '__main__':
    main()
