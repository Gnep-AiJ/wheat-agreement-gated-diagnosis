"""Cross-dataset leakage audit of local public wheat-disease datasets (+EVAL450 sources): pretrained DINOv3-B CLS cosine.

Counts, for each dataset A and other dataset B, images of A with a near-duplicate (cos >= 0.90; also >= 0.95) in B.
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.vendor/transformers_v2'))
OUT = ROOT / 'outputs/paper_hrme/leak_audit'


class DS:
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        import torch
        from PIL import Image
        try:
            im = Image.open(self.paths[i]).convert('RGB').resize((384, 384), Image.Resampling.BILINEAR)
        except Exception:
            im = Image.new('RGB', (384, 384))
        x = torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255
        return (x - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]


def main() -> None:
    import torch
    from transformers import AutoModel
    torch.set_num_threads(6)
    OUT.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'outputs/stage0/data_audit_v2_six_sources/image_manifest.csv').open(encoding='utf-8-sig') as f:
        rows = [r for r in csv.DictReader(f) if r['readable'] in ('True', 'true', '1', 'yes', '')]
    # one image per exact sha within a dataset
    seen, items = set(), []
    for r in rows:
        k = (r['dataset'], r['sha256'])
        if k not in seen:
            seen.add(k); items.append(r)
    cache = OUT / 'feats.npy'
    if cache.exists():
        X = np.load(cache)
    else:
        m = AutoModel.from_pretrained(ROOT / 'outputs/pretrained/dinov3_vitb16', local_files_only=True).eval()
        out = []
        with torch.inference_mode():
            for i, x in enumerate(torch.utils.data.DataLoader(DS([r['absolute_path'] for r in items]), batch_size=32, num_workers=3)):
                out.append(torch.nn.functional.normalize(m(pixel_values=x).pooler_output, dim=1).numpy().astype(np.float16))
                if i % 50 == 0:
                    print('feat', i * 32, len(items), flush=True)
        X = np.concatenate(out); np.save(cache, X)
    ds = np.array([r['dataset'] for r in items])
    names = sorted(set(ds))
    Xt = torch.from_numpy(X.astype(np.float32))
    res = {}
    for a in names:
        ia = np.where(ds == a)[0]
        for b in names:
            if a == b:
                continue
            ib = np.where(ds == b)[0]
            mx = np.concatenate([(Xt[ia[i:i + 2048]] @ Xt[ib].T).max(1).values.numpy() for i in range(0, len(ia), 2048)])
            res[f'{a}|{b}'] = dict(n_a=int(len(ia)), ge090=int((mx >= .90).sum()), ge095=int((mx >= .95).sum()))
            print(a, b, res[f'{a}|{b}'], flush=True)
    (OUT / 'audit.json').write_text(json.dumps(dict(n_images=len(items), per_dataset={n: int((ds == n).sum()) for n in names}, pairs=res), indent=1))


if __name__ == '__main__':
    main()
