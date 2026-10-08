"""P4: feature-level dedup of the iNat fresh test against all local train/dev images (pretrained DINOv3-B CLS, cos >= 0.90). CPU."""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / '.vendor/transformers_v2'))
P4 = ROOT / 'outputs/paper_hrme/p4_inat'


def main() -> None:
    import torch
    from PIL import Image
    from transformers import AutoModel
    torch.set_num_threads(8)
    m = AutoModel.from_pretrained(ROOT / 'outputs/pretrained/dinov3_vitb16', local_files_only=True).eval()
    mean = torch.tensor([.485, .456, .406])[:, None, None]; std = torch.tensor([.229, .224, .225])[:, None, None]

    def feats_from_arrays(arr):
        out = []
        with torch.inference_mode():
            for i in range(0, len(arr), 32):
                x = (torch.from_numpy(np.array(arr[i:i + 32])).permute(0, 3, 1, 2).float() / 255 - mean) / std
                out.append(torch.nn.functional.normalize(m(pixel_values=x).pooler_output, dim=1).numpy())
                if i % 640 == 0:
                    print('feat', i, len(arr), flush=True)
        return np.concatenate(out)

    def load(paths):
        return np.stack([np.array(Image.open(p).convert('RGB').resize((384, 384), Image.Resampling.BILINEAR)) for p in paths])
    items = json.loads((P4 / 'items.json').read_text())
    T = feats_from_arrays(load([it['file'] for it in items]))
    cache = P4 / 'local_feats.npy'
    if cache.exists():
        Lf = np.load(cache)
    else:
        parts = [np.load(ROOT / 'outputs/strong_expert_training_v1/pixels/dino.npy', mmap_mode='r'),
                 np.load(ROOT / 'outputs/mainline_v2/stage3/pixels_inat_train.npy', mmap_mode='r')]
        from scripts.mainline_v2_final import test_sets
        ts = test_sets()
        extra = [p for _, p, _ in ts['inat']] + [p for _, p, _ in ts['iari']]
        Lf = np.concatenate([feats_from_arrays(a) for a in parts] + [feats_from_arrays(load(extra[i:i + 256])) for i in range(0, len(extra), 256)])
        np.save(cache, Lf)
    sim = (T @ Lf.T).max(1)
    keep, seen = [], set()
    for it, s in zip(items, sim):
        it['max_local_cos'] = float(s)
        key = (it['user'], it['observed'], it['label'])
        it['keep'] = bool(s < .90 and key not in seen)
        seen.add(key)
    (P4 / 'items.json').write_text(json.dumps(items, indent=0))
    print('kept', sum(i['keep'] for i in items), 'of', len(items), 'dup>=0.90', int((sim >= .90).sum()),
          {l: sum(i['keep'] and i['label'] == l for i in items) for l in {i['label'] for i in items}})


if __name__ == '__main__':
    main()
