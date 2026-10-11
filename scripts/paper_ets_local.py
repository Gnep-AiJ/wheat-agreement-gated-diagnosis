"""Local expert L (9 DINOv3 iNat models, source-held folds) probabilities on the ETS images. Saves only predictions (no metrics)."""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    import torch
    from PIL import Image
    import scripts.mainline_v2_train_inat  # noqa: F401  (transformers import order)
    from scripts.strong_expert_train import Expert
    E = ROOT / os.environ.get('ETS_DIR', 'outputs/paper_hrme/ets')
    items = json.loads((E / 'items.json').read_text())
    x = np.stack([np.array(Image.open(it['file']).convert('RGB').resize((384, 384), Image.Resampling.BILINEAR)) for it in items])
    x = (torch.from_numpy(x).permute(0, 3, 1, 2).float() / 255 - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]
    out = []
    for s in (42, 43, 44):
        for f in range(3):
            m = Expert('dino'); m.load_state_dict(torch.load(ROOT / f'outputs/mainline_v2/stage3/dino_inat_seed{s}_fold{f}/model.pt', map_location='cpu', weights_only=True))
            m = m.cuda().eval()
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                out.append(torch.cat([m(x[i:i + 16].cuda()).float().sigmoid().cpu() for i in range(0, len(x), 16)]).numpy())
            print('model', s, f, flush=True)
    np.save(E / 'L_dino.npy', np.mean(out, 0))


if __name__ == '__main__':
    main()
