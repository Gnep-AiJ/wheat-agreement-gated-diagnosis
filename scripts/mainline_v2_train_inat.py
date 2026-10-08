"""Mainline V2 stage 3: expanded-DINO recipe, original vs original+iNat-train (matched 2005 draws/epoch).

Usage: python -m scripts.mainline_v2_train_inat --arm orig|inat|inat25|inat50 --seed 44
Only source_held cells are trained. The sealed iNat final-test partition is asserted absent.
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# The old .vendor/transformers copy has an OS deny ACL; import a fresh PyPI copy first so it is cached in sys.modules.
sys.path.insert(0, str(ROOT / '.vendor/transformers_v2'))
from transformers import AutoModel  # noqa: E402,F401  (resolve lazy submodules before the old path is inserted)
import tokenizers  # noqa: E402,F401
from scripts.strong_expert_train import Expert, load_x, read, LABELS
from scripts.system_native_tile_experiment import atomic_json, sha
from src.limited_adaptation import classification_metrics

OUT = ROOT / 'outputs/mainline_v2/stage3'
INAT = ROOT / 'outputs/mainline_v2/inat'


def order_for(n, epoch, fold, seed):
    rng = np.random.default_rng(seed * 1000 + fold * 100 + epoch)
    return np.concatenate([rng.permutation(n) for _ in range(math.ceil(2005 / n))])[:2005]


def inat_rows(arm: str) -> dict:
    with (INAT / 'inat_manifest.csv').open(encoding='utf-8') as f:
        rows = [r for r in csv.DictReader(f)]
    assert all(r['partition'] in ('inat_train', 'inat_final_test_SEALED') for r in rows)
    rows = [r for r in rows if r['partition'] == 'inat_train']
    if arm in ('inat25', 'inat50'):
        obs = sorted({r['observer'] for r in rows})
        frac = 0.25 if arm == 'inat25' else 0.50
        keep = set(np.random.default_rng(20261001).permutation(obs)[:int(round(frac * len(obs)))])
        rows = [r for r in rows if r['observer'] in keep]
    return {'inat_' + r['file']: dict(record_id='inat_' + r['file'], canonical_class=r['label'], path=str(INAT / 'raw' / r['file'])) for r in rows}


def inat_pixels(rows: dict) -> tuple:
    ids = sorted(rows)
    dest = OUT / 'pixels_inat_train.npy'
    meta = dest.with_suffix('.json')
    if dest.exists() and json.loads(meta.read_text())['ids'] == ids:
        return np.load(dest, mmap_mode='r'), {k: i for i, k in enumerate(ids)}
    OUT.mkdir(parents=True, exist_ok=True)
    data = np.zeros((len(ids), 384, 384, 3), np.uint8)
    for i, rid in enumerate(ids):
        with Image.open(rows[rid]['path']) as im:
            data[i] = np.array(im.convert('RGB').resize((384, 384), Image.Resampling.BILINEAR))  # same as training recipe
    np.save(dest, data)
    atomic_json(meta, dict(ids=ids))
    return np.load(dest, mmap_mode='r'), {k: i for i, k in enumerate(ids)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True)
    ap.add_argument('--seed', type=int, required=True)
    a = ap.parse_args()
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = ROOT / 'outputs/stage0/split_candidate_v1/stage0_candidate_manifest.csv'
    rows = {r['record_id']: r for r in read(manifest) if r['partition'] == 'development_oof'}
    extra = read(ROOT / 'outputs/strong_expert_data_v2/eligible_training.csv')
    rows.update({r['record_id']: r for r in extra})
    cache = ROOT / 'outputs/strong_expert_training_v1/pixels/dino.npy'
    meta = json.loads(cache.with_suffix('.json').read_text())
    assert meta['ids'] == list(rows)
    base = np.load(cache, mmap_mode='r')
    index = {rid: i for i, rid in enumerate(rows)}
    add = inat_rows(a.arm) if a.arm != 'orig' else {}
    if add:
        idata, iindex = inat_pixels(inat_rows('inat'))  # one cache for the full inat_train pool
        rows.update(add)
    labels = {rid: set(r['canonical_class'].split(';')) for rid, r in rows.items()}

    def truth(ids):
        return np.array([[l in labels[rid] for l in LABELS] for rid in ids], dtype=np.float32)

    def batch(ids, training=False):
        xs = [load_x(idata if rid.startswith('inat_') else base, [iindex[rid] if rid.startswith('inat_') else index[rid]], training) for rid in ids]
        return torch.cat(xs)

    plan = json.loads((ROOT / 'outputs/system_lazy_cascade_v1/plan.json').read_text())
    seed = a.seed
    for cell in [c for c in plan['cells'] if c['setting'] == 'source_held']:
        folder = OUT / f"dino_{a.arm}_seed{seed}_fold{cell['fold']}"
        folder.mkdir(exist_ok=True)
        if (folder / 'completion.json').exists():
            continue
        ids = cell['fit'] + [r['record_id'] for r in extra] + sorted(add)
        assert not set(ids) & set(cell['calibration'] + cell['threshold'] + cell['evaluation'])
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = Expert('dino').cuda()
        opt = torch.optim.AdamW([dict(params=model.backbone.parameters(), lr=5e-5), dict(params=model.head.parameters(), lr=5e-4)], weight_decay=.01)
        scaler = torch.amp.GradScaler('cuda')
        y = truth(ids)
        history, started = [], time.monotonic()
        for epoch in range(12):
            torch.manual_seed(seed * 1000 + cell['fold'] * 100 + epoch)
            torch.cuda.manual_seed_all(seed * 1000 + cell['fold'] * 100 + epoch)
            order = order_for(len(ids), epoch, cell['fold'], seed)
            model.train()
            losses = []
            for start in range(0, 2005, 32):
                sel = order[start:start + 32]
                progress = epoch + start / 2005
                factor = min(1., progress + .05) if progress < 1 else .1 + .9 * (1 + math.cos(math.pi * (progress - 1) / 11)) / 2
                for g, lr in zip(opt.param_groups, (5e-5, 5e-4)):
                    g['lr'] = lr * factor
                opt.zero_grad(set_to_none=True)
                for off in range(0, len(sel), 8):
                    sub = sel[off:off + 8]
                    x = batch([ids[i] for i in sub], True)
                    target = torch.from_numpy(y[sub]).cuda()
                    with torch.autocast('cuda', dtype=torch.float16):
                        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), target, reduction='sum') / (len(sel) * 6)
                    assert torch.isfinite(loss)
                    scaler.scale(loss).backward()
                    losses.append(float(loss.detach()) * len(sel) / len(sub))
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                scaler.step(opt)
                scaler.update()
            history.append(dict(epoch=epoch + 1, bce=float(np.mean(losses)), seconds=time.monotonic() - started))
            print(folder.name, history[-1], flush=True)
        model.eval()
        metrics = {}
        for part in ('calibration', 'threshold', 'evaluation'):
            eids = cell[part]
            vals = []
            with torch.inference_mode():
                for s in range(0, len(eids), 8):
                    with torch.autocast('cuda', dtype=torch.float16):
                        vals.append(model(batch(eids[s:s + 8])).float().sigmoid().cpu().numpy())
            p = np.concatenate(vals)
            np.savez_compressed(folder / f'{part}.npz', record_ids=eids, probabilities=p)
            metrics[part] = classification_metrics(p, truth(eids))
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, folder / 'model.pt')
        atomic_json(folder / 'completion.json', dict(arm=a.arm, seed=seed, fold=cell['fold'], fit_count=len(ids), n_inat=len(add),
                                                     history=history, metrics=metrics))
        print('COMPLETE', folder.name, metrics['evaluation'], flush=True)
        del model, opt, scaler
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
