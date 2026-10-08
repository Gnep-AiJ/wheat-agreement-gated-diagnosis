"""One frozen native-quadrant comparison; development only, resumable cache."""
import csv
import hashlib
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / '.vendor/transformers'))
from src.limited_adaptation import classification_metrics
from src.native_tiles import quadrants
from transformers import AutoImageProcessor, AutoModel

OUT = ROOT / 'outputs/system_native_tiles_v1'
LABELS = ('healthy', 'leaf_rust', 'powdery_mildew', 'septoria', 'stem_rust', 'yellow_rust')


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_suffix('.part')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(temp, path)


def main() -> None:
    manifest = ROOT / 'outputs/stage0/split_candidate_v1/stage0_candidate_manifest.csv'
    plan_path = ROOT / 'outputs/system_source_split_v1/plan.json'
    assert sha(manifest) == 'ec07f69eee6e5315fbaa7530a54869c7e08d149435afe92deb2ac6debbc7f236'
    with manifest.open(encoding='utf-8-sig', newline='') as f:
        rows = {r['record_id']: r for r in csv.DictReader(f) if r['partition'] == 'development_oof'}
    assert len(rows) == 1418
    plan = json.loads(plan_path.read_text())
    weights = ROOT / 'outputs/pretrained/dinov3_vitb16'
    assert sha(weights / 'model.safetensors') == '9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b'
    files = [manifest, plan_path, Path(__file__).resolve(), ROOT / 'src/native_tiles.py', ROOT / 'src/limited_adaptation.py',
             ROOT / 'research/NATIVE_TILE_PROTOCOL_2026-09-26.md',
             ROOT / '.vendor/transformers/transformers/models/dinov3_vit/modeling_dinov3_vit.py']
    files += [weights / name for name in ('config.json', 'preprocessor_config.json', 'model.safetensors')]
    cache_files = sorted((ROOT / 'outputs/system_detail_v1/features_448').glob('batch_*.npz'))
    files += cache_files
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in files}
    OUT.mkdir(exist_ok=True)
    freeze = OUT / 'input_hashes.json'
    if freeze.exists():
        assert json.loads(freeze.read_text()) == hashes
    else:
        atomic_json(freeze, hashes)
    assert not (OUT / 'results.json').exists(), 'Completed: do not rerun'
    global_features = {}
    for path in cache_files:
        with np.load(path) as data:
            global_features.update(zip(data['record_ids'].astype(str), data['cls'], strict=True))
    assert set(global_features) == set(rows)
    cache = OUT / 'features'
    cache.mkdir(exist_ok=True)
    processor = AutoImageProcessor.from_pretrained(weights, local_files_only=True)
    model, loading = AutoModel.from_pretrained(weights, local_files_only=True, output_loading_info=True)
    assert not any(loading.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'))
    model.eval().requires_grad_(False).to('cuda')
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    native = {}
    for i, (rid, row) in enumerate(rows.items()):
        target = cache / f'{rid}.npz'
        if not target.exists():
            path = Path(row['absolute_path'])
            assert sha(path) == row['sha256']
            with Image.open(path) as im:
                tiles = quadrants(im.convert('RGB'))
            parts = []
            for start in (0, 2):
                pixels = processor(images=tiles[start:start+2], size={'height':448, 'width':448}, return_tensors='pt')['pixel_values'].to('cuda')
                with torch.inference_mode():
                    parts.append(model(pixel_values=pixels).last_hidden_state[:, 0].cpu().numpy())
            tile_features = np.concatenate(parts)
            with target.with_suffix('.part').open('wb') as f:
                np.savez_compressed(f, record_id=rid, cls=tile_features)
            os.replace(target.with_suffix('.part'), target)
        with np.load(target) as data:
            assert data['record_id'].item() == rid
            assert data['cls'].shape == (4, 768) and np.isfinite(data['cls']).all()
            native[rid] = data['cls'].mean(0)
        if i % 100 == 0 or i+1 == len(rows):
            atomic_json(OUT / 'progress.json', dict(completed=i+1, total=len(rows), elapsed_seconds=time.perf_counter()-started))
            print(f'quadrants {i+1}/{len(rows)}', flush=True)
        if time.perf_counter()-started > 3600:
            raise RuntimeError('60-minute extraction limit; retain atomic checkpoints')
    atomic_json(OUT / 'runtime.json', dict(extraction_seconds_this_process=time.perf_counter()-started,
        peak_cuda_mib=torch.cuda.max_memory_allocated()/1048576, extra_backbone_views_per_image=4))
    del model
    torch.cuda.empty_cache()
    truth = {k: [label in r['canonical_class'].split(';') for label in LABELS] for k,r in rows.items()}
    results, all_probabilities, ordered_ids = {}, {}, None
    for arm in ('baseline', 'duplicate_control', 'native_tiles'):
        vectors = {k: v if arm == 'baseline' else np.concatenate((v, v if arm == 'duplicate_control' else native[k]))
                   for k,v in global_features.items()}
        ids, arrays, folds = [], [], []
        for outer in plan['outer']:
            fit, predict, fold = outer['fit_ids'], outer['predict_ids'], outer['fold']
            assert not {rows[k]['source_group'] for k in fit} & {rows[k]['source_group'] for k in predict}
            x, z = (np.stack([vectors[k] for k in group]) for group in (fit,predict))
            y, target = (np.array([truth[k] for k in group]) for group in (fit,predict))
            scaler = StandardScaler().fit(x)
            x, z = scaler.transform(x), scaler.transform(z)
            p = np.zeros((len(predict), 6))
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                for c in range(6):
                    head = LogisticRegression(C=1, max_iter=2000, random_state=42).fit(x,y[:,c])
                    p[:,c] = head.predict_proba(z)[:,1]
            assert np.isfinite(p).all()
            if arm == 'baseline':
                with np.load(ROOT / f'outputs/system_detail_v1/448_cls_fold{fold}.npz') as old:
                    assert old['record_ids'].astype(str).tolist() == predict
                    np.testing.assert_allclose(p,old['probabilities'],atol=1e-8,rtol=1e-6)
            np.savez_compressed(OUT / f'{arm}_fold{fold}.npz', record_ids=predict, probabilities=p)
            ids.extend(predict)
            arrays.append(p)
            folds.append(classification_metrics(p,target))
        assert len(ids) == len(set(ids)) == 1418
        if ordered_ids is None:
            ordered_ids = ids
        assert ordered_ids == ids
        all_probabilities[arm] = np.concatenate(arrays)
        results[arm] = classification_metrics(all_probabilities[arm],np.array([truth[k] for k in ids]))
        results[arm]['folds'] = folds
        print(arm, results[arm]['accuracy'], flush=True)
    target = np.array([truth[k] for k in ordered_ids])
    a = target[np.arange(1418),all_probabilities['native_tiles'].argmax(1)].astype(int)
    comparisons = {}
    sources = sorted({r['source_group'] for r in rows.values()})
    masks = [np.array([rows[k]['source_group'] == source for k in ordered_ids]) for source in sources]
    sizes = np.array([m.sum() for m in masks])
    draws = np.random.default_rng(20260926).integers(len(sources),size=(10000,len(sources)))
    for reference in ('baseline','duplicate_control'):
        b = target[np.arange(1418),all_probabilities[reference].argmax(1)].astype(int)
        deltas = np.array([(a[m]-b[m]).sum() for m in masks])
        interval = np.quantile(deltas[draws].sum(1)/sizes[draws].sum(1),[.025,.975]).tolist()
        gain = float((a-b).mean())
        folds = [x['accuracy']-y['accuracy'] for x,y in zip(results['native_tiles']['folds'],results[reference]['folds'],strict=True)]
        classes = [x-y for x,y in zip(results['native_tiles']['named_top1_recall'],results[reference]['named_top1_recall'],strict=True)]
        comparisons[reference] = dict(gain=gain,fold_gains=folds,named_recall_gains=classes,source_bootstrap_interval=interval,
            fixes=int(((a==1)&(b==0)).sum()),harms=int(((a==0)&(b==1)).sum()),
            gate=bool(gain>=.05 and min(folds)>=-.02 and min(classes)>=-.02 and (reference!='duplicate_control' or interval[0]>0)))
    atomic_json(OUT / 'results.json', dict(scope='exploratory source-held development; no final test', labels=LABELS,
        arms=results, comparisons=comparisons, exploratory_gate=all(c['gate'] for c in comparisons.values())))
    print('Completed; gate=', all(c['gate'] for c in comparisons.values()), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        main()
