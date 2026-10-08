"""Frozen simple full-backbone, data-expansion factorial. Development only."""
import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import timm
from PIL import Image
from safetensors.torch import load_file

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'.vendor/transformers'))
from transformers import AutoModel
from src.limited_adaptation import classification_metrics
from scripts.system_native_tile_experiment import sha,atomic_json

OUT=ROOT/'outputs/strong_expert_training_v1'
LABELS=('healthy','leaf_rust','powdery_mildew','septoria','stem_rust','yellow_rust')

def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))

def epoch_indices(n,epoch,fold,draws=2005):
    rng=np.random.default_rng(42000+fold*100+epoch)
    return np.concatenate([rng.permutation(n) for _ in range(math.ceil(draws/n))])[:draws]

class Expert(torch.nn.Module):
    def __init__(self,name):
        super().__init__()
        self.name=name
        if name=='convnext':
            self.backbone=timm.create_model('convnext_base',pretrained=False,num_classes=1000)
            self.backbone.load_state_dict(load_file(str(ROOT/'outputs/pretrained/convnext_base_384/model.safetensors')),strict=True)
            self.backbone.reset_classifier(0)
            self.backbone.set_grad_checkpointing(True)
            dim=1024
        else:
            self.backbone=AutoModel.from_pretrained(ROOT/'outputs/pretrained/dinov3_vitb16',local_files_only=True)
            self.backbone.gradient_checkpointing_enable()
            dim=768
        self.head=torch.nn.Linear(dim,6)
    def forward(self,x):
        f=self.backbone(x) if self.name=='convnext' else self.backbone(pixel_values=x).pooler_output
        return self.head(f)

def pixels(rows,name):
    folder=OUT/'pixels'
    folder.mkdir(exist_ok=True)
    dest=folder/f'{name}.npy'
    ids=list(rows)
    metadata=folder/f'{name}.json'
    if dest.exists():
        assert json.loads(metadata.read_text())['ids']==ids
        return np.load(dest,mmap_mode='r')
    temp=dest.with_suffix('.part.npy')
    data=np.lib.format.open_memmap(temp,mode='w+',dtype=np.uint8,shape=(len(ids),384,384,3))
    method=Image.Resampling.BICUBIC if name=='convnext' else Image.Resampling.BILINEAR
    for i,rid in enumerate(ids):
        r=rows[rid]
        path=Path(r['absolute_path'])
        assert sha(path)==r['sha256']
        with Image.open(path) as image:
            data[i]=np.array(image.convert('RGB').resize((384,384),method))
        if i%500==0:
            print('pixels',name,i,len(ids),flush=True)
    data.flush()
    del data
    os.replace(temp,dest)
    atomic_json(metadata,dict(ids=ids,sha256=sha(dest)))
    return np.load(dest,mmap_mode='r')

def load_x(data,indices,training=False):
    x=torch.from_numpy(np.array(data[indices])).cuda().permute(0,3,1,2).float()/255.
    if training:
        h=torch.rand(len(x),device='cuda')<.5
        v=torch.rand(len(x),device='cuda')<.5
        x[h]=x[h].flip(-1)
        x[v]=x[v].flip(-2)
    mean=x.new_tensor([.485,.456,.406])[None,:,None,None]
    std=x.new_tensor([.229,.224,.225])[None,:,None,None]
    return (x-mean)/std

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=True)
    if args.preflight:
        result=[]
        for name in ('convnext','dino'):
            torch.manual_seed(42)
            model=Expert(name).cuda().train()
            optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5)
            scaler=torch.amp.GradScaler('cuda')
            torch.cuda.reset_peak_memory_stats()
            times=[]
            for _ in range(4):
                start=time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16):
                    loss=model(torch.rand(8,3,384,384,device='cuda')).square().mean()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                torch.cuda.synchronize()
                times.append(time.monotonic()-start)
            result.append(dict(name=name,microbatch=8,times=times,peak_mib=torch.cuda.max_memory_allocated()/2**20))
            print(result[-1],flush=True)
            del model,optimizer,scaler,loss
            torch.cuda.empty_cache()
        atomic_json(OUT/'preflight.json',dict(results=result))
        return
    assert (OUT/'preflight.json').exists()
    manifest=ROOT/'outputs/stage0/split_candidate_v1/stage0_candidate_manifest.csv'
    additions=ROOT/'outputs/strong_expert_data_v2/eligible_training.csv'
    plan_path=ROOT/'outputs/system_lazy_cascade_v1/plan.json'
    plan=json.loads(plan_path.read_text())
    rows={r['record_id']:r for r in read(manifest) if r['partition']=='development_oof'}
    extra=read(additions)
    assert len(rows)==1418 and len(extra)==1405
    assert not set(rows)&{r['record_id'] for r in extra}
    rows.update({r['record_id']:r for r in extra})
    protected={r['duplicate_component'] for r in read(manifest)}
    assert not protected&{r['duplicate_component'] for r in extra}
    files=[manifest,additions,plan_path,Path(__file__).resolve(),ROOT/'research/STRONG_EXPERT_TRAINING_PROTOCOL_2026-09-27.md',
           ROOT/'src/limited_adaptation.py',ROOT/'outputs/pretrained/convnext_base_384/model.safetensors',
           ROOT/'outputs/pretrained/dinov3_vitb16/model.safetensors']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in files}
    assert hashes[str(files[-2].relative_to(ROOT))]=='bc0cfafc7d7755e8271f51343c4374725bf6409c0434e64028a863c4aca7d9fe'
    assert hashes[str(files[-1].relative_to(ROOT))]=='9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b'
    freeze=OUT/'input_hashes.json'
    if freeze.exists():
        assert json.loads(freeze.read_text())==hashes
    else:
        atomic_json(freeze,hashes)
    index={rid:i for i,rid in enumerate(rows)}
    def truth(ids):
        return np.array([[label in rows[rid]['canonical_class'].split(';') for label in LABELS] for rid in ids],dtype=np.float32)
    for name in ('convnext','dino'):
        data=pixels(rows,name)
        for cell in [c for c in plan['cells'] if c['setting']=='source_held']:
            for arm in ('original','expanded'):
                folder=OUT/f"{name}_{arm}_fold{cell['fold']}"
                folder.mkdir(exist_ok=True)
                if (folder/'completion.json').exists():
                    continue
                ids=cell['fit']+([r['record_id'] for r in extra] if arm=='expanded' else [])
                assert len(ids)==(600 if arm=='original' else 2005)
                assert not set(ids)&set(cell['calibration']+cell['threshold']+cell['evaluation'])
                torch.manual_seed(42)
                torch.cuda.manual_seed_all(42)
                model=Expert(name).cuda()
                optimizer=torch.optim.AdamW([dict(params=model.backbone.parameters(),lr=5e-5),dict(params=model.head.parameters(),lr=5e-4)],weight_decay=.01)
                scaler=torch.amp.GradScaler('cuda')
                checkpoint=folder/'latest.pt'
                history=[]
                first=0
                if checkpoint.exists():
                    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
                    assert state['freeze_sha256']==sha(freeze)
                    model.load_state_dict(state['model'])
                    optimizer.load_state_dict(state['optimizer'])
                    scaler.load_state_dict(state['scaler'])
                    first,history=state['epoch'],state['history']
                    del state
                started=time.monotonic()
                torch.cuda.reset_peak_memory_stats()
                y=truth(ids)
                for epoch in range(first,12):
                    torch.manual_seed(42000+cell['fold']*100+epoch)
                    torch.cuda.manual_seed_all(42000+cell['fold']*100+epoch)
                    order=epoch_indices(len(ids),epoch,cell['fold'])
                    model.train()
                    losses=[]
                    for start in range(0,2005,32):
                        selected=order[start:start+32]
                        progress=epoch+start/2005
                        factor=min(1.,progress+.05) if progress<1 else .1+.9*(1+math.cos(math.pi*(progress-1)/11))/2
                        for group,lr in zip(optimizer.param_groups,(5e-5,5e-4)):
                            group['lr']=lr*factor
                        optimizer.zero_grad(set_to_none=True)
                        for offset in range(0,len(selected),8):
                            subset=selected[offset:offset+8]
                            x=load_x(data,[index[ids[i]] for i in subset],True)
                            target=torch.from_numpy(y[subset]).cuda()
                            with torch.autocast('cuda',dtype=torch.float16):
                                logits=model(x)
                                loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,target,reduction='sum')/(len(selected)*6)
                            assert torch.isfinite(loss)
                            scaler.scale(loss).backward()
                            losses.append(float(loss.detach())*len(selected)/len(subset))
                        scaler.unscale_(optimizer)
                        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                        before=scaler.get_scale()
                        scaler.step(optimizer)
                        scaler.update()
                        if not torch.isfinite(norm):
                            assert scaler.get_scale()<before,'Nonfinite gradients must skip optimizer update'
                    history.append(dict(epoch=epoch+1,bce=float(np.mean(losses)),seconds=time.monotonic()-started,amp_scale=scaler.get_scale()))
                    state=dict(epoch=epoch+1,history=history,model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                        optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),freeze_sha256=sha(freeze))
                    torch.save(state,checkpoint.with_suffix('.part'))
                    os.replace(checkpoint.with_suffix('.part'),checkpoint)
                    del state
                    atomic_json(folder/'progress.json',dict(history=history))
                    print(folder.name,history[-1],flush=True)
                model.eval()
                metrics={}
                for partition in ('calibration','threshold','evaluation'):
                    eval_ids=cell[partition]
                    values=[]
                    with torch.inference_mode():
                        for start in range(0,len(eval_ids),8):
                            x=load_x(data,[index[rid] for rid in eval_ids[start:start+8]])
                            with torch.autocast('cuda',dtype=torch.float16):
                                values.append(model(x).float().sigmoid().cpu().numpy())
                    p=np.concatenate(values)
                    np.savez_compressed(folder/f'{partition}.npz',record_ids=eval_ids,probabilities=p)
                    metrics[partition]=classification_metrics(p,truth(eval_ids))
                atomic_json(folder/'completion.json',dict(name=name,arm=arm,fold=cell['fold'],fit_count=len(ids),history=history,metrics=metrics,
                    checkpoint_sha256=sha(checkpoint),peak_mib=torch.cuda.max_memory_allocated()/2**20))
                print('COMPLETE',folder.name,metrics['evaluation'],flush=True)
                del model,optimizer,scaler,x,loss,logits
                torch.cuda.empty_cache()
    print('ALL 12 CELLS COMPLETE',flush=True)

if __name__=='__main__':
    main()
