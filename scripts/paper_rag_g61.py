"""Retrieval-augmented gpt-6.1-sol diagnosis (RAG-VLM), protocol section 14.

For each query image: retrieve labelled reference cases from the query fold's TRAINING sources only (pretrained DINOv3 CLS,
near-duplicates >= 0.90 excluded), then ask gpt-6.1-sol (WSL Codex) to diagnose the query by comparison.
Usage: python scripts/paper_rag_g61.py pilot | full
Shared resources used: Python = <DATA_ROOT>/.venv; features = outputs/paper_hrme/p4_inat/local_feats.npy (pretrained DINOv3-B CLS
of dev pool + 1405 additions + iNat-train/sealed + IARI, built by scripts/paper_p4_dedup.py).
"""
import csv
import hashlib
import io
import json
import random
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import paper_p2_codex_g61 as P  # noqa: E402  (schema, flags, WSL)

S1 = ROOT / 'outputs/mainline_v2/stage1'
OUT = ROOT / 'outputs/paper_hrme/rag'
RT = '<WSL_HOME>/hrme_rag'
LABELS = ('healthy', 'leaf_rust', 'powdery_mildew', 'septoria', 'stem_rust', 'yellow_rust')
K = {'leaf_rust': 2, 'stem_rust': 2, 'yellow_rust': 2, 'healthy': 1, 'powdery_mildew': 1, 'septoria': 1}
NAME = {'healthy': 'healthy', 'leaf_rust': 'leaf_rust (brown rust)', 'stem_rust': 'stem_rust (black rust)',
        'yellow_rust': 'yellow_rust (stripe rust)', 'powdery_mildew': 'powdery_mildew', 'septoria': 'septoria'}
PREFIX = """REFERENCE CASES: Before the photo to diagnose, you are shown {n} labelled reference photos of wheat from other collections,
retrieved because they look similar. In order, their confirmed labels are:
{lst}
Use them as a visual atlas: compare the query's lesion colour, shape, arrangement and organ with each reference class.
References may come from different cameras/sites than the query; judge by disease symptoms, not by background or image style.
The query photo follows the references (first its overview, then optional zoomed tiles).

"""
RUNNER = r'''set -e
h=$1; d=/tmp/hrme_rag/$h; o=/tmp/hrme_rag_out; mkdir -p $d $o; tar -x -C $d
cp @@RT@@/schema.json $d/schema.json; cd $d
imgs=""; for f in $(ls $d | grep '\.jpg$' | sort); do imgs="$imgs --image $d/$f"; done
timeout 900 <WSL_HOME>/.local/bin/codex exec --ephemeral --ignore-rules --skip-git-repo-check -m gpt-6.1-sol \
  -c 'model_reasoning_effort="medium"' -C $d $imgs --output-schema $d/schema.json --output-last-message $o/$h.txt \
  @@FLAGS@@ "$(cat $d/prompt.txt)" < /dev/null > $o/$h.log 2>&1 || true
cat $o/$h.txt 2>/dev/null || true; rm -rf $d $o/$h.txt $o/$h.log
'''.replace('@@RT@@', RT).replace('@@FLAGS@@', P.FLAGS)


def install() -> None:
    for name, data in (('run.sh', RUNNER), ('schema.json', json.dumps(P.SCHEMA))):
        subprocess.run(P.WSL + ['bash', '-c', f'mkdir -p {RT} && cat > {RT}/{name}'], input=data.encode(), check=True, timeout=60)


def jpeg(path, side=640) -> bytes:
    im = ImageOps.exif_transpose(Image.open(path)).convert('RGB'); im.thumbnail((side, side))
    b = io.BytesIO(); im.save(b, 'JPEG', quality=88); return b.getvalue()


def pools():
    """Rows aligned with local_feats.npy: dino.json ids (dev pool + additions) then iNat-train."""
    rows = {}
    for r in csv.DictReader(open(ROOT / 'outputs/stage0/split_candidate_v1/stage0_candidate_manifest.csv', encoding='utf-8-sig')):
        rows[r['record_id']] = (r['absolute_path'], r['canonical_class'])
    for r in csv.DictReader(open(ROOT / 'outputs/strong_expert_data_v2/eligible_training.csv', encoding='utf-8')):
        rows[r['record_id']] = (r['absolute_path'], r['canonical_class'])
    ids = json.loads((ROOT / 'outputs/strong_expert_training_v1/pixels/dino.json').read_text())['ids']
    inat = {('inat_' + r['file']): (str(ROOT / 'outputs/mainline_v2/inat/raw' / r['file']), r['label'])
            for r in csv.DictReader(open(ROOT / 'outputs/mainline_v2/inat/inat_manifest.csv', encoding='utf-8')) if r['partition'] == 'inat_train'}
    iids = json.loads((ROOT / 'outputs/mainline_v2/stage3/pixels_inat_train.json').read_text())['ids']
    meta = [(rid,) + rows[rid] for rid in ids] + [(i,) + inat[i] for i in iids]
    F = np.load(ROOT / 'outputs/paper_hrme/p4_inat/local_feats.npy')[:len(meta)]
    return meta, F, {rid: k for k, (rid, _, _) in enumerate(meta)}


def references(q_feat, allowed_idx, meta, F):
    sims = F[allowed_idx] @ q_feat
    order = np.argsort(-sims)
    refs, need = [], dict(K)
    for o in order:
        if sims[o] >= .90:
            continue
        i = allowed_idx[o]; lab = meta[i][2]
        if ';' in lab or lab not in need or need[lab] == 0:
            continue
        refs.append((meta[i][1], lab, float(sims[o]))); need[lab] -= 1
        if not any(need.values()):
            break
    order_lab = {l: n for n, l in enumerate(LABELS)}
    return sorted(refs, key=lambda x: (order_lab[x[1]], -x[2]))


def call(query_hashes, refs):
    lst = '\n'.join(f'  reference {i + 1}: {NAME[l]}' for i, (_, l, _) in enumerate(refs))
    prompt = PREFIX.format(n=len(refs), lst=lst) + P.PROMPT
    buf = io.BytesIO(); blobs = []
    with tarfile.open(fileobj=buf, mode='w') as t:
        def add(name, data):
            ti = tarfile.TarInfo(name); ti.size = len(data); t.addfile(ti, io.BytesIO(data)); blobs.append(data)
        for i, (p, _, _) in enumerate(refs):
            add(f'a{i:02d}_ref.jpg', jpeg(p))
        for i, h in enumerate(query_hashes):
            add(f'q{i:02d}_query.jpg', (ROOT / 'outputs/mainline_v2/images' / f'{h}.jpg').read_bytes())
        add('prompt.txt', prompt.encode())
    key = hashlib.sha256(b''.join(blobs) + b'gpt-6.1-sol|medium|rag1').hexdigest()[:32]
    f = OUT / 'responses' / f'{key}.json'
    if f.exists():
        return json.loads(f.read_text(encoding='utf-8'))
    rec = dict(status='error')
    for attempt in range(2):
        out = subprocess.run(P.WSL + ['bash', f'{RT}/run.sh', key], input=buf.getvalue(), capture_output=True, timeout=1000).stdout.decode(errors='replace')
        try:
            p = json.loads(out[out.find('{'):out.rfind('}') + 1])
            if p.get('primary') in LABELS + ('uncertain',):
                rec = dict(status='ok', parsed=p, refs=[(Path(a).name, b, round(c, 3)) for a, b, c in refs]); break
        except Exception:
            rec = dict(status='format_error', text=out[:300])
    if rec['status'] == 'ok':
        f.write_text(json.dumps(rec, ensure_ascii=False), encoding='utf-8')
    return rec


def main() -> None:
    mode = sys.argv[1]
    (OUT / 'responses').mkdir(parents=True, exist_ok=True); install()
    per = json.loads((S1 / 'per_record.json').read_text()); ev = json.loads((S1 / 'sets.json').read_text())['eval']
    prepared = json.loads((S1 / 'prepared.json').read_text())
    plan = json.loads((ROOT / 'outputs/system_lazy_cascade_v1/plan.json').read_text())
    cells = {c['fold']: c for c in plan['cells'] if c['setting'] == 'source_held'}
    meta, F, pos = pools()
    extra = [r['record_id'] for r in csv.DictReader(open(ROOT / 'outputs/strong_expert_data_v2/eligible_training.csv', encoding='utf-8'))]
    inat_ids = [m[0] for m in meta if m[0].startswith('inat_')]
    allowed = {f: np.array(sorted({pos[r] for r in cells[f]['fit'] + extra + inat_ids if r in pos})) for f in range(3)}
    if mode == 'ets':
        return run_ets(meta, F, pos)
    if mode == 'pilot':
        rng = random.Random(20261008); q = []
        for lab, n in (('leaf_rust', 20), ('stem_rust', 20), ('yellow_rust', 20), ('healthy', 10), ('powdery_mildew', 10), ('septoria', 10)):
            c = sorted(r for r in ev if per[r]['truth'] == [lab]); rng.shuffle(c); q += c[:n]
    else:
        q = list(ev)
    print('queries', len(q), flush=True)
    done = [0]

    def work(r):
        refs = references(F[pos[r]], allowed[per[r]['fold']], meta, F)
        rec = call(prepared[r], refs); done[0] += 1
        if done[0] % 10 == 0:
            print('done', done[0], flush=True)
        return r, rec
    with ThreadPoolExecutor(5) as ex:
        res = dict(ex.map(work, q))
    out = {r: (dict(status='ok', primary=v['parsed']['primary'], conf=v['parsed']['confidence'], diag=v['parsed']['diagnosis'])
               if v['status'] == 'ok' else dict(status=v['status'])) for r, v in res.items()}
    (OUT / f'rag_{mode}.json').write_text(json.dumps(out, indent=0))
    g = json.loads((ROOT / 'outputs/paper_hrme/p2_g61/g61_per_record.json').read_text())
    rag = sum(out[r].get('primary') in per[r]['truth'] for r in q); plain = sum(g[r].get('primary') in per[r]['truth'] for r in q)
    print('status', {s: sum(v['status'] == s for v in out.values()) for s in {v['status'] for v in out.values()}})
    print(f'{mode}: RAG top1 {rag}/{len(q)}  plain G61 {plain}/{len(q)}')
    for lab in LABELS:
        rr = [r for r in q if per[r]['truth'] == [lab]]
        if rr:
            print(f'  {lab:15s} RAG {sum(out[r].get("primary") == lab for r in rr)}/{len(rr)}  plain {sum(g[r].get("primary") == lab for r in rr)}/{len(rr)}')


def ets_feats(items):
    import torch
    from PIL import Image
    sys.path.insert(0, str(ROOT / '.vendor/transformers_v2'))
    from transformers import AutoModel
    m = AutoModel.from_pretrained(ROOT / 'outputs/pretrained/dinov3_vitb16', local_files_only=True).eval().cuda()
    out = []
    with torch.inference_mode():
        for i in range(0, len(items), 16):
            x = np.stack([np.array(Image.open(it['file']).convert('RGB').resize((384, 384), Image.Resampling.BILINEAR)) for it in items[i:i + 16]])
            x = (torch.from_numpy(x).permute(0, 3, 1, 2).float() / 255 - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]
            out.append(torch.nn.functional.normalize(m(pixel_values=x.cuda()).pooler_output.float(), dim=1).cpu().numpy())
    return np.concatenate(out)


def run_ets(meta, F, pos):
    """ETS queries; reference pool = all local training pools (single-label rows); predictions only, no metrics."""
    import os
    E = ROOT / os.environ.get('ETS_DIR', 'outputs/paper_hrme/ets')
    items = json.loads((E / 'items.json').read_text()); prep = json.loads((E / 'prepared.json').read_text())
    Q = ets_feats(items)
    allowed = np.arange(len(meta))
    done = [0]

    def work(k):
        it = items[k]
        rec = call(prep[it['file']], references(Q[k], allowed, meta, F)); done[0] += 1
        if done[0] % 10 == 0:
            print('done', done[0], flush=True)
        return it['file'], rec
    with ThreadPoolExecutor(5) as ex:
        res = dict(ex.map(work, range(len(items))))
    out = {k: (dict(status='ok', primary=v['parsed']['primary'], conf=v['parsed']['confidence'], diag=v['parsed']['diagnosis'])
               if v['status'] == 'ok' else dict(status=v['status'])) for k, v in res.items()}
    (E / 'rag.json').write_text(json.dumps(out, indent=0))
    print('status', {s: sum(v['status'] == s for v in out.values()) for s in {v['status'] for v in out.values()}})


if __name__ == '__main__':
    main()
