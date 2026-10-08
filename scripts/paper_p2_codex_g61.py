"""Paper HRME P2: gpt-6.1-sol (WSL Codex) on EVAL450 with the frozen V2 prompt. Rules: research/PAPER_HRME_PROTOCOL_2026-10-07.md.

Same prepared images as V2 (overview + 2x2 tiles when large), staged into WSL under opaque hash names; tools disabled;
one fresh ephemeral session per image; cached per (images, prompt, model, effort).
"""
import hashlib
import io
import json
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.precall_codex_command import DISABLED_FEATURES  # noqa: E402

S1 = ROOT / 'outputs/mainline_v2/stage1'
IMG = ROOT / 'outputs/mainline_v2/images'
OUT = ROOT / 'outputs/paper_hrme/p2_g61'
PROMPT = (ROOT / 'research/MAINLINE_V2_PROMPT_2026-09-29.txt').read_text(encoding='utf-8')
assert hashlib.sha256((ROOT / 'research/MAINLINE_V2_PROMPT_2026-09-29.txt').read_bytes()).hexdigest().startswith('03612a75')
MODEL, EFFORT = 'gpt-6.1-sol', 'medium'
WSL = ['wsl', '-d', 'Ubuntu-24.04', '-u', 'codexblind', '--']
RT = '<WSL_HOME>/hrme_g61'
CLS = ['healthy', 'leaf_rust', 'powdery_mildew', 'septoria', 'stem_rust', 'yellow_rust']
SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'required': ['observations', 'image_type', 'assessability', 'symptoms', 'diagnosis', 'primary', 'confidence', 'summary'],
    'properties': {
        'observations': {'type': 'string'},
        'image_type': {'type': 'string', 'enum': ['leaf_closeup', 'stem', 'spike', 'whole_plant', 'canopy_distant', 'lab_or_specimen', 'non_wheat', 'unclear']},
        'assessability': {'type': 'string', 'enum': ['clear_lesions_visible', 'clear_no_lesions', 'partially_visible', 'cannot_judge']},
        'symptoms': {'type': 'object', 'additionalProperties': False,
                     'required': ['pustules_present', 'pustule_color', 'pustule_arrangement', 'white_powdery_growth', 'necrotic_blotches',
                                  'black_dots_in_lesions', 'affected_organ'],
                     'properties': {
                         'pustules_present': {'type': 'boolean'},
                         'pustule_color': {'type': 'string', 'enum': ['none', 'yellow', 'orange_brown', 'brick_red', 'dark_brown_black', 'other']},
                         'pustule_arrangement': {'type': 'string', 'enum': ['none', 'stripes_along_veins', 'scattered', 'elongated_on_stem_or_sheath', 'other']},
                         'white_powdery_growth': {'type': 'boolean'}, 'necrotic_blotches': {'type': 'boolean'},
                         'black_dots_in_lesions': {'type': 'boolean'},
                         'affected_organ': {'type': 'string', 'enum': ['none', 'leaf', 'stem', 'leaf_sheath', 'spike', 'multiple', 'unclear']}}},
        'diagnosis': {'type': 'array', 'items': {'type': 'string', 'enum': CLS + ['uncertain']}},
        'primary': {'type': 'string', 'enum': CLS + ['uncertain']},
        'confidence': {'type': 'integer'},
        'summary': {'type': 'string'}}}
FLAGS = ' '.join(f'--disable {f}' for f in DISABLED_FEATURES)
RUNNER = r'''set -e
h=$1; d=/tmp/hrme_g61/$h; o=/tmp/hrme_g61_out; mkdir -p $d $o; tar -x -C $d
cp @@RT@@/schema.json $d/schema.json; cd $d
imgs=""; for f in $(ls $d | grep '^img' | sort); do imgs="$imgs --image $d/$f"; done
timeout 600 <WSL_HOME>/.local/bin/codex exec --ephemeral --ignore-rules --skip-git-repo-check -m @@MODEL@@ \
  -c 'model_reasoning_effort="@@EFFORT@@"' -C $d $imgs --output-schema $d/schema.json --output-last-message $o/$h.txt \
  @@FLAGS@@ "$(cat @@RT@@/prompt.txt)" < /dev/null > $o/$h.log 2>&1 || true
cat $o/$h.txt 2>/dev/null || true; rm -rf $d $o/$h.txt $o/$h.log
'''.replace('@@RT@@', RT).replace('@@MODEL@@', MODEL).replace('@@EFFORT@@', EFFORT).replace('@@FLAGS@@', FLAGS)


def install() -> None:
    for name, data in (('run.sh', RUNNER), ('schema.json', json.dumps(SCHEMA)), ('prompt.txt', PROMPT)):
        subprocess.run(WSL + ['bash', '-c', f'mkdir -p {RT} && cat > {RT}/{name}'], input=data.encode(), check=True, timeout=60)


def call(hashes: list) -> dict:
    key = hashlib.sha256(json.dumps([hashes, PROMPT, MODEL, EFFORT]).encode()).hexdigest()[:32]
    f = OUT / 'responses' / f'{key}.json'
    if f.exists():
        return json.loads(f.read_text(encoding='utf-8'))
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w') as t:
        for i, h in enumerate(hashes):
            t.add(IMG / f'{h}.jpg', arcname=f'img{i}.jpg')
    rec = dict(status='error', attempts=0)
    for attempt in range(2):
        rec['attempts'] = attempt + 1
        out = subprocess.run(WSL + ['bash', f'{RT}/run.sh', key], input=buf.getvalue(), capture_output=True, timeout=700).stdout.decode(errors='replace')
        try:
            p = json.loads(out[out.find('{'):out.rfind('}') + 1])
            if p.get('primary') in CLS + ['uncertain'] and isinstance(p.get('confidence'), int):
                rec = dict(status='ok', parsed=p, attempts=attempt + 1)
                break
            rec = dict(status='format_error', text=out[:300], attempts=attempt + 1)
        except Exception:
            rec = dict(status='format_error', text=out[:300], attempts=attempt + 1)
    if rec['status'] == 'ok':
        f.write_text(json.dumps(rec, ensure_ascii=False), encoding='utf-8')
    return rec


def main() -> None:
    (OUT / 'responses').mkdir(parents=True, exist_ok=True)
    install()
    prepared = json.loads((S1 / 'prepared.json').read_text())
    ev = json.loads((S1 / 'sets.json').read_text())['eval']
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else len(ev)
    done = [0]

    def work(r):
        rec = call(prepared[r]); done[0] += 1
        if done[0] % 25 == 0:
            print('done', done[0], flush=True)
        return r, rec
    with ThreadPoolExecutor(4) as ex:
        res = dict(ex.map(work, ev[:limit]))
    out = {r: (dict(status=v['status'], primary=v['parsed']['primary'], conf=v['parsed']['confidence'],
                    diag=v['parsed']['diagnosis'], image_type=v['parsed']['image_type'], assess=v['parsed']['assessability'])
               if v['status'] == 'ok' else dict(status=v['status'])) for r, v in res.items()}
    if limit >= len(ev):
        (OUT / 'g61_per_record.json').write_text(json.dumps(out, indent=0))
    print('status', {s: sum(v['status'] == s for v in out.values()) for s in set(v['status'] for v in out.values())})


if __name__ == '__main__':
    main()
