"""Mainline V2 stage-1 VLM benchmark runner (blind: pixels only, labels joined later).

Usage: python -m scripts.mainline_v2_vlm build
       python -m scripts.mainline_v2_vlm run --model gpt|claude --set eval|wfd|sentinel --round r1 [--workers 6] [--limit N]
Keys are read from ~/.config/wheat_vlm/keys.json (outside the project); never logged.
"""
import argparse
import csv
import hashlib
import io
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/mainline_v2/stage1'
IMG = ROOT / 'outputs/mainline_v2/images'
PROMPT_PATH = ROOT / 'research/MAINLINE_V2_PROMPT_2026-09-29.txt'
MANIFEST = ROOT / 'outputs/stage0/split_candidate_v1/stage0_candidate_manifest.csv'
LABELS = ('healthy', 'leaf_rust', 'powdery_mildew', 'septoria', 'stem_rust', 'yellow_rust')
BUDGET_PX, TILE_TRIGGER_PX = 1_050_000, 4_200_000
# Conservative USD per 1M tokens (no cache discount; reasoning counted as output).
PRICE = {'gpt': (5.0, 40.0), 'claude': (15.0, 75.0)}
PER_CALL_CAP_USD = {'gpt': 0.30, 'claude': 1.00}
PARAMS = {
    'gpt': dict(model='gpt-6-sol', reasoning_effort='medium', detail='high'),
    'claude': dict(model='claude-opus-5-5', max_tokens=8000, effort='medium', anthropic_version='2023-06-01'),
}
_lock = threading.Lock()


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def assert_not_sealed(path: str) -> None:
    p = path.replace('\\', '/').lower()
    assert 'external_holdout' not in p and 'iari' not in p, 'IARI/sealed data must not be touched before stage 5'


def read_manifest() -> dict:
    with MANIFEST.open(encoding='utf-8-sig', newline='') as f:
        return {r['record_id']: r for r in csv.DictReader(f)}


def build() -> None:
    rows = read_manifest()
    ev = []
    for f in range(3):
        ev += [str(x) for x in np.load(ROOT / f'outputs/strong_expert_training_v1/dino_expanded_fold{f}/evaluation.npz')['record_ids']]
    assert len(ev) == 450 and len(set(ev)) == 450
    with (ROOT / 'outputs/leaf_rust_net_correction_v3/CODEX_120_BLIND_RESPONSES_2026-09-24.csv').open(encoding='utf-8') as f:
        wfd = [r['record_id'] for r in csv.DictReader(f)]
    assert len(wfd) == 120 and len(set(wfd)) == 120
    used = set(ev) | set(wfd)
    rng = np.random.default_rng(20260929)
    quota = dict(zip(LABELS, (4, 4, 3, 3, 3, 3)))
    sentinel = []
    for lab in LABELS:
        pool = sorted(r for r, x in rows.items() if x['partition'] == 'development_oof' and x['canonical_class'] == lab and r not in used)
        sentinel += [str(x) for x in rng.choice(pool, quota[lab], replace=False)]
    for rid in used | set(sentinel):
        assert rows[rid]['partition'] == 'development_oof'
        assert_not_sealed(rows[rid]['absolute_path'])
    OUT.mkdir(parents=True, exist_ok=True)
    sets = dict(eval=ev, wfd=wfd, sentinel=sentinel, overlap_eval_wfd=sorted(set(ev) & set(wfd)))
    (OUT / 'sets.json').write_text(json.dumps(sets, indent=1))
    # Blind image preparation: record_id -> list of prepared-image sha256.
    prep = {}
    for rid in sorted(used | set(sentinel)):
        prep[rid] = prepare(rows[rid]['absolute_path'])
    (OUT / 'prepared.json').write_text(json.dumps(prep, indent=1))
    print({k: len(v) for k, v in sets.items()}, 'tiled', sum(len(v) > 1 for v in prep.values()))


def _encode(im: Image.Image) -> str:
    w, h = im.size
    if w * h > BUDGET_PX:
        s = (BUDGET_PX / (w * h)) ** 0.5
        im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, 'JPEG', quality=92)  # fresh RGB image: no EXIF/ICC metadata carried
    b = buf.getvalue()
    h = sha_bytes(b)
    IMG.mkdir(parents=True, exist_ok=True)
    p = IMG / f'{h}.jpg'
    if not p.exists():
        p.write_bytes(b)
    return h


def prepare(path: str) -> list:
    assert_not_sealed(path)
    im = ImageOps.exif_transpose(Image.open(path))
    im = Image.frombytes('RGB', im.size, im.convert('RGB').tobytes())
    out = [_encode(im)]
    w, h = im.size
    if w * h > TILE_TRIGGER_PX:
        for box in [(0, 0, w // 2, h // 2), (w // 2, 0, w, h // 2), (0, h // 2, w // 2, h), (w // 2, h // 2, w, h)]:
            out.append(_encode(im.crop(box)))
    return out


def keys() -> dict:
    return json.loads((Path.home() / '.config/wheat_vlm/keys.json').read_text())


def _loads_tolerant(s: str):
    """Parse a JSON object; if the tail was cut (the relay drops the last few tokens),
    keep only fully completed top-level fields. Returns (obj, repaired)."""
    s = s.strip().rstrip('`').strip()
    for tail in ('', '}', '"}'):
        try:
            return json.loads(s + tail), tail != ''
        except Exception:
            pass
    depth, in_str, esc, cuts = 0, False, False, []
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == chr(92):  # backslash
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in '{[':
            depth += 1
        elif ch in '}]':
            depth -= 1
        elif ch == ',' and depth == 1:
            cuts.append(i)
    for c in reversed(cuts):
        try:
            return json.loads(s[:c] + '}'), True
        except Exception:
            continue
    return None, False


def parse(text: str):
    i = (text or '').find('{')
    if i < 0:
        return None
    j, rep = _loads_tolerant(text[i:])
    if not isinstance(j, dict):
        return None
    j['_repaired'] = rep
    if not isinstance(j.get('confidence'), (int, float)):
        return None  # essential fields must be complete; otherwise format retry
    d = j.get('diagnosis')
    if not isinstance(d, list) or not d or j.get('primary') not in LABELS + ('uncertain',):
        return None
    if any(x not in LABELS + ('uncertain',) for x in d):
        return None
    return j


def cost(model: str, usage: dict) -> float:
    pin, pout = PRICE[model]
    if model == 'gpt':
        i, o = usage.get('prompt_tokens', 0), usage.get('completion_tokens', 0)
    else:
        i = usage.get('input_tokens', 0) + usage.get('cache_creation_input_tokens', 0) + usage.get('cache_read_input_tokens', 0)
        o = usage.get('output_tokens', 0)
    return (i * pin + o * pout) / 1e6


def request(model: str, hashes: list, prompt: str, k: dict, max_tokens: int = 8000, stream: bool = True):
    imgs = [(IMG / f'{h}.jpg').read_bytes() for h in hashes]
    import base64
    b64 = [base64.b64encode(b).decode() for b in imgs]
    if model == 'gpt':
        content = [{'type': 'text', 'text': prompt}] + [
            {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + x, 'detail': 'high'}} for x in b64]
        body = dict(model='gpt-6-sol', reasoning_effort='medium', messages=[{'role': 'user', 'content': content}])
        r = requests.post(k['base'] + '/v1/chat/completions', headers={'Authorization': 'Bearer ' + k['gpt']}, json=body, timeout=300)
    else:
        content = [{'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/jpeg', 'data': x}} for x in b64]
        content.append({'type': 'text', 'text': prompt})
        # Streaming: the relay's non-stream mode deterministically returns empty bodies for some images.
        body = dict(model='claude-opus-5-5', max_tokens=max_tokens, output_config={'effort': 'medium'}, stream=stream,
                    messages=[{'role': 'user', 'content': content}])
        r = requests.post(k['base'] + '/v1/messages', headers={'x-api-key': k['claude'], 'anthropic-version': '2023-06-01'},
                          json=body, timeout=300, stream=stream)
        if r.status_code == 200 and stream:
            r = _collect_stream(r)
    return r


class _Collected:
    """Minimal response-like object assembled from an Anthropic SSE stream."""
    def __init__(self, status_code, payload, text=''):
        self.status_code, self._payload, self.text = status_code, payload, text

    def json(self):
        return self._payload


def _collect_stream(r):
    msg, blocks, usage = {}, {}, {}
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith('data:'):
            continue
        d = json.loads(line[5:])
        t = d.get('type')
        if t == 'message_start':
            msg = d['message']
            usage.update(msg.get('usage') or {})
        elif t == 'content_block_start':
            blocks[d['index']] = dict(d['content_block'])
        elif t == 'content_block_delta' and d['delta'].get('type') == 'text_delta':
            blocks.setdefault(d['index'], {'type': 'text', 'text': ''})
            blocks[d['index']]['text'] = blocks[d['index']].get('text', '') + d['delta']['text']
        elif t == 'message_delta':
            msg['stop_reason'] = d['delta'].get('stop_reason')
            usage.update(d.get('usage') or {})
        elif t == 'error':
            return _Collected(502, None, json.dumps(d)[:400])
    msg['content'] = [blocks[i] for i in sorted(blocks)]
    msg['usage'] = usage
    return _Collected(200, msg)


def extract(model: str, j: dict):
    """Return (text, stop, usage, echoed_model)."""
    if model == 'gpt':
        c = j['choices'][0]
        return c['message'].get('content') or '', c.get('finish_reason'), j.get('usage', {}), j.get('model')
    text = ''.join(b.get('text', '') for b in j.get('content', []) if b.get('type') == 'text')
    return text, j.get('stop_reason'), j.get('usage', {}), j.get('model')


def call_one(model: str, hashes: list, prompt: str, round_id: str, k: dict) -> dict:
    ph = sha_bytes(prompt.encode())
    key = sha_bytes(json.dumps([hashes, ph, model, PARAMS[model], round_id]).encode())
    folder = OUT / 'responses' / model
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'{key}.json'
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    rec = dict(key=key, model=model, round=round_id, images=hashes, prompt_sha=ph, attempts=[], status=None)
    parse_retry, max_tokens, net_retry = 0, 8000, 0
    while True:
        t0 = time.time()
        try:
            r = request(model, hashes, prompt, k, max_tokens, stream=(net_retry % 2 == 0))
            code = r.status_code
            j = r.json() if code == 200 else None
            err = None if code == 200 else r.text[:400]
        except Exception as e:  # network / timeout
            code, j, err = -1, None, repr(e)[:400]
        att = dict(code=code, seconds=round(time.time() - t0, 1), error=err)
        if j is not None:
            text, stop, usage, echoed = extract(model, j)
            att.update(stop=stop, usage=usage, echoed_model=echoed, text=text, cost=cost(model, usage))
        else:
            # No usage returned: conservatively charge the per-call cap unless the request was rejected up front.
            att['cost'] = 0.0 if code in (400, 401, 403, 404, 429) else PER_CALL_CAP_USD[model]
        rec['attempts'].append(att)
        if j is None:
            if code in (-1, 408, 409, 429) or code >= 500:
                net_retry += 1
                if net_retry <= 3:
                    time.sleep(10 * 2 ** net_retry)
                    continue
            rec['status'] = f'http_error_{code}'
            break
        if not text.strip() and stop != 'refusal':
            # Relay sometimes returns an empty body (1 output token): technical failure, not a diagnosis.
            net_retry += 1
            if net_retry <= 3:
                time.sleep(5 * 2 ** net_retry)
                continue
            rec['status'] = 'empty_response'
            break
        if model == 'claude' and stop == 'refusal':
            rec['status'] = 'refusal'
            break
        if stop in ('max_tokens', 'length'):
            if max_tokens == 8000 and model == 'claude':
                max_tokens = 16000
                continue
            rec['status'] = 'truncated'
            break
        parsed = parse(text)
        if parsed is None:
            parse_retry += 1
            if parse_retry <= 1:
                continue
            rec['status'] = 'format_error'
            break
        rec['status'], rec['parsed'] = 'ok', parsed
        break
    rec['cost'] = sum(a.get('cost', 0.0) for a in rec['attempts'])
    # Do not cache hard HTTP failures (e.g. quota) so they can be rerun after the blocker clears.
    if not rec['status'].startswith('http_error'):
        path.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding='utf-8')
    with _lock:
        with (OUT / 'ledger.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(dict(t=time.strftime('%Y-%m-%dT%H:%M:%S'), key=key, model=model, round=round_id,
                                    status=rec['status'], cost=rec['cost'], n_img=len(hashes),
                                    usage=[a.get('usage') for a in rec['attempts']])) + '\n')
    return rec


def run(model: str, which: str, round_id: str, workers: int, limit: int) -> None:
    sets = json.loads((OUT / 'sets.json').read_text())
    prep = json.loads((OUT / 'prepared.json').read_text())
    prompt = PROMPT_PATH.read_text(encoding='utf-8')
    frozen = OUT / 'prompt_sha.txt'
    ph = sha_bytes(prompt.encode())
    if frozen.exists():
        assert frozen.read_text().strip() == ph, 'frozen prompt changed'
    else:
        frozen.write_text(ph)
    ids = sets['eval'] + [x for x in sets['wfd'] if x not in set(sets['eval'])] if which == 'evalwfd' else sets[which]
    # Blind: only prepared-image hashes are dispatched, in a shuffled order.
    jobs = sorted({tuple(prep[r]) for r in ids})
    random.Random(7).shuffle(jobs)
    if limit:
        jobs = jobs[:limit]
    if workers == 0:  # pending-count mode
        ph = sha_bytes(prompt.encode())
        n = sum(not (OUT / 'responses' / model / (sha_bytes(json.dumps([list(h), ph, model, PARAMS[model], round_id]).encode()) + '.json')).exists() for h in jobs)
        print(n)
        return
    k = keys()
    stats, spent, t0 = {}, 0.0, time.time()
    stop_flag = threading.Event()
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(lambda h: None if stop_flag.is_set() else call_one(model, list(h), prompt, round_id, k), h) for h in jobs]
        for i, fu in enumerate(as_completed(futs), 1):
            rec = fu.result()
            if rec is None:
                continue
            stats[rec['status']] = stats.get(rec['status'], 0) + 1
            spent += rec['cost'] if rec['attempts'] else 0
            if rec['status'].startswith('http_error'):
                codes = [a['code'] for a in rec['attempts']]
                if stats[rec['status']] >= 3:
                    stop_flag.set()
                print('HTTP_ERROR', codes, (rec['attempts'][-1].get('error') or '')[:200], flush=True)
            if i % 20 == 0 or i == len(jobs):
                print(f'{model} {which} {round_id} {i}/{len(jobs)} {stats} est${spent:.2f} {time.time()-t0:.0f}s', flush=True)
    print('DONE', model, which, round_id, stats, f'est${spent:.2f}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd')
    ap.add_argument('--model')
    ap.add_argument('--set', default='evalwfd')
    ap.add_argument('--round', default='r1')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    if a.cmd == 'build':
        build()
    else:
        run(a.model, a.set, a.round, a.workers, a.limit)
