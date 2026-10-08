"""Plain gpt-6.1-sol (frozen V2 prompt, WSL Codex) on the ETS images. Saves predictions only (no metrics)."""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
import paper_p2_codex_g61 as P  # noqa: E402
from scripts.mainline_v2_vlm import prepare  # noqa: E402


def main() -> None:
    P.install()
    E = ROOT / os.environ.get('ETS_DIR', 'outputs/paper_hrme/ets')
    items = json.loads((E / 'items.json').read_text())
    prep = {it['file']: prepare(it['file']) for it in items}
    (E / 'prepared.json').write_text(json.dumps(prep, indent=0))
    done = [0]

    def work(it):
        r = P.call(prep[it['file']]); done[0] += 1
        if done[0] % 25 == 0:
            print('done', done[0], flush=True)
        return it['file'], r
    with ThreadPoolExecutor(4) as ex:
        res = dict(ex.map(work, items))
    out = {k: (dict(status='ok', primary=v['parsed']['primary'], conf=v['parsed']['confidence']) if v['status'] == 'ok' else dict(status=v['status']))
           for k, v in res.items()}
    (E / 'g61.json').write_text(json.dumps(out, indent=0))
    print('status', {s: sum(v['status'] == s for v in out.values()) for s in {v['status'] for v in out.values()}})


if __name__ == '__main__':
    main()
