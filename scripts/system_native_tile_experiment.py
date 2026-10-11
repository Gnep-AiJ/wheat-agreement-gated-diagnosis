"""File helpers shared by the training scripts (public version: only `sha` and `atomic_json` are released)."""
import hashlib
import json
import os
from pathlib import Path


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
