"""Small, local G11 utilities. No model, reference, or recognition imports."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = Path(__file__).resolve().parent

def now():
    return datetime.now(timezone.utc).isoformat()

def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def descriptor(path):
    p = Path(path).resolve()
    return dict(path=p.as_posix(), sha256=sha(p), bytes=p.stat().st_size)

def checked(record):
    p = Path(record['path'])
    if descriptor(p) != record:
        raise ValueError('Input hash differs: ' + str(p))
    return p

def write(path, value, *, replace=False):
    text(path, json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n', replace=replace)

def text(path, value, *, replace=False):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = value.encode('utf-8')
    if p.exists() and not replace:
        if p.read_bytes() != data:
            raise FileExistsError('Preserve existing evidence: ' + str(p))
        return
    if replace:
        tmp = p.with_suffix(p.suffix + '.tmp')
        tmp.write_bytes(data)
        tmp.replace(p)
    else:
        with p.open('xb') as f:
            f.write(data)

def append(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a', encoding='utf-8') as f:
        f.write(canonical(value) + '\n')
