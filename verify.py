"""Verify source/data integrity; attachment verification is optional."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parent


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--code-only', action='store_true')
    a = p.parse_args()
    rows = json.loads((ROOT / 'MANIFEST.json').read_text())['files']
    if not a.code_only: rows += json.loads((ROOT / 'assets.json').read_text())['assets']
    for row in rows:
        path = ROOT / row['path']
        assert path.is_file(), row['path']
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], row['path']
    print('Verified {} files.'.format(len(rows)))


if __name__ == '__main__': main()
