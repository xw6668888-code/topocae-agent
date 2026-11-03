"""Download immutable private-release attachments and check their hashes."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def token():
    value = os.environ.get('GH_TOKEN') or os.environ.get('GITHUB_TOKEN')
    if value: return value
    result = subprocess.run(['git', 'credential', 'fill'], input='protocol=https\nhost=github.com\n\n',
                            capture_output=True, text=True, check=True,
                            env={**os.environ, 'GIT_TERMINAL_PROMPT': '0', 'GCM_INTERACTIVE': 'Never'})
    values = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    if not values.get('password'): raise RuntimeError('GitHub read credentials are required for this private repository.')
    return values['password']


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None: new.remove_header('Authorization')
        return new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=['all', 'training', 'test', 'engineering', 'models'], default='all')
    parser.add_argument('--profile', choices=['all','demo'], default='all', help='demo downloads only the three frozen deployment weight files needed by infer.py/run_case.py')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'assets.json').read_text())
    todo = []
    for row in manifest['assets']:
        if args.profile=='demo' and not row['path'].startswith('comparison/topocae/engine/'):
            continue
        if args.group != 'all' and row['group'] != args.group: continue
        p = ROOT / row['path']
        if p.exists():
            if p.stat().st_size != row['bytes'] or sha(p) != row['sha256']:
                raise RuntimeError('Existing asset differs; preserve and investigate: ' + row['path'])
        else: todo.append(row)
    if not todo:
        print('All selected assets are present and verified.'); return
    headers = {'Authorization': 'Bearer ' + token(), 'Accept': 'application/vnd.github+json', 'User-Agent': 'TopoCAE-reproduce'}
    base = 'https://api.github.com/repos/' + manifest['repository']
    request = urllib.request.Request(base + '/releases/tags/' + manifest['release_tag'], headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response: release = json.load(response)
    byname = {r['name']: r for r in release['assets']}
    opener = urllib.request.build_opener(SafeRedirect())
    for row in todo:
        asset = byname[row['asset_name']]
        assert asset['size'] == row['bytes']
        p = ROOT / row['path']; p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + '.part')
        if tmp.exists(): raise RuntimeError('Preserve existing partial download: ' + str(tmp))
        req = urllib.request.Request(asset['url'], headers={**headers, 'Accept': 'application/octet-stream'})
        with opener.open(req, timeout=120) as response, tmp.open('xb') as f:
            while True:
                block = response.read(1024 * 1024)
                if not block: break
                f.write(block)
        assert tmp.stat().st_size == row['bytes'] and sha(tmp) == row['sha256'], row['path']
        tmp.rename(p)
        print(row['path'] + ': verified', flush=True)


if __name__ == '__main__': main()
