"""Assemble the frozen module namespace in a new temporary directory."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

ROOT=Path(__file__).resolve().parent

def prepare_runtime():
    index=json.loads((ROOT/'comparison/topocae/engine/PATH_INDEX.json').read_text())
    runtime=Path(tempfile.mkdtemp(prefix='tc-runtime-')).resolve()
    for row in index['files']:
        source=(ROOT/row['source']).resolve()
        target=(runtime/row['runtime_path']).resolve()
        if not source.is_relative_to(ROOT) or not target.is_relative_to(runtime):
            raise ValueError('Runtime path escapes its approved directory')
        if hashlib.sha256(source.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('Changed frozen dependency: '+row['source'])
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
    return runtime
