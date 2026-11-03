"""G07 owned I/O and reproducible execution utilities; historical artifacts read-only."""
from __future__ import annotations
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / 'tools/aei_paper_goals_v1/g07'
REPORT = ROOT / 'reports/aei_paper_goals_v1/g07'
ARTIFACT = ROOT / 'artifacts/aei_paper_goals_v1/g07'
FROZEN = ROOT / 'paper_release/topocae_p1_legacy_baseline_v1'
SEEDS = (20260718, 20260719, 20260720)

def now():
    return datetime.now(timezone.utc).isoformat()

def owned(path):
    path = Path(path).resolve()
    valid = any(path.is_relative_to(p.resolve()) for p in (TOOLS, REPORT, ARTIFACT))
    valid |= path.parent == ROOT / 'tests/aei_paper_goals_v1' and path.name.startswith('test_g07_')
    if not valid:
        raise ValueError(f'G07 output outside authorized directories: {path}')
    return path

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path, value):
    path = owned(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, path)

def append(path, value):
    path = owned(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps({'at_utc': now(), **value}, ensure_ascii=False, allow_nan=False) + '\n')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def asset(path):
    path = Path(path).resolve()
    return {'path': path.as_posix(), 'bytes': path.stat().st_size, 'sha256': sha(path)}

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def runtime_environment():
    env = os.environ.copy()
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[key] = '4'
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', PYTHONIOENCODING='utf-8',
               WANDB_MODE='disabled', WANDB_DISABLED='true', DGLBACKEND='pytorch')
    for key, suffix in (('DGLDEFAULTDIR','dgl'), ('TORCH_HOME','torch'), ('HF_HOME','huggingface'),
                        ('MPLCONFIGDIR','matplotlib')):
        env[key] = str(ARTIFACT / 'runtime_cache' / suffix)
    return env

def status(completed, remaining, blockers=(), state='running'):
    path = REPORT / 'RUN_STATUS.json'
    old = read(path) if path.exists() else {}
    write(path, {'goal_id':'G07', 'status':state, 'started_at_utc':old.get('started_at_utc',now()),
        'updated_at_utc':now(), 'ended_at_utc':now() if state=='ready_for_review' else None,
        'completed':list(completed), 'remaining':list(remaining), 'blockers':list(blockers),
        'dependency_review_paths':[str(REPORT.parent/'reviews'/f'G{i:02d}_REVIEW.md') for i in (1,2,3,5)],
        'authorization':'G07_RECOGNITION_IMPROVEMENT.md supersedes historical G01-G05 gates for this iteration',
        'review_passed':False, 'review_owner':'original conversation', 'G06_files_modified':False})
