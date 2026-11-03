from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'paper_release/topocae_p1_legacy_baseline_v1'
REPORT = ROOT / 'reports/aei_paper_goals_v1/g03'
ARTIFACT = ROOT / 'artifacts/aei_paper_goals_v1/g03'
DATA = SOURCE / 'output_training_p1_e6r2_r3_graph_v2_2_1_v1'
OLD_REPORT = SOURCE / 'reports/p1_e6r2_r3_m2_m5_controlled_training_v1'
TOOLS = Path(__file__).resolve().parent
SEEDS = (20260718, 20260719, 20260720)
sys.path.insert(0, str(SOURCE))


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def owned(path):
    path = Path(path).resolve()
    allowed = (REPORT, ARTIFACT, TOOLS, ROOT / 'tests/aei_paper_goals_v1')
    if not any(path.is_relative_to(root.resolve()) for root in allowed):
        raise ValueError(f'G03 write outside ownership: {path}')
    return path


def write_json(path, value):
    path = owned(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def hashed(path):
    path = Path(path).resolve()
    return {'path': path.as_posix(), 'bytes': path.stat().st_size, 'sha256': sha256(path)}


def graph_path(sample_id):
    return SOURCE / 'output_generation_p1_e6r2_r1_3_r1_complete_r2_mixed_9300_v1/graphs' / f'{sample_id}.json'


def run_dir(seed):
    return OLD_REPORT / 'training_runs' / f'final_8370_seed_{seed}'


def load_records(split):
    import torch
    from p1_e6r2_r3.training_common import adapt_record
    if split not in ('train', 'val'):
        raise ValueError('G03 accepts original train/val only')
    raw = torch.load(DATA / f'{split}.pt', map_location='cpu', weights_only=False)
    expected = {'train': 7440, 'val': 930}[split]
    if len(raw) != expected or any(r['split'] != split for r in raw):
        raise ValueError('frozen split mismatch')
    return [adapt_record(r) for r in raw]


def decoder_assets():
    base = SOURCE / 'reports/p1_e3c_seed_matched_candidate_v1/pipelines/old_same_instance_seed_20260718'
    paths = (base / 'candidate_model/candidate_threshold_selection.json', base / 'global_decode_selection.json')
    if not all(p.is_file() for p in paths):
        raise ValueError(f'missing frozen decoder assets: {paths}')
    return paths
