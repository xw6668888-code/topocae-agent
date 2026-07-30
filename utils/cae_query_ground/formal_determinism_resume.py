from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

from .constants import CROSSWALK_PATH, REPORT_ROOT, ROOT, SEED
from .formal_pipeline import _quality_log
from .formal_planner import make_formal_record
from .formal_planner_parallel_release import generate_formal_plans_parallel
from .formal_renderer_v3 import render_formal_all
from .io import canonical_json_bytes, read_yaml, sha256_file, write_json


OUTPUT = ROOT / "output_cae_query_ground_v1"
REPORT_PATH = REPORT_ROOT / "deterministic_regeneration.json"


def _update(digest: Any, value: Any) -> None:
    digest.update(canonical_json_bytes(value))
    digest.update(b"\n")


def verify(*, force: bool = True) -> dict[str, Any]:
    plans = generate_formal_plans_parallel(read_yaml(CROSSWALK_PATH), seed=SEED)
    canonical_digests = {split: hashlib.sha256() for split in ("train", "val", "test")}
    rendering_digests = {split: hashlib.sha256() for split in ("train", "val", "test")}
    index_digest = hashlib.sha256()
    quality_log_digest = hashlib.sha256()
    plan_digest = hashlib.sha256()
    counts: Counter[str] = Counter()
    for index, plan in enumerate(plans, start=1):
        _update(plan_digest, plan)
        canonical_query, queries = render_formal_all(plan)
        canonical = make_formal_record(plan, record_kind="canonical", query=canonical_query)
        renderings = [
            make_formal_record(plan, record_kind="rendering", query=query) for query in queries
        ]
        split = canonical["source_ref"]["dataset_split"]
        _update(canonical_digests[split], canonical)
        for rendering in renderings:
            _update(rendering_digests[split], rendering)
        _update(index_digest, {
            "canonical_record_id": canonical["source_ref"]["canonical_record_id"],
            "canonical_semantics_sha256": canonical["source_ref"]["canonical_semantics_sha256"],
            "dataset_split": split,
            "rendering_ids": sorted(item["query"]["rendering_id"] for item in renderings),
        })
        _update(quality_log_digest, _quality_log([plan])[0])
        counts[split] += 1
        if index % 5000 == 0:
            print(f"deterministic regeneration: {index}/{len(plans)}", flush=True)

    expected = {
        **{
            f"canonical_{split}": canonical_digests[split].hexdigest()
            for split in ("train", "val", "test")
        },
        **{
            f"rendering_{split}": rendering_digests[split].hexdigest()
            for split in ("train", "val", "test")
        },
        "canonical_to_rendering_index": index_digest.hexdigest(),
        "quality_filter_log": quality_log_digest.hexdigest(),
    }
    paths = {
        **{
            f"canonical_{split}": OUTPUT / "canonical" / f"{split}.jsonl"
            for split in ("train", "val", "test")
        },
        **{
            f"rendering_{split}": OUTPUT / "renderings" / f"{split}.jsonl"
            for split in ("train", "val", "test")
        },
        "canonical_to_rendering_index": OUTPUT / "manifests" / "canonical_to_rendering.jsonl",
        "quality_filter_log": OUTPUT / "reports" / "quality_filter_log.jsonl",
    }
    actual = {role: sha256_file(path) for role, path in paths.items()}
    mismatches = {
        role: {"expected": expected[role], "actual": actual[role]}
        for role in expected
        if expected[role] != actual[role]
    }
    if mismatches:
        raise RuntimeError(f"Output-level deterministic regeneration failed: {mismatches}")
    result = {
        "schema_version": "cae-query-ground-deterministic-regeneration/1",
        "automatic_pass": True,
        "seed": SEED,
        "canonical_records": len(plans),
        "split_counts": dict(sorted(counts.items())),
        "plan_sha256": plan_digest.hexdigest(),
        "regenerated_file_sha256": expected,
        "existing_file_sha256": actual,
        "mismatches": {},
        "proof": "A fresh source+seed regeneration produced byte-identical canonical, rendering, index, and quality-log JSONL hashes.",
    }
    write_json(REPORT_PATH, result, force=force)
    return result


if __name__ == "__main__":
    import json

    print(json.dumps(verify(), ensure_ascii=False, sort_keys=True))

