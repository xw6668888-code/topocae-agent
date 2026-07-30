from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from utils.cae_query_ground_p3_a1r.frozen_assets import verify_freeze_manifest


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "output_cae_query_ground_v1_1" / "manifests" / "dataset_manifest.json"
FORMAL_ROOT = ROOT / "output_generation_p3_a1r_multi_instance_formal_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().lower()


def line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def validate() -> dict[str, Any]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    failures: list[dict[str, Any]] = []
    for entry in manifest["files"]:
        path = Path(entry["path"])
        if not path.is_file():
            failures.append({"path": str(path), "reason": "missing"})
            continue
        actual_hash = sha256_file(path)
        if actual_hash != str(entry["sha256"]).lower():
            failures.append({"path": str(path), "reason": "hash_mismatch"})
        if line_count(path) != int(entry["record_count"]):
            failures.append({"path": str(path), "reason": "record_count_mismatch"})
    upstream_manifest = Path(manifest["composition"]["upstream_dataset_manifest"])
    if sha256_file(upstream_manifest) != manifest["composition"]["upstream_dataset_manifest_sha256"]:
        failures.append({"path": str(upstream_manifest), "reason": "upstream_manifest_changed"})

    geometry_meta = json.loads((FORMAL_ROOT / "dataset_meta.json").read_text(encoding="utf-8"))
    geometry_audit = json.loads(
        (FORMAL_ROOT / "reports" / "geometry_validity_audit.json").read_text(encoding="utf-8")
    )
    descriptor_audit = json.loads(
        (FORMAL_ROOT / "reports" / "descriptor_and_metric_audit.json").read_text(encoding="utf-8")
    )
    leakage_audit = json.loads(
        (FORMAL_ROOT / "reports" / "split_leakage_audit.json").read_text(encoding="utf-8")
    )
    query_audit = json.loads(
        (FORMAL_ROOT / "reports" / "query_semantic_audit.json").read_text(encoding="utf-8")
    )
    freeze = verify_freeze_manifest(
        ROOT,
        ROOT / "reports" / "p3_a1_cae_queryground_v1" / "freeze" / "freeze_manifest.json",
    )
    coverage = manifest["supplement_coverage"]
    thresholds = {
        "formal_base_models_at_least_510": coverage["multi_instance_base_models"] >= 510,
        "all_at_least_450": coverage["all"] >= 450,
        "count_at_least_450": coverage["count"] >= 450,
        "L2_multi_at_least_900": coverage["L2_executable_multi"] >= 900,
        "L4_at_least_900": coverage["L4_executable"] >= 900,
        "largest_at_least_300": coverage["largest"] >= 300,
        "smallest_at_least_300": coverage["smallest"] >= 300,
        "nearest_at_least_300": coverage["nearest"] >= 300,
        "target_reconstruction_complete": coverage["target_reconstruction"]["passed"]
        == coverage["target_reconstruction"]["total"],
        "metric_recomputation_complete": coverage["metric_recomputation"]["passed"]
        == coverage["metric_recomputation"]["total"],
        "geometry_automatic_pass": geometry_meta["automatic_pass"]
        and geometry_audit["automatic_pass"],
        "descriptor_automatic_pass": descriptor_audit["automatic_pass"],
        "geometry_split_leakage_zero": leakage_audit["automatic_pass"],
        "query_automatic_pass": query_audit["automatic_pass"],
        "query_split_leakage_zero": query_audit["split_leakage"]["automatic_pass"],
        "old_p3_101_unchanged": freeze["automatic_pass"]
        and freeze["matching_asset_count"] == 101
        and freeze["hash_changes"] == 0,
        "old_record_semantics_changed_zero": manifest["composition"]["old_records_semantics_changed"] == 0,
        "scope_guards_all_false": not any(
            bool(value) for value in manifest["scope_guards"].values()
        ),
    }
    automatic_pass = not failures and all(thresholds.values())
    return {
        "schema_version": "cae-query-ground-v1.1-read-only-validation/1",
        "dataset_version": manifest["dataset_version"],
        "file_failures": failures[:50],
        "thresholds": thresholds,
        "old_p3_freeze": {
            "declared_assets": freeze["declared_asset_count"],
            "matching_assets": freeze["matching_asset_count"],
            "hash_changes": freeze["hash_changes"],
        },
        "automatic_pass": automatic_pass,
    }


def main() -> int:
    result = validate()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["automatic_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
