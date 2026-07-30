from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
OLD_ROOT = ROOT / "output_cae_query_ground_v1"
SUPPLEMENT_ROOT = ROOT / "output_generation_p3_a1r_multi_instance_formal_v1"
OUTPUT_ROOT = ROOT / "output_cae_query_ground_v1_1"
VERSION = "cae-query-ground/1.1.0-h3r.01"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().lower()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def file_entry(path: Path, role: str, record_count: int) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "role": role,
        "bytes": path.stat().st_size,
        "record_count": int(record_count),
        "sha256": sha256_file(path),
    }


def _verify_old_manifest(old_manifest: dict[str, Any]) -> None:
    for entry in old_manifest["files"]:
        path = ROOT / entry["path"]
        if not path.is_file() or sha256_file(path) != str(entry["sha256"]).lower():
            raise RuntimeError(f"old P3 dependency mismatch: {path}")


def _old_strata() -> dict[str, int]:
    counts = Counter()
    for split in ("train", "val", "test"):
        path = OLD_ROOT / "canonical" / f"{split}.jsonl"
        for record in read_jsonl(path):
            status = str(record["intent_result"]["status"])
            if status != "VALID":
                counts["clarification_or_invalid"] += 1
                if status == "NEEDS_CLARIFICATION":
                    counts["clarification"] += 1
                else:
                    counts["invalid"] += 1
                continue
            targets = record["ground_truth"].get("targets", [])
            if any(target.get("target_contract") == "whole_part" for target in targets):
                counts["whole_part"] += 1
            feature_targets = [
                target for target in targets if target.get("target_contract") == "feature_face_set"
            ]
            if feature_targets:
                counts["feature_target_valid"] += 1
                candidate_counts = [
                    int(target.get("uniqueness_contract", {}).get("candidate_count", 0))
                    for target in feature_targets
                ]
                if candidate_counts and max(candidate_counts) == 1:
                    counts["singleton_candidate_control"] += 1
    return dict(sorted(counts.items()))


def build() -> dict[str, Any]:
    manifests = OUTPUT_ROOT / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    old_manifest_path = OLD_ROOT / "manifests" / "dataset_manifest.json"
    old_manifest = json.loads(old_manifest_path.read_text(encoding="utf-8"))
    _verify_old_manifest(old_manifest)

    supplement_canonical = SUPPLEMENT_ROOT / "queries" / "canonical.jsonl"
    supplement_renderings = SUPPLEMENT_ROOT / "queries" / "renderings.jsonl"
    supplement_geometry_manifest = SUPPLEMENT_ROOT / "manifests" / "sample_manifest.jsonl"
    query_meta = json.loads(
        (SUPPLEMENT_ROOT / "query_dataset_meta.json").read_text(encoding="utf-8")
    )
    query_audit = query_meta["query_audit"]
    geometry_meta = json.loads((SUPPLEMENT_ROOT / "dataset_meta.json").read_text(encoding="utf-8"))

    index_path = manifests / "supplement_canonical_to_rendering.jsonl"
    by_canonical: dict[str, list[str]] = {}
    for rendering in read_jsonl(supplement_renderings):
        by_canonical.setdefault(str(rendering["canonical_record_id"]), []).append(
            str(rendering["rendering_id"])
        )
    canonical_ids = [
        str(record["source_ref"]["canonical_record_id"])
        for record in read_jsonl(supplement_canonical)
    ]
    if set(canonical_ids) != set(by_canonical):
        raise RuntimeError("supplement canonical/rendering identity mismatch")
    if any(len(by_canonical[item]) != 3 for item in canonical_ids):
        raise RuntimeError("supplement rendering cardinality is not 3")
    index_path.write_text(
        "".join(
            json.dumps(
                {
                    "canonical_record_id": canonical_id,
                    "rendering_ids": sorted(by_canonical[canonical_id]),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
            for canonical_id in canonical_ids
        ),
        encoding="utf-8",
        newline="\n",
    )

    old_counts = old_manifest["counts"]
    supplement_status = Counter(query_audit["status_counts"])
    supplement_splits = Counter(query_audit["canonical_split_distribution"])
    features = Counter(old_counts["features"])
    for feature in features:
        features[feature] += 180
    counts = {
        "canonical_records": int(old_counts["canonical_records"]) + int(query_meta["canonical_records"]),
        "renderings": int(old_counts["renderings"]) + int(query_meta["renderings"]),
        "statuses": {
            "VALID": int(old_counts["statuses"]["VALID"]) + int(supplement_status["VALID"]),
            "NEEDS_CLARIFICATION": (
                int(old_counts["statuses"]["NEEDS_CLARIFICATION"])
                + int(supplement_status["NEEDS_CLARIFICATION"])
            ),
            "INVALID": int(old_counts["statuses"]["INVALID"]),
        },
        "levels": {
            **old_counts["levels"],
            "L2": int(old_counts["levels"]["L2"]) + int(query_audit["L2_executable_multi"]),
            "L4": int(old_counts["levels"]["L4"]) + 1836,
        },
        "splits": {
            split: int(old_counts["splits"][split]) + int(supplement_splits[split])
            for split in ("train", "val", "test")
        },
        "languages": {
            "en-US": int(old_counts["languages"]["en-US"]) + 6120,
            "zh-CN": int(old_counts["languages"]["zh-CN"]) + 3060,
        },
        "features": dict(sorted(features.items())),
        "operations": {
            **old_counts["operations"],
            "local_mesh_size": int(old_counts["operations"]["local_mesh_size"]) + 3060,
        },
    }
    old_strata = _old_strata()
    strata = {
        "old_p3_singleton_candidate_control": old_strata.get("singleton_candidate_control", 0),
        "old_p3_feature_target_valid": old_strata.get("feature_target_valid", 0),
        "old_p3_whole_part": old_strata.get("whole_part", 0),
        "multi_instance_all_count_valid": int(query_audit["L2_executable_multi"]),
        "comparison_valid": (
            int(query_audit["L4_by_quantifier"]["largest"])
            + int(query_audit["L4_by_quantifier"]["smallest"])
        ),
        "spatial_nearest_valid": int(query_audit["L4_by_quantifier"]["nearest"]),
        "clarification_total": counts["statuses"]["NEEDS_CLARIFICATION"],
        "supplement_clarification": int(supplement_status["NEEDS_CLARIFICATION"]),
        "invalid_total": counts["statuses"]["INVALID"],
    }

    old_files = []
    for split in ("train", "val", "test"):
        old_files.append(
            file_entry(
                OLD_ROOT / "canonical" / f"{split}.jsonl",
                f"upstream_v1_canonical_{split}",
                int(old_counts["splits"][split]),
            )
        )
        old_files.append(
            file_entry(
                OLD_ROOT / "renderings" / f"{split}.jsonl",
                f"upstream_v1_renderings_{split}",
                int(old_counts["splits"][split]) * 3,
            )
        )
    files = old_files + [
        file_entry(
            supplement_canonical,
            "p3_a1r_supplement_canonical",
            int(query_meta["canonical_records"]),
        ),
        file_entry(
            supplement_renderings,
            "p3_a1r_supplement_renderings",
            int(query_meta["renderings"]),
        ),
        file_entry(
            index_path,
            "p3_a1r_supplement_canonical_to_rendering_index",
            int(query_meta["canonical_records"]),
        ),
        file_entry(
            supplement_geometry_manifest,
            "p3_a1r_supplement_geometry_manifest",
            int(geometry_meta["sample_count"]),
        ),
    ]
    manifest = {
        "schema_version": "cae-query-ground-dataset-manifest/1.1.0-h3r.01",
        "dataset_version": VERSION,
        "status": "candidate_for_p3_a1r_freeze",
        "composition": {
            "mode": "immutable_upstream_reference_plus_deterministic_supplement",
            "upstream_dataset_version": old_manifest["dataset_version"],
            "upstream_dataset_manifest": str(old_manifest_path.resolve()),
            "upstream_dataset_manifest_sha256": sha256_file(old_manifest_path),
            "supplement_geometry_root": str(SUPPLEMENT_ROOT.resolve()),
            "supplement_geometry_manifest_sha256": sha256_file(supplement_geometry_manifest),
            "old_records_semantics_changed": 0,
        },
        "official_entrypoint": str((ROOT / "read_cae_query_ground_v1_1.py").resolve()),
        "read_only_verifier": str((ROOT / "validate_cae_query_ground_v1_1.py").resolve()),
        "counts": counts,
        "semantic_strata": strata,
        "supplement_coverage": {
            "multi_instance_base_models": int(geometry_meta["sample_count"]),
            "L2_executable_multi": int(query_audit["L2_executable_multi"]),
            "all": int(query_audit["L2_all"]),
            "count": int(query_audit["L2_count"]),
            "L4_executable": int(query_audit["L4_executable"]),
            "largest": int(query_audit["L4_by_quantifier"]["largest"]),
            "smallest": int(query_audit["L4_by_quantifier"]["smallest"]),
            "nearest": int(query_audit["L4_by_quantifier"]["nearest"]),
            "target_reconstruction": query_audit["target_reconstruction"],
            "metric_recomputation": query_audit["metric_recomputation"],
        },
        "files": files,
        "scope_guards": {
            "llm_called": False,
            "model_trained": False,
            "solver_executed": False,
            "p4_entered": False,
            "holdout_run": False,
            "real_part_content_accessed": False,
            "forbidden_data_accessed": False,
        },
    }
    manifest_path = manifests / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    sources = {
        "schema_version": "cae-query-ground-sources/1.1.0-h3r.01",
        "canonical_sources": [
            entry["path"] for entry in files if "canonical_" in entry["role"] and "index" not in entry["role"]
        ],
        "rendering_sources": [
            entry["path"] for entry in files if "renderings_" in entry["role"]
        ],
    }
    (manifests / "dataset_sources.json").write_text(
        json.dumps(sources, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def main() -> int:
    manifest = build()
    print(json.dumps({
        "dataset_version": manifest["dataset_version"],
        "counts": manifest["counts"],
        "semantic_strata": manifest["semantic_strata"],
        "supplement_coverage": manifest["supplement_coverage"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
