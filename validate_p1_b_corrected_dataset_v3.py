from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "output_generation_p1_b_corrected_balanced_5000_v3"
REPORT_DIR = ROOT / "reports" / "p1_b_corrected_dataset_validation"
TARGET_CLASSES = [
    "through_hole_round", "blind_hole_round", "stepped_hole_round", "counterbore_hole",
    "countersink_hole", "obround_hole", "slot_u", "slot_rect", "pocket_rect",
    "pocket_circular", "cylinder_boss", "rib", "web", "rectangular_boss", "fillet",
    "chamfer", "slot_v", "slot_o",
]
REBUILD_ONLY = {"slot_u", "slot_rect", "slot_v", "slot_o", "rib", "web"}
EXPECTED_PROTECTED = {
    "output_generation_v5_1_augmented_full_5400_clean": {"file_count": 21630, "byte_count": 1416242301},
    "output_generation_paper_v1_synthetic_holdout_v1": {"file_count": 2211, "byte_count": 140219897},
}


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    meta = json.loads((DATASET / "dataset_meta.json").read_text(encoding="utf-8"))
    records = _read_jsonl(DATASET / "manifests" / "sample_manifest.jsonl")
    instance_counts = Counter()
    source_counts = Counter()
    derivation_counts = Counter()
    geometry_formats = Counter()
    hash_failures = []
    graph_failures = []
    base_splits: dict[str, set[str]] = defaultdict(set)
    exact_splits: dict[str, set[str]] = defaultdict(set)
    near_splits: dict[str, set[str]] = defaultdict(set)
    gate_counts = Counter()
    finite_value_count = 0
    max_abs_recognition = 0.0
    for position, record in enumerate(records):
        graph_path = Path(record["graph_path"])
        geometry_path = Path(record["geometry_path"])
        if _sha256(graph_path) != record["graph_sha256"]:
            hash_failures.append({"sample_id": record["sample_id"], "artifact": "graph"})
        if _sha256(geometry_path) != record["geometry_sha256"]:
            hash_failures.append({"sample_id": record["sample_id"], "artifact": "geometry"})
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        failures = []
        if graph.get("schema_version") != "topocae_graph_v2_relative_1_1":
            failures.append("wrong_graph_schema")
        if graph.get("sample_id") != record["sample_id"]:
            failures.append("sample_id_mismatch")
        if not graph.get("feature_sets"):
            failures.append("missing_feature_sets")
        macro_counts = Counter(str(feature["macro_feature"]) for feature in graph.get("feature_sets", []))
        if dict(sorted(macro_counts.items())) != record["macro_instance_counts"]:
            failures.append("manifest_instance_count_mismatch")
        provenance = graph.get("dataset_provenance", {})
        if record["source_kind"].startswith("legacy"):
            if set(macro_counts) & REBUILD_ONLY:
                failures.append("legacy_rebuild_only_macro_present")
            if provenance.get("old_rebuild_only_macro_present") is not False:
                failures.append("legacy_provenance_contract_missing")
        else:
            generator = provenance.get("generator")
            gate_hash = provenance.get("human_gate_csv_sha256")
            if generator == "p1_b_continuous_closed_profile_sweep_cut" and gate_hash:
                gate_counts["H_B1"] += 1
            elif generator == "p1_b_supported_rib_web" and gate_hash:
                gate_counts["H_B2"] += 1
            else:
                failures.append("new_generator_gate_provenance_missing")
        for node in graph.get("nodes", []):
            values = node.get("attributes", {}).get("recognition_features_v2", {}).values()
            for value in values:
                number = float(value)
                finite_value_count += 1
                max_abs_recognition = max(max_abs_recognition, abs(number))
                if not math.isfinite(number):
                    failures.append("nonfinite_node_recognition")
                    break
        for edge in graph.get("edge_attributes", []):
            values = edge.get("recognition_features_v2", {}).values()
            for value in values:
                number = float(value)
                finite_value_count += 1
                max_abs_recognition = max(max_abs_recognition, abs(number))
                if not math.isfinite(number):
                    failures.append("nonfinite_edge_recognition")
                    break
        if failures:
            graph_failures.append({"sample_id": record["sample_id"], "failures": sorted(set(failures))})
        instance_counts.update(macro_counts)
        source_counts[record["source_kind"]] += 1
        derivation_counts[record["derivation_kind"]] += 1
        geometry_formats[record["geometry_format"]] += 1
        base_splits[record["base_geometry_family_id"]].add(record["split"])
        exact_splits[record["exact_graph_signature"]].add(record["split"])
        near_splits[record["near_graph_signature"]].add(record["split"])
        if (position + 1) % 500 == 0:
            print(f"validated {position + 1}/{len(records)}", flush=True)
    split_files = {
        split: _read_jsonl(DATASET / "splits" / f"{split}.jsonl")
        for split in ("train", "val", "development_test")
    }
    manifest_split_ids = {
        split: {record["sample_id"] for record in records if record["split"] == split}
        for split in split_files
    }
    split_file_ids = {split: {record["sample_id"] for record in values} for split, values in split_files.items()}
    split_manifest_match = manifest_split_ids == split_file_ids
    protected = {name: _inventory(ROOT / name) for name in EXPECTED_PROTECTED}
    protected_unchanged = protected == EXPECTED_PROTECTED
    leakage = {
        "base_group": sum(len(splits) > 1 for splits in base_splits.values()),
        "exact_signature": sum(len(splits) > 1 for splits in exact_splits.values()),
        "near_signature": sum(len(splits) > 1 for splits in near_splits.values()),
    }
    under = {macro: instance_counts[macro] for macro in TARGET_CLASSES if instance_counts[macro] < 300}
    automatic_pass = (
        len(records) == 5000
        and not hash_failures
        and not graph_failures
        and not under
        and not any(leakage.values())
        and split_manifest_match
        and derivation_counts["legal_face_partition"] >= 80
        and gate_counts["H_B1"] >= 1200
        and gate_counts["H_B2"] >= 600
        and protected_unchanged
        and meta.get("automatic_pass") is True
    )
    payload = {
        "schema_version": "topocae_p1_b_corrected_dataset_independent_validation_1",
        "automatic_pass": automatic_pass,
        "dataset_root": str(DATASET.resolve()),
        "sample_count": len(records),
        "instance_count": sum(instance_counts.values()),
        "instance_counts": {macro: instance_counts[macro] for macro in TARGET_CLASSES},
        "minimum_300_failures": under,
        "source_counts": dict(sorted(source_counts.items())),
        "derivation_counts": dict(sorted(derivation_counts.items())),
        "geometry_artifact_counts": dict(sorted(geometry_formats.items())),
        "human_gate_provenance_counts": dict(sorted(gate_counts.items())),
        "hash_failure_count": len(hash_failures),
        "hash_failures": hash_failures,
        "graph_failure_count": len(graph_failures),
        "graph_failures": graph_failures,
        "finite_recognition_value_count": finite_value_count,
        "max_abs_raw_recognition_feature": max_abs_recognition,
        "split_manifest_match": split_manifest_match,
        "split_sample_counts": {split: len(values) for split, values in split_files.items()},
        "leakage": leakage,
        "protected_inventory": protected,
        "protected_expected": EXPECTED_PROTECTED,
        "protected_unchanged": protected_unchanged,
        "contracts": {
            "legacy_rebuild_only_macro_reused": False,
            "new_slot_gate_provenance_present": gate_counts["H_B1"] >= 1200,
            "new_rib_web_gate_provenance_present": gate_counts["H_B2"] >= 600,
            "all_graphs_v2_1": not any("wrong_graph_schema" in item["failures"] for item in graph_failures),
            "real_data_used": False,
        },
    }
    report_path = REPORT_DIR / "automatic_validation.json"
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_path = REPORT_DIR / "automatic_validation_summary.zh-CN.md"
    summary_path.write_text(_summary(payload), encoding="utf-8")
    manifest = {
        "schema_version": "topocae_p1_b_corrected_dataset_independent_validation_manifest_1",
        "automatic_pass": automatic_pass,
        "files": [
            {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in (report_path, summary_path)
        ],
    }
    (REPORT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "automatic_pass": automatic_pass,
        "sample_count": len(records),
        "instance_count": sum(instance_counts.values()),
        "min_class_count": min(instance_counts[macro] for macro in TARGET_CLASSES),
        "hash_failures": len(hash_failures),
        "graph_failures": len(graph_failures),
        "leakage": leakage,
        "protected_unchanged": protected_unchanged,
    }, ensure_ascii=False), flush=True)
    return 0 if automatic_pass else 1


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _inventory(path: Path) -> dict[str, int]:
    files = [item for item in path.rglob("*") if item.is_file()]
    return {"file_count": len(files), "byte_count": sum(item.stat().st_size for item in files)}


def _summary(payload: dict[str, Any]) -> str:
    lines = [
        "# P1-B 修正数据集独立验证",
        "",
        f"- 自动结论：{'PASS' if payload['automatic_pass'] else 'FAIL'}",
        f"- 样本/实例：{payload['sample_count']} / {payload['instance_count']}",
        f"- hash failures：{payload['hash_failure_count']}",
        f"- graph/provenance failures：{payload['graph_failure_count']}",
        f"- split manifest 一致：{payload['split_manifest_match']}",
        f"- base/exact/near leakage：{payload['leakage']}",
        f"- 受保护旧数据未变化：{payload['protected_unchanged']}",
        "",
        "## 18 类实例",
        "",
    ]
    for macro, count in payload["instance_counts"].items():
        lines.append(f"- {macro}: {count}")
    lines.append("")
    return "\n".join(lines)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


if __name__ == "__main__":
    raise SystemExit(main())
