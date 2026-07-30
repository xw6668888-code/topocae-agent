from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "output_generation_v5_1_augmented_full_5400_clean"
OUTPUT = ROOT / "reports" / "p1_b_corrected_dataset_source_audit"
TARGET_CLASSES = [
    "through_hole_round", "blind_hole_round", "stepped_hole_round", "counterbore_hole",
    "countersink_hole", "obround_hole", "slot_u", "slot_rect", "pocket_rect",
    "pocket_circular", "cylinder_boss", "rib", "web", "rectangular_boss", "fillet",
    "chamfer", "slot_v", "slot_o",
]
REBUILD_ONLY_CLASSES = {"slot_u", "slot_rect", "slot_v", "slot_o", "rib", "web"}


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    index_path = SOURCE / "manifests" / "sample_index.jsonl"
    selected = []
    excluded = []
    selected_instance_counts = Counter()
    excluded_macro_counts = Counter()
    source_roles = Counter()
    with index_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            index_record = json.loads(line)
            graph_path = Path(index_record["graph_path"])
            step_path = Path(index_record["step_path"])
            graph = json.loads(graph_path.read_text(encoding="utf-8"))
            truths = _truth_instances(graph)
            macros = {record["macro_feature"] for record in truths}
            reasons = []
            rebuild_macros = sorted(macros & REBUILD_ONLY_CLASSES)
            if rebuild_macros:
                reasons.append("contains_rebuild_only_old_concept")
            if not graph.get("feature_sets"):
                reasons.append("missing_feature_sets_provenance")
            if not graph.get("ablation_policy", {}).get("valid", False):
                reasons.append("ablation_policy_not_valid")
            merged_source = graph.get("merged_dataset_source", {})
            if not merged_source.get("source_root") or not merged_source.get("source_sample_id"):
                reasons.append("missing_merged_dataset_source")
            if not step_path.exists():
                reasons.append("missing_step")
            if any(not record["face_indices"] for record in truths):
                reasons.append("empty_truth_instance")
            if reasons:
                excluded.append({
                    "sample_id": index_record["sample_id"],
                    "reasons": sorted(set(reasons)),
                    "macro_features": sorted(macros),
                    "rebuild_only_macros": rebuild_macros,
                })
                excluded_macro_counts.update(record["macro_feature"] for record in truths)
                continue
            source_roles[str(merged_source.get("source_role", "unknown"))] += 1
            selected_instance_counts.update(record["macro_feature"] for record in truths)
            selected.append({
                "sample_id": index_record["sample_id"],
                "source_graph_path": str(graph_path.resolve()),
                "source_step_path": str(step_path.resolve()),
                "source_step_sha256": _sha256(step_path),
                "source_graph_sha256": _sha256(graph_path),
                "macro_features": sorted(macros),
                "feature_instance_count": len(truths),
                "source_role": merged_source.get("source_role"),
                "source_root": merged_source.get("source_root"),
                "source_sample_id": merged_source.get("source_sample_id"),
                "reuse_policy": "unaffected_generator_provenance_no_rebuild_only_macro_graph_v2_reextract_required",
            })
    deficits = {
        macro: max(0, 300 - selected_instance_counts[macro])
        for macro in TARGET_CLASSES
    }
    payload = {
        "schema_version": "topocae_p1_b_corrected_dataset_source_audit_1",
        "source_root": str(SOURCE.resolve()),
        "source_sample_count": len(selected) + len(excluded),
        "selected_safe_sample_count": len(selected),
        "excluded_sample_count": len(excluded),
        "selected_instance_counts": {macro: selected_instance_counts[macro] for macro in TARGET_CLASSES},
        "minimum_300_deficits_before_new_generation": deficits,
        "excluded_instance_macro_counts": dict(sorted(excluded_macro_counts.items())),
        "selected_source_roles": dict(sorted(source_roles.items())),
        "rebuild_only_classes": sorted(REBUILD_ONLY_CLASSES),
        "selection_contract": {
            "old_slot_or_rib_web_allowed": False,
            "feature_sets_required": True,
            "valid_noninterference_policy_required": True,
            "merged_source_provenance_required": True,
            "graph_v2_reextraction_required": True,
            "source_artifacts_modified": False,
        },
        "selected_samples": selected,
        "excluded_samples": excluded,
    }
    audit_path = OUTPUT / "source_audit.json"
    audit_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_path = OUTPUT / "source_audit_summary.zh-CN.md"
    summary_path.write_text(_summary(payload), encoding="utf-8")
    manifest = {
        "schema_version": "topocae_p1_b_corrected_dataset_source_audit_manifest_1",
        "files": [
            {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in (audit_path, summary_path)
        ],
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "selected_safe_sample_count": len(selected),
        "excluded_sample_count": len(excluded),
        "selected_instance_counts": payload["selected_instance_counts"],
        "deficits": deficits,
    }, ensure_ascii=False))
    return 0


def _truth_instances(graph: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for feature_set in graph.get("feature_sets", []):
        faces = feature_set.get("node_indices", feature_set.get("face_indices", []))
        records.append({
            "instance_id": feature_set.get("instance_id"),
            "macro_feature": str(feature_set.get("macro_feature", "unknown_feature")),
            "face_indices": sorted(set(int(index) for index in faces)),
        })
    return records


def _summary(payload: dict[str, Any]) -> str:
    lines = [
        "# P1-B 修正数据集旧源审计",
        "",
        f"- 旧源样本：{payload['source_sample_count']}",
        f"- 可复用候选：{payload['selected_safe_sample_count']}",
        f"- 排除：{payload['excluded_sample_count']}",
        "- 任何包含旧槽或旧 rib/web 的样本均排除；旧源只读，复用时必须从 STEP 重提 Graph v2。",
        "",
        "## 可复用实例数 / 距 300 缺口",
        "",
    ]
    for macro in TARGET_CLASSES:
        lines.append(
            f"- {macro}: {payload['selected_instance_counts'][macro]} / deficit {payload['minimum_300_deficits_before_new_generation'][macro]}"
        )
    lines.append("")
    return "\n".join(lines)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


if __name__ == "__main__":
    raise SystemExit(main())
