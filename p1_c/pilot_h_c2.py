from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Extend.TopologyUtils import TopologyExplorer

from core.graph_v2 import extract_graph_v2
from p1_c.all_features_h_c2 import (
    GENERATOR_REVISION,
    H_C2_SECONDARY_BY_PRIMARY,
    MACRO_CLASSES,
    CleanFeatureResult,
    build_feature_sample,
)
from p1_c.geometry import face_area, finite_tree, shape_volume, topology_counts
from utils.ontology import load_layer_index, load_ontology_maps
from utils.step_solid_validation import inspect_step_file, write_step_with_positive_volume


CANONICAL_SEED_BASE = 2026076000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_h_c1(root: Path) -> dict[str, Any]:
    path = root / "output_generation_p1_c1_clean_rib_web_pilot_48" / "human_review_canonical_12.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = ("single_solid", "clean_host", "feature_concept_valid", "decision")
    invalid = []
    for row in rows:
        values = {field: row[field].strip().upper() for field in required}
        if any(values[field] != "PASS" for field in required):
            invalid.append({"review_id": row["review_id"], **values})
    result = {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "row_count": len(rows),
        "pass_count": len(rows) - len(invalid),
        "invalid_rows": invalid,
        "passed": len(rows) == 12 and not invalid,
    }
    if not result["passed"]:
        raise RuntimeError(f"H-C1 is not 12/12 PASS: {result}")
    return result


def exact_labels(result: CleanFeatureResult, root: Path) -> list[dict[str, Any]]:
    semantic_index = load_layer_index(root / "face_semantic_subtype_ontology.yaml", "face_semantic_subtype")
    ontology = load_ontology_maps(root / "CAE_feature_ontology.yaml")
    alias_to_key = semantic_index["alias_to_key"]
    key_to_id = semantic_index["key_to_id"]
    labels: list[dict[str, Any]] = []
    for face in TopologyExplorer(result.shape).faces():
        matches = [tag for tag in result.face_tags if tag.face.IsSame(face)]
        if len(matches) > 1:
            raise ValueError("one final H-C2 face has multiple truth owners")
        if not matches:
            labels.append(
                {
                    "semantic_label": int(key_to_id[alias_to_key["stock_face"]]),
                    "instance_id": 0,
                    "macro_feature": "stock",
                    "macro_feature_id": int(ontology.stock_id),
                    "semantic_role": "stock_face",
                    "face_area": face_area(face),
                }
            )
            continue
        tag = matches[0]
        macro = ontology.resolve_macro_feature(tag.macro_feature)
        semantic_key = alias_to_key[tag.semantic_role]
        labels.append(
            {
                "semantic_label": int(key_to_id[semantic_key]),
                "instance_id": int(tag.instance_id),
                "macro_feature": macro.key,
                "macro_feature_id": int(macro.id),
                "semantic_role": tag.semantic_role,
                "face_area": face_area(face),
            }
        )
    return labels


def build_graph(result: CleanFeatureResult, labels: list[dict[str, Any]], root: Path) -> dict[str, Any]:
    primitive_index = load_layer_index(root / "CAE_feature_ontology.yaml", "primitive_surface")
    graph = extract_graph_v2(result.shape, face_labels=labels, primitive_surface_index=primitive_index)
    graph["generation"] = {
        "stage": "P1-C2",
        "generator_revision": GENERATOR_REVISION,
        "parameters": result.parameters,
        "cleanup_audit": result.cleanup,
    }
    return graph


def audit_sample(
    result: CleanFeatureResult,
    labels: list[dict[str, Any]],
    graph: dict[str, Any],
    step_report: dict[str, Any],
) -> dict[str, Any]:
    parameters = result.parameters
    primary = str(parameters["primary_target"])
    secondary_features = list(parameters["secondary_features"])
    counts = topology_counts(result.shape)
    target_labels = [item for item in labels if item["instance_id"] == 1]
    secondary_labels = [item for item in labels if item["instance_id"] == 2]
    feature_labels = [item for item in labels if item["instance_id"] > 0]
    expected_instances = {1} if not secondary_features else {1, 2}
    actual_instances = {item["instance_id"] for item in feature_labels}
    step_volume = abs(float(step_report["signed_volume"] or 0.0))
    memory_volume = abs(shape_volume(result.shape))
    step_relative_volume_error = abs(step_volume - memory_volume) / max(step_volume, memory_volume, 1e-12)
    target_indices = {index for index, item in enumerate(labels) if item["instance_id"] == 1}
    secondary_indices = {index for index, item in enumerate(labels) if item["instance_id"] == 2}
    checks = {
        "single_solid": counts["solid_count"] == 1 and int(step_report["solid_count"]) == 1,
        "single_shell": counts["shell_count"] == 1 and int(step_report["shell_count"]) == 1,
        "valid_brep": bool(BRepCheck_Analyzer(result.shape).IsValid()) and bool(step_report["valid"]),
        "positive_volume": memory_volume > 1e-9 and step_volume > 1e-9,
        "clean_host": bool(parameters["host_cleanup_before_features"]["passed"]),
        "cleanup_idempotent": bool(result.cleanup["passed"]),
        "zero_redundant_faces": int(result.cleanup["removable_same_domain_face_count"]) == 0,
        "zero_redundant_edges": int(result.cleanup["removable_same_domain_edge_count"]) == 0,
        "label_coverage_100_percent": len(labels) == counts["face_count"] == len(graph["nodes"]),
        "target_exists": bool(target_labels),
        "target_macro_valid": {item["macro_feature"] for item in target_labels} == {primary},
        "secondary_exists_when_required": (not secondary_features) or bool(secondary_labels),
        "secondary_macro_valid": (not secondary_features) or (
            {item["macro_feature"] for item in secondary_labels} == {secondary_features[0]}
        ),
        "exact_instance_set": actual_instances == expected_instances,
        "target_secondary_face_disjoint": not (target_indices & secondary_indices),
        "target_secondary_distinct_macro": (not secondary_features) or primary != secondary_features[0],
        "step_face_count_roundtrip": int(step_report["face_count"]) == counts["face_count"],
        "step_edge_count_roundtrip": int(step_report["edge_count"]) == counts["edge_count"],
        "step_volume_roundtrip": step_relative_volume_error < 1e-9,
        "step_has_no_issues": bool(step_report["readable"]) and not step_report["issues"],
        "graph_finite": finite_tree(graph),
        "graph_feature_set_count": len(graph["feature_sets"]) == len(expected_instances),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "topology": counts,
        "cleanup": result.cleanup,
        "memory_volume": memory_volume,
        "step_relative_volume_error": step_relative_volume_error,
        "target_face_count": len(target_labels),
        "secondary_face_count": len(secondary_labels),
        "feature_face_count": len(feature_labels),
        "step_report": step_report,
    }


def generate_h_c2(root: Path, output: Path) -> dict[str, Any]:
    h_c1 = verify_h_c1(root)
    output.mkdir(parents=True, exist_ok=True)
    step_dir = output / "human_review_canonical_36"
    graph_dir = output / "human_review_canonical_36_graphs_v2"
    generated_at = datetime.now(timezone.utc).isoformat()
    records: list[dict[str, Any]] = []
    for class_index, primary in enumerate(MACRO_CLASSES, 1):
        single_seed = CANONICAL_SEED_BASE + (class_index - 1) * 2
        single_stem = f"{class_index:02d}_{primary}_target_only_seed{single_seed}"
        records.append(
            _write_sample(
                root=root,
                primary=primary,
                secondary=None,
                seed=single_seed,
                step_path=step_dir / f"{single_stem}.step",
                graph_path=graph_dir / f"{single_stem}.json",
                generated_at=generated_at,
            )
        )
        secondary = H_C2_SECONDARY_BY_PRIMARY[primary]
        pair_seed = single_seed + 1
        pair_stem = f"{class_index:02d}_{primary}_plus_{secondary}_seed{pair_seed}"
        records.append(
            _write_sample(
                root=root,
                primary=primary,
                secondary=secondary,
                seed=pair_seed,
                step_path=step_dir / f"{pair_stem}.step",
                graph_path=graph_dir / f"{pair_stem}.json",
                generated_at=generated_at,
            )
        )
    automatic_pass = len(records) == 36 and all(record["audit"]["passed"] for record in records)
    payload = {
        "schema_version": "topocae_p1_c_h_c2_automatic_audit_1",
        "generated_at_utc": generated_at,
        "generator_revision": GENERATOR_REVISION,
        "h_c1_review": h_c1,
        "class_order": list(MACRO_CLASSES),
        "secondary_by_primary": H_C2_SECONDARY_BY_PRIMARY,
        "sample_count": len(records),
        "automatic_pass": automatic_pass,
        "records": records,
    }
    (output / "automatic_audit_36.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_review_csv(output / "human_review_canonical_36.csv", records)
    _write_guide(output / "H-C2_UG_NX_检查指南.zh-CN.md")
    _write_summary(output / "automatic_audit_summary.zh-CN.md", payload)
    files = []
    for path in sorted(item for item in output.rglob("*") if item.is_file() and item.name != "manifest.json"):
        files.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = {
        "schema_version": "topocae_p1_c_h_c2_manifest_1",
        "generated_at_utc": generated_at,
        "generator_revision": GENERATOR_REVISION,
        "h_c1_review_sha256": h_c1["sha256"],
        "automatic_pass": automatic_pass,
        "files": files,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not automatic_pass:
        failed = Counter(
            check for record in records for check, passed in record["audit"]["checks"].items() if not passed
        )
        raise RuntimeError(f"H-C2 automatic audit failed: {dict(failed)}")
    return payload


def _write_sample(
    *,
    root: Path,
    primary: str,
    secondary: str | None,
    seed: int,
    step_path: Path,
    graph_path: Path,
    generated_at: str,
) -> dict[str, Any]:
    result = build_feature_sample(
        primary,
        seed,
        secondary_features=() if secondary is None else (secondary,),
    )
    labels = exact_labels(result, root)
    graph = build_graph(result, labels, root)
    step_path.parent.mkdir(parents=True, exist_ok=True)
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    write_step_with_positive_volume(result.shape, step_path)
    step_report = inspect_step_file(step_path)
    audit = audit_sample(result, labels, graph, step_report)
    graph["p1_c_automatic_audit"] = audit
    graph_path.write_text(json.dumps(graph, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "primary_target": primary,
        "secondary_feature": secondary,
        "context": "target_only" if secondary is None else "pair",
        "seed": int(seed),
        "generated_at_utc": generated_at,
        "generator_revision": GENERATOR_REVISION,
        "step_path": step_path.relative_to(root).as_posix(),
        "graph_path": graph_path.relative_to(root).as_posix(),
        "step_sha256": sha256_file(step_path),
        "graph_sha256": sha256_file(graph_path),
        "parameters": result.parameters,
        "audit": audit,
    }


def _write_review_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = (
        "review_id",
        "primary_target",
        "secondary_feature",
        "context",
        "host_family",
        "seed",
        "step_path",
        "step_sha256",
        "target_valid",
        "secondary_valid",
        "clean_host",
        "composition_valid",
        "decision",
        "notes",
    )
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, record in enumerate(records, 1):
            writer.writerow(
                {
                    "review_id": f"H-C2-{index:02d}",
                    "primary_target": record["primary_target"],
                    "secondary_feature": record["secondary_feature"] or "",
                    "context": record["context"],
                    "host_family": record["parameters"]["host_family"],
                    "seed": record["seed"],
                    "step_path": record["step_path"],
                    "step_sha256": record["step_sha256"],
                    "target_valid": "",
                    "secondary_valid": "N/A" if record["secondary_feature"] is None else "",
                    "clean_host": "",
                    "composition_valid": "N/A" if record["secondary_feature"] is None else "",
                    "decision": "",
                    "notes": "",
                }
            )


def _write_guide(path: Path) -> None:
    lines = [
        "# P1-C Human Gate H-C2：18类单/多特征概念审查",
        "",
        "请在 UG/NX 中打开 `human_review_canonical_36/` 内全部36个 STEP。每类连续两件：第一件 target-only，第二件 target + secondary。",
        "",
        "逐件确认：",
        "",
        "1. `primary_target` 的工程概念正确。",
        "2. pair 样本的 `secondary_feature` 概念正确。",
        "3. target 与 secondary 没有互相吞并、错误交叠或共享真值面，实例边界清楚。",
        "4. 基体只有一个实体，没有共面冗余分割、多余小面或 Boolean 残片。",
        "5. target-only 样本确实没有第二个标注特征。",
        "",
        "在 `human_review_canonical_36.csv` 中填写：",
        "",
        "- `target_valid`：PASS/FAIL。",
        "- `secondary_valid`：pair 填 PASS/FAIL；target-only 保持 N/A。",
        "- `clean_host`：PASS/FAIL。",
        "- `composition_valid`：pair 填 PASS/FAIL；target-only 保持 N/A。",
        "- `decision`：所有适用项均通过才填 PASS，否则填 FAIL。",
        "- `notes`：FAIL 必须说明文件名和问题；PASS 可简写“通过”。",
        "",
        "必须36/36 PASS。不要修改 seed、路径或 SHA-256。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    records = payload["records"]
    lines = [
        "# P1-C H-C2 自动审计摘要",
        "",
        f"- H-C1：{payload['h_c1_review']['pass_count']}/12 PASS；SHA-256 `{payload['h_c1_review']['sha256']}`。",
        f"- H-C2：{sum(record['audit']['passed'] for record in records)}/{len(records)} 自动 PASS。",
        "- 18类各1个 target-only 和1个 pair，合计36件。",
        "- 全部检查单实体、单壳、有效B-Rep、cleanup幂等、零冗余face/edge、标签100%、实例集合、STEP重读和Graph v2 finite。",
        "- rib/web 的 pair 孔位于基体并与薄壁保持净距，不穿过 target 薄壁。",
        "",
        "自动审计通过后进入 Human Gate H-C2；36/36 人工 PASS 前不得生成 Clean-Context-7200。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


__all__ = ["generate_h_c2", "verify_h_c1"]
