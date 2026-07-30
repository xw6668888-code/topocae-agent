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
from p1_c.geometry import face_area, finite_tree, shape_volume, topology_counts
from p1_c.rib_web import GENERATOR_REVISION, SCENARIO_SPECS, CleanRibWebResult, build_rib_web_sample, scenario_keys
from utils.ontology import load_layer_index, load_ontology_maps
from utils.step_solid_validation import inspect_step_file, write_step_with_positive_volume


PILOT_COUNT_PER_SCENARIO = 4
PILOT_SEED_BASE = 2026071800
CANONICAL_SEED_BASE = 2026071900


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_labels(result: CleanRibWebResult, root: Path) -> list[dict[str, Any]]:
    semantic_index = load_layer_index(root / "face_semantic_subtype_ontology.yaml", "face_semantic_subtype")
    ontology = load_ontology_maps(root / "CAE_feature_ontology.yaml")
    alias_to_key = semantic_index["alias_to_key"]
    key_to_id = semantic_index["key_to_id"]
    labels: list[dict[str, Any]] = []
    for face in TopologyExplorer(result.shape).faces():
        matches = [tag for tag in result.face_tags if tag.face.IsSame(face)]
        if len(matches) > 1:
            raise ValueError("multiple P1-C truth records matched one final face")
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
        labels.append(
            {
                "semantic_label": int(key_to_id[alias_to_key[tag.semantic_role]]),
                "instance_id": int(tag.instance_id),
                "macro_feature": macro.key,
                "macro_feature_id": int(macro.id),
                "semantic_role": tag.semantic_role,
                "face_area": face_area(face),
            }
        )
    return labels


def build_graph(result: CleanRibWebResult, labels: list[dict[str, Any]], root: Path) -> dict[str, Any]:
    primitive_index = load_layer_index(root / "CAE_feature_ontology.yaml", "primitive_surface")
    graph = extract_graph_v2(result.shape, face_labels=labels, primitive_surface_index=primitive_index)
    graph["generation"] = {
        "stage": "P1-C1",
        "generator_revision": GENERATOR_REVISION,
        "parameters": result.parameters,
        "cleanup_audit": result.cleanup,
    }
    return graph


def audit_sample(
    result: CleanRibWebResult,
    labels: list[dict[str, Any]],
    graph: dict[str, Any],
    step_report: dict[str, Any],
) -> dict[str, Any]:
    params = result.parameters
    counts = topology_counts(result.shape)
    feature_instance = int(params["feature_instance_id"])
    feature_labels = [item for item in labels if item["instance_id"] == feature_instance]
    feature_macro = str(params["macro_feature"])
    pocket_labels = [item for item in labels if item["macro_feature"] in {"pocket_rect", "pocket_circular"}]
    hole_labels = [item for item in labels if item["macro_feature"] == "through_hole_round"]
    owned_face_indices = [index for index, item in enumerate(labels) if item["instance_id"] > 0]
    step_volume = abs(float(step_report["signed_volume"] or 0.0))
    memory_volume = abs(shape_volume(result.shape))
    step_relative_volume_error = abs(step_volume - memory_volume) / max(step_volume, memory_volume, 1e-12)
    checks = {
        "single_solid": counts["solid_count"] == 1 and int(step_report["solid_count"]) == 1,
        "valid_brep": bool(BRepCheck_Analyzer(result.shape).IsValid()) and bool(step_report["valid"]),
        "positive_volume": memory_volume > 1e-9 and step_volume > 1e-9,
        "no_internal_shell": counts["shell_count"] == 1 and int(step_report["shell_count"]) == 1,
        "cleanup_idempotent": bool(result.cleanup["passed"]),
        "zero_redundant_faces": int(result.cleanup["removable_same_domain_face_count"]) == 0,
        "zero_redundant_edges": int(result.cleanup["removable_same_domain_edge_count"]) == 0,
        "clean_host_before_feature": bool(params["host_cleanup_before_feature"]["passed"]),
        "support_contact": int(params["positive_support_contact_count"]) >= 2,
        "all_declared_contacts_positive": all(item["intersection_volume"] > 1e-7 for item in params["support_contacts"]),
        "thin_wall_proportion": float(params["span_thickness_ratio"]) >= 5.0 and float(params["height_thickness_ratio"]) >= 2.0,
        "feature_label_coverage": len(feature_labels) == int(params["feature_face_count"]) and len(feature_labels) >= 3,
        "feature_macro_consistent": {item["macro_feature"] for item in feature_labels} == {feature_macro},
        "paired_side_faces": sum(item["semantic_role"] == f"{feature_macro}_side_plane" for item in feature_labels) >= 2,
        "label_coverage_100_percent": len(labels) == counts["face_count"] == len(graph["nodes"]),
        "face_multiple_ownership_zero": len(owned_face_indices) == len(set(owned_face_indices)),
        "pocket_independent": params["host_family"] not in {"rect_cavity", "circular_cavity"} or (
            bool(pocket_labels) and {item["instance_id"] for item in pocket_labels} == {1} and feature_instance == 2
        ),
        "hole_independent": not params["has_hole"] or (
            bool(hole_labels) and {item["instance_id"] for item in hole_labels} == {feature_instance + 1}
        ),
        "step_face_count_roundtrip": int(step_report["face_count"]) == counts["face_count"],
        "step_edge_count_roundtrip": int(step_report["edge_count"]) == counts["edge_count"],
        "step_volume_roundtrip": step_relative_volume_error < 1e-9,
        "step_has_no_issues": bool(step_report["readable"]) and not step_report["issues"],
        "graph_finite": finite_tree(graph),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "topology": counts,
        "cleanup": result.cleanup,
        "memory_volume": memory_volume,
        "step_relative_volume_error": step_relative_volume_error,
        "feature_face_count": len(feature_labels),
        "pocket_face_count": len(pocket_labels),
        "hole_face_count": len(hole_labels),
        "support_contacts": params["support_contacts"],
        "step_report": step_report,
    }


def generate_h_c1(root: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    pilot_step_dir = output / "pilot_48" / "steps"
    pilot_graph_dir = output / "pilot_48" / "graphs_v2"
    canonical_step_dir = output / "human_review_canonical_12"
    canonical_graph_dir = output / "human_review_canonical_12_graphs_v2"
    records: list[dict[str, Any]] = []
    canonical_records: list[dict[str, Any]] = []
    generated_at = datetime.now(timezone.utc).isoformat()
    for scenario_index, scenario_key in enumerate(scenario_keys()):
        for replicate in range(PILOT_COUNT_PER_SCENARIO):
            seed = PILOT_SEED_BASE + scenario_index * PILOT_COUNT_PER_SCENARIO + replicate
            stem = f"{scenario_index + 1:02d}_{scenario_key}_r{replicate + 1}_seed{seed}"
            records.append(
                _write_sample(
                    root=root,
                    scenario_key=scenario_key,
                    seed=seed,
                    step_path=pilot_step_dir / f"{stem}.step",
                    graph_path=pilot_graph_dir / f"{stem}.json",
                    generated_at=generated_at,
                    canonical=False,
                )
            )
        seed = CANONICAL_SEED_BASE + scenario_index
        stem = f"{scenario_index + 1:02d}_{scenario_key}_seed{seed}"
        canonical_records.append(
            _write_sample(
                root=root,
                scenario_key=scenario_key,
                seed=seed,
                step_path=canonical_step_dir / f"{stem}.step",
                graph_path=canonical_graph_dir / f"{stem}.json",
                generated_at=generated_at,
                canonical=True,
            )
        )
    all_records = records + canonical_records
    automatic_pass = len(records) == 48 and len(canonical_records) == 12 and all(item["audit"]["passed"] for item in all_records)
    payload = {
        "schema_version": "topocae_p1_c_h_c1_automatic_audit_1",
        "generated_at_utc": generated_at,
        "generator_revision": GENERATOR_REVISION,
        "scenario_specs": SCENARIO_SPECS,
        "pilot_count": len(records),
        "canonical_count": len(canonical_records),
        "automatic_pass": automatic_pass,
        "pilot_records": records,
        "canonical_records": canonical_records,
    }
    audit_path = output / "automatic_audit_60.json"
    audit_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_review_csv(output / "human_review_canonical_12.csv", canonical_records)
    _write_guide(output / "H-C1_UG_NX_检查指南.zh-CN.md")
    _write_summary(output / "automatic_audit_summary.zh-CN.md", payload)
    manifest_records = []
    for path in sorted(item for item in output.rglob("*") if item.is_file() and item.name != "manifest.json"):
        manifest_records.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = {
        "schema_version": "topocae_p1_c_h_c1_manifest_1",
        "generated_at_utc": generated_at,
        "generator_revision": GENERATOR_REVISION,
        "automatic_pass": automatic_pass,
        "files": manifest_records,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not automatic_pass:
        failed = Counter(
            check
            for record in all_records
            for check, passed in record["audit"]["checks"].items()
            if not passed
        )
        raise RuntimeError(f"H-C1 automatic audit failed: {dict(failed)}")
    return payload


def _write_sample(
    *,
    root: Path,
    scenario_key: str,
    seed: int,
    step_path: Path,
    graph_path: Path,
    generated_at: str,
    canonical: bool,
) -> dict[str, Any]:
    result = build_rib_web_sample(scenario_key, seed)
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
        "scenario_key": scenario_key,
        "seed": seed,
        "canonical": canonical,
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
        "scenario_key",
        "macro_feature",
        "host_family",
        "variant",
        "seed",
        "step_path",
        "step_sha256",
        "single_solid",
        "clean_host",
        "feature_concept_valid",
        "decision",
        "notes",
    )
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, record in enumerate(records, 1):
            params = record["parameters"]
            writer.writerow(
                {
                    "review_id": f"H-C1-{index:02d}",
                    "scenario_key": record["scenario_key"],
                    "macro_feature": params["macro_feature"],
                    "host_family": params["host_family"],
                    "variant": params["variant"],
                    "seed": record["seed"],
                    "step_path": record["step_path"],
                    "step_sha256": record["step_sha256"],
                    "single_solid": "",
                    "clean_host": "",
                    "feature_concept_valid": "",
                    "decision": "",
                    "notes": "",
                }
            )


def _write_guide(path: Path) -> None:
    lines = [
        "# P1-C Human Gate H-C1：UG/NX 审查指南",
        "",
        "请打开 `human_review_canonical_12/` 内全部 12 个 STEP；本 Gate 不要求检查 raw face ID。",
        "",
        "逐件确认：",
        "",
        "1. 模型树和测量结果只有一个有效实体。",
        "2. 基体没有重叠体、内部壳、共面分割线、多余小面或不必要台阶。",
        "3. rib/web 是受至少两个支撑区域连接的薄壁增强体，没有吞掉或堵死基体。",
        "4. rect/circular cavity 仍清楚存在；hole 和 transition 变体的概念成立。",
        "5. 孔没有切断薄壁，transition 连续且没有碎片。",
        "",
        "在同级 `human_review_canonical_12.csv` 中只填写：",
        "",
        "- `single_solid`：PASS/FAIL。",
        "- `clean_host`：PASS/FAIL。",
        "- `feature_concept_valid`：PASS/FAIL。",
        "- `decision`：仅当前三项均 PASS 时填 PASS，否则填 FAIL。",
        "- `notes`：FAIL 必须说明文件名和问题；PASS 可简写“通过”。",
        "",
        "必须 12/12 PASS。不要修改 seed、路径或 SHA-256 列。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    records = payload["pilot_records"] + payload["canonical_records"]
    minimum_contact = min(
        item["intersection_volume"]
        for record in records
        for item in record["parameters"]["support_contacts"]
    )
    lines = [
        "# P1-C H-C1 自动审计摘要",
        "",
        f"- Pilot：{payload['pilot_count']}/48。",
        f"- Canonical：{payload['canonical_count']}/12。",
        f"- 自动审计：{sum(record['audit']['passed'] for record in records)}/{len(records)} PASS。",
        f"- Generator revision：`{payload['generator_revision']}`。",
        f"- 最小支撑交叠体积：`{minimum_contact:.9f}`。",
        f"- 最大 cleanup 相对体积误差：`{max(record['audit']['cleanup']['relative_volume_error'] for record in records):.3e}`。",
        f"- 最大 STEP 重读相对体积误差：`{max(record['audit']['step_relative_volume_error'] for record in records):.3e}`。",
        "- 60/60 均检查单实体、有效 B-Rep、单壳、cleanup 幂等、零冗余面/edge、标签互斥、支撑接触、STEP 重读一致和 Graph v2 finite。",
        "",
        "自动审计通过后进入 Human Gate H-C1；未收到 12/12 人工 PASS 前不得生成 H-C2。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


__all__ = ["generate_h_c1"]
