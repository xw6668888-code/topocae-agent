from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Extend.TopologyUtils import TopologyExplorer

from core.AAGExtractor import extract_aag_graph
from features.p1_b_rib_web_v3 import SCENARIO_SPECS, P1BRibWebResult, build_rib_web_sample, scenario_keys
from utils.ontology import export_layer_label_map, load_layer_index, load_ontology_maps
from utils.step_solid_validation import inspect_step_file, write_step_with_positive_volume


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "output_generation_p1_b2_rib_web_pilot_48"
PILOT_SEED_BASE = 202607186000
CANONICAL_SEED_BASE = 202607187000
PILOT_PER_SCENARIO = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate P1-B2 engineering rib/web pilot and H-B2 review package.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def shape_volume(shape: Any) -> float:
    props = GProp_GProps()
    brepgprop.VolumeProperties(shape, props)
    return float(props.Mass())


def face_area(face: Any) -> float:
    props = GProp_GProps()
    brepgprop.SurfaceProperties(face, props)
    return float(props.Mass())


def exact_face_labels(
    result: P1BRibWebResult,
    *,
    semantic_index: dict[str, Any],
    ontology_maps: Any,
) -> list[dict[str, Any]]:
    labels = []
    alias_to_key = semantic_index["alias_to_key"]
    key_to_id = semantic_index["key_to_id"]
    for face in TopologyExplorer(result.shape).faces():
        matches = [tag for tag in result.face_tags if face.IsSame(tag.face)]
        if len(matches) > 1:
            raise ValueError("multiple semantic tags matched one final face")
        if not matches:
            labels.append({
                "semantic_label": key_to_id[alias_to_key["stock_face"]],
                "instance_id": 0,
                "macro_feature": "stock",
                "macro_feature_id": ontology_maps.stock_id,
                "semantic_role": "stock_face",
                "face_area": face_area(face),
            })
            continue
        tag = matches[0]
        macro = ontology_maps.resolve_macro_feature(tag.macro_key)
        semantic_key = alias_to_key[tag.semantic_role]
        labels.append({
            "semantic_label": key_to_id[semantic_key],
            "instance_id": tag.instance_id,
            "macro_feature": macro.key,
            "macro_feature_id": macro.id,
            "semantic_role": tag.semantic_role,
            "face_area": face_area(face),
        })
    return labels


def exact_graph(
    result: P1BRibWebResult,
    *,
    semantic_index: dict[str, Any],
    ontology_maps: Any,
    primitive_index: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    labels = exact_face_labels(result, semantic_index=semantic_index, ontology_maps=ontology_maps)
    graph = extract_aag_graph(result.shape, primitive_surface_index=primitive_index, include_label_fields=False)
    if len(labels) != len(graph["nodes"]):
        raise RuntimeError("topology traversal order changed during graph extraction")
    for node, label in zip(graph["nodes"], labels):
        node.update({key: value for key, value in label.items() if key != "face_area"})
    instances: dict[str, list[int]] = {}
    for node in graph["nodes"]:
        instances.setdefault(str(node["instance_id"]), []).append(node["idx"])
    graph["instances"] = instances
    return graph, labels


def audit_sample(
    result: P1BRibWebResult,
    *,
    step_report: dict[str, Any],
    graph: dict[str, Any],
    labels: list[dict[str, Any]],
) -> dict[str, Any]:
    params = result.parameters
    macro_key = params["macro_key"]
    feature_inst = params["feature_instance_id"]
    feature_labels = [
        label for label in labels
        if label["macro_feature"] == macro_key and label["instance_id"] == feature_inst
    ]
    stock_labels = [label for label in labels if label["macro_feature"] == "stock"]
    pocket_labels = [label for label in labels if label["macro_feature"] in {"pocket_rect", "pocket_circular"}]
    hole_labels = [label for label in labels if label["macro_feature"] == "through_hole"]
    feature_side_role = f"{macro_key}_side_plane"
    feature_side_count = sum(label["semantic_role"] == feature_side_role for label in feature_labels)
    cavity_expected = params["context_kind"] in {"rect_cavity", "circular_cavity"}
    hole_expected = params["variant"] == "hole"
    transition_expected = params["variant"] == "transition"
    largest_feature_area = max((label["face_area"] for label in feature_labels), default=0.0)
    largest_stock_area = max((label["face_area"] for label in stock_labels), default=0.0)
    checks = {
        "single_solid": len(list(TopologyExplorer(result.shape).solids())) == 1 and step_report["solid_count"] == 1,
        "positive_volume": shape_volume(result.shape) > 0.0 and float(step_report["signed_volume"] or 0.0) > 0.0,
        "brep_valid": BRepCheck_Analyzer(result.shape).IsValid() and bool(step_report["valid"]),
        "step_clean": bool(step_report["readable"]) and not step_report["issues"],
        "supported_not_isolated": not params["isolated_thin_plate"] and params["positive_support_contact_count"] >= 2,
        "all_declared_contacts_positive": all(item["intersection_volume"] > 1e-7 for item in params["support_contacts"]),
        "thin_wall_ratio": params["span_thickness_ratio"] >= 5.0 and params["height_thickness_ratio"] >= 2.0,
        "feature_label_coverage": len(feature_labels) == params["feature_face_count"] and bool(feature_labels),
        "one_thin_wall_instance": {label["instance_id"] for label in feature_labels} == {feature_inst},
        "paired_large_side_faces": feature_side_count >= 2,
        "stock_context_preserved": bool(stock_labels),
        "no_stock_scale_face_as_feature": largest_stock_area > largest_feature_area,
        "cavity_instance_independent": (not cavity_expected) or (
            len(pocket_labels) == params["pocket_face_count"]
            and bool(pocket_labels)
            and {label["instance_id"] for label in pocket_labels} == {1}
            and feature_inst == 2
        ),
        "hole_instance_independent": (not hole_expected) or (
            len(hole_labels) == params["hole_face_count"]
            and bool(hole_labels)
            and {label["instance_id"] for label in hole_labels} == {feature_inst + 1}
        ),
        "transition_kept_in_feature": (not transition_expected) or params["feature_face_count"] >= 5,
        "graph_node_count": len(graph["nodes"]) == len(labels),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "in_memory_volume": shape_volume(result.shape),
        "feature_face_count": len(feature_labels),
        "feature_side_face_count": feature_side_count,
        "stock_face_count": len(stock_labels),
        "pocket_face_count": len(pocket_labels),
        "hole_face_count": len(hole_labels),
        "largest_feature_face_area": largest_feature_area,
        "largest_stock_face_area": largest_stock_area,
        "support_contacts": params["support_contacts"],
        "step_report": step_report,
        "face_identity_protocol": "TopoDS_Shape.IsSame; exact topology traversal injection",
    }


def build_and_write_sample(
    *,
    scenario_key: str,
    seed: int,
    step_path: Path,
    graph_path: Path,
    semantic_index: dict[str, Any],
    ontology_maps: Any,
    primitive_index: dict[str, Any],
) -> dict[str, Any]:
    result = build_rib_web_sample(scenario_key, seed)
    graph, labels = exact_graph(
        result,
        semantic_index=semantic_index,
        ontology_maps=ontology_maps,
        primitive_index=primitive_index,
    )
    graph["p1_b_rib_web_parameters"] = result.parameters
    graph["label_maps"] = ontology_maps.export_label_map()
    graph["label_maps"]["face_semantic_subtype"] = export_layer_label_map(semantic_index)["face_semantic_subtype"]
    graph["generation_protocol"] = {
        "stage": "P1-B2",
        "generator": "p1_b_engineering_rib_web_v3",
        "seed": seed,
        "face_identity": "TopoDS_Shape.IsSame",
    }
    step_path.parent.mkdir(parents=True, exist_ok=True)
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    write_step_with_positive_volume(result.shape, step_path)
    step_report = inspect_step_file(step_path)
    sample_audit = audit_sample(result, step_report=step_report, graph=graph, labels=labels)
    graph["p1_b_automatic_audit"] = sample_audit
    graph_path.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "scenario_key": scenario_key,
        "macro_feature": result.parameters["macro_key"],
        "context_kind": result.parameters["context_kind"],
        "variant": result.parameters["variant"],
        "seed": seed,
        "step_path": str(step_path.resolve()),
        "graph_path": str(graph_path.resolve()),
        "step_sha256": sha256_file(step_path),
        "graph_sha256": sha256_file(graph_path),
        "parameters": result.parameters,
        "audit": sample_audit,
    }


def write_review_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "review_id", "scenario_key", "macro_feature", "context_kind", "variant", "seed",
        "step_path", "step_sha256", "review_result", "review_reason",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, record in enumerate(records, 1):
            writer.writerow({
                "review_id": f"H-B2-{index:02d}",
                "scenario_key": record["scenario_key"],
                "macro_feature": record["macro_feature"],
                "context_kind": record["context_kind"],
                "variant": record["variant"],
                "seed": record["seed"],
                "step_path": record["step_path"],
                "step_sha256": record["step_sha256"],
                "review_result": "",
                "review_reason": "",
            })


GUIDE_LINES = [
    "# P1-B2 rib/web Human Gate H-B2：UG/NX 检查指南",
    "",
    "请在 UG/NX 中逐个打开 `human_review_canonical_12/` 内全部 12 个 STEP。本审查只判断工程几何概念，不需要填写 face ID。",
    "",
    "## 场景覆盖",
    "",
    "- 01–03：L 形底板/立壁连接 rib（普通、带孔、连续过渡）。",
    "- 04–06：矩形腔内跨壁 rib（普通、带孔、连续过渡）。",
    "- 07–09：圆形腔内跨壁 rib（普通、带孔、连续过渡）。",
    "- 10–12：连接两个抬高支承体的 web（普通、带孔、连续过渡）。",
    "",
    "## 五项必检内容",
    "",
    "1. rib/web 是否真实连接至少两个支承区域、腔壁或主体区域。",
    "2. 是否存在孤立、只粘在单一基面上的薄板；若存在则 FAIL。",
    "3. 厚度、跨度和高度比例是否像加强结构，而不是主 stock 或大基面。",
    "4. 腔与 rib 是否同时存在且彼此几何清晰；rib 不应吞掉或堵死腔。",
    "5. 带孔或连续过渡场景中，薄壁主体是否仍然完整连接，未被孔切断或被过渡拆成孤立碎片。",
    "",
    "同时确认模型树中只有一个有效实体，无隐藏的第二实体。",
    "",
    "## 填写 CSV",
    "",
    "打开同级 `human_review_canonical_12.csv`，只填写：",
    "",
    "- `review_result`：`PASS` 或 `FAIL`。",
    "- `review_reason`：PASS 可写“五项均通过”；FAIL 写明场景和简短原因。",
    "",
    "不要修改 seed、路径、哈希或其他列。完成 12 行后保存 CSV，并把结论交回 Codex。",
]


def write_guide(path: Path) -> None:
    path.write_text("\n".join(GUIDE_LINES) + "\n", encoding="utf-8")


def write_summary(path: Path, pilot_records: list[dict[str, Any]], canonical_records: list[dict[str, Any]]) -> None:
    records = pilot_records + canonical_records
    failed = [record for record in records if not record["audit"]["passed"]]
    scenario_counts = Counter(record["scenario_key"] for record in pilot_records)
    context_counts = Counter(record["context_kind"] for record in pilot_records)
    variant_counts = Counter(record["variant"] for record in pilot_records)
    positive_contacts = [
        item["intersection_volume"]
        for record in records
        for item in record["parameters"]["support_contacts"]
    ]
    lines = [
        "# P1-B2 rib/web 自动审计报告",
        "",
        f"- Pilot：{len(pilot_records)} 个（12 场景 × {PILOT_PER_SCENARIO}）",
        f"- Canonical：{len(canonical_records)} 个（12 场景 × 1）",
        f"- 自动审计通过：{len(records) - len(failed)}/{len(records)}",
        f"- 场景覆盖：{dict(sorted(scenario_counts.items()))}",
        f"- 上下文覆盖：{dict(sorted(context_counts.items()))}",
        f"- 变体覆盖：{dict(sorted(variant_counts.items()))}",
        f"- 最小支承交叠体积：{min(positive_contacts):.6f} mm³",
        f"- 最小 span/thickness：{min(record['parameters']['span_thickness_ratio'] for record in records):.6f}",
        f"- 最小 height/thickness：{min(record['parameters']['height_thickness_ratio'] for record in records):.6f}",
        "",
        "## 自动验收项",
        "",
        "每个样本均检查：单 solid、正体积、B-Rep/STEP 有效、至少两个正体积支承接触、薄壁比例、成对大侧面、同一薄壁实例、stock 上下文保留、腔与 rib 独立、孔实例独立、过渡不拆分薄壁。",
        "",
        "## 失败项",
        "",
        "无。" if not failed else "\n".join(f"- {record['step_path']}" for record in failed),
        "",
        "## 结论",
        "",
        "自动验收通过后必须停在 Human Gate H-B2；用户完成 12 个 STEP 的 UG/NX 审查前不得开始 Graph v2。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    pilot_step_dir = output_dir / "pilot_48" / "step"
    pilot_graph_dir = output_dir / "pilot_48" / "graph"
    canonical_step_dir = output_dir / "human_review_canonical_12"
    canonical_graph_dir = output_dir / "human_review_canonical_12_graph"
    output_dir.mkdir(parents=True, exist_ok=True)

    semantic_index = load_layer_index(ROOT / "face_semantic_subtype_ontology.yaml", "face_semantic_subtype")
    ontology_maps = load_ontology_maps(ROOT / "CAE_feature_ontology.yaml")
    primitive_index = load_layer_index(ROOT / "CAE_feature_ontology.yaml", "primitive_surface")
    pilot_records: list[dict[str, Any]] = []
    canonical_records: list[dict[str, Any]] = []
    for scenario_index, scenario_key in enumerate(scenario_keys()):
        for replicate in range(PILOT_PER_SCENARIO):
            seed = PILOT_SEED_BASE + scenario_index * PILOT_PER_SCENARIO + replicate
            stem = f"{scenario_index + 1:02d}_{scenario_key}_r{replicate + 1}_seed{seed}"
            pilot_records.append(build_and_write_sample(
                scenario_key=scenario_key,
                seed=seed,
                step_path=pilot_step_dir / f"{stem}.step",
                graph_path=pilot_graph_dir / f"{stem}.json",
                semantic_index=semantic_index,
                ontology_maps=ontology_maps,
                primitive_index=primitive_index,
            ))
        seed = CANONICAL_SEED_BASE + scenario_index
        stem = f"{scenario_index + 1:02d}_{scenario_key}_seed{seed}"
        canonical_records.append(build_and_write_sample(
            scenario_key=scenario_key,
            seed=seed,
            step_path=canonical_step_dir / f"{stem}.step",
            graph_path=canonical_graph_dir / f"{stem}.json",
            semantic_index=semantic_index,
            ontology_maps=ontology_maps,
            primitive_index=primitive_index,
        ))

    records = pilot_records + canonical_records
    payload = {
        "schema_version": "p1_b2_rib_web_pilot_audit_v1",
        "protocol": {
            "pilot_count": 48,
            "canonical_count": 12,
            "scenario_count": 12,
            "pilot_per_scenario": PILOT_PER_SCENARIO,
            "pilot_seed_base": PILOT_SEED_BASE,
            "canonical_seed_base": CANONICAL_SEED_BASE,
            "scenario_specs": SCENARIO_SPECS,
            "face_identity": "TopoDS_Shape.IsSame",
        },
        "automatic_pass": all(record["audit"]["passed"] for record in records),
        "pilot_records": pilot_records,
        "canonical_records": canonical_records,
    }
    (output_dir / "automatic_audit_60.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_review_csv(output_dir / "human_review_canonical_12.csv", canonical_records)
    write_guide(output_dir / "H-B2_UG_NX_检查指南.zh-CN.md")
    write_summary(output_dir / "automatic_audit_summary.zh-CN.md", pilot_records, canonical_records)
    manifest = {
        "schema_version": "p1_b2_human_gate_manifest_v1",
        "automatic_pass": payload["automatic_pass"],
        "files": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in sorted(output_dir.rglob("*"))
            if path.is_file() and path.name != "manifest.json"
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if not payload["automatic_pass"]:
        failed_checks = Counter(
            key for record in records for key, value in record["audit"]["checks"].items() if not value
        )
        raise RuntimeError(f"P1-B2 automatic audit failed: {dict(failed_checks)}")
    print(json.dumps({
        "automatic_pass": True,
        "pilot_count": len(pilot_records),
        "canonical_count": len(canonical_records),
        "output_dir": str(output_dir),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
