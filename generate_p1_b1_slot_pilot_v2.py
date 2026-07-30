from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.Extend.TopologyUtils import TopologyExplorer

import features.p1_b_continuous_slot as slot_core
import generate_p1_b1_slot_pilot as protocol
from core.AAGExtractor import extract_aag_graph
from features.p1_b_continuous_slot import PROFILE_TO_MACRO, P1BContinuousSlotGenerator, SlotBuildResult, all_slot_combinations
from utils.ontology import export_layer_label_map, load_layer_index, load_ontology_maps
from utils.step_solid_validation import inspect_step_file, write_step_with_positive_volume


def rebuild_with_identity_roles(raw_result: SlotBuildResult, stock: Any) -> SlotBuildResult:
    cut = BRepAlgoAPI_Cut(stock, raw_result.tool)
    cut.Build()
    if not cut.IsDone():
        raise ValueError("identity-safe slot cut reconstruction failed")
    shape = cut.Shape()
    parameters = dict(raw_result.parameters)
    role_items = []
    for tool_face in TopologyExplorer(raw_result.tool).faces():
        role = slot_core._tool_face_role(tool_face, parameters, protocol.H)
        if role is None:
            continue
        for modified in cut.Modified(tool_face):
            if modified.ShapeType() != 4:
                continue
            if not any(modified.IsSame(existing) for existing, _ in role_items):
                role_items.append((modified, role))
    parameters["_face_role_items"] = role_items
    parameters["_face_role_by_hash"] = {hash(face): role for face, role in role_items}
    parameters["_feature_face_hashes"] = sorted({hash(face) for face, _ in role_items})
    parameters["feature_face_count"] = len(role_items)
    parameters["terminal_face_count"] = sum(role == slot_core.TERMINAL_ROLE for _, role in role_items)
    return SlotBuildResult(shape, raw_result.tool, raw_result.guide_wire, raw_result.profile_wire, parameters)


def exact_face_labels(shape: Any, parameters: dict[str, Any], generator: P1BContinuousSlotGenerator, ontology_maps: Any) -> list[dict[str, Any]]:
    role_items = parameters["_face_role_items"]
    macro = ontology_maps.resolve_macro_feature(str(parameters["macro_key"]))
    labels = []
    for face in TopologyExplorer(shape).faces():
        role = next((candidate_role for candidate, candidate_role in role_items if face.IsSame(candidate)), None)
        if role is None:
            labels.append({
                "semantic_label": generator.semantic_map["stock_face"],
                "instance_id": 0,
                "macro_feature": "stock",
                "macro_feature_id": ontology_maps.stock_id,
            })
        else:
            labels.append({
                "semantic_label": generator.semantic_map[role],
                "instance_id": int(parameters["inst"]),
                "macro_feature": macro.key,
                "macro_feature_id": macro.id,
                "semantic_role": role,
            })
    return labels


def exact_graph(shape: Any, parameters: dict[str, Any], generator: P1BContinuousSlotGenerator, ontology_maps: Any, primitive_index: dict[str, Any]) -> dict[str, Any]:
    labels = exact_face_labels(shape, parameters, generator, ontology_maps)
    graph = extract_aag_graph(shape, primitive_surface_index=primitive_index, include_label_fields=False)
    if len(labels) != len(graph["nodes"]):
        raise RuntimeError("topology traversal order changed during graph extraction")
    for node, label in zip(graph["nodes"], labels):
        node.update(label)
    graph["instances"] = {
        "0": [node["idx"] for node in graph["nodes"] if node["instance_id"] == 0],
        "1": [node["idx"] for node in graph["nodes"] if node["instance_id"] == 1],
    }
    return graph


def exact_audit(result: SlotBuildResult, step_report: dict[str, Any], graph: dict[str, Any]) -> dict[str, Any]:
    parameters = result.parameters
    feature_nodes = [node for node in graph["nodes"] if node["instance_id"] == 1]
    stock_nodes = [node for node in graph["nodes"] if node["instance_id"] == 0]
    expected_terminal = 1 if parameters["end_condition"] == "open_closed" else 0
    boundary = protocol.boundary_contract(parameters)
    guide = parameters["guide_audit"]
    checks = {
        "single_solid": len(list(TopologyExplorer(result.shape).solids())) == 1 and step_report["solid_count"] == 1,
        "positive_volume": protocol.volume(result.shape) > 0.0 and float(step_report["signed_volume"] or 0.0) > 0.0,
        "brep_valid": bool(step_report["valid"]),
        "step_clean": bool(step_report["readable"]) and not step_report["issues"],
        "guide_single_edge": bool(guide["single_edge"]),
        "guide_analytic": bool(guide["analytic_curve_type_ok"]),
        "guide_c1": bool(guide["at_least_c1"]),
        "guide_not_polyline": not bool(guide["polyline"]),
        "length_width_ratio": float(parameters["path_length_profile_width_ratio"]) >= 4.0,
        "end_condition": all(boundary.values()) and int(parameters["terminal_face_count"]) == expected_terminal,
        "feature_faces_exist": bool(parameters["_face_role_items"]),
        "feature_label_coverage": len(feature_nodes) == len(parameters["_face_role_items"]),
        "no_feature_face_as_stock": len(feature_nodes) + len(stock_nodes) == len(graph["nodes"]),
        "single_feature_instance": {node["instance_id"] for node in feature_nodes} == {1},
        "no_fixed_segment_peak": parameters["guide_family"] == "straight" or int(parameters["feature_face_count"]) not in protocol.FORBIDDEN_SEGMENT_COUNTS,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "boundary_contract": boundary,
        "in_memory_volume": protocol.volume(result.shape),
        "feature_face_count": int(parameters["feature_face_count"]),
        "terminal_face_count": int(parameters["terminal_face_count"]),
        "stock_face_count": len(stock_nodes),
        "graph_feature_face_count": len(feature_nodes),
        "guide_audit": guide,
        "step_report": step_report,
        "face_identity_protocol": "TopoDS_Shape.IsSame; graph labels injected by exact topology traversal order",
    }


def build_and_write_sample(
    *, generator: P1BContinuousSlotGenerator, ontology_maps: Any, primitive_index: dict[str, Any], semantic_index: dict[str, Any],
    profile_shape: str, guide_family: str, end_condition: str, seed: int, step_path: Path, graph_path: Path,
) -> dict[str, Any]:
    stock = protocol.make_stock()
    raw_result = generator.build_one(
        stock, protocol.L, protocol.W, protocol.H,
        profile_shape=profile_shape, guide_family=guide_family, end_condition=end_condition, seed=seed, instance_id=1,
    )
    result = rebuild_with_identity_roles(raw_result, stock)
    graph = exact_graph(result.shape, result.parameters, generator, ontology_maps, primitive_index)
    graph["p1_b_slot_parameters"] = protocol.serializable_parameters(result.parameters)
    graph["label_maps"] = ontology_maps.export_label_map()
    graph["label_maps"]["face_semantic_subtype"] = export_layer_label_map(semantic_index)["face_semantic_subtype"]
    graph["generation_protocol"] = {
        "stage": "P1-B1", "generator": "p1_b_continuous_slot",
        "stock_dimensions_mm": [protocol.L, protocol.W, protocol.H], "seed": seed,
        "face_identity": "TopoDS_Shape.IsSame",
    }
    step_path.parent.mkdir(parents=True, exist_ok=True)
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    write_step_with_positive_volume(result.shape, step_path)
    step_report = inspect_step_file(step_path)
    audit = exact_audit(result, step_report, graph)
    graph["p1_b_automatic_audit"] = audit
    graph_path.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "profile_shape": profile_shape, "macro_feature": PROFILE_TO_MACRO[profile_shape],
        "guide_family": guide_family, "end_condition": end_condition, "seed": seed,
        "step_path": str(step_path.resolve()), "graph_path": str(graph_path.resolve()),
        "step_sha256": protocol.sha256_file(step_path), "graph_sha256": protocol.sha256_file(graph_path),
        "parameters": protocol.serializable_parameters(result.parameters), "audit": audit,
    }


def main() -> int:
    args = protocol.parse_args()
    output_dir = args.output_dir.resolve()
    pilot_step_dir = output_dir / "pilot_80" / "step"
    pilot_graph_dir = output_dir / "pilot_80" / "graph"
    canonical_dir = output_dir / "human_review_canonical_20"
    canonical_graph_dir = output_dir / "human_review_canonical_20_graph"
    output_dir.mkdir(parents=True, exist_ok=True)
    semantic_index = protocol.load_p1_b_semantic_index()
    ontology_maps = load_ontology_maps(protocol.ROOT / "CAE_feature_ontology.yaml")
    primitive_index = load_layer_index(protocol.ROOT / "CAE_feature_ontology.yaml", "primitive_surface")
    generator = P1BContinuousSlotGenerator(semantic_index)
    pilot_records, canonical_records = [], []
    for combination_index, (profile_shape, guide_family, end_condition) in enumerate(all_slot_combinations()):
        stem = f"{combination_index + 1:02d}_{PROFILE_TO_MACRO[profile_shape]}_{guide_family}_{end_condition}"
        for replicate in range(protocol.REQUIRED_PILOT_PER_COMBINATION):
            seed = protocol.PILOT_SEED_BASE + combination_index * protocol.REQUIRED_PILOT_PER_COMBINATION + replicate
            sample_stem = f"{stem}_r{replicate + 1}_seed{seed}"
            pilot_records.append(build_and_write_sample(
                generator=generator, ontology_maps=ontology_maps, primitive_index=primitive_index, semantic_index=semantic_index,
                profile_shape=profile_shape, guide_family=guide_family, end_condition=end_condition, seed=seed,
                step_path=pilot_step_dir / f"{sample_stem}.step", graph_path=pilot_graph_dir / f"{sample_stem}.json",
            ))
        seed = protocol.CANONICAL_SEED_BASE + combination_index
        canonical_records.append(build_and_write_sample(
            generator=generator, ontology_maps=ontology_maps, primitive_index=primitive_index, semantic_index=semantic_index,
            profile_shape=profile_shape, guide_family=guide_family, end_condition=end_condition, seed=seed,
            step_path=canonical_dir / f"{stem}_seed{seed}.step", graph_path=canonical_graph_dir / f"{stem}_seed{seed}.json",
        ))
    records = pilot_records + canonical_records
    payload = {
        "schema_version": "p1_b1_slot_pilot_audit_v2",
        "protocol": {
            "pilot_count": 80, "canonical_count": 20,
            "pilot_per_combination": protocol.REQUIRED_PILOT_PER_COMBINATION,
            "combination_count": len(all_slot_combinations()),
            "pilot_seed_base": protocol.PILOT_SEED_BASE, "canonical_seed_base": protocol.CANONICAL_SEED_BASE,
            "forbidden_fixed_segment_counts": sorted(protocol.FORBIDDEN_SEGMENT_COUNTS),
            "face_identity": "TopoDS_Shape.IsSame",
        },
        "automatic_pass": all(record["audit"]["passed"] for record in records),
        "pilot_records": pilot_records, "canonical_records": canonical_records,
    }
    audit_path = output_dir / "automatic_audit_100.json"
    audit_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    protocol.write_review_csv(output_dir / "human_review_canonical_20.csv", canonical_records)
    protocol.write_nx_guide(output_dir / "H-B1_UG_NX_检查指南.zh-CN.md")
    protocol.write_summary(output_dir / "automatic_audit_summary.zh-CN.md", pilot_records, canonical_records)
    manifest = {
        "schema_version": "p1_b1_human_gate_manifest_v2", "automatic_pass": payload["automatic_pass"],
        "files": [{"path": str(path.resolve()), "sha256": protocol.sha256_file(path)} for path in sorted(output_dir.rglob("*")) if path.is_file() and path.name != "manifest.json"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if not payload["automatic_pass"]:
        failed_checks = Counter(key for record in records for key, value in record["audit"]["checks"].items() if not value)
        raise RuntimeError(f"P1-B1 automatic audit failed: {dict(failed_checks)}")
    print(json.dumps({"automatic_pass": True, "pilot_count": 80, "canonical_count": 20, "output_dir": str(output_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
