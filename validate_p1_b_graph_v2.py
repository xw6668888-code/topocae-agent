from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_Transform
from OCC.Core.BRepFeat import BRepFeat_SplitShape
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.GeomAbs import GeomAbs_Plane
from OCC.Core.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Trsf
from OCC.Extend.TopologyUtils import TopologyExplorer

from core.graph_v2_1 import canonical_graph_signature, extract_graph_v2
from features.p1_b_continuous_slot import P1BContinuousSlotGenerator, all_slot_combinations
from features.p1_b_rib_web_v3 import build_rib_web_sample, scenario_keys
from generate_p1_b1_slot_pilot import CANONICAL_SEED_BASE as SLOT_SEED_BASE
from generate_p1_b1_slot_pilot import L, W, H, load_p1_b_semantic_index, make_stock
from generate_p1_b1_slot_pilot_v2 import rebuild_with_identity_roles
from generate_p1_b2_rib_web_pilot import CANONICAL_SEED_BASE as RIB_WEB_SEED_BASE
from utils.graph_v2_validation import compare_graph_v2_invariance, compare_partition_invariance
from utils.ontology import load_layer_index, load_ontology_maps
from utils.training_export_v2_1 import export_graph_v2_training_record, fit_training_quantile_transform_from_graphs


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "reports" / "p1_b_graph_v2"


def slot_graphs(semantic_index: dict[str, Any], ontology_maps: Any, primitive_index: dict[str, Any]) -> list[dict[str, Any]]:
    generator = P1BContinuousSlotGenerator(semantic_index)
    records = []
    for index, (profile_shape, guide_family, end_condition) in enumerate(all_slot_combinations()):
        stock = make_stock()
        raw_result = generator.build_one(
            stock, L, W, H,
            profile_shape=profile_shape,
            guide_family=guide_family,
            end_condition=end_condition,
            seed=SLOT_SEED_BASE + index,
            instance_id=1,
        )
        result = rebuild_with_identity_roles(raw_result, stock)
        labels = []
        macro = ontology_maps.resolve_macro_feature(result.parameters["macro_key"])
        for face in TopologyExplorer(result.shape).faces():
            role = next((candidate_role for candidate_face, candidate_role in result.parameters["_face_role_items"] if face.IsSame(candidate_face)), None)
            if role is None:
                labels.append(stock_label())
            else:
                semantic_key = semantic_index["alias_to_key"][role]
                labels.append({
                    "semantic_label": semantic_index["key_to_id"][semantic_key],
                    "instance_id": 1,
                    "macro_feature": macro.key,
                    "macro_feature_id": macro.id,
                    "semantic_role": role,
                })
        graph = extract_graph_v2(result.shape, face_labels=labels, primitive_surface_index=primitive_index)
        records.append({
            "sample_id": f"slot_{index + 1:02d}_{profile_shape}_{guide_family}_{end_condition}",
            "family": "slot",
            "shape": result.shape,
            "graph": graph,
        })
    return records


def rib_web_graphs(semantic_index: dict[str, Any], ontology_maps: Any, primitive_index: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for index, scenario_key in enumerate(scenario_keys()):
        result = build_rib_web_sample(scenario_key, RIB_WEB_SEED_BASE + index)
        labels = []
        for face in TopologyExplorer(result.shape).faces():
            match = next((tag for tag in result.face_tags if face.IsSame(tag.face)), None)
            if match is None:
                labels.append(stock_label())
            else:
                macro = ontology_maps.resolve_macro_feature(match.macro_key)
                semantic_key = semantic_index["alias_to_key"][match.semantic_role]
                labels.append({
                    "semantic_label": semantic_index["key_to_id"][semantic_key],
                    "instance_id": match.instance_id,
                    "macro_feature": macro.key,
                    "macro_feature_id": macro.id,
                    "semantic_role": match.semantic_role,
                })
        graph = extract_graph_v2(result.shape, face_labels=labels, primitive_surface_index=primitive_index)
        records.append({
            "sample_id": f"rib_web_{index + 1:02d}_{scenario_key}",
            "family": "rib_web",
            "shape": result.shape,
            "graph": graph,
        })
    return records


def stock_label() -> dict[str, Any]:
    return {"semantic_label": 0, "instance_id": 0, "macro_feature": "stock", "macro_feature_id": 0, "semantic_role": "stock_face"}


def rotate_shape(shape: Any) -> Any:
    transform = gp_Trsf()
    transform.SetRotation(gp_Ax1(gp_Pnt(0.0, 0.0, 0.0), gp_Dir(1.0, 2.0, 3.0)), 0.73)
    return BRepBuilderAPI_Transform(shape, transform, True).Shape()


def scale_shape(shape: Any, factor: float = 3.7) -> Any:
    transform = gp_Trsf()
    transform.SetScale(gp_Pnt(0.0, 0.0, 0.0), factor)
    return BRepBuilderAPI_Transform(shape, transform, True).Shape()


def split_box_pair() -> tuple[Any, Any]:
    shape = BRepPrimAPI_MakeBox(gp_Pnt(-10.0, -8.0, 0.0), 20.0, 16.0, 10.0).Shape()
    top_face = None
    for face in TopologyExplorer(shape).faces():
        surface = BRepAdaptor_Surface(face)
        if surface.GetType() == GeomAbs_Plane:
            plane = surface.Plane()
            if abs(plane.Axis().Direction().Z()) > 0.9 and abs(plane.Location().Z() - 10.0) < 0.1:
                top_face = face
                break
    if top_face is None:
        raise RuntimeError("split validation top face not found")
    splitter = BRepFeat_SplitShape(shape)
    splitter.Add(BRepBuilderAPI_MakeEdge(gp_Pnt(-10.0, 0.0, 10.0), gp_Pnt(10.0, 0.0, 10.0)).Edge(), top_face)
    splitter.Build()
    return shape, splitter.Shape()


def graph_audit(record: dict[str, Any]) -> dict[str, Any]:
    graph = record["graph"]
    node_values = [
        float(value)
        for node in graph["nodes"]
        for value in node["attributes"]["recognition_features_v2"].values()
    ]
    edge_values = [
        float(value)
        for edge in graph["edge_attributes"]
        for value in edge["recognition_features_v2"].values()
    ]
    all_face_indices = {int(node["idx"]) for node in graph["nodes"]}
    patch_face_indices = [face for patch in graph["canonical_patches"] for face in patch["face_indices"]]
    patch_label_safe = all(len(patch["instance_ids"]) == 1 and len(patch["macro_features"]) == 1 for patch in graph["canonical_patches"])
    checks = {
        "finite_node_features": all(math.isfinite(value) and abs(value) <= 1e4 for value in node_values),
        "finite_edge_features": all(math.isfinite(value) and abs(value) <= 1e4 for value in edge_values),
        "all_faces_mapped_once": len(patch_face_indices) == len(set(patch_face_indices)) and set(patch_face_indices) == all_face_indices,
        "patches_respect_labels": patch_label_safe,
        "raw_physical_preserved": graph["part_metadata_v2"]["raw_physical"]["total_surface_area"] > 0.0,
        "directed_edges_aligned": len(graph["edges"]) == len(graph["edge_attributes"]),
    }
    return {
        "sample_id": record["sample_id"],
        "family": record["family"],
        "passed": all(checks.values()),
        "checks": checks,
        "face_count": len(graph["nodes"]),
        "edge_count": len(graph["edges"]),
        "patch_count": len(graph["canonical_patches"]),
        "merged_patch_count": sum(patch["face_count"] > 1 for patch in graph["canonical_patches"]),
        "max_abs_node_feature": max((abs(value) for value in node_values), default=0.0),
        "max_abs_edge_feature": max((abs(value) for value in edge_values), default=0.0),
    }


def invariance_audits(records: list[dict[str, Any]], primitive_index: dict[str, Any]) -> list[dict[str, Any]]:
    selected_indices = [0, 2, 4, 6, 9, 14, 19, 20, 24, 28, 30, 31]
    audits = []
    for record_index in selected_indices:
        record = records[record_index]
        reference = extract_graph_v2(record["shape"], primitive_surface_index=primitive_index)
        rotation_graph = extract_graph_v2(rotate_shape(record["shape"]), primitive_surface_index=primitive_index)
        scale_graph = extract_graph_v2(scale_shape(record["shape"]), primitive_surface_index=primitive_index)
        rotation = compare_graph_v2_invariance(reference, rotation_graph)
        scale = compare_graph_v2_invariance(reference, scale_graph)
        audits.append({
            "sample_id": record["sample_id"],
            "rotation": rotation,
            "uniform_scale": scale,
            "passed": rotation["score"] >= 0.98 and scale["score"] >= 0.98,
        })
    return audits


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    semantic_index = load_p1_b_semantic_index()
    ontology_maps = load_ontology_maps(ROOT / "CAE_feature_ontology.yaml")
    primitive_index = load_layer_index(ROOT / "CAE_feature_ontology.yaml", "primitive_surface")
    records = slot_graphs(semantic_index, ontology_maps, primitive_index) + rib_web_graphs(semantic_index, ontology_maps, primitive_index)
    graph_audits = [graph_audit(record) for record in records]
    invariance = invariance_audits(records, primitive_index)
    reference_shape, partitioned_shape = split_box_pair()
    partition = compare_partition_invariance(
        extract_graph_v2(reference_shape, primitive_surface_index=primitive_index),
        extract_graph_v2(partitioned_shape, primitive_surface_index=primitive_index),
    )
    quantile_transform = fit_training_quantile_transform_from_graphs(
        [record["graph"] for record in records],
        split_name="train",
        lower_quantile=0.005,
        upper_quantile=0.995,
    )
    training_records = [export_graph_v2_training_record(record["graph"], quantile_transform=quantile_transform) for record in records]
    model_values = [
        float(value)
        for record in training_records
        for row in record["node_features"] + record["edge_features"]
        for value in row
    ]
    automatic_pass = (
        all(item["passed"] for item in graph_audits)
        and all(item["passed"] for item in invariance)
        and partition["score"] >= 0.98
        and all(math.isfinite(value) and abs(value) <= 32.0 for value in model_values)
    )
    payload = {
        "schema_version": "p1_b_graph_v2_validation_1",
        "automatic_pass": automatic_pass,
        "graph_count": len(records),
        "slot_graph_count": sum(record["family"] == "slot" for record in records),
        "rib_web_graph_count": sum(record["family"] == "rib_web" for record in records),
        "graph_audits": graph_audits,
        "invariance_audits": invariance,
        "partition_audit": partition,
        "quantile_transform": quantile_transform.to_dict(),
        "training_feature_audit": {
            "value_count": len(model_values),
            "all_finite": all(math.isfinite(value) for value in model_values),
            "max_abs": max((abs(value) for value in model_values), default=0.0),
            "fit_split": quantile_transform.fitted_split,
        },
    }
    json_path = OUTPUT_DIR / "automatic_validation.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    rotation_scores = [item["rotation"]["score"] for item in invariance]
    scale_scores = [item["uniform_scale"]["score"] for item in invariance]
    summary_lines = [
        "# P1-B Graph v2 自动验证报告",
        "",
        f"- 自动结论：{'PASS' if automatic_pass else 'FAIL'}",
        f"- Canonical 工程图：{len(records)}（槽 20，rib/web 12）",
        f"- 图字段/finite/patch 标签审计：{sum(item['passed'] for item in graph_audits)}/{len(graph_audits)}",
        f"- 任意轴旋转不变性：min={min(rotation_scores):.6f}, mean={sum(rotation_scores)/len(rotation_scores):.6f}",
        f"- 统一尺度不变性：min={min(scale_scores):.6f}, mean={sum(scale_scores)/len(scale_scores):.6f}",
        f"- 合法 face partition canonical signature：{partition['score']:.6f}",
        f"- 原始 face/partition face：{partition['reference_face_count']}/{partition['partitioned_face_count']}",
        f"- 原始 patch/partition patch：{partition['reference_patch_count']}/{partition['partitioned_patch_count']}",
        f"- 训练向量：{len(model_values)} values，finite={payload['training_feature_audit']['all_finite']}，max_abs={payload['training_feature_audit']['max_abs']:.6f}",
        "- 分位数边界只用声明为 train 的 32 个合成 canonical 图拟合；未使用验证集或真实件。",
        "- 原始面积、长度、半径、体积保留在 graph/object metadata；模型输入使用相对量与 log1p/训练分位数裁剪。",
        "- canonical patch 非破坏性保留原始 face ID，且不跨锐边、不同 instance 或 macro。",
    ]
    summary_path = OUTPUT_DIR / "automatic_validation_summary.zh-CN.md"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "p1_b_graph_v2_validation_manifest_1",
        "automatic_pass": automatic_pass,
        "files": [
            {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in (json_path, summary_path)
        ],
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "automatic_pass": automatic_pass,
        "graph_count": len(records),
        "rotation_min": min(rotation_scores),
        "scale_min": min(scale_scores),
        "partition_score": partition["score"],
        "output_dir": str(OUTPUT_DIR.resolve()),
    }, ensure_ascii=False))
    return 0 if automatic_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
