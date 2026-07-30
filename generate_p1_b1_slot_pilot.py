from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.GProp import GProp_GProps
from OCC.Core.gp import gp_Pnt
from OCC.Extend.TopologyUtils import TopologyExplorer

from core.AAGExtractor import extract_aag_graph
from features.p1_b_continuous_slot import (
    PROFILE_TO_MACRO,
    P1BContinuousSlotGenerator,
    all_slot_combinations,
)
from utils.ontology import export_layer_label_map, load_layer_index, load_ontology_maps
from utils.step_solid_validation import inspect_step_file, write_step_with_positive_volume


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "output_generation_p1_b1_slot_pilot_80"
PILOT_SEED_BASE = 202607180100
CANONICAL_SEED_BASE = 202607181000
L, W, H = 100.0, 80.0, 25.0
REQUIRED_PILOT_PER_COMBINATION = 4
FORBIDDEN_SEGMENT_COUNTS = {16, 32, 48}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and audit the P1-B1 continuous-slot pilot and H-B1 package.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_p1_b_semantic_index() -> dict[str, Any]:
    index = load_layer_index(ROOT / "face_semantic_subtype_ontology.yaml", "face_semantic_subtype")
    overlay_path = ROOT / "configs" / "p1_b" / "slot_semantic_overlay.json"
    overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
    for label in overlay["labels"]:
        label_id = int(label["id"])
        key = str(label["key"])
        if key in index["key_to_id"] and index["key_to_id"][key] != label_id:
            raise ValueError(f"semantic key collision for {key}")
        if str(label_id) in index["id_to_key"] and index["id_to_key"][str(label_id)] != key:
            raise ValueError(f"semantic id collision for {label_id}")
        index["id_to_key"][str(label_id)] = key
        index["key_to_id"][key] = label_id
        index["alias_to_key"][key] = key
        for alias in label.get("current_code_aliases", []):
            index["alias_to_key"][str(alias)] = key
        if bool(label.get("enabled", True)) and key not in index["enabled_keys"]:
            index["enabled_keys"].append(key)
    index["ontology_path"] = f"{index['ontology_path']} + {overlay_path}"
    return index


def make_stock():
    return BRepPrimAPI_MakeBox(gp_Pnt(-L / 2.0, -W / 2.0, 0.0), L, W, H).Shape()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def volume(shape: Any) -> float:
    props = GProp_GProps()
    brepgprop.VolumeProperties(shape, props)
    return float(props.Mass())


def build_face_infos(shape: Any, parameters: dict[str, Any], generator: P1BContinuousSlotGenerator, ontology_maps: Any) -> dict[int, dict[str, Any]]:
    feature_roles = parameters["_face_role_by_hash"]
    macro_key = str(parameters["macro_key"])
    macro = ontology_maps.resolve_macro_feature(macro_key)
    face_infos = {}
    for face in TopologyExplorer(shape).faces():
        face_hash = hash(face)
        role = feature_roles.get(face_hash)
        if role is None:
            face_infos[face_hash] = {
                "semantic_label": generator.semantic_map["stock_face"],
                "instance_id": 0,
                "macro_feature": "stock",
                "macro_feature_id": ontology_maps.stock_id,
                "raw_macro_feature": "stock",
            }
        else:
            face_infos[face_hash] = {
                "semantic_label": generator.semantic_map[role],
                "instance_id": int(parameters["inst"]),
                "macro_feature": macro.key,
                "macro_feature_id": macro.id,
                "raw_macro_feature": macro.raw_key,
                "semantic_role": role,
            }
    return face_infos


def boundary_contract(parameters: dict[str, Any]) -> dict[str, bool]:
    half_u, half_v = L / 2.0, W / 2.0
    tol = 1e-7

    def on_boundary(point: list[float] | tuple[float, float]) -> bool:
        return math.isclose(abs(float(point[0])), half_u, abs_tol=tol) or math.isclose(abs(float(point[1])), half_v, abs_tol=tol)

    def inside(point: list[float] | tuple[float, float], margin: float = 0.0) -> bool:
        return abs(float(point[0])) <= half_u - margin and abs(float(point[1])) <= half_v - margin

    end_condition = parameters["end_condition"]
    start = parameters["visible_start_xy"]
    end = parameters["visible_end_xy"]
    if end_condition == "open_open":
        return {"start_on_boundary": on_boundary(start), "end_on_boundary": on_boundary(end), "closed_inside": True}
    if end_condition == "open_closed":
        return {"start_on_boundary": on_boundary(start), "end_on_boundary": not on_boundary(end) and inside(end), "closed_inside": True}
    definition = parameters["guide_definition"]
    center = definition["circle_center_xy"]
    extent = float(definition["circle_radius"]) + float(parameters["profile_width"]) / 2.0
    extrema = ((center[0] + extent, center[1]), (center[0] - extent, center[1]), (center[0], center[1] + extent), (center[0], center[1] - extent))
    return {"start_on_boundary": True, "end_on_boundary": True, "closed_inside": all(inside(point, tol) for point in extrema)}


def audit_sample(result: Any, step_report: dict[str, Any], graph: dict[str, Any]) -> dict[str, Any]:
    parameters = result.parameters
    feature_roles = parameters["_face_role_by_hash"]
    graph_feature_nodes = [node for node in graph["nodes"] if int(node.get("instance_id", 0)) == int(parameters["inst"])]
    graph_stock_nodes = [node for node in graph["nodes"] if int(node.get("instance_id", 0)) == 0]
    expected_terminal = 1 if parameters["end_condition"] == "open_closed" else 0
    boundary = boundary_contract(parameters)
    checks = {
        "single_solid": len(list(TopologyExplorer(result.shape).solids())) == 1 and step_report["solid_count"] == 1,
        "positive_volume": volume(result.shape) > 0.0 and float(step_report["signed_volume"] or 0.0) > 0.0,
        "brep_valid": BRepCheck_Analyzer(result.shape).IsValid() and bool(step_report["valid"]),
        "step_clean": bool(step_report["readable"]) and not step_report["issues"],
        "guide_single_edge": bool(parameters["guide_audit"]["single_edge"]),
        "guide_analytic": bool(parameters["guide_audit"]["analytic_curve_type_ok"]),
        "guide_c1": bool(parameters["guide_audit"]["at_least_c1"]),
        "guide_not_polyline": not bool(parameters["guide_audit"]["polyline"]),
        "length_width_ratio": float(parameters["path_length_profile_width_ratio"]) >= 4.0,
        "end_condition": all(boundary.values()) and int(parameters["terminal_face_count"]) == expected_terminal,
        "feature_faces_exist": bool(feature_roles),
        "feature_label_coverage": len(graph_feature_nodes) == len(feature_roles),
        "no_feature_face_as_stock": not ({node["idx"] for node in graph_feature_nodes} & {node["idx"] for node in graph_stock_nodes}),
        "single_feature_instance": {int(node["instance_id"]) for node in graph_feature_nodes} == {1},
        "no_fixed_segment_peak": parameters["guide_family"] == "straight" or int(parameters["feature_face_count"]) not in FORBIDDEN_SEGMENT_COUNTS,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "boundary_contract": boundary,
        "in_memory_volume": volume(result.shape),
        "feature_face_count": int(parameters["feature_face_count"]),
        "terminal_face_count": int(parameters["terminal_face_count"]),
        "stock_face_count": len(graph_stock_nodes),
        "graph_feature_face_count": len(graph_feature_nodes),
        "guide_audit": parameters["guide_audit"],
        "step_report": step_report,
    }


def serializable_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in parameters.items() if not key.startswith("_")}


def build_and_write_sample(
    *,
    generator: P1BContinuousSlotGenerator,
    ontology_maps: Any,
    primitive_index: dict[str, Any],
    semantic_index: dict[str, Any],
    profile_shape: str,
    guide_family: str,
    end_condition: str,
    seed: int,
    step_path: Path,
    graph_path: Path,
) -> dict[str, Any]:
    result = generator.build_one(
        make_stock(), L, W, H,
        profile_shape=profile_shape,
        guide_family=guide_family,
        end_condition=end_condition,
        seed=seed,
        instance_id=1,
    )
    face_infos = build_face_infos(result.shape, result.parameters, generator, ontology_maps)
    graph = extract_aag_graph(result.shape, face_infos, primitive_surface_index=primitive_index)
    graph["p1_b_slot_parameters"] = serializable_parameters(result.parameters)
    graph["label_maps"] = ontology_maps.export_label_map()
    graph["label_maps"]["face_semantic_subtype"] = export_layer_label_map(semantic_index)["face_semantic_subtype"]
    graph["generation_protocol"] = {
        "stage": "P1-B1",
        "generator": "p1_b_continuous_slot",
        "stock_dimensions_mm": [L, W, H],
        "seed": seed,
    }
    step_path.parent.mkdir(parents=True, exist_ok=True)
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    write_step_with_positive_volume(result.shape, step_path)
    step_report = inspect_step_file(step_path)
    sample_audit = audit_sample(result, step_report, graph)
    graph["p1_b_automatic_audit"] = sample_audit
    graph_path.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "profile_shape": profile_shape,
        "macro_feature": PROFILE_TO_MACRO[profile_shape],
        "guide_family": guide_family,
        "end_condition": end_condition,
        "seed": seed,
        "step_path": str(step_path.resolve()),
        "graph_path": str(graph_path.resolve()),
        "step_sha256": sha256_file(step_path),
        "graph_sha256": sha256_file(graph_path),
        "parameters": serializable_parameters(result.parameters),
        "audit": sample_audit,
    }


def write_review_csv(path: Path, canonical_records: list[dict[str, Any]]) -> None:
    fields = [
        "review_id", "profile_shape", "macro_feature", "guide_family", "end_condition",
        "guide_curve_kind", "seed", "step_path", "step_sha256", "review_result", "review_reason",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for review_id, record in enumerate(canonical_records, 1):
            writer.writerow({
                "review_id": f"H-B1-{review_id:02d}",
                "profile_shape": record["profile_shape"],
                "macro_feature": record["macro_feature"],
                "guide_family": record["guide_family"],
                "end_condition": record["end_condition"],
                "guide_curve_kind": record["parameters"]["guide_curve_kind"],
                "seed": record["seed"],
                "step_path": record["step_path"],
                "step_sha256": record["step_sha256"],
                "review_result": "",
                "review_reason": "",
            })


def write_nx_guide(path: Path) -> None:
    path.write_text(
        """# P1-B1 槽几何 Human Gate H-B1：UG/NX 检查指南\n\n"
        "本审查只判断几何概念是否正确，不需要填写 face ID。请打开 `human_review_canonical_20/` 内全部 20 个 STEP。\n\n"
        "## 建议操作\n\n"
        "1. 在 NX 中逐个导入 STEP，确认 Part Navigator/模型树中只有一个有效实体。\n"
        "2. 使用剖切、测量或旋转视图检查截面与槽端。必要时隐藏基准或打开边显示。\n"
        "3. 对每个 STEP 检查下列七项；任一项不满足即填 FAIL。\n\n"
        "## 七项必检内容\n\n"
        "1. 模型树是否只有一个有效实体。\n"
        "2. 截面是否分别为矩形、U、V、O 的精确解析形状，而不是折线近似。\n"
        "3. 弯槽 guide 是否视觉连续、没有折角和分段面带。\n"
        "4. `open_open` 通槽是否两端都无残余实体。\n"
        "5. `open_closed` 半通槽是否恰好一端开放、另一端有完整终止壁。\n"
        "6. `closed_loop` 闭槽是否首尾连续，且整圈不与基体边界相交。\n"
        "7. 是否存在埋在实体内部、没有开放到表面的隐藏通道。\n\n"
        "## 填写 CSV\n\n"
        "打开同级 `human_review_canonical_20.csv`，只填写两列：\n\n"
        "- `review_result`：填写 `PASS` 或 `FAIL`。\n"
        "- `review_reason`：PASS 可简写“七项均通过”；FAIL 请写简短原因和可识别的几何位置。\n\n"
        "不要修改 STEP 文件、seed、哈希或其他 CSV 列。完成 20 行后保存 CSV，并把结论交回 Codex。\n"
        """,
        encoding="utf-8",
    )


def write_summary(path: Path, pilot_records: list[dict[str, Any]], canonical_records: list[dict[str, Any]]) -> None:
    combination_counts = Counter((record["profile_shape"], record["guide_family"], record["end_condition"]) for record in pilot_records)
    failed = [record for record in pilot_records + canonical_records if not record["audit"]["passed"]]
    curved_counts = Counter(
        record["audit"]["feature_face_count"]
        for record in pilot_records
        if record["guide_family"] != "straight"
    )
    lines = [
        "# P1-B1 连续槽自动审计报告",
        "",
        f"- Pilot：{len(pilot_records)} 个（20 组合 × {REQUIRED_PILOT_PER_COMBINATION}）",
        f"- Canonical：{len(canonical_records)} 个（20 组合 × 1）",
        f"- 自动审计通过：{len(pilot_records) + len(canonical_records) - len(failed)}/{len(pilot_records) + len(canonical_records)}",
        f"- 固定种子：pilot base={PILOT_SEED_BASE}，canonical base={CANONICAL_SEED_BASE}",
        "- 宿主：100×80×25 mm 单一长方体，平面宿主面",
        "- 曲线协议：单条解析 line/circular arc/circle edge；禁止采样折线",
        "",
        "## 自动验收项",
        "",
        "每个样本均检查：单 solid、正体积、B-Rep valid、STEP 可回读且无 issue、guide 单 edge/C1/解析曲线、端部状态、槽面标签覆盖、无槽面误标 stock、单实例、曲线槽特征面数不为 16/32/48。",
        "",
        "## 组合覆盖",
        "",
    ]
    for combination in all_slot_combinations():
        lines.append(f"- `{combination[0]} / {combination[1]} / {combination[2]}`：{combination_counts[combination]} 个")
    lines.extend([
        "",
        "## 曲线槽特征面数分布",
        "",
        f"`{dict(sorted(curved_counts.items()))}`",
        "",
        "## 失败项",
        "",
        "无。" if not failed else "\n".join(f"- {record['step_path']}" for record in failed),
        "",
        "## 结论",
        "",
        "自动验收通过后必须进入 Human Gate H-B1；在用户完成 20 个 STEP 的 UG/NX 审查前，不得开始 rib/web。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    pilot_step_dir = output_dir / "pilot_80" / "step"
    pilot_graph_dir = output_dir / "pilot_80" / "graph"
    canonical_dir = output_dir / "human_review_canonical_20"
    canonical_graph_dir = output_dir / "human_review_canonical_20_graph"
    output_dir.mkdir(parents=True, exist_ok=True)

    semantic_index = load_p1_b_semantic_index()
    ontology_maps = load_ontology_maps(ROOT / "CAE_feature_ontology.yaml")
    primitive_index = load_layer_index(ROOT / "CAE_feature_ontology.yaml", "primitive_surface")
    generator = P1BContinuousSlotGenerator(semantic_index)

    pilot_records = []
    canonical_records = []
    for combination_index, (profile_shape, guide_family, end_condition) in enumerate(all_slot_combinations()):
        stem = f"{combination_index + 1:02d}_{PROFILE_TO_MACRO[profile_shape]}_{guide_family}_{end_condition}"
        for replicate in range(REQUIRED_PILOT_PER_COMBINATION):
            seed = PILOT_SEED_BASE + combination_index * REQUIRED_PILOT_PER_COMBINATION + replicate
            sample_stem = f"{stem}_r{replicate + 1}_seed{seed}"
            pilot_records.append(build_and_write_sample(
                generator=generator,
                ontology_maps=ontology_maps,
                primitive_index=primitive_index,
                semantic_index=semantic_index,
                profile_shape=profile_shape,
                guide_family=guide_family,
                end_condition=end_condition,
                seed=seed,
                step_path=pilot_step_dir / f"{sample_stem}.step",
                graph_path=pilot_graph_dir / f"{sample_stem}.json",
            ))
        seed = CANONICAL_SEED_BASE + combination_index
        canonical_records.append(build_and_write_sample(
            generator=generator,
            ontology_maps=ontology_maps,
            primitive_index=primitive_index,
            semantic_index=semantic_index,
            profile_shape=profile_shape,
            guide_family=guide_family,
            end_condition=end_condition,
            seed=seed,
            step_path=canonical_dir / f"{stem}_seed{seed}.step",
            graph_path=canonical_graph_dir / f"{stem}_seed{seed}.json",
        ))

    all_records = pilot_records + canonical_records
    audit_payload = {
        "schema_version": "p1_b1_slot_pilot_audit_v1",
        "protocol": {
            "pilot_count": 80,
            "canonical_count": 20,
            "pilot_per_combination": REQUIRED_PILOT_PER_COMBINATION,
            "combination_count": len(all_slot_combinations()),
            "pilot_seed_base": PILOT_SEED_BASE,
            "canonical_seed_base": CANONICAL_SEED_BASE,
            "forbidden_fixed_segment_counts": sorted(FORBIDDEN_SEGMENT_COUNTS),
        },
        "automatic_pass": all(record["audit"]["passed"] for record in all_records),
        "pilot_records": pilot_records,
        "canonical_records": canonical_records,
    }
    (output_dir / "automatic_audit_100.json").write_text(json.dumps(audit_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_review_csv(output_dir / "human_review_canonical_20.csv", canonical_records)
    write_nx_guide(output_dir / "H-B1_UG_NX_检查指南.zh-CN.md")
    write_summary(output_dir / "automatic_audit_summary.zh-CN.md", pilot_records, canonical_records)
    manifest = {
        "schema_version": "p1_b1_human_gate_manifest_v1",
        "automatic_pass": audit_payload["automatic_pass"],
        "files": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in sorted(output_dir.rglob("*")) if path.is_file()
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if not audit_payload["automatic_pass"]:
        raise RuntimeError("P1-B1 automatic audit failed; inspect automatic_audit_100.json")
    print(json.dumps({
        "automatic_pass": True,
        "pilot_count": len(pilot_records),
        "canonical_count": len(canonical_records),
        "output_dir": str(output_dir),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
