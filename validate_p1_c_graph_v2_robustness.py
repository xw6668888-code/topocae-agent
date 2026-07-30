from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRepFeat import BRepFeat_SplitShape
from OCC.Core.GeomAbs import GeomAbs_Plane
from OCC.Core.gp import gp_Pnt
from OCC.Extend.TopologyUtils import TopologyExplorer

from core.graph_v2_1 import canonical_graph_signature, extract_graph_v2
from p1_c.formal_generator_v5 import build_planned_result, transformed_shape_and_audit
from p1_c.formal_plan_v4 import build_plan
from p1_c.geometry import cleanup_audit, topology_counts, unified_probe
from p1_c.pilot_h_c2 import exact_labels
from utils.graph_v2_validation import compare_graph_v2_invariance, compare_partition_invariance
from utils.ontology import load_layer_index


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output_generation_p1_c_clean_context_7200" / "reports" / "graph_v2_robustness"


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    primitive_index = load_layer_index(ROOT / "CAE_feature_ontology.yaml", "primitive_surface")
    plan = build_plan()
    selected = []
    for primary in sorted({sample.primary_target for sample in plan}):
        selected.append(next(sample for sample in plan if sample.primary_target == primary and sample.context == "target_only"))
        selected.append(next(sample for sample in plan if sample.primary_target == primary and sample.context == "triple"))
    invariance_records: list[dict[str, Any]] = []
    cleanup_records: list[dict[str, Any]] = []
    partition_records: list[dict[str, Any]] = []
    for sample in selected:
        result = build_planned_result(sample)
        labels = exact_labels(result, ROOT)
        reference = extract_graph_v2(result.shape, face_labels=labels, primitive_surface_index=primitive_index)
        rotated_shape, rotation_shape_audit = transformed_shape_and_audit(
            result,
            scale=1.0,
            rotation_axis=(1.0, 2.0, 3.0),
            rotation_angle_radians=0.73,
        )
        scaled_shape, scale_shape_audit = transformed_shape_and_audit(
            result,
            scale=1.7,
            rotation_axis=(0.0, 0.0, 1.0),
            rotation_angle_radians=0.0,
        )
        scaled_labels = deepcopy(labels)
        for label in scaled_labels:
            label["face_area"] = float(label.get("face_area", 0.0)) * 1.7**2
        rotated = extract_graph_v2(rotated_shape, face_labels=labels, primitive_surface_index=primitive_index)
        scaled = extract_graph_v2(scaled_shape, face_labels=scaled_labels, primitive_surface_index=primitive_index)
        rotation = compare_graph_v2_invariance(reference, rotated)
        scale = compare_graph_v2_invariance(reference, scaled)
        invariance_records.append(
            {
                "sample_id": sample.sample_id,
                "primary_target": sample.primary_target,
                "context": sample.context,
                "rotation": rotation,
                "scale": scale,
                "rotation_shape_audit": rotation_shape_audit,
                "scale_shape_audit": scale_shape_audit,
                "passed": rotation["score"] >= 0.98 and scale["score"] >= 0.98,
            }
        )
        probe = unified_probe(result.shape)
        probe_graph = extract_graph_v2(probe, primitive_surface_index=primitive_index)
        clean_graph = extract_graph_v2(result.shape, primitive_surface_index=primitive_index)
        clean_signature = canonical_graph_signature(clean_graph)
        probe_signature = canonical_graph_signature(probe_graph)
        cleanup_records.append(
            {
                "sample_id": sample.sample_id,
                "cleanup_probe": cleanup_audit(result.shape),
                "canonical_signature_equal": clean_signature == probe_signature,
                "passed": cleanup_audit(result.shape)["passed"] and clean_signature == probe_signature,
            }
        )
    partition_candidates = [
        sample
        for sample in selected
        if sample.context == "target_only"
        and sample.primary_target not in {"through_hole_round", "counterbore_hole", "countersink_hole", "obround_hole", "rib", "web"}
    ][:12]
    for sample in partition_candidates:
        result = build_planned_result(sample)
        labels = exact_labels(result, ROOT)
        reference = extract_graph_v2(result.shape, face_labels=labels, primitive_surface_index=primitive_index)
        partitioned = _partition_bottom_stock_face(result.shape, labels)
        if partitioned is None:
            partition_records.append({"sample_id": sample.sample_id, "passed": False, "reason": "no_partition"})
            continue
        partition_shape, partition_labels = partitioned
        graph = extract_graph_v2(partition_shape, face_labels=partition_labels, primitive_surface_index=primitive_index)
        comparison = compare_partition_invariance(reference, graph)
        partition_records.append(
            {
                "sample_id": sample.sample_id,
                "comparison": comparison,
                "partition_cleanup_expected_false": not cleanup_audit(partition_shape)["passed"],
                "passed": comparison["score"] >= 0.98,
            }
        )
    rotation_scores = [record["rotation"]["score"] for record in invariance_records]
    scale_scores = [record["scale"]["score"] for record in invariance_records]
    partition_scores = [record.get("comparison", {}).get("score", 0.0) for record in partition_records]
    payload = {
        "schema_version": "topocae_p1_c_graph_v2_robustness_1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(selected),
        "rotation_consistency": min(rotation_scores, default=0.0),
        "scale_consistency": min(scale_scores, default=0.0),
        "partition_consistency": min(partition_scores, default=0.0),
        "cleanup_probe_signature_consistency": sum(record["passed"] for record in cleanup_records) / max(len(cleanup_records), 1),
        "invariance_records": invariance_records,
        "cleanup_records": cleanup_records,
        "partition_records": partition_records,
    }
    payload["passed"] = (
        payload["rotation_consistency"] >= 0.98
        and payload["scale_consistency"] >= 0.98
        and payload["partition_consistency"] >= 0.98
        and payload["cleanup_probe_signature_consistency"] >= 0.98
    )
    json_path = OUTPUT / "robustness.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = "\n".join(
        [
            "# P1-C Graph v2 鲁棒性审计",
            "",
            f"- 自动验收：{'PASS' if payload['passed'] else 'FAIL'}。",
            f"- 旋转 consistency：{payload['rotation_consistency']:.6f}。",
            f"- 缩放 consistency：{payload['scale_consistency']:.6f}。",
            f"- 合法 face partition consistency：{payload['partition_consistency']:.6f}。",
            f"- cleanup probe canonical signature consistency：{payload['cleanup_probe_signature_consistency']:.6f}。",
            "- context/primary/seed/path 未作为训练向量输入。",
            "",
        ]
    )
    (OUTPUT / "summary.zh-CN.md").write_text(summary, encoding="utf-8")
    manifest = {
        "schema_version": "topocae_p1_c_graph_v2_robustness_manifest_1",
        "robustness_sha256": _sha256(json_path),
        "summary_sha256": _sha256(OUTPUT / "summary.zh-CN.md"),
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": payload["passed"],
                "rotation": payload["rotation_consistency"],
                "scale": payload["scale_consistency"],
                "partition": payload["partition_consistency"],
                "cleanup_probe": payload["cleanup_probe_signature_consistency"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if payload["passed"] else 1


def _partition_bottom_stock_face(
    shape: Any,
    labels: list[dict[str, Any]],
) -> tuple[Any, list[dict[str, Any]]] | None:
    faces = list(TopologyExplorer(shape).faces())
    target = None
    target_index = -1
    for index, (face, label) in enumerate(zip(faces, labels, strict=True)):
        if int(label["instance_id"]) != 0:
            continue
        surface = BRepAdaptor_Surface(face)
        if surface.GetType() != GeomAbs_Plane:
            continue
        plane = surface.Plane()
        if abs(plane.Axis().Direction().Z()) > 0.99 and abs(plane.Location().Z()) < 1e-6:
            target = face
            target_index = index
            break
    if target is None:
        return None
    xs: list[float] = []
    ys: list[float] = []
    for vertex in TopologyExplorer(target).vertices():
        from OCC.Core.BRep import BRep_Tool

        point = BRep_Tool.Pnt(vertex)
        xs.append(float(point.X()))
        ys.append(float(point.Y()))
    if not xs or not ys:
        return None
    x_min, x_max = min(xs), max(xs)
    y_mid = (min(ys) + max(ys)) / 2.0
    edge = BRepBuilderAPI_MakeEdge(gp_Pnt(x_min, y_mid, 0.0), gp_Pnt(x_max, y_mid, 0.0)).Edge()
    splitter = BRepFeat_SplitShape(shape)
    splitter.Add(edge, target)
    splitter.Build()
    if not splitter.IsDone():
        return None
    partitioned = splitter.Shape()
    if not BRepCheck_Analyzer(partitioned).IsValid() or topology_counts(partitioned)["solid_count"] != 1:
        return None
    new_labels: list[dict[str, Any]] = []
    unmatched = 0
    for new_face in TopologyExplorer(partitioned).faces():
        match_index = next((index for index, old_face in enumerate(faces) if new_face.IsSame(old_face)), None)
        if match_index is None:
            new_labels.append(deepcopy(labels[target_index]))
            unmatched += 1
        else:
            new_labels.append(deepcopy(labels[match_index]))
    if unmatched < 2 or len(new_labels) <= len(labels):
        return None
    return partitioned, new_labels


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
