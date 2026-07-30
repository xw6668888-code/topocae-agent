"""Official corrected-dataset builder using fast native BREP geometry artifacts."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRepTools import breptools
from OCC.Extend.TopologyUtils import TopologyExplorer

import build_p1_b_corrected_dataset_5000 as base
from core.graph_v2_1 import canonical_graph_signature, extract_graph_v2


base.OUTPUT = base.ROOT / "output_generation_p1_b_corrected_balanced_5000_v2"
OUTPUT = base.OUTPUT


def _emit_shape_graph_v2(
    self: base.BuildState,
    sample_id: str,
    shape: Any,
    labels: list[dict[str, Any]],
    *,
    feature_sets: list[dict[str, Any]] | None = None,
    feature_parameters: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not BRepCheck_Analyzer(shape).IsValid() or len(list(TopologyExplorer(shape).solids())) != 1:
        raise ValueError("generated derivative must be a valid single solid")
    geometry_dir = OUTPUT / "geometries"
    geometry_dir.mkdir(parents=True, exist_ok=True)
    geometry_path = geometry_dir / f"{sample_id}.brep"
    geometry_path.write_text(breptools.WriteToString(shape), encoding="utf-8")
    expected = base._face_sequence_signature(shape)
    graph = extract_graph_v2(shape, face_labels=labels, primitive_surface_index=self.primitive_index)
    base._validate_face_sequence(expected, graph, relative_area_tolerance=1e-8)
    if feature_sets is not None:
        graph["feature_sets"] = deepcopy(feature_sets)
    if feature_parameters:
        for feature_set in graph.get("feature_sets", []):
            instance_id = int(feature_set.get("instance_id", 0) or 0)
            if instance_id in feature_parameters:
                feature_set["geometry_parameters"] = deepcopy(feature_parameters[instance_id])
    graph["native_geometry_artifact"] = {
        "path": str(geometry_path.resolve()),
        "format": "OpenCASCADE_BREP_text",
        "sha256": base._sha256(geometry_path),
        "serialization": "breptools.WriteToString",
    }
    return graph


def _finalize_sample_v2(
    self: base.BuildState,
    sample_id: str,
    graph: dict[str, Any],
    step_path: Path,
    *,
    base_id: str,
    derivation_kind: str,
    derivation_parameters: dict[str, Any],
    source_kind: str,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    graph_path = OUTPUT / "graphs" / f"{sample_id}.json"
    native_record = graph.get("native_geometry_artifact")
    if step_path.exists():
        geometry_path = step_path
        geometry_format = "STEP"
    elif native_record:
        geometry_path = Path(native_record["path"])
        geometry_format = "OpenCASCADE_BREP_text"
    else:
        raise FileNotFoundError(f"no geometry artifact for {sample_id}")
    graph["sample_id"] = sample_id
    graph["source_geometry_path"] = str(geometry_path.resolve())
    graph["source_geometry_format"] = geometry_format
    graph["source_step_path"] = str(step_path.resolve()) if geometry_format == "STEP" else None
    graph["base_geometry_family_id"] = base_id
    graph["derivation_kind"] = derivation_kind
    graph["derivation_parameters"] = derivation_parameters
    graph["dataset_provenance"] = provenance
    exact_signature = base._signature_hash(canonical_graph_signature(graph, ndigits=6))
    near_signature = base._signature_hash(canonical_graph_signature(graph, ndigits=4))
    graph["geometry_signatures"] = {
        "exact_graph_signature": exact_signature,
        "near_graph_signature": near_signature,
        "rotation_scale_invariant": True,
        "canonical_patch_aware": True,
    }
    graph_path.write_text(json.dumps(graph, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    macro_counts = Counter(str(feature_set["macro_feature"]) for feature_set in graph.get("feature_sets", []))
    forbidden = sorted(set(macro_counts) & base.REBUILD_ONLY) if source_kind.startswith("legacy") else []
    if forbidden:
        raise ValueError(f"legacy sample unexpectedly contains rebuild-only macros: {forbidden}")
    geometry_sha256 = base._sha256(geometry_path)
    record = {
        "sample_id": sample_id,
        "graph_path": str(graph_path.resolve()),
        "step_path": str(geometry_path.resolve()),
        "geometry_path": str(geometry_path.resolve()),
        "geometry_format": geometry_format,
        "graph_sha256": base._sha256(graph_path),
        "step_sha256": geometry_sha256,
        "geometry_sha256": geometry_sha256,
        "base_geometry_family_id": base_id,
        "derivation_kind": derivation_kind,
        "source_kind": source_kind,
        "macro_instance_counts": dict(sorted(macro_counts.items())),
        "feature_instance_count": sum(macro_counts.values()),
        "node_count": len(graph.get("nodes", [])),
        "edge_count": len(graph.get("edges", [])),
        "exact_graph_signature": exact_signature,
        "near_graph_signature": near_signature,
    }
    self.samples.append(record)
    return record


base.BuildState._emit_shape_graph = _emit_shape_graph_v2
base.BuildState._finalize_sample = _finalize_sample_v2


def main() -> int:
    exit_code = base.main()
    meta_path = OUTPUT / "dataset_meta.json"
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    formats = Counter(record["geometry_format"] for record in _read_manifest(OUTPUT / "manifests" / "sample_manifest.jsonl"))
    payload["schema_version"] = "topocae_p1_b_corrected_balanced_dataset_2"
    payload["geometry_artifact_counts"] = dict(sorted(formats.items()))
    payload["contracts"]["geometry_artifact_protocol"] = (
        "legacy identities retain hash-pinned STEP; generated/derived training shapes use exact OpenCASCADE BREP text"
    )
    payload["contracts"]["graph_extracted_from_actual_transformed_brep"] = True
    payload["contracts"]["native_brep_is_training_geometry_not_human_gate_substitute"] = True
    meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_path = OUTPUT / "reports" / "dataset_summary.zh-CN.md"
    with summary_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            "\n几何载体：旧安全 identity 保留原 STEP；新生成与旋转/缩放/partition 派生以 OpenCASCADE BREP text 固化，Graph v2 直接从实际变换后的 B-Rep 提取。\n"
        )
    artifact_path = OUTPUT / "manifests" / "artifact_manifest.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["schema_version"] = "topocae_p1_b_corrected_dataset_artifact_manifest_2"
    artifact["dataset_meta_sha256"] = base._sha256(meta_path)
    artifact["summary_sha256"] = base._sha256(summary_path)
    artifact["geometry_artifact_counts"] = dict(sorted(formats.items()))
    artifact_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "official_schema": payload["schema_version"],
        "automatic_pass": payload["automatic_pass"],
        "geometry_artifact_counts": dict(sorted(formats.items())),
        "dataset_meta_sha256": base._sha256(meta_path),
    }, ensure_ascii=False), flush=True)
    return exit_code


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
