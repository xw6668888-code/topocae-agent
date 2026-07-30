from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
from typing import Any, Iterable

from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_Transform
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRepFeat import BRepFeat_SplitShape
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.GProp import GProp_GProps
from OCC.Core.GeomAbs import (
    GeomAbs_BezierSurface,
    GeomAbs_BSplineSurface,
    GeomAbs_Cone,
    GeomAbs_Cylinder,
    GeomAbs_OffsetSurface,
    GeomAbs_Plane,
    GeomAbs_Sphere,
    GeomAbs_SurfaceOfExtrusion,
    GeomAbs_SurfaceOfRevolution,
    GeomAbs_Torus,
)
from OCC.Core.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Trsf
from OCC.Extend.TopologyUtils import TopologyExplorer

from core.graph_v2_1 import canonical_graph_signature, extract_graph_v2
from features.p1_b_continuous_slot import P1BContinuousSlotGenerator, all_slot_combinations
from features.p1_b_rib_web_v3 import build_rib_web_sample, scenario_keys
from generate_p1_b1_slot_pilot import load_p1_b_semantic_index
from generate_p1_b1_slot_pilot_v2 import rebuild_with_identity_roles
from utils.ontology import load_layer_index, load_ontology_maps
from utils.step_solid_validation import read_step_shape, write_step_with_positive_volume


ROOT = Path(__file__).resolve().parent
SOURCE_AUDIT = ROOT / "reports" / "p1_b_corrected_dataset_source_audit" / "source_audit.json"
OUTPUT = ROOT / "output_generation_p1_b_corrected_balanced_5000_v1"
TARGET_SAMPLE_COUNT = 5000
TARGET_CLASSES = [
    "through_hole_round", "blind_hole_round", "stepped_hole_round", "counterbore_hole",
    "countersink_hole", "obround_hole", "slot_u", "slot_rect", "pocket_rect",
    "pocket_circular", "cylinder_boss", "rib", "web", "rectangular_boss", "fillet",
    "chamfer", "slot_v", "slot_o",
]
REBUILD_ONLY = {"slot_u", "slot_rect", "slot_v", "slot_o", "rib", "web"}
SLOT_PROFILE_TO_MACRO = {"rect": "slot_rect", "u": "slot_u", "v": "slot_v", "o": "slot_o"}
H_B1_CSV_SHA256 = "B3C2E2E2D7D47D72416BFDF657A552CBB58087EF9FBE58DCFB783FA815AD7F3A"
H_B2_CSV_SHA256 = "2B50AFCBA9DD954289CAEF43F1ABDB6F7F7EB005E3DBD33E0735F694D4B7DC37"
PRIMITIVE_NAMES = {
    int(GeomAbs_Plane): "plane",
    int(GeomAbs_Cylinder): "cylinder",
    int(GeomAbs_Cone): "cone",
    int(GeomAbs_Sphere): "sphere",
    int(GeomAbs_Torus): "torus",
    int(GeomAbs_BezierSurface): "bezier_surface",
    int(GeomAbs_BSplineSurface): "bspline_surface",
    int(GeomAbs_SurfaceOfRevolution): "surface_of_revolution",
    int(GeomAbs_SurfaceOfExtrusion): "surface_of_extrusion",
    int(GeomAbs_OffsetSurface): "offset_surface",
}


def main() -> int:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"output directory already exists and is non-empty: {OUTPUT}")
    for directory in (OUTPUT / "steps", OUTPUT / "graphs", OUTPUT / "manifests", OUTPUT / "splits", OUTPUT / "reports"):
        directory.mkdir(parents=True, exist_ok=True)
    audit = json.loads(SOURCE_AUDIT.read_text(encoding="utf-8"))
    semantic_index = load_p1_b_semantic_index()
    ontology_maps = load_ontology_maps(ROOT / "CAE_feature_ontology.yaml")
    primitive_index = load_layer_index(ROOT / "CAE_feature_ontology.yaml", "primitive_surface")
    state = BuildState(semantic_index, ontology_maps, primitive_index)

    print(f"[1/4] re-extracting {len(audit['selected_samples'])} legacy-safe STEP files", flush=True)
    legacy_valid = []
    legacy_rejected = []
    for position, source in enumerate(audit["selected_samples"]):
        try:
            valid_record = state.emit_legacy_identity(source, position)
            legacy_valid.append(valid_record)
        except Exception as exc:
            legacy_rejected.append({"sample_id": source["sample_id"], "reason": f"{type(exc).__name__}: {exc}"})
        if (position + 1) % 100 == 0:
            print(f"  legacy {position + 1}/{len(audit['selected_samples'])}; accepted={len(legacy_valid)} rejected={len(legacy_rejected)}", flush=True)

    print("[2/4] generating Human-Gate-approved slot families", flush=True)
    slot_partition_count = 0
    for profile in ("rect", "u", "v", "o"):
        generated, partitions = state.generate_slot_family(profile, base_count=100, partition_target=25)
        slot_partition_count += partitions
        print(f"  {SLOT_PROFILE_TO_MACRO[profile]}: {generated} samples; legal partitions={partitions}", flush=True)

    print("[3/4] generating Human-Gate-approved rib/web families", flush=True)
    for macro in ("rib", "web"):
        generated = state.generate_rib_web_family(macro, base_count=100)
        print(f"  {macro}: {generated} samples", flush=True)

    remaining = TARGET_SAMPLE_COUNT - len(state.samples)
    if remaining < 0:
        raise RuntimeError(f"base generation exceeded target by {-remaining} samples")
    print(f"[4/4] generating {remaining} grouped legacy rotation/scale derivatives", flush=True)
    selected_derivative_bases = _select_derivative_bases(legacy_valid, remaining)
    for position, base in enumerate(selected_derivative_bases):
        state.emit_legacy_derivative(base, position)
        if (position + 1) % 100 == 0:
            print(f"  derivatives {position + 1}/{remaining}", flush=True)

    if len(state.samples) != TARGET_SAMPLE_COUNT:
        raise RuntimeError(f"expected {TARGET_SAMPLE_COUNT} samples, got {len(state.samples)}")
    split_audit = _assign_grouped_splits(state.samples)
    payload = _write_dataset_artifacts(
        state,
        legacy_valid=legacy_valid,
        legacy_rejected=legacy_rejected,
        slot_partition_count=slot_partition_count,
        split_audit=split_audit,
        source_audit=audit,
    )
    print(json.dumps({
        "automatic_pass": payload["automatic_pass"],
        "sample_count": payload["sample_count"],
        "instance_count": payload["instance_count"],
        "minimum_class_count": min(payload["instance_counts"].values()),
        "legacy_identity_count": payload["source_counts"]["legacy_identity"],
        "new_generator_count": payload["source_counts"]["new_generator"],
        "derived_count": payload["source_counts"]["legacy_derived"],
        "partition_count": payload["derivation_counts"].get("legal_face_partition", 0),
        "split_leakage": payload["split_audit"]["leakage_detected"],
    }, ensure_ascii=False), flush=True)
    return 0 if payload["automatic_pass"] else 1


class BuildState:
    def __init__(self, semantic_index: dict[str, Any], ontology_maps: Any, primitive_index: dict[str, Any]) -> None:
        self.semantic_index = semantic_index
        self.ontology_maps = ontology_maps
        self.primitive_index = primitive_index
        self.samples: list[dict[str, Any]] = []
        self.slot_generator = P1BContinuousSlotGenerator(semantic_index)

    def emit_legacy_identity(self, source: dict[str, Any], position: int) -> dict[str, Any]:
        old_graph = json.loads(Path(source["source_graph_path"]).read_text(encoding="utf-8"))
        labels = _labels_from_graph(old_graph, self.semantic_index)
        expected = _expected_sequence_from_graph(old_graph)
        sample_id = f"legacy_{position:05d}"
        base_id = f"legacy::{source['source_root']}::{source['source_sample_id']}"
        destination_step = OUTPUT / "steps" / f"{sample_id}.step"
        shutil.copy2(source["source_step_path"], destination_step)
        shape = read_step_shape(destination_step)
        graph = extract_graph_v2(shape, face_labels=labels, primitive_surface_index=self.primitive_index)
        _validate_face_sequence(expected, graph, relative_area_tolerance=1e-3)
        graph["feature_sets"] = deepcopy(old_graph["feature_sets"])
        provenance = {
            "source_kind": "legacy_identity",
            "source_sample_id": source["sample_id"],
            "source_step_sha256": source["source_step_sha256"],
            "source_graph_sha256": source["source_graph_sha256"],
            "reuse_policy": source["reuse_policy"],
            "old_rebuild_only_macro_present": False,
        }
        record = self._finalize_sample(
            sample_id,
            graph,
            destination_step,
            base_id=base_id,
            derivation_kind="identity",
            derivation_parameters={},
            source_kind="legacy_identity",
            provenance=provenance,
        )
        return {
            "sample_record": record,
            "source": source,
            "base_id": base_id,
            "identity_graph_path": record["graph_path"],
            "identity_step_path": record["step_path"],
            "macro_features": list(record["macro_instance_counts"]),
        }

    def emit_legacy_derivative(self, base: dict[str, Any], position: int) -> None:
        base_graph = json.loads(Path(base["identity_graph_path"]).read_text(encoding="utf-8"))
        shape = read_step_shape(Path(base["identity_step_path"]))
        labels = _labels_from_graph(base_graph, self.semantic_index)
        kind = "rigid_rotation" if position % 2 == 0 else "uniform_scale"
        if kind == "rigid_rotation":
            axis = [1.0 + (position % 3), 2.0 + (position % 5), 3.0 + (position % 7)]
            angle = 0.31 + 0.07 * (position % 9)
            transformed = _rotate_shape(shape, axis, angle)
            parameters = {"axis": axis, "angle_radians": angle}
        else:
            factors = [0.55, 0.75, 1.4, 1.8, 2.3]
            factor = factors[(position // 2) % len(factors)]
            transformed = _scale_shape(shape, factor)
            parameters = {"factor": factor}
        sample_id = f"legacy_derived_{position:04d}"
        graph = self._emit_shape_graph(
            sample_id,
            transformed,
            labels,
            feature_sets=deepcopy(base_graph.get("feature_sets", [])),
        )
        self._finalize_sample(
            sample_id,
            graph,
            OUTPUT / "steps" / f"{sample_id}.step",
            base_id=base["base_id"],
            derivation_kind=kind,
            derivation_parameters=parameters,
            source_kind="legacy_derived",
            provenance={
                "source_kind": "legacy_derived",
                "base_sample_id": base["sample_record"]["sample_id"],
                "source_step_sha256": base["source"]["source_step_sha256"],
                "old_rebuild_only_macro_present": False,
            },
        )

    def generate_slot_family(self, profile: str, *, base_count: int, partition_target: int) -> tuple[int, int]:
        combinations = [item for item in all_slot_combinations() if item[0] == profile]
        successful = 0
        emitted = 0
        partitions = 0
        attempt = 0
        macro = SLOT_PROFILE_TO_MACRO[profile]
        while successful < base_count:
            combination = combinations[successful % len(combinations)]
            _, guide_family, end_condition = combination
            seed = 810000 + list(SLOT_PROFILE_TO_MACRO).index(profile) * 10000 + attempt
            rng = random.Random(seed)
            length = rng.uniform(82.0, 145.0)
            width = rng.uniform(66.0, min(112.0, length * 0.85))
            height = rng.uniform(20.0, 38.0)
            stock = BRepPrimAPI_MakeBox(gp_Pnt(-length / 2.0, -width / 2.0, 0.0), length, width, height).Shape()
            attempt += 1
            try:
                raw = self.slot_generator.build_one(
                    stock,
                    length,
                    width,
                    height,
                    profile_shape=profile,
                    guide_family=guide_family,
                    end_condition=end_condition,
                    seed=seed,
                    instance_id=1,
                )
                result = rebuild_with_identity_roles(raw, stock)
                labels, face_items = _slot_labels(result, self.semantic_index, self.ontology_maps)
                base_id = f"new_slot::{macro}::{successful:04d}"
                params = {
                    "source_generator": "p1_b_continuous_closed_profile_sweep_cut",
                    "profile_shape": profile,
                    "guide_family": guide_family,
                    "end_condition": end_condition,
                    "seed": seed,
                    "stock_dimensions": [length, width, height],
                    "human_gate_csv_sha256": H_B1_CSV_SHA256,
                    **{key: value for key, value in result.parameters.items() if not str(key).startswith("_")},
                }
                feature_parameters = {1: params}
                variants = [
                    ("identity", result.shape, {}),
                    ("rigid_rotation", _rotate_shape(result.shape, [1.0, 2.0 + successful % 3, 3.0], 0.37 + 0.03 * (successful % 11)), {
                        "axis": [1.0, 2.0 + successful % 3, 3.0], "angle_radians": 0.37 + 0.03 * (successful % 11)
                    }),
                    ("uniform_scale", _scale_shape(result.shape, [0.6, 1.5, 2.1][successful % 3]), {
                        "factor": [0.6, 1.5, 2.1][successful % 3]
                    }),
                ]
                for kind, shape, transform_parameters in variants:
                    sample_id = f"new_{macro}_{successful:04d}_{kind}"
                    graph = self._emit_shape_graph(sample_id, shape, labels, feature_parameters=feature_parameters)
                    self._finalize_sample(
                        sample_id,
                        graph,
                        OUTPUT / "steps" / f"{sample_id}.step",
                        base_id=base_id,
                        derivation_kind=kind,
                        derivation_parameters=transform_parameters,
                        source_kind="new_generator",
                        provenance={
                            "source_kind": "new_generator",
                            "generator": "p1_b_continuous_closed_profile_sweep_cut",
                            "human_gate_csv_sha256": H_B1_CSV_SHA256,
                            "concept_rebuilt": True,
                        },
                    )
                    emitted += 1
                if successful < partition_target:
                    partitioned = _partition_slot_stock_face(result.shape, face_items, labels, length, width)
                    if partitioned is not None:
                        partition_shape, partition_labels = partitioned
                        sample_id = f"new_{macro}_{successful:04d}_partition"
                        graph = self._emit_shape_graph(
                            sample_id,
                            partition_shape,
                            partition_labels,
                            feature_parameters=feature_parameters,
                        )
                        self._finalize_sample(
                            sample_id,
                            graph,
                            OUTPUT / "steps" / f"{sample_id}.step",
                            base_id=base_id,
                            derivation_kind="legal_face_partition",
                            derivation_parameters={"partitioned_stock_face": True},
                            source_kind="new_generator",
                            provenance={
                                "source_kind": "new_generator",
                                "generator": "p1_b_continuous_closed_profile_sweep_cut",
                                "human_gate_csv_sha256": H_B1_CSV_SHA256,
                                "concept_rebuilt": True,
                                "legal_modeling_history_variant": True,
                            },
                        )
                        partitions += 1
                        emitted += 1
                successful += 1
            except Exception:
                if attempt > base_count * 20:
                    raise
                continue
        return emitted, partitions

    def generate_rib_web_family(self, macro: str, *, base_count: int) -> int:
        keys = [key for key in scenario_keys() if key.startswith(f"{macro}_")]
        successful = 0
        attempt = 0
        emitted = 0
        while successful < base_count:
            scenario = keys[successful % len(keys)]
            seed = 910000 + (0 if macro == "rib" else 10000) + attempt
            attempt += 1
            try:
                result = build_rib_web_sample(scenario, seed)
                labels = _rib_web_labels(result, self.semantic_index, self.ontology_maps)
                base_id = f"new_rib_web::{macro}::{successful:04d}"
                feature_parameters = {
                    int(instance_id): {
                        "source_generator": "p1_b_supported_rib_web",
                        "scenario": scenario,
                        "seed": seed,
                        "human_gate_csv_sha256": H_B2_CSV_SHA256,
                        **deepcopy(result.parameters),
                    }
                    for instance_id in sorted({int(label["instance_id"]) for label in labels if int(label["instance_id"]) > 0})
                }
                rotation_axis = [2.0, 1.0 + successful % 4, 3.0]
                rotation_angle = 0.29 + 0.04 * (successful % 13)
                scale_factor = [0.65, 1.45, 2.0][successful % 3]
                variants = [
                    ("identity", result.shape, {}),
                    ("rigid_rotation", _rotate_shape(result.shape, rotation_axis, rotation_angle), {
                        "axis": rotation_axis, "angle_radians": rotation_angle
                    }),
                    ("uniform_scale", _scale_shape(result.shape, scale_factor), {"factor": scale_factor}),
                ]
                for kind, shape, transform_parameters in variants:
                    sample_id = f"new_{macro}_{successful:04d}_{kind}"
                    graph = self._emit_shape_graph(sample_id, shape, labels, feature_parameters=feature_parameters)
                    self._finalize_sample(
                        sample_id,
                        graph,
                        OUTPUT / "steps" / f"{sample_id}.step",
                        base_id=base_id,
                        derivation_kind=kind,
                        derivation_parameters=transform_parameters,
                        source_kind="new_generator",
                        provenance={
                            "source_kind": "new_generator",
                            "generator": "p1_b_supported_rib_web",
                            "human_gate_csv_sha256": H_B2_CSV_SHA256,
                            "concept_rebuilt": True,
                            "scenario": scenario,
                        },
                    )
                    emitted += 1
                successful += 1
            except Exception:
                if attempt > base_count * 20:
                    raise
                continue
        return emitted

    def _emit_shape_graph(
        self,
        sample_id: str,
        shape: Any,
        labels: list[dict[str, Any]],
        *,
        feature_sets: list[dict[str, Any]] | None = None,
        feature_parameters: dict[int, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if not BRepCheck_Analyzer(shape).IsValid() or len(list(TopologyExplorer(shape).solids())) != 1:
            raise ValueError("generated derivative must be a valid single solid")
        expected = _face_sequence_signature(shape)
        step_path = OUTPUT / "steps" / f"{sample_id}.step"
        with _silence_native_output():
            write_step_with_positive_volume(shape, step_path)
        exported = read_step_shape(step_path)
        graph = extract_graph_v2(exported, face_labels=labels, primitive_surface_index=self.primitive_index)
        _validate_face_sequence(expected, graph, relative_area_tolerance=2e-3)
        if feature_sets is not None:
            graph["feature_sets"] = deepcopy(feature_sets)
        if feature_parameters:
            for feature_set in graph.get("feature_sets", []):
                instance_id = int(feature_set.get("instance_id", 0) or 0)
                if instance_id in feature_parameters:
                    feature_set["geometry_parameters"] = deepcopy(feature_parameters[instance_id])
        return graph

    def _finalize_sample(
        self,
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
        graph["sample_id"] = sample_id
        graph["source_step_path"] = str(step_path.resolve())
        graph["base_geometry_family_id"] = base_id
        graph["derivation_kind"] = derivation_kind
        graph["derivation_parameters"] = derivation_parameters
        graph["dataset_provenance"] = provenance
        exact_signature = _signature_hash(canonical_graph_signature(graph, ndigits=6))
        near_signature = _signature_hash(canonical_graph_signature(graph, ndigits=4))
        graph["geometry_signatures"] = {
            "exact_graph_signature": exact_signature,
            "near_graph_signature": near_signature,
            "rotation_scale_invariant": True,
            "canonical_patch_aware": True,
        }
        graph_path.write_text(json.dumps(graph, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        macro_counts = Counter(str(feature_set["macro_feature"]) for feature_set in graph.get("feature_sets", []))
        forbidden = sorted(set(macro_counts) & REBUILD_ONLY) if source_kind.startswith("legacy") else []
        if forbidden:
            raise ValueError(f"legacy sample unexpectedly contains rebuild-only macros: {forbidden}")
        record = {
            "sample_id": sample_id,
            "graph_path": str(graph_path.resolve()),
            "step_path": str(step_path.resolve()),
            "graph_sha256": _sha256(graph_path),
            "step_sha256": _sha256(step_path),
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


def _select_derivative_bases(valid: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count > len(valid):
        raise ValueError("derivative selection currently requires unique base samples")
    priority = []
    used = set()
    for macro, needed in (("fillet", 110), ("obround_hole", 10)):
        matches = [record for record in valid if macro in record["macro_features"] and record["base_id"] not in used]
        for record in matches[:needed]:
            priority.append(record)
            used.add(record["base_id"])
    remaining = [record for record in valid if record["base_id"] not in used]
    remaining.sort(key=lambda record: hashlib.sha256(record["base_id"].encode("utf-8")).hexdigest())
    priority.extend(remaining[: max(0, count - len(priority))])
    return priority[:count]


def _assign_grouped_splits(samples: list[dict[str, Any]]) -> dict[str, Any]:
    base_ids = sorted({record["base_geometry_family_id"] for record in samples})
    parent = {base_id: base_id for base_id in base_ids}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    owners: dict[tuple[str, str], str] = {}
    for record in samples:
        base_id = record["base_geometry_family_id"]
        for kind in ("exact_graph_signature", "near_graph_signature"):
            key = (kind, record[kind])
            if key in owners:
                union(base_id, owners[key])
            else:
                owners[key] = base_id
    components: dict[str, list[str]] = defaultdict(list)
    for base_id in base_ids:
        components[find(base_id)].append(base_id)
    base_to_split = {}
    for component in components.values():
        key = "|".join(sorted(component))
        bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:12], 16) % 1000
        split = "train" if bucket < 800 else "val" if bucket < 900 else "development_test"
        for base_id in component:
            base_to_split[base_id] = split
    split_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in samples:
        record["split"] = base_to_split[record["base_geometry_family_id"]]
        split_records[record["split"]].append(record)
    for split in ("train", "val", "development_test"):
        path = OUTPUT / "splits" / f"{split}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in sorted(split_records[split], key=lambda item: item["sample_id"]):
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    leakage = {}
    for signature_name in ("exact_graph_signature", "near_graph_signature"):
        sets = {
            split: {record[signature_name] for record in records}
            for split, records in split_records.items()
        }
        leakage[signature_name] = {
            "train_val": len(sets.get("train", set()) & sets.get("val", set())),
            "train_development_test": len(sets.get("train", set()) & sets.get("development_test", set())),
            "val_development_test": len(sets.get("val", set()) & sets.get("development_test", set())),
        }
    base_leakage = {}
    base_sets = {
        split: {record["base_geometry_family_id"] for record in records}
        for split, records in split_records.items()
    }
    base_leakage["train_val"] = len(base_sets.get("train", set()) & base_sets.get("val", set()))
    base_leakage["train_development_test"] = len(base_sets.get("train", set()) & base_sets.get("development_test", set()))
    base_leakage["val_development_test"] = len(base_sets.get("val", set()) & base_sets.get("development_test", set()))
    leakage_detected = any(value for group in leakage.values() for value in group.values()) or any(base_leakage.values())
    return {
        "split_sample_counts": {split: len(records) for split, records in sorted(split_records.items())},
        "split_base_group_counts": {split: len(base_sets[split]) for split in sorted(base_sets)},
        "signature_component_count": len(components),
        "base_signature_leakage": base_leakage,
        "graph_signature_leakage": leakage,
        "leakage_detected": bool(leakage_detected),
    }


def _write_dataset_artifacts(
    state: BuildState,
    *,
    legacy_valid: list[dict[str, Any]],
    legacy_rejected: list[dict[str, Any]],
    slot_partition_count: int,
    split_audit: dict[str, Any],
    source_audit: dict[str, Any],
) -> dict[str, Any]:
    instance_counts = Counter()
    source_counts = Counter()
    derivation_counts = Counter()
    split_instance_counts: dict[str, Counter] = defaultdict(Counter)
    for record in state.samples:
        instance_counts.update(record["macro_instance_counts"])
        source_counts[record["source_kind"]] += 1
        derivation_counts[record["derivation_kind"]] += 1
        split_instance_counts[record["split"]].update(record["macro_instance_counts"])
    missing_or_under = {macro: instance_counts[macro] for macro in TARGET_CLASSES if instance_counts[macro] < 300}
    manifest_path = OUTPUT / "manifests" / "sample_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in sorted(state.samples, key=lambda item: item["sample_id"]):
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    automatic_pass = (
        len(state.samples) == TARGET_SAMPLE_COUNT
        and not missing_or_under
        and not split_audit["leakage_detected"]
        and slot_partition_count >= 80
        and source_counts["legacy_identity"] == len(legacy_valid)
        and all(not (set(record["macro_instance_counts"]) & REBUILD_ONLY) for record in state.samples if record["source_kind"].startswith("legacy"))
    )
    payload = {
        "schema_version": "topocae_p1_b_corrected_balanced_dataset_1",
        "automatic_pass": automatic_pass,
        "dataset_root": str(OUTPUT.resolve()),
        "sample_count": len(state.samples),
        "instance_count": sum(instance_counts.values()),
        "target_classes": TARGET_CLASSES,
        "instance_counts": {macro: instance_counts[macro] for macro in TARGET_CLASSES},
        "minimum_300_failures": missing_or_under,
        "source_counts": {
            "legacy_identity": source_counts["legacy_identity"],
            "legacy_derived": source_counts["legacy_derived"],
            "new_generator": source_counts["new_generator"],
        },
        "derivation_counts": dict(sorted(derivation_counts.items())),
        "legacy_source_audit_sha256": _sha256(SOURCE_AUDIT),
        "legacy_selected_count": source_audit["selected_safe_sample_count"],
        "legacy_reextract_accepted_count": len(legacy_valid),
        "legacy_reextract_rejected_count": len(legacy_rejected),
        "legacy_reextract_rejections": legacy_rejected,
        "human_gate_hashes": {"H_B1": H_B1_CSV_SHA256, "H_B2": H_B2_CSV_SHA256},
        "slot_legal_partition_count": slot_partition_count,
        "split_audit": split_audit,
        "split_instance_counts": {
            split: {macro: counts[macro] for macro in TARGET_CLASSES}
            for split, counts in sorted(split_instance_counts.items())
        },
        "contracts": {
            "old_slot_or_rib_web_reused": False,
            "new_slot_generator_gate_passed": True,
            "new_rib_web_generator_gate_passed": True,
            "graph_schema": "topocae_graph_v2_relative_1_1",
            "base_family_grouped_split": True,
            "rotation_scale_partition_derivatives_same_split": True,
            "real_data_used": False,
            "source_dataset_modified": False,
        },
    }
    meta_path = OUTPUT / "dataset_meta.json"
    meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_path = OUTPUT / "reports" / "dataset_summary.zh-CN.md"
    summary_path.write_text(_dataset_summary(payload), encoding="utf-8")
    artifact_manifest = {
        "schema_version": "topocae_p1_b_corrected_dataset_artifact_manifest_1",
        "automatic_pass": automatic_pass,
        "dataset_meta_sha256": _sha256(meta_path),
        "sample_manifest_sha256": _sha256(manifest_path),
        "summary_sha256": _sha256(summary_path),
        "sample_count": len(state.samples),
        "aggregate_graph_bytes": sum(Path(record["graph_path"]).stat().st_size for record in state.samples),
        "aggregate_step_bytes": sum(Path(record["step_path"]).stat().st_size for record in state.samples),
    }
    (OUTPUT / "manifests" / "artifact_manifest.json").write_text(
        json.dumps(artifact_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return payload


def _labels_from_graph(graph: dict[str, Any], semantic_index: dict[str, Any]) -> list[dict[str, Any]]:
    id_to_key = semantic_index.get("id_to_key", {})
    labels = []
    for node in graph.get("nodes", []):
        semantic_id = int(node.get("semantic_label", 0) or 0)
        role = id_to_key.get(semantic_id, id_to_key.get(str(semantic_id), "stock_face"))
        labels.append({
            "semantic_label": semantic_id,
            "instance_id": int(node.get("instance_id", 0) or 0),
            "macro_feature": str(node.get("macro_feature", "stock")),
            "macro_feature_id": int(node.get("macro_feature_id", 0) or 0),
            "semantic_role": str(role),
        })
    return labels


def _slot_labels(result: Any, semantic_index: dict[str, Any], ontology_maps: Any) -> tuple[list[dict[str, Any]], list[tuple[Any, dict[str, Any]]]]:
    macro = ontology_maps.resolve_macro_feature(result.parameters["macro_key"])
    labels = []
    items = []
    for face in TopologyExplorer(result.shape).faces():
        role = next((candidate_role for candidate_face, candidate_role in result.parameters["_face_role_items"] if face.IsSame(candidate_face)), None)
        if role is None:
            label = _stock_label()
        else:
            semantic_key = semantic_index["alias_to_key"][role]
            label = {
                "semantic_label": semantic_index["key_to_id"][semantic_key],
                "instance_id": 1,
                "macro_feature": macro.key,
                "macro_feature_id": macro.id,
                "semantic_role": role,
            }
        labels.append(label)
        items.append((face, label))
    return labels, items


def _rib_web_labels(result: Any, semantic_index: dict[str, Any], ontology_maps: Any) -> list[dict[str, Any]]:
    labels = []
    for face in TopologyExplorer(result.shape).faces():
        match = next((tag for tag in result.face_tags if face.IsSame(tag.face)), None)
        if match is None:
            labels.append(_stock_label())
        else:
            macro = ontology_maps.resolve_macro_feature(match.macro_key)
            semantic_key = semantic_index["alias_to_key"][match.semantic_role]
            labels.append({
                "semantic_label": semantic_index["key_to_id"][semantic_key],
                "instance_id": int(match.instance_id),
                "macro_feature": macro.key,
                "macro_feature_id": macro.id,
                "semantic_role": match.semantic_role,
            })
    return labels


def _stock_label() -> dict[str, Any]:
    return {
        "semantic_label": 0,
        "instance_id": 0,
        "macro_feature": "stock",
        "macro_feature_id": 0,
        "semantic_role": "stock_face",
    }


def _partition_slot_stock_face(
    shape: Any,
    face_items: list[tuple[Any, dict[str, Any]]],
    labels: list[dict[str, Any]],
    length: float,
    width: float,
) -> tuple[Any, list[dict[str, Any]]] | None:
    target = None
    for face, label in face_items:
        if int(label["instance_id"]) != 0:
            continue
        surface = BRepAdaptor_Surface(face)
        if surface.GetType() != GeomAbs_Plane:
            continue
        plane = surface.Plane()
        if abs(plane.Axis().Direction().Z()) > 0.99 and abs(plane.Location().Z()) < 1e-5:
            target = face
            break
    if target is None:
        return None
    splitter = BRepFeat_SplitShape(shape)
    edge = BRepBuilderAPI_MakeEdge(
        gp_Pnt(-length / 2.0, 0.0, 0.0), gp_Pnt(length / 2.0, 0.0, 0.0)
    ).Edge()
    splitter.Add(edge, target)
    splitter.Build()
    partitioned = splitter.Shape()
    old_faces = [item[0] for item in face_items]
    new_labels = []
    unmatched = 0
    for new_face in TopologyExplorer(partitioned).faces():
        match_index = next((index for index, old_face in enumerate(old_faces) if new_face.IsSame(old_face)), None)
        if match_index is None:
            new_labels.append(_stock_label())
            unmatched += 1
        else:
            new_labels.append(deepcopy(labels[match_index]))
    if unmatched < 2 or len(new_labels) <= len(labels):
        return None
    if sum(int(label["instance_id"]) > 0 for label in new_labels) != sum(int(label["instance_id"]) > 0 for label in labels):
        return None
    if not BRepCheck_Analyzer(partitioned).IsValid() or len(list(TopologyExplorer(partitioned).solids())) != 1:
        return None
    return partitioned, new_labels


def _face_sequence_signature(shape: Any) -> list[dict[str, Any]]:
    result = []
    for face in TopologyExplorer(shape).faces():
        props = GProp_GProps()
        brepgprop.SurfaceProperties(face, props)
        result.append({"primitive_surface": _primitive_name(face), "area": float(props.Mass())})
    return result


def _expected_sequence_from_graph(graph: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "primitive_surface": str(node.get("primitive_surface", "unknown_surface")),
            "area": float(node.get("attributes", {}).get("physical", {}).get("area", 0.0) or 0.0),
        }
        for node in graph.get("nodes", [])
    ]


def _validate_face_sequence(expected: list[dict[str, Any]], graph: dict[str, Any], *, relative_area_tolerance: float) -> None:
    nodes = graph.get("nodes", [])
    if len(expected) != len(nodes):
        raise ValueError(f"STEP face count changed: expected {len(expected)}, got {len(nodes)}")
    for index, (reference, node) in enumerate(zip(expected, nodes)):
        primitive = str(node.get("primitive_surface", "unknown_surface"))
        area = float(node.get("attributes", {}).get("physical", {}).get("area", 0.0) or 0.0)
        relative_error = abs(area - reference["area"]) / max(1.0, abs(reference["area"]))
        if primitive != reference["primitive_surface"] or relative_error > relative_area_tolerance:
            raise ValueError(
                f"STEP face sequence mismatch at {index}: {reference['primitive_surface']}/{reference['area']} vs {primitive}/{area}"
            )


def _primitive_name(face: Any) -> str:
    return PRIMITIVE_NAMES.get(int(BRepAdaptor_Surface(face).GetType()), "unknown_surface")


def _rotate_shape(shape: Any, axis: Iterable[float], angle: float) -> Any:
    values = list(axis)
    transform = gp_Trsf()
    transform.SetRotation(gp_Ax1(gp_Pnt(0.0, 0.0, 0.0), gp_Dir(*values)), float(angle))
    return BRepBuilderAPI_Transform(shape, transform, True).Shape()


def _scale_shape(shape: Any, factor: float) -> Any:
    transform = gp_Trsf()
    transform.SetScale(gp_Pnt(0.0, 0.0, 0.0), float(factor))
    return BRepBuilderAPI_Transform(shape, transform, True).Shape()


def _signature_hash(signature: Any) -> str:
    canonical = json.dumps(signature, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest().upper()


def _dataset_summary(payload: dict[str, Any]) -> str:
    lines = [
        "# P1-B 修正平衡数据集报告",
        "",
        f"- 自动结论：{'PASS' if payload['automatic_pass'] else 'FAIL'}",
        f"- 样本数：{payload['sample_count']}",
        f"- 有效实例数：{payload['instance_count']}",
        f"- 旧安全 STEP Graph v2 重提：{payload['source_counts']['legacy_identity']}",
        f"- 新槽/rib/web 生成：{payload['source_counts']['new_generator']}",
        f"- 旧安全 geometry rotation/scale 派生：{payload['source_counts']['legacy_derived']}",
        f"- 合法 face partition：{payload['slot_legal_partition_count']}",
        f"- split 泄漏：{payload['split_audit']['leakage_detected']}",
        "",
        "## 18 类实例覆盖",
        "",
    ]
    for macro in TARGET_CLASSES:
        lines.append(f"- {macro}: {payload['instance_counts'][macro]}")
    lines.extend([
        "",
        "槽与 rib/web 全部来自通过 H-B1/H-B2 的新生成器；任何含旧错误概念的旧样本均未复用。",
        "旋转、缩放和 face partition 派生与 base geometry family 固定在同一 split。",
        "",
    ])
    return "\n".join(lines)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


@contextmanager
def _silence_native_output():
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    null_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null_fd, 1)
        os.dup2(null_fd, 2)
        yield
    finally:
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)
        os.close(null_fd)


if __name__ == "__main__":
    raise SystemExit(main())
