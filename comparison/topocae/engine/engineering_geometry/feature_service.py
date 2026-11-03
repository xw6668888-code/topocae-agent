"""Ground validated language plans in frozen P1 feature objects."""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

from build_cases import ROOT, build_cad, build_cae, hole_geometry, load_case, sha, write_json
from plan_contract import PlanError, validate_plan

FROZEN_CHECKPOINT = "491655bfec5591c5a6a1b7f6300294f4818c29bf60b4c1b8dd5ff6078a057054"
INFERENCE_SCRIPT = ROOT / "paper_release/topocae_p1_legacy_baseline_v1/scripts/run_inference.py"


def recognize(step_path, directory):
    directory.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run([sys.executable, str(INFERENCE_SCRIPT), str(step_path),
                           "--output-dir", str(directory), "--device", "cpu"],
                          capture_output=True, encoding="utf-8", errors="replace")
    (directory / "inference.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode:
        raise PlanError("recognition_failed", "See inference.log")
    return directory / (step_path.stem + "_recognition.json")


def catalog_for(source, prediction_path):
    case = load_case(source=source, prediction_path=prediction_path)
    _, _, faces, prediction, _ = case
    if prediction["model"]["checkpoint_sha256"].lower() != FROZEN_CHECKPOINT:
        raise PlanError("unregistered_recognizer")
    records = []
    used = set()
    for item in prediction["feature_objects"]:
        numbers = item["original_face_numbers"]
        if not numbers or len(numbers) != len(set(numbers)) or any(n < 1 or n > len(faces) for n in numbers):
            raise PlanError("invalid_prediction_face_indices")
        if used.intersection(numbers):
            raise PlanError("overlapping_prediction_objects")
        used.update(numbers)
        params, error = {}, None
        if item["class"] == "blind_hole_round" and item["topology_valid"]:
            try:
                hole = hole_geometry(item, faces)
                params = {"radius_mm": hole["radius_before_mm"], "depth_mm": hole["depth_mm"]}
            except RuntimeError as exc:
                error = str(exc)
        records.append({"instance_id": "f%04d" % item["object_index"],
                        "class": item["class"], "parameters": params,
                        "center_mm": item["geometry_summary"]["area_weighted_center"],
                        "confidence": item["class_confidence"], "topology_valid": item["topology_valid"],
                        "geometry_error": error, "prediction_object": item})
    catalog = {"schema_version": "feature_catalog/1", "source_sha256": sha(source),
               "prediction_sha256": sha(prediction_path), "model": prediction["model"],
               "truth_used": False, "instances": records}
    return case, catalog


def public_catalog(catalog):
    return {"schema_version": catalog["schema_version"], "geometry_version": catalog["source_sha256"],
            "instances": [{"instance_id": r["instance_id"], "class": r["class"],
                           "parameters": r["parameters"], "topology_valid": r["topology_valid"]}
                          for r in catalog["instances"]]}


def resolve_target(target, catalog):
    matches = [r for r in catalog["instances"] if r["class"] == target["feature_class"]]
    if target["radius_mm"] is not None:
        if matches and any("radius_mm" not in r["parameters"] for r in matches):
            raise PlanError("unsupported", "A verified radius is unavailable for this feature class")
        matches = [r for r in matches if abs(r["parameters"]["radius_mm"] - target["radius_mm"]) < 1e-6]
    if not matches:
        raise PlanError("no_target", "No recognized object matches this selector")
    if any(not r["topology_valid"] or r["geometry_error"] for r in matches):
        raise PlanError("invalid_geometry", "A selected object failed its geometry checks")
    if target["scope"] == "single" and len(matches) != 1:
        raise PlanError("needs_clarification", "Multiple matching instances: " + ", ".join(r["instance_id"] for r in matches))
    return matches


def ground(plan, catalog):
    validate_plan(plan)
    if plan["status"] != "ready":
        return {"status": plan["status"], "reason": plan["reason"], "clarifications": plan["clarifications"]}
    bindings = {}
    try:
        for op in plan["operations"]:
            action = op["action"]
            if "target" not in op:
                continue
            allowed = {"set_blind_hole_radius": "blind_hole_round", "fix_features": "counterbore_hole",
                       "apply_force": "obround_hole"}[action]
            if op["target"]["feature_class"] != allowed:
                raise PlanError("unsupported", "Feature class is outside this operator's contract")
            records = resolve_target(op["target"], catalog)
            if action == "set_blind_hole_radius":
                radii = {r["parameters"].get("radius_mm") for r in records}
                if None in radii or len(radii) != 1:
                    raise PlanError("unsupported", "This edit requires a common verified source radius")
                if not 0 < op["new_radius_mm"] < next(iter(radii)):
                    raise PlanError("unsupported", "Only radius reduction is currently implemented")
            bindings[action] = records
        if "fix_features" in bindings:
            fixed = {n for r in bindings["fix_features"] for n in r["prediction_object"]["original_face_numbers"]}
            loaded = {n for r in bindings["apply_force"] for n in r["prediction_object"]["original_face_numbers"]}
            if fixed & loaded:
                raise PlanError("overlapping_regions")
    except PlanError as exc:
        return {"status": exc.code, "reason": str(exc)}
    return {"status": "ready", "bindings": bindings, "source_sha256": catalog["source_sha256"],
            "prediction_sha256": catalog["prediction_sha256"]}


def execute(plan, grounded, case, output):
    if grounded["status"] != "ready":
        raise PlanError("nonready_execution")
    if sha(case[0]) != grounded["source_sha256"] or sha(case[4]) != grounded["prediction_sha256"]:
        raise PlanError("stale_grounding")
    ops = {op["action"]: op for op in plan["operations"]}
    bindings = grounded["bindings"]
    def objects(action):
        return [r["prediction_object"] for r in bindings[action]]
    if plan["domain"] == "cad":
        radius = bindings["set_blind_hole_radius"][0]["parameters"]["radius_mm"]
        return build_cad(case=case, output=output, selected=objects("set_blind_hole_radius"),
                         old_radius=radius, new_radius=ops["set_blind_hole_radius"]["new_radius_mm"])
    material = {k: v for k, v in ops["set_material"].items() if k != "action"}
    force = ops["apply_force"]
    return build_cae(case=case, output=output, fixed=objects("fix_features"), loaded=objects("apply_force"),
                     material=material, mesh_mm=ops["set_global_mesh"]["size_mm"], force_N=force["magnitude_N"],
                     force_basis=force["basis"], direction=force["direction"])
