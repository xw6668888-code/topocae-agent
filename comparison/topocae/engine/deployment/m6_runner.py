"""Prepare, run, and audit the 13 frozen-E3 G10 engineering executions.

The default commands only freeze/prepare jobs.  CAD edits and Abaqus solves are
separate explicit commands, use unique run directories, and emit exactly one
case-level execution receipt.  Historical G01-G09 files are always read-only.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Iterator

from g10_common import (
    ARTIFACT, REPORT, ROOT, checked, descriptor, digest, now, read_json,
    resource_lock, sha256, write_json, write_text,
)
from m6_contract import (
    ABAQUS_RELEASE, BindingError, CAUSALITY_PATH, DEPLOYMENT_SEED, E3_SCHEMA,
    E3_SYSTEM, MANIFEST_PATH, MATERIALS, M6_ROOT, OCC_VERSION,
    assert_frozen_binding, build_manifest, case_from_manifest, object_id,
    resolve_plan, validate_prediction, verify_manifest,
)


CASE_SUITE = ROOT / "tools/aei_case_suite"
PILOT = ROOT / "tools/aei_two_case_pilot"
ABAQUS = Path("F:/ABAQUS2020/commands/abaqus.bat")
RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def _safe_run_id(value: str) -> str:
    if not RUN_ID.fullmatch(value) or value in {".", ".."}:
        raise ValueError("run-id must be 1-64 safe filename characters")
    return value


def _vector_add(a: list[float], b: list[float]) -> list[float]:
    return [x + y for x, y in zip(a, b)]


def _vector_mul(a: list[float], value: float) -> list[float]:
    return [x * value for x in a]


def _vector_dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _unit(a: list[float]) -> list[float]:
    length = math.sqrt(sum(value * value for value in a))
    if length <= 1e-12:
        raise BindingError("zero_direction")
    return [value / length for value in a]


@contextmanager
def _case_suite_imports() -> Iterator[None]:
    original = list(sys.path)
    for path in (str(CASE_SUITE), str(PILOT)):
        if path not in sys.path:
            sys.path.insert(0, path)
    try:
        yield
    finally:
        sys.path[:] = original


def _surface_family(row: dict[str, Any]) -> str:
    if "radius_mm" in row:
        return "cylinder"
    if "normal" in row:
        return "plane"
    return row.get("type", "other")


def _geometry_summary(obj: dict[str, Any], faces: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [faces[number - 1] for number in obj["original_face_numbers"]]
    area = sum(row["area_mm2"] for row in rows)
    center = [
        sum(row["center_mm"][axis] * row["area_mm2"] for row in rows) / area
        for axis in range(3)
    ]
    families: dict[str, int] = {}
    for row in rows:
        family = _surface_family(row)
        families[family] = families.get(family, 0) + 1
    return {
        "member_face_count": len(rows),
        "total_area": area,
        "surface_family_counts": families,
        "area_weighted_center": center,
        "positive_radii": list(obj.get("parameters", {}).get("radii_mm", [])),
    }


def _load_adapter(case: dict[str, Any]) -> dict[str, Any]:
    """Load source geometry and adapt only the frozen E3 output, never truth."""
    source = checked(case["source_step"])
    prediction_path = checked(case["prediction"])
    prediction = read_json(prediction_path)
    validate_prediction(prediction, case["source_step"])
    assert_frozen_binding(case["frozen_binding"], case["plan"], prediction)
    with _case_suite_imports():
        import OCC
        if str(getattr(OCC, "VERSION", "")) != OCC_VERSION:
            raise BindingError("pythonocc_version_changed", str(getattr(OCC, "VERSION", "missing")))
        from build_cases import properties, read_step
        from preflight import measured
        from OCC.Extend.TopologyUtils import TopologyExplorer

        shape = read_step(source)
        occ_faces = list(TopologyExplorer(shape).faces())
        if len(occ_faces) != len(prediction["face_macro_classes"]):
            raise BindingError("step_prediction_face_count_changed", case["case_id"])
        face_rows = [measured(face, index + 1)
                     for index, face in enumerate(occ_faces)]
        for row, macro_class in zip(face_rows, prediction["face_macro_classes"]):
            row["predicted_class"] = macro_class
        objects = deepcopy(prediction["feature_objects"])
        for obj in objects:
            obj["object_id"] = object_id(prediction, obj)
            obj["geometry_version"] = prediction["input_step"]["sha256"]
            obj["recognition_source"] = E3_SYSTEM
            obj["geometry_summary"] = _geometry_summary(obj, face_rows)
        volume, center = properties(shape, True)
    model = {
        "name": E3_SYSTEM,
        "deployment_seed": DEPLOYMENT_SEED,
        "pythonocc_version": OCC_VERSION,
        "checkpoint_sha256": prediction["source_identity"]["R"]["checkpoint"]["sha256"],
        "source_identity_digest": digest(prediction["source_identity"]),
        "composite_system": True,
    }
    compatible_prediction = deepcopy(prediction)
    compatible_prediction.update({
        "input": prediction["input_step"],
        "model": model,
        "feature_objects": objects,
    })
    data = {
        "schema": "g10_e3_engineering_object_service_v1",
        "part": case["part"],
        "source_step": case["source_step"],
        "source_sha256": case["source_step"]["sha256"],
        "prediction": case["prediction"],
        "prediction_sha256": case["prediction"]["sha256"],
        "system": E3_SYSTEM,
        "deployment_seed": DEPLOYMENT_SEED,
        "geometry_version": case["source_step"]["sha256"],
        "volume_mm3": volume,
        "center_of_mass_mm": center,
        "objects": objects,
        "faces": face_rows,
        "frozen_binding": case["frozen_binding"],
        "object_id_scope": "valid only for this exact STEP geometry version",
        "truth_used": False,
    }
    return {
        "source": source,
        "prediction_path": prediction_path,
        "prediction": prediction,
        "compatible_prediction": compatible_prediction,
        "shape": shape,
        "occ_faces": occ_faces,
        "data": data,
        "case_tuple": (source, shape, occ_faces, compatible_prediction, prediction_path),
    }


def _select(data: dict[str, Any], selector: dict[str, Any]) -> list[dict[str, Any]]:
    classes = set(selector["feature_classes"])
    radius = selector.get("radius_mm")
    matches = []
    for obj in data["objects"]:
        if obj["class"] not in classes:
            continue
        radii = obj["geometry_summary"]["positive_radii"]
        if radius is not None and not any(abs(value - radius) <= 1e-5 for value in radii):
            continue
        matches.append(obj)
    if not matches:
        raise BindingError("no_matching_object")
    if selector.get("scope") == "single" and len(matches) != 1:
        raise BindingError("ambiguous_object_selection")
    return matches


def _roles(data: dict[str, Any], objects: list[dict[str, Any]],
           role: str) -> list[dict[str, Any]]:
    groups = []
    for obj in objects:
        group = obj.get("roles", {}).get(role)
        if not isinstance(group, dict) or not group.get("face_numbers"):
            raise BindingError("missing_or_ambiguous_role", f"{obj['class']}:{role}")
        if group["object_index"] != obj["object_index"] or group["class"] != obj["class"]:
            raise BindingError("role_object_identity_changed")
        record = deepcopy(group)
        record["object_id"] = obj["object_id"]
        record["geometry_version"] = data["geometry_version"]
        groups.append(record)
    return groups


def _compile_cae(case: dict[str, Any], adapter: dict[str, Any], case_root: Path,
                 attempt_dir: Path, manifest_descriptor: dict[str, Any]) -> Path:
    """Compile a case-specific Abaqus program with immutable E3 bindings."""
    plan = case["plan"]
    operations = {operation["action"]: operation for operation in plan["operations"]}
    material = MATERIALS[operations["set_material"]["material_id"]]
    data = adapter["data"]
    fixed = _roles(data, _select(data, operations["fix_features"]["target"]),
                   operations["fix_features"]["role"])
    local = []
    if "set_local_mesh" in operations:
        operation = operations["set_local_mesh"]
        local.append({
            "size_mm": operation["size_mm"],
            "regions": _roles(data, _select(data, operation["target"]),
                              "whole_feature"),
        })

    loads = []
    resultant = [0.0, 0.0, 0.0]
    force_scale = 0.0
    for action in ("apply_force", "apply_traction", "apply_pressure", "apply_centrifugal"):
        if action not in operations:
            continue
        operation = operations[action]
        if action == "apply_centrifugal":
            axis = _unit(operation["axis_direction"])
            relative = [data["center_of_mass_mm"][i] - operation["axis_point_mm"][i]
                        for i in range(3)]
            radial = [relative[i] - axis[i] * _vector_dot(relative, axis)
                      for i in range(3)]
            resultant = _vector_mul(
                radial,
                material["density_tonne_mm3"] * data["volume_mm3"]
                * operation["angular_velocity_rad_s"] ** 2,
            )
            force_scale = math.sqrt(sum(value * value for value in resultant))
            loads.append({
                "kind": "centrifugal",
                "angular_velocity_rad_s": operation["angular_velocity_rad_s"],
                "axis_point_mm": operation["axis_point_mm"],
                "axis_direction": axis,
            })
            continue
        groups = _roles(data, _select(data, operation["target"]), operation["role"])
        total_area = sum(group["area_mm2"] for group in groups)
        fixed_faces = {number for group in fixed for number in group["face_numbers"]}
        for group in groups:
            if fixed_faces & set(group["face_numbers"]):
                raise BindingError("overlapping_fixed_and_loaded_regions")
            if action == "apply_force":
                amount = operation["magnitude_N"] if operation["basis"] == "per_instance" \
                    else operation["magnitude_N"] * group["area_mm2"] / total_area
                direction = _unit(operation["direction"])
                contribution = _vector_mul(direction, amount)
                loads.append({**group, "kind": "force", "force_N": amount,
                              "direction": direction})
                force_scale += amount
            elif action == "apply_traction":
                direction = _unit(operation["direction"])
                amount = operation["magnitude_MPa"] * group["area_mm2"]
                contribution = _vector_mul(direction, amount)
                loads.append({**group, "kind": "traction",
                              "magnitude_MPa": operation["magnitude_MPa"],
                              "direction": direction})
                force_scale += amount
            else:
                contribution = [0.0, 0.0, 0.0]
                for number in group["face_numbers"]:
                    face = data["faces"][number - 1]
                    if "normal" in face:
                        contribution = _vector_add(
                            contribution,
                            _vector_mul(face["normal"],
                                        -operation["magnitude_MPa"] * face["area_mm2"]),
                        )
                    elif "radius_mm" not in face:
                        raise BindingError("pressure_resultant_geometry_unsupported")
                loads.append({**group, "kind": "pressure",
                              "magnitude_MPa": operation["magnitude_MPa"]})
                force_scale += math.sqrt(sum(value * value for value in contribution))
            resultant = _vector_add(resultant, contribution)

    semantic_plan = case_root / "semantic_plan.json"
    write_json(semantic_plan, plan, overwrite=False)
    with _case_suite_imports():
        from cae_executor import import_face_record

        source_faces = [import_face_record(face, index + 1)
                        for index, face in enumerate(adapter["occ_faces"])]
    template_path = CASE_SUITE / "abaqus_generic.template.py"
    verifier_path = CASE_SUITE / "abaqus_verification.py"
    binding_files = [
        manifest_descriptor,
        case["source_step"],
        case["prediction"],
        descriptor(semantic_plan),
        descriptor(template_path),
        descriptor(verifier_path),
    ]
    payload = {
        "schema": "g10_e3_cae_execution_plan_v1",
        "case_id": case["case_id"],
        "part": case["part"],
        "source_step": str(adapter["source"]),
        "source_sha256": case["source_step"]["sha256"],
        "geometry_version": case["source_step"]["sha256"],
        "prediction_path": str(adapter["prediction_path"]),
        "prediction_sha256": case["prediction"]["sha256"],
        "model": adapter["compatible_prediction"]["model"],
        "manifest_sha256": manifest_descriptor["sha256"],
        "required_abaqus_release": ABAQUS_RELEASE,
        "semantic_plan_sha256": descriptor(semantic_plan)["sha256"],
        "frozen_binding": case["frozen_binding"],
        "binding_files": binding_files,
        "script_directory": str(case_root.resolve()),
        "run_directory": str(attempt_dir.resolve()),
        "execution_mode": "g10_frozen_e3_user_confirmed_reference_plan",
        "truth_used": False,
        "material": material,
        "global_mesh_mm": operations["set_global_mesh"]["size_mm"],
        "local_mesh": local,
        "fixed_regions": fixed,
        "loads": loads,
        "expected_force_N": resultant,
        "force_balance_scale_N": force_scale,
        "mapping_tolerances": {"vertex_distance_mm": 1e-4, "relative_area": 0.005},
        "source_faces": source_faces,
    }
    execution_plan = case_root / "execution_plan.json"
    write_json(execution_plan, payload, overwrite=False)
    template = template_path.read_text(encoding="ascii")
    verification = verifier_path.read_text(encoding="ascii")
    template = template.replace("# __VERIFICATION_FUNCTION__", verification)
    old_run = "RUN_DIR = tempfile.mkdtemp(prefix='run_',dir=SCRIPT_DIR)"
    new_run = ("RUN_DIR = PLAN['run_directory']\n"
               "if not os.path.isdir(RUN_DIR):\n"
               "    os.makedirs(RUN_DIR)")
    if template.count(old_run) != 1:
        raise BindingError("abaqus_template_run_directory_hook_changed")
    template = template.replace(old_run, new_run)
    old_job = ("    jobname='AEI_'+PLAN['case_id']\n"
               "    suffix=1\n"
               "    while jobname in mdb.jobs:\n"
               "        jobname='AEI_'+PLAN['case_id']+'_%d'%suffix\n"
               "        suffix+=1")
    if template.count(old_job) != 1:
        raise BindingError("abaqus_template_job_name_hook_changed")
    template = template.replace(old_job, "    jobname='final'")
    binding_guard = """
def file_sha256(path):
    value=hashlib.sha256()
    with open(path,'rb') as stream:
        while True:
            block=stream.read(1024*1024)
            if not block:
                break
            value.update(block)
    return value.hexdigest()


def verify_binding_files():
    for record in PLAN['binding_files']:
        path=record['path']
        if not os.path.isfile(path) or os.path.getsize(path)!=record['bytes'] or file_sha256(path)!=record['sha256']:
            raise RuntimeError('Frozen G10 binding file changed: '+path)


"""
    marker = "# __VERIFICATION_FUNCTION__"
    # The marker was replaced above; insert before the verifier's first function.
    verifier_start = "def verify_odb(odb, plan):"
    if template.count(verifier_start) != 1:
        raise BindingError("abaqus_verifier_hook_changed")
    template = template.replace(verifier_start, binding_guard + verifier_start)
    old_try = "try:\n    main(audit)"
    if template.count(old_try) != 1:
        raise BindingError("abaqus_template_binding_call_hook_changed")
    template = template.replace(old_try, "try:\n    verify_binding_files()\n    main(audit)")
    old_audit = ("audit={'case':PLAN['part'],'status':'FAIL','truth_used':False,'execution_mode':PLAN['execution_mode'],\n"
                 "       'model_checkpoint_sha256':PLAN['model']['checkpoint_sha256'],'prediction_sha256':PLAN['prediction_sha256']}")
    new_audit = ("audit={'case':PLAN['part'],'status':'FAIL','truth_used':False,'execution_mode':PLAN['execution_mode'],\n"
                 "       'model_checkpoint_sha256':PLAN['model']['checkpoint_sha256'],'prediction_sha256':PLAN['prediction_sha256'],\n"
                 "       'manifest_sha256':PLAN['manifest_sha256'],'semantic_plan_sha256':PLAN['semantic_plan_sha256'],\n"
                 "       'geometry_version':PLAN['geometry_version'],'frozen_binding':PLAN['frozen_binding']}")
    if template.count(old_audit) != 1:
        raise BindingError("abaqus_template_audit_hook_changed")
    template = template.replace(old_audit, new_audit)
    script = case_root / "final.py"
    rendered = template.replace(
        "__EMBEDDED_PLAN_JSON__",
        repr(json.dumps(payload, ensure_ascii=True, separators=(",", ":"))),
    )
    write_text(script, rendered, overwrite=False)
    _write_odb_reaudit(case_root, payload, verifier_path)
    return script


def _write_odb_reaudit(case_root: Path, payload: dict[str, Any],
                       verifier_path: Path) -> None:
    verifier = verifier_path.read_text(encoding="ascii")
    source = """# -*- coding: ascii -*-
from __future__ import print_function
from odbAccess import openOdb
import hashlib, json, os, sys, traceback
try:
    import builtins
except ImportError:
    import __builtin__ as builtins
sum, min, max, abs = builtins.sum, builtins.min, builtins.max, builtins.abs
PLAN = json.loads(__PLAN__)

__VERIFIER__

def sha(path):
    value=hashlib.sha256()
    with open(path,'rb') as stream:
        while True:
            block=stream.read(1024*1024)
            if not block:
                break
            value.update(block)
    return value.hexdigest()

out=os.path.join(PLAN['run_directory'],'independent_odb_audit.json')
odb_path=os.path.join(PLAN['run_directory'],'final.odb')
result={'schema':'g10_independent_odb_audit_v1','status':'FAIL','truth_used':False,
        'prediction_sha256':PLAN['prediction_sha256'],'manifest_sha256':PLAN['manifest_sha256'],
        'geometry_version':PLAN['geometry_version'],'odb_path':odb_path}
try:
    if not os.path.isfile(odb_path):
        raise RuntimeError('final.odb is absent')
    odb=openOdb(path=odb_path,readOnly=True)
    try:
        result.update(verify_odb(odb,PLAN))
        result['odb_sha256']=sha(odb_path)
        result['instance_count']=len(odb.rootAssembly.instances)
        result['element_count']=sum(len(instance.elements) for instance in odb.rootAssembly.instances.values())
        result['node_count']=sum(len(instance.nodes) for instance in odb.rootAssembly.instances.values())
        element_types=sorted(set(element.type for instance in odb.rootAssembly.instances.values() for element in instance.elements))
        result['element_types']=element_types
        if element_types != ['C3D10']:
            result['status']='FAIL'
            result['verification_error']='unexpected_element_types'
    finally:
        odb.close()
except Exception:
    result['error']=traceback.format_exc()
with open(out,'w') as handle:
    json.dump(result,handle,indent=2)
print(out)
if result.get('status')!='PASS':
    raise SystemExit(2)
"""
    source = source.replace("__PLAN__", repr(json.dumps(payload, ensure_ascii=True)))
    source = source.replace("__VERIFIER__", verifier)
    write_text(case_root / "odb_reaudit.py", source, overwrite=False)


def prepare_run(run_id: str, case_ids: list[str] | None = None) -> dict[str, Any]:
    run_id = _safe_run_id(run_id)
    manifest = verify_manifest() if MANIFEST_PATH.exists() else build_manifest()
    available = {case["case_id"] for case in manifest["cases"]}
    chosen = sorted(available if not case_ids else set(case_ids))
    unknown = set(chosen) - available
    if unknown:
        raise BindingError("unknown_case", ",".join(sorted(unknown)))
    run_root = M6_ROOT / "runs" / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    manifest_descriptor = descriptor(MANIFEST_PATH)
    rows = []
    try:
        with resource_lock("occ", f"g10-m6-prepare-{run_id}"):
            for case_id in chosen:
                case = case_from_manifest(manifest, case_id)
                case_root = run_root / case_id
                case_root.mkdir(parents=True, exist_ok=False)
                attempt_dir = case_root / "attempts" / "attempt_001"
                adapter = _load_adapter(case)
                write_json(case_root / "object_service.json", adapter["data"], overwrite=False)
                if case["domain"] == "cae":
                    attempt_dir.mkdir(parents=True, exist_ok=False)
                    program = _compile_cae(
                        case, adapter, case_root, attempt_dir, manifest_descriptor,
                    )
                    program_descriptor = descriptor(program)
                else:
                    write_json(case_root / "semantic_plan.json", case["plan"], overwrite=False)
                    program_descriptor = None
                job = {
                    "schema": "g10_m6_prepared_job_v1",
                    "prepared_utc": now(),
                    "run_id": run_id,
                    "case_id": case_id,
                    "part": case["part"],
                    "domain": case["domain"],
                    "manifest": manifest_descriptor,
                    "source_step": case["source_step"],
                    "prediction": case["prediction"],
                    "object_service": descriptor(case_root / "object_service.json"),
                    "semantic_plan": descriptor(case_root / "semantic_plan.json"),
                    "program": program_descriptor,
                    "attempt_directory": attempt_dir.resolve().as_posix(),
                    "frozen_binding": case["frozen_binding"],
                    "status": "prepared_not_executed",
                    "truth_used": False,
                }
                write_json(case_root / "job_spec.json", job, overwrite=False)
                rows.append({
                    "case_id": case_id,
                    "domain": case["domain"],
                    "job_spec": descriptor(case_root / "job_spec.json"),
                })
        result = {
            "schema": "g10_m6_prepared_run_v1",
            "prepared_utc": now(),
            "run_id": run_id,
            "manifest": manifest_descriptor,
            "cases": rows,
            "prepared_count": len(rows),
            "execution_count": 0,
            "status": "prepared_not_executed",
            "truth_used": False,
        }
        write_json(run_root / "PREPARED.json", result, overwrite=False)
        return result
    except BaseException as error:
        write_json(run_root / "PREPARATION_FAILURE.json", {
            "schema": "g10_m6_preparation_failure_v1",
            "at_utc": now(), "run_id": run_id,
            "error_type": type(error).__name__, "error": str(error),
            "prepared_cases": rows, "truth_used": False,
        }, overwrite=False)
        raise


def _verify_job(run_id: str, case_id: str, expected_domain: str) -> tuple[dict, dict, Path]:
    manifest = verify_manifest()
    case = case_from_manifest(manifest, case_id)
    if case["domain"] != expected_domain:
        raise BindingError("wrong_execution_domain")
    case_root = M6_ROOT / "runs" / _safe_run_id(run_id) / case_id
    job_path = case_root / "job_spec.json"
    job = read_json(job_path)
    if job.get("schema") != "g10_m6_prepared_job_v1" \
            or job.get("status") != "prepared_not_executed" \
            or job.get("truth_used") is not False:
        raise BindingError("invalid_prepared_job")
    for key in ("manifest", "source_step", "prediction", "object_service", "semantic_plan"):
        checked(job[key])
    if job.get("program") is not None:
        checked(job["program"])
    if job["manifest"] != descriptor(MANIFEST_PATH) \
            or job["source_step"] != case["source_step"] \
            or job["prediction"] != case["prediction"] \
            or job["frozen_binding"] != case["frozen_binding"]:
        raise BindingError("prepared_job_binding_changed")
    prediction = read_json(checked(job["prediction"]))
    validate_prediction(prediction, case["source_step"])
    assert_frozen_binding(case["frozen_binding"], case["plan"], prediction)
    receipt = case_root / "execution_receipt.json"
    if receipt.exists():
        raise FileExistsError(f"Execution receipt already exists: {receipt}")
    return case, job, case_root


@contextmanager
def _patched_cad_adapter(adapter: dict[str, Any]) -> Iterator[Any]:
    with _case_suite_imports():
        import cad_executor
        original_load = cad_executor.load
        original_select = cad_executor.select
        try:
            cad_executor.load = lambda _part: (adapter["case_tuple"], adapter["data"])
            cad_executor.select = _select
            yield cad_executor
        finally:
            cad_executor.load = original_load
            cad_executor.select = original_select


def _independent_cad_audit(case: dict[str, Any], job: dict[str, Any], output: Path,
                           legacy_audit_path: Path) -> dict[str, Any]:
    audit = read_json(legacy_audit_path)
    with _case_suite_imports():
        from build_cases import properties, read_step
        from OCC.Extend.TopologyUtils import TopologyExplorer

        reread = read_step(output)
        independent_volume, _ = properties(reread, True)
        solids = len(list(TopologyExplorer(reread).solids()))
    checks = {
        "legacy_postcondition_pass": audit.get("status") == "PASS",
        "truth_not_used": audit.get("truth_used") is False,
        "prediction_hash_bound": audit.get("prediction_sha256") == job["prediction"]["sha256"],
        "source_version_bound": audit.get("source_sha256") == case["source_step"]["sha256"],
        "source_unchanged": audit.get("source_file_unchanged", True) is True,
        "output_hash_bound": audit.get("output_sha256", sha256(output)) == sha256(output),
        "roundtrip_one_valid_solid": solids == 1,
        "roundtrip_volume_matches_executor": abs(
            independent_volume - float(audit["volume_after_mm3"])
        ) <= max(1e-6, abs(independent_volume) * 1e-10),
    }
    action = case["plan"]["operations"][0]["action"]
    if action == "set_hole_radius" and "checks" in audit:
        radius_checks = [row.get("radius_and_axis_extent_verified") is True
                         for row in audit["checks"]
                         if "radius_and_axis_extent_verified" in row]
        checks["radius_axis_extent_check"] = bool(radius_checks) and all(radius_checks)
    if action == "set_hole_radius" and "target_checks" in audit:
        radius_checks = [row.get("pass") is True for row in audit["target_checks"]]
        checks["radius_axis_extent_check"] = bool(radius_checks) and all(radius_checks)
    if action == "set_hole_spacing":
        checks["spacing_and_central_anchor_check"] = any(
            row.get("central_hole_unchanged") is True for row in audit.get("checks", [])
        ) and all(row.get("translated_hole_geometry_verified") is True
                  for row in audit.get("checks", [])
                  if "translated_hole_geometry_verified" in row)
    if action == "set_thickness":
        thickness_checks = [row.get("symmetric_thickness_verified") is True
                            for row in audit.get("checks", [])
                            if "symmetric_thickness_verified" in row]
        checks["symmetric_midplane_check"] = bool(thickness_checks) and all(thickness_checks)
    passed = bool(checks) and all(checks.values())
    return {
        "schema": "g10_independent_cad_roundtrip_audit_v1",
        "audited_utc": now(),
        "case_id": case["case_id"],
        "status": "PASS" if passed else "FAIL",
        "source_step": case["source_step"],
        "prediction": job["prediction"],
        "output_step": descriptor(output),
        "executor_audit": descriptor(legacy_audit_path),
        "independent_roundtrip_volume_mm3": independent_volume,
        "independent_roundtrip_solid_count": solids,
        "checks": checks,
        "truth_used": False,
    }


def run_cad(run_id: str, case_id: str) -> dict[str, Any]:
    case, job, case_root = _verify_job(run_id, case_id, "cad")
    attempt = Path(job["attempt_directory"])
    launch = case_root / "launch_state.json"
    write_json(launch, {
        "schema": "g10_m6_launch_state_v1", "at_utc": now(),
        "case_id": case_id, "state": "started_receipt_not_yet_written",
        "attempt_directory": attempt.as_posix(), "truth_used": False,
    }, overwrite=False)
    started = time.perf_counter()
    try:
        with resource_lock("occ", f"g10-m6-cad-{run_id}-{case_id}"):
            adapter = _load_adapter(case)
            with _patched_cad_adapter(adapter) as cad_executor:
                output = cad_executor.execute(
                    case["part"], case["plan"], attempt,
                    execution_mode="g10_frozen_e3_user_confirmed_reference_plan",
                )
            legacy_audit = attempt / "edit_audit.json"
            independent = _independent_cad_audit(case, job, output, legacy_audit)
            write_json(case_root / "independent_cad_audit.json", independent,
                       overwrite=False)
            final = case_root / "final.step"
            shutil.copy2(output, final)
        passed = independent["status"] == "PASS"
        receipt = {
            "schema": "g10_m6_execution_receipt_v1",
            "completed_utc": now(), "run_id": run_id, "case_id": case_id,
            "domain": "cad", "status": "PASS" if passed else "FAIL",
            "actually_executed": True,
            "manifest": job["manifest"], "job_spec": descriptor(case_root / "job_spec.json"),
            "source_step": case["source_step"], "prediction": case["prediction"],
            "semantic_plan": job["semantic_plan"], "frozen_binding": case["frozen_binding"],
            "final_step": descriptor(final),
            "executor_audit": descriptor(legacy_audit),
            "independent_audit": descriptor(case_root / "independent_cad_audit.json"),
            "elapsed_seconds": time.perf_counter() - started,
            "attempt_count": 1, "truth_used": False,
        }
    except BaseException as error:
        receipt = {
            "schema": "g10_m6_execution_receipt_v1",
            "completed_utc": now(), "run_id": run_id, "case_id": case_id,
            "domain": "cad", "status": "FAIL", "actually_executed": True,
            "manifest": job["manifest"], "job_spec": descriptor(case_root / "job_spec.json"),
            "source_step": case["source_step"], "prediction": case["prediction"],
            "semantic_plan": job["semantic_plan"], "frozen_binding": case["frozen_binding"],
            "error_type": type(error).__name__, "error": str(error),
            "elapsed_seconds": time.perf_counter() - started,
            "attempt_count": 1, "truth_used": False,
        }
        write_json(case_root / "execution_receipt.json", receipt, overwrite=False)
        raise
    write_json(case_root / "execution_receipt.json", receipt, overwrite=False)
    return receipt


def _stop_owned_process(process: subprocess.Popen) -> None:
    if process.poll() is None:
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True, check=False,
        )
    process.wait(timeout=30)


def _run_process(command: list[str], cwd: Path, environment: dict[str, str],
                 log_path: Path, timeout: int) -> tuple[int, bool, float]:
    started = time.perf_counter()
    timed_out = False
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, cwd=cwd, env=environment, stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            while process.poll() is None:
                if time.perf_counter() - started > timeout:
                    timed_out = True
                    _stop_owned_process(process)
                    break
                time.sleep(2)
        except BaseException:
            _stop_owned_process(process)
            raise
    return process.returncode, timed_out, time.perf_counter() - started


def _abaqus_release() -> dict[str, Any]:
    if not ABAQUS.is_file():
        raise BindingError("abaqus_launcher_missing", str(ABAQUS))
    run = subprocess.run(
        [str(ABAQUS), "information=release"], capture_output=True,
        text=True, timeout=60, check=False,
    )
    result = {
        "launcher": descriptor(ABAQUS),
        "return_code": run.returncode,
        "required_release": ABAQUS_RELEASE,
        "release_matched": ABAQUS_RELEASE in run.stdout,
        "stdout_sha256": hashlib.sha256(run.stdout.encode("utf-8")).hexdigest(),
        "stderr_sha256": hashlib.sha256(run.stderr.encode("utf-8")).hexdigest(),
    }
    if run.returncode or not result["release_matched"]:
        raise BindingError("abaqus_release_changed")
    return result


def run_cae(run_id: str, case_id: str, *, timeout: int = 7200,
            memory_mb: int = 20480, prepare_only: bool = False,
            skip_mesh: bool = False, solver: str = "AUTO") -> dict[str, Any]:
    case, job, case_root = _verify_job(run_id, case_id, "cae")
    if solver not in {"AUTO", "DIRECT", "ITERATIVE"}:
        raise ValueError("solver must be AUTO, DIRECT, or ITERATIVE")
    if timeout < 60 or memory_mb < 1024:
        raise ValueError("unsafe timeout or Abaqus memory allowance")
    attempt = Path(job["attempt_directory"])
    launch = case_root / "launch_state.json"
    write_json(launch, {
        "schema": "g10_m6_launch_state_v1", "at_utc": now(),
        "case_id": case_id, "state": "started_receipt_not_yet_written",
        "attempt_directory": attempt.as_posix(), "truth_used": False,
    }, overwrite=False)
    script = checked(job["program"])
    started = time.perf_counter()
    abaqus_release = None
    receipt: dict[str, Any]
    try:
        abaqus_release = _abaqus_release()
        env = dict(
            os.environ,
            AEI_MATRIX_SOLVER=solver,
            AEI_PREPARE_ONLY="1" if prepare_only else "0",
            AEI_MEMORY_MB=str(memory_mb),
            AEI_SKIP_MESH="1" if skip_mesh else "0",
        )
        command = [
            "cmd", "/d", "/s", "/c",
            f'call "{ABAQUS}" cae noGUI="{script}"',
        ]
        with resource_lock("abaqus", f"g10-m6-cae-{run_id}-{case_id}"):
            code, timed_out, elapsed = _run_process(
                command, attempt, env, case_root / "launch.log", timeout,
            )
            audit_path = attempt / "execution_audit.json"
            audit = read_json(audit_path) if audit_path.exists() else {}
            independent_path = attempt / "independent_odb_audit.json"
            reaudit_code = None
            if not prepare_only and not skip_mesh and code == 0 \
                    and audit.get("status") == "PASS":
                reaudit_command = [
                    "cmd", "/d", "/s", "/c",
                    f'call "{ABAQUS}" python "{case_root / "odb_reaudit.py"}"',
                ]
                reaudit_code, reaudit_timeout, reaudit_elapsed = _run_process(
                    reaudit_command, attempt, env,
                    case_root / "odb_reaudit.log", min(timeout, 1800),
                )
                timed_out |= reaudit_timeout
                elapsed += reaudit_elapsed
            independent = read_json(independent_path) if independent_path.exists() else {}
        actually_executed = not prepare_only and not skip_mesh
        if actually_executed:
            passed = code == 0 and not timed_out and audit.get("status") == "PASS" \
                and reaudit_code == 0 and independent.get("status") == "PASS"
            status = "PASS" if passed else "FAIL"
        else:
            expected = "MODEL_READY" if skip_mesh else "PREPARED"
            status = expected if code == 0 and audit.get("status") == expected else "FAIL"
        artifacts = {}
        for name in ("final.cae", "final.odb", "final.inp", "execution_audit.json",
                     "independent_odb_audit.json"):
            path = attempt / name
            if path.exists():
                artifacts[name] = descriptor(path)
        receipt = {
            "schema": "g10_m6_execution_receipt_v1",
            "completed_utc": now(), "run_id": run_id, "case_id": case_id,
            "domain": "cae", "status": status,
            "actually_executed": actually_executed,
            "prepare_only": prepare_only, "skip_mesh": skip_mesh,
            "manifest": job["manifest"], "job_spec": descriptor(case_root / "job_spec.json"),
            "source_step": case["source_step"], "prediction": case["prediction"],
            "semantic_plan": job["semantic_plan"], "program": job["program"],
            "frozen_binding": case["frozen_binding"],
            "return_code": code, "independent_reaudit_return_code": reaudit_code,
            "timed_out": timed_out, "solver": solver, "memory_mb": memory_mb,
            "abaqus_release": abaqus_release,
            "artifacts": artifacts, "elapsed_seconds": elapsed,
            "attempt_count": 1, "truth_used": False,
        }
    except BaseException as error:
        receipt = {
            "schema": "g10_m6_execution_receipt_v1",
            "completed_utc": now(), "run_id": run_id, "case_id": case_id,
            "domain": "cae", "status": "FAIL",
            "actually_executed": not prepare_only and not skip_mesh,
            "prepare_only": prepare_only, "skip_mesh": skip_mesh,
            "manifest": job["manifest"], "job_spec": descriptor(case_root / "job_spec.json"),
            "source_step": case["source_step"], "prediction": case["prediction"],
            "semantic_plan": job["semantic_plan"], "program": job["program"],
            "frozen_binding": case["frozen_binding"],
            "abaqus_release": abaqus_release,
            "error_type": type(error).__name__, "error": str(error),
            "elapsed_seconds": time.perf_counter() - started,
            "attempt_count": 1, "truth_used": False,
        }
        write_json(case_root / "execution_receipt.json", receipt, overwrite=False)
        raise
    write_json(case_root / "execution_receipt.json", receipt, overwrite=False)
    return receipt


def audit_run(run_id: str) -> dict[str, Any]:
    manifest = verify_manifest()
    run_root = M6_ROOT / "runs" / _safe_run_id(run_id)
    prepared = read_json(run_root / "PREPARED.json")
    rows = []
    for row in prepared["cases"]:
        case_id = row["case_id"]
        case = case_from_manifest(manifest, case_id)
        case_root = run_root / case_id
        receipt_path = case_root / "execution_receipt.json"
        if not receipt_path.exists():
            rows.append({"case_id": case_id, "domain": case["domain"],
                         "status": "NOT_EXECUTED", "receipt": None})
            continue
        receipt = read_json(receipt_path)
        for key in ("manifest", "job_spec", "source_step", "prediction", "semantic_plan"):
            checked(receipt[key])
        if receipt.get("truth_used") is not False \
                or receipt["manifest"] != descriptor(MANIFEST_PATH) \
                or receipt["frozen_binding"] != case["frozen_binding"]:
            raise BindingError("execution_receipt_binding_changed", case_id)
        for value in receipt.get("artifacts", {}).values():
            checked(value)
        if receipt.get("final_step"):
            checked(receipt["final_step"])
        if receipt.get("executor_audit"):
            checked(receipt["executor_audit"])
        if receipt.get("independent_audit"):
            checked(receipt["independent_audit"])
        rows.append({
            "case_id": case_id, "domain": case["domain"],
            "status": receipt["status"], "actually_executed": receipt["actually_executed"],
            "receipt": descriptor(receipt_path),
        })
    completed = [row for row in rows if row["status"] == "PASS"
                 and row.get("actually_executed") is True]
    result = {
        "schema": "g10_m6_run_audit_v1", "audited_utc": now(),
        "run_id": run_id, "prepared_count": len(rows),
        "actual_pass_count": len(completed),
        "cad_actual_pass_count": sum(row["domain"] == "cad" for row in completed),
        "cae_actual_pass_count": sum(row["domain"] == "cae" for row in completed),
        "all_13_execution_pass": len(completed) == 13,
        "execution_experiment_complete": all(
            row["status"] != "NOT_EXECUTED" for row in rows
        ),
        "rows": rows, "truth_used": False,
    }
    output = run_root / "RUN_AUDIT.json"
    if output.exists():
        raise FileExistsError(f"Preserve existing run audit: {output}")
    write_json(output, result, overwrite=False)
    return result


def infer_arbitrary_step(step: Path, inference_id: str, device: str = "cpu") -> dict[str, Any]:
    """Run the frozen G09 E3 arbitrary-STEP service into a G10-owned output."""
    inference_id = _safe_run_id(inference_id)
    step = step.resolve()
    if not step.is_file() or step.suffix.lower() not in {".stp", ".step"}:
        raise ValueError("An existing STEP file is required")
    output_root = M6_ROOT / "arbitrary_inference" / inference_id
    output_root.mkdir(parents=True, exist_ok=False)
    try:
        with resource_lock("occ", f"g10-m6-infer-{inference_id}"):
            g09 = ROOT / "tools/aei_paper_goals_v1/g09"
            g07 = ROOT / "tools/aei_paper_goals_v1/g07"
            original = list(sys.path)
            try:
                for path in (str(g09), str(g07)):
                    if path not in sys.path:
                        sys.path.insert(0, path)
                from infer_step_e3 import E3StepService
                service = E3StepService(device=device)
                result = service.infer(step)
            finally:
                sys.path[:] = original
        validate_prediction(result, descriptor(step))
        output = output_root / "prediction.json"
        write_json(output, result, overwrite=False)
        receipt = {
            "schema": "g10_m6_arbitrary_step_inference_receipt_v1",
            "completed_utc": now(), "inference_id": inference_id,
            "input_step": descriptor(step), "prediction": descriptor(output),
            "system": E3_SYSTEM, "device": device,
            "object_count": len(result["feature_objects"]),
            "truth_used": False,
        }
        write_json(output_root / "inference_receipt.json", receipt, overwrite=False)
        return receipt
    except BaseException as error:
        write_json(output_root / "INFERENCE_FAILURE.json", {
            "schema": "g10_m6_arbitrary_step_inference_failure_v1",
            "at_utc": now(), "inference_id": inference_id,
            "input_step": descriptor(step), "error_type": type(error).__name__,
            "error": str(error), "truth_used": False,
        }, overwrite=False)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("manifest")
    sub.add_parser("verify")
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--run-id", required=True)
    prepare.add_argument("--cases", nargs="*")
    cad = sub.add_parser("run-cad")
    cad.add_argument("--run-id", required=True)
    cad.add_argument("--case", required=True)
    cae = sub.add_parser("run-cae")
    cae.add_argument("--run-id", required=True)
    cae.add_argument("--case", required=True)
    cae.add_argument("--timeout", type=int, default=7200)
    cae.add_argument("--memory-mb", type=int, default=20480)
    cae.add_argument("--solver", choices=("AUTO", "DIRECT", "ITERATIVE"), default="AUTO")
    cae.add_argument("--prepare-only", action="store_true")
    cae.add_argument("--skip-mesh", action="store_true")
    audit = sub.add_parser("audit")
    audit.add_argument("--run-id", required=True)
    inference = sub.add_parser("infer")
    inference.add_argument("step", type=Path)
    inference.add_argument("--inference-id", required=True)
    inference.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.command == "manifest":
        result = build_manifest()
        summary = {"manifest": descriptor(MANIFEST_PATH), "denominator": result["denominator"],
                   "causality": descriptor(CAUSALITY_PATH), "truth_used": False}
    elif args.command == "verify":
        result = verify_manifest()
        summary = {"verified": descriptor(MANIFEST_PATH), "denominator": result["denominator"],
                   "truth_used": False}
    elif args.command == "prepare":
        summary = prepare_run(args.run_id, args.cases)
    elif args.command == "run-cad":
        summary = run_cad(args.run_id, args.case)
    elif args.command == "run-cae":
        summary = run_cae(
            args.run_id, args.case, timeout=args.timeout,
            memory_mb=args.memory_mb, prepare_only=args.prepare_only,
            skip_mesh=args.skip_mesh, solver=args.solver,
        )
    elif args.command == "audit":
        summary = audit_run(args.run_id)
    else:
        summary = infer_arbitrary_step(args.step, args.inference_id, args.device)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
