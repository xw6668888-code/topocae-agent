from __future__ import annotations

"""Generate the frozen, collision-audited Synthetic-Holdout-v3."""

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import random
import shutil
from typing import Any

from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.gp import gp_Pnt

import generate_p1_c_clean_context_7200 as generation
from p1_c import all_features
from p1_c import formal_generator_v7
from p1_c import formal_plan_v4
from p1_c.all_features import CleanFeatureResult, MACRO_CLASSES
from p1_c.formal_plan import PlannedSample
from p1_c.oriented_cylinder_polarity_v1 import augment_graph_with_oriented_cylinder_polarity
from p1_c.rib_web_v3 import RIB_HOST_FAMILIES, WEB_HOST_FAMILIES
from utils.step_solid_validation import read_step_shape


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "p1_c" / "holdout_v3_generation_plan_v1.json"
OUTPUT = ROOT / "output_generation_p1_c_synthetic_holdout_v3"
SELECTION = ROOT / "reports" / "p1_c_freeze_v1" / "preholdout_selection.json"
SIDECAR = ROOT / "reports" / "p1_c_freeze_v1" / "preholdout_selection.sha256.json"
TRAIN_MANIFEST = ROOT / "output_generation_p1_c_clean_context_7200_strict_concept_r2" / "manifests" / "sample_manifest.jsonl"
HOLDOUT_V2_FINGERPRINT = ROOT / "output_generation_p1_b_synthetic_holdout_v2" / "graph_step_manifest.json"
PLAN_PATH = OUTPUT / "manifests" / "generation_plan.jsonl"
PLAN_FREEZE = OUTPUT / "manifests" / "generation_plan_freeze.json"
GENERATOR_REVISION = "p1_c_synthetic_holdout_v3_generator_1"
PLAN_REVISION = "p1_c_synthetic_holdout_v3_plan_1"


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest().lower()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def verify_selection() -> str:
    sidecar = read_json(SIDECAR)
    actual = digest(SELECTION)
    if actual != sidecar.get("sha256") or not sidecar.get("verified_after_write"):
        raise RuntimeError("preholdout selection hash mismatch")
    return actual


def holdout_host_template(plan: PlannedSample) -> str:
    if plan.host_family_kind == "holdout_general_template_v1":
        return plan.host_family
    return f"holdout_supported_scaled::{plan.host_family}"


def build_plan(config: dict[str, Any]) -> list[PlannedSample]:
    contexts = config["context_quota_per_class"]
    general_hosts = tuple(config["general_holdout_host_templates"])
    axes = tuple(tuple(float(value) for value in axis) for axis in config["rotation_axes"])
    angles = tuple(float(value) for value in config["rotation_angles_radians"])
    general_scales = tuple(float(value) for value in config["general_uniform_scales"])
    supported_scales = tuple(float(value) for value in config["supported_wall_uniform_scales"])
    records: list[PlannedSample] = []
    for primary_index, primary in enumerate(MACRO_CLASSES):
        stream = formal_plan_v4._secondary_stream(primary_index)
        cursor = 0
        class_index = 0
        for context in ("target_only", "pair", "triple"):
            for context_index in range(int(contexts[context])):
                if context == "target_only":
                    secondary: tuple[str, ...] = ()
                elif context == "pair":
                    secondary = (stream[cursor],)
                    cursor += 1
                else:
                    secondary = (stream[cursor], stream[cursor + 1])
                    cursor += 2
                if len(set(secondary)) != len(secondary):
                    raise RuntimeError("duplicate secondary in Holdout-v3 triple")
                macros = (primary, *secondary)
                walls = [macro for macro in macros if macro in {"rib", "web"}]
                if len(walls) > 1:
                    raise RuntimeError("forbidden rib/web composition")
                if walls:
                    families = RIB_HOST_FAMILIES if walls[0] == "rib" else WEB_HOST_FAMILIES
                    host = families[(primary_index + class_index) % len(families)]
                    host_kind = "holdout_supported_template_v1"
                    scale = supported_scales[(primary_index * 3 + class_index) % len(supported_scales)]
                else:
                    host = general_hosts[(primary_index + class_index) % len(general_hosts)]
                    host_kind = "holdout_general_template_v1"
                    scale = general_scales[(primary_index * 3 + class_index) % len(general_scales)]
                seed = int(config["seed_namespace"]) + primary_index * 10000 + class_index
                sample_id = f"p1c_holdout_v3_{primary_index + 1:02d}_{primary}_{class_index:03d}_{context}"
                template_id = host if host_kind == "holdout_general_template_v1" else f"holdout_supported_scaled::{host}"
                records.append(PlannedSample(
                    sample_id=sample_id,
                    primary_target=primary,
                    primary_index=primary_index,
                    class_sample_index=class_index,
                    context=context,
                    context_index=context_index,
                    secondary_features=secondary,
                    split="holdout_v3",
                    seed=seed,
                    host_family=host,
                    host_family_kind=host_kind,
                    rotation_axis=axes[(primary_index + class_index) % len(axes)],
                    rotation_angle_radians=angles[(primary_index * 2 + class_index) % len(angles)],
                    uniform_scale=scale,
                    base_geometry_family_id=f"p1c-holdout-v3-base::{primary_index:02d}::{class_index:03d}::{template_id}",
                    parametric_family_id=f"p1c-holdout-v3-param::{primary}::{context}::{template_id}::{class_index:03d}",
                ))
                class_index += 1
        if cursor != 50:
            raise RuntimeError(f"Holdout-v3 secondary cursor mismatch: {primary}/{cursor}")
    audit_plan(records, config)
    return records


def audit_plan(records: list[PlannedSample], config: dict[str, Any]) -> dict[str, Any]:
    primary = Counter(row.primary_target for row in records)
    contexts = Counter((row.primary_target, row.context) for row in records)
    ids = [row.sample_id for row in records]
    seeds = [row.seed for row in records]
    expected_contexts = config["context_quota_per_class"]
    checks = {
        "sample_count_1440": len(records) == 1440,
        "all_18_classes": set(primary) == set(MACRO_CLASSES),
        "80_per_class": all(primary[name] == 80 for name in MACRO_CLASSES),
        "context_40_30_10": all(
            contexts[(name, context)] == int(quota)
            for name in MACRO_CLASSES for context, quota in expected_contexts.items()
        ),
        "unique_ids": len(ids) == len(set(ids)),
        "unique_seeds": len(seeds) == len(set(seeds)),
        "new_seed_namespace": min(seeds) >= int(config["seed_namespace"]),
        "new_general_templates_present": all(
            any(row.host_family == template for row in records)
            for template in config["general_holdout_host_templates"]
        ),
        "supported_wall_new_scale_range": all(
            row.uniform_scale > 1.6
            for row in records if row.host_family_kind == "holdout_supported_template_v1"
        ),
    }
    payload = {
        "schema_version": "topocae_p1_c_holdout_v3_plan_audit_1",
        "passed": all(checks.values()),
        "checks": checks,
        "sample_count": len(records),
        "primary_counts": dict(primary),
        "context_counts": {f"{macro}::{context}": contexts[(macro, context)] for macro in MACRO_CLASSES for context in expected_contexts},
        "host_template_counts": dict(Counter(holdout_host_template(row) for row in records)),
    }
    if not payload["passed"]:
        raise RuntimeError(f"Holdout-v3 plan audit failed: {payload}")
    return payload


def prepare() -> dict[str, Any]:
    if OUTPUT.exists():
        raise FileExistsError("Holdout-v3 output already exists; use generate only for a prepared/resumable output")
    config = read_json(CONFIG_PATH)
    selection_sha = verify_selection()
    if SELECTION.stat().st_mtime_ns >= CONFIG_PATH.stat().st_mtime_ns:
        raise RuntimeError("preholdout selection must predate Holdout-v3 config")
    for name in ("steps", "graphs", "state/records", "manifests", "reports", "human_review_36"):
        (OUTPUT / name).mkdir(parents=True, exist_ok=True)
    plan = build_plan(config)
    with PLAN_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        for row in plan:
            payload = row.to_dict()
            payload["holdout_host_template_id"] = holdout_host_template(row)
            handle.write(json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\n")
    audit = audit_plan(plan, config)
    audit_path = OUTPUT / "manifests" / "generation_plan_audit.json"
    write_json(audit_path, audit)
    freeze = {
        "schema_version": "topocae_p1_c_holdout_v3_plan_freeze_1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan_revision": PLAN_REVISION,
        "generator_revision": GENERATOR_REVISION,
        "sample_count": len(plan),
        "generation_plan_sha256": digest(PLAN_PATH),
        "generation_plan_audit_sha256": digest(audit_path),
        "generation_config_sha256": digest(CONFIG_PATH),
        "preholdout_selection_sha256": selection_sha,
        "holdout_results_seen": False,
        "old_holdout_v2_fingerprint_manifest_only": str(HOLDOUT_V2_FINGERPRINT),
        "old_holdout_v2_fingerprint_sha256": digest(HOLDOUT_V2_FINGERPRINT),
        "contracts": config["contracts"],
    }
    write_json(PLAN_FREEZE, freeze)
    return {"passed": True, "output": str(OUTPUT), "plan_freeze": freeze}


_original_make_host = all_features._make_host


def _holdout_make_host(host_family: str, rng: random.Random):
    config = read_json(CONFIG_PATH)
    bounds = config["general_holdout_host_templates"].get(host_family)
    if bounds is None:
        return _original_make_host(host_family, rng)
    length = rng.uniform(*bounds["L"])
    width = rng.uniform(*bounds["W"])
    height = rng.uniform(*bounds["H"])
    shape = BRepPrimAPI_MakeBox(gp_Pnt(-length / 2.0, -width / 2.0, 0.0), length, width, height).Shape()
    return shape, {
        "family": host_family,
        "construction_method": "holdout_v3_single_box_template",
        "length": length, "width": width, "height": height, "top_z": height,
        "origin": [-length / 2.0, -width / 2.0, 0.0],
        "holdout_v3_new_template": True,
        "training_range_reused": False,
    }


def _build_holdout_result(plan: PlannedSample) -> CleanFeatureResult:
    all_features._make_host = _holdout_make_host
    result = formal_generator_v7.build_planned_result(plan)
    parameters = deepcopy(result.parameters)
    parameters["holdout_v3"] = {
        "plan_revision": PLAN_REVISION,
        "generator_revision": GENERATOR_REVISION,
        "host_template_id": holdout_host_template(plan),
        "new_seed_namespace": True,
        "new_parameter_range": True,
        "used_for_development": False,
    }
    return CleanFeatureResult(result.shape, result.face_tags, parameters, result.cleanup)


def load_plan() -> list[PlannedSample]:
    result = []
    with PLAN_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            payload.pop("holdout_host_template_id", None)
            payload["secondary_features"] = tuple(payload["secondary_features"])
            payload["rotation_axis"] = tuple(payload["rotation_axis"])
            result.append(PlannedSample(**payload))
    return result


def load_collision_sets() -> dict[str, set[str]]:
    sets = {key: set() for key in ("base_id", "base_signature", "step", "exact", "near", "parametric")}
    with TRAIN_MANIFEST.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            sets["base_id"].add(str(row.get("base_geometry_family_id", "")))
            sets["base_signature"].add(str(row.get("base_geometry_signature", "")))
            sets["step"].add(str(row.get("step_sha256", "")).lower())
            sets["exact"].add(str(row.get("exact_graph_signature", "")))
            sets["near"].add(str(row.get("near_graph_signature", "")))
            sets["parametric"].add(str(row.get("parametric_family_id", "")))
    old = read_json(HOLDOUT_V2_FINGERPRINT)
    for row in old["records"]:
        sets["base_id"].add(str(row.get("base_geometry_family_id", "")))
        sets["step"].add(str(row.get("step_sha256", "")).lower())
        sets["exact"].add(str(row.get("exact_graph_signature", "")))
        sets["near"].add(str(row.get("near_graph_signature", "")))
    return sets


def collision_kinds(record: dict[str, Any], sets: dict[str, set[str]]) -> list[str]:
    values = {
        "base_id": str(record["base_geometry_family_id"]),
        "base_signature": str(record["base_geometry_signature"]),
        "step": str(record["step_sha256"]).lower(),
        "exact": str(record["exact_graph_signature"]),
        "near": str(record["near_graph_signature"]),
        "parametric": str(record["parametric_family_id"]),
    }
    return [key for key, value in values.items() if value and value in sets[key]]


def add_collision_record(record: dict[str, Any], sets: dict[str, set[str]]) -> None:
    sets["base_id"].add(str(record["base_geometry_family_id"]))
    sets["base_signature"].add(str(record["base_geometry_signature"]))
    sets["step"].add(str(record["step_sha256"]).lower())
    sets["exact"].add(str(record["exact_graph_signature"]))
    sets["near"].add(str(record["near_graph_signature"]))
    sets["parametric"].add(str(record["parametric_family_id"]))


def generate() -> dict[str, Any]:
    if not PLAN_FREEZE.is_file():
        raise RuntimeError("prepare must freeze the Holdout-v3 plan first")
    freeze = read_json(PLAN_FREEZE)
    if digest(PLAN_PATH) != freeze["generation_plan_sha256"]:
        raise RuntimeError("Holdout-v3 plan drift")
    if verify_selection() != freeze["preholdout_selection_sha256"]:
        raise RuntimeError("selection drift")
    if (OUTPUT / "holdout_generation_manifest.json").exists():
        raise RuntimeError("Holdout-v3 generation already complete and immutable")
    config = read_json(CONFIG_PATH)
    generation.build_planned_result = _build_holdout_result
    generation.transformed_shape_and_audit = formal_generator_v7.transformed_shape_and_audit
    generation.GENERATOR_REVISION = GENERATOR_REVISION
    generation.PLAN_REVISION = PLAN_REVISION
    generation.MAX_GENERATION_ATTEMPTS = int(config["maximum_generation_attempts"])
    generation.RETRY_SEED_STRIDE = int(config["retry_seed_stride"])
    sets = load_collision_sets()
    plan = load_plan()
    resumed = generated = 0
    for position, planned in enumerate(plan, 1):
        state_path = OUTPUT / "state" / "records" / f"{planned.sample_id}.json"
        if state_path.exists():
            record = read_json(state_path)
            generation._verify_record_files(record, OUTPUT)
            add_collision_record(record, sets)
            resumed += 1
            continue
        accepted = None
        failures = []
        for collision_attempt in range(int(config["collision_retry_count"])):
            candidate = replace(planned, seed=planned.seed + collision_attempt * int(config["collision_retry_stride"]))
            record = generation._generate_one(candidate, OUTPUT)
            kinds = collision_kinds(record, sets)
            if kinds:
                failures.append({"collision_attempt": collision_attempt, "kinds": kinds})
                Path(record["step_path"]).unlink(missing_ok=True)
                Path(record["graph_path"]).unlink(missing_ok=True)
                continue
            graph_path = Path(record["graph_path"])
            graph = read_json(graph_path)
            shape = read_step_shape(Path(record["step_path"]))
            graph = augment_graph_with_oriented_cylinder_polarity(shape, graph)
            graph["dataset"] = "Synthetic-Holdout-v3"
            graph["holdout_v3_provenance"] = {
                "preholdout_selection_sha256": freeze["preholdout_selection_sha256"],
                "plan_freeze_sha256": digest(PLAN_FREEZE),
                "holdout_host_template_id": holdout_host_template(planned),
                "collision_attempt": collision_attempt,
                "old_holdout_v2_fingerprint_only": True,
                "used_for_development": False,
            }
            graph_path.write_text(json.dumps(graph, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
            record.update({
                "holdout_host_template_id": holdout_host_template(planned),
                "frozen_planned_seed": planned.seed,
                "collision_attempt": collision_attempt,
                "collision_failures_before_acceptance": failures,
                "graph_sha256": digest(graph_path),
                "graph_polarity_revision": "p1_c_oriented_cylinder_polarity_v1",
            })
            accepted = record
            break
        if accepted is None:
            raise RuntimeError(f"Holdout-v3 collision retries exhausted: {planned.sample_id}/{failures}")
        write_json(state_path, accepted)
        add_collision_record(accepted, sets)
        generated += 1
        if position % 25 == 0 or position == len(plan):
            progress = {
                "schema_version": "topocae_p1_c_holdout_v3_generation_progress_1",
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "completed_count": position,
                "generated_this_run": generated,
                "resumed_count": resumed,
                "last_sample_id": planned.sample_id,
            }
            write_json(OUTPUT / "state" / "progress.json", progress)
            print(f"P1-C Holdout-v3 generation {position}/{len(plan)} new={generated} resumed={resumed}", flush=True)
    return {"passed": True, "sample_count": len(plan), "generated_this_run": generated, "resumed_count": resumed}


def finalize() -> dict[str, Any]:
    if (OUTPUT / "holdout_generation_manifest.json").exists():
        raise RuntimeError("Holdout-v3 generation already finalized")
    plan = load_plan()
    records = []
    for planned in plan:
        path = OUTPUT / "state" / "records" / f"{planned.sample_id}.json"
        if not path.is_file():
            raise RuntimeError(f"missing Holdout-v3 state: {planned.sample_id}")
        record = read_json(path)
        generation._verify_record_files(record, OUTPUT)
        records.append(record)
    primary = Counter(row["primary_target"] for row in records)
    contexts = Counter((row["primary_target"], row["context"]) for row in records)
    collision_fields = {
        "base_id": [row["base_geometry_family_id"] for row in records],
        "base_signature": [row["base_geometry_signature"] for row in records],
        "step": [row["step_sha256"].lower() for row in records],
        "exact": [row["exact_graph_signature"] for row in records],
        "near": [row["near_graph_signature"] for row in records],
        "parametric": [row["parametric_family_id"] for row in records],
    }
    reference = load_collision_sets()
    external_collision_counts = {
        key: sum(value in reference[key] for value in values)
        for key, values in collision_fields.items()
    }
    internal_collision_counts = {
        key: len(values) - len(set(values)) for key, values in collision_fields.items()
    }
    cleanup_pass = sum(row["cleanup_audit"]["passed"] for row in records)
    label_pass = sum(row["label_audit"]["passed"] for row in records)
    graph_pass = sum(row["graph_input_audit"]["passed"] for row in records)
    step_pass = sum(row["step_roundtrip_audit"]["passed"] for row in records)
    quality = {
        "schema_version": "topocae_p1_c_holdout_v3_quality_report_1",
        "sample_count": len(records),
        "primary_counts": dict(primary),
        "context_counts": {f"{macro}::{context}": contexts[(macro, context)] for macro in MACRO_CLASSES for context in ("target_only", "pair", "triple")},
        "cleanup_pass_count": cleanup_pass,
        "label_pass_count": label_pass,
        "graph_pass_count": graph_pass,
        "step_roundtrip_pass_count": step_pass,
        "all_graphs_have_polarity": all(row["graph_polarity_revision"] == "p1_c_oriented_cylinder_polarity_v1" for row in records),
        "automatic_pass": (
            len(records) == 1440 and all(primary[name] == 80 for name in MACRO_CLASSES)
            and all(contexts[(name, context)] == quota for name in MACRO_CLASSES for context, quota in {"target_only": 40, "pair": 30, "triple": 10}.items())
            and cleanup_pass == label_pass == graph_pass == step_pass == 1440
        ),
    }
    leakage = {
        "schema_version": "topocae_p1_c_holdout_v3_leakage_audit_1",
        "p1_c_train_val_development_and_holdout_v2_fingerprint_reference_only": True,
        "external_collision_counts": external_collision_counts,
        "internal_collision_counts": internal_collision_counts,
        "all_zero": not any(external_collision_counts.values()) and not any(internal_collision_counts.values()),
        "holdout_v2_fingerprint_manifest_path": str(HOLDOUT_V2_FINGERPRINT),
        "holdout_v2_fingerprint_manifest_sha256": digest(HOLDOUT_V2_FINGERPRINT),
        "holdout_v2_predictions_or_errors_read": False,
    }
    if not quality["automatic_pass"] or not leakage["all_zero"]:
        raise RuntimeError(f"Holdout-v3 final audit failed: quality={quality}, leakage={leakage}")
    manifest_path = OUTPUT / "manifests" / "sample_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in sorted(records, key=lambda item: item["sample_id"]):
            handle.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")
    fingerprint = {
        "schema_version": "topocae_p1_c_holdout_v3_fingerprint_manifest_1",
        "records": [{
            "sample_id": row["sample_id"], "step_sha256": row["step_sha256"],
            "base_geometry_family_id": row["base_geometry_family_id"],
            "base_geometry_signature": row["base_geometry_signature"],
            "exact_graph_signature": row["exact_graph_signature"], "near_graph_signature": row["near_graph_signature"],
        } for row in sorted(records, key=lambda item: item["sample_id"])],
    }
    write_json(OUTPUT / "manifests" / "fingerprint_manifest.json", fingerprint)
    write_json(OUTPUT / "reports" / "quality_report.json", quality)
    write_json(OUTPUT / "reports" / "leakage_audit.json", leakage)

    review_records = []
    for macro in MACRO_CLASSES:
        single = next(row for row in records if row["primary_target"] == macro and row["context"] == "target_only")
        multi = next(row for row in records if row["primary_target"] == macro and row["context"] == "pair")
        for context_label, row in (("single", single), ("multi", multi)):
            destination = OUTPUT / "human_review_36" / f"{macro}_{context_label}.step"
            shutil.copy2(row["step_path"], destination)
            review_records.append({
                "macro_feature": macro, "review_context": context_label, "sample_id": row["sample_id"],
                "source_step_path": row["step_path"], "review_step_path": str(destination),
                "step_sha256": digest(destination), "human_gate": False,
            })
    write_json(OUTPUT / "human_review_36" / "review_package_manifest.json", {
        "schema_version": "topocae_p1_c_holdout_v3_review_package_1",
        "record_count": len(review_records), "records": review_records,
        "purpose": "reporting archive only; not a return-work Human Gate",
    })
    freeze = read_json(PLAN_FREEZE)
    complete = {
        "schema_version": "topocae_p1_c_holdout_v3_generation_manifest_1",
        "dataset_name": "Synthetic-Holdout-v3",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(records),
        "generation_plan_freeze_sha256": digest(PLAN_FREEZE),
        "preholdout_selection_sha256": freeze["preholdout_selection_sha256"],
        "sample_manifest_sha256": digest(manifest_path),
        "fingerprint_manifest_sha256": digest(OUTPUT / "manifests" / "fingerprint_manifest.json"),
        "quality_report_sha256": digest(OUTPUT / "reports" / "quality_report.json"),
        "leakage_audit_sha256": digest(OUTPUT / "reports" / "leakage_audit.json"),
        "review_package_manifest_sha256": digest(OUTPUT / "human_review_36" / "review_package_manifest.json"),
        "inference_pass_count": 0,
        "contracts": {
            "holdout_used_for_development": False, "old_holdout_results_used": False,
            "holdout_v2_fingerprint_manifest_only": True, "test_part_s3_used": False,
            "second_inference_run_forbidden": True,
        },
    }
    write_json(OUTPUT / "holdout_generation_manifest.json", complete)
    return {"automatic_pass": True, "sample_count": len(records), "leakage": leakage, "quality": quality, "output": str(OUTPUT)}


def status() -> dict[str, Any]:
    count = len(list((OUTPUT / "state" / "records").glob("*.json"))) if OUTPUT.exists() else 0
    return {"prepared": PLAN_FREEZE.is_file(), "completed_records": count, "finalized": (OUTPUT / "holdout_generation_manifest.json").is_file()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "generate", "finalize", "status"))
    args = parser.parse_args()
    result = {"prepare": prepare, "generate": generate, "finalize": finalize, "status": status}[args.command]()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if result.get("passed", result.get("automatic_pass", True)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
