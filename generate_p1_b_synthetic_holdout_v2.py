from __future__ import annotations

"""Generate collision-free Synthetic-Holdout-v2 after selection freeze."""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any

from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.gp import gp_Pnt

import build_p1_b_corrected_dataset_5000 as corrected
from build_p1_b_corrected_dataset_5000_v3 import rebuild_with_host_height_roles
from core.graph_v2_1 import canonical_graph_signature, extract_graph_v2
from features.p1_b_continuous_slot import P1BContinuousSlotGenerator, PROFILE_TO_MACRO, all_slot_combinations
from features.p1_b_rib_web_v3 import build_rib_web_sample, scenario_keys
from generate_paper_v1_synthetic_holdout import acceptable_step_report, generation_resources
from main import CATALOG_GENERATOR_BY_MACRO, build_single_dataset_sample, is_valid_catalog_sample
from utils.step_solid_validation import inspect_step_file, read_step_shape, write_step_with_positive_volume


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "p1_b_holdout_v2_generation.json"
SELECTION = ROOT / "reports" / "p1_b_freeze_v1" / "preholdout_selection.json"
SIDECAR = ROOT / "reports" / "p1_b_freeze_v1" / "preholdout_selection.sha256.json"
TRAIN_MANIFEST = ROOT / "output_generation_p1_b_corrected_balanced_5000_v3" / "manifests" / "sample_manifest.jsonl"
SPECIAL_MACROS = {"slot_u", "slot_rect", "slot_v", "slot_o", "rib", "web"}
SLOT_PROFILE_BY_MACRO = {macro: profile for profile, macro in PROFILE_TO_MACRO.items()}


def main() -> int:
    config = _read_json(CONFIG_PATH)
    selection_sha = _verify_selection()
    output = ROOT / config["output_root"]
    complete_manifest = output / "holdout_generation_manifest.json"
    if complete_manifest.exists():
        raise FileExistsError("Synthetic-Holdout-v2 is complete and immutable")
    started_ns = time.time_ns()
    if SELECTION.stat().st_mtime_ns >= started_ns:
        raise RuntimeError("preholdout selection must strictly predate Holdout generation")
    graph_dir = output / "graphs"
    step_dir = output / "steps"
    raw_dir = output / "raw_generation"
    for directory in (graph_dir, step_dir, raw_dir):
        directory.mkdir(parents=True, exist_ok=True)
    target_classes = list(config["target_macro_features"])
    per_class = int(config["samples_per_class"])
    expected = len(target_classes) * per_class
    if expected != int(config["sample_count"]):
        raise ValueError("sample_count must equal 18 * samples_per_class")
    if int(config["seed_end"]) - int(config["seed_start"]) + 1 != expected:
        raise ValueError("seed range does not match sample count")
    training_base, training_exact, training_near = _training_signatures()
    progress_path = output / "generation_progress.json"
    accepted, failures = _load_progress(progress_path, selection_sha)
    accepted_by_id = {record["sample_id"]: record for record in accepted}
    holdout_base = {record["base_geometry_family_id"] for record in accepted}
    holdout_exact = {record["exact_graph_signature"] for record in accepted}
    holdout_near = {record["near_graph_signature"] for record in accepted}
    resources = generation_resources()
    slot_generator = P1BContinuousSlotGenerator(resources["semantic_subtype_index"])
    global_index = 0
    for macro in target_classes:
        for class_index in range(per_class):
            seed = int(config["seed_start"]) + global_index
            global_index += 1
            sample_id = f"p1b_holdout_v2_{macro}_{class_index:03d}"
            if sample_id in accepted_by_id:
                continue
            step_path = step_dir / f"{sample_id}.step"
            graph_path = graph_dir / f"{sample_id}.json"
            record = None
            for attempt in range(int(config["max_class_attempts_per_sample"])):
                attempt_seed = seed + attempt * 10_000_000
                try:
                    if macro in SPECIAL_MACROS:
                        graph, generator_meta = _special_graph(
                            macro, class_index, attempt_seed, step_path, resources, slot_generator
                        )
                    else:
                        graph, generator_meta = _catalog_graph(
                            macro, attempt_seed, step_path, raw_dir / f"{sample_id}.json", config, resources
                        )
                except (RuntimeError, ValueError, KeyError):
                    failures["generation_or_label_failure"] += 1
                    continue
                if not _target_is_present(graph, macro):
                    failures["target_macro_missing"] += 1
                    continue
                step_report = inspect_step_file(step_path)
                if not acceptable_step_report(step_report):
                    failures["step_quality_failure"] += 1
                    continue
                base_id = f"p1b_holdout_v2::{macro}::{class_index:03d}::{seed}"
                exact = corrected._signature_hash(canonical_graph_signature(graph, ndigits=6))
                near = corrected._signature_hash(canonical_graph_signature(graph, ndigits=4))
                if base_id in training_base or base_id in holdout_base:
                    failures["base_collision"] += 1
                    continue
                if exact in training_exact or exact in holdout_exact:
                    failures["exact_signature_collision"] += 1
                    continue
                if near in training_near or near in holdout_near:
                    failures["near_signature_collision"] += 1
                    continue
                graph.update({
                    "sample_id": sample_id,
                    "dataset": "Synthetic-Holdout-v2",
                    "source_step_path": str(step_path.resolve()),
                    "base_geometry_family_id": base_id,
                    "geometry_signatures": {
                        "exact_graph_signature": exact,
                        "near_graph_signature": near,
                        "exact_ndigits": 6,
                        "near_ndigits": 4,
                    },
                    "p1_b_holdout_provenance": {
                        "sample_seed": seed,
                        "attempt_seed": attempt_seed,
                        "accepted_attempt": attempt + 1,
                        "target_macro_feature": macro,
                        "generator": generator_meta,
                        "preholdout_selection_sha256": selection_sha,
                        "used_for_development": False,
                    },
                })
                _write_json(graph_path, graph)
                record = {
                    "sample_id": sample_id,
                    "target_macro_feature": macro,
                    "class_index": class_index,
                    "sample_seed": seed,
                    "attempt_seed": attempt_seed,
                    "accepted_attempt": attempt + 1,
                    "graph_path": str(graph_path.resolve()),
                    "step_path": str(step_path.resolve()),
                    "graph_sha256": _sha256(graph_path),
                    "step_sha256": _sha256(step_path),
                    "base_geometry_family_id": base_id,
                    "exact_graph_signature": exact,
                    "near_graph_signature": near,
                    "face_count": len(graph.get("nodes", [])),
                    "feature_instance_count": len(graph.get("feature_sets", [])),
                    "step_validation": step_report,
                }
                accepted.append(record)
                accepted_by_id[sample_id] = record
                holdout_base.add(base_id)
                holdout_exact.add(exact)
                holdout_near.add(near)
                break
            if record is None:
                raise RuntimeError(f"failed to generate collision-free {sample_id}")
            if len(accepted) % 20 == 0 or len(accepted) == expected:
                _write_progress(progress_path, selection_sha, accepted, failures)
                print(f"holdout_v2_generation={len(accepted)}/{expected} class={macro} failures={sum(failures.values())}", flush=True)
    accepted.sort(key=lambda item: item["sample_id"])
    class_counts = Counter(item["target_macro_feature"] for item in accepted)
    if len(accepted) != expected or any(class_counts[name] != per_class for name in target_classes):
        raise RuntimeError("Holdout-v2 balance invariant failed")
    audit = {
        "schema_version": "topocae_p1_b_holdout_v2_leakage_audit_1",
        "training_sample_count": sum(1 for _ in TRAIN_MANIFEST.open("r", encoding="utf-8")),
        "holdout_sample_count": len(accepted),
        "base_collision_with_training": sum(item["base_geometry_family_id"] in training_base for item in accepted),
        "exact_collision_with_training": sum(item["exact_graph_signature"] in training_exact for item in accepted),
        "near_collision_with_training": sum(item["near_graph_signature"] in training_near for item in accepted),
        "holdout_internal_base_collision": len(accepted) - len(holdout_base),
        "holdout_internal_exact_collision": len(accepted) - len(holdout_exact),
        "holdout_internal_near_collision": len(accepted) - len(holdout_near),
        "status": "pass",
    }
    quality = {
        "schema_version": "topocae_p1_b_holdout_v2_quality_1",
        "sample_count": len(accepted),
        "valid_single_solid_count": sum(acceptable_step_report(item["step_validation"]) for item in accepted),
        "class_counts": dict(sorted(class_counts.items())),
        "samples_per_class": per_class,
        "generation_failures": dict(sorted(failures.items())),
        "all_18_classes_present": set(class_counts) == set(target_classes),
        "slot_and_rib_web_use_human_gated_generators": True,
        "automatic_pass": True,
    }
    _write_json(output / "graph_step_manifest.json", {"schema_version": "topocae_p1_b_holdout_v2_graph_step_manifest_1", "records": accepted})
    _write_json(output / "leakage_audit.json", audit)
    _write_json(output / "quality_report.json", quality)
    _write_json(complete_manifest, {
        "schema_version": "topocae_p1_b_holdout_v2_generation_manifest_1",
        "phase": "P1-B",
        "dataset_name": "Synthetic-Holdout-v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(accepted),
        "seed_start": config["seed_start"],
        "seed_end": config["seed_end"],
        "generation_config_sha256": _sha256(CONFIG_PATH),
        "preholdout_selection_sha256": selection_sha,
        "preholdout_strictly_predates_generation": SELECTION.stat().st_mtime_ns < started_ns,
        "inference_pass_count": 0,
        "contracts": {
            "old_holdout_reused": False,
            "old_slot_or_rib_web_generator_used": False,
            "holdout_used_for_development": False,
            "test_part_s3_used": False,
        },
    })
    print(json.dumps({"automatic_pass": True, "sample_count": len(accepted), "leakage": audit, "output": str(output)}, ensure_ascii=False), flush=True)
    return 0


def _special_graph(
    macro: str, class_index: int, seed: int, step_path: Path,
    resources: dict[str, Any], slot_generator: P1BContinuousSlotGenerator,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if macro.startswith("slot_"):
        profile = SLOT_PROFILE_BY_MACRO[macro]
        combinations = [item for item in all_slot_combinations() if item[0] == profile]
        _, guide, end = combinations[class_index % len(combinations)]
        rng = random.Random(seed)
        length = rng.uniform(87.0, 151.0)
        width = rng.uniform(69.0, min(119.0, length * 0.86))
        height = rng.uniform(21.0, 41.0)
        stock = BRepPrimAPI_MakeBox(gp_Pnt(-length / 2, -width / 2, 0), length, width, height).Shape()
        raw = slot_generator.build_one(
            stock, length, width, height, profile_shape=profile,
            guide_family=guide, end_condition=end, seed=seed, instance_id=1,
        )
        result = rebuild_with_host_height_roles(raw, stock)
        labels, _ = corrected._slot_labels(result, resources["semantic_subtype_index"], resources["ontology_maps"])
        shape = result.shape
        meta = {"name": "p1_b_continuous_closed_profile_sweep_cut", "profile": profile, "guide": guide, "end_condition": end}
    else:
        keys = [key for key in scenario_keys() if key.startswith(f"{macro}_")]
        scenario = keys[class_index % len(keys)]
        result = build_rib_web_sample(scenario, seed)
        shape = result.shape
        labels = corrected._rib_web_labels(result, resources["semantic_subtype_index"], resources["ontology_maps"])
        meta = {"name": "p1_b_supported_rib_web_v3", "scenario": scenario}
    expected = corrected._face_sequence_signature(shape)
    write_step_with_positive_volume(shape, step_path)
    reread = read_step_shape(step_path)
    graph = extract_graph_v2(reread, face_labels=labels, primitive_surface_index=resources["primitive_surface_index"])
    corrected._validate_face_sequence(expected, graph, relative_area_tolerance=1e-8)
    return graph, meta


def _catalog_graph(
    macro: str, seed: int, step_path: Path, raw_path: Path,
    config: dict[str, Any], resources: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    random.seed(seed)
    bounds = config["stock_size_range"]
    raw = build_single_dataset_sample(
        random.uniform(*bounds["L"]), random.uniform(*bounds["W"]), random.uniform(*bounds["H"]),
        str(step_path), str(raw_path), ontology_maps=resources["ontology_maps"],
        primitive_surface_index=resources["primitive_surface_index"],
        semantic_subtype_index=resources["semantic_subtype_index"],
        idealization_tag_index=resources["idealization_tag_index"],
        primitive_label_map=resources["primitive_label_map"], semantic_label_map=resources["semantic_label_map"],
        idealization_label_map=resources["idealization_label_map"],
        generator_names=[CATALOG_GENERATOR_BY_MACRO[macro]], instance_count_range=(1, 1),
        require_nonempty_sample=True, max_generation_attempts=int(config["max_generation_attempts"]),
        write_step=True, write_graph=True, generation_context=resources["generation_context"],
    )
    if not is_valid_catalog_sample(raw, macro):
        raise ValueError("catalog graph does not contain exactly the requested macro")
    labels = corrected._labels_from_graph(raw, resources["semantic_subtype_index"])
    shape = read_step_shape(step_path)
    graph = extract_graph_v2(shape, face_labels=labels, primitive_surface_index=resources["primitive_surface_index"])
    if len(graph.get("nodes", [])) != len(labels):
        raise ValueError("STEP face sequence does not match generated labels")
    return graph, {"name": CATALOG_GENERATOR_BY_MACRO[macro]}


def _target_is_present(graph: dict[str, Any], macro: str) -> bool:
    return any(str(item.get("macro_feature")) == macro and item.get("node_indices") for item in graph.get("feature_sets", []))


def _training_signatures() -> tuple[set[str], set[str], set[str]]:
    base: set[str] = set(); exact: set[str] = set(); near: set[str] = set()
    with TRAIN_MANIFEST.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            base.add(str(record["base_geometry_family_id"]))
            exact.add(str(record["exact_graph_signature"]))
            near.add(str(record["near_graph_signature"]))
    return base, exact, near


def _verify_selection() -> str:
    if not SELECTION.is_file() or not SIDECAR.is_file():
        raise FileNotFoundError("preholdout selection and sidecar are required")
    sidecar = _read_json(SIDECAR)
    digest = _sha256(SELECTION)
    if digest != sidecar.get("sha256") or not sidecar.get("verified_after_write"):
        raise RuntimeError("preholdout selection hash verification failed")
    return digest


def _load_progress(path: Path, selection_sha: str) -> tuple[list[dict[str, Any]], Counter[str]]:
    if not path.exists():
        return [], Counter()
    payload = _read_json(path)
    if payload.get("preholdout_selection_sha256") != selection_sha:
        raise RuntimeError("partial Holdout belongs to another frozen selection")
    return list(payload.get("accepted_records", [])), Counter(payload.get("failure_counts", {}))


def _write_progress(path: Path, selection_sha: str, accepted: list[dict[str, Any]], failures: Counter[str]) -> None:
    _write_json(path, {
        "schema_version": "topocae_p1_b_holdout_v2_generation_progress_1",
        "preholdout_selection_sha256": selection_sha,
        "accepted_count": len(accepted),
        "failure_counts": dict(failures),
        "accepted_records": accepted,
    })


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


if __name__ == "__main__":
    raise SystemExit(main())
