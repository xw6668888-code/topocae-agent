from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time
from typing import Any

import yaml

from audit_paper_v1_provenance import (
    base_geometry_key,
    rotation_invariant_graph_signature,
)
from main import (
    CATALOG_GENERATOR_BY_MACRO,
    build_single_dataset_sample,
    is_valid_catalog_sample,
)
from p1_a2_protocol import (
    DATASET_CONFIG,
    HOLDOUT_CONFIG,
    PREHOLDOUT_SELECTION,
    PREHOLDOUT_SIDECAR,
    REPO_ROOT,
    load_json,
    resolve_repo_path,
    sha256_file,
    stable_hash,
)
from utils.generation_context import build_generation_context
from utils.ontology import export_layer_label_map, load_layer_index, load_ontology_maps
from utils.step_solid_validation import inspect_step_file


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def verify_preholdout() -> tuple[dict, str]:
    if not PREHOLDOUT_SELECTION.is_file() or not PREHOLDOUT_SIDECAR.is_file():
        raise FileNotFoundError("preholdout selection must be written before Holdout generation")
    sidecar = load_json(PREHOLDOUT_SIDECAR)
    digest = sha256_file(PREHOLDOUT_SELECTION)
    if digest != sidecar["sha256"] or not sidecar["verified_after_write"]:
        raise RuntimeError("preholdout selection hash verification failed")
    return load_json(PREHOLDOUT_SELECTION), digest


def formal_provenance_sets(dataset_config: dict) -> tuple[set[str], set[str], set[str], set[str]]:
    formal = dataset_config["formal_synthetic"]
    graph_root = resolve_repo_path(formal["graph_root"])
    sample_ids: set[str] = set()
    base_keys: set[str] = set()
    exact: set[str] = set()
    near: set[str] = set()
    for split_path in formal["splits"].values():
        payload = load_json(resolve_repo_path(split_path))
        ids = payload if isinstance(payload, list) else payload["samples"]
        for sample_id in ids:
            if sample_id in sample_ids:
                continue
            sample_ids.add(sample_id)
            graph = load_json(graph_root / f"{sample_id}.json")
            base_keys.add(base_geometry_key(graph, sample_id))
            exact.add(rotation_invariant_graph_signature(graph, ndigits=6))
            near.add(rotation_invariant_graph_signature(graph, ndigits=3))
    return sample_ids, base_keys, exact, near


def generation_resources() -> dict[str, Any]:
    ontology_path = REPO_ROOT / "CAE_feature_ontology.yaml"
    semantic_path = REPO_ROOT / "face_semantic_subtype_ontology.yaml"
    ontology_maps = load_ontology_maps(ontology_path)
    primitive_surface_index = load_layer_index(ontology_path, "primitive_surface")
    semantic_subtype_index = load_layer_index(semantic_path, "face_semantic_subtype")
    idealization_tag_index = load_layer_index(ontology_path, "idealization_tag")
    static_config = yaml.safe_load(
        (REPO_ROOT / "configs" / "static_structural_v3_1_balanced_3000.yaml").read_text(
            encoding="utf-8"
        )
    )
    generation_context = build_generation_context(static_config["generation"])
    # A standalone rib has no L-plate corner context.  Use the generator's
    # established boundary placement mode so the quality gate can retry actual
    # rib candidates instead of deterministically producing empty samples.
    generation_context["rib_features"] = {
        "placement_modes": ["boundary"],
        "corner_probability": 0.0,
        "cavity_probability": 0.0,
    }
    return {
        "ontology_maps": ontology_maps,
        "primitive_surface_index": primitive_surface_index,
        "semantic_subtype_index": semantic_subtype_index,
        "idealization_tag_index": idealization_tag_index,
        "primitive_label_map": export_layer_label_map(primitive_surface_index),
        "semantic_label_map": export_layer_label_map(semantic_subtype_index),
        "idealization_label_map": export_layer_label_map(idealization_tag_index),
        "generation_context": generation_context,
    }


def acceptable_step_report(report: dict) -> bool:
    return bool(
        report.get("readable")
        and report.get("solid_count") == 1
        and report.get("valid")
        and not report.get("issues")
        and float(report.get("signed_volume", 0.0)) > 0.0
    )


def load_progress(
    progress_path: Path,
    *,
    selection_sha: str,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    if not progress_path.is_file():
        return [], Counter()
    progress = load_json(progress_path)
    if progress.get("preholdout_selection_sha256") != selection_sha:
        raise RuntimeError("Existing Holdout progress belongs to a different frozen selection")
    return (
        list(progress.get("accepted_records", [])),
        Counter(progress.get("failure_counts", {})),
    )


def main() -> None:
    selection, selection_sha = verify_preholdout()
    dataset_config = load_json(DATASET_CONFIG)
    config = load_json(HOLDOUT_CONFIG)
    output_root = resolve_repo_path(config["output_root"])
    complete_manifest = output_root / "holdout_generation_manifest.json"
    if complete_manifest.exists():
        raise FileExistsError("Synthetic-Holdout-v1 is already complete and immutable")

    started_ns = time.time_ns()
    if PREHOLDOUT_SELECTION.stat().st_mtime_ns >= started_ns:
        raise RuntimeError("preholdout selection must predate Holdout output creation")
    graphs_root = output_root / "graphs"
    steps_root = output_root / "steps"
    graphs_root.mkdir(parents=True, exist_ok=True)
    steps_root.mkdir(parents=True, exist_ok=True)
    progress_path = output_root / "generation_progress.json"

    original_ids, original_base, original_exact, original_near = formal_provenance_sets(
        dataset_config
    )
    resources = generation_resources()
    accepted, failure_counts = load_progress(
        progress_path, selection_sha=selection_sha
    )
    accepted_by_id = {record["sample_id"]: record for record in accepted}
    holdout_base = {record["base_geometry_key"] for record in accepted}
    holdout_exact = {record["exact_signature"] for record in accepted}
    holdout_near = {record["near_signature"] for record in accepted}
    target_classes = list(config["target_macro_features"])
    per_class = int(config["samples_per_class"])
    expected_count = len(target_classes) * per_class
    if expected_count != int(config["sample_count"]):
        raise ValueError("sample_count must equal target class count times samples_per_class")

    global_index = 0
    for class_name in target_classes:
        generator_name = CATALOG_GENERATOR_BY_MACRO[class_name]
        for class_index in range(per_class):
            sample_seed = int(config["seed_start"]) + global_index
            sample_id = f"holdout_{class_name}_{class_index:03d}"
            global_index += 1
            if sample_id in accepted_by_id:
                continue
            step_path = steps_root / f"{sample_id}.step"
            graph_path = graphs_root / f"{sample_id}.json"
            accepted_record = None
            for attempt in range(int(config["max_class_attempts_per_sample"])):
                attempt_seed = sample_seed + attempt * 10_000_000
                random.seed(attempt_seed)
                size = config["stock_size_range"]
                try:
                    graph = build_single_dataset_sample(
                        random.uniform(*size["L"]),
                        random.uniform(*size["W"]),
                        random.uniform(*size["H"]),
                        str(step_path),
                        str(graph_path),
                        ontology_maps=resources["ontology_maps"],
                        primitive_surface_index=resources["primitive_surface_index"],
                        semantic_subtype_index=resources["semantic_subtype_index"],
                        idealization_tag_index=resources["idealization_tag_index"],
                        primitive_label_map=resources["primitive_label_map"],
                        semantic_label_map=resources["semantic_label_map"],
                        idealization_label_map=resources["idealization_label_map"],
                        generator_names=[generator_name],
                        instance_count_range=tuple(config["instance_count_range"]),
                        require_nonempty_sample=True,
                        max_generation_attempts=int(config["max_generation_attempts"]),
                        write_step=True,
                        write_graph=True,
                        generation_context=resources["generation_context"],
                    )
                except RuntimeError:
                    failure_counts["generation_candidate_exhausted"] += 1
                    continue
                if not is_valid_catalog_sample(graph, class_name):
                    failure_counts["class_mismatch"] += 1
                    continue
                step_report = inspect_step_file(step_path)
                if not acceptable_step_report(step_report):
                    failure_counts["step_quality"] += 1
                    continue
                base_key = base_geometry_key(
                    {**graph, "base_sample_id": f"holdout_seed_{sample_seed}"}, sample_id
                )
                exact = rotation_invariant_graph_signature(graph, ndigits=6)
                near = rotation_invariant_graph_signature(graph, ndigits=3)
                if base_key in original_base or base_key in holdout_base:
                    failure_counts["base_provenance_collision"] += 1
                    continue
                if exact in original_exact or exact in holdout_exact:
                    failure_counts["exact_signature_collision"] += 1
                    continue
                if near in original_near or near in holdout_near:
                    failure_counts["near_signature_collision"] += 1
                    continue
                graph["base_sample_id"] = f"holdout_seed_{sample_seed}"
                graph["p1_a2_holdout_provenance"] = {
                    "dataset": "Synthetic-Holdout-v1",
                    "sample_seed": sample_seed,
                    "accepted_attempt": attempt + 1,
                    "attempt_seed": attempt_seed,
                    "target_macro_feature": class_name,
                    "preholdout_selection_sha256": selection_sha,
                }
                write_json(graph_path, graph)
                accepted_record = {
                    "sample_id": sample_id,
                    "target_macro_feature": class_name,
                    "class_index": class_index,
                    "sample_seed": sample_seed,
                    "attempt_seed": attempt_seed,
                    "accepted_attempt": attempt + 1,
                    "graph_path": str(graph_path.resolve()),
                    "step_path": str(step_path.resolve()),
                    "graph_sha256": sha256_file(graph_path),
                    "step_sha256": sha256_file(step_path),
                    "face_count": len(graph.get("nodes", [])),
                    "feature_instance_count": len(graph.get("feature_sets", [])),
                    "base_geometry_key": base_key,
                    "exact_signature": exact,
                    "near_signature": near,
                    "step_validation": step_report,
                }
                holdout_base.add(base_key)
                holdout_exact.add(exact)
                holdout_near.add(near)
                accepted.append(accepted_record)
                accepted_by_id[sample_id] = accepted_record
                break
            if accepted_record is None:
                raise RuntimeError(
                    f"Failed to generate collision-free valid {sample_id} after "
                    f"{config['max_class_attempts_per_sample']} attempts"
                )
            if len(accepted) % 20 == 0 or len(accepted) == expected_count:
                write_json(
                    progress_path,
                    {
                        "schema": "topocae_synthetic_holdout_v1_progress",
                        "preholdout_selection_sha256": selection_sha,
                        "accepted_count": len(accepted),
                        "failure_counts": dict(failure_counts),
                        "accepted_records": accepted,
                    },
                )
                print(
                    f"holdout_generation={len(accepted)}/{expected_count} "
                    f"class={class_name} failures={sum(failure_counts.values())}"
                )

    accepted.sort(key=lambda record: record["sample_id"])
    class_counts = Counter(record["target_macro_feature"] for record in accepted)
    if len(accepted) != expected_count or any(
        class_counts[class_name] != per_class for class_name in target_classes
    ):
        raise RuntimeError("Holdout class balance invariant failed")
    direct_valid = sum(acceptable_step_report(record["step_validation"]) for record in accepted)
    fingerprint_audit = {
        "schema": "topocae_synthetic_holdout_v1_fingerprint_audit",
        "formal_original_sample_count": len(original_ids),
        "holdout_sample_count": len(accepted),
        "base_provenance_collision_with_original_count": sum(
            record["base_geometry_key"] in original_base for record in accepted
        ),
        "rotation_invariant_exact_collision_with_original_count": sum(
            record["exact_signature"] in original_exact for record in accepted
        ),
        "rotation_invariant_near_collision_with_original_count": sum(
            record["near_signature"] in original_near for record in accepted
        ),
        "holdout_internal_base_collision_count": len(accepted) - len(holdout_base),
        "holdout_internal_exact_collision_count": len(accepted) - len(holdout_exact),
        "holdout_internal_near_collision_count": len(accepted) - len(holdout_near),
        "signature_method": "rotation_invariant_graph_signature ndigits=6 exact and ndigits=3 near",
        "status": "pass",
    }
    quality_report = {
        "schema": "topocae_synthetic_holdout_v1_quality_report",
        "requested_sample_count": expected_count,
        "accepted_sample_count": len(accepted),
        "direct_occ_valid_single_solid_count": direct_valid,
        "invalid_or_non_single_solid_count": len(accepted) - direct_valid,
        "class_counts": dict(sorted(class_counts.items())),
        "samples_per_class": per_class,
        "generation_retry_counts": dict(sorted(failure_counts.items())),
        "status": "pass" if direct_valid == expected_count else "fail",
    }
    graph_step_manifest = {
        "schema": "topocae_synthetic_holdout_v1_graph_step_manifest",
        "sample_count": len(accepted),
        "records": accepted,
    }
    completed_ns = time.time_ns()
    manifest = {
        "schema": "topocae_synthetic_holdout_v1_generation_manifest",
        "phase": "P1-A2",
        "dataset_name": "Synthetic-Holdout-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(accepted),
        "samples_per_class": per_class,
        "class_counts": dict(sorted(class_counts.items())),
        "seed_start": config["seed_start"],
        "seed_end": config["seed_end"],
        "generation_config_path": str(HOLDOUT_CONFIG.resolve()),
        "generation_config_sha256": sha256_file(HOLDOUT_CONFIG),
        "preholdout_selection_path": str(PREHOLDOUT_SELECTION.resolve()),
        "preholdout_selection_sha256": selection_sha,
        "preholdout_mtime_ns": PREHOLDOUT_SELECTION.stat().st_mtime_ns,
        "holdout_generation_started_ns": started_ns,
        "holdout_generation_completed_ns": completed_ns,
        "preholdout_strictly_predates_generation": PREHOLDOUT_SELECTION.stat().st_mtime_ns < started_ns,
        "formal_original_seed": 20260623,
        "new_seed_range_disjoint": int(config["seed_start"]) > 20260623,
        "generation_context_overrides": {
            "standalone_rib_placement_modes": ["boundary"],
            "reason": "standalone samples have no L-plate corner or cavity parent context",
        },
        "selection_locked_variant": selection["selected_variant"],
        "inference_pass_count": 0,
    }
    write_json(output_root / "fingerprint_audit.json", fingerprint_audit)
    write_json(output_root / "quality_report.json", quality_report)
    write_json(output_root / "graph_step_manifest.json", graph_step_manifest)
    write_json(complete_manifest, manifest)
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "sample_count": len(accepted),
                "class_balance": dict(sorted(class_counts.items())),
                "solid_valid": direct_valid,
                "fingerprint_audit": fingerprint_audit["status"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
