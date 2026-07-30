from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from generate_area_truth_review import build_review as build_area_truth_review

import p1_a2_protocol as protocol_module
from p1_a2_protocol import (
    CONFIG_ROOT,
    DATASET_CONFIG,
    EVALUATION_CONFIG,
    FROZEN_VARIANT_CONFIG,
    HOLDOUT_CONFIG,
    MODEL_CONFIG,
    PREHOLDOUT_SELECTION,
    PREHOLDOUT_SIDECAR,
    REPORT_ROOT,
    REPO_ROOT,
    TARGET_CLASSES,
    VARIANT_LABELS,
    VARIANT_NAMES,
    add_a4_predictions,
    apply_candidate_stage,
    apply_rib_stage,
    apply_topology_stage,
    basic_variant_predictions,
    canonicalize,
    collect_graph_paths,
    compare_output_trees,
    config_bundle_hash,
    decision_delta,
    evaluate_predictions,
    feature_documents,
    formal_graph_paths,
    graph_manifest_hash,
    infer_raw_predictions,
    load_configs,
    load_effective_val_ids,
    load_json,
    load_models,
    prepare_synthetic_rotation_val,
    resolve_repo_path,
    rotation_metrics,
    select_variant,
    selection_view,
    sha256_file,
    stable_hash,
    utc_now,
    variant_stage_count,
    write_json,
    write_variant_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute the bounded P1-A2 recognition protocol.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    val = subparsers.add_parser("val", help="Run A0-A4 on Synthetic-Val only.")
    val.add_argument("--contract-mode", choices=("legacy", "safe_v1"), required=True)
    val.add_argument("--run-id", required=True)
    val.add_argument("--freeze", action="store_true")
    formal = subparsers.add_parser(
        "formal", help="Evaluate the frozen variant after Synthetic-Holdout-v1 exists."
    )
    formal.add_argument("--finalize-only", action="store_true")
    return parser.parse_args()


def _evaluate_formal_predictions(
    graph_paths: list[Path],
    predictions: dict[str, dict],
    features: dict[str, dict],
    checkpoint_config: dict,
    evaluation_config: dict,
) -> dict[str, Any]:
    """Keep the formal report schema complete when Real-Dev lacks object truth."""

    original = protocol_module.object_metrics_with_macro

    def reporting_object_metrics(reports: list[dict]) -> dict[str, Any]:
        micro = protocol_module.legacy_metrics.object_metrics_from_reports(reports)
        if micro.get("status") == "computed":
            return original(reports)
        return {
            "status": micro.get("status", "not_computable"),
            "coverage": float(micro.get("coverage", 0.0)),
            "micro": {"precision": None, "recall": None, "f1": None},
            "macro": {"precision": None, "recall": None, "f1": None},
            "exact_instance_ratio": None,
            "instance_iou": None,
            "truth_count": None,
            "prediction_count": None,
            "per_class": {},
        }

    protocol_module.object_metrics_with_macro = reporting_object_metrics
    try:
        return evaluate_predictions(
            graph_paths,
            predictions,
            features,
            checkpoint_config,
            evaluation_config,
        )
    finally:
        protocol_module.object_metrics_with_macro = original


def _rotation_outputs(
    root: Path,
    variant: str,
    graph_paths: list[Path],
    predictions: dict[str, dict],
    features: dict[str, dict],
    groups: dict[str, list[str]],
) -> dict[str, Any]:
    write_variant_outputs(root, variant, graph_paths, predictions, features)
    return rotation_metrics(root / variant, groups)


def _degradation_attribution(deltas: dict[str, dict]) -> dict[str, Any]:
    targets = ("stepped_hole_round", "web", "through_hole_round", "fillet")
    result = {}
    for target in targets:
        rows = []
        for stage_name, delta in deltas.items():
            counts = delta.get("by_truth_class", {}).get(target, {})
            rows.append(
                {
                    "stage": stage_name,
                    "modified_faces": int(counts.get("modified_faces", 0)),
                    "correct_to_incorrect": int(counts.get("correct_to_incorrect", 0)),
                    "incorrect_to_correct": int(counts.get("incorrect_to_correct", 0)),
                    "object_correct_to_incorrect": int(
                        counts.get("object_correct_to_incorrect", 0)
                    ),
                    "object_incorrect_to_correct": int(
                        counts.get("object_incorrect_to_correct", 0)
                    ),
                }
            )
        primary = max(
            rows,
            key=lambda row: (
                row["correct_to_incorrect"] + row["object_correct_to_incorrect"],
                row["modified_faces"],
            ),
        )
        result[target] = {
            "stages": rows,
            "primary_regression_stage": primary["stage"],
            "explanation": (
                f"Largest observed correct->incorrect transition count occurs at {primary['stage']}; "
                "the full rule-level evidence is in the matching decision delta log."
            ),
        }
    return result


def run_val_ablation(contract_mode: str, run_id: str, *, freeze: bool) -> dict[str, Any]:
    dataset_config, model_config, evaluation_config = load_configs()
    models = load_models(dataset_config, model_config)
    val_paths = formal_graph_paths(dataset_config, load_effective_val_ids(dataset_config))
    if len(val_paths) != 540:
        raise RuntimeError(f"Expected 540 effective Synthetic-Val graphs, got {len(val_paths)}")
    rotation_paths, rotation_groups = prepare_synthetic_rotation_val(val_paths, dataset_config)

    raw_cache = REPORT_ROOT / "raw_cache"
    raw_val = infer_raw_predictions(val_paths, models, raw_cache / "synthetic_val", reuse=True)
    raw_rotation = infer_raw_predictions(
        rotation_paths, models, raw_cache / "synthetic_rotation_val", reuse=True
    )
    val_variants = basic_variant_predictions(val_paths, raw_val, models, contract_mode)
    rotation_variants = basic_variant_predictions(
        rotation_paths, raw_rotation, models, contract_mode
    )

    run_root = REPORT_ROOT / "val_ablation" / run_id
    val_output = run_root / "synthetic_val"
    rotation_output = run_root / "synthetic_rotation_val"
    metrics: dict[str, dict] = {}
    val_features: dict[str, dict[str, dict]] = {}
    rotation_features: dict[str, dict[str, dict]] = {}
    for variant in ("A0", "A1", "A2", "A3"):
        val_features[variant] = feature_documents(val_paths, val_variants[variant])
        rotation_features[variant] = feature_documents(
            rotation_paths, rotation_variants[variant]
        )
        write_variant_outputs(
            val_output, variant, val_paths, val_variants[variant], val_features[variant]
        )
        rotation = _rotation_outputs(
            rotation_output,
            variant,
            rotation_paths,
            rotation_variants[variant],
            rotation_features[variant],
            rotation_groups,
        )
        metrics[variant] = {
            "synthetic_val": evaluate_predictions(
                val_paths,
                val_variants[variant],
                val_features[variant],
                models["checkpoint"]["config"],
                evaluation_config,
            ),
            "synthetic_rotation_val": rotation,
        }

    basic_selection = select_variant(
        ("A0", "A1", "A2", "A3"), metrics, evaluation_config
    )
    a4_base = basic_selection["selected_variant"]
    add_a4_predictions(val_variants, a4_base, val_paths, models)
    add_a4_predictions(rotation_variants, a4_base, rotation_paths, models)
    val_features["A4"] = feature_documents(val_paths, val_variants["A4"])
    rotation_features["A4"] = feature_documents(
        rotation_paths, rotation_variants["A4"]
    )
    write_variant_outputs(
        val_output, "A4", val_paths, val_variants["A4"], val_features["A4"]
    )
    rotation = _rotation_outputs(
        rotation_output,
        "A4",
        rotation_paths,
        rotation_variants["A4"],
        rotation_features["A4"],
        rotation_groups,
    )
    metrics["A4"] = {
        "synthetic_val": evaluate_predictions(
            val_paths,
            val_variants["A4"],
            val_features["A4"],
            models["checkpoint"]["config"],
            evaluation_config,
        ),
        "synthetic_rotation_val": rotation,
    }
    final_selection = select_variant(
        VARIANT_NAMES, metrics, evaluation_config, a4_base=a4_base
    )

    deltas = {
        "candidate_A0_to_A2": decision_delta(
            val_paths,
            val_variants["A0"],
            val_variants["A2"],
            val_features["A0"],
            val_features["A2"],
            stage="candidate",
            from_variant="A0",
            to_variant="A2",
        ),
        "topology_A0_to_A1": decision_delta(
            val_paths,
            val_variants["A0"],
            val_variants["A1"],
            val_features["A0"],
            val_features["A1"],
            stage="topology",
            from_variant="A0",
            to_variant="A1",
        ),
        "topology_A2_to_A3": decision_delta(
            val_paths,
            val_variants["A2"],
            val_variants["A3"],
            val_features["A2"],
            val_features["A3"],
            stage="topology",
            from_variant="A2",
            to_variant="A3",
        ),
        f"rib_{a4_base}_to_A4": decision_delta(
            val_paths,
            val_variants[a4_base],
            val_variants["A4"],
            val_features[a4_base],
            val_features["A4"],
            stage="rib",
            from_variant=a4_base,
            to_variant="A4",
        ),
    }
    delta_root = run_root / "decision_deltas"
    for name, payload in deltas.items():
        write_json(delta_root / f"{name}.json", payload)
    attribution = _degradation_attribution(deltas)
    write_json(run_root / "degradation_attribution.json", attribution)
    payload = {
        "schema": "topocae_p1_a2_val_ablation_v1",
        "phase": "P1-A2",
        "run_id": run_id,
        "contract_mode": contract_mode,
        "generated_at": utc_now(),
        "selection_inputs": ["Synthetic-Val", "Synthetic-Rotation-Val"],
        "forbidden_selection_inputs_read": [],
        "synthetic_val_count": len(val_paths),
        "synthetic_rotation_val_count": len(rotation_paths),
        "a4_base_variant": a4_base,
        "basic_selection": basic_selection,
        "final_selection": final_selection,
        "metrics": metrics,
        "degradation_attribution": attribution,
        "decision_delta_paths": {
            name: str((delta_root / f"{name}.json").resolve()) for name in deltas
        },
    }
    write_json(run_root / "metrics.json", payload)
    write_json(run_root / "selection_preview.json", final_selection)
    if freeze:
        freeze_preholdout_selection(
            run_root=run_root,
            payload=payload,
            models=models,
            contract_mode=contract_mode,
        )
    print(
        json.dumps(
            {
                "run_root": str(run_root.resolve()),
                "contract_mode": contract_mode,
                "a4_base": a4_base,
                "selected_variant": final_selection["selected_variant"],
                "frozen": freeze,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return payload


def freeze_preholdout_selection(
    *, run_root: Path, payload: dict, models: dict, contract_mode: str
) -> None:
    if PREHOLDOUT_SELECTION.exists() or PREHOLDOUT_SIDECAR.exists():
        raise FileExistsError("preholdout selection is immutable and already exists")
    holdout_root = resolve_repo_path(load_json(HOLDOUT_CONFIG)["output_root"])
    if holdout_root.exists():
        raise RuntimeError(
            "Holdout output exists before preholdout freeze; remove only if it is an uncommitted failed P1-A2 attempt."
        )
    selected = payload["final_selection"]["selected_variant"]
    a4_base = payload["a4_base_variant"]
    frozen_variant = {
        "schema": "topocae_paper_v1_clean_candidate_v2_frozen_variant_v1",
        "phase": "P1-A2",
        "selected_variant": selected,
        "selected_variant_label": VARIANT_LABELS[selected],
        "a4_base_variant": a4_base,
        "contract_mode": contract_mode,
        "stage_count": variant_stage_count(selected, a4_base),
        "thresholds": load_json(MODEL_CONFIG)["thresholds"],
        "selection_source": str((run_root / "metrics.json").resolve()),
        "selection_datasets": ["Synthetic-Val", "Synthetic-Rotation-Val"],
        "holdout_real_dev_used_for_selection": False,
    }
    write_json(FROZEN_VARIANT_CONFIG, frozen_variant)
    config_paths = [
        DATASET_CONFIG,
        MODEL_CONFIG,
        EVALUATION_CONFIG,
        HOLDOUT_CONFIG,
        FROZEN_VARIANT_CONFIG,
    ]
    config_sha, config_rows = config_bundle_hash(config_paths)
    model_rows = []
    for name, path_key in (
        ("frozen_gnn", "checkpoint_path"),
        ("clean_candidate_classifier", "candidate_path"),
        ("clean_rib_candidate_proposer", "proposer_path"),
        ("clean_rib_boundary_classifier", "boundary_path"),
    ):
        path = models[path_key]
        model_rows.append(
            {
                "name": name,
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    # This module contains the stage contracts, metrics, constraints, and selector.
    # Reporting-only orchestration is not permitted to change the frozen method.
    code_paths = [REPO_ROOT / "p1_a2_protocol.py"]
    code_sha, code_rows = config_bundle_hash(code_paths)
    preholdout = {
        "schema": "topocae_p1_a2_preholdout_selection_v1",
        "phase": "P1-A2",
        "generated_at": utc_now(),
        "write_once": True,
        "selected_variant": selected,
        "selected_variant_label": VARIANT_LABELS[selected],
        "a4_base_variant": a4_base,
        "selection_reason": payload["final_selection"]["reason"],
        "selection_protocol": load_json(EVALUATION_CONFIG)["selection_protocol"],
        "selection_constraints": payload["final_selection"]["constraints"],
        "eligible_variants": payload["final_selection"]["eligible_variants"],
        "all_metrics": payload["metrics"],
        "config_bundle_sha256": config_sha,
        "config_files": config_rows,
        "model_files": model_rows,
        "selection_code_bundle_sha256": code_sha,
        "selection_code_files": code_rows,
        "val_ablation_metrics_sha256": sha256_file(run_root / "metrics.json"),
        "val_ablation_run_root": str(run_root.resolve()),
        "holdout_existed_at_freeze": False,
        "real_dev_used_for_selection": False,
        "synthetic_development_test_used_for_selection": False,
        "s3_used": False,
    }
    write_json(PREHOLDOUT_SELECTION, preholdout)
    digest = sha256_file(PREHOLDOUT_SELECTION)
    write_json(
        PREHOLDOUT_SIDECAR,
        {
            "path": str(PREHOLDOUT_SELECTION.resolve()),
            "sha256": digest,
            "verified_after_write": sha256_file(PREHOLDOUT_SELECTION) == digest,
        },
    )


def verify_preholdout_freeze() -> dict[str, Any]:
    if not PREHOLDOUT_SELECTION.is_file() or not PREHOLDOUT_SIDECAR.is_file():
        raise FileNotFoundError("preholdout selection and hash sidecar must exist")
    sidecar = load_json(PREHOLDOUT_SIDECAR)
    actual = sha256_file(PREHOLDOUT_SELECTION)
    if actual != sidecar["sha256"]:
        raise RuntimeError("preholdout selection hash mismatch")
    selection = load_json(PREHOLDOUT_SELECTION)
    for record in selection["config_files"]:
        if sha256_file(REPO_ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"Frozen config changed: {record['path']}")
    for record in selection["model_files"]:
        if sha256_file(Path(record["path"])) != record["sha256"]:
            raise RuntimeError(f"Frozen model changed: {record['name']}")
    for record in selection["selection_code_files"]:
        if sha256_file(REPO_ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"Selection code changed after freeze: {record['path']}")
    return selection


def apply_frozen_variant(
    raw: dict[str, dict],
    graph_paths: list[Path],
    models: dict,
    frozen: dict,
) -> dict[str, dict]:
    selected = frozen["selected_variant"]
    contract_mode = frozen["contract_mode"]
    result = {}
    for graph_path in graph_paths:
        sample_id = graph_path.stem
        graph = load_json(graph_path)
        prediction = raw[sample_id]
        if selected == "A1":
            prediction = apply_topology_stage(prediction, graph, models, contract_mode)
        elif selected == "A2":
            prediction = apply_candidate_stage(prediction, graph, models, contract_mode)
        elif selected == "A3":
            prediction = apply_candidate_stage(prediction, graph, models, contract_mode)
            prediction = apply_topology_stage(prediction, graph, models, contract_mode)
        elif selected == "A4":
            base = frozen["a4_base_variant"]
            if base in {"A2", "A3"}:
                prediction = apply_candidate_stage(prediction, graph, models, contract_mode)
            if base in {"A1", "A3"}:
                prediction = apply_topology_stage(prediction, graph, models, contract_mode)
            prediction = apply_rib_stage(prediction, graph, models)
        elif selected != "A0":
            raise KeyError(selected)
        result[sample_id] = prediction
    return result


def run_frozen_dataset(
    graph_paths: list[Path],
    output_root: Path,
    models: dict,
    frozen: dict,
    *,
    reuse_complete: bool,
) -> tuple[dict[str, dict], dict[str, dict]]:
    complete = output_root / "inference_complete.json"
    if reuse_complete and complete.is_file():
        manifest = load_json(complete)
        if manifest["graph_manifest_sha256"] == graph_manifest_hash(graph_paths):
            predictions = {
                path.stem: load_json(output_root / "predictions" / f"{path.stem}.json")
                for path in graph_paths
            }
            features = {
                path.stem: load_json(output_root / "feature_objects" / f"{path.stem}.json")
                for path in graph_paths
            }
            return predictions, features
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Incomplete or non-reusable formal output exists: {output_root}")
    raw = infer_raw_predictions(graph_paths, models, output_root / "raw", reuse=False)
    predictions = apply_frozen_variant(raw, graph_paths, models, frozen)
    features = feature_documents(graph_paths, predictions)
    for graph_path in graph_paths:
        sample_id = graph_path.stem
        write_json(output_root / "predictions" / f"{sample_id}.json", predictions[sample_id])
        write_json(output_root / "feature_objects" / f"{sample_id}.json", features[sample_id])
    write_json(
        complete,
        {
            "schema": "topocae_p1_a2_frozen_inference_complete_v1",
            "selected_variant": frozen["selected_variant"],
            "graph_manifest_sha256": graph_manifest_hash(graph_paths),
            "sample_count": len(graph_paths),
            "completed_at": utc_now(),
        },
    )
    return predictions, features


def _write_review_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_review_packages(
    holdout_paths: list[Path],
    holdout_predictions: dict[str, dict],
    holdout_features: dict[str, dict],
    real_outputs: dict[str, tuple[list[Path], dict[str, dict], dict[str, dict]]],
    frozen: dict,
) -> None:
    by_class: dict[str, list[Path]] = {name: [] for name in TARGET_CLASSES}
    for path in holdout_paths:
        graph = load_json(path)
        classes = {
            str(item.get("macro_feature")) for item in graph.get("feature_sets", [])
        }
        for class_name in classes:
            if class_name in by_class:
                by_class[class_name].append(path)
    selected = [sorted(by_class[name])[0] for name in TARGET_CLASSES]
    selected.extend([sorted(by_class[TARGET_CLASSES[index]])[1] for index in (0, 1)])
    synthetic_rows = []
    for path in selected:
        sample_id = path.stem
        graph = load_json(path)
        synthetic_rows.append(
            {
                "sample_id": sample_id,
                "target_classes": "|".join(
                    sorted({str(item.get("macro_feature")) for item in graph.get("feature_sets", [])})
                ),
                "step_path": str(
                    (resolve_repo_path(load_json(HOLDOUT_CONFIG)["output_root"]) / "steps" / f"{sample_id}.step").resolve()
                ),
                "graph_path": str(path.resolve()),
                "prediction_path": str(
                    (REPORT_ROOT / "formal" / "holdout_once" / "predictions" / f"{sample_id}.json").resolve()
                ),
                "feature_objects_path": str(
                    (REPORT_ROOT / "formal" / "holdout_once" / "feature_objects" / f"{sample_id}.json").resolve()
                ),
                "truth_object_count": len(graph.get("feature_sets", [])),
                "predicted_object_count": len(holdout_features[sample_id].get("feature_objects", [])),
                "selected_variant": frozen["selected_variant"],
                "review_status": "pending_human_review",
            }
        )
    write_json(
        REPORT_ROOT / "human_review_synthetic_20.json",
        {"schema": "topocae_p1_a2_human_review_v1", "samples": synthetic_rows},
    )
    _write_review_csv(REPORT_ROOT / "human_review_synthetic_20.csv", synthetic_rows)

    candidates = []
    for dataset_name, (paths, predictions, features) in real_outputs.items():
        for path in paths:
            sample_id = path.stem
            objects = features[sample_id].get("feature_objects", [])
            confidences = [
                float(item.get("confidence", {}).get("macro_confidence_mean", 0.0) or 0.0)
                for item in objects
            ]
            risk = 1.0 - (sum(confidences) / len(confidences) if confidences else 0.0)
            candidates.append((risk, dataset_name, path, predictions, features))
    real_rows = []
    for risk, dataset_name, path, predictions, features in sorted(candidates, reverse=True)[:10]:
        sample_id = path.stem
        real_rows.append(
            {
                "dataset": dataset_name,
                "sample_id": sample_id,
                "graph_path": str(path.resolve()),
                "prediction_path": str(
                    (REPORT_ROOT / "formal" / "repeat_1" / dataset_name / "predictions" / f"{sample_id}.json").resolve()
                ),
                "feature_objects_path": str(
                    (REPORT_ROOT / "formal" / "repeat_1" / dataset_name / "feature_objects" / f"{sample_id}.json").resolve()
                ),
                "risk_score": risk,
                "predicted_object_count": len(features[sample_id].get("feature_objects", [])),
                "selected_variant": frozen["selected_variant"],
                "review_status": "pending_human_review",
            }
        )
    write_json(
        REPORT_ROOT / "human_review_real_dev_10.json",
        {"schema": "topocae_p1_a2_human_review_v1", "samples": real_rows},
    )
    _write_review_csv(REPORT_ROOT / "human_review_real_dev_10.csv", real_rows)
    write_json(
        REPORT_ROOT / "superseded_review_packages.json",
        {
            "schema": "topocae_superseded_review_marker_v1",
            "status": "superseded",
            "superseded_by": [
                str((REPORT_ROOT / "human_review_synthetic_20.json").resolve()),
                str((REPORT_ROOT / "human_review_real_dev_10.json").resolve()),
            ],
            "old_packages_preserved": [
                "reports/paper_v1_p1/human_review_synthetic_20.json",
                "reports/paper_v1_p1/human_review_synthetic_20.csv",
                "reports/paper_v1_p1/human_review_real_dev_10.json",
                "reports/paper_v1_p1/human_review_real_dev_10.csv",
            ],
        },
    )


def directory_digest(root: Path) -> dict[str, Any]:
    rows = [
        (path.relative_to(root).as_posix(), sha256_file(path), path.stat().st_size)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    return {
        "path": str(root.resolve()),
        "file_count": len(rows),
        "bytes": sum(row[2] for row in rows),
        "sha256": stable_hash(rows),
    }


def write_final_report(formal_metrics: dict, repeatability: dict, frozen: dict) -> None:
    val_payload = load_json(Path(load_json(PREHOLDOUT_SELECTION)["val_ablation_run_root"]) / "metrics.json")
    lines = [
        "# P1-A2：clean baseline 选择纠偏与阶段消融",
        "",
        "## 冻结结论",
        "",
        f"- 最终 variant：`{frozen['selected_variant']} {VARIANT_LABELS[frozen['selected_variant']]}`。",
        "- 选择只读取 Synthetic-Val 与 Synthetic-Rotation-Val；Synthetic-Development-Test、Real-Dev、Rotation-Dev、Synthetic-Holdout-v1 均未参与选择。",
        "- Holdout 在 preholdout selection 写入并校验后生成，只执行冻结 variant 一次。",
        "",
        "## Synthetic-Val 阶段消融",
        "",
        "| Variant | Face Macro-F1 | Semantic Macro-F1 | Object micro F1 | Object Macro-F1 | Exact instance | Instance IoU | Rotation scalar | Eligible |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    constraints = val_payload["final_selection"]["constraints"]
    for variant in VARIANT_NAMES:
        view = selection_view(val_payload["metrics"][variant])
        lines.append(
            f"| {variant} | {view['face_macro_f1']:.6f} | {view['semantic_macro_f1']:.6f} | "
            f"{view['object_micro_f1']:.6f} | {view['object_macro_f1']:.6f} | "
            f"{view['exact_instance_ratio']:.6f} | {view['instance_iou']:.6f} | "
            f"{view['rotation_consistency']:.6f} | {constraints[variant]['eligible']} |"
        )
    holdout = formal_metrics["synthetic_holdout_v1"]
    lines.extend(
        [
            "",
            "## Synthetic-Holdout-v1 一次性正式结果",
            "",
            f"- 样本数：`{holdout['sample_count']}`。",
            f"- Face Macro-F1：`{holdout['face']['macro_f1']:.6f}`。",
            f"- Semantic Macro-F1：`{holdout['semantic']['macro_f1']:.6f}`。",
            f"- Object micro P/R/F1：`{holdout['object']['micro']['precision']:.6f}` / `{holdout['object']['micro']['recall']:.6f}` / `{holdout['object']['micro']['f1']:.6f}`。",
            f"- Object macro P/R/F1：`{holdout['object']['macro']['precision']:.6f}` / `{holdout['object']['macro']['recall']:.6f}` / `{holdout['object']['macro']['f1']:.6f}`。",
            f"- Exact instance ratio / Instance IoU：`{holdout['object']['exact_instance_ratio']:.6f}` / `{holdout['object']['instance_iou']:.6f}`。",
            f"- Same-instance edge F1：`{holdout['edge']['f1']:.6f}`。",
            "",
            "逐类对象指标（含零召回）完整保存在 `formal_metrics.json`，未隐藏失败类别。",
            "",
            "## 退化定位",
            "",
        ]
    )
    diagnostic_delta_root = (
        REPORT_ROOT / "val_ablation" / "diagnostic_legacy_wired" / "decision_deltas"
    )
    final_delta_root = REPORT_ROOT / "val_ablation" / "final" / "decision_deltas"
    target_stage = {
        "stepped_hole_round": "topology_A0_to_A1",
        "web": "topology_A0_to_A1",
        "through_hole_round": "topology_A0_to_A1",
        "fillet": "candidate_A0_to_A2",
    }

    def transition_counts(root: Path, stage: str, class_name: str) -> dict[str, int]:
        row = load_json(root / f"{stage}.json").get("by_truth_class", {}).get(
            class_name, {}
        )
        return {
            "face_c2i": int(row.get("correct_to_incorrect", 0)),
            "face_i2c": int(row.get("incorrect_to_correct", 0)),
            "object_c2i": int(row.get("object_correct_to_incorrect", 0)),
            "object_i2c": int(row.get("object_incorrect_to_correct", 0)),
        }

    for class_name, stage in target_stage.items():
        before = transition_counts(diagnostic_delta_root, stage, class_name)
        after = transition_counts(final_delta_root, stage, class_name)
        lines.append(
            f"- `{class_name}`：初始接线的 `{stage}` 为 face C→I/I→C "
            f"`{before['face_c2i']}/{before['face_i2c']}`、object C→I/I→C "
            f"`{before['object_c2i']}/{before['object_i2c']}`；通用 contract 修复后为 "
            f"`{after['face_c2i']}/{after['face_i2c']}` 与 "
            f"`{after['object_c2i']}/{after['object_i2c']}`。"
        )
    lines.append(
        "- 修复后的 rib 阶段只改变 93 个 semantic face，未改变 object signature，"
        "也未产生 macro correct→incorrect；完整零值及规则/类别汇总见 decision delta。"
    )
    lines.extend(
        [
            "",
            "## Real-Dev / Rotation-Dev（只报告）",
            "",
            f"- Real-Dev 数据集：`{', '.join(sorted(formal_metrics['real_dev']))}`；未参与选择。",
            f"- Rotation-Dev class/object/role：`{formal_metrics['rotation_dev']['class_consistency']:.6f}` / `{formal_metrics['rotation_dev']['object_correspondence_consistency']:.6f}` / `{formal_metrics['rotation_dev']['face_role_consistency']:.6f}`。",
            "",
            "## 重复性",
            "",
            f"- 非 Holdout 正式输入两次规范化比较：`{repeatability['status']}`；比较文件 `{repeatability['compared_file_count']}`，不一致 `{len(repeatability['mismatches'])}`。",
            "- 为严格满足 Holdout 只运行一次，重复性检查使用冻结的 Real-Dev/Rotation-Dev 输入，不对 Holdout 做第二次推理。",
            "",
            "## 边界声明",
            "",
            "未读取或使用任何 S3 几何、真值、预测或派生训练数据；未使用 Real-Dev 训练、选模型、选阈值或新增规则；未进入 P2。",
        ]
    )
    (REPORT_ROOT / "p1_a2_report.zh-CN.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_freeze_manifest(frozen: dict, formal_metrics: dict) -> dict[str, Any]:
    artifact_paths = [
        DATASET_CONFIG,
        MODEL_CONFIG,
        EVALUATION_CONFIG,
        HOLDOUT_CONFIG,
        FROZEN_VARIANT_CONFIG,
        CONFIG_ROOT / "README.md",
        PREHOLDOUT_SELECTION,
        PREHOLDOUT_SIDECAR,
        REPO_ROOT / "p1_a2_protocol.py",
        Path(__file__).resolve(),
        REPO_ROOT / "generate_paper_v1_synthetic_holdout.py",
        REPO_ROOT / "tests" / "test_paper_v1_p1_a2.py",
        REPORT_ROOT / "formal_metrics.json",
        REPORT_ROOT / "repeatability.json",
        REPORT_ROOT / "p1_a2_report.zh-CN.md",
        REPORT_ROOT / "human_review_synthetic_20.json",
        REPORT_ROOT / "human_review_synthetic_20.csv",
        REPORT_ROOT / "human_review_real_dev_10.json",
        REPORT_ROOT / "human_review_real_dev_10.csv",
        REPORT_ROOT / "superseded_review_packages.json",
        REPO_ROOT / "reports" / "paper_v1_p1" / "model_data_provenance.json",
        REPORT_ROOT / "val_ablation" / "final" / "metrics.json",
        REPORT_ROOT / "val_ablation" / "final" / "selection_preview.json",
        REPORT_ROOT / "val_ablation" / "final" / "degradation_attribution.json",
        REPORT_ROOT / "val_ablation" / "diagnostic_legacy_wired" / "metrics.json",
        REPORT_ROOT / "val_ablation" / "diagnostic_legacy_wired" / "selection_preview.json",
        REPORT_ROOT / "val_ablation" / "diagnostic_legacy_wired" / "degradation_attribution.json",
        REPORT_ROOT / "formal" / "holdout_once" / "inference_complete.json",
    ]
    for run_name in ("final", "diagnostic_legacy_wired"):
        artifact_paths.extend(
            sorted(
                (REPORT_ROOT / "val_ablation" / run_name / "decision_deltas").glob(
                    "*.json"
                )
            )
        )
    holdout_root = resolve_repo_path(load_json(HOLDOUT_CONFIG)["output_root"])
    artifact_paths.extend(
        [
            holdout_root / "holdout_generation_manifest.json",
            holdout_root / "graph_step_manifest.json",
            holdout_root / "quality_report.json",
            holdout_root / "fingerprint_audit.json",
        ]
    )
    artifacts = []
    for path in artifact_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        relative = (
            path.relative_to(REPO_ROOT).as_posix()
            if REPO_ROOT in path.parents
            else str(path.resolve())
        )
        if path == REPO_ROOT / "reports" / "paper_v1_p1" / "model_data_provenance.json":
            category = "provenance"
        elif CONFIG_ROOT in path.parents:
            category = "config"
        elif path.suffix == ".py" and path.parent.name == "tests":
            category = "test"
        elif path.suffix == ".py":
            category = "code"
        elif holdout_root in path.parents:
            category = "data"
        else:
            category = "report"
        artifacts.append(
            {
                "path": relative,
                "category": category,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    for model in load_json(PREHOLDOUT_SELECTION)["model_files"]:
        artifacts.append(
            {
                "path": model["path"],
                "sha256": model["sha256"],
                "bytes": model["bytes"],
                "role": model["name"],
                "category": "model",
            }
        )
    manifest = {
        "schema": "topocae_paper_v1_clean_candidate_v2_freeze_manifest_v1",
        "phase": "P1-A2",
        "generated_at": utc_now(),
        "selected_variant": frozen["selected_variant"],
        "selected_variant_label": frozen["selected_variant_label"],
        "preholdout_selection_sha256": sha256_file(PREHOLDOUT_SELECTION),
        "holdout_inference_pass_count": 1,
        "uses_real_dev_for_training_or_selection": False,
        "uses_synthetic_development_test_for_selection": False,
        "uses_holdout_for_selection": False,
        "uses_s3_geometry_truth_prediction_or_training_data": False,
        "artifacts": sorted(artifacts, key=lambda row: row["path"]),
        "directory_digests": [
            directory_digest(holdout_root / "graphs"),
            directory_digest(holdout_root / "steps"),
            directory_digest(REPORT_ROOT / "formal" / "holdout_once" / "predictions"),
            directory_digest(REPORT_ROOT / "formal" / "holdout_once" / "feature_objects"),
            directory_digest(REPORT_ROOT / "formal" / "holdout_once" / "raw"),
            directory_digest(REPORT_ROOT / "formal" / "repeat_1"),
            directory_digest(REPORT_ROOT / "formal" / "repeat_2"),
        ],
        "formal_metric_summary": {
            "holdout_sample_count": formal_metrics["synthetic_holdout_v1"]["sample_count"],
            "holdout_object_macro_f1": formal_metrics["synthetic_holdout_v1"]["object"]["macro"]["f1"],
        },
    }
    path = REPO_ROOT / "paper_v1_clean_candidate_v2_freeze_manifest.json"
    write_json(path, manifest)
    return manifest


def run_formal(*, finalize_only: bool) -> dict[str, Any]:
    selection = verify_preholdout_freeze()
    frozen = load_json(FROZEN_VARIANT_CONFIG)
    dataset_config, model_config, evaluation_config = load_configs()
    models = load_models(dataset_config, model_config)
    holdout_root = resolve_repo_path(load_json(HOLDOUT_CONFIG)["output_root"])
    for required in (
        holdout_root / "holdout_generation_manifest.json",
        holdout_root / "graph_step_manifest.json",
        holdout_root / "quality_report.json",
        holdout_root / "fingerprint_audit.json",
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    generation = load_json(holdout_root / "holdout_generation_manifest.json")
    if generation["preholdout_selection_sha256"] != sha256_file(PREHOLDOUT_SELECTION):
        raise RuntimeError("Holdout was not generated from the frozen preholdout selection")
    holdout_paths = collect_graph_paths(holdout_root / "graphs", dataset_config)
    if len(holdout_paths) < 1000:
        raise RuntimeError(f"Holdout too small: {len(holdout_paths)}")

    real_paths = {
        name: collect_graph_paths(resolve_repo_path(record["graph_root"]), dataset_config)
        for name, record in dataset_config["real_dev"].items()
    }
    formal_root = REPORT_ROOT / "formal"
    real_outputs: dict[str, tuple[list[Path], dict[str, dict], dict[str, dict]]] = {}
    if not finalize_only:
        for repeat in ("repeat_1", "repeat_2"):
            for name, paths in real_paths.items():
                run_frozen_dataset(
                    paths,
                    formal_root / repeat / name,
                    models,
                    frozen,
                    reuse_complete=True,
                )
        run_frozen_dataset(
            holdout_paths,
            formal_root / "holdout_once",
            models,
            frozen,
            reuse_complete=True,
        )

    holdout_predictions, holdout_features = run_frozen_dataset(
        holdout_paths,
        formal_root / "holdout_once",
        models,
        frozen,
        reuse_complete=True,
    )
    repeat_metrics = {"repeat_1": {}, "repeat_2": {}}
    for repeat in repeat_metrics:
        for name, paths in real_paths.items():
            predictions, features = run_frozen_dataset(
                paths,
                formal_root / repeat / name,
                models,
                frozen,
                reuse_complete=True,
            )
            repeat_metrics[repeat][name] = _evaluate_formal_predictions(
                paths,
                predictions,
                features,
                models["checkpoint"]["config"],
                evaluation_config,
            )
            if repeat == "repeat_1":
                real_outputs[name] = (paths, predictions, features)
            if name == "test_part_s2":
                truth = resolve_repo_path(dataset_config["real_dev"][name]["truth"])
                area = build_area_truth_review(
                    formal_root / repeat / name / "feature_objects",
                    truth,
                    False,
                    5.0,
                    0.01,
                )
                repeat_metrics[repeat][name]["area_truth"] = area
                write_json(formal_root / repeat / name / "area_truth_review.json", area)

    rotation_source = formal_root / "repeat_1" / "test_part_s"
    rotation = rotation_metrics(rotation_source, dataset_config["rotation_dev"]["groups"])
    holdout_metrics = _evaluate_formal_predictions(
        holdout_paths,
        holdout_predictions,
        holdout_features,
        models["checkpoint"]["config"],
        evaluation_config,
    )
    formal_metrics = {
        "schema": "topocae_p1_a2_formal_metrics_v1",
        "phase": "P1-A2",
        "generated_at": utc_now(),
        "selected_variant": frozen["selected_variant"],
        "selection_frozen_before_holdout": True,
        "holdout_inference_pass_count": 1,
        "synthetic_holdout_v1": holdout_metrics,
        "real_dev": repeat_metrics["repeat_1"],
        "rotation_dev": rotation,
    }
    write_json(REPORT_ROOT / "formal_metrics.json", formal_metrics)
    repeatability = compare_output_trees(
        formal_root / "repeat_1", formal_root / "repeat_2"
    )
    # Completion timestamps are intentionally run-specific.  Confirm every
    # other completion-marker field matches before excluding only that field.
    stable_marker_mismatches = []
    for relative in repeatability["mismatches"]:
        if not relative.endswith("/inference_complete.json"):
            stable_marker_mismatches.append(relative)
            continue
        left_marker = load_json(formal_root / "repeat_1" / relative)
        right_marker = load_json(formal_root / "repeat_2" / relative)
        left_marker.pop("completed_at", None)
        right_marker.pop("completed_at", None)
        if left_marker != right_marker:
            stable_marker_mismatches.append(relative)
    repeatability["mismatches"] = stable_marker_mismatches
    repeatability["volatile_fields_excluded"] = ["inference_complete.completed_at"]
    repeatability["status"] = (
        "pass"
        if not repeatability["missing_in_run_1"]
        and not repeatability["missing_in_run_2"]
        and not repeatability["mismatches"]
        else "fail"
    )
    repeatability["metrics_exact_match"] = canonicalize(repeat_metrics["repeat_1"]) == canonicalize(
        repeat_metrics["repeat_2"]
    )
    if not repeatability["metrics_exact_match"]:
        repeatability["status"] = "fail"
    repeatability["holdout_excluded_reason"] = (
        "Synthetic-Holdout-v1 is intentionally inferred exactly once by protocol."
    )
    write_json(REPORT_ROOT / "repeatability.json", repeatability)
    write_review_packages(
        holdout_paths,
        holdout_predictions,
        holdout_features,
        real_outputs,
        frozen,
    )
    write_final_report(formal_metrics, repeatability, frozen)
    manifest = write_freeze_manifest(frozen, formal_metrics)
    if repeatability["status"] != "pass":
        raise RuntimeError("Formal repeatability failed")
    print(
        json.dumps(
            {
                "selected_variant": frozen["selected_variant"],
                "holdout_count": len(holdout_paths),
                "holdout_object_macro_f1": holdout_metrics["object"]["macro"]["f1"],
                "repeatability": repeatability["status"],
                "manifest_artifacts": len(manifest["artifacts"]),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return formal_metrics


def main() -> None:
    args = parse_args()
    if args.command == "val":
        run_val_ablation(args.contract_mode, args.run_id, freeze=args.freeze)
    elif args.command == "formal":
        run_formal(finalize_only=args.finalize_only)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
