from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

from audit_paper_v1_provenance import (
    REPO_ROOT,
    assert_allowed_path,
    load_json,
    load_split_ids,
    resolve_repo_path,
    stable_hash,
    write_json,
)
from build_candidate_classifier_dataset import build_candidate_records_for_graph
from build_rib_boundary_classifier_dataset import build_rib_boundary_records_for_graph
from build_rib_boundary_hardcase_dataset import build_hardcase_records_for_graph
from run_gnn_inference import build_model_from_checkpoint, predict_one_graph
from train_candidate_topology_classifier import train_candidate_classifier
from utils.rib_candidate_features import extract_rib_candidate_features


DEFAULT_DATASET_CONFIG = REPO_ROOT / "configs" / "paper_v1_clean_candidate" / "datasets.json"
DEFAULT_MODEL_CONFIG = REPO_ROOT / "configs" / "paper_v1_clean_candidate" / "models.json"
DEFAULT_EVALUATION_CONFIG = REPO_ROOT / "configs" / "paper_v1_clean_candidate" / "evaluation.json"
DEFAULT_EXCLUSIONS = REPO_ROOT / "configs" / "paper_v1_clean_candidate" / "quality_exclusions.json"
SPLIT_NAMES = ("train", "val", "test")
RIB_POSITIVE_LABEL = "rib_candidate"
RIB_NEGATIVE_LABEL = "not_rib_candidate"
RIB_PROPOSAL_MACROS = {
    "stock",
    "rib",
    "web",
    "fillet",
    "chamfer",
    "cylinder_boss",
    "rectangular_boss",
    "pocket_rect",
    "pocket_circular",
    "slot_u",
    "slot_o",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate formal-split Synthetic-only auxiliary datasets, fit clean classifiers, "
            "and select rib thresholds on Synthetic-Val only."
        )
    )
    parser.add_argument("--dataset-config", type=Path, default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--evaluation-config", type=Path, default=DEFAULT_EVALUATION_CONFIG)
    parser.add_argument("--quality-exclusions", type=Path, default=DEFAULT_EXCLUSIONS)
    parser.add_argument("--force-raw-predictions", action="store_true")
    parser.add_argument("--skip-raw-predictions", action="store_true")
    return parser.parse_args()


def load_effective_splits(dataset_config: dict, exclusion_path: Path) -> dict[str, list[str]]:
    splits = load_split_ids(dataset_config)
    exclusions = load_json(exclusion_path)["exclusions"]
    return {
        split_name: [
            sample_id
            for sample_id in sample_ids
            if sample_id not in exclusions.get(split_name, {})
        ]
        for split_name, sample_ids in splits.items()
    }


def graph_and_prediction_paths(
    sample_id: str,
    graph_root: Path,
    prediction_root: Path,
) -> tuple[Path, Path]:
    return graph_root / f"{sample_id}.json", prediction_root / f"{sample_id}.json"


def generate_raw_predictions(
    *,
    dataset_config: dict,
    checkpoint_path: Path,
    graph_root: Path,
    prediction_root: Path,
    splits: dict[str, list[str]],
    force: bool,
) -> dict:
    checkpoint_path = assert_allowed_path(checkpoint_path, dataset_config)
    graph_root = assert_allowed_path(graph_root, dataset_config)
    prediction_root = assert_allowed_path(prediction_root, dataset_config)
    prediction_root.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    device = torch.device("cpu")
    model = build_model_from_checkpoint(checkpoint, device)
    model.eval()
    written = 0
    reused = 0
    sample_ids = [sample_id for split in SPLIT_NAMES for sample_id in splits[split]]
    with torch.no_grad():
        for index, sample_id in enumerate(sample_ids, start=1):
            graph_path, prediction_path = graph_and_prediction_paths(
                sample_id, graph_root, prediction_root
            )
            graph_path = assert_allowed_path(graph_path, dataset_config)
            prediction_path = assert_allowed_path(prediction_path, dataset_config)
            if prediction_path.is_file() and not force:
                existing = load_json(prediction_path)
                if existing.get("sample_id") == sample_id and isinstance(existing.get("faces"), list):
                    reused += 1
                    continue
            graph = load_json(graph_path)
            prediction = predict_one_graph(
                model=model,
                graph_data=graph,
                graph_path=graph_path,
                checkpoint_config=checkpoint["config"],
                ndigits=6,
                device=device,
                topology_postprocess=False,
                same_instance_threshold=0.5,
            )
            write_json(prediction_path, prediction)
            written += 1
            if index % 500 == 0 or index == len(sample_ids):
                print(f"raw_prediction_progress={index}/{len(sample_ids)}")
    manifest = {
        "schema": "topocae_paper_v1_raw_prediction_manifest_v1",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "sample_count": len(sample_ids),
        "written": written,
        "reused": reused,
        "topology_postprocess": False,
        "auxiliary_classifiers": [],
        "splits_hash": stable_hash(splits),
    }
    write_json(prediction_root / "raw_prediction_manifest.json", manifest)
    return manifest


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def undirected_adjacency(graph: dict) -> dict[int, set[int]]:
    adjacency = {int(node.get("idx", index)): set() for index, node in enumerate(graph.get("nodes", []))}
    for edge in graph.get("edges", []) or []:
        if not isinstance(edge, list) or len(edge) != 2:
            continue
        source, target = int(edge[0]), int(edge[1])
        if source in adjacency and target in adjacency:
            adjacency[source].add(target)
            adjacency[target].add(source)
    return adjacency


def deterministic_rank(sample_id: str, face_idx: int) -> str:
    return hashlib.sha256(f"{sample_id}:{face_idx}".encode("utf-8")).hexdigest()


def build_rib_candidate_records_for_graph(
    *,
    sample_id: str,
    graph: dict,
    prediction: dict,
    graph_path: Path,
    prediction_path: Path,
    max_background_negatives: int = 12,
) -> list[dict]:
    nodes = {int(node.get("idx", index)): node for index, node in enumerate(graph.get("nodes", []))}
    pred_faces = {int(face["idx"]): face for face in prediction.get("faces", [])}
    adjacency = undirected_adjacency(graph)
    true_rib_faces = {
        face_idx for face_idx, node in nodes.items() if node.get("macro_feature") == "rib"
    }
    adjacent_to_true = {
        neighbor for face_idx in true_rib_faces for neighbor in adjacency.get(face_idx, set())
    } - true_rib_faces
    mandatory_negatives = {
        face_idx
        for face_idx, face in pred_faces.items()
        if face_idx not in true_rib_faces and face.get("macro_feature") == "rib"
    } | adjacent_to_true
    background_pool = [
        face_idx
        for face_idx, face in pred_faces.items()
        if face_idx not in true_rib_faces
        and face_idx not in mandatory_negatives
        and face.get("macro_feature") in RIB_PROPOSAL_MACROS
    ]
    background_pool.sort(key=lambda face_idx: deterministic_rank(sample_id, face_idx))
    selected = sorted(true_rib_faces | mandatory_negatives | set(background_pool[:max_background_negatives]))
    records = []
    for face_idx in selected:
        prediction_face = pred_faces.get(face_idx, {})
        is_positive = face_idx in true_rib_faces
        if is_positive:
            source = "synthetic_truth_positive"
            weight = 5.0
        elif face_idx in adjacent_to_true:
            source = "synthetic_adjacent_hard_negative"
            weight = 3.0
        elif prediction_face.get("macro_feature") == "rib":
            source = "synthetic_prediction_hard_negative"
            weight = 3.0
        else:
            source = "synthetic_background_negative"
            weight = 1.0
        records.append(
            {
                "sample_id": sample_id,
                "graph_path": str(graph_path.resolve()),
                "prediction_path": str(prediction_path.resolve()),
                "source": source,
                "label": RIB_POSITIVE_LABEL if is_positive else RIB_NEGATIVE_LABEL,
                "candidate_face_idx": face_idx,
                "sample_weight": weight,
                "features": extract_rib_candidate_features(
                    graph, face_idx, prediction_face=prediction_face
                ),
            }
        )
    return records


def deduplicate_boundary_records(records: list[dict]) -> list[dict]:
    selected: dict[tuple, dict] = {}
    for record in records:
        key = (
            record.get("sample_id"),
            record.get("label"),
            tuple(record.get("seed_face_indices", [])),
            int(record.get("candidate_face_idx", -1)),
            record.get("source"),
        )
        selected[key] = record
    return [selected[key] for key in sorted(selected, key=str)]


def encode_and_write_dataset(
    *,
    output_dir: Path,
    records_by_split: dict[str, list[dict]],
    label_to_id: dict[str, int],
    schema: str,
    source_graph_root: Path,
    truth_source: str,
    splits: dict[str, list[str]],
    exclusion_path: Path,
    extra_metadata: dict[str, Any] | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_records = [record for split in SPLIT_NAMES for record in records_by_split[split]]
    feature_schema = sorted(
        {name for record in all_records for name in record.get("features", {}).keys()}
    )
    id_to_label = {str(value): key for key, value in label_to_id.items()}
    label_counts = {
        split_name: dict(Counter(record["label"] for record in records_by_split[split_name]))
        for split_name in SPLIT_NAMES
    }
    metadata = {
        "schema": schema,
        "record_count": len(all_records),
        "split_counts": {name: len(records_by_split[name]) for name in SPLIT_NAMES},
        "label_counts_by_split": label_counts,
        "feature_schema": feature_schema,
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "paper_clean_provenance": {
            "source_graph_roots": [str(source_graph_root.resolve())],
            "truth_source": truth_source,
            "split": "formal Synthetic-Train/Val/Test; model fit on Train and threshold selection on Val",
            "formal_split_enforced": True,
            "formal_splits_hash": stable_hash(splits),
            "quality_exclusions_path": str(exclusion_path.resolve()),
            "contains_real_data": False,
            "contains_s3_derived_data": False,
        },
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    write_json(output_dir / "metadata.json", metadata)
    for split_name in SPLIT_NAMES:
        output_path = output_dir / f"{split_name}.jsonl"
        with output_path.open("w", encoding="utf-8") as stream:
            for record in records_by_split[split_name]:
                encoded = dict(record)
                encoded["label_id"] = label_to_id[record["label"]]
                encoded["feature_schema"] = feature_schema
                encoded["feature_vector"] = [
                    float(record.get("features", {}).get(name, 0.0)) for name in feature_schema
                ]
                stream.write(json.dumps(encoded, ensure_ascii=False) + "\n")
    return metadata


def prepare_datasets(
    *,
    dataset_config: dict,
    model_config: dict,
    splits: dict[str, list[str]],
    exclusion_path: Path,
) -> dict:
    formal = dataset_config["formal_synthetic"]
    graph_root = assert_allowed_path(resolve_repo_path(formal["graph_root"]), dataset_config)
    prediction_root = assert_allowed_path(
        resolve_repo_path(model_config["training_data"]["raw_gnn_predictions"]), dataset_config
    )
    output_paths = {
        key: assert_allowed_path(resolve_repo_path(value), dataset_config)
        for key, value in model_config["training_data"].items()
        if key != "raw_gnn_predictions"
    }
    legacy_candidate_metadata = load_json(
        REPO_ROOT / "output_generation_v5_1_candidate_topology_dataset" / "metadata.json"
    )
    target_macros = set(legacy_candidate_metadata["target_macros"])
    candidate_by_split: dict[str, list[dict]] = {name: [] for name in SPLIT_NAMES}
    proposer_by_split: dict[str, list[dict]] = {name: [] for name in SPLIT_NAMES}
    boundary_by_split: dict[str, list[dict]] = {name: [] for name in SPLIT_NAMES}

    for split_name in SPLIT_NAMES:
        for index, sample_id in enumerate(splits[split_name], start=1):
            graph_path, prediction_path = graph_and_prediction_paths(
                sample_id, graph_root, prediction_root
            )
            graph = load_json(assert_allowed_path(graph_path, dataset_config))
            prediction = load_json(assert_allowed_path(prediction_path, dataset_config))
            candidate_by_split[split_name].extend(
                build_candidate_records_for_graph(
                    sample_id=sample_id,
                    graph_data=graph,
                    prediction_data=prediction,
                    graph_path=graph_path,
                    prediction_path=prediction_path,
                    target_macros=target_macros,
                    min_iou=0.6,
                    min_both_coverage=0.75,
                    same_instance_threshold=0.5,
                )
            )
            proposer_by_split[split_name].extend(
                build_rib_candidate_records_for_graph(
                    sample_id=sample_id,
                    graph=graph,
                    prediction=prediction,
                    graph_path=graph_path,
                    prediction_path=prediction_path,
                )
            )
            boundary_records = build_rib_boundary_records_for_graph(
                sample_id=sample_id, graph_data=graph, graph_path=graph_path
            )
            boundary_records.extend(
                build_hardcase_records_for_graph(
                    sample_id=sample_id,
                    graph_data=graph,
                    graph_path=graph_path,
                    max_contaminants=2,
                )
            )
            boundary_by_split[split_name].extend(
                deduplicate_boundary_records(boundary_records)
            )
            if index % 500 == 0 or index == len(splits[split_name]):
                print(f"dataset_prepare_progress={split_name}:{index}/{len(splits[split_name])}")

    candidate_labels = dict(legacy_candidate_metadata["label_to_id"])
    candidate_meta = encode_and_write_dataset(
        output_dir=output_paths["candidate_classifier"],
        records_by_split=candidate_by_split,
        label_to_id={str(key): int(value) for key, value in candidate_labels.items()},
        schema="candidate_topology_classifier_dataset_paper_clean_v1",
        source_graph_root=graph_root,
        truth_source="programmatic Synthetic feature sets plus frozen raw-GNN proposals",
        splits=splits,
        exclusion_path=exclusion_path,
        extra_metadata={"target_macros": sorted(target_macros)},
    )
    proposer_meta = encode_and_write_dataset(
        output_dir=output_paths["rib_candidate_proposer"],
        records_by_split=proposer_by_split,
        label_to_id={RIB_NEGATIVE_LABEL: 0, RIB_POSITIVE_LABEL: 1},
        schema="rib_candidate_proposer_dataset_paper_clean_v1",
        source_graph_root=graph_root,
        truth_source="programmatic Synthetic rib face labels and frozen raw-GNN predictions",
        splits=splits,
        exclusion_path=exclusion_path,
        extra_metadata={"max_background_negatives_per_sample": 12},
    )
    boundary_meta = encode_and_write_dataset(
        output_dir=output_paths["rib_boundary_classifier"],
        records_by_split=boundary_by_split,
        label_to_id={"not_rib_member": 0, "rib_member": 1},
        schema="rib_boundary_classifier_dataset_paper_clean_v1",
        source_graph_root=graph_root,
        truth_source="programmatic Synthetic rib instances plus synthetic fragment/contamination hard cases",
        splits=splits,
        exclusion_path=exclusion_path,
        extra_metadata={"max_contaminants": 2},
    )
    return {
        "candidate_classifier": candidate_meta,
        "rib_candidate_proposer": proposer_meta,
        "rib_boundary_classifier": boundary_meta,
    }


def train_models(model_config: dict, dataset_config: dict) -> dict[str, dict]:
    training = model_config["training_data"]
    model_paths = model_config["clean_components"]
    specifications = {
        "candidate_classifier": {
            "run_name": Path(model_paths["candidate_classifier"]).parent.name,
            "seed": 20260625,
            "n_estimators": 300,
            "max_depth": 18,
            "hard_source_weight": 4.0,
        },
        "rib_candidate_proposer": {
            "run_name": Path(model_paths["rib_candidate_proposer"]).parent.name,
            "seed": 20260716,
            "n_estimators": 500,
            "max_depth": 18,
            "hard_source_weight": 1.0,
        },
        "rib_boundary_classifier": {
            "run_name": Path(model_paths["rib_boundary_classifier"]).parent.name,
            "seed": 20260717,
            "n_estimators": 500,
            "max_depth": 20,
            "hard_source_weight": 1.0,
        },
    }
    results = {}
    for component_name, specification in specifications.items():
        dataset_dir = assert_allowed_path(resolve_repo_path(training[component_name]), dataset_config)
        model_path = assert_allowed_path(resolve_repo_path(model_paths[component_name]), dataset_config)
        result = train_candidate_classifier(
            dataset_dir=dataset_dir,
            output_dir=model_path.parent.parent,
            run_name=specification["run_name"],
            model_kind="random_forest",
            seed=specification["seed"],
            n_estimators=specification["n_estimators"],
            max_depth=specification["max_depth"],
            hard_source_weight=specification["hard_source_weight"],
        )
        results[component_name] = result
        print(
            f"trained={component_name} val_macro_f1="
            f"{result['metrics']['val']['macro_f1']:.6f} test_macro_f1="
            f"{result['metrics']['test']['macro_f1']:.6f}"
        )
    return results


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def positive_probabilities(bundle: dict, records: list[dict], positive_id: int) -> np.ndarray:
    estimator = bundle["estimator"]
    features = np.asarray([record["feature_vector"] for record in records], dtype=np.float32)
    probabilities = estimator.predict_proba(features)
    classes = list(getattr(estimator, "classes_", []))
    if positive_id not in classes:
        classifier = estimator.named_steps.get("classifier") if hasattr(estimator, "named_steps") else estimator
        classes = list(getattr(classifier, "classes_", []))
    if positive_id not in classes:
        raise ValueError(f"Positive class {positive_id} not found in estimator classes {classes}")
    return probabilities[:, classes.index(positive_id)]


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = probabilities >= threshold
    positives = labels == 1
    tp = int(np.sum(predictions & positives))
    fp = int(np.sum(predictions & ~positives))
    fn = int(np.sum(~predictions & positives))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def select_threshold(
    *,
    model_path: Path,
    dataset_dir: Path,
    evaluation_config: dict,
    positive_label: str,
) -> dict:
    bundle = joblib.load(model_path)
    label_to_id = bundle["metadata"]["label_to_id"]
    positive_id = int(label_to_id[positive_label])
    selection = evaluation_config["threshold_selection"]
    minimum = float(selection["grid_min"])
    maximum = float(selection["grid_max"])
    step = float(selection["grid_step"])
    grid = np.arange(minimum, maximum + step / 2.0, step)
    val_records = load_jsonl(dataset_dir / "val.jsonl")
    test_records = load_jsonl(dataset_dir / "test.jsonl")
    val_labels = np.asarray([1 if int(record["label_id"]) == positive_id else 0 for record in val_records])
    test_labels = np.asarray([1 if int(record["label_id"]) == positive_id else 0 for record in test_records])
    val_probabilities = positive_probabilities(bundle, val_records, positive_id)
    test_probabilities = positive_probabilities(bundle, test_records, positive_id)
    curve = [binary_metrics(val_labels, val_probabilities, float(threshold)) for threshold in grid]
    best = max(curve, key=lambda item: (item["f1"], item["precision"], item["threshold"]))
    return {
        "selection_split": "Synthetic-Val",
        "objective": selection["objective"],
        "positive_label": positive_label,
        "selected_threshold": best["threshold"],
        "selected_val_metrics": best,
        "test_metrics_at_selected_threshold": binary_metrics(
            test_labels, test_probabilities, best["threshold"]
        ),
        "val_curve": curve,
    }


def select_and_write_thresholds(
    *,
    dataset_config: dict,
    model_config: dict,
    evaluation_config: dict,
) -> dict:
    components = model_config["clean_components"]
    training = model_config["training_data"]
    proposer = select_threshold(
        model_path=assert_allowed_path(resolve_repo_path(components["rib_candidate_proposer"]), dataset_config),
        dataset_dir=assert_allowed_path(resolve_repo_path(training["rib_candidate_proposer"]), dataset_config),
        evaluation_config=evaluation_config,
        positive_label=RIB_POSITIVE_LABEL,
    )
    boundary = select_threshold(
        model_path=assert_allowed_path(resolve_repo_path(components["rib_boundary_classifier"]), dataset_config),
        dataset_dir=assert_allowed_path(resolve_repo_path(training["rib_boundary_classifier"]), dataset_config),
        evaluation_config=evaluation_config,
        positive_label="rib_member",
    )
    payload = {
        "schema": "topocae_paper_v1_clean_selected_thresholds_v1",
        "phase": "P1-A1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidate_classifier_min_confidence": 0.5,
        "rib_candidate_proposer": proposer,
        "rib_boundary_classifier": boundary,
        "real_dev_used_for_selection": False,
        "s3_used_for_selection": False,
    }
    output_path = REPO_ROOT / "configs" / "paper_v1_clean_candidate" / "selected_thresholds.json"
    write_json(output_path, payload)
    write_json(REPO_ROOT / "reports" / "paper_v1_p1" / "threshold_selection.json", payload)
    return payload


def main() -> None:
    args = parse_args()
    dataset_config = load_json(args.dataset_config.resolve())
    model_config = load_json(args.model_config.resolve())
    evaluation_config = load_json(args.evaluation_config.resolve())
    splits = load_effective_splits(dataset_config, args.quality_exclusions.resolve())
    formal = dataset_config["formal_synthetic"]
    graph_root = resolve_repo_path(formal["graph_root"])
    checkpoint_path = resolve_repo_path(model_config["frozen_gnn"]["path"])
    prediction_root = resolve_repo_path(model_config["training_data"]["raw_gnn_predictions"])

    raw_manifest = None
    if not args.skip_raw_predictions:
        raw_manifest = generate_raw_predictions(
            dataset_config=dataset_config,
            checkpoint_path=checkpoint_path,
            graph_root=graph_root,
            prediction_root=prediction_root,
            splits=splits,
            force=args.force_raw_predictions,
        )
    dataset_metadata = prepare_datasets(
        dataset_config=dataset_config,
        model_config=model_config,
        splits=splits,
        exclusion_path=args.quality_exclusions.resolve(),
    )
    training_results = train_models(model_config, dataset_config)
    thresholds = select_and_write_thresholds(
        dataset_config=dataset_config,
        model_config=model_config,
        evaluation_config=evaluation_config,
    )
    summary = {
        "schema": "topocae_paper_v1_clean_auxiliary_preparation_v1",
        "phase": "P1-A1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "effective_split_counts": {name: len(values) for name, values in splits.items()},
        "raw_prediction_manifest": raw_manifest,
        "dataset_metadata": dataset_metadata,
        "training_metrics": {
            name: result["metrics"] for name, result in training_results.items()
        },
        "selected_thresholds": {
            "rib_candidate_proposer": thresholds["rib_candidate_proposer"]["selected_threshold"],
            "rib_boundary_classifier": thresholds["rib_boundary_classifier"]["selected_threshold"],
        },
    }
    output_path = REPO_ROOT / "reports" / "paper_v1_p1" / "clean_auxiliary_preparation.json"
    write_json(output_path, summary)
    print(json.dumps({"output": str(output_path.resolve()), **summary["effective_split_counts"], **summary["selected_thresholds"]}, indent=2))


if __name__ == "__main__":
    main()
