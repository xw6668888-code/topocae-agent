from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import joblib
import torch
import yaml

import evaluate_paper_v1_recognition as legacy_metrics
from audit_paper_v1_provenance import (
    REPO_ROOT,
    assert_allowed_path,
    rotation_invariant_graph_signature,
)
from evaluate_paper_v1_recognition_v2 import fast_bootstrap_object_metrics
from evaluate_paper_v1_recognition_v5 import normalize_output_contract
from prediction_to_feature_objects import build_feature_objects
from run_gnn_inference import build_model_from_checkpoint, predict_one_graph
from utils.candidate_classifier_postprocess import (
    REJECT_LABEL,
    build_inference_candidate_proposals,
    classify_candidate_proposals,
    target_macros_from_feature_schema,
)
from utils.rib_boundary_classifier_postprocess import (
    apply_rib_boundary_classifier_postprocess,
    load_rib_boundary_classifier_bundle,
)
from utils.rib_candidate_proposer_postprocess import (
    apply_rib_candidate_proposer_postprocess,
    load_rib_candidate_proposer_bundle,
)
from utils.topology_postprocess import (
    apply_topology_postprocess,
    connected_components_for_prediction,
    demote_faces_to_stock,
    remap_faces_to_macro,
)
from utils.topology_rules import validate_feature_topology


CONFIG_ROOT = REPO_ROOT / "configs" / "paper_v1_clean_candidate_v2"
REPORT_ROOT = REPO_ROOT / "reports" / "paper_v1_p1_a2"
DATASET_CONFIG = CONFIG_ROOT / "datasets.json"
MODEL_CONFIG = CONFIG_ROOT / "models.json"
EVALUATION_CONFIG = CONFIG_ROOT / "evaluation.json"
HOLDOUT_CONFIG = CONFIG_ROOT / "holdout_generation.json"
PREHOLDOUT_SELECTION = REPORT_ROOT / "preholdout_selection.json"
PREHOLDOUT_SIDECAR = REPORT_ROOT / "preholdout_selection.sha256.json"
FROZEN_VARIANT_CONFIG = CONFIG_ROOT / "frozen_variant.json"

TARGET_CLASSES = (
    "stepped_hole_round",
    "through_hole_round",
    "blind_hole_round",
    "counterbore_hole",
    "countersink_hole",
    "obround_hole",
    "slot_u",
    "slot_v",
    "slot_o",
    "slot_rect",
    "pocket_rect",
    "pocket_circular",
    "cylinder_boss",
    "rectangular_boss",
    "rib",
    "web",
    "fillet",
    "chamfer",
)
VARIANT_NAMES = ("A0", "A1", "A2", "A3", "A4")
VARIANT_LABELS = {
    "A0": "Raw-GNN",
    "A1": "Raw-GNN + topology postprocess",
    "A2": "Raw-GNN + clean candidate classifier",
    "A3": "Raw-GNN + clean candidate classifier + topology postprocess",
    "A4": "Val-best basic + clean rib proposer/boundary",
}
OUTPUT_LOCATION_KEYS = {
    "source_prediction_path",
    "prediction_path",
    "feature_objects_path",
    "output_path",
    "output_root",
    "run_id",
    "generated_at",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = normalize_output_contract(payload)
    path.write_text(
        json.dumps(normalized, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def config_bundle_hash(paths: Iterable[Path]) -> tuple[str, list[dict[str, Any]]]:
    rows = [
        {
            "path": path.relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(paths)
    ]
    return stable_hash(rows), rows


def load_configs() -> tuple[dict, dict, dict]:
    return load_json(DATASET_CONFIG), load_json(MODEL_CONFIG), load_json(EVALUATION_CONFIG)


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def load_effective_val_ids(dataset_config: dict) -> list[str]:
    formal = dataset_config["formal_synthetic"]
    val_payload = load_json(resolve_repo_path(formal["splits"]["val"]))
    val_ids = val_payload if isinstance(val_payload, list) else val_payload["samples"]
    exclusions = load_json(resolve_repo_path(formal["quality_exclusions"]))["exclusions"]
    return [sample_id for sample_id in val_ids if sample_id not in exclusions.get("val", {})]


def formal_graph_paths(dataset_config: dict, sample_ids: Iterable[str]) -> list[Path]:
    root = assert_allowed_path(
        resolve_repo_path(dataset_config["formal_synthetic"]["graph_root"]),
        dataset_config,
    )
    paths = [assert_allowed_path(root / f"{sample_id}.json", dataset_config) for sample_id in sample_ids]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing formal graph files: {missing[:5]}")
    return paths


def collect_graph_paths(root: Path, dataset_config: dict) -> list[Path]:
    safe_root = assert_allowed_path(root, dataset_config)
    paths = []
    for path in sorted(safe_root.glob("*.json")):
        payload = load_json(path)
        if isinstance(payload.get("nodes"), list) and isinstance(payload.get("edges"), list):
            paths.append(path)
    return paths


def load_models(dataset_config: dict, model_config: dict) -> dict[str, Any]:
    checkpoint_path = assert_allowed_path(
        resolve_repo_path(model_config["frozen_gnn"]["path"]), dataset_config
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    device = torch.device("cpu")
    model = build_model_from_checkpoint(checkpoint, device)
    model.eval()
    clean = model_config["clean_components"]
    candidate_path = assert_allowed_path(resolve_repo_path(clean["candidate_classifier"]), dataset_config)
    proposer_path = assert_allowed_path(resolve_repo_path(clean["rib_candidate_proposer"]), dataset_config)
    boundary_path = assert_allowed_path(resolve_repo_path(clean["rib_boundary_classifier"]), dataset_config)
    return {
        "checkpoint": checkpoint,
        "checkpoint_path": checkpoint_path,
        "model": model,
        "device": device,
        "candidate": joblib.load(candidate_path),
        "candidate_path": candidate_path,
        "proposer": load_rib_candidate_proposer_bundle(proposer_path),
        "proposer_path": proposer_path,
        "boundary": load_rib_boundary_classifier_bundle(boundary_path),
        "boundary_path": boundary_path,
        "thresholds": model_config["thresholds"],
    }


def graph_manifest_hash(graph_paths: list[Path]) -> str:
    return stable_hash([(path.stem, sha256_file(path)) for path in graph_paths])


def infer_raw_predictions(
    graph_paths: list[Path],
    models: dict[str, Any],
    output_dir: Path,
    *,
    reuse: bool,
) -> dict[str, dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    expected = {
        "checkpoint_sha256": sha256_file(models["checkpoint_path"]),
        "graph_manifest_sha256": graph_manifest_hash(graph_paths),
        "sample_count": len(graph_paths),
    }
    if reuse and manifest_path.is_file():
        manifest = load_json(manifest_path)
        if all(manifest.get(key) == value for key, value in expected.items()):
            cached = {path.stem: load_json(output_dir / f"{path.stem}.json") for path in graph_paths}
            if len(cached) == len(graph_paths):
                return cached

    predictions: dict[str, dict] = {}
    with torch.no_grad():
        for index, graph_path in enumerate(graph_paths, start=1):
            graph = load_json(graph_path)
            prediction = predict_one_graph(
                model=models["model"],
                graph_data=graph,
                graph_path=graph_path,
                checkpoint_config=models["checkpoint"]["config"],
                ndigits=6,
                device=models["device"],
                topology_postprocess=False,
                same_instance_threshold=float(models["thresholds"]["same_instance"]),
            )
            predictions[graph_path.stem] = prediction
            write_json(output_dir / f"{graph_path.stem}.json", prediction)
            if index % 100 == 0 or index == len(graph_paths):
                print(f"raw_inference={index}/{len(graph_paths)} output={output_dir}")
    write_json(
        manifest_path,
        {
            "schema": "topocae_p1_a2_raw_cache_v1",
            **expected,
            "samples": [path.stem for path in graph_paths],
        },
    )
    return predictions


def semantic_parent_maps() -> tuple[dict[str, str], dict[int, str]]:
    ontology = yaml.safe_load(
        (REPO_ROOT / "face_semantic_subtype_ontology.yaml").read_text(encoding="utf-8")
    )
    entries = ontology["layers"]["face_semantic_subtype"]["labels"]
    by_key = {str(item["key"]): str(item["parent_macro_feature"]) for item in entries}
    by_id = {int(item["id"]): str(item["parent_macro_feature"]) for item in entries}
    return by_key, by_id


SEMANTIC_PARENT_BY_KEY, SEMANTIC_PARENT_BY_ID = semantic_parent_maps()


def face_semantic_compatible(face: dict) -> bool:
    parent = SEMANTIC_PARENT_BY_KEY.get(str(face.get("semantic_subtype")))
    if parent is None:
        parent = SEMANTIC_PARENT_BY_ID.get(int(face.get("semantic_subtype_id", -1)))
    return parent == str(face.get("macro_feature"))


def semantic_compatibility_report(predictions: dict[str, dict]) -> dict[str, Any]:
    total = compatible = 0
    by_macro: dict[str, Counter] = defaultdict(Counter)
    examples = []
    for sample_id, prediction in sorted(predictions.items()):
        for face in prediction.get("faces", []):
            total += 1
            macro = str(face.get("macro_feature"))
            if face_semantic_compatible(face):
                compatible += 1
                by_macro[macro]["compatible"] += 1
            else:
                by_macro[macro]["incompatible"] += 1
                if len(examples) < 50:
                    examples.append(
                        {
                            "sample_id": sample_id,
                            "face_idx": int(face["idx"]),
                            "macro_feature": macro,
                            "semantic_subtype": face.get("semantic_subtype"),
                        }
                    )
    return {
        "face_count": total,
        "compatible_face_count": compatible,
        "incompatible_face_count": total - compatible,
        "compatibility": legacy_metrics.safe_divide(compatible, total),
        "by_macro": {key: dict(value) for key, value in sorted(by_macro.items())},
        "examples": examples,
    }


def _copy_face_contract_fields(target: dict, source: dict) -> None:
    for key in (
        "macro_feature_id",
        "macro_feature",
        "semantic_subtype_id",
        "semantic_subtype",
        "macro_confidence",
        "semantic_confidence",
    ):
        if key in source:
            target[key] = deepcopy(source[key])


def apply_candidate_contract_safe(
    prediction: dict,
    graph: dict,
    bundle: dict,
    *,
    min_confidence: float,
    same_instance_threshold: float,
) -> dict:
    corrected = deepcopy(prediction)
    metadata = bundle["metadata"]
    feature_schema = list(metadata["feature_schema"])
    id_to_label = {int(key): value for key, value in metadata["id_to_label"].items()}
    target_macros = target_macros_from_feature_schema(feature_schema)
    proposals = build_inference_candidate_proposals(
        graph,
        corrected,
        target_macros,
        same_instance_threshold=same_instance_threshold,
    )
    classified = classify_candidate_proposals(
        graph,
        corrected,
        proposals,
        estimator=bundle["estimator"],
        feature_schema=feature_schema,
        id_to_label=id_to_label,
    )
    faces = corrected["faces"]
    face_by_idx = {int(face["idx"]): face for face in faces}
    records = []
    occupied: set[int] = set()
    for record in sorted(classified, key=lambda item: float(item["confidence"]), reverse=True):
        indices = [int(value) for value in record["face_indices"]]
        action = "skipped_low_confidence"
        if float(record["confidence"]) < min_confidence:
            pass
        elif record["predicted_label"] == REJECT_LABEL:
            action = "rejected_proposal_preserved"
        elif set(indices) & occupied:
            action = "skipped_overlap"
        else:
            target_macro = str(record["predicted_label"])
            current = [face_by_idx[idx] for idx in indices if idx in face_by_idx]
            protected = [
                face
                for face in current
                if str(face.get("macro_feature")) not in {"stock", target_macro}
            ]
            if protected:
                action = "skipped_nonstock_overwrite_without_hard_topology"
            else:
                snapshots = {int(face["idx"]): deepcopy(face) for face in current}
                remap_faces_to_macro(
                    faces,
                    indices,
                    target_macro,
                    graph,
                    ["p1_a2_candidate_contract_remap"],
                )
                changed = [face_by_idx[idx] for idx in indices if idx in face_by_idx]
                if not all(face_semantic_compatible(face) for face in changed):
                    for idx, snapshot in snapshots.items():
                        face_by_idx[idx].clear()
                        face_by_idx[idx].update(snapshot)
                    action = "skipped_semantic_macro_incompatibility"
                else:
                    occupied.update(indices)
                    action = "accepted_remap"
        records.append(
            {
                "action": action,
                "source": record["source"],
                "proposal_macro": record["proposal_macro"],
                "predicted_label": record["predicted_label"],
                "confidence": float(record["confidence"]),
                "face_indices": indices,
                "face_count": len(indices),
            }
        )
    corrected["candidate_classifier_postprocess"] = {
        "enabled": True,
        "contract_version": "p1_a2_safe_v1",
        "candidate_count": len(classified),
        "accepted_count": sum(record["action"] == "accepted_remap" for record in records),
        "preserved_reject_count": sum(
            record["action"] == "rejected_proposal_preserved" for record in records
        ),
        "min_confidence": min_confidence,
        "records": records,
    }
    return corrected


def apply_topology_contract_safe(
    prediction: dict,
    graph: dict,
    *,
    same_instance_threshold: float,
) -> dict:
    before = deepcopy(prediction)
    components = connected_components_for_prediction(
        before.get("faces", []), graph, same_instance_threshold=same_instance_threshold
    )
    protected: dict[int, dict] = {}
    for component in components:
        macro = str(component["macro_feature"])
        indices = [int(value) for value in component["face_indices"]]
        try:
            valid = bool(validate_feature_topology(macro, indices, graph)["valid"])
        except (KeyError, TypeError, ValueError):
            valid = False
        if valid:
            source_faces = {int(face["idx"]): face for face in before.get("faces", [])}
            for idx in indices:
                protected[idx] = source_faces[idx]

    corrected = apply_topology_postprocess(
        before, graph, same_instance_threshold=same_instance_threshold
    )
    restored_valid = []
    restored_compatibility = []
    before_faces = {int(face["idx"]): face for face in before.get("faces", [])}
    for face in corrected.get("faces", []):
        idx = int(face["idx"])
        source = before_faces[idx]
        if idx in protected and face.get("macro_feature") != source.get("macro_feature"):
            _copy_face_contract_fields(face, source)
            restored_valid.append(idx)
        if not face_semantic_compatible(face) and face_semantic_compatible(source):
            _copy_face_contract_fields(face, source)
            restored_compatibility.append(idx)
    corrected.setdefault("topology_postprocess", {})["contract"] = {
        "version": "p1_a2_safe_v1",
        "rule": "topology-valid input components cannot be macro-overwritten; no new semantic/macro incompatibility",
        "restored_topology_valid_face_indices": sorted(set(restored_valid)),
        "restored_semantic_compatibility_face_indices": sorted(set(restored_compatibility)),
    }
    return corrected


def apply_candidate_stage(prediction: dict, graph: dict, models: dict, contract_mode: str) -> dict:
    threshold = float(models["thresholds"]["candidate_classifier"])
    same = float(models["thresholds"]["same_instance"])
    if contract_mode == "safe_v1":
        return apply_candidate_contract_safe(
            prediction,
            graph,
            models["candidate"],
            min_confidence=threshold,
            same_instance_threshold=same,
        )
    from utils.candidate_classifier_postprocess import apply_candidate_classifier_postprocess

    return apply_candidate_classifier_postprocess(
        prediction,
        graph,
        models["candidate"],
        min_confidence=threshold,
        same_instance_threshold=same,
    )


def apply_topology_stage(prediction: dict, graph: dict, models: dict, contract_mode: str) -> dict:
    same = float(models["thresholds"]["same_instance"])
    graph_for_postprocess = {
        **graph,
        "prediction_edge_same_instance": prediction.get("edge_same_instance", []),
    }
    if contract_mode == "safe_v1":
        return apply_topology_contract_safe(
            prediction, graph_for_postprocess, same_instance_threshold=same
        )
    return apply_topology_postprocess(
        prediction, graph_for_postprocess, same_instance_threshold=same
    )


def apply_rib_stage(prediction: dict, graph: dict, models: dict) -> dict:
    result = apply_rib_candidate_proposer_postprocess(
        prediction,
        graph,
        models["proposer"],
        min_confidence=float(models["thresholds"]["rib_candidate_proposer"]),
    )
    return apply_rib_boundary_classifier_postprocess(
        result,
        graph,
        models["boundary"],
        min_confidence=float(models["thresholds"]["rib_boundary_classifier"]),
    )


def basic_variant_predictions(
    graph_paths: list[Path], raw: dict[str, dict], models: dict, contract_mode: str
) -> dict[str, dict[str, dict]]:
    variants = {name: {} for name in ("A0", "A1", "A2", "A3")}
    for graph_path in graph_paths:
        graph = load_json(graph_path)
        sample_id = graph_path.stem
        a0 = deepcopy(raw[sample_id])
        a1 = apply_topology_stage(a0, graph, models, contract_mode)
        a2 = apply_candidate_stage(a0, graph, models, contract_mode)
        a3 = apply_topology_stage(a2, graph, models, contract_mode)
        for variant, prediction in (("A0", a0), ("A1", a1), ("A2", a2), ("A3", a3)):
            variants[variant][sample_id] = prediction
    return variants


def add_a4_predictions(
    variants: dict[str, dict[str, dict]],
    base_variant: str,
    graph_paths: list[Path],
    models: dict,
) -> None:
    variants["A4"] = {}
    for graph_path in graph_paths:
        graph = load_json(graph_path)
        sample_id = graph_path.stem
        variants["A4"][sample_id] = apply_rib_stage(
            variants[base_variant][sample_id], graph, models
        )


def feature_documents(
    graph_paths: list[Path], predictions: dict[str, dict]
) -> dict[str, dict]:
    documents = {}
    for graph_path in graph_paths:
        graph = load_json(graph_path)
        documents[graph_path.stem] = build_feature_objects(
            predictions[graph_path.stem], graph, ndigits=6
        )
    return documents


def object_metrics_with_macro(reports: list[dict]) -> dict[str, Any]:
    micro = legacy_metrics.object_metrics_from_reports(reports)
    per_class = micro.get("per_class", {})
    truth_classes = [
        metrics for metrics in per_class.values() if metrics["tp"] + metrics["fn"] > 0
    ]
    macro = {
        name: (
            sum(float(metrics[name]) for metrics in truth_classes) / len(truth_classes)
            if truth_classes
            else None
        )
        for name in ("precision", "recall", "f1")
    }
    enriched = {}
    for class_name, metrics in sorted(per_class.items()):
        enriched[class_name] = {
            **metrics,
            "support": int(metrics["tp"] + metrics["fn"]),
        }
    return {
        "status": micro["status"],
        "coverage": micro["coverage"],
        "micro": {name: micro[name] for name in ("precision", "recall", "f1")},
        "macro": macro,
        "exact_instance_ratio": micro["exact_instance_ratio"],
        "instance_iou": micro["instance_iou"],
        "truth_count": micro["truth_count"],
        "prediction_count": micro["prediction_count"],
        "per_class": enriched,
    }


def evaluate_predictions(
    graph_paths: list[Path],
    predictions: dict[str, dict],
    features: dict[str, dict],
    checkpoint_config: dict,
    evaluation_config: dict,
) -> dict[str, Any]:
    macro_pairs: list[tuple[int, int]] = []
    semantic_pairs: list[tuple[int, int]] = []
    edge_reports = []
    object_reports = []
    for graph_path in graph_paths:
        graph = load_json(graph_path)
        sample_id = graph_path.stem
        current_macro, current_semantic = legacy_metrics.face_and_semantic_pairs(
            predictions[sample_id], graph
        )
        macro_pairs.extend(current_macro)
        semantic_pairs.extend(current_semantic)
        edge_reports.append(legacy_metrics.edge_counts(predictions[sample_id], graph))
        object_reports.append(
            legacy_metrics.object_match_report(
                features[sample_id], graph, float(evaluation_config["object_match_iou"])
            )
        )
    labels = checkpoint_config["label_spaces"]
    bootstrap = evaluation_config["bootstrap"]
    return {
        "sample_count": len(graph_paths),
        "face": legacy_metrics.classification_report(
            macro_pairs, labels["macro_feature_id_to_key"]
        ),
        "semantic": legacy_metrics.classification_report(
            semantic_pairs, labels["semantic_subtype_id_to_key"]
        ),
        "object": object_metrics_with_macro(object_reports),
        "edge": legacy_metrics.aggregate_edge(edge_reports),
        "bootstrap": fast_bootstrap_object_metrics(
            object_reports,
            replicates=int(bootstrap["replicates"]),
            confidence=float(bootstrap["confidence"]),
            seed=int(bootstrap["seed"]),
        ),
        "semantic_macro_compatibility": semantic_compatibility_report(predictions),
    }


def write_variant_outputs(
    output_root: Path,
    variant: str,
    graph_paths: list[Path],
    predictions: dict[str, dict],
    features: dict[str, dict],
) -> None:
    prediction_root = output_root / variant / "predictions"
    feature_root = output_root / variant / "feature_objects"
    for graph_path in graph_paths:
        sample_id = graph_path.stem
        write_json(prediction_root / f"{sample_id}.json", predictions[sample_id])
        write_json(feature_root / f"{sample_id}.json", features[sample_id])
    write_json(
        output_root / variant / "manifest.json",
        {
            "variant": variant,
            "label": VARIANT_LABELS[variant],
            "sample_count": len(graph_paths),
            "graph_manifest_sha256": graph_manifest_hash(graph_paths),
        },
    )


def rotation_scalar(rotation: dict) -> float:
    values = [
        rotation.get("class_consistency"),
        rotation.get("object_correspondence_consistency"),
        rotation.get("face_role_consistency"),
    ]
    return min(float(value) for value in values if value is not None)


def selection_view(metrics: dict) -> dict[str, Any]:
    val = metrics["synthetic_val"]
    rotation = metrics["synthetic_rotation_val"]
    return {
        "object_macro_f1": val["object"]["macro"]["f1"],
        "object_macro_precision": val["object"]["macro"]["precision"],
        "object_macro_recall": val["object"]["macro"]["recall"],
        "object_micro_f1": val["object"]["micro"]["f1"],
        "face_macro_f1": val["face"]["macro_f1"],
        "semantic_macro_f1": val["semantic"]["macro_f1"],
        "exact_instance_ratio": val["object"]["exact_instance_ratio"],
        "instance_iou": val["object"]["instance_iou"],
        "rotation_consistency": rotation_scalar(rotation),
        "rotation": rotation,
        "per_class_object": val["object"]["per_class"],
    }


def variant_stage_count(variant: str, a4_base: str | None = None) -> int:
    if variant == "A0":
        return 1
    if variant in {"A1", "A2"}:
        return 2
    if variant == "A3":
        return 3
    if variant == "A4":
        if a4_base is None:
            raise ValueError("A4 stage count requires its base variant")
        return variant_stage_count(a4_base) + 2
    raise KeyError(variant)


def evaluate_selection_constraints(
    variant: str,
    views: dict[str, dict],
    protocol: dict,
) -> dict[str, Any]:
    baseline = views["A0"]
    current = views[variant]
    face_drop = baseline["face_macro_f1"] - current["face_macro_f1"]
    semantic_drop = baseline["semantic_macro_f1"] - current["semantic_macro_f1"]
    rotation_drop = baseline["rotation_consistency"] - current["rotation_consistency"]
    zero_recall = []
    for class_name, base_class in baseline["per_class_object"].items():
        current_class = current["per_class_object"].get(
            class_name, {"recall": 0.0, "support": base_class["support"]}
        )
        if (
            int(base_class["support"]) >= int(protocol["no_nonzero_to_zero_recall_min_support"])
            and float(base_class["recall"]) > 0.0
            and float(current_class["recall"]) == 0.0
        ):
            zero_recall.append(class_name)
    checks = {
        "face_macro_f1_drop_ok": face_drop
        <= float(protocol["face_macro_f1_max_drop_from_a0"]),
        "semantic_macro_f1_drop_ok": semantic_drop
        <= float(protocol["semantic_macro_f1_max_drop_from_a0"]),
        "no_supported_nonzero_to_zero_recall": not zero_recall,
        "synthetic_rotation_drop_ok": rotation_drop
        <= float(protocol["synthetic_rotation_consistency_max_drop_from_a0"]),
    }
    return {
        "eligible": all(checks.values()),
        "checks": checks,
        "face_macro_f1_drop": face_drop,
        "semantic_macro_f1_drop": semantic_drop,
        "rotation_consistency_drop": rotation_drop,
        "nonzero_to_zero_recall_classes": zero_recall,
    }


def select_variant(
    variants: Iterable[str],
    metrics: dict[str, dict],
    evaluation_config: dict,
    *,
    a4_base: str | None = None,
) -> dict[str, Any]:
    variant_list = list(variants)
    views = {variant: selection_view(metrics[variant]) for variant in variant_list}
    if "A0" not in views:
        views["A0"] = selection_view(metrics["A0"])
    protocol = evaluation_config["selection_protocol"]
    constraints = {
        variant: evaluate_selection_constraints(variant, views, protocol)
        for variant in variant_list
    }
    eligible = [variant for variant in variant_list if constraints[variant]["eligible"]]
    if not eligible:
        selected = "A0"
    else:
        selected = max(
            eligible,
            key=lambda variant: (
                float(views[variant]["object_macro_f1"]),
                float(views[variant]["exact_instance_ratio"]),
                float(views[variant]["instance_iou"]),
                -variant_stage_count(variant, a4_base),
            ),
        )
    return {
        "selected_variant": selected,
        "selection_views": {variant: views[variant] for variant in variant_list},
        "constraints": constraints,
        "eligible_variants": eligible,
        "protocol": protocol,
        "reason": (
            "Highest Synthetic-Val object Macro-F1 among constrained variants; ties use exact instance ratio, instance IoU, then fewer stages."
        ),
    }


def _transition(before_correct: bool, after_correct: bool) -> str:
    if before_correct and not after_correct:
        return "correct_to_incorrect"
    if not before_correct and after_correct:
        return "incorrect_to_correct"
    if before_correct:
        return "unchanged_correct"
    return "unchanged_incorrect"


def _changed_rule_map(after: dict, stage: str) -> dict[int, set[str]]:
    mapping: dict[int, set[str]] = defaultdict(set)
    if stage == "candidate":
        records = after.get("candidate_classifier_postprocess", {}).get("records", [])
        for record in records:
            if record.get("action") not in {"accepted_remap", "rejected_prediction_component_to_stock"}:
                continue
            rule = f"candidate:{record.get('action')}:{record.get('proposal_macro')}->{record.get('predicted_label')}"
            for idx in record.get("face_indices", []):
                mapping[int(idx)].add(rule)
    elif stage == "topology":
        post = after.get("topology_postprocess", {})
        for record in post.get("invalid_components", []) + post.get("repaired_components", []):
            reasons = record.get("source_reasons") or record.get("reasons") or record.get("repair_evidence") or ["unspecified"]
            rule = f"topology:{record.get('action')}:{'|'.join(str(item) for item in reasons)}"
            for idx in record.get("face_indices", []):
                mapping[int(idx)].add(rule)
    elif stage == "rib":
        for key in ("rib_candidate_proposer_postprocess", "rib_boundary_classifier_postprocess"):
            for record in after.get(key, {}).get("records", []):
                if record.get("action") not in {"promoted_to_rib", "pruned_to_stock", "expanded_to_rib"}:
                    continue
                idxs = record.get("face_indices") or [record.get("candidate_face_idx")]
                rule = f"rib:{key}:{record.get('action')}"
                for idx in idxs:
                    if idx is not None:
                        mapping[int(idx)].add(rule)
    return mapping


def decision_delta(
    graph_paths: list[Path],
    before_predictions: dict[str, dict],
    after_predictions: dict[str, dict],
    before_features: dict[str, dict],
    after_features: dict[str, dict],
    *,
    stage: str,
    from_variant: str,
    to_variant: str,
) -> dict[str, Any]:
    face_transitions = Counter()
    object_transitions = Counter()
    by_class: dict[str, Counter] = defaultdict(Counter)
    by_rule: dict[str, Counter] = defaultdict(Counter)
    changed_faces = 0
    changed_object_signatures = 0
    added_objects = 0
    removed_objects = 0
    for graph_path in graph_paths:
        sample_id = graph_path.stem
        graph = load_json(graph_path)
        truth_faces = {int(node["idx"]): int(node["macro_feature_id"]) for node in graph["nodes"]}
        truth_names = {int(node["idx"]): str(node["macro_feature"]) for node in graph["nodes"]}
        before_faces = {int(face["idx"]): face for face in before_predictions[sample_id]["faces"]}
        after_faces = {int(face["idx"]): face for face in after_predictions[sample_id]["faces"]}
        rules = _changed_rule_map(after_predictions[sample_id], stage)
        for idx, truth in truth_faces.items():
            before_correct = int(before_faces[idx]["macro_feature_id"]) == truth
            after_correct = int(after_faces[idx]["macro_feature_id"]) == truth
            transition = _transition(before_correct, after_correct)
            face_transitions[transition] += 1
            by_class[truth_names[idx]][transition] += 1
            changed = (
                before_faces[idx].get("macro_feature_id") != after_faces[idx].get("macro_feature_id")
                or before_faces[idx].get("semantic_subtype_id") != after_faces[idx].get("semantic_subtype_id")
            )
            if changed:
                changed_faces += 1
                by_class[truth_names[idx]]["modified_faces"] += 1
                for rule in rules.get(idx, {f"{stage}:unattributed"}):
                    by_rule[rule][transition] += 1
                    by_rule[rule]["modified_faces"] += 1

        def signatures(document: dict) -> set[tuple[str, tuple[int, ...]]]:
            return {
                (str(item["macro_feature"]), tuple(sorted(int(idx) for idx in item["face_indices"])))
                for item in document.get("feature_objects", [])
            }

        before_set = signatures(before_features[sample_id])
        after_set = signatures(after_features[sample_id])
        removed = before_set - after_set
        added = after_set - before_set
        removed_objects += len(removed)
        added_objects += len(added)
        changed_object_signatures += len(removed) + len(added)
        truth_set = {
            (str(item["macro_feature"]), tuple(sorted(int(idx) for idx in item["node_indices"])))
            for item in graph.get("feature_sets", [])
            if item.get("node_indices")
        }
        for truth_signature in truth_set:
            transition = _transition(truth_signature in before_set, truth_signature in after_set)
            object_transitions[transition] += 1
            by_class[truth_signature[0]][f"object_{transition}"] += 1
    return {
        "schema": "topocae_p1_a2_decision_delta_v1",
        "stage": stage,
        "from_variant": from_variant,
        "to_variant": to_variant,
        "modified_face_count": changed_faces,
        "modified_object_signature_count": changed_object_signatures,
        "added_object_count": added_objects,
        "removed_object_count": removed_objects,
        "face_transitions": dict(face_transitions),
        "object_truth_transitions": dict(object_transitions),
        "by_truth_class": {key: dict(value) for key, value in sorted(by_class.items())},
        "by_rule": {key: dict(value) for key, value in sorted(by_rule.items())},
    }


def _rotate_vector(vector: list[float], rotation_id: int) -> list[float]:
    x, y, z = (float(value) for value in vector)
    if rotation_id == 1:
        return [-y, x, z]
    if rotation_id == 2:
        return [z, y, -x]
    raise ValueError(rotation_id)


def rotate_graph(graph: dict, rotation_id: int, *, source_sample_id: str) -> dict:
    result = deepcopy(graph)
    for node in result.get("nodes", []):
        attributes = node.get("attributes", {})
        physical = attributes.get("physical", {})
        specific = attributes.get("specific", {})
        if isinstance(physical.get("center_of_mass"), list) and len(physical["center_of_mass"]) == 3:
            physical["center_of_mass"] = _rotate_vector(physical["center_of_mass"], rotation_id)
        for key in ("normal", "axis", "apex"):
            if isinstance(specific.get(key), list) and len(specific[key]) == 3:
                specific[key] = _rotate_vector(specific[key], rotation_id)
        shape = attributes.get("shape_descriptors", {})
        extents = [shape.get("bbox_dx"), shape.get("bbox_dy"), shape.get("bbox_dz")]
        if all(value is not None for value in extents):
            dx, dy, dz = (float(value) for value in extents)
            if rotation_id == 1:
                dx, dy, dz = dy, dx, dz
            else:
                dx, dy, dz = dz, dy, dx
            shape["bbox_dx"], shape["bbox_dy"], shape["bbox_dz"] = dx, dy, dz
    result["rotation_source_sample_id"] = source_sample_id
    result["p1_a2_rotation"] = {
        "rotation_id": rotation_id,
        "type": "rigid_axis_permutation",
        "face_index_correspondence": "stable",
    }
    return result


def prepare_synthetic_rotation_val(
    val_graph_paths: list[Path], dataset_config: dict
) -> tuple[list[Path], dict[str, list[str]]]:
    output_root = REPORT_ROOT / "synthetic_rotation_val" / "graphs"
    manifest_path = REPORT_ROOT / "synthetic_rotation_val" / "manifest.json"
    if manifest_path.is_file():
        manifest = load_json(manifest_path)
        paths = [output_root / f"{sample_id}.json" for sample_id in manifest["sample_ids"]]
        if all(path.is_file() for path in paths):
            return paths, manifest["groups"]

    selected: dict[str, Path] = {}
    used: set[str] = set()
    for target in TARGET_CLASSES:
        for graph_path in val_graph_paths:
            if graph_path.stem in used:
                continue
            graph = load_json(graph_path)
            classes = {str(item.get("macro_feature")) for item in graph.get("feature_sets", [])}
            if target in classes:
                selected[target] = graph_path
                used.add(graph_path.stem)
                break
        if target not in selected:
            raise RuntimeError(f"Synthetic-Val has no unique rotation source for {target}")

    paths = []
    groups = {}
    source_rows = []
    for target, source_path in selected.items():
        source_graph = load_json(source_path)
        sample_ids = []
        base_id = f"rotation_{target}_base"
        base_graph = deepcopy(source_graph)
        base_graph["rotation_source_sample_id"] = source_path.stem
        base_path = output_root / f"{base_id}.json"
        write_json(base_path, base_graph)
        paths.append(base_path)
        sample_ids.append(base_id)
        for rotation_id in (1, 2):
            sample_id = f"rotation_{target}_r{rotation_id}"
            rotated_path = output_root / f"{sample_id}.json"
            write_json(
                rotated_path,
                rotate_graph(source_graph, rotation_id, source_sample_id=source_path.stem),
            )
            paths.append(rotated_path)
            sample_ids.append(sample_id)
        groups[target] = sample_ids
        source_rows.append(
            {
                "target_class": target,
                "source_sample_id": source_path.stem,
                "source_graph_sha256": sha256_file(source_path),
                "rotation_sample_ids": sample_ids,
            }
        )
    write_json(
        manifest_path,
        {
            "schema": "topocae_p1_a2_synthetic_rotation_val_v1",
            "source_split": "Synthetic-Val",
            "sample_count": len(paths),
            "sample_ids": [path.stem for path in paths],
            "groups": groups,
            "sources": source_rows,
            "selection_allowed": True,
        },
    )
    return paths, groups


def rotation_metrics(
    output_root: Path,
    groups: dict[str, list[str]],
) -> dict[str, Any]:
    return legacy_metrics.rotation_consistency(
        prediction_dir=output_root / "predictions",
        feature_dir=output_root / "feature_objects",
        groups=groups,
    )


def canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: canonicalize(item)
            for key, item in sorted(value.items())
            if key not in OUTPUT_LOCATION_KEYS
        }
    if isinstance(value, list):
        return [canonicalize(item) for item in value]
    return value


def compare_output_trees(left: Path, right: Path) -> dict[str, Any]:
    left_files = {
        path.relative_to(left).as_posix(): path
        for path in left.rglob("*.json")
        if path.name != "manifest.json"
    }
    right_files = {
        path.relative_to(right).as_posix(): path
        for path in right.rglob("*.json")
        if path.name != "manifest.json"
    }
    missing_left = sorted(set(right_files) - set(left_files))
    missing_right = sorted(set(left_files) - set(right_files))
    mismatches = []
    for relative in sorted(set(left_files) & set(right_files)):
        if stable_hash(canonicalize(load_json(left_files[relative]))) != stable_hash(
            canonicalize(load_json(right_files[relative]))
        ):
            mismatches.append(relative)
    return {
        "compared_file_count": len(set(left_files) & set(right_files)),
        "missing_in_run_1": missing_left,
        "missing_in_run_2": missing_right,
        "mismatches": mismatches,
        "status": "pass" if not missing_left and not missing_right and not mismatches else "fail",
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def formal_signature_sets(dataset_config: dict) -> tuple[set[str], set[str], set[str]]:
    formal = dataset_config["formal_synthetic"]
    graph_root = resolve_repo_path(formal["graph_root"])
    sample_ids: set[str] = set()
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
            exact.add(rotation_invariant_graph_signature(graph, ndigits=6))
            near.add(rotation_invariant_graph_signature(graph, ndigits=3))
    return sample_ids, exact, near
