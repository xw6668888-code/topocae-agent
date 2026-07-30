from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

import torch

from models.p1_b_candidate_topology_classifier import predict_candidate_record
from utils.p1_b_candidate_features import build_candidate_tensor_record as build_p1_b_candidate_record
from utils.p1_b_candidate_proposer import generate_candidate_proposals
from utils.p1_b_global_decode_v2 import decode_feature_objects
from utils.p1_b_instance_assembly import assemble_same_instance_candidates
from utils.p1_b_topology_verifier_v4 import verify_candidate_prediction as verify_p1_b
from utils.p1_c_candidate_features_polarity_v1 import build_candidate_tensor_record as build_p1_c_candidate_record
from utils.p1_c_topology_verifier_polarity_v1 import verify_candidate_prediction as verify_p1_c

from .common import (
    OUTPUT_ROOT,
    ROOT,
    candidate_area,
    canonical_object_signature,
    connected_macro_objects,
    normalize_objects,
    one_to_one_area_match,
    read_json,
    strip_graph_truth,
    strip_prediction_truth,
    write_csv,
    write_json,
)
from .runtime import get_spec, load_candidate_model, load_gnn, predict_faces


LAYERED_ROOT = OUTPUT_ROOT / "layered_comparison"
AREA_TRUTH_PATH = ROOT / "reports/p1_b_real_dev_v1/area_truth.json"
EXACT_TRUTH_METRICS = ROOT / "reports/p1_b_real_dev_v1/metrics.json"

HOLE_CLASSES = {
    "through_hole_round", "blind_hole_round", "stepped_hole_round",
    "counterbore_hole", "countersink_hole", "obround_hole",
}
SLOT_CLASSES = {"slot_rect", "slot_u", "slot_v", "slot_o"}
RIB_WEB_CLASSES = {"rib", "web"}
TRANSITION_CLASSES = {"fillet", "chamfer"}
FAMILY_MEMBERS = {
    "hole": HOLE_CLASSES,
    "slot": SLOT_CLASSES,
    "rib_web": RIB_WEB_CLASSES,
    "transition": TRANSITION_CLASSES,
}


def _instrument_pipeline(
    pipeline_key: str,
    graph: Mapping[str, Any],
    face_prediction: Mapping[str, Any],
    candidate_model,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, float]]:
    spec = get_spec(pipeline_key)
    graph_clean = strip_graph_truth(graph)
    prediction_clean = strip_prediction_truth(face_prediction)
    same = read_json(spec.same_instance_selection)
    candidate_thresholds = read_json(spec.candidate_selection)
    global_selection = read_json(spec.global_selection)
    if pipeline_key == "p1_c":
        selected = global_selection["selected"]
        candidate_thresholds = dict(candidate_thresholds)
        candidate_thresholds["min_candidate_confidence"] = float(selected["min_candidate_confidence"])
        build_record = build_p1_c_candidate_record
        verify = verify_p1_c
        subset_tolerance = float(selected["subset_confidence_tolerance"])
    else:
        build_record = build_p1_b_candidate_record
        verify = verify_p1_b
        subset_tolerance = float(global_selection["global_subset_confidence_tolerance"])

    started_l1 = time.perf_counter()
    assembly = assemble_same_instance_candidates(
        graph_clean,
        prediction_clean,
        global_threshold=float(same["global_threshold"]),
        threshold_selection_sha256=str(same["selection_sha256"]),
    )
    proposal_result = generate_candidate_proposals(graph_clean, prediction_clean, assembly)
    raw_macros = {
        int(face.get("idx", position)): str(face.get("macro_feature", face.get("raw_macro_feature", "stock")))
        for position, face in enumerate(prediction_clean.get("faces", []))
        if isinstance(face, Mapping)
    }
    tensor_records = []
    model_predictions = []
    for proposal in proposal_result["proposals"]:
        tensor_record = build_record(graph_clean, prediction_clean, proposal)
        model_prediction = predict_candidate_record(candidate_model, tensor_record, device=str(device))
        model_prediction.setdefault("candidate_id", proposal.get("candidate_id"))
        model_prediction.setdefault("face_indices", proposal.get("face_indices", []))
        model_prediction["face_macro_predictions"] = raw_macros
        tensor_records.append(tensor_record)
        model_predictions.append(model_prediction)
    l1_seconds = time.perf_counter() - started_l1

    started_l2 = time.perf_counter()
    verified = [
        verify(graph_clean, proposal, prediction, thresholds=candidate_thresholds)
        for proposal, prediction in zip(proposal_result["proposals"], model_predictions)
    ]
    l2_seconds = time.perf_counter() - started_l2

    started_l3 = time.perf_counter()
    decoded = decode_feature_objects(
        graph_clean,
        verified,
        subset_confidence_tolerance=subset_tolerance,
        selection_sha256=str(global_selection["selection_sha256"]),
    )
    l3_seconds = time.perf_counter() - started_l3
    return {
        "assembly": assembly,
        "candidate_proposals": proposal_result,
        "candidate_tensor_records": tensor_records,
        "candidate_model_predictions": model_predictions,
        "verified_candidates": verified,
        "decoded_feature_objects": decoded,
        "candidate_thresholds": candidate_thresholds,
    }, {"L1_seconds": l1_seconds, "L2_seconds": l2_seconds, "L3_seconds": l3_seconds}


def _classified_candidate_objects(
    graph: Mapping[str, Any],
    model_predictions: Sequence[Mapping[str, Any]],
    membership_threshold: float,
) -> list[dict[str, Any]]:
    records = []
    for prediction in model_predictions:
        classes = list(prediction.get("macro_classes", []))
        probabilities = list(prediction.get("macro_probabilities", []))
        if not classes or len(classes) != len(probabilities):
            continue
        macro = str(classes[max(range(len(probabilities)), key=lambda index: float(probabilities[index]))])
        original_faces = [int(value) for value in prediction.get("face_indices", [])]
        membership = list(prediction.get("membership_probabilities", []))
        faces = [
            face for face, probability in zip(original_faces, membership)
            if float(probability) >= membership_threshold
        ]
        if not faces:
            continue
        records.append({
            "candidate_id": prediction.get("candidate_id"),
            "macro_feature": macro,
            "face_indices": sorted(set(faces)),
            "area": candidate_area(graph, faces),
            "macro_confidence": max(float(value) for value in probabilities),
            "validity_score": float(prediction.get("validity_score", 0.0)),
        })
    return normalize_objects(graph, records)


def _verified_objects(graph: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    accepted = []
    for record in records:
        if record.get("abstain") or str(record.get("status")) != "accepted":
            continue
        faces = record.get("refined_face_indices", [])
        if not faces or str(record.get("predicted_macro")) == "unknown":
            continue
        accepted.append({
            "candidate_id": record.get("candidate_id"),
            "macro_feature": record.get("predicted_macro"),
            "face_indices": faces,
            "area": candidate_area(graph, faces),
            "final_confidence": record.get("final_confidence"),
        })
    return normalize_objects(graph, accepted)


def _raw_objects(graph: Mapping[str, Any], prediction: Mapping[str, Any], field: str = "macro_feature") -> list[dict[str, Any]]:
    macros = {
        int(face.get("idx", position)): str(face.get(field, face.get("macro_feature", "stock")))
        for position, face in enumerate(prediction.get("faces", []))
        if isinstance(face, Mapping)
    }
    return normalize_objects(graph, connected_macro_objects(graph, macros))


def _iou(left: Iterable[int], right: Iterable[int]) -> float:
    a, b = set(left), set(right)
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _exact_diagnostics(
    truths: Sequence[Mapping[str, Any]],
    proposals: Sequence[Mapping[str, Any]],
    classified: Sequence[Mapping[str, Any]],
    verified: Sequence[Mapping[str, Any]],
    final_objects: Sequence[Mapping[str, Any]],
    decoded: Mapping[str, Any],
) -> dict[str, Any]:
    if not truths:
        return {
            "truth_count": 0,
            "scope": "unavailable: no exact face-set truth for this sample",
            "proposal_recall": {"iou_0_5": None, "iou_0_75": None, "iou_0_95": None},
            "candidate_conditional_macro_accuracy": None,
            "candidate_conditional_macro_f1": None,
            "membership_mean_iou": None,
            "exact_boundary_ratio": None,
            "verifier_false_veto": None,
            "verifier_false_accept_reviewed_subset": None,
            "verifier_demotion": sum(bool(record.get("abstain")) for record in verified),
            "decoder_merge_loss": None,
            "decoder_split_loss": None,
            "decoder_conflict_loss": int(decoded.get("global_conflict_reject_count", 0)),
        }

    proposal_best = [max((_iou(truth["truth_face_indices"], record.get("face_indices", [])) for record in proposals), default=0.0) for truth in truths]
    conditional_pairs = []
    best_membership = []
    exact_count = 0
    false_veto = 0
    verified_by_id = {record.get("candidate_id"): record for record in verified}
    for truth in truths:
        candidates = sorted(
            ((_iou(truth["truth_face_indices"], record.get("face_indices", [])), record) for record in classified),
            key=lambda pair: pair[0], reverse=True,
        )
        best_iou, best = candidates[0] if candidates else (0.0, None)
        best_membership.append(best_iou)
        exact_count += best_iou == 1.0
        if best is not None and best_iou >= 0.5:
            correct = str(best.get("macro_feature")) == str(truth.get("macro_feature"))
            conditional_pairs.append((str(truth.get("macro_feature")), str(best.get("macro_feature"))))
            verified_record = verified_by_id.get(best.get("object_id"))
            if correct and (verified_record is None or verified_record.get("abstain") or verified_record.get("status") != "accepted"):
                false_veto += 1
    conditional_correct = sum(expected == predicted for expected, predicted in conditional_pairs)
    conditional_accuracy = conditional_correct / len(conditional_pairs) if conditional_pairs else None
    conditional_f1 = _macro_f1_pairs(conditional_pairs) if conditional_pairs else None

    reviewed_union = set().union(*(set(truth["truth_face_indices"]) for truth in truths))
    false_accept = 0
    for record in verified:
        if record.get("abstain") or record.get("status") != "accepted":
            continue
        faces = set(record.get("refined_face_indices", []))
        if not faces & reviewed_union:
            continue
        if not any(
            str(record.get("predicted_macro")) == str(truth.get("macro_feature"))
            and _iou(faces, truth["truth_face_indices"]) >= 0.5
            for truth in truths
        ):
            false_accept += 1

    split_loss = sum(
        sum(_iou(truth["truth_face_indices"], record.get("face_indices", [])) > 0 for record in final_objects) > 1
        for truth in truths
    )
    merge_loss = sum(
        sum(_iou(record.get("face_indices", []), truth["truth_face_indices"]) > 0 for truth in truths) > 1
        for record in final_objects
    )
    return {
        "truth_count": len(truths),
        "scope": "frozen reviewed exact face-set subset only",
        "proposal_recall": {
            "iou_0_5": sum(value >= 0.5 for value in proposal_best) / len(truths),
            "iou_0_75": sum(value >= 0.75 for value in proposal_best) / len(truths),
            "iou_0_95": sum(value >= 0.95 for value in proposal_best) / len(truths),
        },
        "candidate_conditional_macro_accuracy": conditional_accuracy,
        "candidate_conditional_macro_f1": conditional_f1,
        "membership_mean_iou": sum(best_membership) / len(best_membership),
        "exact_boundary_ratio": exact_count / len(truths),
        "verifier_false_veto": false_veto,
        "verifier_false_accept_reviewed_subset": false_accept,
        "verifier_demotion": sum(bool(record.get("abstain")) for record in verified),
        "decoder_merge_loss": merge_loss,
        "decoder_split_loss": split_loss,
        "decoder_conflict_loss": int(decoded.get("global_conflict_reject_count", 0)),
    }


def _macro_f1_pairs(pairs: Sequence[tuple[str, str]]) -> float:
    classes = sorted({value for pair in pairs for value in pair})
    scores = []
    for label in classes:
        tp = sum(expected == label and predicted == label for expected, predicted in pairs)
        fp = sum(expected != label and predicted == label for expected, predicted in pairs)
        fn = sum(expected == label and predicted != label for expected, predicted in pairs)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def _layer_transition(previous: Sequence[Mapping[str, Any]], current: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    previous_faces = set().union(*(set(record.get("face_indices", [])) for record in previous)) if previous else set()
    current_faces = set().union(*(set(record.get("face_indices", [])) for record in current)) if current else set()
    return {
        "retained_faces": len(previous_faces & current_faces),
        "added_faces": len(current_faces - previous_faces),
        "deleted_faces": len(previous_faces - current_faces),
        "changed_face_membership": len(previous_faces ^ current_faces),
        "previous_candidate_count": len(previous),
        "current_candidate_count": len(current),
        "candidate_count_delta": len(current) - len(previous),
    }


def _changed_face_ratio(
    graph: Mapping[str, Any], prediction: Mapping[str, Any], final_objects: Sequence[Mapping[str, Any]],
) -> float:
    raw = {
        int(face.get("idx", position)): str(face.get("macro_feature", "stock"))
        for position, face in enumerate(prediction.get("faces", [])) if isinstance(face, Mapping)
    }
    final: dict[int, set[str]] = defaultdict(set)
    for record in final_objects:
        for index in record.get("face_indices", []):
            final[int(index)].add(str(record.get("macro_feature", "unknown")))
    changed = 0
    for index in range(len(graph.get("nodes", []))):
        final_value = next(iter(final[index])) if len(final[index]) == 1 else ("stock" if not final[index] else "+".join(sorted(final[index])))
        changed += raw.get(index, "stock") != final_value
    return changed / max(len(graph.get("nodes", [])), 1)


def _evaluate_frozen_sample(
    pipeline_key: str,
    sample_key: str,
    truths: Sequence[Mapping[str, Any]],
    exact_truths: Sequence[Mapping[str, Any]],
    candidate_model,
    gnn,
    gnn_config,
    device: torch.device,
) -> dict[str, Any]:
    report_root = ROOT / ("reports/p1_b_real_dev_v1" if pipeline_key == "p1_b" else "reports/p1_c_real_dev_v1")
    graph_path = report_root / "graphs_v2" / f"{sample_key}.json"
    frozen_prediction_path = report_root / "predictions" / f"{sample_key}.json"
    graph = read_json(graph_path)
    started_l0 = time.perf_counter()
    prediction = predict_faces(get_spec(pipeline_key), strip_graph_truth(graph), gnn, gnn_config, device)
    l0_seconds = time.perf_counter() - started_l0
    frozen_prediction = read_json(frozen_prediction_path)
    raw_reproduced = [str(face.get("macro_feature")) for face in prediction.get("faces", [])]
    raw_frozen = [str(face.get("macro_feature")) for face in frozen_prediction.get("faces", [])]
    raw_signature_match = raw_reproduced == raw_frozen
    pipeline, timings = _instrument_pipeline(pipeline_key, graph, prediction, candidate_model, device)
    raw_objects = _raw_objects(graph, prediction)
    classified = _classified_candidate_objects(
        graph,
        pipeline["candidate_model_predictions"],
        float(pipeline["candidate_thresholds"]["membership_threshold"]),
    )
    verified = _verified_objects(graph, pipeline["verified_candidates"])
    final_objects = normalize_objects(graph, pipeline["decoded_feature_objects"]["feature_objects"])
    stages = {"L0_raw_gnn": raw_objects, "L1_candidate_classifier": classified, "L2_verifier": verified, "L3_decoder": final_objects}
    stage_metrics = {name: one_to_one_area_match(truths, records) for name, records in stages.items()}
    for value in stage_metrics.values():
        value.pop("matches", None)
        value.pop("unmatched_truth_indices", None)
        value.pop("unmatched_prediction_indices", None)
    return {
        "system": get_spec(pipeline_key).name,
        "sample_key": sample_key,
        "truth_count": len(truths),
        "raw_prediction_signature_matches_frozen": raw_signature_match,
        "stage_objects": stages,
        "stage_metrics": stage_metrics,
        "transitions": {
            "L0_to_L1": _layer_transition(raw_objects, classified),
            "L1_to_L2": _layer_transition(classified, verified),
            "L2_to_L3": _layer_transition(verified, final_objects),
        },
        "raw_to_final_macro_changed_face_ratio": _changed_face_ratio(graph, prediction, final_objects),
        "exact_diagnostics": _exact_diagnostics(
            exact_truths,
            pipeline["candidate_proposals"]["proposals"],
            classified,
            pipeline["verified_candidates"],
            final_objects,
            pipeline["decoded_feature_objects"],
        ),
        "timing": {"L0_seconds": l0_seconds, **timings},
        "pipeline_counts": {
            "face_count": len(graph.get("nodes", [])),
            "proposal_count": len(pipeline["candidate_proposals"]["proposals"]),
            "classified_candidate_count": len(classified),
            "verified_accepted_count": len(verified),
            "decoded_object_count": len(final_objects),
            "abstention_count": int(pipeline["decoded_feature_objects"].get("abstention_count", 0)),
            "global_conflict_reject_count": int(pipeline["decoded_feature_objects"].get("global_conflict_reject_count", 0)),
        },
    }


def _legacy_candidate_macros(prediction: Mapping[str, Any]) -> dict[int, str]:
    result = {
        int(face.get("idx", position)): str(face.get("raw_macro_feature", face.get("macro_feature", "stock")))
        for position, face in enumerate(prediction.get("faces", [])) if isinstance(face, Mapping)
    }
    records = prediction.get("candidate_classifier_postprocess", {}).get("records", [])
    for record in records:
        if not str(record.get("action", "")).startswith("accepted"):
            continue
        label = str(record.get("predicted_label", record.get("proposal_macro", "candidate")))
        for index in record.get("face_indices", []):
            result[int(index)] = label
    return result


def _evaluate_legacy_sample(sample_key: str, truths: Sequence[Mapping[str, Any]], exact_truths: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    sample_id = sample_key.split("/", 1)[1]
    base = ROOT / "output_real_step_s2_v5_6_a_rolefix"
    graph = read_json(ROOT / f"output_real_step_s2_v5_4_g_looprole/graphs/{sample_id}.json")
    prediction = read_json(base / f"predictions/edge_aware_v5_6_a_candidate_topology_post/{sample_id}.json")
    final_payload = read_json(base / f"feature_objects/edge_aware_v5_6_a_candidate_topology_post/{sample_id}.json")
    raw_macros = {
        int(face.get("idx", position)): str(face.get("raw_macro_feature", face.get("macro_feature", "stock")))
        for position, face in enumerate(prediction.get("faces", [])) if isinstance(face, Mapping)
    }
    candidate_macros = _legacy_candidate_macros(prediction)
    final_macros = {
        int(face.get("idx", position)): str(face.get("macro_feature", "stock"))
        for position, face in enumerate(prediction.get("faces", [])) if isinstance(face, Mapping)
    }
    raw_objects = normalize_objects(graph, connected_macro_objects(graph, raw_macros))
    candidate_objects = normalize_objects(graph, connected_macro_objects(graph, candidate_macros))
    topology_objects = normalize_objects(graph, connected_macro_objects(graph, final_macros))
    final_objects = normalize_objects(graph, final_payload.get("feature_objects", []))
    stages = {"L0_raw_gnn": raw_objects, "L1_candidate_classifier": candidate_objects, "L2_verifier": topology_objects, "L3_decoder": final_objects}
    stage_metrics = {name: one_to_one_area_match(truths, records) for name, records in stages.items()}
    for value in stage_metrics.values():
        value.pop("matches", None)
        value.pop("unmatched_truth_indices", None)
        value.pop("unmatched_prediction_indices", None)
    candidate_records = prediction.get("candidate_classifier_postprocess", {}).get("records", [])
    topology = prediction.get("topology_postprocess", {})
    pseudo_verified = [
        {
            "candidate_id": f"legacy_{index}",
            "status": "accepted",
            "abstain": False,
            "predicted_macro": record.get("macro_feature"),
            "refined_face_indices": record.get("face_indices", []),
        }
        for index, record in enumerate(topology_objects)
    ]
    return {
        "system": "legacy v5_6_a_rolefix",
        "sample_key": sample_key,
        "truth_count": len(truths),
        "raw_prediction_signature_matches_frozen": True,
        "stage_objects": stages,
        "stage_metrics": stage_metrics,
        "transitions": {
            "L0_to_L1": _layer_transition(raw_objects, candidate_objects),
            "L1_to_L2": _layer_transition(candidate_objects, topology_objects),
            "L2_to_L3": _layer_transition(topology_objects, final_objects),
        },
        "raw_to_final_macro_changed_face_ratio": sum(raw_macros.get(index, "stock") != final_macros.get(index, "stock") for index in raw_macros) / max(len(raw_macros), 1),
        "exact_diagnostics": _exact_diagnostics(exact_truths, candidate_records, candidate_objects, pseudo_verified, final_objects, {"global_conflict_reject_count": 0}),
        "timing": {"L0_seconds": None, "L1_seconds": None, "L2_seconds": None, "L3_seconds": None},
        "pipeline_counts": {
            "face_count": len(graph.get("nodes", [])),
            "proposal_count": int(prediction.get("candidate_classifier_postprocess", {}).get("candidate_count", len(candidate_records))),
            "classified_candidate_count": len(candidate_objects),
            "verified_accepted_count": len(topology_objects),
            "decoded_object_count": len(final_objects),
            "abstention_count": 0,
            "global_conflict_reject_count": 0,
            "topology_repaired_component_count": int(topology.get("repaired_component_count", 0)),
            "topology_demoted_face_count": int(topology.get("demoted_face_count", 0)),
        },
    }


def _aggregate(system: str, view: str, samples: Sequence[Mapping[str, Any]], truth_by_sample: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    stage_names = ("L0_raw_gnn", "L1_candidate_classifier", "L2_verifier", "L3_decoder")
    stage_summary = {}
    per_class_rows = []
    for stage in stage_names:
        truth_count = prediction_count = matched_count = 0
        per_class = defaultdict(lambda: Counter(truth=0, prediction=0, matched=0))
        for sample in samples:
            truths = list(truth_by_sample[sample["sample_key"]])
            predictions = list(sample["stage_objects"][stage])
            matching = one_to_one_area_match(truths, predictions)
            truth_count += matching["truth_count"]
            prediction_count += matching["prediction_count"]
            matched_count += matching["matched_count"]
            for truth in truths:
                per_class[str(truth["macro_feature"])]["truth"] += 1
            for prediction in predictions:
                per_class[str(prediction["macro_feature"])]["prediction"] += 1
            for pair in matching["matches"]:
                macro = str(truths[pair["truth_index"]]["macro_feature"])
                per_class[macro]["matched"] += 1
        precision = matched_count / prediction_count if prediction_count else 0.0
        recall = matched_count / truth_count if truth_count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        stage_summary[stage] = {
            "truth_count": truth_count,
            "prediction_count": prediction_count,
            "matched_count": matched_count,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        for macro, counts in sorted(per_class.items()):
            p = counts["matched"] / counts["prediction"] if counts["prediction"] else 0.0
            r = counts["matched"] / counts["truth"] if counts["truth"] else 0.0
            row = {
                "system": system, "view": view, "layer": stage, "macro_feature": macro,
                "family": _family(macro), "truth": counts["truth"], "prediction": counts["prediction"], "matched": counts["matched"],
                "precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
            }
            per_class_rows.append(row)
        for family, members in FAMILY_MEMBERS.items():
            selected = [counts for macro, counts in per_class.items() if macro in members]
            counts = Counter({key: sum(value[key] for value in selected) for key in ("truth", "prediction", "matched")})
            p = counts["matched"] / counts["prediction"] if counts["prediction"] else 0.0
            r = counts["matched"] / counts["truth"] if counts["truth"] else 0.0
            per_class_rows.append({
                "system": system, "view": view, "layer": stage, "macro_feature": f"__family__{family}",
                "family": family, "truth": counts["truth"], "prediction": counts["prediction"], "matched": counts["matched"],
                "precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
            })

    exact_records = [sample["exact_diagnostics"] for sample in samples if sample["exact_diagnostics"]["truth_count"]]
    exact_truth_count = sum(record["truth_count"] for record in exact_records)
    exact_summary = {
        "truth_count": exact_truth_count,
        "scope": "frozen reviewed exact face-set subset only",
        "proposal_recall_iou_0_5": _weighted(exact_records, ("proposal_recall", "iou_0_5")),
        "proposal_recall_iou_0_75": _weighted(exact_records, ("proposal_recall", "iou_0_75")),
        "proposal_recall_iou_0_95": _weighted(exact_records, ("proposal_recall", "iou_0_95")),
        "candidate_conditional_macro_accuracy": _weighted(exact_records, ("candidate_conditional_macro_accuracy",)),
        "candidate_conditional_macro_f1": _weighted(exact_records, ("candidate_conditional_macro_f1",)),
        "membership_mean_iou": _weighted(exact_records, ("membership_mean_iou",)),
        "exact_boundary_ratio": _weighted(exact_records, ("exact_boundary_ratio",)),
        "verifier_false_veto": sum(record["verifier_false_veto"] for record in exact_records),
        "verifier_false_accept_reviewed_subset": sum(record["verifier_false_accept_reviewed_subset"] for record in exact_records),
        "verifier_demotion": sum(record["verifier_demotion"] for record in exact_records),
        "decoder_merge_loss": sum(record["decoder_merge_loss"] for record in exact_records),
        "decoder_split_loss": sum(record["decoder_split_loss"] for record in exact_records),
        "decoder_conflict_loss": sum(record["decoder_conflict_loss"] for record in exact_records),
    }
    timing = {}
    for layer in ("L0_seconds", "L1_seconds", "L2_seconds", "L3_seconds"):
        values = [sample["timing"][layer] for sample in samples if sample["timing"][layer] is not None]
        timing[layer] = {
            "total": sum(values) if values else None,
            "mean": sum(values) / len(values) if values else None,
            "sample_count": len(values),
        }
    return {
        "system": system,
        "view": view,
        "sample_count": len(samples),
        "sample_keys": [sample["sample_key"] for sample in samples],
        "area_truth_protocol": {
            "source": str(AREA_TRUTH_PATH.resolve()),
            "absolute_tolerance": 5.0,
            "relative_tolerance": 0.01,
            "connected_same_macro_merge": False,
            "exact_face_set_truth_available_for_all": False,
        },
        "layers": stage_summary,
        "exact_subset": exact_summary,
        "raw_to_final_macro_changed_face_ratio": sum(sample["raw_to_final_macro_changed_face_ratio"] for sample in samples) / max(len(samples), 1),
        "timing": timing,
        "pipeline_count_totals": {key: sum(int(sample["pipeline_counts"].get(key, 0)) for sample in samples) for key in (
            "face_count", "proposal_count", "classified_candidate_count", "verified_accepted_count", "decoded_object_count", "abstention_count", "global_conflict_reject_count"
        )},
        "raw_prediction_reproduction_all_match": all(sample["raw_prediction_signature_matches_frozen"] for sample in samples),
        "per_class_rows": per_class_rows,
    }


def _weighted(records: Sequence[Mapping[str, Any]], path: tuple[str, ...]) -> float | None:
    numerator = denominator = 0.0
    for record in records:
        value: Any = record
        for key in path:
            value = value.get(key) if isinstance(value, Mapping) else None
        if value is None:
            continue
        weight = float(record["truth_count"])
        numerator += float(value) * weight
        denominator += weight
    return numerator / denominator if denominator else None


def _family(macro: str) -> str:
    for family, members in FAMILY_MEMBERS.items():
        if macro in members:
            return family
    return "other"


def _per_sample_rows(system: str, view: str, samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for sample in samples:
        for layer, metric in sample["stage_metrics"].items():
            rows.append({
                "system": system,
                "view": view,
                "sample_key": sample["sample_key"],
                "layer": layer,
                "truth": metric["truth_count"],
                "prediction": metric["prediction_count"],
                "matched": metric["matched_count"],
                "precision": metric["precision"],
                "recall": metric["recall"],
                "f1": metric["f1"],
                "raw_to_final_macro_changed_face_ratio": sample["raw_to_final_macro_changed_face_ratio"],
                "proposal_recall_iou_0_5": sample["exact_diagnostics"]["proposal_recall"]["iou_0_5"],
                "membership_mean_iou": sample["exact_diagnostics"]["membership_mean_iou"],
                "exact_boundary_ratio": sample["exact_diagnostics"]["exact_boundary_ratio"],
                "layer_seconds": sample["timing"][layer.replace("_raw_gnn", "").replace("_candidate_classifier", "").replace("_verifier", "").replace("_decoder", "") + "_seconds"],
            })
    return rows


def _strip_stage_objects(sample: Mapping[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in sample.items() if key != "stage_objects"}
    result["stage_object_signatures"] = {
        layer: canonical_object_signature(records) for layer, records in sample["stage_objects"].items()
    }
    return result


def _report(metrics: Mapping[str, Any]) -> str:
    views = metrics["views"]
    s2 = {record["system"]: record for record in views["S2-Overlap"]}
    full = {record["system"]: record for record in views["Real-Dev-Full"]}
    legacy = s2["legacy v5_6_a_rolefix"]
    raw_matched = legacy["layers"]["L0_raw_gnn"]["matched_count"]
    candidate_matched = legacy["layers"]["L1_candidate_classifier"]["matched_count"]
    final_matched = legacy["layers"]["L3_decoder"]["matched_count"]
    classifier_increment = candidate_matched - raw_matched
    topology_increment = final_matched - candidate_matched
    p1_b = full["P1-B frozen"]
    p1_c = full["P1-C frozen"]
    layer_order = ("L0_raw_gnn", "L1_candidate_classifier", "L2_verifier", "L3_decoder")
    first_degradation = next(
        (layer for layer in layer_order if p1_c["layers"][layer]["f1"] < p1_b["layers"][layer]["f1"]),
        "none",
    )
    exact = p1_b["exact_subset"]
    candidates = {
        "candidate coverage": 1.0 - float(exact["proposal_recall_iou_0_5"] or 0.0),
        "candidate macro": 1.0 - float(exact["candidate_conditional_macro_accuracy"] or 0.0),
        "decoder boundary": 1.0 - float(exact["exact_boundary_ratio"] or 0.0),
        "verifier veto": exact["verifier_false_veto"] / max(exact["truth_count"], 1),
    }
    dominant = max(candidates, key=candidates.get)
    lines = [
        "# P1-D Lite 四层流水线统一比较", "",
        "## 评价口径", "",
        "本报告严格分离 `S2-Overlap` 与 `Real-Dev-Full`。对象 P/R/F1 使用冻结面积真值、绝对 5 mm² 或相对 1% 的一对一 macro+area 匹配；不启用 connected same-macro truth merge。proposal IoU、membership IoU 和 exact boundary 只在 6 个冻结人工 exact face-set 对象上报告，不能外推到全部 Real-Dev。", "",
        "旧 rolefix 没有可执行命令、源码 revision 和完整阈值快照，因此只能从冻结 prediction/feature-object 产物重建 L0–L3，层运行时间记为不可用。P1-B/P1-C 的 L0–L3 均由冻结模型和阈值在 CPU 上重放。", "",
        "## 主要结论", "",
        f"- 旧 rolefix 在 S2-Overlap 的最终 object F1 为 `{legacy['layers']['L3_decoder']['f1']:.6f}`。按 matched truth object 计，raw GNN 直接贡献 `{raw_matched}/{final_matched}`；candidate classifier 净增 `{classifier_increment}`；topology postprocess/最终解码净增 `{topology_increment}`。这说明其高分是学习输出与长期规则重组的联合结果，而非 raw GNN 单独达到 1.0。",
        f"- P1-C 相比 P1-B 在 Real-Dev-Full 首次可见退化层为 `{first_degradation}`：P1-B/P1-C 该层 F1 分别为 `{p1_b['layers'][first_degradation]['f1']:.6f}` / `{p1_c['layers'][first_degradation]['f1']:.6f}`。",
        f"- 在当前 exact 子集的可测信号中，主导失败暂定为 `{dominant}`；各信号为 {json.dumps(candidates, ensure_ascii=False)}。该判断只用于 Task 1 定位，最终优先级以 Task 2 Oracle F1 增益为准。",
        f"- P1-B/P1-C Real-Dev-Full 最终 object F1 为 `{p1_b['layers']['L3_decoder']['f1']:.6f}` / `{p1_c['layers']['L3_decoder']['f1']:.6f}`；P1-B 保持当前可运行最佳。",
        "", "## 能用于论文与不能用于论文的结论", "",
        "可用于论文：受控开发域上，candidate/topology 层能显著改变 raw GNN 结果；P1-C 的同生成器提升没有转化为 Real-Dev 提升；误差必须按 face/proposal/classifier/verifier/decoder 分解。", "",
        "不能用于论文外推：旧 rolefix 的 S2 1.0、任何面积匹配结果、以及本报告的 Real-Dev 数字都不能描述为未知真实件泛化；缺少 exact face-set truth 的样本不能声称 face Macro-F1 或 exact boundary accuracy。", "",
    ]
    return "\n".join(lines)


def main() -> int:
    truth_payload = read_json(AREA_TRUTH_PATH)
    truth_by_sample = truth_payload["samples"]
    exact_records = read_json(EXACT_TRUTH_METRICS)["reviewed_face_sets"]["records"]
    exact_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in exact_records:
        exact_by_sample[str(record["sample_key"])].append({
            "truth_id": record["truth_id"],
            "macro_feature": record["macro_feature"],
            "truth_face_indices": record["truth_face_indices"],
        })
    full_samples = list(truth_by_sample)
    s2_samples = [sample for sample in full_samples if sample.startswith("test_part_s2/")]
    device = torch.device("cpu")
    sample_results: dict[str, list[dict[str, Any]]] = {}
    for pipeline_key in ("p1_b", "p1_c"):
        spec = get_spec(pipeline_key)
        gnn, gnn_config = load_gnn(spec, device)
        candidate_model = load_candidate_model(spec, device)
        sample_results[pipeline_key] = [
            _evaluate_frozen_sample(
                pipeline_key, sample, truth_by_sample[sample], exact_by_sample.get(sample, []),
                candidate_model, gnn, gnn_config, device,
            )
            for sample in full_samples
        ]
    legacy_results = [
        _evaluate_legacy_sample(sample, truth_by_sample[sample], exact_by_sample.get(sample, []))
        for sample in s2_samples
    ]

    views: dict[str, list[dict[str, Any]]] = {"S2-Overlap": [], "Real-Dev-Full": []}
    all_per_class = []
    all_per_sample = []
    for pipeline_key in ("p1_b", "p1_c"):
        system = get_spec(pipeline_key).name
        records = sample_results[pipeline_key]
        s2_records = [record for record in records if record["sample_key"] in s2_samples]
        for view, selected in (("S2-Overlap", s2_records), ("Real-Dev-Full", records)):
            summary = _aggregate(system, view, selected, truth_by_sample)
            all_per_class.extend(summary.pop("per_class_rows"))
            views[view].append(summary)
            all_per_sample.extend(_per_sample_rows(system, view, selected))
    legacy_summary = _aggregate("legacy v5_6_a_rolefix", "S2-Overlap", legacy_results, truth_by_sample)
    all_per_class.extend(legacy_summary.pop("per_class_rows"))
    views["S2-Overlap"].append(legacy_summary)
    all_per_sample.extend(_per_sample_rows("legacy v5_6_a_rolefix", "S2-Overlap", legacy_results))
    views["Real-Dev-Full"].append({
        "system": "legacy v5_6_a_rolefix",
        "view": "Real-Dev-Full",
        "available": False,
        "reason": "frozen legacy artifacts cover only S2 and do not contain a reproducible full-manifest entry",
    })

    metrics = {
        "schema_version": "topocae_p1_d_lite_layered_comparison_1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": 20260720,
        "views": views,
        "sample_details": {
            "P1-B frozen": [_strip_stage_objects(record) for record in sample_results["p1_b"]],
            "P1-C frozen": [_strip_stage_objects(record) for record in sample_results["p1_c"]],
            "legacy v5_6_a_rolefix": [_strip_stage_objects(record) for record in legacy_results],
        },
        "contracts": {
            "truth_used_only_for_offline_evaluation": True,
            "exact_metrics_reported_only_where_exact_truth_exists": True,
            "different_views_not_compared_in_one_numeric_column": True,
            "historical_outputs_modified": False,
        },
    }
    write_json(LAYERED_ROOT / "metrics.json", metrics)
    write_csv(LAYERED_ROOT / "per_class.csv", all_per_class)
    write_csv(LAYERED_ROOT / "per_sample.csv", all_per_sample)
    report = _report(metrics)
    report_path = LAYERED_ROOT / "report.zh-CN.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(LAYERED_ROOT.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

