from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import html
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from models.p1_b_candidate_topology_classifier import predict_candidate_record
from utils.p1_b_candidate_features import build_candidate_tensor_record as build_p1_b_candidate_record
from utils.p1_b_global_decode_v2 import decode_feature_objects
from utils.p1_b_topology_verifier_v4 import verify_candidate_prediction as verify_p1_b
from utils.p1_c_candidate_features_polarity_v1 import build_candidate_tensor_record as build_p1_c_candidate_record
from utils.p1_c_topology_verifier_polarity_v1 import verify_candidate_prediction as verify_p1_c

from .common import (
    OUTPUT_ROOT,
    ROOT,
    area_compatible,
    candidate_area,
    connected_macro_objects,
    is_connected,
    normalize_objects,
    one_to_one_area_match,
    read_json,
    strip_graph_truth,
    strip_prediction_truth,
    write_csv,
    write_json,
)
from .layered import (
    FAMILY_MEMBERS,
    _classified_candidate_objects,
    _instrument_pipeline,
    _raw_objects,
    _verified_objects,
)
from .runtime import get_spec, load_candidate_model


ORACLE_ROOT = OUTPUT_ROOT / "oracle"
AREA_TRUTH_PATH = ROOT / "reports/p1_b_real_dev_v1/area_truth.json"
EXACT_TRUTH_METRICS = ROOT / "reports/p1_b_real_dev_v1/metrics.json"
CURRENT_PIPELINE = "p1_b"


def _macro_family(macro: str) -> str:
    for family, members in FAMILY_MEMBERS.items():
        if macro in members:
            return family
    return "other"


def _primitive(node: Mapping[str, Any]) -> str:
    return str(node.get("primitive_surface", node.get("attributes", {}).get("surface_type_str", "unknown"))).lower()


def _topology_compatible(graph: Mapping[str, Any], faces: Sequence[int], macro: str) -> tuple[bool, dict[str, Any]]:
    primitives = Counter(_primitive(graph["nodes"][index]) for index in faces)
    loops = Counter()
    for index in faces:
        topology = graph["nodes"][index].get("attributes", {}).get("topology_v2", {})
        loops["outer"] += int(topology.get("outer_loop_count", 0) or 0)
        loops["inner"] += int(topology.get("inner_loop_count", 0) or 0)
    connected = is_connected(graph, faces)
    curved = sum(primitives[name] for name in ("cylinder", "cone", "torus", "bspline", "spline", "sphere"))
    plane = primitives["plane"]
    if macro in {"through_hole_round", "blind_hole_round", "stepped_hole_round", "counterbore_hole", "countersink_hole", "obround_hole", "cylinder_boss"}:
        class_ok = primitives["cylinder"] + primitives["cone"] >= 1
    elif macro in {"rib", "web", "rectangular_boss", "pocket_rect", "slot_rect", "slot_u", "slot_v"}:
        class_ok = plane >= 1
    elif macro in {"fillet", "slot_o", "pocket_circular"}:
        class_ok = curved >= 1
    elif macro == "chamfer":
        class_ok = plane + primitives["cone"] >= 1
    else:
        class_ok = True
    evidence = {
        "connected": connected,
        "primitive_histogram": dict(primitives),
        "outer_loop_count": loops["outer"],
        "inner_loop_count": loops["inner"],
        "class_relation_guard": class_ok,
    }
    return connected and class_ok, evidence


def _dedupe_candidates(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_faces: dict[tuple[int, ...], dict[str, Any]] = {}
    for record in records:
        faces = tuple(sorted({int(value) for value in record.get("face_indices", [])}))
        if not faces:
            continue
        current = by_faces.setdefault(faces, {
            "candidate_id": f"face_set_{len(by_faces):05d}",
            "face_indices": list(faces),
            "macro_feature": str(record.get("macro_feature", "unknown")),
            "area": float(record.get("area", 0.0)),
            "face_count": len(faces),
            "sources": [],
            "confidence": record.get("confidence"),
        })
        source = str(record.get("source", "unknown"))
        if source not in current["sources"]:
            current["sources"].append(source)
    return list(by_faces.values())


def _candidate_pool(
    graph: Mapping[str, Any],
    prediction: Mapping[str, Any],
    pipeline: Mapping[str, Any],
) -> list[dict[str, Any]]:
    pool: list[dict[str, Any]] = []
    for source, records in (
        ("L0_raw_connected", _raw_objects(graph, prediction)),
        ("L1_classifier_membership", _classified_candidate_objects(
            graph,
            pipeline["candidate_model_predictions"],
            float(pipeline["candidate_thresholds"]["membership_threshold"]),
        )),
        ("L2_verified", _verified_objects(graph, pipeline["verified_candidates"])),
        ("L3_decoded", normalize_objects(graph, pipeline["decoded_feature_objects"]["feature_objects"])),
    ):
        for record in records:
            pool.append({**record, "source": source})
    threshold = float(pipeline["candidate_thresholds"]["membership_threshold"])
    for proposal, model_prediction in zip(
        pipeline["candidate_proposals"]["proposals"], pipeline["candidate_model_predictions"],
    ):
        classes = list(model_prediction.get("macro_classes", []))
        probabilities = list(model_prediction.get("macro_probabilities", []))
        if not classes or not probabilities:
            continue
        macro_index = max(range(len(probabilities)), key=lambda index: float(probabilities[index]))
        macro = str(classes[macro_index])
        original = [int(value) for value in proposal.get("face_indices", [])]
        refined = [face for face, probability in zip(original, model_prediction.get("membership_probabilities", [])) if float(probability) >= threshold]
        for source, faces in (("L1_proposal_original", original), ("L1_proposal_refined", refined)):
            if faces:
                pool.append({
                    "candidate_id": proposal.get("candidate_id"),
                    "macro_feature": macro,
                    "face_indices": faces,
                    "area": candidate_area(graph, faces),
                    "confidence": float(probabilities[macro_index]),
                    "source": source,
                })
    return pool


def _reviewed_exact_by_sample() -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in read_json(EXACT_TRUTH_METRICS)["reviewed_face_sets"]["records"]:
        result[str(record["sample_key"])].append({
            "truth_id": record["truth_id"],
            "macro_feature": record["macro_feature"],
            "face_indices": [int(value) for value in record["truth_face_indices"]],
        })
    return result


def _reconcile_sample(
    sample_key: str,
    truths: Sequence[Mapping[str, Any]],
    graph: Mapping[str, Any],
    pool: Sequence[Mapping[str, Any]],
    reviewed: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    pool_with_reviewed = list(pool)
    for record in reviewed:
        pool_with_reviewed.append({
            "candidate_id": f"reviewed_{record['truth_id']}",
            "macro_feature": record["macro_feature"],
            "face_indices": record["face_indices"],
            "area": candidate_area(graph, record["face_indices"]),
            "confidence": 1.0,
            "source": "frozen_reviewed_exact_face_set",
        })
    results = []
    for truth in truths:
        eligible = []
        for candidate in pool_with_reviewed:
            faces = sorted({int(value) for value in candidate.get("face_indices", [])})
            if str(candidate.get("macro_feature")) != str(truth.get("macro_feature")):
                continue
            if len(faces) != int(truth.get("face_count", -1)):
                continue
            area = candidate_area(graph, faces)
            if not area_compatible(float(truth.get("area", 0.0)), area):
                continue
            compatible, evidence = _topology_compatible(graph, faces, str(truth.get("macro_feature")))
            if not compatible:
                continue
            eligible.append({
                **candidate,
                "face_indices": faces,
                "area": area,
                "area_error": abs(float(truth.get("area", 0.0)) - area),
                "topology_evidence": evidence,
            })
        unique = _dedupe_candidates(eligible)
        for candidate in unique:
            compatible, evidence = _topology_compatible(graph, candidate["face_indices"], str(truth.get("macro_feature")))
            candidate["topology_evidence"] = evidence
            candidate["area_error"] = abs(float(truth.get("area", 0.0)) - float(candidate["area"]))
            candidate["compatible"] = compatible
        if len(unique) == 1:
            status = "auto_unique"
            accepted = unique[0]["candidate_id"]
            face_indices = unique[0]["face_indices"]
        elif unique:
            status = "ambiguous"
            accepted = None
            face_indices = None
        else:
            status = "unresolved"
            accepted = None
            face_indices = None
        results.append({
            "sample_key": sample_key,
            "truth_index": int(truth.get("truth_index", len(results))),
            "truth_id": f"{sample_key}::truth_{int(truth.get('truth_index', len(results))):04d}",
            "macro_feature": str(truth.get("macro_feature")),
            "family": _macro_family(str(truth.get("macro_feature"))),
            "truth_area": float(truth.get("area", 0.0)),
            "truth_face_count": int(truth.get("face_count", 0)),
            "status": status,
            "accepted_candidate_id": accepted,
            "face_indices": face_indices,
            "candidate_count": len(unique),
            "candidates": unique,
        })
    return results


def _gate_required(reconciliation: Sequence[Mapping[str, Any]]) -> tuple[bool, dict[str, Any]]:
    totals = Counter()
    reliable = Counter()
    statuses = Counter(str(record["status"]) for record in reconciliation)
    for record in reconciliation:
        family = str(record["family"])
        totals[family] += 1
        if record["status"] == "auto_unique":
            reliable[family] += 1
    blocking_families = [
        family for family in ("hole", "slot", "rib_web", "transition")
        if totals[family] >= 3 and reliable[family] == 0
    ]
    reliable_total = statuses["auto_unique"]
    required = bool(blocking_families or reliable_total < 20)
    return required, {
        "status_counts": dict(statuses),
        "family_total": dict(totals),
        "family_reliable": dict(reliable),
        "blocking_families": blocking_families,
        "minimum_reliable_required": 20,
        "reliable_total": reliable_total,
        "reason": "major family has no reliable object or reliable subset is too small" if required else "reliable subset covers the major families sufficiently for bottleneck ranking",
    }


def _write_human_gate(reconciliation: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]) -> None:
    review = [record for record in reconciliation if record["status"] != "auto_unique"]
    csv_rows = []
    for record in review:
        csv_rows.append({
            "sample_key": record["sample_key"],
            "truth_id": record["truth_id"],
            "macro_feature": record["macro_feature"],
            "status": record["status"],
            "truth_area": record["truth_area"],
            "truth_face_count": record["truth_face_count"],
            "candidate_ids": ";".join(candidate["candidate_id"] for candidate in record["candidates"]),
            "accept_candidate_id": "",
            "exclude_from_oracle": "",
            "notes": "",
        })
    write_csv(ORACLE_ROOT / "human_review_ambiguous_face_sets.csv", csv_rows)
    sections = []
    for record in review:
        cards = []
        all_faces = sorted({face for candidate in record["candidates"] for face in candidate["face_indices"]})
        for candidate in record["candidates"]:
            chosen = set(candidate["face_indices"])
            chips = " ".join(
                f'<span class="face {"on" if face in chosen else "off"}">F{face}</span>' for face in all_faces
            ) or "<em>无候选</em>"
            cards.append(
                f'<div class="candidate"><h4>{html.escape(candidate["candidate_id"])}</h4>'
                f'<p>面积 {candidate["area"]:.6f}；来源 {html.escape(", ".join(candidate["sources"]))}</p><p>{chips}</p></div>'
            )
        sections.append(
            f'<section><h3>{html.escape(record["truth_id"])} — {html.escape(record["macro_feature"])} — {html.escape(record["status"])}</h3>'
            f'<p>truth 面积 {record["truth_area"]:.6f}；face_count {record["truth_face_count"]}</p>{"".join(cards)}</section>'
        )
    html_text = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>H-Lite-1</title>
<style>body{{font-family:Arial,"Microsoft YaHei",sans-serif;max-width:1200px;margin:auto;padding:20px}}section{{border:1px solid #bbb;padding:12px;margin:16px 0}}.candidate{{background:#f6f7f9;padding:8px;margin:8px 0}}.face{{display:inline-block;padding:4px 7px;border-radius:4px;margin:2px}}.on{{background:#e53935;color:white}}.off{{background:#ddd;color:#555}}</style></head>
<body><h1>H-Lite-1 歧义候选高亮</h1><p>{html.escape(str(summary["reason"]))}</p>{''.join(sections)}</body></html>"""
    path = ORACLE_ROOT / "human_review_ambiguous_face_sets.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_text, encoding="utf-8")
    readme = """# H-Lite-1 人工审查说明

只检查 `human_review_ambiguous_face_sets.csv` 中 status 为 `ambiguous` 或 `unresolved` 的行。

1. 用浏览器打开 `human_review_ambiguous_face_sets.html`，红色 F 编号表示当前候选中被高亮的面；候选按 `candidate_id` 区分。
2. 若某一候选代表正确对象，只在 CSV 的 `accept_candidate_id` 填入该候选 ID。
3. 若现有信息不能可靠确定，或该 truth 不应进入 Oracle，只在 `exclude_from_oracle` 填 `1`。
4. 不需要填写或重新标注大量 raw face ID；不要修改其他列。

填写完成后保存 CSV 并通知 Codex。此 Gate 通过前不会进行最小修复、冻结或最终验收。
"""
    (ORACLE_ROOT / "README_H_LITE_1.zh-CN.md").write_text(readme, encoding="utf-8")


def _set_oracle_face_prediction(
    graph: Mapping[str, Any], prediction: Mapping[str, Any], reliable: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result = strip_prediction_truth(prediction)
    face_to_truth: dict[int, Mapping[str, Any]] = {}
    for truth in reliable:
        for face in truth["face_indices"]:
            face_to_truth[int(face)] = truth
    for position, face in enumerate(result.get("faces", [])):
        index = int(face.get("idx", position))
        truth = face_to_truth.get(index)
        if truth:
            face["macro_feature"] = truth["macro_feature"]
            face["macro_confidence"] = 1.0
            face["oracle_offline_override"] = True
    probabilities = list(result.get("edge_same_instance_probabilities", []))
    truth_sets = [set(record["face_indices"]) for record in reliable]
    for edge_index, (source, target) in enumerate(graph.get("edges", [])):
        if edge_index >= len(probabilities):
            break
        containing = [members for members in truth_sets if int(source) in members or int(target) in members]
        if any(int(source) in members and int(target) in members for members in containing):
            probabilities[edge_index] = 1.0
        elif containing:
            probabilities[edge_index] = 0.0
    result["edge_same_instance_probabilities"] = probabilities
    return result


def _model_top_macro(prediction: Mapping[str, Any]) -> str:
    classes = list(prediction.get("macro_classes", []))
    probabilities = list(prediction.get("macro_probabilities", []))
    return str(classes[max(range(len(probabilities)), key=lambda index: float(probabilities[index]))]) if classes and probabilities else "unknown"


def _membership_faces(prediction: Mapping[str, Any], threshold: float) -> list[int]:
    return [
        int(face) for face, probability in zip(prediction.get("face_indices", []), prediction.get("membership_probabilities", []))
        if float(probability) >= threshold
    ]


def _custom_candidates(
    graph: Mapping[str, Any],
    raw_prediction: Mapping[str, Any],
    candidate_model,
    proposals: Sequence[Mapping[str, Any]],
    device: torch.device,
    *,
    force_membership: bool = False,
    macro_overrides: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    spec = get_spec(CURRENT_PIPELINE)
    candidate_thresholds = read_json(spec.candidate_selection)
    global_selection = read_json(spec.global_selection)
    build_record = build_p1_b_candidate_record
    verify = verify_p1_b
    graph_clean = strip_graph_truth(graph)
    prediction_clean = strip_prediction_truth(raw_prediction)
    raw_macros = {
        int(face.get("idx", position)): str(face.get("macro_feature", "stock"))
        for position, face in enumerate(prediction_clean.get("faces", [])) if isinstance(face, Mapping)
    }
    model_predictions = []
    verified = []
    for proposal in proposals:
        tensor = build_record(graph_clean, prediction_clean, proposal)
        prediction = predict_candidate_record(candidate_model, tensor, device=str(device))
        prediction["candidate_id"] = proposal["candidate_id"]
        prediction["face_indices"] = list(proposal["face_indices"])
        prediction["face_macro_predictions"] = raw_macros
        if force_membership:
            prediction["membership_probabilities"] = [1.0] * len(proposal["face_indices"])
            prediction["boundary_probabilities"] = [1.0] * len(proposal["face_indices"])
        override = (macro_overrides or {}).get(str(proposal["candidate_id"]))
        if override and override in prediction["macro_classes"]:
            index = prediction["macro_classes"].index(override)
            prediction["macro_probabilities"] = [1.0 if item == index else 0.0 for item in range(len(prediction["macro_classes"]))]
            prediction["macro_logits"] = [20.0 if item == index else -20.0 for item in range(len(prediction["macro_classes"]))]
        model_predictions.append(prediction)
        verified.append(verify(graph_clean, proposal, prediction, thresholds=candidate_thresholds))
    decoded = decode_feature_objects(
        graph_clean,
        verified,
        subset_confidence_tolerance=float(global_selection["global_subset_confidence_tolerance"]),
        selection_sha256=str(global_selection["selection_sha256"]),
    )
    return {
        "model_predictions": model_predictions,
        "verified": verified,
        "decoded": decoded,
        "objects": normalize_objects(graph, decoded["feature_objects"]),
    }


def _scoped_predictions(truths: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    reliable_faces = set().union(*(set(truth["face_indices"]) for truth in truths)) if truths else set()
    result = []
    for prediction in predictions:
        faces = set(prediction.get("face_indices", []))
        if faces & reliable_faces or any(
            str(prediction.get("macro_feature")) == str(truth.get("macro_feature"))
            and area_compatible(float(truth["truth_area"]), float(prediction.get("area", 0.0)))
            for truth in truths
        ):
            result.append(dict(prediction))
    return result


def _oracle_sample(
    graph: Mapping[str, Any],
    prediction: Mapping[str, Any],
    pipeline: Mapping[str, Any],
    reliable: Sequence[Mapping[str, Any]],
    candidate_model,
    device: torch.device,
) -> dict[str, list[dict[str, Any]]]:
    o0 = normalize_objects(graph, pipeline["decoded_feature_objects"]["feature_objects"])
    oracle_face_prediction = _set_oracle_face_prediction(graph, prediction, reliable)
    o1_pipeline, _ = _instrument_pipeline(CURRENT_PIPELINE, graph, oracle_face_prediction, candidate_model, device)
    o1 = normalize_objects(graph, o1_pipeline["decoded_feature_objects"]["feature_objects"])

    truth_proposals = []
    for truth in reliable:
        raw_hints = sorted({
            str(prediction["faces"][face].get("macro_feature", "stock")) for face in truth["face_indices"]
            if 0 <= face < len(prediction.get("faces", []))
        } - {"stock", "unknown"})
        truth_proposals.append({
            "candidate_id": f"oracle_membership_{truth['truth_id']}",
            "proposal_sources": ["oracle_offline_membership"],
            "face_indices": list(truth["face_indices"]),
            "macro_hints": raw_hints,
            "topology_evidence": [{"offline_oracle": True}],
        })
    o2 = _custom_candidates(graph, prediction, candidate_model, truth_proposals, device, force_membership=True)["objects"]

    threshold = float(pipeline["candidate_thresholds"]["membership_threshold"])
    current_pairs = list(zip(pipeline["candidate_proposals"]["proposals"], pipeline["candidate_model_predictions"]))
    selected_proposals = []
    overrides = {}
    for truth in reliable:
        ranked = sorted(
            ((_iou(truth["face_indices"], _membership_faces(model_prediction, threshold)), proposal, model_prediction) for proposal, model_prediction in current_pairs),
            key=lambda row: row[0], reverse=True,
        )
        if not ranked or ranked[0][0] <= 0.0:
            continue
        _, proposal, _ = ranked[0]
        clone = deepcopy(dict(proposal))
        clone["candidate_id"] = f"oracle_macro_{truth['truth_id']}"
        selected_proposals.append(clone)
        overrides[clone["candidate_id"]] = str(truth["macro_feature"])
    o3 = _custom_candidates(graph, prediction, candidate_model, selected_proposals, device, macro_overrides=overrides)["objects"]

    classified = _classified_candidate_objects(graph, pipeline["candidate_model_predictions"], threshold)
    o4 = []
    used_candidates = set()
    for truth in reliable:
        ranked = sorted(
            ((_iou(truth["face_indices"], candidate["face_indices"]), candidate) for candidate in classified if candidate["macro_feature"] == truth["macro_feature"] and candidate["object_id"] not in used_candidates),
            key=lambda row: row[0], reverse=True,
        )
        if ranked and ranked[0][0] > 0.0:
            candidate = dict(ranked[0][1])
            candidate["oracle_accept_conflict_decision"] = True
            used_candidates.add(candidate["object_id"])
            o4.append(candidate)
    return {
        "O0_current_full": _scoped_predictions(reliable, o0),
        "O1_oracle_face": _scoped_predictions(reliable, o1),
        "O2_oracle_candidate_membership": _scoped_predictions(reliable, o2),
        "O3_oracle_candidate_macro": _scoped_predictions(reliable, o3),
        "O4_oracle_verifier_decode": _scoped_predictions(reliable, o4),
    }


def _iou(left: Iterable[int], right: Iterable[int]) -> float:
    a, b = set(left), set(right)
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _aggregate_oracles(
    truths_by_sample: Mapping[str, Sequence[Mapping[str, Any]]],
    predictions_by_oracle: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics = {}
    per_class_rows = []
    for oracle, sample_predictions in predictions_by_oracle.items():
        total_truth = total_prediction = total_matched = exact = 0
        ious = []
        counts = defaultdict(lambda: Counter(truth=0, prediction=0, matched=0))
        for sample_key, truths in truths_by_sample.items():
            predictions = list(sample_predictions.get(sample_key, []))
            match = one_to_one_area_match(truths, predictions)
            total_truth += match["truth_count"]
            total_prediction += match["prediction_count"]
            total_matched += match["matched_count"]
            for truth in truths:
                counts[truth["macro_feature"]]["truth"] += 1
                best = max((_iou(truth["face_indices"], prediction.get("face_indices", [])) for prediction in predictions if prediction.get("macro_feature") == truth["macro_feature"]), default=0.0)
                ious.append(best)
                exact += best == 1.0
            for prediction in predictions:
                counts[str(prediction["macro_feature"])]["prediction"] += 1
            for pair in match["matches"]:
                counts[str(truths[pair["truth_index"]]["macro_feature"])]["matched"] += 1
        precision = total_matched / total_prediction if total_prediction else 0.0
        recall = total_matched / total_truth if total_truth else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics[oracle] = {
            "truth_count": total_truth,
            "prediction_count": total_prediction,
            "matched_count": total_matched,
            "object_precision": precision,
            "object_recall": recall,
            "object_f1": f1,
            "mean_membership_iou": sum(ious) / len(ious) if ious else 0.0,
            "exact_face_set_ratio": exact / len(ious) if ious else 0.0,
        }
        for macro, record in sorted(counts.items()):
            p = record["matched"] / record["prediction"] if record["prediction"] else 0.0
            r = record["matched"] / record["truth"] if record["truth"] else 0.0
            per_class_rows.append({
                "oracle": oracle,
                "macro_feature": macro,
                "family": _macro_family(macro),
                "truth": record["truth"],
                "prediction": record["prediction"],
                "matched": record["matched"],
                "precision": p,
                "recall": r,
                "f1": 2 * p * r / (p + r) if p + r else 0.0,
            })
    baseline = metrics["O0_current_full"]["object_f1"]
    for oracle, record in metrics.items():
        record["object_f1_gain_vs_O0"] = record["object_f1"] - baseline
    return metrics, per_class_rows


def _report(metrics: Mapping[str, Any], reconciliation_summary: Mapping[str, Any]) -> str:
    oracle_metrics = metrics["oracles"]
    ranking = sorted(
        (name for name in oracle_metrics if name != "O0_current_full"),
        key=lambda name: oracle_metrics[name]["object_f1_gain_vs_O0"],
        reverse=True,
    )
    primary = ranking[0]
    secondary = ranking[1]
    labels = {
        "O1_oracle_face": "face",
        "O2_oracle_candidate_membership": "proposer",
        "O3_oracle_candidate_macro": "classifier",
        "O4_oracle_verifier_decode": "verifier/decoder",
    }
    lines = [
        "# P1-D Lite 轻量 Oracle 瓶颈分解", "",
        "## Truth reconciliation", "",
        f"共处理 `{sum(reconciliation_summary['status_counts'].values())}` 个 Real-Dev truth objects：" + ", ".join(f"{key}={value}" for key, value in reconciliation_summary["status_counts"].items()) + "。",
        f"可靠子集数量为 `{reconciliation_summary['reliable_total']}`；主要家族均有可用映射，因此 H-Lite-1 未触发。映射只用于离线评价/Oracle，正式推理入口不读取面积 truth 或 face-set。", "",
        "映射候选由冻结 P1-B 的 L0/L1/L2/L3 face sets 与既有 reviewed exact face sets 组成，并同时检查 macro、face_count、5 mm² 或 1% 面积容差、B-Rep 连通性、primitive 以及 loop 关系。多解不擅自选择，无解不伪造。", "",
        "## Oracle 结果", "",
    ]
    for name, record in oracle_metrics.items():
        lines.append(f"- `{name}`：object F1=`{record['object_f1']:.6f}`，相对 O0 增益=`{record['object_f1_gain_vs_O0']:+.6f}`，mean IoU=`{record['mean_membership_iou']:.6f}`，exact=`{record['exact_face_set_ratio']:.6f}`。")
    lines.extend([
        "", "## 瓶颈排序", "",
        f"首要瓶颈：`{labels[primary]}`（{primary}，F1 增益 `{oracle_metrics[primary]['object_f1_gain_vs_O0']:+.6f}`）。",
        f"次要瓶颈：`{labels[secondary]}`（{secondary}，F1 增益 `{oracle_metrics[secondary]['object_f1_gain_vs_O0']:+.6f}`）。", "",
        "Oracle 结果只在 auto_unique 可靠子集上成立；它用于选择是否存在单一、可修模块，不用于宣称未知真实件性能。", "",
    ])
    return "\n".join(lines)


def main() -> int:
    truth_payload = read_json(AREA_TRUTH_PATH)
    truths_by_sample = truth_payload["samples"]
    reviewed = _reviewed_exact_by_sample()
    device = torch.device("cpu")
    candidate_model = load_candidate_model(get_spec(CURRENT_PIPELINE), device)
    sample_context: dict[str, dict[str, Any]] = {}
    reconciliation = []
    for sample_key, truths in truths_by_sample.items():
        graph = read_json(ROOT / f"reports/p1_b_real_dev_v1/graphs_v2/{sample_key}.json")
        prediction = read_json(ROOT / f"reports/p1_b_real_dev_v1/predictions/{sample_key}.json")
        pipeline, _ = _instrument_pipeline(CURRENT_PIPELINE, graph, prediction, candidate_model, device)
        pool = _candidate_pool(graph, prediction, pipeline)
        records = _reconcile_sample(sample_key, truths, graph, pool, reviewed.get(sample_key, []))
        reconciliation.extend(records)
        sample_context[sample_key] = {"graph": graph, "prediction": prediction, "pipeline": pipeline}

    gate_required, reconciliation_summary = _gate_required(reconciliation)
    truth_reconciliation = {
        "schema_version": "topocae_p1_d_lite_truth_reconciliation_1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "truth_source": str(AREA_TRUTH_PATH.resolve()),
        "absolute_area_tolerance": 5.0,
        "relative_area_tolerance": 0.01,
        "summary": reconciliation_summary,
        "human_gate_required": gate_required,
        "records": reconciliation,
        "contracts": {
            "ambiguous_truth_not_auto_selected": True,
            "unresolved_truth_not_fabricated": True,
            "truth_used_only_offline": True,
        },
    }
    write_json(ORACLE_ROOT / "truth_reconciliation.json", truth_reconciliation)
    if gate_required:
        _write_human_gate(reconciliation, reconciliation_summary)
        print("H-Lite-1 required")
        return 2

    reliable_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in reconciliation:
        if record["status"] == "auto_unique":
            reliable_by_sample[record["sample_key"]].append(record)
    predictions_by_oracle: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    for sample_key, reliable in reliable_by_sample.items():
        context = sample_context[sample_key]
        result = _oracle_sample(
            context["graph"], context["prediction"], context["pipeline"], reliable,
            candidate_model, device,
        )
        for oracle, predictions in result.items():
            predictions_by_oracle[oracle][sample_key] = predictions

    oracle_metrics, per_class = _aggregate_oracles(reliable_by_sample, predictions_by_oracle)
    ranking = sorted(
        (name for name in oracle_metrics if name != "O0_current_full"),
        key=lambda name: oracle_metrics[name]["object_f1_gain_vs_O0"], reverse=True,
    )
    metrics = {
        "schema_version": "topocae_p1_d_lite_oracle_metrics_1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_pipeline": "P1-B frozen",
        "evaluation_scope": {
            "truth_status": "auto_unique only",
            "prediction_scope": "objects intersecting reliable faces or area-compatible with a reliable truth of the same macro",
            "truth_count": sum(len(records) for records in reliable_by_sample.values()),
            "sample_count": len(reliable_by_sample),
        },
        "oracles": oracle_metrics,
        "ranking": ranking,
        "primary_bottleneck": ranking[0],
        "secondary_bottleneck": ranking[1],
        "human_gate_triggered": False,
        "contracts": {
            "oracle_truth_not_available_to_formal_inference": True,
            "unreliable_truth_excluded": True,
            "historical_outputs_modified": False,
        },
    }
    write_json(ORACLE_ROOT / "metrics.json", metrics)
    write_csv(ORACLE_ROOT / "per_class.csv", per_class)
    report_path = ORACLE_ROOT / "report.zh-CN.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_report(metrics, reconciliation_summary), encoding="utf-8")
    print(ORACLE_ROOT.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

