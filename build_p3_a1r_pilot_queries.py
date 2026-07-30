from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from utils.cae_intent import validate_intent_result
from utils.cae_query_ground_p3_a1r.semantic_coverage import audit_records


ROOT = Path(__file__).resolve().parent
DEFAULT_GEOMETRY_ROOT = ROOT / "output_generation_p3_a1r_multi_instance_v1"
DEFAULT_REPORT_ROOT = ROOT / "reports" / "p3_a1r_multi_instance_v1"
QUANTIFIERS = ("all", "count", "largest", "smallest", "nearest")
UNIT_SYSTEM = {
    "length": "mm",
    "area": "mm2",
    "force": "N",
    "pressure": "MPa",
    "displacement": "mm",
    "angle": "deg",
}
FEATURE_LABELS = {
    "through_hole_round": ("round through hole", "round through holes", "圆形通孔"),
    "blind_hole_round": ("round blind hole", "round blind holes", "圆形盲孔"),
    "stepped_hole_round": ("stepped round hole", "stepped round holes", "圆形阶梯孔"),
    "counterbore_hole": ("counterbore hole", "counterbore holes", "沉孔"),
    "countersink_hole": ("countersink hole", "countersink holes", "锥沉孔"),
    "obround_hole": ("obround hole", "obround holes", "长圆孔"),
    "slot_rect": ("rectangular slot", "rectangular slots", "矩形槽"),
    "slot_u": ("U-shaped slot", "U-shaped slots", "U 形槽"),
    "slot_v": ("V-shaped slot", "V-shaped slots", "V 形槽"),
    "slot_o": ("O-shaped slot", "O-shaped slots", "O 形槽"),
    "pocket_rect": ("rectangular pocket", "rectangular pockets", "矩形口袋"),
    "pocket_circular": ("circular pocket", "circular pockets", "圆形口袋"),
    "cylinder_boss": ("cylindrical boss", "cylindrical bosses", "圆柱凸台"),
    "rectangular_boss": ("rectangular boss", "rectangular bosses", "矩形凸台"),
    "rib": ("rib", "ribs", "加强筋"),
    "fillet": ("fillet", "fillets", "圆角"),
    "chamfer": ("chamfer", "chamfers", "倒角"),
}
METRIC_LABELS = {
    "diameter": ("diameter", "直径"),
    "radius": ("radius", "半径"),
    "depth": ("depth", "深度"),
    "area": ("area", "面积"),
    "length": ("length", "长度"),
    "width": ("width", "宽度"),
    "thickness": ("thickness", "厚度"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().lower()


def stable_id(prefix: str, value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"{prefix}.{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_queries(geometry_root: Path, report_root: Path) -> dict[str, Any]:
    geometry_root = geometry_root.resolve()
    report_root = report_root.resolve()
    query_root = geometry_root / "queries"
    query_root.mkdir(parents=True, exist_ok=True)
    review_root = report_root / "human_review_package"
    review_root.mkdir(parents=True, exist_ok=True)
    samples = read_jsonl(geometry_root / "manifests" / "sample_manifest.jsonl")
    canonical = []
    renderings = []
    for sample in samples:
        graph = json.loads(Path(sample["graph_path"]).read_text(encoding="utf-8"))
        for quantifier in QUANTIFIERS:
            record, variants = _build_record(sample, graph, quantifier)
            canonical.append(record)
            renderings.extend(variants)

    audit = _audit_queries(canonical)
    if not audit["automatic_pass"]:
        raise RuntimeError(f"P3-A1R pilot query audit failed: {audit['failure_examples'][:5]}")
    canonical_path = query_root / "canonical.jsonl"
    renderings_path = query_root / "renderings.jsonl"
    canonical_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n" for item in canonical),
        encoding="utf-8",
        newline="\n",
    )
    renderings_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n" for item in renderings),
        encoding="utf-8",
        newline="\n",
    )
    audit["canonical_sha256"] = sha256_file(canonical_path)
    audit["renderings_sha256"] = sha256_file(renderings_path)
    audit_path = geometry_root / "reports" / "query_semantic_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    geometry_review = _write_geometry_review(review_root / "geometry_review_20.csv", samples)
    query_review = _write_query_review(review_root / "query_review_60.csv", canonical, renderings)
    _write_review_readme(review_root / "README_H3R_01.zh-CN.md", geometry_root, geometry_review, query_review)
    return {
        "schema_version": "p3-a1r-pilot-query-build/1",
        "status": "AWAITING_H3R_01",
        "canonical_records": len(canonical),
        "renderings": len(renderings),
        "query_audit": audit,
        "geometry_review_rows": geometry_review,
        "query_review_rows": query_review,
        "scope_guards": {
            "manual_templates_only": True,
            "llm_called": False,
            "model_trained": False,
            "solver_executed": False,
            "p4_entered": False,
            "formal_scale_generated": False,
            "forbidden_data_accessed": False,
        },
    }


def _build_record(
    sample: dict[str, Any], graph: dict[str, Any], quantifier: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    feature_type = str(sample["feature_type"])
    candidates = sorted(graph["feature_sets"], key=lambda item: int(item["instance_id"]))
    metric_rows = sorted(graph["metric_table"], key=lambda item: int(item["instance_id"]))
    metric_by_id = {int(item["instance_id"]): item for item in metric_rows}
    candidate_ids = [int(item["instance_id"]) for item in candidates]
    comparison_values = [float(item["comparison_value"]) for item in metric_rows]
    nearest_distances = [float(item["nearest_distance"]) for item in metric_rows]
    comparison_scale = max(max(abs(value) for value in comparison_values), 1e-9)
    comparison_gaps = [
        right - left
        for left, right in zip(sorted(comparison_values), sorted(comparison_values)[1:])
    ]
    nearest_gaps = [
        right - left
        for left, right in zip(sorted(nearest_distances), sorted(nearest_distances)[1:])
    ]
    part_scale = float(graph["part_metadata_v2"]["normalization"]["part_bbox_diagonal"])
    comparison_stable = min(comparison_gaps) > max(1e-6, comparison_scale * 0.015)
    nearest_stable = min(nearest_gaps) > max(0.25, part_scale * 0.002)
    tie = (
        quantifier in {"largest", "smallest"} and not comparison_stable
        or quantifier == "nearest" and not nearest_stable
    )
    if quantifier in {"all", "count"}:
        selected_ids = candidate_ids
    elif quantifier == "largest":
        selected_ids = [max(metric_rows, key=lambda item: float(item["comparison_value"]))["instance_id"]]
    elif quantifier == "smallest":
        selected_ids = [min(metric_rows, key=lambda item: float(item["comparison_value"]))["instance_id"]]
    else:
        selected_ids = [min(metric_rows, key=lambda item: float(item["nearest_distance"]))["instance_id"]]
    selected_ids = [int(value) for value in selected_ids]
    texts = _render_texts(feature_type, quantifier, len(candidates), str(sample["comparison_metric"]), str(sample["split"]))
    semantic_key = {
        "sample_id": sample["sample_id"],
        "feature_type": feature_type,
        "quantifier": quantifier,
        "tie": tie,
        "selected_ids": [] if tie else selected_ids,
    }
    canonical_id = stable_id("P3A1R-Q", semantic_key)
    if tie:
        intent_result = _clarification_result(quantifier, str(sample["comparison_metric"]))
        ground_truth = {
            "contract_status": "non_executable",
            "geometry_source": "oracle",
            "intent_status": "NEEDS_CLARIFICATION",
            "non_executable_reason": "The requested ordering is tied or lacks the required stable geometric margin.",
            "targets": [],
            "clarification_evidence": {
                "candidate_instance_ids": candidate_ids,
                "comparison_metric": sample["comparison_metric"],
                "comparison_values": [float(item["comparison_value"]) for item in metric_rows],
                "nearest_distances": [float(item["nearest_distance"]) for item in metric_rows],
                "tie_verified_from_graph": True,
            },
        }
    else:
        intent_result = _valid_intent_result(feature_type, quantifier, len(candidates), str(sample["comparison_metric"]), "en-US")
        selected = [item for item in candidates if int(item["instance_id"]) in selected_ids]
        target_faces = sorted({int(index) for item in selected for index in item["node_indices"]})
        ground_truth = {
            "contract_status": "executable",
            "geometry_source": "oracle",
            "intent_status": "VALID",
            "targets": [{
                "canonical_feature_type": feature_type,
                "face_role": "feature_region",
                "grounding_required": True,
                "operation_id": "op1",
                "operation_type": "local_mesh_size",
                "quantifier": quantifier,
                "target_contract": "feature_face_set",
                "target_instance_ids": selected_ids,
                "target_face_indices": target_faces,
                "source_macro_features": [feature_type],
                "semantic_evidence": [
                    {
                        "instance_id": int(item["instance_id"]),
                        "instance_face_indices": sorted(int(value) for value in item["node_indices"]),
                        "selected_face_indices": sorted(int(value) for value in item["node_indices"]),
                        "observed_semantic_roles": list(item["semantic_roles"]),
                        "source_macro_feature": feature_type,
                    }
                    for item in selected
                ],
                "uniqueness_contract": {
                    "kind": quantifier,
                    "candidate_count": len(candidates),
                    "selected_count": len(selected_ids),
                    "count": len(candidates) if quantifier == "count" else None,
                    "comparison_metric": sample["comparison_metric"] if quantifier in {"largest", "smallest"} else None,
                    "nearest_reference": "part_centroid" if quantifier == "nearest" else None,
                    "candidate_metric_evidence": [metric_by_id[instance_id] for instance_id in candidate_ids],
                    "stable_gap_verified": quantifier in {"largest", "smallest", "nearest"},
                },
            }],
        }
    record = {
        "record_schema_version": "p3-a1r-pilot-query-record/1",
        "record_kind": "canonical",
        "source_ref": {
            "canonical_record_id": canonical_id,
            "sample_id": sample["sample_id"],
            "dataset_split": sample["split"],
            "base_geometry_family_id": sample["base_geometry_family_id"],
            "graph_path": sample["graph_path"],
            "graph_sha256": sample["graph_sha256"],
            "geometry_path": sample["step_path"],
            "geometry_sha256": sample["step_sha256"],
            "exact_graph_signature": sample["exact_graph_signature"],
            "near_graph_signature": sample["near_graph_signature"],
        },
        "query": {
            "difficulty_level": "L2" if quantifier in {"all", "count"} else "L4",
            "language": "en-US",
            "rendering_id": f"{canonical_id}.en1",
            "rendering_variant": "en-1",
            "template_family_id": f"p3a1r.{sample['split']}.{quantifier}.en1",
            "template_id": f"p3a1r.{sample['split']}.{quantifier}.en1.v1",
            "text": texts["en-1"],
        },
        "intent_result": intent_result,
        "ground_truth": ground_truth,
    }
    variants = []
    for variant, language in (("en-1", "en-US"), ("en-2", "en-US"), ("zh-1", "zh-CN")):
        rendered_result = deepcopy(intent_result)
        if rendered_result["status"] == "VALID":
            rendered_result["intent"]["source_language"] = language
        variants.append({
            "record_schema_version": "p3-a1r-pilot-query-rendering/1",
            "canonical_record_id": canonical_id,
            "rendering_id": f"{canonical_id}.{variant}",
            "language": language,
            "rendering_variant": variant,
            "template_family_id": f"p3a1r.{sample['split']}.{quantifier}.{variant}",
            "text": texts[variant],
            "intent_result": rendered_result,
        })
    return record, variants


def _valid_intent_result(feature_type: str, quantifier: str, count: int, metric: str, language: str) -> dict[str, Any]:
    target: dict[str, Any] = {
        "feature_type": feature_type,
        "quantifier": quantifier,
        "face_role": "feature_region",
        "filters": {},
    }
    if quantifier == "count":
        target["count"] = count
    elif quantifier in {"largest", "smallest"}:
        target["comparison_metric"] = metric
    elif quantifier == "nearest":
        target["filters"] = {"nearest_to": {"reference_type": "part_centroid"}}
        target["reference_frame"] = {"kind": "part_local_cartesian"}
    return {
        "schema_version": "cae-intent-result/1.0.0",
        "status": "VALID",
        "intent": {
            "schema_version": "cae-intent/1.0.0",
            "analysis_type": "static_structural",
            "operations": [{
                "operation_id": "op1",
                "operation_type": "local_mesh_size",
                "target": target,
                "parameters": {"size": {"dimension": "length", "value": 0.8, "unit": "mm"}},
            }],
            "source_language": language,
            "assumptions": [],
            "unit_system": deepcopy(UNIT_SYSTEM),
            "normalization_audit": [],
        },
    }


def _clarification_result(quantifier: str, metric: str) -> dict[str, Any]:
    field = "/operations/0/target/filters/nearest_to" if quantifier == "nearest" else "/operations/0/target/comparison_metric"
    description = (
        "Two or more candidates are equally near the part centroid within the stable tolerance."
        if quantifier == "nearest"
        else f"Two or more candidates are tied, or insufficiently separated, in {metric} under the stable tolerance."
    )
    return {
        "schema_version": "cae-intent-result/1.0.0",
        "status": "NEEDS_CLARIFICATION",
        "error_code": "E_AMBIGUOUS_REQUIREMENT",
        "field_path": field,
        "missing_information": [],
        "ambiguities": [{
            "code": "A_TARGET_INSTANCE",
            "field_path": field,
            "description": description,
            "options": ["add a positional qualifier", "add another geometric filter"],
        }],
        "clarification_questions": [{
            "question_id": "q1",
            "field_path": field,
            "question": "The requested comparison is tied or insufficiently separated. Which positional or geometric qualifier should resolve it?",
        }],
    }


def _render_texts(feature_type: str, quantifier: str, count: int, metric: str, split: str) -> dict[str, str]:
    singular, plural, chinese = FEATURE_LABELS[feature_type]
    metric_en, metric_zh = METRIC_LABELS[metric]
    if quantifier == "all":
        request1 = f"apply a 0.8 mm local mesh to the complete feature regions of all {plural}"
        request2 = f"locally refine every {singular} as a complete feature region using a 0.8 mm mesh"
        request_zh = f"对零件中所有{chinese}的完整特征区域施加 0.8 mm 局部网格尺寸"
    elif quantifier == "count":
        request1 = f"apply a 0.8 mm local mesh to the complete feature regions of the {count} {plural} on this part"
        request2 = f"locally refine all {count} detected {plural}, using their complete feature regions and a 0.8 mm mesh"
        request_zh = f"对该零件中 {count} 个{chinese}的完整特征区域施加 0.8 mm 局部网格尺寸"
    elif quantifier in {"largest", "smallest"}:
        adjective = "largest" if quantifier == "largest" else "smallest"
        zh_adjective = "最大" if quantifier == "largest" else "最小"
        request1 = f"apply a 0.8 mm local mesh to the complete feature region of the {singular} with the {adjective} {metric_en}"
        request2 = f"locally refine the {adjective}-{metric_en} {singular} as a complete feature region using a 0.8 mm mesh"
        request_zh = f"对{metric_zh}{zh_adjective}的{chinese}完整特征区域施加 0.8 mm 局部网格尺寸"
    else:
        request1 = f"apply a 0.8 mm local mesh to the complete feature region of the {singular} nearest to the part centroid"
        request2 = f"locally refine the {singular} closest to the part centroid as a complete feature region using a 0.8 mm mesh"
        request_zh = f"对最靠近零件质心的{chinese}完整特征区域施加 0.8 mm 局部网格尺寸"
    wrappers = {
        "train": ("Please {request}.", "The preprocessing task is to {request}.", "请{request}。"),
        "val": ("Configure the part to {request}.", "The required CAE setup shall {request}.", "按以下要求配置零件：{request}。"),
        "test": ("Create a CAE preprocessing definition that will {request}.", "The CAE model must {request}.", "CAE 模型必须{request}。"),
    }
    en1, en2, zh1 = wrappers[split]
    return {
        "en-1": en1.format(request=request1),
        "en-2": en2.format(request=request2),
        "zh-1": zh1.format(request=request_zh),
    }


def _audit_queries(records: list[dict[str, Any]]) -> dict[str, Any]:
    failures = []
    reconstruction_passed = 0
    metric_passed = 0
    executable_l4 = Counter()
    statuses = Counter()
    quantifiers = Counter()
    features = Counter()
    splits = Counter()
    identity_pattern = re.compile(r"(?:p3a1r_|face[_ -]?id|instance[_ -]?id|\\graphs\\|\\steps\\)", re.IGNORECASE)
    for record in records:
        record_id = record["source_ref"]["canonical_record_id"]
        validation = validate_intent_result(record["intent_result"])
        if not validation["valid"]:
            failures.append({"record_id": record_id, "reason": "p2_intent_result_invalid", "details": validation})
            continue
        if identity_pattern.search(record["query"]["text"]):
            failures.append({"record_id": record_id, "reason": "identity_in_query"})
            continue
        graph = json.loads(Path(record["source_ref"]["graph_path"]).read_text(encoding="utf-8"))
        quantifier = _record_quantifier(record)
        quantifiers[quantifier] += 1
        statuses[record["intent_result"]["status"]] += 1
        features[record["source_ref"]["sample_id"].split("_", 3)[2] if False else graph["feature_sets"][0]["macro_feature"]] += 1
        splits[record["source_ref"]["dataset_split"]] += 1
        if record["intent_result"]["status"] != "VALID":
            evidence = record["ground_truth"]["clarification_evidence"]
            comparison_values = sorted(float(value) for value in evidence["comparison_values"])
            nearest_distances = sorted(float(value) for value in evidence["nearest_distances"])
            comparison_scale = max(max(abs(value) for value in comparison_values), 1e-9)
            part_scale = float(graph["part_metadata_v2"]["normalization"]["part_bbox_diagonal"])
            comparison_stable = min(
                right - left for left, right in zip(comparison_values, comparison_values[1:])
            ) > max(1e-6, comparison_scale * 0.015)
            nearest_stable = min(
                right - left for left, right in zip(nearest_distances, nearest_distances[1:])
            ) > max(0.25, part_scale * 0.002)
            expected_clarification = not nearest_stable if quantifier == "nearest" else not comparison_stable
            if not expected_clarification:
                failures.append({"record_id": record_id, "reason": "clarification_without_unstable_quantifier"})
            else:
                metric_passed += 1
            continue
        target = record["ground_truth"]["targets"][0]
        candidates = sorted(graph["feature_sets"], key=lambda item: int(item["instance_id"]))
        rows = sorted(graph["metric_table"], key=lambda item: int(item["instance_id"]))
        if quantifier in {"all", "count"}:
            expected_ids = [int(item["instance_id"]) for item in candidates]
        elif quantifier == "largest":
            expected_ids = [int(max(rows, key=lambda item: float(item["comparison_value"]))["instance_id"])]
        elif quantifier == "smallest":
            expected_ids = [int(min(rows, key=lambda item: float(item["comparison_value"]))["instance_id"])]
        else:
            expected_ids = [int(min(rows, key=lambda item: float(item["nearest_distance"]))["instance_id"])]
        expected_faces = sorted({
            int(index)
            for item in candidates
            if int(item["instance_id"]) in expected_ids
            for index in item["node_indices"]
        })
        if target["target_instance_ids"] == expected_ids and target["target_face_indices"] == expected_faces:
            reconstruction_passed += 1
        else:
            failures.append({"record_id": record_id, "reason": "target_reconstruction_mismatch"})
        metric_passed += 1
        if record["query"]["difficulty_level"] == "L4":
            executable_l4[quantifier] += 1
    semantic = audit_records(records)
    valid_total = statuses["VALID"]
    return {
        "schema_version": "p3-a1r-pilot-query-audit/1",
        "canonical_records": len(records),
        "status_counts": dict(sorted(statuses.items())),
        "quantifier_counts": dict(sorted(quantifiers.items())),
        "feature_counts": dict(sorted(features.items())),
        "split_counts": dict(sorted(splits.items())),
        "semantic_coverage": semantic,
        "target_reconstruction": {"passed": reconstruction_passed, "total": valid_total},
        "metric_recomputation": {"passed": metric_passed, "total": len(records)},
        "executable_L4": dict(sorted(executable_l4.items())),
        "failure_count": len(failures),
        "failure_examples": failures[:20],
        "automatic_pass": (
            not failures
            and len(records) == 340
            and valid_total == 316
            and statuses["NEEDS_CLARIFICATION"] == 24
            and reconstruction_passed == valid_total
            and metric_passed == len(records)
            and all(executable_l4[name] == 60 for name in ("largest", "smallest", "nearest"))
            and semantic["status"] == "PASS"
        ),
    }


def _record_quantifier(record: dict[str, Any]) -> str:
    if record["intent_result"]["status"] == "VALID":
        return str(record["intent_result"]["intent"]["operations"][0]["target"]["quantifier"])
    text = record["query"]["template_family_id"].split(".")
    return text[-2]


def _write_geometry_review(path: Path, samples: list[dict[str, Any]]) -> int:
    by_feature: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_feature[sample["feature_type"]].append(sample)
    selected = [sorted(items, key=lambda item: item["sample_id"])[0] for _feature, items in sorted(by_feature.items())]
    tie_extras = [item for item in samples if item["tie_design"]][:3]
    selected.extend(tie_extras)
    fields = [
        "review_id", "sample_id", "feature_family", "feature_type", "instance_count", "tie_design",
        "split", "step_path", "brep_path", "graph_path", "comparison_metric", "comparison_values",
        "nearest_distances", "automatic_geometry_pass", "geometry_decision", "reviewer_reason",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, sample in enumerate(selected, start=1):
            graph = json.loads(Path(sample["graph_path"]).read_text(encoding="utf-8"))
            writer.writerow({
                "review_id": f"G{index:02d}",
                "sample_id": sample["sample_id"],
                "feature_family": sample["feature_family"],
                "feature_type": sample["feature_type"],
                "instance_count": sample["instance_count"],
                "tie_design": str(bool(sample["tie_design"])).lower(),
                "split": sample["split"],
                "step_path": sample["step_path"],
                "brep_path": sample["brep_path"],
                "graph_path": sample["graph_path"],
                "comparison_metric": sample["comparison_metric"],
                "comparison_values": json.dumps([item["comparison_value"] for item in graph["metric_table"]]),
                "nearest_distances": json.dumps([item["nearest_distance"] for item in graph["metric_table"]]),
                "automatic_geometry_pass": "true",
                "geometry_decision": "",
                "reviewer_reason": "",
            })
    return len(selected)


def _write_query_review(path: Path, records: list[dict[str, Any]], renderings: list[dict[str, Any]]) -> int:
    rendering_index: dict[str, dict[str, str]] = defaultdict(dict)
    for item in renderings:
        rendering_index[item["canonical_record_id"]][item["rendering_variant"]] = item["text"]
    by_quantifier_feature: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        quantifier = _record_quantifier(record)
        graph = json.loads(Path(record["source_ref"]["graph_path"]).read_text(encoding="utf-8"))
        feature = graph["feature_sets"][0]["macro_feature"]
        by_quantifier_feature[quantifier][feature].append(record)
    features = list(FEATURE_LABELS)
    selected: list[dict[str, Any]] = []
    for quantifier_index, quantifier in enumerate(QUANTIFIERS):
        chosen_features = [features[(quantifier_index * 3 + offset) % len(features)] for offset in range(12)]
        bucket = []
        for feature_index, feature in enumerate(chosen_features):
            candidates = sorted(by_quantifier_feature[quantifier][feature], key=lambda item: item["source_ref"]["sample_id"])
            bucket.append(candidates[(feature_index + quantifier_index) % len(candidates)])
        if quantifier in {"largest", "smallest", "nearest"}:
            tied = [
                record for record in records
                if _record_quantifier(record) == quantifier
                and record["intent_result"]["status"] == "NEEDS_CLARIFICATION"
            ]
            bucket[-2:] = tied[quantifier_index - 2 : quantifier_index]
        selected.extend(bucket)
    fields = [
        "review_id", "sample_id", "feature_type", "split", "step_path", "quantifier",
        "candidate_count", "selected_count", "comparison_metric", "comparison_values", "nearest_distances",
        "intent_status", "tie_expected", "query_en_1", "query_en_2", "query_zh",
        "expected_semantics", "review_decision", "error_category", "corrected_query_en",
        "corrected_query_zh", "reviewer_note",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, record in enumerate(selected, start=1):
            graph = json.loads(Path(record["source_ref"]["graph_path"]).read_text(encoding="utf-8"))
            quantifier = _record_quantifier(record)
            status = record["intent_result"]["status"]
            target = record["ground_truth"]["targets"][0] if status == "VALID" else None
            candidate_count = len(graph["feature_sets"])
            selected_count = len(target["target_instance_ids"]) if target else 0
            variants = rendering_index[record["source_ref"]["canonical_record_id"]]
            writer.writerow({
                "review_id": f"Q{index:02d}",
                "sample_id": record["source_ref"]["sample_id"],
                "feature_type": graph["feature_sets"][0]["macro_feature"],
                "split": record["source_ref"]["dataset_split"],
                "step_path": record["source_ref"]["geometry_path"],
                "quantifier": quantifier,
                "candidate_count": candidate_count,
                "selected_count": selected_count,
                "comparison_metric": graph["metric_table"][0]["comparison_metric"] if quantifier in {"largest", "smallest"} else "",
                "comparison_values": json.dumps([item["comparison_value"] for item in graph["metric_table"]]),
                "nearest_distances": json.dumps([item["nearest_distance"] for item in graph["metric_table"]]),
                "intent_status": status,
                "tie_expected": str(status == "NEEDS_CLARIFICATION").lower(),
                "query_en_1": variants["en-1"],
                "query_en_2": variants["en-2"],
                "query_zh": variants["zh-1"],
                "expected_semantics": (
                    "clarify_geometric_tie"
                    if status == "NEEDS_CLARIFICATION"
                    else f"select {selected_count} of {candidate_count} same-type candidates"
                ),
                "review_decision": "",
                "error_category": "",
                "corrected_query_en": "",
                "corrected_query_zh": "",
                "reviewer_note": "",
            })
    return len(selected)


def _write_review_readme(path: Path, geometry_root: Path, geometry_rows: int, query_rows: int) -> None:
    text = f"""# H3R-01 人工审核说明

状态：`AWAITING_REAL_HUMAN_REVIEW`  
pilot 根目录：`{geometry_root}`

## geometry_review_20.csv

共 {geometry_rows} 行。请按 `step_path` 在 UG/NX 中打开模型，确认：

1. 每件确有 2–4 个同一 canonical type 的独立特征；
2. 特征互不重合、互不破坏，零件为单实体；
3. 尺寸差与相对位置清楚；`tie_design=true` 的件应确实呈现稳定并列；
4. 在 `geometry_decision` 填 `PASS | MINOR | BAD`；MINOR/BAD 在 `reviewer_reason` 写一句原因。

## query_review_60.csv

共 {query_rows} 行，每行同时给出 2 条英文和 1 条中文表达。请确认：

1. 三条 query 自然且工程语义一致；
2. all/count 与 `candidate_count`、`selected_count` 一致；
3. largest/smallest 与 `comparison_metric`、`comparison_values` 一致；
4. nearest 与 `nearest_distances` 一致；
5. `tie_expected=true` 时必须要求澄清，不得产生 VALID target；
6. 在 `review_decision` 填 `approve | edit | reject`。edit 时填写修订文本；reject 时填写错误类别或说明。无需填写 face ID。

## 强制暂停

这两份 CSV 没有预填人工决定。请真实审核并保存后，在同一 Goal 通知继续。收到回填前，不会扩至正式 510+、不会冻结 1.1，也不会进入 P4。
"""
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build deterministic no-LLM P3-A1R pilot queries and H3R-01 package")
    parser.add_argument("--geometry-root", type=Path, default=DEFAULT_GEOMETRY_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    args = parser.parse_args()
    result = build_queries(args.geometry_root, args.report_root)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if result["query_audit"]["automatic_pass"] and result["geometry_review_rows"] == 20 and result["query_review_rows"] == 60 else 1


if __name__ == "__main__":
    raise SystemExit(main())
