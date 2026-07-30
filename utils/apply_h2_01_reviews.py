"""Validate and apply signed H2-01 decisions to P2-A1 artifacts."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from utils.build_cae_ir_cases import intent, operation, quantity, target
from utils.cae_intent import load_feature_role_policy, validate_intent, validate_intent_result


DECISIONS = {"approve", "edit", "reject"}
IDENTITY_WORDS = {
    "raw_face_id", "face_id", "face_ids", "faces", "sample_id", "part_id",
    "model_file", "model_filename", "file_name", "filename", "area_truth", "truth_area",
    "geometry_source", "coverage_status", "abstention_reasons",
}
ROLE_EDIT_KEYS = {
    "proposed_compatible", "selection_status", "semantic_subtype_candidates",
    "region_kind", "requires_complete_instance", "may_include_transition",
    "engineering_explanation", "recognizer_runtime_note",
}
REQUIREMENT_EDIT_KEYS = {
    "requirement_text", "source_language", "expected_status", "analysis_type",
    "operation_type", "feature_type", "face_role", "quantifier",
    "typed_parameters_json", "missing_information", "ambiguities",
    "clarification_question", "engineering_rationale",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_correction(row: dict[str, str], allowed_keys: set[str]) -> dict[str, Any]:
    if row["review_decision"] != "edit":
        return {}
    try:
        correction = json.loads(row.get("corrected", ""))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{row['review_id']}: corrected is not valid JSON: {exc}") from exc
    if not isinstance(correction, dict) or not correction:
        raise ValueError(f"{row['review_id']}: corrected must be a non-empty JSON object")
    unknown = set(correction) - allowed_keys
    if unknown:
        raise ValueError(f"{row['review_id']}: unsupported corrected keys: {sorted(unknown)}")
    serialized = json.dumps(correction, ensure_ascii=False).lower()
    if any(f'"{word}"' in serialized for word in IDENTITY_WORDS):
        raise ValueError(f"{row['review_id']}: corrected introduces a forbidden identity/envelope field")
    return correction


def validate_reviews(role_rows: list[dict[str, str]], requirement_rows: list[dict[str, str]]) -> None:
    if not role_rows or not requirement_rows:
        raise ValueError("H2 review CSV is empty")
    role_ids = [row["review_id"] for row in role_rows]
    requirement_ids = [row["review_id"] for row in requirement_rows]
    if len(role_ids) != len(set(role_ids)) or len(requirement_ids) != len(set(requirement_ids)):
        raise ValueError("Duplicate H2 review_id")
    for row in role_rows + requirement_rows:
        decision = row.get("review_decision", "").strip()
        if decision not in DECISIONS:
            raise ValueError(f"{row.get('review_id')}: review_decision must be approve/edit/reject")
        row["review_decision"] = decision
        if decision == "reject" and not row.get("reviewer_note", "").strip():
            raise ValueError(f"{row['review_id']}: rejected row requires reviewer_note")
    for row in role_rows:
        parse_correction(row, ROLE_EDIT_KEYS)
    for row in requirement_rows:
        parse_correction(row, REQUIREMENT_EDIT_KEYS)
    if len(role_rows) != 302:
        raise ValueError(f"Expected all 302 role rows, got {len(role_rows)}")
    if len(requirement_rows) < 30:
        raise ValueError("At least 30 requirement rows must be reviewed")


def role_final_compatibility(row: dict[str, str]) -> tuple[bool, dict[str, Any]]:
    correction = parse_correction(row, ROLE_EDIT_KEYS)
    proposed = correction.get("proposed_compatible", row["proposed_compatible"])
    if proposed not in {"yes", "no"}:
        raise ValueError(f"{row['review_id']}: proposed_compatible must be yes/no")
    # A reject always rejects the feature-role-operation combination. An approved
    # pre-filled 'no' confirms incompatibility and never flips it to allowed.
    compatible = row["review_decision"] != "reject" and proposed == "yes"
    return compatible, correction


def apply_policy_reviews(policy: dict[str, Any], rows: list[dict[str, str]]) -> dict[str, Any]:
    reviewed = copy.deepcopy(policy)
    indexed: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    whole_part_rows = []
    for row in rows:
        if row["feature_type"] == "__whole_part__":
            whole_part_rows.append(row)
        else:
            indexed[(row["feature_type"], row["face_role"])][row["operation_type"]] = row

    operation_order = reviewed["operation_types"]
    allowed_total = 0
    rejected_total = 0
    for feature_name, feature in reviewed["features"].items():
        feature_allowed: set[str] = set()
        for role in feature["selectable_roles"]:
            role_rows = indexed.get((feature_name, role["role"]), {})
            if set(role_rows) != {"fixed_support", "displacement", "pressure", "surface_traction", "local_mesh_size"}:
                raise ValueError(f"Incomplete review matrix for {feature_name}.{role['role']}")
            allowed = []
            operation_review = {}
            for operation_type, row in role_rows.items():
                compatible, correction = role_final_compatibility(row)
                if role["selection_status"].startswith("unsupported"):
                    compatible = False
                if compatible:
                    allowed.append(operation_type)
                    feature_allowed.add(operation_type)
                    allowed_total += 1
                else:
                    rejected_total += 1
                operation_review[operation_type] = {
                    "review_id": row["review_id"],
                    "decision": row["review_decision"],
                    "compatible": compatible,
                    "reviewer_note": row.get("reviewer_note", ""),
                    "correction": correction,
                }
            role["allowed_operations"] = [op for op in operation_order if op in allowed]
            role["operation_review"] = operation_review
        feature["allowed_operations"] = [op for op in operation_order if op in feature_allowed]

    whole_part = {}
    for row in whole_part_rows:
        compatible, correction = role_final_compatibility(row)
        if not compatible:
            raise ValueError(f"Whole-part operation rejected without a replacement: {row['review_id']}")
        whole_part[row["operation_type"]] = {
            "review_id": row["review_id"], "decision": row["review_decision"],
            "compatible": True, "reviewer_note": row.get("reviewer_note", ""),
            "correction": correction,
        }
    if set(whole_part) != {"material_assignment", "global_mesh_size"}:
        raise ValueError("Whole-part operation review is incomplete")
    reviewed["policy_revision"] = "h2_01_approved_2026-07-20"
    reviewed["h2_01"] = {
        "status": "completed",
        "reviewed_at_local_date": "2026-07-20",
        "role_rows_reviewed": len(rows),
        "compatible_feature_role_operation_rows": allowed_total,
        "incompatible_feature_role_operation_rows": rejected_total,
        "whole_part_operation_review": whole_part,
    }
    return reviewed


def approved_requirements(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    approved = []
    rejected = []
    for row in rows:
        correction = parse_correction(row, REQUIREMENT_EDIT_KEYS)
        if row["review_decision"] == "reject":
            rejected.append({
                "review_id": row["review_id"],
                "requirement_text": row["requirement_text"],
                "reviewer_note": row["reviewer_note"],
            })
            continue
        value: dict[str, Any] = {
            key: row[key]
            for key in (
                "review_id", "requirement_text", "source_language", "expected_status",
                "analysis_type", "operation_type", "feature_type", "face_role", "quantifier",
                "missing_information", "ambiguities", "clarification_question",
                "engineering_rationale",
            )
        }
        value["typed_parameters"] = json.loads(row["typed_parameters_json"] or "{}")
        for key, corrected in correction.items():
            if key == "typed_parameters_json":
                value["typed_parameters"] = json.loads(corrected) if isinstance(corrected, str) else corrected
            else:
                value[key] = corrected
        value["review_decision"] = row["review_decision"]
        value["reviewer_note"] = row.get("reviewer_note", "")
        approved.append(value)
    if len(approved) < 30:
        raise ValueError(f"Only {len(approved)} approved/edited requirements remain")
    return approved, rejected


def allowed_matrix(role_rows: list[dict[str, str]]) -> set[tuple[str, str, str]]:
    allowed = set()
    for row in role_rows:
        compatible, _ = role_final_compatibility(row)
        if compatible and row["feature_type"] != "__whole_part__" and not row["selection_status"].startswith("unsupported"):
            allowed.add((row["feature_type"], row["face_role"], row["operation_type"]))
    return allowed


def reconcile_cases(
    cases: list[dict[str, Any]],
    role_rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    allowed = allowed_matrix(role_rows)
    by_operation: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for feature, role, operation_type in sorted(allowed):
        by_operation[operation_type].append((feature, role))
    changed_valid = 0
    for case in cases:
        if case["expected_status"] != "VALID":
            continue
        for op in case["payload"]["operations"]:
            target_value = op["target"]
            if "feature_type" not in target_value:
                continue
            key = (target_value["feature_type"], target_value["face_role"], op["operation_type"])
            if key in allowed:
                continue
            changed_valid += 1
            if "feature_coverage" in case["tags"]:
                op["operation_type"] = "local_mesh_size"
                op["parameters"] = {"size": quantity("length", 0.8, "mm")}
                case["tags"] = [tag for tag in case["tags"] if not tag.startswith("operation:")]
                case["tags"].append("operation:local_mesh_size")
            else:
                same_feature = [candidate for candidate in by_operation[op["operation_type"]] if candidate[0] == target_value["feature_type"]]
                replacement = (same_feature or by_operation[op["operation_type"]])[0]
                target_value["feature_type"], target_value["face_role"] = replacement
            case["description"] += " H2-01 approved-policy target applied."

    existing_ids = {case["case_id"] for case in cases}
    added_negatives = 0
    for row in role_rows:
        if row["feature_type"] == "__whole_part__" or row["review_decision"] != "reject" or row["proposed_compatible"] != "yes":
            continue
        case_id = f"H2I-{row['review_id']}"
        if case_id in existing_ids:
            continue
        payload = intent([
            operation(
                f"h2_rejected_{row['review_id'].lower()}",
                row["operation_type"],
                target(row["feature_type"], row["face_role"]),
            )
        ])
        case = {
            "case_id": case_id,
            "expected_status": "INVALID",
            "description": f"H2-01 rejects {row['operation_type']} on {row['feature_type']}.{row['face_role']}: {row['reviewer_note']}",
            "payload": payload,
            "tags": ["h2_role_rejection", "role_operation_negative"],
        }
        cases.append(case)
        existing_ids.add(case_id)
        added_negatives += 1
    return cases, {"changed_valid_cases": changed_valid, "added_h2_negative_cases": added_negatives}


def add_schema_examples(root: Path, cases: list[dict[str, Any]]) -> None:
    valid_cases = [case for case in cases if case["expected_status"] == "VALID"]
    clarification = next(case["payload"] for case in cases if case["expected_status"] == "NEEDS_CLARIFICATION")
    operation_types = [
        "fixed_support", "displacement", "pressure", "surface_traction",
        "material_assignment", "global_mesh_size", "local_mesh_size",
    ]
    examples = []
    for operation_type in operation_types:
        sample = next(
            case["payload"] for case in valid_cases
            if any(op["operation_type"] == operation_type for op in case["payload"]["operations"])
        )
        if sample not in examples:
            examples.append(sample)
    intent_path = root / "schemas" / "cae_intent.schema.json"
    intent_schema = json.loads(intent_path.read_text(encoding="utf-8"))
    intent_schema["examples"] = examples
    intent_path.write_text(json.dumps(intent_schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    result_path = root / "schemas" / "cae_intent_result.schema.json"
    result_schema = json.loads(result_path.read_text(encoding="utf-8"))
    result_schema["examples"] = [
        {"schema_version": "cae-intent-result/1.0.0", "status": "VALID", "intent": examples[0]},
        clarification,
    ]
    result_path.write_text(json.dumps(result_schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    context_path = root / "schemas" / "cae_grounding_context.schema.json"
    context_schema = json.loads(context_path.read_text(encoding="utf-8"))
    base = {
        "schema_version": "cae-grounding-context/1.0.0",
        "recognizer_capability_level": "LIMITED_CONTROLLED_DOMAIN",
        "feature_object_schema_version": "topocae_recognizer_dev_feature_objects_1",
    }
    context_schema["examples"] = [
        dict(base, geometry_source="oracle", coverage_status="ACCEPTED", abstention_reasons=[]),
        dict(base, geometry_source="predicted", coverage_status="ABSTAIN", abstention_reasons=["R_SLOT_PREDICTED_UNSUPPORTED"]),
        dict(base, geometry_source="human_confirmed", coverage_status="ACCEPTED", abstention_reasons=[]),
    ]
    context_path.write_text(json.dumps(context_schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    report_dir = root / "reports" / "p2_a1_cae_ir_v1"
    review_dir = report_dir / "human_review_package"
    role_csv = review_dir / "feature_role_policy_review.csv"
    requirement_csv = review_dir / "cae_requirement_review_40.csv"
    summary_path = report_dir / "h2_01_decision_summary.json"
    approved_path = report_dir / "approved_requirements.jsonl"
    snapshot_dir = report_dir / "h2_pre_update"
    if summary_path.exists() or approved_path.exists() or snapshot_dir.exists():
        raise FileExistsError("H2 application outputs already exist; refusing to reapply signed decisions")

    role_rows = read_csv(role_csv)
    requirement_rows = read_csv(requirement_csv)
    validate_reviews(role_rows, requirement_rows)
    policy_path = root / "configs" / "cae_feature_role_policy.yaml"
    cases_path = root / "tests" / "fixtures" / "cae_ir_v1" / "cases.jsonl"
    schema_paths = [
        root / "schemas" / "cae_intent.schema.json",
        root / "schemas" / "cae_intent_result.schema.json",
        root / "schemas" / "cae_grounding_context.schema.json",
    ]
    snapshot_dir.mkdir(parents=True)
    for path in [policy_path, cases_path, *schema_paths]:
        shutil.copy2(path, snapshot_dir / path.name)

    policy_before = sha256(policy_path)
    cases_before = sha256(cases_path)
    reviewed_policy = apply_policy_reviews(load_feature_role_policy(policy_path), role_rows)
    policy_path.write_text(yaml.safe_dump(reviewed_policy, allow_unicode=True, sort_keys=False, width=120), encoding="utf-8")

    approved, rejected_requirements = approved_requirements(requirement_rows)
    approved_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in approved),
        encoding="utf-8",
    )
    cases = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    cases, reconciliation = reconcile_cases(cases, role_rows)
    for case in cases:
        if case["expected_status"] == "VALID":
            result = validate_intent(case["payload"])
            if not result["valid"]:
                raise AssertionError(f"Reconciled VALID failed {case['case_id']}: {result['errors'][:3]}")
        elif case["expected_status"] == "NEEDS_CLARIFICATION":
            if not validate_intent_result(case["payload"])["valid"]:
                raise AssertionError(f"Clarification failed after H2: {case['case_id']}")
        elif validate_intent(case["payload"])["valid"]:
            raise AssertionError(f"Reconciled INVALID accepted: {case['case_id']}")
    cases_path.write_text(
        "".join(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n" for case in cases),
        encoding="utf-8",
    )
    add_schema_examples(root, cases)

    summary = {
        "schema_version": "topocae_h2_01_decision_summary_1",
        "applied_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "role_review": {
            "rows": len(role_rows),
            "decisions": dict(Counter(row["review_decision"] for row in role_rows)),
            "final_compatible_feature_role_operations": len(allowed_matrix(role_rows)),
            "final_incompatible_feature_role_operations": 300 - len(allowed_matrix(role_rows)),
        },
        "requirement_review": {
            "rows": len(requirement_rows),
            "decisions": dict(Counter(row["review_decision"] for row in requirement_rows)),
            "approved_or_edited": len(approved),
            "rejected": rejected_requirements,
        },
        "fixture_reconciliation": reconciliation,
        "hashes": {
            "policy_before": policy_before,
            "policy_after": sha256(policy_path),
            "cases_before": cases_before,
            "cases_after": sha256(cases_path),
            "signed_role_csv": sha256(role_csv),
            "signed_requirement_csv": sha256(requirement_csv),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
