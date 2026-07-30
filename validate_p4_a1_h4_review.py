"""Fail-closed validation for the human-filled P4-A1 H4-01 review package."""

from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

import generate_p4_a1_h4_package as h4_generator


ROOT = Path(__file__).resolve().parent
PACKAGE_DIR = (
    ROOT
    / "reports"
    / "p4_a1_language_grounding_v1"
    / "human_review_package"
)
REVIEW_CSV = PACKAGE_DIR / "H4-01_val_language_review_60.csv"
TEMPLATE_NAME = "H4-01_acceptance_record.template.json"
ALLOWED_DECISIONS = frozenset({"approve", "edit", "reject"})
REVIEW_COLUMNS = frozenset({"reviewer_decision", "reviewer_note"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _find_acceptance_records() -> list[tuple[Path, dict[str, Any]]]:
    records = []
    for path in sorted(PACKAGE_DIR.glob("*.json")):
        if path.name == TEMPLATE_NAME:
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if value.get("schema_version") == "topocae-p4-a1-h4-acceptance/1":
            records.append((path, value))
    return records


def _validate_acceptance(
    path: Path, value: dict[str, Any], decision_counts: Counter
) -> list[str]:
    errors = []
    if value.get("status") not in {"ACCEPTED", "PASS"}:
        errors.append("acceptance_status_not_accepted")
    if not str(value.get("reviewer_name", "")).strip():
        errors.append("reviewer_name_missing")
    if not str(value.get("reviewer_role", "")).strip():
        errors.append("reviewer_role_missing")
    if not str(value.get("reviewed_at_utc", "")).strip():
        errors.append("reviewed_at_utc_missing")
    if value.get("gate_decision") not in {"approve", "PASS"}:
        errors.append("gate_decision_not_approved")
    recorded_counts = value.get("decision_counts")
    expected_counts = {
        "approve": decision_counts["approve"],
        "edit": decision_counts["edit"],
        "reject": decision_counts["reject"],
    }
    if recorded_counts != expected_counts:
        errors.append("decision_counts_mismatch")
    if value.get("all_rows_reviewed") is not True:
        errors.append("all_rows_reviewed_not_true")
    attestation = value.get("attestation", {})
    for field in (
        "review_was_performed_by_a_real_human",
        "no_test_query_or_ground_truth_was_read",
        "no_face_id_annotation_was_requested",
    ):
        if attestation.get(field) is not True:
            errors.append(f"attestation_false:{field}")
    return [f"{path.name}:{error}" for error in errors]


def validate() -> dict[str, Any]:
    if not REVIEW_CSV.is_file():
        raise FileNotFoundError(REVIEW_CSV)
    actual_headers, actual_rows = _read_csv(REVIEW_CSV)

    with tempfile.TemporaryDirectory(
        prefix="p4_a1_h4_reference_", dir=ROOT
    ) as temp_dir:
        original_package_dir = h4_generator.PACKAGE_DIR
        try:
            h4_generator.PACKAGE_DIR = Path(temp_dir)
            generated = h4_generator.generate()
            reference_csv = Path(generated["csv"])
            reference_headers, reference_rows = _read_csv(reference_csv)
        finally:
            h4_generator.PACKAGE_DIR = original_package_dir

    errors: list[str] = []
    if actual_headers != reference_headers:
        errors.append("header_mismatch")
    if len(actual_rows) != 60 or len(reference_rows) != 60:
        errors.append(
            f"row_count_mismatch:actual={len(actual_rows)}:"
            f"reference={len(reference_rows)}"
        )
    protected_columns = [
        column for column in reference_headers if column not in REVIEW_COLUMNS
    ]
    for index, (actual, reference) in enumerate(
        zip(actual_rows, reference_rows), start=1
    ):
        for column in protected_columns:
            if actual.get(column) != reference.get(column):
                errors.append(
                    f"protected_cell_changed:row={index}:column={column}"
                )
        decision = (actual.get("reviewer_decision") or "").strip().lower()
        note = (actual.get("reviewer_note") or "").strip()
        if decision not in ALLOWED_DECISIONS:
            errors.append(f"invalid_decision:row={index}:{decision!r}")
        if decision in {"edit", "reject"} and not note:
            errors.append(f"note_required:row={index}:decision={decision}")

    decisions = Counter(
        (row.get("reviewer_decision") or "").strip().lower()
        for row in actual_rows
    )
    acceptance_records = _find_acceptance_records()
    acceptance_errors: list[str] = []
    accepted_record: Path | None = None
    if not acceptance_records:
        acceptance_errors.append("real_human_acceptance_record_missing")
    else:
        for path, value in acceptance_records:
            record_errors = _validate_acceptance(path, value, decisions)
            if not record_errors and accepted_record is None:
                accepted_record = path
            acceptance_errors.extend(record_errors)

    rows_valid = not errors
    h4_accepted = rows_valid and accepted_record is not None
    if h4_accepted:
        status = "PASS"
    elif rows_valid:
        status = "ROWS_COMPLETE_PENDING_HUMAN_ATTESTATION"
    else:
        status = "FAIL"
    return {
        "schema_version": "topocae-p4-a1-h4-review-validation/1",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "source_split": "val",
        "review_csv": str(REVIEW_CSV.relative_to(ROOT)).replace("\\", "/"),
        "review_csv_sha256": _sha256(REVIEW_CSV),
        "row_count": len(actual_rows),
        "decision_counts": {
            "approve": decisions["approve"],
            "edit": decisions["edit"],
            "reject": decisions["reject"],
        },
        "protected_columns_unchanged": not any(
            error.startswith("protected_cell_changed") for error in errors
        ),
        "row_validation_errors": errors,
        "acceptance_record": (
            str(accepted_record.relative_to(ROOT)).replace("\\", "/")
            if accepted_record is not None
            else None
        ),
        "acceptance_errors": acceptance_errors,
        "h4_accepted": h4_accepted,
        "h4_prerequisite_satisfied": h4_accepted,
        "test_unlock_authorized": False,
        "test_unlock_reason": (
            "C2/C3 train/val completion and post-H4 freeze are still required"
            if h4_accepted
            else "H4 acceptance is incomplete"
        ),
    }


def main() -> int:
    result = validate()
    output_path = PACKAGE_DIR / "H4-01_review_validation.json"
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "row_count": result["row_count"],
                "decision_counts": result["decision_counts"],
                "protected_columns_unchanged":
                    result["protected_columns_unchanged"],
                "h4_accepted": result["h4_accepted"],
                "h4_prerequisite_satisfied":
                    result["h4_prerequisite_satisfied"],
                "test_unlock_authorized":
                    result["test_unlock_authorized"],
                "test_unlock_reason": result["test_unlock_reason"],
                "acceptance_errors": result["acceptance_errors"],
                "output": str(output_path.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result["status"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
