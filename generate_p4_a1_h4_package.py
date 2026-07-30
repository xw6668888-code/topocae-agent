"""Generate the deterministic 60-row P4-A1 H4-01 package from val only."""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from utils.cae_language_grounding.records import iter_language_examples


ROOT = Path(__file__).resolve().parent
KEYWORD_PREDICTIONS = (
    ROOT
    / "output_p4_a1_language_grounding_v1"
    / "baselines"
    / "keyword"
    / "val"
    / "predictions.jsonl"
)
PACKAGE_DIR = (
    ROOT
    / "reports"
    / "p4_a1_language_grounding_v1"
    / "human_review_package"
)
FEATURE_TYPES = {
    "through_hole_round",
    "blind_hole_round",
    "stepped_hole_round",
    "counterbore_hole",
    "countersink_hole",
    "obround_hole",
    "slot_rect",
    "slot_u",
    "slot_v",
    "slot_o",
    "pocket_rect",
    "pocket_circular",
    "rectangular_boss",
    "cylinder_boss",
    "chamfer",
    "fillet",
    "rib",
}
REQUIRED_QUANTIFIERS = {"all", "count", "largest", "smallest", "nearest"}
REQUIRED_OPERATIONS = {
    "fixed_support",
    "pressure",
    "surface_traction",
    "displacement",
    "local_mesh_size",
}


def _stable_key(value: str) -> str:
    return hashlib.sha256(
        ("topocae-p4-a1-h4-01:" + value).encode("utf-8")
    ).hexdigest()


def _load_keyword_predictions() -> dict[str, Mapping[str, Any]]:
    result = {}
    with KEYWORD_PREDICTIONS.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            result[str(row["example_id"])] = row
    return result


def _operations(result: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    intent = result.get("intent")
    if not isinstance(intent, Mapping):
        return []
    operations = intent.get("operations", [])
    return [item for item in operations if isinstance(item, Mapping)]


def _dimensions(
    oracle: Mapping[str, Any], prediction: Mapping[str, Any]
) -> tuple[set[str], set[str], set[str]]:
    operations = _operations(oracle) or _operations(prediction)
    features, quantifiers, operation_types = set(), set(), set()
    for operation in operations:
        operation_types.add(str(operation.get("operation_type", "")))
        target = operation.get("target")
        if isinstance(target, Mapping):
            feature = target.get("feature_type")
            quantifier = target.get("quantifier")
            if feature:
                features.add(str(feature))
            if quantifier:
                quantifiers.add(str(quantifier))
    return features, quantifiers, operation_types


def _compact_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _operation_summary(result: Mapping[str, Any]) -> str:
    if result.get("status") != "VALID":
        return _compact_json(
            {
                "status": result.get("status"),
                "error_code": result.get("error_code"),
                "field_path": result.get("field_path"),
            }
        )
    summary = []
    for operation in _operations(result):
        target = operation.get("target", {})
        summary.append(
            {
                "operation_type": operation.get("operation_type"),
                "feature_type": target.get("feature_type"),
                "face_role": target.get("face_role"),
                "quantifier": target.get("quantifier"),
                "filters": target.get("filters", {}),
                "parameters": operation.get("parameters", {}),
            }
        )
    return _compact_json({"status": "VALID", "operations": summary})


def _field_summary(result: Mapping[str, Any], field: str) -> str:
    values = []
    for operation in _operations(result):
        target = operation.get("target", {})
        if field == "feature_type":
            value = target.get("feature_type")
        elif field == "face_role":
            value = target.get("face_role")
        elif field == "quantifier_filter":
            value = {
                "quantifier": target.get("quantifier"),
                "filters": target.get("filters", {}),
            }
        else:
            value = operation.get("parameters", {})
        if value not in (None, {}, []):
            values.append(value)
    return _compact_json(values)


def _select_rows(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def take_status(status: str, per_language: int) -> None:
        for language in ("zh-CN", "en-US"):
            pool = sorted(
                (
                    row
                    for row in candidates
                    if row["status"] == status
                    and row["language"] == language
                    and row["example_id"] not in selected_ids
                ),
                key=lambda row: _stable_key(row["example_id"]),
            )
            if len(pool) < per_language:
                raise RuntimeError(
                    f"Insufficient {language}/{status} val examples"
                )
            for row in pool[:per_language]:
                selected.append(row)
                selected_ids.add(row["example_id"])

    take_status("INVALID", 2)
    take_status("NEEDS_CLARIFICATION", 4)

    counts = {
        "feature": Counter(),
        "quantifier": Counter(),
        "operation": Counter(),
        "language": Counter(row["language"] for row in selected),
    }
    covered_feature: set[str] = set()
    covered_quantifier: set[str] = set()
    covered_operation: set[str] = set()
    valid_pool = [
        row
        for row in candidates
        if row["status"] == "VALID" and row["example_id"] not in selected_ids
    ]
    while sum(row["status"] == "VALID" for row in selected) < 48:
        eligible = [
            row
            for row in valid_pool
            if row["example_id"] not in selected_ids
            and counts["language"][row["language"]] < 30
        ]
        if not eligible:
            raise RuntimeError("Unable to satisfy H4 language/status quotas")

        def score(row: Mapping[str, Any]):
            features = row["features"] & FEATURE_TYPES
            quantifiers = row["quantifiers"] & REQUIRED_QUANTIFIERS
            operations = row["operation_types"] & REQUIRED_OPERATIONS
            coverage = (
                100 * len(features - covered_feature)
                + 100 * len(quantifiers - covered_quantifier)
                + 100 * len(operations - covered_operation)
            )
            balance = (
                sum(20 / (1 + counts["feature"][value]) for value in features)
                + sum(
                    8 / (1 + counts["quantifier"][value])
                    for value in quantifiers
                )
                + sum(
                    8 / (1 + counts["operation"][value])
                    for value in operations
                )
            )
            language_need = 10 * (
                30 - counts["language"][str(row["language"])]
            )
            return (
                coverage + balance + language_need,
                _stable_key(str(row["example_id"])),
            )

        chosen = max(eligible, key=score)
        selected.append(chosen)
        selected_ids.add(chosen["example_id"])
        counts["language"][chosen["language"]] += 1
        for value in chosen["features"] & FEATURE_TYPES:
            covered_feature.add(value)
            counts["feature"][value] += 1
        for value in chosen["quantifiers"] & REQUIRED_QUANTIFIERS:
            covered_quantifier.add(value)
            counts["quantifier"][value] += 1
        for value in chosen["operation_types"] & REQUIRED_OPERATIONS:
            covered_operation.add(value)
            counts["operation"][value] += 1

    if len(selected) != 60:
        raise RuntimeError(f"Expected 60 rows, selected {len(selected)}")
    language_counts = Counter(row["language"] for row in selected)
    status_counts = Counter(row["status"] for row in selected)
    if language_counts != {"zh-CN": 30, "en-US": 30}:
        raise RuntimeError(f"Language quota failure: {language_counts}")
    if status_counts != {
        "VALID": 48,
        "NEEDS_CLARIFICATION": 8,
        "INVALID": 4,
    }:
        raise RuntimeError(f"Status quota failure: {status_counts}")
    if covered_feature != FEATURE_TYPES:
        raise RuntimeError(
            f"Missing feature types: {sorted(FEATURE_TYPES-covered_feature)}"
        )
    if covered_quantifier != REQUIRED_QUANTIFIERS:
        raise RuntimeError(
            "Missing quantifiers: "
            + str(sorted(REQUIRED_QUANTIFIERS - covered_quantifier))
        )
    if covered_operation != REQUIRED_OPERATIONS:
        raise RuntimeError(
            "Missing operations: "
            + str(sorted(REQUIRED_OPERATIONS - covered_operation))
        )
    return sorted(selected, key=lambda row: _stable_key(row["example_id"]))


def generate() -> dict[str, Any]:
    predictions = _load_keyword_predictions()
    candidates = []
    for example in iter_language_examples("val"):
        prediction_row = predictions.get(example.example_id)
        if prediction_row is None:
            raise RuntimeError(
                f"Missing keyword prediction for {example.example_id}"
            )
        prediction = prediction_row["prediction"]
        features, quantifiers, operation_types = _dimensions(
            example.oracle_intent_result, prediction
        )
        candidates.append(
            {
                "example_id": example.example_id,
                "canonical_record_id": example.canonical_record_id,
                "language": example.language,
                "difficulty_level": example.difficulty_level,
                "status": str(example.oracle_intent_result["status"]),
                "text": example.text,
                "oracle": example.oracle_intent_result,
                "prediction": prediction,
                "features": features,
                "quantifiers": quantifiers,
                "operation_types": operation_types,
            }
        )
    selected = _select_rows(candidates)
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = PACKAGE_DIR / "H4-01_val_language_review_60.csv"
    fieldnames = [
        "sample_id",
        "language",
        "difficulty_level",
        "stratum_feature_type",
        "stratum_quantifier",
        "oracle_status",
        "stratum_operation_type",
        "query",
        "baseline_name",
        "parsed_operation_summary",
        "feature_type",
        "face_role",
        "quantifier_filter",
        "parameters_units_direction",
        "clarification_requested",
        "oracle_summary",
        "reviewer_decision",
        "reviewer_note",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(selected, 1):
            prediction = row["prediction"]
            writer.writerow(
                {
                    "sample_id": f"H4-01-{index:03d}",
                    "language": row["language"],
                    "difficulty_level": row["difficulty_level"],
                    "stratum_feature_type": "|".join(
                        sorted(row["features"])
                    ),
                    "stratum_quantifier": "|".join(
                        sorted(row["quantifiers"])
                    ),
                    "oracle_status": row["status"],
                    "stratum_operation_type": "|".join(
                        sorted(row["operation_types"])
                    ),
                    "query": row["text"],
                    "baseline_name": "keyword",
                    "parsed_operation_summary": _operation_summary(
                        prediction
                    ),
                    "feature_type": _field_summary(
                        prediction, "feature_type"
                    ),
                    "face_role": _field_summary(prediction, "face_role"),
                    "quantifier_filter": _field_summary(
                        prediction, "quantifier_filter"
                    ),
                    "parameters_units_direction": _field_summary(
                        prediction, "parameters"
                    ),
                    "clarification_requested": (
                        "yes"
                        if prediction.get("status")
                        == "NEEDS_CLARIFICATION"
                        else "no"
                    ),
                    "oracle_summary": _operation_summary(row["oracle"]),
                    "reviewer_decision": "",
                    "reviewer_note": "",
                }
            )

    def union_count(key: str) -> Counter:
        counter = Counter()
        for row in selected:
            for value in row[key]:
                counter[value] += 1
        return counter

    manifest = {
        "schema_version": "topocae-p4-a1-h4-package/1",
        "gate": "H4-01",
        "source_split": "val",
        "selection_algorithm": (
            "sha256-stable status quotas plus greedy dimension coverage"
        ),
        "row_count": len(selected),
        "review_csv": csv_path.name,
        "distributions": {
            "language": dict(
                sorted(Counter(row["language"] for row in selected).items())
            ),
            "status": dict(
                sorted(Counter(row["status"] for row in selected).items())
            ),
            "feature_type": dict(sorted(union_count("features").items())),
            "quantifier": dict(
                sorted(union_count("quantifiers").items())
            ),
            "operation_type": dict(
                sorted(union_count("operation_types").items())
            ),
        },
        "coverage_checks": {
            "languages_zh_en": True,
            "all_17_feature_types": True,
            "required_quantifiers": sorted(REQUIRED_QUANTIFIERS),
            "all_three_statuses": True,
            "required_operations": sorted(REQUIRED_OPERATIONS),
            "raw_face_id_review_required": False,
        },
    }
    (PACKAGE_DIR / "package_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {
        "csv": str(csv_path.resolve()),
        "manifest": str((PACKAGE_DIR / "package_manifest.json").resolve()),
        "row_count": len(selected),
        "distributions": manifest["distributions"],
    }


if __name__ == "__main__":
    print(json.dumps(generate(), ensure_ascii=False, indent=2))
