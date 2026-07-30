"""Run the scoped P2-A1 test suite and write an evidence-rich summary."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import unittest
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jsonschema
import yaml

from utils.cae_intent import normalize_intent_units, validate_intent, validate_intent_result


ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "tests" / "fixtures" / "cae_ir_v1" / "cases.jsonl"
TEST_NAMES = ["tests.test_cae_intent_schema", "tests.test_cae_intent_semantics"]


def load_cases() -> list[dict[str, Any]]:
    return [json.loads(line) for line in CASES_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def post_run_fixture_audit(cases: list[dict[str, Any]]) -> dict[str, Any]:
    executed = 0
    passed = 0

    def check(condition: bool, message: str) -> None:
        nonlocal executed, passed
        executed += 1
        if not condition:
            raise AssertionError(message)
        passed += 1

    ids = [case["case_id"] for case in cases]
    check(len(ids) == len(set(ids)), "case IDs are not unique")
    statuses = Counter(case["expected_status"] for case in cases)
    tags = Counter(tag for case in cases for tag in case["tags"])
    covered_features: set[str] = set()
    covered_operations: set[str] = set()
    for case in cases:
        check(len(case["description"]) >= 20 and "placeholder" not in case["description"].lower(), f"non-substantive {case['case_id']}")
        check(case == json.loads(json.dumps(case, ensure_ascii=False, sort_keys=True)), f"round-trip failed {case['case_id']}")
        status = case["expected_status"]
        if status == "VALID":
            validation = validate_intent(case["payload"])
            check(validation["valid"], f"VALID rejected {case['case_id']}")
            for operation in case["payload"]["operations"]:
                covered_operations.add(operation["operation_type"])
                if "feature_type" in operation["target"]:
                    covered_features.add(operation["target"]["feature_type"])
            if "unit_conversion" in case["tags"]:
                normalized = normalize_intent_units(case["payload"])
                check(bool(normalized["normalization_audit"]), f"missing unit audit {case['case_id']}")
                check(normalized == normalize_intent_units(normalized), f"normalization not idempotent {case['case_id']}")
        elif status == "NEEDS_CLARIFICATION":
            validation = validate_intent_result(case["payload"])
            check(validation["valid"], f"clarification rejected {case['case_id']}")
            check("intent" not in case["payload"] and "operations" not in case["payload"], f"partial intent leaked {case['case_id']}")
            check(bool(case["payload"]["clarification_questions"]), f"missing question {case['case_id']}")
        else:
            validation = validate_intent(case["payload"])
            check(not validation["valid"], f"INVALID accepted {case['case_id']}")
            check(all(item["error_code"].startswith("E_") for item in validation["errors"]), f"unstable error code {case['case_id']}")

    for status, minimum in (("VALID", 50), ("NEEDS_CLARIFICATION", 35), ("INVALID", 35)):
        check(statuses[status] >= minimum, f"{status} count below {minimum}")
    for tag, minimum in (("unit_conversion", 20), ("role_operation_negative", 15), ("multi_operation", 10), ("quantifier_or_filter", 10)):
        check(tags[tag] >= minimum, f"{tag} count below {minimum}")
    expected_features = {
        "through_hole_round", "blind_hole_round", "stepped_hole_round", "counterbore_hole",
        "countersink_hole", "obround_hole", "slot_rect", "slot_u", "slot_v", "slot_o",
        "pocket_rect", "pocket_circular", "cylinder_boss", "rectangular_boss", "rib", "fillet", "chamfer",
    }
    expected_operations = {
        "fixed_support", "displacement", "pressure", "surface_traction",
        "material_assignment", "global_mesh_size", "local_mesh_size",
    }
    check(covered_features == expected_features, "feature coverage mismatch")
    check(covered_operations == expected_operations, "operation coverage mismatch")
    return {
        "assertion_checks_executed": executed,
        "assertion_checks_passed": passed,
        "status_counts": dict(sorted(statuses.items())),
        "tag_counts": dict(sorted(tags.items())),
        "covered_features": sorted(covered_features),
        "covered_operations": sorted(covered_operations),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")

    suite = unittest.defaultTestLoader.loadTestsFromNames(TEST_NAMES)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        return 1
    cases = load_cases()
    evidence = post_run_fixture_audit(cases)
    methods_passed = result.testsRun - len(result.failures) - len(result.errors) - len(result.unexpectedSuccesses)
    summary = {
        "schema_version": "topocae_p2_a1_test_case_summary_1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": r"W:\python\envs\feagpt\python.exe -m utils.run_p2_a1_cae_ir_tests --output reports\p2_a1_cae_ir_v1\test_case_summary.json",
        "python": {"executable": sys.executable, "version": platform.python_version()},
        "dependencies": {"jsonschema": jsonschema.__version__, "PyYAML": yaml.__version__},
        "unittest": {
            "test_methods_total": result.testsRun,
            "test_methods_passed": methods_passed,
            "failures": len(result.failures),
            "errors": len(result.errors),
            "skipped": len(result.skipped),
        },
        "fixtures": {
            "path": str(CASES_PATH),
            "case_total": len(cases),
            **evidence,
        },
        "rates": {
            "valid_acceptance": 1.0,
            "invalid_rejection": 1.0,
            "clarification_structured_return": 1.0,
            "overall_fixture_pass": 1.0,
        },
        "scope_guards": {
            "llm_called": False,
            "model_trained": False,
            "abaqus_executed": False,
            "holdout_rerun": False,
            "forbidden_dataset_content_read": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
