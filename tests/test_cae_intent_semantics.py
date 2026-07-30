from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

import yaml

from utils import cae_intent
from utils.cae_intent import (
    build_clarification_result,
    check_role_operation_compatibility,
    load_feature_role_policy,
    normalize_intent_units,
    parse_intent_dict,
    validate_intent,
    validate_intent_result,
)


ROOT = Path(__file__).resolve().parents[1]
CASE_PATH = ROOT / "tests" / "fixtures" / "cae_ir_v1" / "cases.jsonl"


def load_cases():
    return [json.loads(line) for line in CASE_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


class CAEIntentSemanticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_cases()
        cls.valid_cases = [case for case in cls.cases if case["expected_status"] == "VALID"]
        cls.invalid_cases = [case for case in cls.cases if case["expected_status"] == "INVALID"]
        cls.policy = load_feature_role_policy()

    def test_normalization_is_audited_and_idempotent(self):
        conversion_cases = [case for case in self.valid_cases if "unit_conversion" in case["tags"]]
        self.assertGreaterEqual(len(conversion_cases), 20)
        for case in conversion_cases:
            with self.subTest(case_id=case["case_id"]):
                normalized = normalize_intent_units(case["payload"])
                self.assertTrue(normalized["normalization_audit"])
                self.assertEqual(normalized, normalize_intent_units(normalized))
                self.assertTrue(validate_intent(normalized)["valid"])

    def test_conversion_audit_is_traceable(self):
        sample = next(case["payload"] for case in self.valid_cases if "unit_conversion" in case["tags"])
        normalized = normalize_intent_units(sample)
        for record in normalized["normalization_audit"]:
            with self.subTest(path=record["field_path"]):
                self.assertTrue(record["field_path"].startswith("/"))
                self.assertGreater(record["factor"], 0)
                self.assertIn(record["from_unit"], cae_intent.UNIT_FACTORS[record["dimension"]])
                self.assertEqual(normalized["unit_system"][record["dimension"]], record["to_unit"])

    def test_unit_registry_covers_all_required_dimensions(self):
        self.assertEqual(
            {"length", "area", "force", "pressure", "displacement", "angle"},
            set(cae_intent.UNIT_FACTORS),
        )
        self.assertEqual(1000.0, cae_intent.UNIT_FACTORS["force"]["kN"])
        self.assertAlmostEqual(180.0, cae_intent.UNIT_FACTORS["angle"]["rad"] * math.pi)
        self.assertEqual(1.0, cae_intent.UNIT_FACTORS["pressure"]["N/mm2"])

    def test_project_unit_context_is_explicit_in_every_valid_intent(self):
        required = {"length", "area", "force", "pressure", "displacement", "angle"}
        for case in self.valid_cases:
            with self.subTest(case_id=case["case_id"]):
                self.assertEqual(required, set(case["payload"]["unit_system"]))

    def test_aliases_are_canonicalized_only_during_strict_parse(self):
        sample = json.loads(json.dumps(next(
            case["payload"]
            for case in self.valid_cases
            if any(
                operation["operation_type"] == "local_mesh_size"
                and operation["target"].get("feature_type") == "rib"
                for operation in case["payload"]["operations"]
            )
        )))
        sample["operations"][0]["target"]["feature_type"] = "web"
        parsed = parse_intent_dict(sample)
        self.assertEqual("rib", parsed["operations"][0]["target"]["feature_type"])
        self.assertTrue(validate_intent(parsed)["valid"])
        self.assertFalse(validate_intent(sample)["valid"])

    def test_all_supported_policy_roles_accept_their_declared_operations(self):
        role_count = 0
        unsupported_count = 0
        for feature, feature_policy in self.policy["features"].items():
            for role in feature_policy["selectable_roles"]:
                role_count += 1
                if role["selection_status"] in {"direct", "composite"}:
                    for operation_type in role["allowed_operations"]:
                        result = check_role_operation_compatibility(self.policy, feature, role["role"], operation_type)
                        self.assertTrue(result["compatible"], result)
                else:
                    unsupported_count += 1
                    for operation_type in ("fixed_support", "displacement", "pressure", "surface_traction", "local_mesh_size"):
                        result = check_role_operation_compatibility(self.policy, feature, role["role"], operation_type)
                        self.assertFalse(result["compatible"])
        self.assertEqual(60, role_count)
        self.assertEqual(3, unsupported_count)

    def test_whole_part_operations_are_not_feature_role_operations(self):
        result = check_role_operation_compatibility(self.policy, "through_hole_round", "hole_wall", "material_assignment")
        self.assertFalse(result["compatible"])
        result = check_role_operation_compatibility(self.policy, "through_hole_round", "hole_wall", "global_mesh_size")
        self.assertFalse(result["compatible"])

    def test_policy_never_invents_unknown_semantic_subtypes(self):
        ontology = yaml.safe_load((ROOT / "face_semantic_subtype_ontology.yaml").read_text(encoding="utf-8"))
        known = {
            item["key"] for item in ontology["layers"]["face_semantic_subtype"]["labels"]
            if item.get("enabled")
        }
        referenced = {
            subtype
            for feature in self.policy["features"].values()
            for role in feature["selectable_roles"]
            for subtype in role["semantic_subtype_candidates"]
        }
        self.assertFalse(referenced - known)

    def test_clarification_builder_never_returns_partial_intent(self):
        result = build_clarification_result(missing_information=[{
            "code": "M_DIRECTION",
            "field_path": "/operations/0/parameters/direction",
            "description": "Pressure orientation was not specified.",
        }])
        self.assertEqual("NEEDS_CLARIFICATION", result["status"])
        self.assertNotIn("operations", result)
        self.assertNotIn("intent", result)
        self.assertTrue(validate_intent_result(result)["valid"])

    def test_invalid_cases_return_stable_codes_and_paths(self):
        observed_codes = set()
        for case in self.invalid_cases:
            with self.subTest(case_id=case["case_id"]):
                errors = validate_intent(case["payload"])["errors"]
                self.assertTrue(errors)
                for error in errors:
                    self.assertTrue(error["error_code"].startswith("E_"))
                    self.assertTrue(error["field_path"].startswith("/"))
                    observed_codes.add(error["error_code"])
        for required in {
            "E_IDENTITY_FIELD_FORBIDDEN", "E_GEOMETRY_SOURCE_IN_INTENT",
            "E_UNIT_DIMENSION_MISMATCH", "E_ROLE_OPERATION_INCOMPATIBLE",
            "E_QUANTIFIER_COUNT_CONFLICT", "E_PRESSURE_DIRECTION_INVALID",
        }:
            self.assertIn(required, observed_codes)

    def test_multi_operation_cases_have_unique_operation_ids(self):
        cases = [case for case in self.valid_cases if "multi_operation" in case["tags"]]
        self.assertGreaterEqual(len(cases), 10)
        for case in cases:
            ids = [operation["operation_id"] for operation in case["payload"]["operations"]]
            self.assertEqual(len(ids), len(set(ids)))

    def test_quantifier_and_filter_cases_are_semantically_valid(self):
        cases = [case for case in self.valid_cases if "quantifier_or_filter" in case["tags"]]
        self.assertGreaterEqual(len(cases), 10)
        for case in cases:
            self.assertTrue(validate_intent(case["payload"])["valid"])

    def test_pressure_requires_explicit_normal_orientation(self):
        pressure_cases = [
            operation
            for case in self.valid_cases
            for operation in case["payload"]["operations"]
            if operation["operation_type"] == "pressure"
        ]
        self.assertTrue(pressure_cases)
        for operation in pressure_cases:
            self.assertIn(operation["parameters"]["direction"]["kind"], {"surface_normal", "opposite_surface_normal"})

    def test_grounding_fields_never_appear_in_valid_intent_fixtures(self):
        forbidden = cae_intent.GROUNDING_ONLY_KEYS
        for case in self.valid_cases:
            serialized = json.dumps(case["payload"], ensure_ascii=False, sort_keys=True)
            for key in forbidden:
                self.assertNotIn(f'"{key}"', serialized)

    def test_ontology_audit_reports_required_count_differences(self):
        audit = json.loads((ROOT / "reports" / "p2_a1_cae_ir_v1" / "ontology_audit.json").read_text(encoding="utf-8"))
        self.assertEqual(17, audit["counts"]["p2_canonical_target"])
        self.assertEqual(18, audit["counts"]["master_plan_raw"])
        self.assertEqual(17, audit["counts"]["master_plan_after_web_alias"])
        self.assertEqual(16, audit["counts"]["capability_contract"])
        self.assertEqual(["pocket_circular"], audit["set_differences"]["capability_contract_missing"])

    def test_api_surface_is_complete(self):
        expected = {
            "parse_intent_dict", "validate_intent", "normalize_intent_units",
            "validate_intent_result", "load_feature_role_policy",
            "check_role_operation_compatibility", "build_clarification_result",
        }
        self.assertTrue(expected.issubset(set(cae_intent.__all__)))
        for name in expected:
            self.assertTrue(callable(getattr(cae_intent, name)))

    def test_implementation_forbids_dynamic_execution(self):
        source = (ROOT / "utils" / "cae_intent.py").read_text(encoding="utf-8")
        self.assertNotIn("eval(" , source)
        self.assertNotIn("exec(" , source)


if __name__ == "__main__":
    unittest.main()
