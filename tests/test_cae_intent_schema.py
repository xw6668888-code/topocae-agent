from __future__ import annotations

import json
import unittest
from collections import Counter
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator

from utils.cae_intent import validate_intent, validate_intent_result


ROOT = Path(__file__).resolve().parents[1]
CASE_PATH = ROOT / "tests" / "fixtures" / "cae_ir_v1" / "cases.jsonl"
SCHEMA_PATHS = [
    ROOT / "schemas" / "cae_intent.schema.json",
    ROOT / "schemas" / "cae_intent_result.schema.json",
    ROOT / "schemas" / "cae_grounding_context.schema.json",
]


def load_cases():
    return [json.loads(line) for line in CASE_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


class CAEIntentSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_cases()

    def test_all_three_schemas_are_valid_draft_2020_12(self):
        for path in SCHEMA_PATHS:
            with self.subTest(path=path.name):
                schema = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual("https://json-schema.org/draft/2020-12/schema", schema["$schema"])
                Draft202012Validator.check_schema(schema)

    def test_every_declared_object_schema_is_closed(self):
        def visit(value, path=()):
            if isinstance(value, dict):
                if value.get("type") == "object":
                    self.assertIs(value.get("additionalProperties"), False, f"open object at {path}")
                for key, item in value.items():
                    visit(item, path + (key,))
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    visit(item, path + (index,))

        for schema_path in SCHEMA_PATHS:
            visit(json.loads(schema_path.read_text(encoding="utf-8")), (schema_path.name,))

    def test_fixture_status_contracts(self):
        for case in self.cases:
            with self.subTest(case_id=case["case_id"]):
                if case["expected_status"] == "VALID":
                    result = validate_intent(case["payload"])
                    self.assertTrue(result["valid"], result["errors"][:3])
                elif case["expected_status"] == "NEEDS_CLARIFICATION":
                    result = validate_intent_result(case["payload"])
                    self.assertTrue(result["valid"], result["errors"][:3])
                    self.assertNotIn("intent", case["payload"])
                    self.assertNotIn("operations", case["payload"])
                else:
                    result = validate_intent(case["payload"])
                    self.assertFalse(result["valid"])

    def test_fixture_minimum_counts_and_tags(self):
        statuses = Counter(case["expected_status"] for case in self.cases)
        tags = Counter(tag for case in self.cases for tag in case["tags"])
        self.assertGreaterEqual(len(self.cases), 120)
        self.assertGreaterEqual(statuses["VALID"], 50)
        self.assertGreaterEqual(statuses["NEEDS_CLARIFICATION"], 35)
        self.assertGreaterEqual(statuses["INVALID"], 35)
        self.assertGreaterEqual(tags["unit_conversion"], 20)
        self.assertGreaterEqual(tags["role_operation_negative"], 15)
        self.assertGreaterEqual(tags["multi_operation"], 10)
        self.assertGreaterEqual(tags["quantifier_or_filter"], 10)

    def test_all_features_and_operations_are_covered(self):
        intent_schema = json.loads(SCHEMA_PATHS[0].read_text(encoding="utf-8"))
        definitions = intent_schema["$defs"]
        expected_features = set(definitions["featureType"]["enum"])
        expected_operations = {
            "fixed_support", "displacement", "pressure", "surface_traction",
            "material_assignment", "global_mesh_size", "local_mesh_size",
        }
        covered_features = set()
        covered_operations = set()
        for case in self.cases:
            if case["expected_status"] != "VALID":
                continue
            for operation in case["payload"]["operations"]:
                covered_operations.add(operation["operation_type"])
                if "feature_type" in operation["target"]:
                    covered_features.add(operation["target"]["feature_type"])
        self.assertEqual(expected_features, covered_features)
        self.assertEqual(expected_operations, covered_operations)

    def test_valid_result_embeds_only_a_fully_valid_intent(self):
        sample = next(case["payload"] for case in self.cases if case["expected_status"] == "VALID")
        result = {"schema_version": "cae-intent-result/1.0.0", "status": "VALID", "intent": sample}
        self.assertTrue(validate_intent_result(result)["valid"])
        result["intent"]["operations"][0]["target"]["face_id"] = 1
        validation = validate_intent_result(result)
        self.assertFalse(validation["valid"])
        self.assertIn("E_IDENTITY_FIELD_FORBIDDEN", {item["error_code"] for item in validation["errors"]})

    def test_nonvalid_result_cannot_smuggle_operations(self):
        sample = next(case["payload"] for case in self.cases if case["expected_status"] == "NEEDS_CLARIFICATION")
        smuggled = json.loads(json.dumps(sample))
        smuggled["operations"] = []
        validation = validate_intent_result(smuggled)
        self.assertFalse(validation["valid"])
        self.assertIn("E_NONVALID_RESULT_HAS_INTENT", {item["error_code"] for item in validation["errors"]})

    def test_grounding_context_tracks_are_schema_isolated(self):
        schema = json.loads((ROOT / "schemas" / "cae_grounding_context.schema.json").read_text(encoding="utf-8"))
        validator = Draft202012Validator(schema)
        base = {
            "schema_version": "cae-grounding-context/1.0.0",
            "geometry_source": "oracle",
            "recognizer_capability_level": "LIMITED_CONTROLLED_DOMAIN",
            "coverage_status": "ACCEPTED",
            "abstention_reasons": [],
            "feature_object_schema_version": "topocae_recognizer_dev_feature_objects_1",
        }
        for source in ("oracle", "predicted", "human_confirmed"):
            with self.subTest(source=source):
                value = dict(base, geometry_source=source)
                self.assertEqual([], list(validator.iter_errors(value)))
        abstain = dict(base, geometry_source="predicted", coverage_status="ABSTAIN",
                       abstention_reasons=["R_SLOT_PREDICTED_UNSUPPORTED"])
        self.assertEqual([], list(validator.iter_errors(abstain)))
        self.assertTrue(list(validator.iter_errors(dict(abstain, abstention_reasons=[]))))

    def test_json_round_trip_is_lossless_for_every_fixture(self):
        for case in self.cases:
            with self.subTest(case_id=case["case_id"]):
                self.assertEqual(case, json.loads(json.dumps(case, ensure_ascii=False, sort_keys=True)))

    def test_policy_and_alias_documents_have_17_canonical_features(self):
        policy = yaml.safe_load((ROOT / "configs" / "cae_feature_role_policy.yaml").read_text(encoding="utf-8"))
        aliases = yaml.safe_load((ROOT / "configs" / "paper_v1" / "cae_feature_aliases.yaml").read_text(encoding="utf-8"))
        self.assertEqual(17, policy["canonical_feature_count"])
        self.assertEqual(17, len(policy["features"]))
        self.assertEqual(17, aliases["canonical_feature_count"])
        self.assertFalse(aliases["compatibility_rules"]["web"]["emit_as_canonical_output"])


if __name__ == "__main__":
    unittest.main()
