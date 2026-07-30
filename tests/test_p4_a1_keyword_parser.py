from __future__ import annotations

import unittest

from utils.cae_intent import validate_intent_result
from utils.cae_language_grounding.keyword_parser import parse_keyword_query
from utils.cae_language_grounding.records import (
    INFERENCE_FORBIDDEN_KEYS,
    assert_inference_input_safe,
    iter_language_examples,
)


class KeywordParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = list(iter_language_examples("train"))

    def test_inference_view_contains_only_text_and_language(self):
        for example in self.train[:100]:
            value = example.inference_input()
            self.assertEqual(set(value), {"text", "language"})
            self.assertFalse(INFERENCE_FORBIDDEN_KEYS & set(value))
            assert_inference_input_safe(value)

    def test_outputs_are_always_valid_p2_results(self):
        for example in self.train[:1000]:
            prediction = parse_keyword_query(**example.inference_input())
            validation = validate_intent_result(prediction)
            self.assertTrue(validation["valid"], validation["errors"])

    def test_required_operation_and_quantifier_coverage(self):
        operations = set()
        quantifiers = set()
        features = set()
        roles = set()
        for example in self.train:
            oracle = example.oracle_intent_result
            if oracle["status"] != "VALID":
                continue
            prediction = parse_keyword_query(**example.inference_input())
            if prediction["status"] != "VALID":
                continue
            for operation in prediction["intent"]["operations"]:
                operations.add(operation["operation_type"])
                target = operation["target"]
                if "feature_type" in target:
                    features.add(target["feature_type"])
                    roles.add((target["feature_type"], target["face_role"]))
                    quantifiers.add(target["quantifier"])
            if (
                len(operations) == 7
                and len(features) == 17
                and len(quantifiers) >= 6
            ):
                break
        self.assertEqual(
            operations,
            {
                "fixed_support",
                "displacement",
                "pressure",
                "surface_traction",
                "material_assignment",
                "global_mesh_size",
                "local_mesh_size",
            },
        )
        self.assertEqual(len(features), 17)
        self.assertTrue(
            {"all", "count", "exactly_one", "largest", "smallest", "nearest"}
            <= quantifiers
        )
        self.assertGreaterEqual(len(roles), 45)

    def test_train_invalid_negation_and_missing_reference(self):
        invalid = next(
            item
            for item in self.train
            if item.oracle_intent_result["status"] == "INVALID"
        )
        missing = next(
            item
            for item in self.train
            if item.oracle_intent_result["status"] == "NEEDS_CLARIFICATION"
            and item.oracle_intent_result["error_code"] == "E_MISSING_INFORMATION"
        )
        self.assertEqual(
            parse_keyword_query(**invalid.inference_input())["status"], "INVALID"
        )
        self.assertEqual(
            parse_keyword_query(**missing.inference_input())["status"],
            "NEEDS_CLARIFICATION",
        )


if __name__ == "__main__":
    unittest.main()
