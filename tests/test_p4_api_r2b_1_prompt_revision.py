from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import yaml

from utils.cae_intent import validate_intent_result
from utils.cae_query_frame import compile_query_frame, validate_query_frame
from utils import cae_query_frame_runtime_r2b_1 as runtime


ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = (
    ROOT
    / "reports"
    / "p2_a1_cae_ir_v1"
    / "freeze"
    / "configs"
    / "cae_feature_role_policy.yaml"
)


class PromptRevisionTests(unittest.TestCase):
    def test_compatibility_table_is_exactly_policy_derived(self) -> None:
        policy = yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8"))
        expected = []
        for feature, feature_value in sorted(policy["features"].items()):
            for role in sorted(
                feature_value["selectable_roles"], key=lambda item: item["role"]
            ):
                compatible = sorted(
                    operation
                    for operation, review in role["operation_review"].items()
                    if review["compatible"] is True
                )
                expected.append(
                    {
                        "feature": feature,
                        "face_role": role["role"],
                        "compatible_operations": compatible,
                    }
                )
        self.assertEqual(expected, runtime.feature_role_compatibility_table())

    def test_general_rules_and_examples_are_embedded_deterministically(self) -> None:
        first = runtime.build_query_frame_runtime_prompt()
        second = runtime.build_query_frame_runtime_prompt()
        self.assertEqual(first.encode("utf-8"), second.encode("utf-8"))
        for value in (
            runtime.feature_role_compatibility_table(),
            runtime.status_decision_rules(),
            runtime.quantifier_rules(),
            runtime.multi_operation_checklist(),
        ):
            encoded = json.dumps(
                value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            self.assertIn(encoded, first)
        rule_ids = {item["rule_id"] for item in runtime.status_decision_rules()}
        self.assertEqual(
            {
                "unsupported_set_difference",
                "nearest_without_reference",
                "required_operation_information_missing",
                "no_unconfirmed_assumptions",
            },
            rule_ids,
        )
        self.assertEqual("all", runtime.quantifier_rules()["all/every/所有"])
        self.assertEqual(
            "exactly_one", runtime.quantifier_rules()["unique/the only/唯一一个"]
        )
        self.assertEqual(
            "count+instance_count=N", runtime.quantifier_rules()["explicit integer N"]
        )
        self.assertEqual(
            [
                "action",
                "target",
                "selector",
                "value",
                "unit",
                "direction_or_dof",
                "material_parameters",
                "mesh_size",
            ],
            runtime.multi_operation_checklist(),
        )

    def test_five_general_example_categories_validate_end_to_end(self) -> None:
        examples = {item["name"]: item for item in runtime.runtime_examples()}
        required = {
            "unsupported_set_difference",
            "nearest_without_reference",
            "explicit_local_face_role",
            "count_and_exactly_one",
            "multi_operation_parameter_preservation",
        }
        self.assertTrue(required <= set(examples))
        for name in required:
            frame = examples[name]["frame"]
            self.assertTrue(validate_query_frame(frame)["valid"], name)
            self.assertTrue(validate_intent_result(compile_query_frame(frame))["valid"], name)
        local = examples["explicit_local_face_role"]["frame"]["operations"]
        self.assertEqual("rounded_end_wall", local[0]["face_role"])
        self.assertEqual("bevel_face", local[1]["face_role"])
        count_ops = examples["count_and_exactly_one"]["frame"]["operations"]
        self.assertEqual(("count", 3), (count_ops[0]["selector"], count_ops[0]["instance_count"]))
        self.assertEqual("exactly_one", count_ops[1]["selector"])

    def test_prompt_has_no_smoke_specific_identifiers(self) -> None:
        prompt = runtime.build_query_frame_runtime_prompt()
        for forbidden in ("RF.", "P3A1R-Q.", "sample_id"):
            self.assertNotIn(forbidden, prompt)


class OfflineDiagnosisTests(unittest.TestCase):
    def test_field_level_categories_preserve_canonical_exact(self) -> None:
        import run_p4_api_r2b_1_prompt_revision_smoke as r2b1

        expected = {
            "status": "VALID",
            "intent": {
                "operations": [
                    {
                        "operation_id": "op1",
                        "operation_type": "local_mesh_size",
                        "target": {
                            "scope": "feature",
                            "feature_type": "slot_o",
                            "face_role": "side_wall",
                            "quantifier": "count",
                            "count": 2,
                            "filters": {},
                        },
                        "parameters": {"size": {"value": 1, "unit": "mm", "dimension": "length"}},
                    }
                ]
            },
        }
        predicted = json.loads(json.dumps(expected))
        predicted["intent"]["operations"][0]["target"]["quantifier"] = "all"
        predicted["intent"]["operations"][0]["target"].pop("count")
        predicted["intent"]["operations"][0]["parameters"]["size"]["value"] = 2
        row = r2b1.classify_field_differences(expected, predicted)
        self.assertFalse(row["canonical_semantic_exact"])
        self.assertTrue(row["categories"]["quantifier_count_error"])
        self.assertTrue(row["categories"]["parameter_value_or_unit_error"])
        self.assertFalse(row["categories"]["action_error"])


class ExecutionContractTests(unittest.TestCase):
    def test_new_client_uses_isolated_cache_namespace(self) -> None:
        import execute_p4_api_r2b_1_prompt_revision_smoke as delivery

        class Cache:
            def __init__(self):
                self.names = []

            def get(self, namespace, request_hash):
                self.names.append(namespace)
                return {"response": {}}

        with tempfile.TemporaryDirectory() as raw:
            client = object.__new__(delivery.R2B1Client)
            client.cache = Cache()
            client.ledger = type("Ledger", (), {"completed": lambda self, value: True})()
            client._fetch(sample_id="s", stage="first_round", request={"x": 1})
        self.assertEqual(["p4_api_r2b_1_first_round"], client.cache.names)

    def test_stop_states_are_exactly_the_supervisor_contract(self) -> None:
        import execute_p4_api_r2b_1_prompt_revision_smoke as delivery

        passing = {key: True for key in delivery.ACCEPTANCE_CHECK_NAMES}
        self.assertEqual(
            "READY_FOR_P4_API_R2C_DISJOINT_100_VAL_REVIEW",
            delivery.stop_state(passing, external_blocker=None),
        )
        passing["canonical_semantic_exact_ge_12"] = False
        self.assertEqual(
            "DEEPSEEK_COMPACT_FRAME_INSUFFICIENT_WITH_EVIDENCE",
            delivery.stop_state(passing, external_blocker=None),
        )
        self.assertEqual(
            "BLOCKED_WITH_EVIDENCE",
            delivery.stop_state(passing, external_blocker="E_HTTP_401"),
        )


if __name__ == "__main__":
    unittest.main()
