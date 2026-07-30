from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest

from run_p4_api_r1e_transport_schema import mock_audit
from utils.cae_language_grounding.deepseek_strict_transport import (
    PROFILE_CONTRACT,
    PROFILE_NAME,
    STRICT_BASE_URL,
    TOOL_NAME,
    TransportContractError,
    build_mock_tool_payload,
    build_transport_schema,
    canonical_to_transport,
    parse_mock_tool_call_response,
    recompute_r1c_combined_schema_metrics,
    transport_to_canonical,
    validate_deepseek_strict_subset_schema,
    validate_transport_instance,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "cae_ir_v1" / "cases.jsonl"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid_result() -> dict:
    for line in FIXTURES.read_text(encoding="utf-8").splitlines():
        case = json.loads(line)
        if case["expected_status"] == "VALID":
            return {
                "schema_version": "cae-intent-result/1.0.0",
                "status": "VALID",
                "intent": case["payload"],
            }
    raise AssertionError("missing VALID fixture")


def find_presence_wrapper(value):
    if isinstance(value, dict):
        if value.get("presence") == "ABSENT" and set(value) == {"presence"}:
            return value
        for child in value.values():
            found = find_presence_wrapper(child)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = find_presence_wrapper(child)
            if found is not None:
                return found
    return None


class R1ETransportSchemaTests(unittest.TestCase):
    def test_delivery_mock_audit_treats_no_real_client_as_a_pass(self):
        audit = mock_audit()
        self.assertEqual("PASS", audit["status"])
        self.assertTrue(audit["checks"]["real_client_not_created"])

    def test_transport_schema_passes_independent_strict_subset_validator(self):
        validation = validate_deepseek_strict_subset_schema(build_transport_schema())
        self.assertTrue(validation["valid"], validation["errors"][:3])

    def test_round_trip_is_semantically_exact(self):
        canonical = valid_result()
        transport = canonical_to_transport(canonical)
        self.assertTrue(validate_transport_instance(transport)["valid"])
        self.assertEqual(canonical, transport_to_canonical(transport))

    def test_presence_wrapper_does_not_use_engineering_sentinel_values(self):
        transport = canonical_to_transport(valid_result())
        wrapper = find_presence_wrapper(transport)
        self.assertIsNotNone(wrapper)
        self.assertEqual({"presence": "ABSENT"}, wrapper)
        wrapper["presence"] = "PRESENT"
        self.assertFalse(validate_transport_instance(transport)["valid"])

    def test_missing_field_wrong_tag_and_illegal_enum_are_rejected(self):
        canonical = valid_result()
        transport = canonical_to_transport(canonical)
        missing = deepcopy(transport)
        del missing["result"]
        self.assertFalse(validate_transport_instance(missing)["valid"])

        bad_tag = deepcopy(transport)
        bad_tag["result"]["status"] = "UNKNOWN"
        self.assertFalse(validate_transport_instance(bad_tag)["valid"])

        bad_enum = deepcopy(transport)
        bad_enum["transport_schema_version"] = "wrong-version"
        self.assertFalse(validate_transport_instance(bad_enum)["valid"])

    def test_profile_requires_beta_strict_tool_and_forced_choice(self):
        self.assertEqual(PROFILE_NAME, PROFILE_CONTRACT["name"])
        payload = build_mock_tool_payload(
            base_url=STRICT_BASE_URL,
            model="mock-model",
            messages=[{"role": "user", "content": "mock"}],
        )
        function = payload["tools"][0]["function"]
        self.assertTrue(function["strict"])
        self.assertEqual(TOOL_NAME, function["name"])
        self.assertEqual(TOOL_NAME, payload["tool_choice"]["function"]["name"])
        with self.assertRaisesRegex(TransportContractError, "E_STRICT_BETA_ENDPOINT"):
            build_mock_tool_payload(
                base_url="https://api.deepseek.com",
                model="mock-model",
                messages=[],
            )

    def test_mock_tool_call_response_has_no_content_fallback(self):
        canonical = valid_result()
        arguments = json.dumps(canonical_to_transport(canonical), ensure_ascii=False)
        response = {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "type": "function",
                        "function": {"name": TOOL_NAME, "arguments": arguments},
                    }]
                }
            }]
        }
        self.assertEqual(canonical, parse_mock_tool_call_response(response))
        with self.assertRaisesRegex(TransportContractError, "E_TOOL_RESPONSE_SHAPE"):
            parse_mock_tool_call_response({"choices": [{"message": {"content": arguments}}]})

    def test_r1c_combined_schema_metrics_are_weighted_from_splits(self):
        report = ROOT / "reports" / "p4_api_r1c_structured_v1"
        train = json.loads((report / "train_metrics.json").read_text(encoding="utf-8"))
        val = json.loads((report / "val_metrics.json").read_text(encoding="utf-8"))
        combined = json.loads((report / "combined_metrics.json").read_text(encoding="utf-8"))
        corrected = recompute_r1c_combined_schema_metrics(train, val)
        self.assertAlmostEqual(0.07, corrected["schema_valid_rate_first_round"])
        self.assertAlmostEqual(0.89, corrected["schema_valid_rate_after_retry"])
        self.assertEqual(93, corrected["retry_count"])
        self.assertAlmostEqual(0.93, corrected["retry_rate"])
        self.assertNotEqual(
            corrected["schema_valid_rate_after_retry"],
            combined["language_cae_ir"]["schema_valid_rate_after_retry"],
        )

    def test_frozen_contract_hashes_are_unchanged(self):
        expected = {
            "schemas/cae_intent_result.schema.json": "00d5da9125031158786bd853c172f0aae1fb63c0424fd19b3ff82f6918e37727",
            "schemas/cae_intent.schema.json": "8e94cd9a6f3dad42ccad58d47eb82e801e2cb66870ac03dcd940a56d076aeac4",
            "utils/cae_language_grounding/llm_adapter.py": "5c6137cefb44175e3f8818a59cedc87c585ab1adb761cabec521afe3e9dc6fe1",
            "utils/cae_language_grounding/llm_adapter_r0.py": "56e37a521316961f4e44682446ed0ef03a7255784f37e4afd3938a615f121710",
        }
        self.assertEqual(
            expected,
            {name: sha256(ROOT / name) for name in expected},
        )


if __name__ == "__main__":
    unittest.main()
