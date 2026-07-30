from __future__ import annotations

import copy
import json
import unittest

from utils.cae_language_grounding.dataset import (
    DatasetIsolationError,
    iter_canonical_records,
)
from utils.cae_language_grounding.direct_llm import (
    expand_and_verify_direct_selection,
    make_direct_inference_input,
)
from utils.cae_language_grounding.evaluation import MemoryCache
from utils.cae_language_grounding.llm_adapter import (
    LLMRuntimeConfig,
    OpenAICompatibleJSONClient,
)
from utils.cae_language_grounding.records import (
    INFERENCE_FORBIDDEN_KEYS,
    iter_language_examples,
)


def _walk_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


class P4A1LLMAndLeakageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.example = next(iter(iter_language_examples("val")))
        cls.cache = MemoryCache()
        cls.memory, _ = cls.cache.get(
            str(cls.example.source_ref["graph_path"])
        )

    def test_test_split_is_fail_closed_before_h4(self):
        with self.assertRaises(DatasetIsolationError):
            next(iter(iter_canonical_records("test")))

    def test_structured_adapter_retries_once_without_silent_repair(self):
        requests = []
        valid = copy.deepcopy(self.example.oracle_intent_result)

        def transport(url, headers, payload, timeout):
            requests.append((url, dict(headers), copy.deepcopy(payload), timeout))
            content = "not-json" if len(requests) == 1 else json.dumps(valid)
            return {
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7},
            }

        config = LLMRuntimeConfig(
            base_url="http://runtime.invalid/v1",
            model="open-source-model",
            api_key="secret-not-for-logs",
            seed=17,
            timeout_seconds=2,
            revision="rev",
            dtype="bf16",
            quantization="none",
            backend="mock",
            gpu="mock-gpu",
            context_length=4096,
            constrained_decoding=True,
        )
        result = OpenAICompatibleJSONClient(
            config, transport=transport
        ).infer_structured(self.example.inference_input())
        self.assertFalse(result.schema_valid_first_round)
        self.assertTrue(result.schema_valid_after_retry)
        self.assertEqual(result.retry_count, 1)
        self.assertEqual(result.backend_request_count, 2)
        self.assertEqual(result.value, valid)
        self.assertIn("response_format", requests[0][2])
        self.assertNotIn("secret-not-for-logs", json.dumps(config.public_metadata()))

    def test_language_inference_input_contains_only_text_and_language(self):
        value = self.example.inference_input()
        self.assertEqual(set(value), {"text", "language"})
        self.assertFalse(INFERENCE_FORBIDDEN_KEYS & set(_walk_keys(value)))

    def test_direct_memory_has_no_raw_face_or_answer_fields(self):
        value = make_direct_inference_input(
            text=self.example.text,
            language=self.example.language,
            memory=self.memory,
        )
        keys = set(_walk_keys(value))
        self.assertFalse(INFERENCE_FORBIDDEN_KEYS & keys)
        self.assertNotIn("face_indices", keys)
        self.assertNotIn("complete_instance_face_indices", keys)
        self.assertNotIn("source_ref", keys)

    def test_direct_range_verifier_rejects_unknown_object(self):
        selection = {
            "schema_version": "topocae-direct-selection/1.0.0",
            "status": "VALID",
            "error_code": None,
            "message": "",
            "selections": [
                {
                    "selection_id": "op1",
                    "operation_type": "fixed_support",
                    "selected_object_ids": ["obj-9999"],
                    "face_role": "feature_region",
                }
            ],
        }
        result = expand_and_verify_direct_selection(selection, self.memory)
        self.assertEqual(result["status"], "INVALID")
        self.assertEqual(result["error_code"], "E_DIRECT_INVALID_OBJECT_ID")
        self.assertEqual(
            result["verifier"]["illegal_object_id_count"], 1
        )
        self.assertEqual(result["verifier"]["illegal_face_id_count"], 0)
        self.assertEqual(result["verifier"]["silent_fallback_count"], 0)


if __name__ == "__main__":
    unittest.main()
