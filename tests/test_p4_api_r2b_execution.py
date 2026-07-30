from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from utils.cae_language_grounding.llm_adapter_r0 import LLMTransportError
from utils.cae_query_frame_runtime import runtime_examples

import run_p4_api_r2b_20_val_smoke_v2 as r2b


def response(content: str, *, model: str = "deepseek-v4-pro") -> dict:
    return {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        "model": model,
    }


class QueueTransport:
    def __init__(self, values):
        self.values = list(values)
        self.calls = []

    def __call__(self, url, headers, payload, timeout):
        self.calls.append((url, headers, payload, timeout))
        value = self.values.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


class R2BExecutionTests(unittest.TestCase):
    def test_structural_failure_repairs_once_without_previous_output(self) -> None:
        valid = runtime_examples()[0]["frame"]
        transport = QueueTransport(
            [response("{}"), response(json.dumps(valid, ensure_ascii=False))]
        )
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            outcome = r2b.R2BClient(
                api_key="test-key",
                ledger=r2b.RequestLedger(root / "ledger.json"),
                cache_dir=root / "cache",
                transport=transport,
            ).infer(
                sample_id="s1",
                inference_input={"text": "Apply pressure.", "language": "en-US"},
            )
        self.assertTrue(outcome["post_repair_frame_valid"])
        self.assertEqual(outcome["repair_count"], 1)
        self.assertEqual(len(transport.calls), 2)
        repair_messages = transport.calls[1][2]["messages"]
        self.assertEqual([row["role"] for row in repair_messages], ["system", "user"])
        self.assertNotIn("previous", repair_messages[1]["content"].lower())

    def test_cache_replay_requires_completed_ledger_and_costs_no_new_request(self) -> None:
        valid = runtime_examples()[0]["frame"]
        transport = QueueTransport([response(json.dumps(valid, ensure_ascii=False))])
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            ledger = r2b.RequestLedger(root / "ledger.json")
            first = r2b.R2BClient(
                api_key="test-key",
                ledger=ledger,
                cache_dir=root / "cache",
                transport=transport,
            ).infer(
                sample_id="s1",
                inference_input={"text": "Apply pressure.", "language": "en-US"},
            )
            replay_transport = QueueTransport([])
            replay = r2b.R2BClient(
                api_key="test-key",
                ledger=r2b.RequestLedger(root / "ledger.json"),
                cache_dir=root / "cache",
                transport=replay_transport,
            ).infer(
                sample_id="s1",
                inference_input={"text": "Apply pressure.", "language": "en-US"},
            )
            ledger_value = json.loads((root / "ledger.json").read_text(encoding="utf-8"))
        self.assertFalse(first["cache_hit"])
        self.assertTrue(replay["cache_hit"])
        self.assertEqual(len(replay_transport.calls), 0)
        self.assertEqual(ledger_value["counters"]["backend_requests"], 1)
        self.assertEqual(ledger_value["counters"]["input_tokens"], 100)
        self.assertEqual(ledger_value["counters"]["output_tokens"], 50)

    def test_transport_failure_does_not_repair(self) -> None:
        transport = QueueTransport(
            [LLMTransportError("E_HTTP_401", status_code=401, retryable=False)]
        )
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            outcome = r2b.R2BClient(
                api_key="test-key",
                ledger=r2b.RequestLedger(root / "ledger.json"),
                cache_dir=root / "cache",
                transport=transport,
            ).infer(
                sample_id="s1",
                inference_input={"text": "Apply pressure.", "language": "en-US"},
            )
        self.assertEqual(outcome["failure_phase"], "transport")
        self.assertEqual(outcome["repair_count"], 0)
        self.assertEqual(len(transport.calls), 1)

    def test_actual_model_mismatch_fails_closed_without_repair(self) -> None:
        valid = runtime_examples()[0]["frame"]
        transport = QueueTransport(
            [response(json.dumps(valid), model="unexpected-model")]
        )
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            outcome = r2b.R2BClient(
                api_key="test-key",
                ledger=r2b.RequestLedger(root / "ledger.json"),
                cache_dir=root / "cache",
                transport=transport,
            ).infer(
                sample_id="s1",
                inference_input={"text": "Apply pressure.", "language": "en-US"},
            )
        self.assertEqual(outcome["failure_code"], "R2B_E_ACTUAL_MODEL_MISMATCH")
        self.assertEqual(outcome["repair_count"], 0)


if __name__ == "__main__":
    unittest.main()

