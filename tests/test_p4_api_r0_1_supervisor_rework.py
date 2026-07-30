from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib
import json
from pathlib import Path
import tempfile
import unittest

from p4_api_r0_1_audit import verify_full_freeze
from utils.cae_language_grounding.llm_adapter import (
    LLMRuntimeConfig,
    OpenAICompatibleJSONClient,
)
from utils.cae_language_grounding.run_identity import (
    RunIdentityError,
    build_run_manifest,
    prepare_immutable_run_directory,
)


SCHEMA = {
    "type": "object",
    "required": ["answer"],
    "properties": {"answer": {"type": "string"}},
}


def valid(value):
    ok = isinstance(value, dict) and isinstance(value.get("answer"), str)
    return {"valid": ok, "errors": [] if ok else [{"code": "bad"}]}


def response():
    return {
        "model": "served-model",
        "choices": [{
            "finish_reason": "stop",
            "message": {"content": '{"answer":"ok"}'},
        }],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1},
    }


def config(cache_dir: str, **updates):
    base = LLMRuntimeConfig(
        base_url="https://one.invalid/v1",
        model="model-a",
        api_key=None,
        seed=3,
        timeout_seconds=1,
        revision="rev-a",
        dtype="none",
        quantization="none",
        backend="backend-a",
        gpu="none",
        context_length=2048,
        constrained_decoding=True,
        cache_dir=cache_dir,
    )
    return replace(base, **updates)


def infer(client):
    return client.infer_json(
        inference_input={"text": "offline", "language": "en-US"},
        system_prompt="Return JSON.",
        schema_name="structured",
        schema=SCHEMA,
        validator=valid,
    )


class CacheProvenanceTests(unittest.TestCase):
    def test_endpoint_and_backend_isolate_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = []

            def transport(*_):
                calls.append(None)
                return response()

            first = infer(OpenAICompatibleJSONClient(
                config(directory), transport=transport
            ))
            hit = infer(OpenAICompatibleJSONClient(
                config(directory), transport=transport
            ))
            endpoint_change = infer(OpenAICompatibleJSONClient(
                config(directory, base_url="https://two.invalid/v1"),
                transport=transport,
            ))
            backend_change = infer(OpenAICompatibleJSONClient(
                config(directory, backend="backend-b"), transport=transport
            ))
            self.assertFalse(first.cache_hit)
            self.assertTrue(hit.cache_hit)
            self.assertFalse(endpoint_change.cache_hit)
            self.assertFalse(backend_change.cache_hit)
            self.assertEqual(len(calls), 3)
            self.assertEqual(len({
                first.request_hash,
                endpoint_change.request_hash,
                backend_change.request_hash,
            }), 3)

    def test_public_endpoint_metadata_never_contains_credentials(self):
        cfg = config(
            "cache",
            base_url="https://fixture-user:fixture-pass@one.invalid/v1?api_key=fixture-secret",
        )
        encoded = json.dumps(cfg.public_metadata())
        self.assertNotIn("fixture-user", encoded)
        self.assertNotIn("fixture-pass", encoded)
        self.assertNotIn("fixture-secret", encoded)
        self.assertIn("endpoint_sha256", cfg.public_metadata())


class ImmutableRunTests(unittest.TestCase):
    def manifest(self, **updates):
        values = {
            "endpoint": "https://one.invalid/v1",
            "backend": "backend-a",
            "provider_profile": "openai_json_schema",
            "model": "model-a",
            "revision": "rev-a",
            "prompt": "prompt",
            "schema": SCHEMA,
            "runtime_config": {"temperature": 0, "seed": 3},
            "baseline": "structured",
            "split": "val",
            "offset": 0,
            "limit": 1,
            "shard_index": 0,
            "shard_count": 1,
        }
        values.update(updates)
        return build_run_manifest(**values)

    def test_configuration_changes_create_distinct_output_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            first = self.manifest()
            model_change = self.manifest(model="model-b")
            endpoint_change = self.manifest(endpoint="https://two.invalid/v1")
            first_path = prepare_immutable_run_directory(directory, first)
            resumed_path = prepare_immutable_run_directory(directory, first)
            model_path = prepare_immutable_run_directory(directory, model_change)
            endpoint_path = prepare_immutable_run_directory(directory, endpoint_change)
            self.assertEqual(first_path, resumed_path)
            self.assertEqual(len({first_path, model_path, endpoint_path}), 3)

    def test_manifest_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.manifest()
            run_dir = prepare_immutable_run_directory(directory, manifest)
            path = run_dir / "run_manifest.json"
            changed = dict(manifest)
            changed["identity_sha256"] = "0" * 64
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(
                RunIdentityError, "E_RUN_ID_MANIFEST_MISMATCH"
            ):
                prepare_immutable_run_directory(directory, manifest)


class CanonicalImportTests(unittest.TestCase):
    def test_canonical_and_legacy_modules_load_independently(self):
        canonical = importlib.import_module(
            "utils.cae_language_grounding.llm_adapter"
        )
        legacy = importlib.import_module(
            "utils.cae_language_grounding.llm_adapter_legacy"
        )
        self.assertEqual(Path(canonical.__file__).name, "llm_adapter.py")
        self.assertEqual(Path(legacy.__file__).name, "llm_adapter_legacy.py")
        self.assertNotEqual(canonical.__file__, legacy.__file__)
        self.assertFalse((Path(canonical.__file__).parent / "llm_adapter").exists())

        calls = []
        legacy_config = legacy.LLMRuntimeConfig(
            base_url="https://legacy.invalid/v1",
            model="legacy-model",
            api_key=None,
            seed=1,
            timeout_seconds=1,
            revision="legacy",
            dtype="none",
            quantization="none",
            backend="mock",
            gpu="none",
            context_length=1024,
            constrained_decoding=True,
        )
        outcome = legacy.OpenAICompatibleJSONClient(
            legacy_config,
            transport=lambda *_: calls.append(None) or response(),
        ).infer_json(
            inference_input={"text": "offline", "language": "en-US"},
            system_prompt="Return JSON.",
            schema_name="legacy",
            schema=SCHEMA,
            validator=valid,
        )
        self.assertTrue(outcome.schema_valid_after_retry)
        self.assertEqual(len(calls), 1)


class FullFreezeAuditTests(unittest.TestCase):
    def test_recomputes_bytes_and_sha256_with_stable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset = root / "asset.bin"
            asset.write_bytes(b"abc")
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps({"assets": [{
                "path": "asset.bin",
                "bytes": 3,
                "sha256": hashlib.sha256(b"abc").hexdigest(),
            }]}), encoding="utf-8")
            passed = verify_full_freeze(
                (("PX", manifest_path),), root=root
            )
            self.assertEqual(passed["status"], "PASS")
            self.assertEqual(passed["stages"][0]["matching_count"], 1)

            asset.write_bytes(b"abd")
            failed = verify_full_freeze(
                (("PX", manifest_path),), root=root
            )
            self.assertEqual(failed["status"], "FAIL")
            self.assertEqual(
                failed["stages"][0]["changes"][0]["error_code"],
                "E_FROZEN_ASSET_SHA256",
            )


if __name__ == "__main__":
    unittest.main()
