from __future__ import annotations

import json
from pathlib import Path
import unittest

from utils.cae_language_grounding.deepseek_strict_transport import (
    build_transport_schema,
    canonical_to_transport,
    transport_to_canonical,
    validate_deepseek_strict_subset_schema,
    validate_transport_instance,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "cae_ir_v1" / "cases.jsonl"


def walk(value):
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)


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


class R1E1DialectRepairTests(unittest.TestCase):
    def test_provider_schema_uses_only_singular_def_dialect(self):
        schema = build_transport_schema()
        dictionaries = [node for node in walk(schema) if isinstance(node, dict)]
        self.assertIn("$def", schema)
        self.assertFalse(any("$defs" in node for node in dictionaries))

    def test_all_refs_use_singular_def_prefix(self):
        refs = [
            node["$ref"]
            for node in walk(build_transport_schema())
            if isinstance(node, dict) and "$ref" in node
        ]
        self.assertTrue(refs)
        self.assertTrue(all(ref.startswith("#/$def/") for ref in refs), refs[:3])

    def test_title_and_unsupported_keywords_are_absent(self):
        schema = build_transport_schema()
        dictionaries = [node for node in walk(schema) if isinstance(node, dict)]
        self.assertFalse(any("title" in node for node in dictionaries))
        validation = validate_deepseek_strict_subset_schema(schema)
        self.assertTrue(validation["valid"], validation["errors"][:3])
        self.assertEqual([], validation["errors"])

    def test_local_jsonschema_adapter_does_not_mutate_provider_schema(self):
        schema_before = build_transport_schema()
        canonical = valid_result()
        transport = canonical_to_transport(canonical)
        self.assertTrue(validate_transport_instance(transport)["valid"])
        self.assertEqual(canonical, transport_to_canonical(transport))
        self.assertEqual(schema_before, build_transport_schema())
        self.assertIn("$def", build_transport_schema())
        self.assertNotIn("$defs", build_transport_schema())


if __name__ == "__main__":
    unittest.main()
