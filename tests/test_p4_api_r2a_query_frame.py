from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "p4_api_r2a_compact_query_frame_v1"
SCHEMA_PATH = REPORT / "query_frame.schema.json"


def valid_mesh_frame() -> dict:
    return {
        "frame_version": "cae-query-frame/1.0.0",
        "status": "valid",
        "language": "en-US",
        "operations": [{
            "action": "local_mesh_size",
            "feature": "through_hole_round",
            "face_role": "feature_region",
            "selector": "exactly_one",
            "mesh_size": {"value": 0.75, "unit": "mm"},
        }],
    }


class QueryFrameContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from utils import cae_query_frame

        cls.api = cae_query_frame
        cls.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    def test_schema_is_small_closed_draft_2020_12_object(self):
        self.assertLessEqual(len(SCHEMA_PATH.read_bytes()), 10 * 1024)
        self.assertEqual("https://json-schema.org/draft/2020-12/schema", self.schema["$schema"])
        self.assertEqual("object", self.schema["type"])
        self.assertIs(self.schema["additionalProperties"], False)
        Draft202012Validator.check_schema(self.schema)

    def test_schema_has_no_external_or_recursive_references(self):
        references = []

        def walk(value):
            if isinstance(value, dict):
                if "$ref" in value:
                    references.append(value["$ref"])
                for child in value.values():
                    walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)

        walk(self.schema)
        self.assertTrue(references)
        self.assertTrue(all(ref.startswith("#/$defs/") for ref in references))

        definitions = self.schema["$defs"]
        graph = {name: set() for name in definitions}
        for name, definition in definitions.items():
            local = []
            walk_definition = lambda value: None

            def walk_definition(value):
                if isinstance(value, dict):
                    ref = value.get("$ref")
                    if isinstance(ref, str) and ref.startswith("#/$defs/"):
                        local.append(ref.rsplit("/", 1)[-1])
                    for child in value.values():
                        walk_definition(child)
                elif isinstance(value, list):
                    for child in value:
                        walk_definition(child)

            walk_definition(definition)
            graph[name].update(local)

        def visit(node, active, finished):
            if node in active:
                self.fail(f"recursive schema definition: {node}")
            if node in finished:
                return
            active.add(node)
            for child in graph[node]:
                visit(child, active, finished)
            active.remove(node)
            finished.add(node)

        finished = set()
        for name in graph:
            visit(name, set(), finished)

    def test_every_declared_field_has_chinese_description_and_objects_are_closed(self):
        def walk(value, path=()):
            if isinstance(value, dict):
                if value.get("type") == "object":
                    self.assertIs(value.get("additionalProperties"), False, path)
                for name, field in value.get("properties", {}).items():
                    description = field.get("description", "")
                    self.assertTrue(any("\u4e00" <= char <= "\u9fff" for char in description), (path, name))
                for name, child in value.items():
                    walk(child, path + (name,))
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    walk(child, path + (index,))

        walk(self.schema)

    def test_contract_has_three_statuses_seven_actions_and_no_identity_fields(self):
        serialized = json.dumps(self.schema, ensure_ascii=False).lower()
        for status in ("valid", "needs_clarification", "invalid"):
            self.assertIn(f'"{status}"', serialized)
        for action in (
            "fixed_support", "displacement", "pressure", "surface_traction",
            "material_assignment", "global_mesh_size", "local_mesh_size",
        ):
            self.assertIn(f'"{action}"', serialized)
        for forbidden in (
            "face_id", "face_index", "face_indices", "feature_instance_id",
            "instance_id", "ground_truth", "source_identity", "source_ref",
        ):
            self.assertNotIn(f'"{forbidden}"', serialized)

    def test_valid_single_operation_compiles_to_exact_frozen_result(self):
        actual = self.api.compile_query_frame(valid_mesh_frame())
        expected = {
            "schema_version": "cae-intent-result/1.0.0",
            "status": "VALID",
            "intent": {
                "schema_version": "cae-intent/1.0.0",
                "analysis_type": "static_structural",
                "operations": [{
                    "operation_id": "op1",
                    "operation_type": "local_mesh_size",
                    "target": {
                        "feature_type": "through_hole_round",
                        "quantifier": "exactly_one",
                        "face_role": "feature_region",
                        "filters": {},
                    },
                    "parameters": {
                        "size": {"dimension": "length", "value": 0.75, "unit": "mm"},
                    },
                }],
                "source_language": "en-US",
                "assumptions": [],
                "unit_system": {
                    "length": "mm", "area": "mm2", "force": "N",
                    "pressure": "MPa", "displacement": "mm", "angle": "deg",
                },
                "normalization_audit": [],
            },
        }
        self.assertEqual(expected, actual)

    def test_multi_operation_support_generates_stable_operation_ids(self):
        frame = valid_mesh_frame()
        frame["operations"] = [
            {
                "action": "material_assignment",
                "material": {
                    "name": "steel",
                    "youngs_modulus": {"value": 210, "unit": "GPa"},
                    "poisson_ratio": 0.3,
                },
            },
            {"action": "global_mesh_size", "mesh_size": {"value": 5, "unit": "mm"}},
            copy.deepcopy(valid_mesh_frame()["operations"][0]),
        ]
        result = self.api.compile_query_frame(frame)
        self.assertEqual(["op1", "op2", "op3"], [op["operation_id"] for op in result["intent"]["operations"]])
        self.assertEqual(
            ["material_assignment", "global_mesh_size", "local_mesh_size"],
            [op["operation_type"] for op in result["intent"]["operations"]],
        )

    def test_nonvalid_states_compile_without_partial_intent(self):
        clarification = {
            "frame_version": "cae-query-frame/1.0.0",
            "status": "needs_clarification",
            "error_code": "E_MISSING_INFORMATION",
            "error_path": "/operations/0/parameters/direction",
            "missing_information": [{
                "code": "M_DIRECTION",
                "path": "/operations/0/parameters/direction",
                "description": "Pressure direction is required.",
            }],
            "ambiguities": [],
            "clarification_questions": [{
                "key": "q1",
                "path": "/operations/0/parameters/direction",
                "question": "Which pressure direction should be used?",
            }],
        }
        result = self.api.compile_query_frame(clarification)
        self.assertEqual("NEEDS_CLARIFICATION", result["status"])
        self.assertNotIn("intent", result)

        invalid = {
            "frame_version": "cae-query-frame/1.0.0",
            "status": "invalid",
            "error_code": "E_UNSUPPORTED_ANALYSIS",
            "error_path": "/analysis_type",
            "missing_information": [],
            "ambiguities": [],
            "clarification_questions": [],
        }
        self.assertEqual("INVALID", self.api.compile_query_frame(invalid)["status"])

    def test_fail_closed_errors_have_stable_codes(self):
        cases = []
        wrong_root = []
        cases.append((wrong_root, "QF_E_JSON_TYPE"))
        invalid_enum = valid_mesh_frame()
        invalid_enum["operations"][0]["action"] = "mesh_it_somehow"
        cases.append((invalid_enum, "QF_E_INVALID_ENUM"))
        missing = valid_mesh_frame()
        del missing["operations"][0]["mesh_size"]
        cases.append((missing, "QF_E_REQUIRED_FIELD"))
        unexpected = valid_mesh_frame()
        unexpected["unexpected_field"] = 7
        cases.append((unexpected, "QF_E_UNEXPECTED_FIELD"))
        conflict = valid_mesh_frame()
        conflict["operations"][0]["magnitude"] = {"value": 5, "unit": "MPa"}
        cases.append((conflict, "QF_E_CONFLICT"))

        for frame, expected_code in cases:
            with self.subTest(expected_code=expected_code):
                with self.assertRaises(self.api.QueryFrameCompileError) as caught:
                    self.api.compile_query_frame(frame)
                self.assertEqual(expected_code, caught.exception.code)
                self.assertTrue(caught.exception.field_path.startswith("/"))

    def test_same_input_has_byte_identical_output(self):
        frame = valid_mesh_frame()
        first = self.api.compile_query_frame_json(frame)
        second = self.api.compile_query_frame_json(copy.deepcopy(frame))
        self.assertIsInstance(first, bytes)
        self.assertEqual(first, second)
        self.assertEqual(self.api.compile_query_frame(frame), json.loads(first.decode("utf-8")))

    def test_inverse_mapping_is_semantically_exact(self):
        canonical = self.api.compile_query_frame(valid_mesh_frame())
        restored_frame = self.api.canonical_to_query_frame(canonical)
        self.assertEqual(canonical, self.api.compile_query_frame(restored_frame))


if __name__ == "__main__":
    unittest.main()
