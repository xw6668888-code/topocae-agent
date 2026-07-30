from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import unittest

from utils.cae_language_grounding.dataset import iter_canonical_records
from utils.cae_language_grounding.grounder import (
    GroundingError,
    ground_intent_result,
    validate_grounding_result,
)
from utils.cae_language_grounding.memory import build_feature_object_memory


ROOT = Path(__file__).resolve().parents[1]


def _load_memory(record):
    path = Path(record["source_ref"]["graph_path"])
    raw = path.read_bytes()
    return build_feature_object_memory(
        json.loads(raw.decode("utf-8")),
        source_graph_sha256=hashlib.sha256(raw).hexdigest(),
    )


def _target_by_operation(record):
    return {
        target["operation_id"]: target
        for target in record.get("ground_truth", {}).get("targets", [])
    }


class DeterministicGrounderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = list(iter_canonical_records("train"))

    def _assert_matches_ground_truth(self, record):
        memory = _load_memory(record)
        result = ground_intent_result(record["intent_result"], memory)
        self.assertTrue(validate_grounding_result(result)["valid"])
        self.assertEqual(result["status"], record["intent_result"]["status"])
        if result["status"] != "VALID":
            self.assertEqual(result["operations"], [])
            return
        expected = _target_by_operation(record)
        self.assertEqual(len(result["operations"]), len(expected))
        for operation in result["operations"]:
            target = expected[operation["operation_id"]]
            self.assertEqual(operation["status"], "VALID")
            self.assertEqual(
                operation["target_contract"], target["target_contract"]
            )
            if target["target_contract"] == "whole_part":
                self.assertEqual(operation["selected_face_indices"], [])
            else:
                self.assertEqual(
                    operation["selected_face_indices"],
                    target["target_face_indices"],
                )
            self.assertFalse(operation["verifier"]["silent_fallback_used"])

    def test_all_required_quantifiers_match_oracle_faces(self):
        required = {
            "all",
            "count",
            "exactly_one",
            "largest",
            "smallest",
            "nearest",
        }
        found = {}
        for record in self.records:
            if record["intent_result"]["status"] != "VALID":
                continue
            for operation in record["intent_result"]["intent"]["operations"]:
                quantifier = operation.get("target", {}).get("quantifier")
                if quantifier in required and quantifier not in found:
                    found[quantifier] = record
            if set(found) == required:
                break
        self.assertEqual(set(found), required)
        for record in found.values():
            self._assert_matches_ground_truth(record)

    def test_every_quantity_filter_matches_oracle_faces(self):
        required = {
            "diameter",
            "radius",
            "depth",
            "area",
            "length",
            "width",
            "thickness",
        }
        found = {}
        for record in self.records:
            if record["intent_result"]["status"] != "VALID":
                continue
            for operation in record["intent_result"]["intent"]["operations"]:
                for name in operation.get("target", {}).get("filters", {}):
                    if name in required and name not in found:
                        found[name] = record
            if set(found) == required:
                break
        self.assertEqual(set(found), required)
        for record in found.values():
            self._assert_matches_ground_truth(record)

    def test_three_operation_intent_and_whole_part_targets(self):
        record = next(
            record
            for record in self.records
            if record["intent_result"]["status"] == "VALID"
            and len(record["intent_result"]["intent"]["operations"]) == 3
        )
        self._assert_matches_ground_truth(record)

    def test_nonvalid_p2_statuses_do_not_run_grounding(self):
        for status in ("NEEDS_CLARIFICATION", "INVALID"):
            record = next(
                item
                for item in self.records
                if item["intent_result"]["status"] == status
            )
            self._assert_matches_ground_truth(record)

    def test_tied_largest_requests_clarification(self):
        graph_path = (
            ROOT
            / "output_generation_p3_a1r_multi_instance_formal_v1"
            / "graphs"
            / "p3a1r_formal_01_through_hole_round_030.json"
        )
        raw = graph_path.read_bytes()
        memory = build_feature_object_memory(
            json.loads(raw.decode("utf-8")),
            source_graph_sha256=hashlib.sha256(raw).hexdigest(),
        )
        source = next(
            record
            for record in self.records
            if record["intent_result"]["status"] == "VALID"
            and record["intent_result"]["intent"]["operations"][0]
            .get("target", {})
            .get("quantifier")
            == "largest"
            and record["intent_result"]["intent"]["operations"][0]["target"][
                "feature_type"
            ]
            == "through_hole_round"
        )
        intent_result = copy.deepcopy(source["intent_result"])
        result = ground_intent_result(intent_result, memory)
        self.assertEqual(result["status"], "NEEDS_CLARIFICATION")
        self.assertEqual(result["error_code"], "E_DESCRIPTOR_TIE")

    def test_invalid_p2_result_is_rejected(self):
        record = next(
            item
            for item in self.records
            if item["intent_result"]["status"] == "VALID"
        )
        invalid = copy.deepcopy(record["intent_result"])
        invalid["intent"]["operations"][0]["target"]["face_role"] = "not_a_role"
        with self.assertRaises(GroundingError):
            ground_intent_result(invalid, _load_memory(record))


if __name__ == "__main__":
    unittest.main()
