from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from utils.cae_intent import load_feature_role_policy
from utils.cae_language_grounding.dataset import (
    DatasetIsolationError,
    DecodeAudit,
    iter_canonical_records,
)
from utils.cae_language_grounding.memory import (
    build_feature_object_memory,
    canonical_memory_bytes,
    validate_feature_object_memory,
)


ROOT = Path(__file__).resolve().parents[1]
SUPPLEMENT_ROOT = ROOT / "output_generation_p3_a1r_multi_instance_formal_v1"
FORBIDDEN_KEYS = {
    "query",
    "ground_truth",
    "target_instance_ids",
    "target_face_indices",
    "semantic_evidence",
    "uniqueness_contract",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _walk_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


class FeatureObjectMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = load_feature_role_policy()
        cls.train_samples = {}
        split_path = SUPPLEMENT_ROOT / "splits" / "train.jsonl"
        with split_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                cls.train_samples.setdefault(record["feature_type"], record)

    def _memory_for_sample(self, sample):
        graph_path = Path(sample["graph_path"])
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        return build_feature_object_memory(
            graph,
            source_graph_sha256=_sha256(graph_path),
            policy=self.policy,
        )

    def test_all_17_features_and_all_policy_roles_are_covered(self):
        self.assertEqual(set(self.train_samples), set(self.policy["features"]))
        covered_roles = set()
        for feature_type, sample in sorted(self.train_samples.items()):
            memory = self._memory_for_sample(sample)
            self.assertTrue(validate_feature_object_memory(memory)["valid"])
            for obj in memory["objects"]:
                if obj["canonical_feature_type"] != feature_type:
                    continue
                actual_roles = {
                    role["face_role"] for role in obj["face_role_map"]
                }
                expected_roles = {
                    role["role"]
                    for role in self.policy["features"][feature_type][
                        "selectable_roles"
                    ]
                }
                self.assertEqual(actual_roles, expected_roles)
                covered_roles.update(
                    (feature_type, role) for role in actual_roles
                )
                maximum = memory["part"]["face_index_contract"]["maximum"]
                for role in obj["face_role_map"]:
                    self.assertTrue(
                        all(0 <= index <= maximum for index in role["face_indices"])
                    )
                    if role["selection_status"].startswith("unsupported"):
                        self.assertEqual(role["status"], "UNSUPPORTED")
                        self.assertEqual(role["face_indices"], [])
                        self.assertTrue(role["abstention_required"])
        expected_count = sum(
            len(feature["selectable_roles"])
            for feature in self.policy["features"].values()
        )
        self.assertEqual(expected_count, 60)
        self.assertEqual(len(covered_roles), 60)

    def test_memory_is_byte_deterministic_and_query_independent(self):
        sample = self.train_samples["counterbore_hole"]
        first = self._memory_for_sample(sample)
        second = self._memory_for_sample(sample)
        self.assertEqual(canonical_memory_bytes(first), canonical_memory_bytes(second))
        self.assertFalse(FORBIDDEN_KEYS & set(_walk_keys(first)))
        self.assertTrue(
            all(obj["object_id"].startswith("obj-") for obj in first["objects"])
        )
        self.assertFalse(
            any(
                str(face) == obj["object_id"]
                for obj in first["objects"]
                for face in obj["topology"]["complete_instance_face_indices"]
            )
        )

    def test_rotation_and_coordinate_fields_are_explicit(self):
        memory = self._memory_for_sample(
            self.train_samples["through_hole_round"]
        )
        invariance = memory["field_invariance"]
        self.assertIn(
            "/objects/*/descriptors/distance_to_part_reference",
            invariance["rotation_invariant"],
        )
        self.assertIn(
            "/objects/*/descriptors/centroid",
            invariance["rigid_rotation_covariant"],
        )
        self.assertTrue(memory["part"]["reference"]["coordinate_dependent"])

    def test_legacy_singleton_graph_gets_labeled_bbox_reference(self):
        path = (
            ROOT
            / "output_generation_p1_b_corrected_balanced_5000_v3"
            / "graphs"
            / "legacy_01929.json"
        )
        graph = json.loads(path.read_text(encoding="utf-8"))
        memory = build_feature_object_memory(
            graph,
            source_graph_sha256=_sha256(path),
            policy=self.policy,
        )
        self.assertEqual(
            memory["part"]["reference"]["kind"], "part_bbox_center"
        )
        self.assertEqual(
            memory["objects"][0]["descriptor_contract"]["availability"],
            "derived_legacy_singleton",
        )
        self.assertIsNone(
            memory["objects"][0]["descriptors"]["comparison_metric"]
        )


class DatasetIsolationTests(unittest.TestCase):
    def test_test_split_is_rejected_before_h4(self):
        with self.assertRaises(DatasetIsolationError):
            list(iter_canonical_records("test"))

    def test_val_iteration_never_decodes_test_lines(self):
        audit = DecodeAudit("canonical", "val")
        records = list(iter_canonical_records("val", audit=audit))
        self.assertEqual(len(records), 2440)
        self.assertEqual(audit.decoded_val_lines, 2440)
        self.assertEqual(audit.decoded_test_lines, 0)
        self.assertEqual(audit.opaque_test_lines_crossed, 340)
        self.assertTrue(
            all(
                record.get("source_ref", {}).get("dataset_split") == "val"
                for record in records
            )
        )


if __name__ == "__main__":
    unittest.main()
