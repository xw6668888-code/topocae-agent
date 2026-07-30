from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from p1_a2_protocol import (
    CONFIG_ROOT,
    PREHOLDOUT_SELECTION,
    PREHOLDOUT_SIDECAR,
    REPORT_ROOT,
    REPO_ROOT,
    TARGET_CLASSES,
    VARIANT_NAMES,
    face_semantic_compatible,
    load_json,
    select_variant,
    sha256_file,
    variant_stage_count,
)
from utils.topology_postprocess import remap_faces_to_macro


def selection_metrics(
    *,
    object_macro_f1: float,
    face: float = 0.95,
    semantic: float = 0.91,
    recall: float = 0.8,
    support: int = 20,
    rotation: float = 1.0,
) -> dict:
    return {
        "synthetic_val": {
            "face": {"macro_f1": face},
            "semantic": {"macro_f1": semantic},
            "object": {
                "macro": {
                    "precision": object_macro_f1,
                    "recall": object_macro_f1,
                    "f1": object_macro_f1,
                },
                "micro": {"f1": object_macro_f1},
                "exact_instance_ratio": object_macro_f1,
                "instance_iou": object_macro_f1,
                "per_class": {
                    "web": {"recall": recall, "support": support}
                },
            },
        },
        "synthetic_rotation_val": {
            "class_consistency": rotation,
            "object_correspondence_consistency": rotation,
            "face_role_consistency": rotation,
        },
    }


class PaperV1P1A2UnitTests(unittest.TestCase):
    def test_five_stage_enable_combinations_are_frozen(self) -> None:
        models = load_json(CONFIG_ROOT / "models.json")
        self.assertEqual(tuple(models["variants"]), VARIANT_NAMES)
        self.assertEqual(models["variants"]["A0"], ["raw_gnn"])
        self.assertEqual(models["variants"]["A1"], ["raw_gnn", "topology"])
        self.assertEqual(models["variants"]["A2"], ["raw_gnn", "candidate"])
        self.assertEqual(
            models["variants"]["A3"], ["raw_gnn", "candidate", "topology"]
        )
        self.assertEqual(
            models["variants"]["A4"],
            ["val_best_basic", "rib_proposer", "rib_boundary"],
        )
        self.assertEqual(
            [variant_stage_count(name, "A0") for name in VARIANT_NAMES],
            [1, 2, 2, 3, 3],
        )

    def test_unsupported_macro_is_not_in_candidate_label_space(self) -> None:
        metadata = load_json(
            REPO_ROOT
            / "reports"
            / "paper_v1_p1"
            / "training_data"
            / "candidate_topology_clean"
            / "metadata.json"
        )
        target_macros = {
            name.removeprefix("proposal_macro_")
            for name in metadata["feature_schema"]
            if name.startswith("proposal_macro_")
        }
        self.assertNotIn("stepped_hole_round", target_macros)
        self.assertNotIn("rectangular_boss", target_macros)
        self.assertNotIn("chamfer", target_macros)

    def test_semantic_macro_compatibility_after_supported_remap(self) -> None:
        faces = [
            {
                "idx": 0,
                "macro_feature_id": 0,
                "macro_feature": "stock",
                "semantic_subtype_id": 0,
                "semantic_subtype": "stock_face",
            }
        ]
        graph = {
            "nodes": [
                {
                    "idx": 0,
                    "primitive_surface": "cylinder",
                    "attributes": {"physical": {}, "specific": {}},
                }
            ]
        }
        remap_faces_to_macro(
            faces, [0], "through_hole_round", graph, ["unit_test_contract"]
        )
        self.assertTrue(face_semantic_compatible(faces[0]))

    def test_val_only_selection_constraints_and_fallback(self) -> None:
        evaluation = load_json(CONFIG_ROOT / "evaluation.json")
        metrics = {
            "A0": selection_metrics(object_macro_f1=0.80),
            "A1": selection_metrics(object_macro_f1=0.90, face=0.94),
            "A2": selection_metrics(object_macro_f1=0.91, recall=0.0),
            "A3": selection_metrics(object_macro_f1=0.92, semantic=0.89),
            "A4": selection_metrics(object_macro_f1=0.93, rotation=0.99),
        }
        selection = select_variant(
            VARIANT_NAMES, metrics, evaluation, a4_base="A0"
        )
        self.assertEqual(selection["selected_variant"], "A0")
        self.assertEqual(selection["eligible_variants"], ["A0"])
        datasets = load_json(CONFIG_ROOT / "datasets.json")
        self.assertEqual(datasets["formal_synthetic"]["selection_split"], "val")
        self.assertFalse(datasets["synthetic_holdout_v1"]["selection_allowed"])
        self.assertFalse(datasets["rotation_dev"]["selection_allowed"])


class PaperV1P1A2ArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        if not PREHOLDOUT_SELECTION.is_file():
            self.skipTest("P1-A2 artifacts are not frozen yet")

    def test_preholdout_hash_is_consistent_and_earlier_than_holdout(self) -> None:
        sidecar = load_json(PREHOLDOUT_SIDECAR)
        self.assertEqual(sha256_file(PREHOLDOUT_SELECTION), sidecar["sha256"])
        holdout = REPO_ROOT / "output_generation_paper_v1_synthetic_holdout_v1"
        manifest = load_json(holdout / "holdout_generation_manifest.json")
        self.assertEqual(manifest["preholdout_selection_sha256"], sidecar["sha256"])
        self.assertTrue(manifest["preholdout_strictly_predates_generation"])
        self.assertLess(
            manifest["preholdout_mtime_ns"], manifest["holdout_generation_started_ns"]
        )

    def test_s3_real_dev_and_holdout_are_forbidden_for_selection(self) -> None:
        selection = load_json(PREHOLDOUT_SELECTION)
        self.assertFalse(selection["s3_used"])
        self.assertFalse(selection["real_dev_used_for_selection"])
        self.assertFalse(selection["synthetic_development_test_used_for_selection"])
        self.assertFalse(selection["holdout_existed_at_freeze"])
        datasets = load_json(CONFIG_ROOT / "datasets.json")
        lowered = json.dumps(datasets["excluded_tokens"]).lower()
        self.assertIn("test_part_s3", lowered)

    def test_holdout_is_balanced_valid_and_collision_free(self) -> None:
        root = REPO_ROOT / "output_generation_paper_v1_synthetic_holdout_v1"
        quality = load_json(root / "quality_report.json")
        fingerprint = load_json(root / "fingerprint_audit.json")
        self.assertEqual(quality["accepted_sample_count"], 1080)
        self.assertEqual(quality["direct_occ_valid_single_solid_count"], 1080)
        self.assertEqual(set(quality["class_counts"]), set(TARGET_CLASSES))
        self.assertEqual(set(quality["class_counts"].values()), {60})
        for key, value in fingerprint.items():
            if key.endswith("collision_with_original_count") or key.startswith(
                "holdout_internal_"
            ) and key.endswith("collision_count"):
                self.assertEqual(value, 0, key)

    def test_final_variant_satisfies_val_protocol(self) -> None:
        selection = load_json(PREHOLDOUT_SELECTION)
        selected = selection["selected_variant"]
        self.assertTrue(selection["selection_constraints"][selected]["eligible"])

    def test_formal_report_keeps_every_target_class_and_one_holdout_pass(self) -> None:
        metrics_path = REPORT_ROOT / "formal_metrics.json"
        if not metrics_path.is_file():
            self.skipTest("formal evaluation has not completed")
        metrics = load_json(metrics_path)
        per_class = metrics["synthetic_holdout_v1"]["object"]["per_class"]
        self.assertTrue(set(TARGET_CLASSES).issubset(per_class))
        self.assertEqual(metrics["holdout_inference_pass_count"], 1)
        manifest = load_json(
            REPO_ROOT / "paper_v1_clean_candidate_v2_freeze_manifest.json"
        )
        self.assertEqual(manifest["holdout_inference_pass_count"], 1)

    def test_two_normalized_formal_runs_match_exactly(self) -> None:
        repeatability_path = REPORT_ROOT / "repeatability.json"
        if not repeatability_path.is_file():
            self.skipTest("formal repeatability has not completed")
        repeatability = load_json(repeatability_path)
        self.assertEqual(repeatability["status"], "pass")
        self.assertEqual(repeatability["mismatches"], [])
        self.assertTrue(repeatability["metrics_exact_match"])

    def test_v2_freeze_manifest_hashes(self) -> None:
        path = REPO_ROOT / "paper_v1_clean_candidate_v2_freeze_manifest.json"
        if not path.is_file():
            self.skipTest("v2 freeze manifest has not completed")
        manifest = load_json(path)
        for artifact in manifest["artifacts"]:
            artifact_path = Path(artifact["path"])
            if not artifact_path.is_absolute():
                artifact_path = REPO_ROOT / artifact_path
            digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            self.assertEqual(digest, artifact["sha256"], artifact["path"])


if __name__ == "__main__":
    unittest.main()
