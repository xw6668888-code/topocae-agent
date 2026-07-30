import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from prediction_to_feature_objects import build_feature_objects, load_json
from utils.selectable_regions import build_selectable_regions


def by_role(regions):
    return {region["role"]: region for region in regions}


class SelectableRegionsTest(unittest.TestCase):
    def test_load_json_accepts_utf8_bom_prediction_files(self):
        with TemporaryDirectory() as temp_dir:
            prediction_path = Path(temp_dir) / "prediction.json"
            prediction_path.write_text('{"faces": []}', encoding="utf-8-sig")

            self.assertEqual(load_json(prediction_path), {"faces": []})

    def test_blind_hole_defaults_to_cylindrical_wall_not_bottom(self):
        descriptors = [
            {"face_idx": 1, "semantic_subtype": "blind_hole_cylinder"},
            {"face_idx": 2, "semantic_subtype": "blind_hole_bottom"},
        ]

        regions = by_role(build_selectable_regions("blind_hole_round", descriptors))

        self.assertEqual(regions["hole_cylindrical_wall"]["face_indices"], [1])
        self.assertIn("hole_constraint", regions["hole_cylindrical_wall"]["is_default_for"])
        self.assertEqual(regions["hole_bottom_face"]["face_indices"], [2])
        self.assertNotIn("hole_constraint", regions["hole_bottom_face"]["is_default_for"])

    def test_obround_hole_has_wall_aggregate_region(self):
        descriptors = [
            {"face_idx": 3, "semantic_subtype": "obround_hole_side_plane"},
            {"face_idx": 4, "semantic_subtype": "obround_hole_side_plane"},
            {"face_idx": 5, "semantic_subtype": "obround_hole_end_cylinder"},
            {"face_idx": 6, "semantic_subtype": "obround_hole_end_cylinder"},
        ]

        regions = by_role(build_selectable_regions("obround_hole", descriptors))

        self.assertEqual(regions["obround_hole_wall_faces"]["face_indices"], [3, 4, 5, 6])
        self.assertEqual(regions["obround_side_faces"]["face_indices"], [3, 4])
        self.assertEqual(regions["obround_end_cylinders"]["face_indices"], [5, 6])

    def test_pressure_can_target_hole_and_u_slot_wall_regions(self):
        through_hole_regions = by_role(build_selectable_regions(
            "through_hole_round",
            [{"face_idx": 1, "semantic_subtype": "through_hole_cylinder"}],
        ))
        slot_u_regions = by_role(build_selectable_regions(
            "slot_u",
            [
                {"face_idx": 2, "semantic_subtype": "slot_u_side_plane"},
                {"face_idx": 3, "semantic_subtype": "slot_u_end_cylinder"},
            ],
        ))

        self.assertIn("pressure_load", through_hole_regions["hole_cylindrical_wall"]["intended_uses"])
        self.assertIn("pressure_load", slot_u_regions["slot_side_faces"]["intended_uses"])
        self.assertIn("pressure_load", slot_u_regions["slot_end_cylinders"]["intended_uses"])

    def test_rib_shell_candidate_uses_side_faces_only(self):
        descriptors = [
            {"face_idx": 7, "semantic_subtype": "rib_side_plane"},
            {"face_idx": 8, "semantic_subtype": "rib_side_plane"},
            {"face_idx": 9, "semantic_subtype": "rib_top_plane"},
        ]

        regions = by_role(build_selectable_regions("rib", descriptors))

        self.assertEqual(regions["thin_wall_side_faces"]["face_indices"], [7, 8])
        self.assertIn("shell_candidate", regions["thin_wall_side_faces"]["is_default_for"])
        self.assertEqual(regions["rib_top_face"]["face_indices"], [9])
        self.assertNotIn("shell_candidate", regions["rib_top_face"]["is_default_for"])

    def test_feature_object_export_contains_selectable_regions(self):
        prediction = {
            "model_id": "mock-part",
            "faces": [
                {
                    "idx": 0,
                    "macro_feature_id": 1,
                    "macro_feature": "blind_hole_round",
                    "macro_confidence": 0.98,
                    "semantic_subtype_id": 10,
                    "semantic_subtype": "blind_hole_cylinder",
                    "semantic_subtype_confidence": 0.97,
                },
                {
                    "idx": 1,
                    "macro_feature_id": 1,
                    "macro_feature": "blind_hole_round",
                    "macro_confidence": 0.96,
                    "semantic_subtype_id": 11,
                    "semantic_subtype": "blind_hole_bottom",
                    "semantic_subtype_confidence": 0.95,
                },
            ],
        }
        graph = {
            "nodes": [
                {
                    "idx": 0,
                    "primitive_surface": "cylinder",
                    "attributes": {
                        "physical": {
                            "area": 3.0,
                            "center_of_mass": [0.0, 0.0, 1.0],
                        },
                        "geometry": {
                            "radius": 2.5,
                            "axis": [0.0, 0.0, 1.0],
                        },
                    },
                },
                {
                    "idx": 1,
                    "primitive_surface": "plane",
                    "attributes": {
                        "physical": {
                            "area": 4.0,
                            "center_of_mass": [0.0, 0.0, 0.0],
                        },
                        "geometry": {
                            "normal": [0.0, 0.0, 1.0],
                        },
                    },
                },
            ],
            "edges": [[0, 1], [1, 0]],
        }

        output = build_feature_objects(prediction, graph)
        feature_objects = output["feature_objects"]

        self.assertEqual(len(feature_objects), 1)
        regions = by_role(feature_objects[0]["selectable_regions"])
        self.assertEqual(regions["hole_cylindrical_wall"]["face_indices"], [0])
        self.assertEqual(regions["hole_bottom_face"]["face_indices"], [1])


if __name__ == "__main__":
    unittest.main()
