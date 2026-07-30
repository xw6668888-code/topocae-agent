import unittest

from features.blind_hole import BlindHoleGenerator
from features.circular_pocket import CircularPocketGenerator
from features.rect_pocket import RectPocketGenerator
from features.slot_rect import SlotRectGenerator
from features.slot_u import USlotGenerator
from prediction_to_feature_objects import build_feature_objects
from utils.feature_geometry import attach_feature_parameters, build_feature_parameter_lookup, normalize_geometry_parameters
from utils.feature_geometry_validation import validate_feature_geometry_separation, validate_hole_pocket_separation


class FeatureGeometryParametersTest(unittest.TestCase):
    def test_blind_hole_geometry_parameters_include_depth_diameter_ratio(self):
        params = normalize_geometry_parameters(
            "blind_hole_round",
            {"inst": 1, "cx": 0.0, "cy": 0.0, "cz": 20.0, "axis": "x", "R": 2.5, "Depth": 6.0},
        )

        self.assertEqual(params["radius"], 2.5)
        self.assertEqual(params["diameter"], 5.0)
        self.assertEqual(params["depth"], 6.0)
        self.assertEqual(params["axis"], "x")
        self.assertEqual(params["cz"], 20.0)
        self.assertAlmostEqual(params["depth_to_diameter_ratio"], 1.2)
        self.assertEqual(params["cylindrical_recess_family"], "deep")

    def test_circular_pocket_geometry_parameters_are_shallow_recesses(self):
        params = normalize_geometry_parameters(
            "pocket_circular",
            {"inst": 2, "cx": 0.0, "cy": 0.0, "R": 5.0, "Depth": 3.0},
        )

        self.assertEqual(params["radius"], 5.0)
        self.assertEqual(params["diameter"], 10.0)
        self.assertEqual(params["depth"], 3.0)
        self.assertAlmostEqual(params["depth_to_diameter_ratio"], 0.3)
        self.assertEqual(params["cylindrical_recess_family"], "shallow")

    def test_rectangular_recess_geometry_parameters_include_aspect_ratio(self):
        slot_params = normalize_geometry_parameters(
            "slot_rect",
            {"inst": 1, "w": 4.0, "l": 24.0, "depth": 3.0},
        )
        pocket_params = normalize_geometry_parameters(
            "pocket_rect",
            {"inst": 2, "w": 10.0, "l": 15.0, "depth": 3.0},
        )

        self.assertEqual(slot_params["width"], 4.0)
        self.assertEqual(slot_params["length"], 24.0)
        self.assertAlmostEqual(slot_params["planform_aspect_ratio"], 6.0)
        self.assertEqual(slot_params["rectangular_recess_family"], "elongated_slot")
        self.assertAlmostEqual(pocket_params["planform_aspect_ratio"], 1.5)
        self.assertEqual(pocket_params["rectangular_recess_family"], "compact_pocket")

    def test_slot_geometry_parameters_preserve_swept_profile_metadata(self):
        params = normalize_geometry_parameters(
            "slot_rect",
            {
                "inst": 1,
                "w": 4.0,
                "l": 24.0,
                "depth": 3.0,
                "construction_method": "swept_closed_profile",
                "profile_shape": "rectangle",
                "sweep_path_length": 24.0,
                "profile_width": 4.0,
                "profile_depth": 3.0,
                "path_type": "straight",
                "path_curve_kind": "line",
                "path_on_face": True,
                "path_start_on_boundary": True,
                "path_end_on_boundary": True,
                "anchor_face_kind": "existing_planar_face",
                "path_boundary_kind": "rectangular_workspace_boundary",
            },
        )

        self.assertEqual(params["construction_method"], "swept_closed_profile")
        self.assertEqual(params["profile_shape"], "rectangle")
        self.assertEqual(params["sweep_path_length"], 24.0)
        self.assertEqual(params["profile_width"], 4.0)
        self.assertEqual(params["profile_depth"], 3.0)
        self.assertEqual(params["path_type"], "straight")
        self.assertEqual(params["path_curve_kind"], "line")
        self.assertTrue(params["path_on_face"])
        self.assertTrue(params["path_start_on_boundary"])
        self.assertTrue(params["path_end_on_boundary"])
        self.assertEqual(params["anchor_face_kind"], "existing_planar_face")
        self.assertEqual(params["path_boundary_kind"], "rectangular_workspace_boundary")

    def test_vector_axis_parameters_are_preserved_as_axis_vector(self):
        params = normalize_geometry_parameters(
            "fillet",
            {"inst": 1, "axis": [0.0, 1.0, 0.0], "radius": 2.0},
        )

        self.assertEqual(params["axis_vector"], [0.0, 1.0, 0.0])
        self.assertNotIn("axis", params)

    def test_attach_feature_parameters_adds_geometry_to_feature_sets(self):
        graph_data = {
            "feature_sets": [
                {
                    "instance_id": 1,
                    "macro_feature": "blind_hole_round",
                    "macro_feature_id": 2,
                    "node_indices": [10, 11],
                }
            ]
        }
        params_by_instance = {
            1: {
                "source_generator": "BlindHoleGenerator",
                "raw_parameters": {"inst": 1, "R": 2.5, "Depth": 6.0},
                "geometry_parameters": {
                    "radius": 2.5,
                    "diameter": 5.0,
                    "depth": 6.0,
                    "depth_to_diameter_ratio": 1.2,
                },
            }
        }

        attach_feature_parameters(graph_data, params_by_instance)

        feature_set = graph_data["feature_sets"][0]
        self.assertEqual(feature_set["source_generator"], "BlindHoleGenerator")
        self.assertEqual(feature_set["geometry_parameters"]["depth_to_diameter_ratio"], 1.2)

    def test_hole_pocket_separation_validator_accepts_clear_cases(self):
        graph_data = {
            "feature_sets": [
                {
                    "instance_id": 1,
                    "macro_feature": "blind_hole_round",
                    "geometry_parameters": {"depth_to_diameter_ratio": 1.1},
                },
                {
                    "instance_id": 2,
                    "macro_feature": "pocket_circular",
                    "geometry_parameters": {"depth_to_diameter_ratio": 0.3},
                },
            ]
        }

        report = validate_hole_pocket_separation(graph_data)

        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["issues"], [])

    def test_hole_pocket_separation_validator_flags_ambiguous_cases(self):
        graph_data = {
            "feature_sets": [
                {
                    "instance_id": 1,
                    "macro_feature": "blind_hole_round",
                    "geometry_parameters": {"depth_to_diameter_ratio": 0.4},
                },
                {
                    "instance_id": 2,
                    "macro_feature": "pocket_circular",
                    "geometry_parameters": {"depth_to_diameter_ratio": 0.8},
                },
            ]
        }

        report = validate_hole_pocket_separation(graph_data)

        self.assertEqual(report["status"], "issues_found")
        self.assertEqual(
            [issue["code"] for issue in report["issues"]],
            ["blind_hole_too_shallow", "circular_pocket_too_deep"],
        )

    def test_feature_geometry_validator_flags_rectangular_recess_ambiguity(self):
        graph_data = {
            "feature_sets": [
                {
                    "instance_id": 1,
                    "macro_feature": "slot_rect",
                    "geometry_parameters": {"planform_aspect_ratio": 2.8},
                },
                {
                    "instance_id": 2,
                    "macro_feature": "pocket_rect",
                    "geometry_parameters": {"planform_aspect_ratio": 2.2},
                },
            ]
        }

        report = validate_feature_geometry_separation(graph_data)

        self.assertEqual(report["status"], "issues_found")
        self.assertEqual(
            [issue["code"] for issue in report["issues"]],
            ["slot_rect_not_elongated_enough", "pocket_rect_too_elongated"],
        )

    def test_generator_parameter_ranges_separate_blind_holes_from_circular_pockets(self):
        self.assertGreaterEqual(BlindHoleGenerator.depth_to_diameter_ratio_range[0], 0.8)
        self.assertLessEqual(CircularPocketGenerator.depth_to_diameter_ratio_range[1], 0.5)

    def test_generator_parameter_ranges_separate_rectangular_slots_from_pockets(self):
        self.assertGreaterEqual(SlotRectGenerator.aspect_ratio_range[0], 4.0)
        self.assertLessEqual(RectPocketGenerator.aspect_ratio_range[1], 1.8)

    def test_u_slot_swept_path_is_longer_than_profile_width(self):
        self.assertGreaterEqual(USlotGenerator.sweep_to_profile_ratio_range[0], 4.0)

    def test_feature_objects_preserve_source_geometry_parameters_when_available(self):
        prediction = {
            "sample_id": "mock-part",
            "faces": [
                {
                    "idx": 0,
                    "macro_feature_id": 2,
                    "macro_feature": "blind_hole_round",
                    "macro_confidence": 1.0,
                    "semantic_subtype_id": 6,
                    "semantic_subtype": "blind_hole_cylinder",
                    "semantic_confidence": 1.0,
                },
                {
                    "idx": 1,
                    "macro_feature_id": 2,
                    "macro_feature": "blind_hole_round",
                    "macro_confidence": 1.0,
                    "semantic_subtype_id": 7,
                    "semantic_subtype": "blind_hole_bottom",
                    "semantic_confidence": 1.0,
                },
            ],
        }
        graph = {
            "nodes": [
                {
                    "idx": 0,
                    "primitive_surface": "cylinder",
                    "attributes": {
                        "physical": {"area": 1.0, "center_of_mass": [0.0, 0.0, 1.0]},
                        "specific": {"radius": 2.5, "axis": [0.0, 0.0, 1.0]},
                    },
                },
                {
                    "idx": 1,
                    "primitive_surface": "plane",
                    "attributes": {
                        "physical": {"area": 1.0, "center_of_mass": [0.0, 0.0, 0.0]},
                        "specific": {"normal": [0.0, 0.0, 1.0]},
                    },
                },
            ],
            "edges": [[0, 1], [1, 0]],
            "feature_sets": [
                {
                    "instance_id": 1,
                    "macro_feature": "blind_hole_round",
                    "node_indices": [0, 1],
                    "geometry_parameters": {
                        "radius": 2.5,
                        "diameter": 5.0,
                        "depth": 6.0,
                        "depth_to_diameter_ratio": 1.2,
                    },
                }
            ],
        }

        output = build_feature_objects(prediction, graph)
        feature_object = output["feature_objects"][0]

        self.assertEqual(feature_object["geometry_parameters"]["depth_to_diameter_ratio"], 1.2)
        self.assertEqual(
            feature_object["fingerprint"]["geometry_parameters"]["depth_to_diameter_ratio"],
            1.2,
        )

    def test_boss_feature_parameter_lookup_resolves_concrete_boss_type(self):
        class FakeOntologyMaps:
            def resolve_macro_feature(self, raw_key):
                if raw_key != "cylinder_boss":
                    raise KeyError(raw_key)

                class Resolution:
                    key = "cylinder_boss"
                    raw_key = "cylinder_boss"

                return Resolution()

        class FakeBossGenerator:
            feature_name = "boss_feature"

        lookup = build_feature_parameter_lookup(
            [
                (
                    FakeBossGenerator(),
                    [{"inst": 1, "type": "cylinder", "R1": 4.0, "R2": 4.0, "h": 8.0}],
                )
            ],
            FakeOntologyMaps(),
        )

        self.assertEqual(lookup[1]["macro_feature"], "cylinder_boss")
        self.assertEqual(lookup[1]["raw_parameters"]["type"], "cylinder")


if __name__ == "__main__":
    unittest.main()
