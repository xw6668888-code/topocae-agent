import unittest
from pathlib import Path

from match_simulation_requirements import resolve_path
from utils.requirement_matching import match_requirement, match_requirements


def feature_object(feature_id, macro_feature, regions):
    return {
        "feature_object_id": feature_id,
        "macro_feature": macro_feature,
        "selectable_regions": regions,
    }


def region(role, faces, intended_uses, defaults=()):
    return {
        "role": role,
        "face_indices": faces,
        "source_semantic_subtypes": [],
        "intended_uses": list(intended_uses),
        "is_default_for": list(defaults),
        "selection_priority": 1,
    }


class RequirementMatchingTest(unittest.TestCase):
    def test_cli_path_resolution_falls_back_to_current_workspace(self):
        resolved = resolve_path("tests/test_requirement_matching.py", Path.cwd() / "configs")

        self.assertEqual(resolved, (Path.cwd() / "tests" / "test_requirement_matching.py").resolve())

    def test_fix_all_blind_holes_selects_default_wall_region_only(self):
        feature_objects = [
            feature_object(
                "hole-1",
                "blind_hole_round",
                [
                    region("hole_cylindrical_wall", [10], ["fixed_support"], ["hole_constraint"]),
                    region("hole_bottom_face", [11], ["pressure_load"]),
                ],
            ),
            feature_object(
                "hole-2",
                "blind_hole_round",
                [
                    region("hole_cylindrical_wall", [20], ["fixed_support"], ["hole_constraint"]),
                    region("hole_bottom_face", [21], ["pressure_load"]),
                ],
            ),
        ]

        result = match_requirement("固定零件的所有盲孔", feature_objects)

        self.assertEqual([match["face_indices"] for match in result["matches"]], [[10], [20]])
        self.assertEqual({match["role"] for match in result["matches"]}, {"hole_cylindrical_wall"})
        self.assertEqual(result["warnings"], [])

    def test_matches_include_geometry_parameters_for_downstream_filtering(self):
        candidate = feature_object(
            "hole-1",
            "blind_hole_round",
            [region("hole_cylindrical_wall", [10], ["fixed_support"], ["hole_constraint"])],
        )
        candidate["geometry_parameters"] = {
            "diameter": 5.0,
            "depth": 6.0,
            "depth_to_diameter_ratio": 1.2,
        }

        result = match_requirement("固定直径为5的盲孔", [candidate])

        self.assertEqual(result["matches"][0]["geometry_parameters"]["diameter"], 5.0)

    def test_stepped_hole_shoulder_and_small_wall_request_selects_both_roles(self):
        feature_objects = [
            feature_object(
                "stepped-1",
                "stepped_hole_round",
                [
                    region("large_hole_cylindrical_wall", [1], ["fixed_support"], ["hole_constraint"]),
                    region("step_shoulder_face", [2], ["fixed_support"]),
                    region("small_hole_cylindrical_wall", [3], ["fixed_support"], ["hole_constraint"]),
                ],
            )
        ]

        result = match_requirement("固定阶梯孔的台阶面和小孔壁", feature_objects)

        self.assertEqual(
            [(match["role"], match["face_indices"]) for match in result["matches"]],
            [("step_shoulder_face", [2]), ("small_hole_cylindrical_wall", [3])],
        )

    def test_rectangular_cavity_inner_wall_falls_back_to_rectangular_slot(self):
        feature_objects = [
            feature_object(
                "slot-1",
                "slot_rect",
                [region("slot_side_faces", [4, 5], ["pressure_load"])],
            )
        ]

        result = match_requirement("在矩形腔内壁施加压力", feature_objects)

        self.assertEqual(len(result["matches"]), 1)
        self.assertEqual(result["matches"][0]["macro_feature"], "slot_rect")
        self.assertEqual(result["matches"][0]["role"], "slot_side_faces")
        self.assertEqual(result["matches"][0]["face_indices"], [4, 5])
        self.assertEqual(result["warnings"], [])

    def test_rectangular_cavity_bottom_can_match_pocket_and_slot_without_cross_warnings(self):
        feature_objects = [
            feature_object(
                "pocket-1",
                "pocket_rect",
                [region("pocket_bottom_face", [1], ["fixed_support"])],
            ),
            feature_object(
                "slot-1",
                "slot_rect",
                [region("slot_bottom_face", [2], ["fixed_support"])],
            ),
        ]

        result = match_requirement("固定所有矩形腔的底部", feature_objects)

        self.assertEqual(
            [(match["macro_feature"], match["role"], match["face_indices"]) for match in result["matches"]],
            [
                ("pocket_rect", "pocket_bottom_face", [1]),
                ("slot_rect", "slot_bottom_face", [2]),
            ],
        )
        self.assertEqual(result["warnings"], [])

    def test_u_slot_inner_wall_selects_side_and_end_wall_regions(self):
        feature_objects = [
            feature_object(
                "u-slot-1",
                "slot_u",
                [
                    region("slot_side_faces", [6, 7], ["pressure_load"]),
                    region("slot_end_cylinders", [8], ["pressure_load"]),
                    region("slot_bottom_face", [9], ["pressure_load"]),
                ],
            )
        ]

        result = match_requirement("在U型槽内壁施加压力载荷", feature_objects)

        self.assertEqual(
            [(match["role"], match["face_indices"]) for match in result["matches"]],
            [("slot_side_faces", [6, 7]), ("slot_end_cylinders", [8])],
        )

    def test_cylinder_top_and_bottom_reports_unavailable_bottom_face(self):
        feature_objects = [
            feature_object(
                "boss-1",
                "cylinder_boss",
                [region("boss_top_face", [12], ["fixed_support", "surface_load"])],
            )
        ]

        result = match_requirement("固定圆柱顶面和底面", feature_objects)

        self.assertEqual(len(result["matches"]), 1)
        self.assertEqual(result["matches"][0]["role"], "boss_top_face")
        self.assertEqual(result["warnings"][0]["reason"], "region_not_available")
        self.assertEqual(result["warnings"][0]["requested_role"], "boss_bottom_face")

    def test_batch_matching_preserves_requirement_ids(self):
        output = match_requirements(
            [
                {"id": "r1", "text": "选取圆角"},
                {"id": "r2", "text": "在通孔内壁加载压力"},
            ],
            [
                feature_object(
                    "fillet-1",
                    "fillet",
                    [region("transition_faces", [30], ["mesh_refinement", "defeaturing_candidate"])],
                )
            ],
        )

        self.assertEqual(output["requirements"][0]["requirement_id"], "r1")
        self.assertEqual(output["requirements"][0]["action"], "selection")
        self.assertEqual(output["requirements"][0]["matches"][0]["face_indices"], [30])
        self.assertEqual(output["requirements"][1]["warnings"][0]["reason"], "feature_not_found")


if __name__ == "__main__":
    unittest.main()
