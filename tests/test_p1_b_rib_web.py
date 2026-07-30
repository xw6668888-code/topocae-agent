from __future__ import annotations

import random
import unittest
from collections import Counter

from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Extend.TopologyUtils import TopologyExplorer

from features.p1_b_rib_web_v3 import SCENARIO_SPECS, build_rib_web_sample, scenario_keys


class P1BRibWebEngineeringTests(unittest.TestCase):
    def test_scenario_matrix_has_twelve_engineering_cases(self):
        self.assertEqual(12, len(SCENARIO_SPECS))
        self.assertEqual(12, len(set(scenario_keys())))
        contexts = Counter(item[2] for item in SCENARIO_SPECS)
        self.assertEqual({"l_corner": 3, "rect_cavity": 3, "circular_cavity": 3, "two_supports": 3}, dict(contexts))
        for context in contexts:
            self.assertEqual({"plain", "hole", "transition"}, {item[3] for item in SCENARIO_SPECS if item[2] == context})

    def test_all_scenarios_are_valid_supported_single_solids(self):
        for index, scenario_key in enumerate(scenario_keys()):
            with self.subTest(scenario_key=scenario_key):
                result = build_rib_web_sample(scenario_key, 202607185000 + index)
                params = result.parameters
                self.assertTrue(BRepCheck_Analyzer(result.shape).IsValid())
                self.assertEqual(1, len(list(TopologyExplorer(result.shape).solids())))
                props = GProp_GProps()
                brepgprop.VolumeProperties(result.shape, props)
                self.assertGreater(props.Mass(), 0.0)
                self.assertFalse(params["isolated_thin_plate"])
                self.assertGreaterEqual(params["positive_support_contact_count"], 2)
                self.assertTrue(all(item["intersection_volume"] > 0.0 for item in params["support_contacts"]))
                self.assertGreaterEqual(params["span_thickness_ratio"], 5.0)
                self.assertGreaterEqual(params["height_thickness_ratio"], 2.0)
                self.assertGreaterEqual(params["feature_face_count"], 2)
                final_faces = list(TopologyExplorer(result.shape).faces())
                self.assertTrue(all(any(tag.face.IsSame(face) for face in final_faces) for tag in result.face_tags))

    def test_thin_wall_faces_share_one_instance_and_stock_context_is_not_absorbed(self):
        for index, scenario_key in enumerate(scenario_keys()):
            result = build_rib_web_sample(scenario_key, 202607185100 + index)
            params = result.parameters
            feature_tags = [tag for tag in result.face_tags if tag.macro_key == params["macro_key"]]
            self.assertTrue(feature_tags)
            self.assertEqual({params["feature_instance_id"]}, {tag.instance_id for tag in feature_tags})
            self.assertEqual(params["feature_face_count"], len(feature_tags))
            self.assertLess(len(result.face_tags), len(list(TopologyExplorer(result.shape).faces())))
            self.assertNotIn("boundary", params["context_kind"])

    def test_cavity_and_rib_remain_independent_instances(self):
        cavity_scenarios = [item[0] for item in SCENARIO_SPECS if item[2] in {"rect_cavity", "circular_cavity"}]
        for index, scenario_key in enumerate(cavity_scenarios):
            result = build_rib_web_sample(scenario_key, 202607185200 + index)
            params = result.parameters
            pocket_tags = [tag for tag in result.face_tags if tag.macro_key in {"pocket_rect", "pocket_circular"}]
            rib_tags = [tag for tag in result.face_tags if tag.macro_key == "rib"]
            self.assertTrue(pocket_tags)
            self.assertTrue(rib_tags)
            self.assertEqual({1}, {tag.instance_id for tag in pocket_tags})
            self.assertEqual({2}, {tag.instance_id for tag in rib_tags})
            self.assertTrue(params["cavity_and_rib_instances_independent"])
            self.assertTrue(params["connects_opposed_cavity_walls"])

    def test_hole_and_transition_variants_preserve_thin_wall_body(self):
        for index, (scenario_key, macro_key, _context, variant) in enumerate(SCENARIO_SPECS):
            result = build_rib_web_sample(scenario_key, 202607185300 + index)
            params = result.parameters
            feature_tags = [tag for tag in result.face_tags if tag.macro_key == macro_key]
            self.assertGreaterEqual(len(feature_tags), 2)
            if variant == "hole":
                hole_tags = [tag for tag in result.face_tags if tag.macro_key == "through_hole"]
                self.assertTrue(params["has_hole"])
                self.assertTrue(hole_tags)
                self.assertTrue(params["hole_parameters"]["passes_only_through_thin_wall"])
                self.assertNotEqual(params["feature_instance_id"], hole_tags[0].instance_id)
            elif variant == "transition":
                self.assertFalse(params["has_hole"])
                self.assertNotEqual("none", params["transition_kind"])
                self.assertGreaterEqual(params["feature_face_count"], 5)

    def test_seed_is_local_and_reproducible(self):
        random.seed(731)
        expected_next = random.random()
        random.seed(731)
        first = build_rib_web_sample("web_two_supports_transition", 202607185400)
        self.assertEqual(expected_next, random.random())
        second = build_rib_web_sample("web_two_supports_transition", 202607185400)
        for key in ("span", "thickness", "height", "span_axis", "support_contacts"):
            self.assertEqual(first.parameters[key], second.parameters[key])


if __name__ == "__main__":
    unittest.main()
