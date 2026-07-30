import unittest

from utils.colored_step_rib_truth import (
    apply_colored_rib_truth_to_template,
    extract_step_face_color_overrides,
    is_background_color,
    match_colored_step_faces_to_graph,
)


class ColoredStepRibTruthTest(unittest.TestCase):
    def test_medium_steel_is_background_and_other_colors_are_rib(self):
        self.assertTrue(is_background_color("Medium Steel", [0.596063172350652, 0.666666666666667, 0.686259250782025]))
        self.assertTrue(is_background_color("", [0.5961, 0.6667, 0.6863]))
        self.assertFalse(is_background_color("Red", [1.0, 0.0, 0.0]))
        self.assertFalse(is_background_color("rib_1", [0.0, 1.0, 0.0]))

    def test_applies_colored_faces_to_template_reviews(self):
        template = {
            "schema": "rib_face_truth_template_v1",
            "samples": [
                {
                    "sample_id": "partA",
                    "candidate_faces": [
                        {"face_idx": 1, "review": {"is_rib_face": None, "rib_instance_id": "", "notes": ""}},
                        {"face_idx": 2, "review": {"is_rib_face": None, "rib_instance_id": "", "notes": ""}},
                        {"face_idx": 3, "review": {"is_rib_face": None, "rib_instance_id": "", "notes": ""}},
                        {"face_idx": 4, "review": {"is_rib_face": None, "rib_instance_id": "", "notes": ""}},
                    ],
                }
            ],
        }
        colored_truth_by_sample = {
            "partA": {
                "rib_faces": {
                    1: {"color_key": "1.0000,0.0000,0.0000", "color_name": "Red"},
                    3: {"color_key": "0.0000,1.0000,0.0000", "color_name": "Green"},
                },
                "background_faces": {
                    2: {"color_key": "0.5961,0.6667,0.6863", "color_name": "Medium Steel"}
                },
            }
        }

        reviewed, diagnostics = apply_colored_rib_truth_to_template(template, colored_truth_by_sample)

        candidates = {
            candidate["face_idx"]: candidate["review"]
            for candidate in reviewed["samples"][0]["candidate_faces"]
        }
        self.assertTrue(candidates[1]["is_rib_face"])
        self.assertTrue(candidates[3]["is_rib_face"])
        self.assertFalse(candidates[2]["is_rib_face"])
        self.assertIsNone(candidates[4]["is_rib_face"])
        self.assertNotEqual(candidates[1]["rib_instance_id"], candidates[3]["rib_instance_id"])
        self.assertEqual(diagnostics["samples"][0]["filled_true"], 2)
        self.assertEqual(diagnostics["samples"][0]["filled_false"], 1)
        self.assertEqual(diagnostics["samples"][0]["left_null"], 1)

    def test_appends_colored_rib_faces_that_were_missing_from_template_candidates(self):
        template = {
            "schema": "rib_face_truth_template_v1",
            "samples": [
                {
                    "sample_id": "partA",
                    "candidate_faces": [
                        {"face_idx": 1, "review": {"is_rib_face": None, "rib_instance_id": "", "notes": ""}},
                    ],
                }
            ],
        }
        colored_truth_by_sample = {
            "partA": {
                "rib_faces": {
                    1: {"color_key": "1.0000,0.0000,0.0000", "color_name": "Red"},
                    5: {"color_key": "1.0000,0.0000,0.0000", "color_name": "Red"},
                },
                "background_faces": {},
            }
        }

        reviewed, diagnostics = apply_colored_rib_truth_to_template(template, colored_truth_by_sample)

        candidate_by_idx = {candidate["face_idx"]: candidate for candidate in reviewed["samples"][0]["candidate_faces"]}
        self.assertIn(5, candidate_by_idx)
        self.assertTrue(candidate_by_idx[5]["review"]["is_rib_face"])
        self.assertIn("colored_step_rib_face_outside_template", candidate_by_idx[5]["candidate_reasons"])
        self.assertEqual(diagnostics["samples"][0]["appended_colored_rib_candidates"], 1)

    def test_matches_colored_step_faces_to_graph_by_geometry_fingerprint(self):
        graph_data = {
            "nodes": [
                graph_node(0, "plane", 1000.0, [0.0, 0.0, 0.0], {"normal": [0.0, 0.0, 1.0]}),
                graph_node(1, "plane", 20.0, [10.0, 0.0, 5.0], {"normal": [1.0, 0.0, 0.0]}),
                graph_node(2, "plane", 22.0, [10.0, 2.0, 5.0], {"normal": [-1.0, 0.0, 0.0]}),
            ]
        }
        colored_faces = [
            {
                "source_face_idx": 8,
                "primitive_surface": "plane",
                "area": 20.0004,
                "center": [10.0002, 0.0001, 5.0001],
                "specific": {"normal": [1.0, 0.0, 0.0]},
                "color_name": "Red",
                "rgb": [1.0, 0.0, 0.0],
            }
        ]

        result = match_colored_step_faces_to_graph(colored_faces, graph_data)

        self.assertEqual(result["rib_faces"][1]["source_face_idx"], 8)
        self.assertEqual(result["unmatched_colored_faces"], [])

    def test_extracts_overriding_styled_item_colors_by_advanced_face_order(self):
        step_text = """
        #10=OVER_RIDING_STYLED_ITEM('',(#20),#100,#30);
        #11=OVER_RIDING_STYLED_ITEM('',(#21),#102,#30);
        #20=PRESENTATION_STYLE_ASSIGNMENT((#40));
        #21=PRESENTATION_STYLE_ASSIGNMENT((#41));
        #40=SURFACE_STYLE_USAGE(.BOTH.,#50);
        #41=SURFACE_STYLE_USAGE(.BOTH.,#51);
        #50=SURFACE_SIDE_STYLE('',(#60));
        #51=SURFACE_SIDE_STYLE('',(#61));
        #60=SURFACE_STYLE_FILL_AREA(#70);
        #61=SURFACE_STYLE_FILL_AREA(#71);
        #70=FILL_AREA_STYLE('',(#80));
        #71=FILL_AREA_STYLE('',(#81));
        #80=FILL_AREA_STYLE_COLOUR('',#90);
        #81=FILL_AREA_STYLE_COLOUR('',#91);
        #90=COLOUR_RGB('Deep Red',1.,0.188235294117647,0.188235294117647);
        #91=COLOUR_RGB('Medium Steel',0.596063172350652,0.666666666666667,0.686259250782025);
        #100=ADVANCED_FACE('',(#1),#200,.T.);
        #101=ADVANCED_FACE('',(#2),#201,.T.);
        #102=ADVANCED_FACE('',(#3),#202,.T.);
        """

        extracted = extract_step_face_color_overrides(step_text)

        self.assertEqual(extracted["face_count"], 3)
        self.assertEqual(extracted["face_colors"][0]["color_name"], "Deep Red")
        self.assertEqual(extracted["face_colors"][2]["color_name"], "Medium Steel")
        self.assertNotIn(1, extracted["face_colors"])


    def test_extracts_draughting_pre_defined_colour_as_face_override(self):
        step_text = """
        #10=OVER_RIDING_STYLED_ITEM('',(#20),#100,#30);
        #20=PRESENTATION_STYLE_ASSIGNMENT((#40));
        #40=SURFACE_STYLE_USAGE(.BOTH.,#50);
        #50=SURFACE_SIDE_STYLE('',(#60));
        #60=SURFACE_STYLE_FILL_AREA(#70);
        #70=FILL_AREA_STYLE('',(#80));
        #80=FILL_AREA_STYLE_COLOUR('',#90);
        #90=DRAUGHTING_PRE_DEFINED_COLOUR('red');
        #100=ADVANCED_FACE('',(#1),#200,.T.);
        """

        extracted = extract_step_face_color_overrides(step_text)

        self.assertEqual(extracted["face_count"], 1)
        self.assertEqual(extracted["face_colors"][0]["color_name"], "red")
        self.assertEqual(extracted["face_colors"][0]["rgb"], [1.0, 0.0, 0.0])
def graph_node(idx, primitive, area, center, specific):
    return {
        "idx": idx,
        "primitive_surface": primitive,
        "attributes": {
            "physical": {"area": area, "center_of_mass": center},
            "specific": specific,
        },
    }


if __name__ == "__main__":
    unittest.main()

