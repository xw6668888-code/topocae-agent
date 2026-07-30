from __future__ import annotations

import unittest

from utils.cae_text_evidence import CAETextEvidenceExtractor, EvidenceInputError


class CAETextEvidenceExtractorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.extractor = CAETextEvidenceExtractor()

    @staticmethod
    def _items(result, category):
        return [item for item in result["evidence"] if item["category"] == category]

    def test_rejects_fields_other_than_text_and_language(self):
        with self.assertRaises(EvidenceInputError):
            self.extractor.extract({"text": "Apply pressure.", "language": "en-US", "extra": 1})

    def test_rejects_unsupported_language(self):
        with self.assertRaises(EvidenceInputError):
            self.extractor.extract({"text": "Apply pressure.", "language": "fr-FR"})

    def test_english_pressure_has_spans_and_normalized_units(self):
        text = "Apply 2.5 GPa pressure normal to all hole inner walls."
        result = self.extractor.extract({"text": text, "language": "en-US"})
        pressure = next(item for item in self._items(result, "quantity") if item["normalized"]["dimension"] == "pressure")
        self.assertEqual(self._items(result, "operation_cue")[0]["normalized"], "pressure")
        self.assertEqual(pressure["normalized"]["value"], 2500.0)
        self.assertEqual(pressure["normalized"]["unit"], "MPa")
        self.assertEqual(text[pressure["span"]["start"]:pressure["span"]["end"]], pressure["text"])
        self.assertEqual(self._items(result, "selector")[0]["normalized"], {"selector": "all"})
        self.assertEqual(self._items(result, "feature_mention")[0]["normalized"], "through_hole_round")
        self.assertEqual(self._items(result, "face_role_mention")[0]["normalized"], "hole_wall")
        self.assertEqual(self._items(result, "direction")[0]["normalized"], {"kind": "normal"})
        self.assertTrue(all(item["rule_id"].startswith("P4R3A_") for item in result["evidence"]))

    def test_chinese_displacement_and_frame_are_normalized(self):
        text = "\u5728\u5168\u5c40\u5750\u6807\u7cfb\u6cbf\u8d1fY\u65b9\u5411\u5bf9\u6070\u597d\u4e00\u4e2a\u5706\u5b54\u7aef\u9762\u65bd\u52a0\u4f4d\u79fb U2=-0.2\u5398\u7c73\u3002"
        result = self.extractor.extract({"text": text, "language": "zh-CN"})
        self.assertEqual(self._items(result, "operation_cue")[0]["normalized"], "displacement")
        self.assertEqual(self._items(result, "selector")[0]["normalized"], {"selector": "exactly_one"})
        self.assertEqual(self._items(result, "coordinate_frame")[0]["normalized"], "global")
        self.assertEqual(self._items(result, "direction")[0]["normalized"], {"kind": "vector", "vector": [0.0, -1.0, 0.0]})
        self.assertEqual(self._items(result, "displacement_component")[0]["normalized"], {"dof": "U2", "value": -2.0, "unit": "mm"})

    def test_material_and_mesh_use_shared_normalization(self):
        text = "Assign steel with E=210 GPa and Poisson ratio 0.30; set global mesh size to 0.4 cm."
        result = self.extractor.extract({"text": text, "language": "en-US"})
        self.assertEqual(len(result["clauses"]), 2)
        youngs = self._items(result, "youngs_modulus")[0]
        mesh = self._items(result, "mesh_size")[0]
        self.assertEqual(youngs["normalized"], {"value": 210000.0, "unit": "MPa"})
        self.assertEqual(self._items(result, "poisson_ratio")[0]["normalized"], 0.3)
        self.assertEqual(self._items(result, "material_name")[0]["normalized"], "steel")
        self.assertEqual(mesh["normalized"], {"value": 4.0, "unit": "mm"})
        self.assertNotEqual(youngs["clause_index"], mesh["clause_index"])

    def test_count_comparison_and_dofs_are_explicit(self):
        result = self.extractor.extract({"text": "Fix the three smallest slot bottom faces in U1, U2 and U3.", "language": "en-US"})
        selectors = [item["normalized"] for item in self._items(result, "selector")]
        self.assertIn({"selector": "count", "count": 3}, selectors)
        self.assertIn({"selector": "smallest", "compare_by": "area"}, selectors)
        self.assertEqual([item["normalized"] for item in self._items(result, "degree_of_freedom")], ["U1", "U2", "U3"])

    def test_nearest_reference_presence_and_absence(self):
        present = self.extractor.extract({"text": "Apply 5 MPa pressure normal to the hole nearest to the global X axis.", "language": "en-US"})
        absent = self.extractor.extract({"text": "Apply 5 MPa pressure normal to the nearest hole.", "language": "en-US"})
        self.assertEqual(self._items(present, "selector")[0]["normalized"], {"selector": "nearest"})
        self.assertEqual(self._items(present, "spatial_reference")[0]["normalized"], {"kind": "axis", "axis": "X"})
        self.assertIn("nearest_to", [item["normalized"]["parameter"] for item in self._items(absent, "missing_required_parameter")])

    def test_set_difference_and_complex_exclusion(self):
        result = self.extractor.extract({"text": "Mesh all holes except the two nearest holes but not the countersinks at 1 mm.", "language": "en-US"})
        kinds = [item["normalized"]["kind"] for item in self._items(result, "exclusion")]
        self.assertIn("set_difference", kinds)
        self.assertIn("complex_exclusion", kinds)

    def test_action_anchors_prevent_parameter_leakage(self):
        text = "Apply 8 MPa pressure normal to all hole walls and then set local mesh size on every fillet to 0.8 mm."
        result = self.extractor.extract({"text": text, "language": "en-US"})
        pressure = next(item for item in self._items(result, "quantity") if item["normalized"]["dimension"] == "pressure")
        mesh = self._items(result, "mesh_size")[0]
        self.assertEqual(len(result["clauses"]), 2)
        self.assertEqual(pressure["clause_index"], 0)
        self.assertEqual(mesh["clause_index"], 1)

    def test_missing_parameters_are_evidence_not_status(self):
        result = self.extractor.extract({"text": "Apply pressure to the hole wall.", "language": "en-US"})
        self.assertNotIn("status", result)
        missing = {item["normalized"]["parameter"] for item in self._items(result, "missing_required_parameter")}
        self.assertEqual(missing, {"magnitude", "direction", "selector"})


if __name__ == "__main__":
    unittest.main()
