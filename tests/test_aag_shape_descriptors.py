import unittest

from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox

from core.AAGExtractor import extract_aag_graph


class AAGShapeDescriptorsTest(unittest.TestCase):
    def test_planar_faces_include_brep_shape_descriptors(self):
        shape = BRepPrimAPI_MakeBox(10.0, 20.0, 30.0).Shape()

        graph = extract_aag_graph(shape, include_label_fields=False)

        self.assertEqual(len(graph["nodes"]), 6)
        large_faces = [
            node for node in graph["nodes"]
            if abs(node["attributes"]["physical"]["area"] - 600.0) < 1e-6
        ]
        self.assertTrue(large_faces)

        descriptors = large_faces[0]["attributes"]["shape_descriptors"]
        self.assertGreater(descriptors["bbox_diagonal"], 0.0)
        self.assertAlmostEqual(descriptors["projected_aspect_ratio"], 1.5, places=5)
        self.assertAlmostEqual(descriptors["projected_area_fill_ratio"], 1.0, places=5)
        self.assertGreaterEqual(descriptors["vertex_count"], 4)
        self.assertGreaterEqual(descriptors["edge_count"], 4)

    def test_neighborhood_descriptors_are_backfilled_after_edges(self):
        shape = BRepPrimAPI_MakeBox(10.0, 20.0, 30.0).Shape()

        graph = extract_aag_graph(shape, include_label_fields=False)

        for node in graph["nodes"]:
            descriptors = node["attributes"]["shape_descriptors"]
            self.assertGreater(descriptors["neighbor_count"], 0)
            self.assertGreater(descriptors["neighbor_area_mean"], 0.0)


if __name__ == "__main__":
    unittest.main()
