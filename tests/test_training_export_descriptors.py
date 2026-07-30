import unittest

from utils.training_export import NODE_FEATURE_SCHEMA, build_numeric_feature_vector


class TrainingExportDescriptorsTest(unittest.TestCase):
    def test_shape_descriptors_are_exported_to_node_feature_vector(self):
        node = {
            "attributes": {
                "type_onehot": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "physical": {"area": 12.0, "center_of_mass": [1.0, 2.0, 3.0]},
                "specific": {},
                "shape_descriptors": {
                    "bbox_dx": 4.0,
                    "bbox_dy": 5.0,
                    "bbox_dz": 0.0,
                    "bbox_longest_extent": 5.0,
                    "bbox_middle_extent": 4.0,
                    "bbox_shortest_extent": 0.0,
                    "bbox_diagonal": 6.403124,
                    "bbox_aspect_ratio": 1.25,
                    "uv_u_span": 4.0,
                    "uv_v_span": 5.0,
                    "uv_aspect_ratio": 1.25,
                    "projected_long_extent": 5.0,
                    "projected_short_extent": 4.0,
                    "projected_aspect_ratio": 1.25,
                    "projected_area_fill_ratio": 0.96,
                    "vertex_count": 4,
                    "edge_count": 4,
                    "neighbor_count": 4,
                    "neighbor_area_mean": 10.0,
                    "neighbor_area_max": 15.0,
                    "neighbor_projected_aspect_mean": 2.0,
                    "neighbor_long_extent_mean": 8.0,
                },
            }
        }

        vector = build_numeric_feature_vector(node, ndigits=6)

        self.assertEqual(len(vector), len(NODE_FEATURE_SCHEMA))
        self.assertEqual(vector[NODE_FEATURE_SCHEMA.index("bbox_dx")], 4.0)
        self.assertEqual(vector[NODE_FEATURE_SCHEMA.index("projected_aspect_ratio")], 1.25)
        self.assertEqual(vector[NODE_FEATURE_SCHEMA.index("neighbor_long_extent_mean")], 8.0)


if __name__ == "__main__":
    unittest.main()
