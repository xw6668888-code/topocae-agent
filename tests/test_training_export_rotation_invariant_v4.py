import unittest

from utils.training_export import (
    NODE_FEATURE_MODE_ROTATION_INVARIANT,
    build_gnn_record,
    build_node_feature_vector,
    node_feature_schema_for_mode,
)


def make_node(center, normal, axis, bbox, uv):
    return {
        "idx": 0,
        "primitive_surface_id": 2,
        "macro_feature_id": 1,
        "semantic_label": 5,
        "derived_macro_feature_ids": [],
        "idealization_tag_ids": [],
        "instance_id": 1,
        "attributes": {
            "type_onehot": [0, 1] + [0] * 9,
            "physical": {"area": 31.415926, "center_of_mass": center},
            "specific": {
                "normal": normal,
                "axis": axis,
                "radius": 2.0,
                "apex": [9.0, 8.0, 7.0],
                "semi_angle": 0.0,
            },
            "shape_descriptors": {
                "bbox_dx": bbox[0],
                "bbox_dy": bbox[1],
                "bbox_dz": bbox[2],
                "bbox_longest_extent": 10.0,
                "bbox_middle_extent": 4.0,
                "bbox_shortest_extent": 0.0,
                "bbox_diagonal": 10.77033,
                "bbox_aspect_ratio": 2.5,
                "uv_u_span": uv[0],
                "uv_v_span": uv[1],
                "uv_aspect_ratio": 2.5,
                "projected_long_extent": 10.0,
                "projected_short_extent": 4.0,
                "projected_aspect_ratio": 2.5,
                "projected_area_fill_ratio": 0.92,
                "vertex_count": 4,
                "edge_count": 4,
                "neighbor_count": 3,
                "neighbor_area_mean": 20.0,
                "neighbor_area_max": 40.0,
                "neighbor_projected_aspect_mean": 1.5,
                "neighbor_long_extent_mean": 12.0,
            },
        },
    }


class RotationInvariantV4ExportTest(unittest.TestCase):
    def test_rotation_invariant_schema_excludes_absolute_direction_fields(self):
        schema = node_feature_schema_for_mode(NODE_FEATURE_MODE_ROTATION_INVARIANT)

        forbidden_terms = [
            "center_of_mass",
            "normal_",
            "axis_",
            "apex_",
            "bbox_dx",
            "bbox_dy",
            "bbox_dz",
            "uv_u_span",
            "uv_v_span",
        ]
        for feature_name in schema:
            for forbidden in forbidden_terms:
                self.assertNotIn(forbidden, feature_name)

    def test_rotation_invariant_vector_is_unchanged_by_axis_rotation(self):
        node_a = make_node(
            center=[1.0, 2.0, 3.0],
            normal=[0.0, 0.0, 1.0],
            axis=[0.0, 0.0, 1.0],
            bbox=[10.0, 4.0, 0.0],
            uv=[10.0, 4.0],
        )
        node_b = make_node(
            center=[-2.0, 1.0, 3.0],
            normal=[0.0, -1.0, 0.0],
            axis=[0.0, -1.0, 0.0],
            bbox=[4.0, 0.0, 10.0],
            uv=[4.0, 10.0],
        )

        vector_a = build_node_feature_vector(
            node_a,
            ndigits=6,
            node_feature_mode=NODE_FEATURE_MODE_ROTATION_INVARIANT,
        )
        vector_b = build_node_feature_vector(
            node_b,
            ndigits=6,
            node_feature_mode=NODE_FEATURE_MODE_ROTATION_INVARIANT,
        )

        self.assertEqual(vector_a, vector_b)

    def test_gnn_record_can_use_rotation_invariant_node_features(self):
        sample_record = {
            "sample_id": "sample",
            "graph_path": "graphs/sample.json",
            "step_path": "steps/sample.step",
        }
        graph_data = {
            "nodes": [
                make_node(
                    center=[1.0, 2.0, 3.0],
                    normal=[0.0, 0.0, 1.0],
                    axis=[0.0, 0.0, 1.0],
                    bbox=[10.0, 4.0, 0.0],
                    uv=[10.0, 4.0],
                )
            ],
            "edges": [],
            "feature_sets": [],
            "derived_label_summary": {},
        }

        record = build_gnn_record(
            sample_record,
            graph_data,
            ndigits=6,
            node_feature_mode=NODE_FEATURE_MODE_ROTATION_INVARIANT,
        )

        self.assertEqual(record["node_feature_mode"], NODE_FEATURE_MODE_ROTATION_INVARIANT)
        self.assertEqual(
            len(record["node_features"][0]),
            len(node_feature_schema_for_mode(NODE_FEATURE_MODE_ROTATION_INVARIANT)),
        )


if __name__ == "__main__":
    unittest.main()
