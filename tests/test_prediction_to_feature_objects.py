import unittest

from prediction_to_feature_objects import build_feature_objects
from tests.test_topology_postprocess import face, graph, graph_node


class PredictionToFeatureObjectsTest(unittest.TestCase):
    def test_adjacent_chamfer_faces_are_exported_as_single_face_transition_objects(self):
        graph_data = graph(
            [
                graph_node(0, "plane", normal=[1, 0, 0], descriptors={"area": 14.7}),
                graph_node(1, "plane", normal=[0, 1, 0], descriptors={"area": 14.7}),
            ],
            [
                (0, 1, "line", "convex", 52.0, 0.785),
                (1, 0, "line", "convex", 52.0, 0.785),
            ],
        )
        prediction = {
            "sample_id": "sample",
            "faces": [
                face(0, macro_id=26, macro="chamfer"),
                face(1, macro_id=26, macro="chamfer"),
            ],
        }

        result = build_feature_objects(prediction, graph_data)

        self.assertEqual([obj["face_indices"] for obj in result["feature_objects"]], [[0], [1]])
        self.assertEqual([obj["macro_feature"] for obj in result["feature_objects"]], ["chamfer", "chamfer"])



    def test_assembled_instance_candidates_can_be_exported_directly(self):
        graph_data = graph(
            [
                graph_node(0, "plane", normal=[1, 0, 0], descriptors={"area": 12.0}),
                graph_node(1, "plane", normal=[0, 1, 0], descriptors={"area": 13.0}),
            ],
            [],
        )
        prediction = {
            "sample_id": "sample",
            "faces": [
                face(0, macro_id=20, macro="rib"),
                face(1, macro_id=20, macro="rib"),
            ],
            "instance_candidate_postprocess": {
                "enabled": True,
                "records": [
                    {
                        "action": "accepted_remap",
                        "source": "assembled_instance_candidate",
                        "proposal_macro": "web",
                        "predicted_label": "rib",
                        "confidence": 0.91,
                        "face_indices": [0, 1],
                    }
                ],
            },
        }

        result = build_feature_objects(
            prediction,
            graph_data,
            grouping_mode="assembled_instance_candidates",
        )

        self.assertEqual([obj["face_indices"] for obj in result["feature_objects"]], [[0, 1]])
        obj = result["feature_objects"][0]
        self.assertEqual(obj["grouping_method"], "assembled_instance_candidates")
        self.assertEqual(obj["macro_feature"], "rib")
        self.assertEqual(obj["source_assembly_record"]["source"], "assembled_instance_candidate")
        self.assertEqual(obj["source_assembly_record"]["confidence"], 0.91)


    def test_assembled_instance_candidates_skip_records_demoted_to_stock(self):
        graph_data = graph(
            [
                graph_node(0, "plane", normal=[1, 0, 0], descriptors={"area": 12.0}),
                graph_node(1, "plane", normal=[0, 1, 0], descriptors={"area": 13.0}),
            ],
            [],
        )
        prediction = {
            "sample_id": "sample",
            "faces": [
                face(0, macro_id=0, macro="stock"),
                face(1, macro_id=0, macro="stock"),
            ],
            "instance_candidate_postprocess": {
                "enabled": True,
                "records": [
                    {
                        "action": "accepted_remap",
                        "source": "assembled_instance_candidate",
                        "proposal_macro": "web",
                        "predicted_label": "rib",
                        "confidence": 0.91,
                        "face_indices": [0, 1],
                    }
                ],
            },
        }

        result = build_feature_objects(
            prediction,
            graph_data,
            grouping_mode="assembled_instance_candidates",
        )

        self.assertEqual(result["feature_objects"], [])
        self.assertEqual(result["feature_object_count"], 0)

if __name__ == "__main__":
    unittest.main()

