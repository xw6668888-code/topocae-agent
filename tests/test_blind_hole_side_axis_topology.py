import random
import unittest
from pathlib import Path

from main import generate_annotated_sample
from utils.generation_context import build_generation_context
from utils.ontology import export_layer_label_map, load_layer_index, load_ontology_maps
from utils.topology_rules import validate_feature_topology


ROOT = Path(__file__).resolve().parents[1]


class BlindHoleSideAxisTopologyTest(unittest.TestCase):
    def test_side_axis_blind_hole_keeps_cylindrical_wall_and_bottom_in_same_instance(self):
        ontology_maps = load_ontology_maps(ROOT / "CAE_feature_ontology.yaml")
        primitive_index = load_layer_index(ROOT / "CAE_feature_ontology.yaml", "primitive_surface")
        idealization_index = load_layer_index(ROOT / "CAE_feature_ontology.yaml", "idealization_tag")
        semantic_index = load_layer_index(ROOT / "face_semantic_subtype_ontology.yaml", "face_semantic_subtype")

        for axis in ("x", "y"):
            with self.subTest(axis=axis):
                random.seed(20260606)
                _shape, graph_data, feature_count = generate_annotated_sample(
                    L=120.0,
                    W=110.0,
                    H=50.0,
                    ontology_maps=ontology_maps,
                    primitive_surface_index=primitive_index,
                    semantic_subtype_index=semantic_index,
                    idealization_tag_index=idealization_index,
                    primitive_label_map=export_layer_label_map(primitive_index),
                    semantic_label_map=export_layer_label_map(semantic_index),
                    idealization_label_map=export_layer_label_map(idealization_index),
                    generator_names=["BlindHoleGenerator"],
                    instance_count_range=(1, 1),
                    require_nonempty_sample=True,
                    max_generation_attempts=10,
                    generation_context=build_generation_context({
                        "stock_profiles": {"enabled": False},
                        "feature_orientation": {"hole_axes": [axis]},
                        "feature_density": {"enabled": True, "max_total_instances_per_sample": 1, "max_instances_per_generator": 1},
                    }),
                )

                self.assertEqual(feature_count, 1)
                feature_set = graph_data["feature_sets"][0]
                self.assertEqual(feature_set["macro_feature"], "blind_hole_round")
                validation = validate_feature_topology(
                    feature_set["macro_feature"],
                    feature_set["node_indices"],
                    graph_data,
                )

                self.assertTrue(validation["valid"], validation)
                self.assertGreaterEqual(len(feature_set["node_indices"]), 2)


if __name__ == "__main__":
    unittest.main()
