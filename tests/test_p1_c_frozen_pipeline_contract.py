from __future__ import annotations

import json
from pathlib import Path
import unittest

from p1_c_general_recognition_pipeline_v1 import strip_inference_truth, strip_prediction_truth


ROOT = Path(__file__).resolve().parents[1]


class FrozenPipelineContractTests(unittest.TestCase):
    def test_graph_truth_is_stripped_before_inference(self) -> None:
        graph_path = next((ROOT / "output_generation_p1_c_synthetic_holdout_v3" / "graphs").glob("*.json"))
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        stripped = strip_inference_truth(graph)
        self.assertNotIn("feature_sets", stripped)
        for node in stripped["nodes"]:
            self.assertFalse({"macro_feature", "semantic_role", "instance_id"} & set(node))

    def test_prediction_evaluation_truth_is_stripped(self) -> None:
        prediction = {
            "truth": {"bad": True}, "faces": [{"idx": 0, "macro_feature": "rib", "truth_macro_feature": "web"}]
        }
        stripped = strip_prediction_truth(prediction)
        self.assertNotIn("truth", stripped)
        self.assertNotIn("truth_macro_feature", stripped["faces"][0])
        self.assertEqual(stripped["faces"][0]["macro_feature"], "rib")


if __name__ == "__main__":
    unittest.main()
