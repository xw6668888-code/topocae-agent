import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import main


class StepWriteQualityGateTest(unittest.TestCase):
    def test_build_single_dataset_sample_retries_when_written_step_is_invalid(self):
        generated = [
            (
                object(),
                {"nodes": [], "feature_sets": [], "generation_quality": {}},
                1,
            ),
            (
                object(),
                {"nodes": [], "feature_sets": [], "generation_quality": {}},
                2,
            ),
        ]
        invalid_report = {
            "step_path": "sample.step",
            "readable": True,
            "solid_count": 1,
            "shell_count": 1,
            "face_count": 6,
            "edge_count": 12,
            "signed_volume": 1.0,
            "valid": False,
            "issues": ["invalid_topology"],
        }
        valid_report = {
            **invalid_report,
            "valid": True,
            "issues": [],
        }

        with TemporaryDirectory() as temp_dir:
            output_step = Path(temp_dir) / "sample.step"
            output_json = Path(temp_dir) / "sample.json"

            with patch.object(main, "generate_annotated_sample", side_effect=generated) as generate_mock, patch.object(
                main,
                "write_step_with_positive_volume",
            ), patch.object(main, "inspect_step_file", side_effect=[invalid_report, valid_report]):
                graph_data = main.build_single_dataset_sample(
                    L=10.0,
                    W=10.0,
                    H=10.0,
                    output_step=output_step,
                    output_json=output_json,
                    ontology_maps=None,
                    primitive_surface_index=None,
                    semantic_subtype_index=None,
                    idealization_tag_index=None,
                    primitive_label_map=None,
                    semantic_label_map=None,
                    idealization_label_map=None,
                    generator_names=[],
                    instance_count_range=(1, 1),
                    require_nonempty_sample=True,
                    max_generation_attempts=2,
                    write_step=True,
                    write_graph=True,
                    generation_context=None,
                )

            self.assertEqual(2, generate_mock.call_count)
            self.assertEqual(1, graph_data["generation_quality"]["step_write_rejected_attempt_count"])
            written_graph = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual([], written_graph["generation_quality"]["step_write_validation"]["issues"])


if __name__ == "__main__":
    unittest.main()
