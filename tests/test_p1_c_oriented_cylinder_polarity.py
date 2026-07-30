from __future__ import annotations

import json
from pathlib import Path
import unittest

from OCC.Extend.DataExchange import read_step_file
from OCC.Extend.TopologyUtils import TopologyExplorer

from p1_c.oriented_cylinder_polarity_v1 import FEATURE_NAME, oriented_cylindrical_radial_alignment
from utils.p1_c_candidate_features_polarity_v1 import CANDIDATE_FEATURE_NAME, ENGINEERED_FEATURE_SCHEMA
from utils.p1_b_candidate_features import ENGINEERED_FEATURE_SCHEMA as BASE_SCHEMA
from utils.p1_c_strict_run_paths import DATASET_ROOT


class OrientedCylinderPolarityTests(unittest.TestCase):
    def _feature_cylinder_value(self, macro: str) -> float:
        graph_path = next((DATASET_ROOT / "graphs").glob(f"*{macro}*target_only*.json"))
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        shape = read_step_file(str(DATASET_ROOT / "steps" / f"{graph['sample_id']}.step"), verbosity=False)
        for face, node in zip(TopologyExplorer(shape).faces(), graph["nodes"]):
            if node.get("macro_feature") == macro and node.get("primitive_surface") == "cylinder":
                return oriented_cylindrical_radial_alignment(face)
        self.fail(f"no feature cylinder for {macro}")

    def test_pocket_and_boss_have_opposite_material_polarity(self) -> None:
        self.assertLess(self._feature_cylinder_value("pocket_circular"), -0.999)
        self.assertGreater(self._feature_cylinder_value("cylinder_boss"), 0.999)

    def test_candidate_schema_adds_exactly_one_generic_feature(self) -> None:
        self.assertEqual(ENGINEERED_FEATURE_SCHEMA[:-1], list(BASE_SCHEMA))
        self.assertEqual(ENGINEERED_FEATURE_SCHEMA[-1], CANDIDATE_FEATURE_NAME)
        self.assertIn("oriented_cylindrical", FEATURE_NAME)


if __name__ == "__main__":
    unittest.main()
