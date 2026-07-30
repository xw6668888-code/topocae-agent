from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.ontology import load_layer_index, load_ontology_maps


def load_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("These regression checks require PyYAML.") from exc

    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


class StaticStructuralV1ScopeTest(unittest.TestCase):
    def test_macro_ontology_matches_v1_scope(self) -> None:
        maps = load_ontology_maps(ROOT / "CAE_feature_ontology.yaml")
        enabled = {item.key for item in maps.macro_features if item.enabled}

        self.assertIn("obround_hole", enabled)
        self.assertIn("rectangular_boss", enabled)
        self.assertEqual(maps.resolve_macro_feature("pad").key, "rectangular_boss")
        self.assertEqual(maps.resolve_macro_feature("rectangle_boss").key, "rectangular_boss")

        excluded = {
            "bolt_hole_pattern",
            "flange",
            "support_pad",
            "bearing_seat",
            "gusset",
            "notch_relief",
            "triangle_prism_boss",
            "rectangle_prism_boss",
            "triangle_frustum_boss",
            "rectangle_frustum_boss",
        }
        self.assertTrue(excluded.isdisjoint(enabled))

    def test_semantic_subtype_recommendations_match_v1_scope(self) -> None:
        subtype_index = load_layer_index(ROOT / "face_semantic_subtype_ontology.yaml", "face_semantic_subtype")
        enabled = set(subtype_index["enabled_keys"])

        self.assertIn("obround_hole_side_plane", enabled)
        self.assertIn("obround_hole_end_cylinder", enabled)
        self.assertIn("pad_top_plane", enabled)
        self.assertIn("pad_side_plane", enabled)

        excluded = {
            "flange_top_plane",
            "flange_side_plane",
            "support_pad_top_plane",
            "bolt_hole_pattern_hole_cylinder",
            "bearing_seat_side_cylinder",
            "notch_relief_side_plane",
            "cone_boss_side",
        }
        self.assertTrue(excluded.isdisjoint(enabled))

    def test_default_config_uses_v1_generators_and_targets(self) -> None:
        config = load_yaml(ROOT / "configs" / "default_dataset_config.yaml")
        generators = set(config["generation"]["enabled_generators"])
        targets = set(config["generation"]["coverage_control"]["target_macro_features"])

        self.assertIn("ObroundHoleGenerator", generators)
        self.assertIn("PadGenerator", generators)
        self.assertNotIn("BoltHolePatternGenerator", generators)
        self.assertNotIn("FlangeGenerator", generators)
        self.assertNotIn("SupportPadGenerator", generators)

        self.assertIn("obround_hole", targets)
        self.assertIn("rectangular_boss", targets)
        self.assertNotIn("bolt_hole_pattern", targets)
        self.assertNotIn("flange", targets)


if __name__ == "__main__":
    unittest.main()
