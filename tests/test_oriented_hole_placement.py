import unittest

from features.oriented_hole import is_axis_placement_entry_exposed


class OrientedHolePlacementTest(unittest.TestCase):
    def test_composite_top_pad_blocks_base_top_hole_entry(self):
        placement = {
            "axis": "z",
            "zone_a": 0.0,
            "zone_b": 0.0,
        }
        stock_profile = {
            "stock_type": "composite",
            "components": [
                {"role": "base", "type": "box", "center": [0.0, 0.0, 20.0], "size": [100.0, 100.0, 40.0]},
                {"role": "top_pad", "type": "box", "center": [0.0, 0.0, 50.0], "size": [40.0, 40.0, 20.0]},
            ],
        }

        self.assertFalse(is_axis_placement_entry_exposed(placement, radius=5.0, stock_profile=stock_profile))

    def test_composite_top_pad_does_not_block_distant_base_top_hole_entry(self):
        placement = {
            "axis": "z",
            "zone_a": 35.0,
            "zone_b": 35.0,
        }
        stock_profile = {
            "stock_type": "composite",
            "components": [
                {"role": "base", "type": "box", "center": [0.0, 0.0, 20.0], "size": [100.0, 100.0, 40.0]},
                {"role": "top_pad", "type": "box", "center": [0.0, 0.0, 50.0], "size": [30.0, 30.0, 20.0]},
            ],
        }

        self.assertTrue(is_axis_placement_entry_exposed(placement, radius=5.0, stock_profile=stock_profile))

    def test_cylinder_side_axis_entry_is_not_exposed_by_rectangular_workspace(self):
        placement = {
            "axis": "x",
            "zone_a": 0.0,
            "zone_b": 20.0,
        }
        stock_profile = {
            "stock_type": "cylinder",
            "dimensions": {"radius": 50.0, "H": 40.0},
        }

        self.assertFalse(is_axis_placement_entry_exposed(placement, radius=5.0, stock_profile=stock_profile))


if __name__ == "__main__":
    unittest.main()
