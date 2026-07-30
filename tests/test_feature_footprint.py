import unittest

from utils.feature_footprint import capsule_footprint, circle_footprint, footprint_conflicts, polyline_capsule_footprint


class FeatureFootprintTest(unittest.TestCase):
    def test_circle_footprints_conflict_with_clearance(self):
        existing = [circle_footprint(0.0, 0.0, 5.0, operation="cut")]
        candidate = circle_footprint(9.0, 0.0, 4.0, operation="cut")

        self.assertTrue(footprint_conflicts(candidate, existing, clearance=1.0))

    def test_capsule_slot_footprint_conflicts_with_crossing_circle(self):
        existing = [capsule_footprint(0.0, 0.0, 40.0, 6.0, 0.0, operation="cut")]
        candidate = circle_footprint(0.0, 2.0, 3.0, operation="cut")

        self.assertTrue(footprint_conflicts(candidate, existing, clearance=1.0))

    def test_non_overlapping_cut_footprints_do_not_conflict(self):
        existing = [capsule_footprint(0.0, 0.0, 40.0, 6.0, 0.0, operation="cut")]
        candidate = circle_footprint(0.0, 30.0, 3.0, operation="cut")

        self.assertFalse(footprint_conflicts(candidate, existing, clearance=1.0))

    def test_polyline_capsule_conflicts_with_circle_near_any_segment(self):
        slot = polyline_capsule_footprint([(-20.0, 0.0), (0.0, 8.0), (20.0, 0.0)], width=6.0, feature_kind="slot_u")
        hole = circle_footprint(0.0, 8.0, 3.0, feature_kind="blind_hole_round")

        self.assertTrue(footprint_conflicts(slot, [hole], clearance=1.0))

    def test_polyline_capsule_does_not_conflict_with_far_circle(self):
        slot = polyline_capsule_footprint([(-20.0, 0.0), (0.0, 8.0), (20.0, 0.0)], width=6.0, feature_kind="slot_u")
        hole = circle_footprint(0.0, 30.0, 3.0, feature_kind="blind_hole_round")

        self.assertFalse(footprint_conflicts(slot, [hole], clearance=1.0))

    def test_polyline_capsule_conflicts_with_crossing_capsule(self):
        slot = polyline_capsule_footprint([(-20.0, 0.0), (0.0, 8.0), (20.0, 0.0)], width=6.0, feature_kind="slot_u")
        crossing_slot = capsule_footprint(0.0, 0.0, 30.0, 6.0, 1.57079632679, operation="cut")

        self.assertTrue(footprint_conflicts(slot, [crossing_slot], clearance=1.0))


if __name__ == "__main__":
    unittest.main()
