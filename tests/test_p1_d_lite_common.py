from __future__ import annotations

import unittest

from p1_d_lite.common import area_compatible, is_connected, strip_graph_truth


class P1DLiteCommonTests(unittest.TestCase):
    def test_placeholder(self) -> None:
        self.assertTrue(area_compatible(100.0, 101.0))


if __name__ == "__main__":
    unittest.main()
