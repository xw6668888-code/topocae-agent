"""Load the revision-state guard before unittest imports other P1-C modules."""

from __future__ import annotations

import unittest

from p1_c.runtime_state_isolation_v1 import install


install()


class RuntimeStateGuardBootstrapTests(unittest.TestCase):
    def test_guard_is_idempotent(self) -> None:
        install()


if __name__ == "__main__":
    unittest.main()
