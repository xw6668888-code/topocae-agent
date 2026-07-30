"""Bootstrap the final captured-reference guard before all P1-C tests."""

from __future__ import annotations

import unittest

from p1_c.runtime_state_isolation_v3 import install


install()


class RuntimeStateGuardV3BootstrapTests(unittest.TestCase):
    def test_guard_is_idempotent(self) -> None:
        install()


if __name__ == "__main__":
    unittest.main()
