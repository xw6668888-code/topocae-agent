"""Bootstrap the refined state guard before all other P1-C tests."""

from __future__ import annotations

import unittest

from p1_c.runtime_state_isolation_v2 import install


install()


class RuntimeStateGuardV2BootstrapTests(unittest.TestCase):
    def test_guard_is_idempotent(self) -> None:
        install()


if __name__ == "__main__":
    unittest.main()
