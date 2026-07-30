"""Run the read-only formal P3-A1 freeze regression suite."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromName(
        "tests.cae_query_ground_formal_regression"
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "automatic_pass": result.wasSuccessful(),
        "writes_performed": False,
    }, sort_keys=True))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())

