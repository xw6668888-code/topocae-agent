"""Run the final read-only P3-A1 pilot regression suite from any working directory."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TEST_MODULE = "tests.cae_query_ground_pilot_regression"


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromName(TEST_MODULE)
    result = unittest.TextTestRunner(verbosity=2, stream=sys.stderr).run(suite)
    print(json.dumps({
        "suite": TEST_MODULE,
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "automatic_pass": result.wasSuccessful(),
        "writes_performed": False,
    }, ensure_ascii=False, sort_keys=True))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())

