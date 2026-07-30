from __future__ import annotations

from pathlib import Path
import sys

import train_p1_c_gnn_iteration1 as iteration1
import train_p1_c_gnn_v2 as cuda_safe


ROOT = Path(__file__).resolve().parent
VARIANT = "iteration2_full_cosine_schedule"
protocol = iteration1.protocol
protocol.VARIANTS = (*protocol.VARIANTS, VARIANT)
protocol.PACKAGE_BY_VARIANT[VARIANT] = "clean_7200"
protocol.PLAN = ROOT / "reports" / "p1_c_iterations_v1" / "iteration_2_hypothesis.json"
protocol.baseline._environment = cuda_safe._environment_without_cudnn_probe


def main() -> int:
    if "--patience" in sys.argv:
        raise RuntimeError("iteration 2 patience is frozen by the wrapper")
    sys.argv.extend(["--patience", "56"])
    return protocol.main()


if __name__ == "__main__":
    raise SystemExit(main())
