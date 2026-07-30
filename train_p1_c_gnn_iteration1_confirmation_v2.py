from __future__ import annotations

from pathlib import Path

import train_p1_c_gnn_iteration1 as iteration
import train_p1_c_gnn_v2 as cuda_safe


ROOT = Path(__file__).resolve().parent
iteration.protocol.PLAN = (
    ROOT / "reports" / "p1_c_iterations_v1" / "iteration_1_confirmation_plan_v2.json"
)
iteration.protocol.baseline._environment = cuda_safe._environment_without_cudnn_probe


def main() -> int:
    return iteration.main()


if __name__ == "__main__":
    raise SystemExit(main())
