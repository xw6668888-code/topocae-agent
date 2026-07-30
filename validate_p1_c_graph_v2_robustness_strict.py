"""Run the Graph v2 robustness protocol on the strict regenerated dataset."""

from __future__ import annotations

from pathlib import Path

import validate_p1_c_graph_v2_robustness as protocol
from p1_c.formal_generator_v7 import (
    build_planned_result,
    transformed_shape_and_audit,
)
from p1_c.formal_plan_v6 import build_plan


ROOT = Path(__file__).resolve().parent
OUTPUT = (
    ROOT
    / "output_generation_p1_c_clean_context_7200_strict_concept_r2"
    / "reports"
    / "graph_v2_robustness"
)

protocol.OUTPUT = OUTPUT
protocol.build_plan = build_plan
protocol.build_planned_result = build_planned_result
protocol.transformed_shape_and_audit = transformed_shape_and_audit


def main() -> int:
    return protocol.main()


if __name__ == "__main__":
    raise SystemExit(main())
