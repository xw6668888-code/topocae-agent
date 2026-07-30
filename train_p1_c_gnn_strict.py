"""Train the frozen P1-C architecture on strict regenerated packages."""

from __future__ import annotations

from pathlib import Path

import train_p1_c_gnn as protocol
import train_p1_c_gnn_v2  # applies the CUDA environment probe fix only


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "output_generation_p1_c_clean_context_7200_strict_concept_r2"
PACKAGE_ROOT = DATASET / "training_packages"
OUTPUT_ROOT = ROOT / "reports" / "p1_c_strict_training_runs_v2"
PLAN = ROOT / "configs" / "p1_c" / "training_plan_strict_v2.json"
ROBUSTNESS = DATASET / "reports" / "graph_v2_robustness" / "robustness.json"
_ORIGINAL_READ_JSON = protocol._read_json


def _read_json(path: Path):
    normalized = Path(path).resolve()
    legacy_robustness = (
        ROOT
        / "output_generation_p1_c_clean_context_7200"
        / "reports"
        / "graph_v2_robustness"
        / "robustness.json"
    ).resolve()
    if normalized == legacy_robustness:
        normalized = ROBUSTNESS.resolve()
    return _ORIGINAL_READ_JSON(normalized)


protocol.PACKAGE_ROOT = PACKAGE_ROOT
protocol.OUTPUT_ROOT = OUTPUT_ROOT
protocol.PLAN = PLAN
protocol._read_json = _read_json


def main() -> int:
    return protocol.main()


if __name__ == "__main__":
    raise SystemExit(main())
