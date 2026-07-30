from __future__ import annotations

from pathlib import Path

import build_p1_b_candidate_training_package_v2 as corrected
from utils.p1_c_run_paths import LegacySeedRunRoot


ROOT = Path(__file__).resolve().parent
implementation = corrected.implementation
implementation.RUN_ROOT = LegacySeedRunRoot()
implementation.TRAINING_METADATA = (
    ROOT
    / "output_generation_p1_c_clean_context_7200"
    / "training_packages"
    / "clean_7200"
    / "metadata.json"
)


def main() -> int:
    return implementation.main()


if __name__ == "__main__":
    raise SystemExit(main())
