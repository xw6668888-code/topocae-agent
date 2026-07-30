"""Build train-only-normalized packages from the strict regenerated dataset."""

from __future__ import annotations

from pathlib import Path

import build_p1_c_training_package as protocol


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "output_generation_p1_c_clean_context_7200_strict_concept_r2"
PACKAGE_ROOT = DATASET / "training_packages"

protocol.DATASET = DATASET
protocol.PACKAGE_ROOT = PACKAGE_ROOT


def main() -> int:
    return protocol.main()


if __name__ == "__main__":
    raise SystemExit(main())
