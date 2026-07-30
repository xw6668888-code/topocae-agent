"""Align strict package head declarations with the frozen P1-B architecture."""

from __future__ import annotations

from pathlib import Path

import repair_p1_c_training_package_metadata_v2 as protocol


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "output_generation_p1_c_clean_context_7200_strict_concept_r2"
PACKAGES = DATASET / "training_packages"

protocol.DATASET = DATASET
protocol.PACKAGES = PACKAGES


def main() -> int:
    return protocol.main()


if __name__ == "__main__":
    raise SystemExit(main())
