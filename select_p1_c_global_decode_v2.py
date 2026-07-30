from __future__ import annotations

from pathlib import Path

import select_p1_b_global_decode_v2 as protocol
from utils.p1_c_run_paths import LegacySeedRunRoot


ROOT = Path(__file__).resolve().parent
protocol.RUN_ROOT = LegacySeedRunRoot()
protocol.TRAINING_METADATA = (
    ROOT / "output_generation_p1_c_clean_context_7200" / "training_packages" / "clean_7200" / "metadata.json"
)


def main() -> int:
    return protocol.main()


if __name__ == "__main__":
    raise SystemExit(main())
