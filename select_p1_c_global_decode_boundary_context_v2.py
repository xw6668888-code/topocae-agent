from __future__ import annotations

from pathlib import Path

import select_p1_b_global_decode_v2 as protocol
from utils.p1_c_candidate_iteration_paths import CandidateIterationRunRoot
from utils.p1_c_topology_verifier_v5 import verify_candidate_prediction


ROOT = Path(__file__).resolve().parent
protocol.RUN_ROOT = CandidateIterationRunRoot()
protocol.TRAINING_METADATA = (
    ROOT / "output_generation_p1_c_clean_context_7200" / "training_packages" / "clean_7200" / "metadata.json"
)
protocol.verify_candidate_prediction = verify_candidate_prediction


def main() -> int:
    return protocol.main()


if __name__ == "__main__":
    raise SystemExit(main())
