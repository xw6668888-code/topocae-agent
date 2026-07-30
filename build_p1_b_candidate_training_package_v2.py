from __future__ import annotations

"""Corrected entry point for the candidate-package builder.

The canonical seven-source vocabulary lives in ``p1_b_candidate_features``;
expose it to the builder's proposer import without mutating the established
proposal implementation.
"""

from utils.p1_b_candidate_features import PROPOSAL_SOURCES
import utils.p1_b_candidate_proposer as proposer

proposer.PROPOSAL_SOURCES = PROPOSAL_SOURCES

import build_p1_b_candidate_training_package as implementation


if __name__ == "__main__":
    raise SystemExit(implementation.main())
