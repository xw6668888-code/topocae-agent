from __future__ import annotations

"""Compatibility entry point for the frozen P1-B Real-Dev evaluation.

The underlying proposer exposes its records as ``candidates``.  The reporting
driver also wants a scalar diagnostic count, so this adapter derives that
count without changing any candidate, prediction, threshold, or decision.
"""

import run_p1_b_real_dev_evaluation as base


_run_frozen_pipeline = base.run_p1_b_general_recognition


def _run_with_proposal_count(*args, **kwargs):
    result = _run_frozen_pipeline(*args, **kwargs)
    proposal_result = result["candidate_proposals"]
    proposal_result.setdefault("proposal_count", len(proposal_result.get("candidates", [])))
    return result


base.run_p1_b_general_recognition = _run_with_proposal_count


if __name__ == "__main__":
    raise SystemExit(base.main())
