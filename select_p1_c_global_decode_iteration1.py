from __future__ import annotations

"""Reuse the strict selector against iteration-1 candidate artifacts."""

import select_p1_c_global_decode_strict as protocol
from p1_c.iteration1_provenance import preflight_iteration_model
from utils.p1_c_iteration1_run_paths import IterationRunDir


protocol.preflight_candidate_model = preflight_iteration_model
protocol.strict_run_dir = lambda variant, seed: IterationRunDir(variant, seed)


if __name__ == "__main__":
    raise SystemExit(protocol.main())

