from __future__ import annotations

"""Evaluate iteration-1 polarity candidates with unchanged verifier/decode."""

from pathlib import Path

import evaluate_p1_c_synthetic_pipeline_strict as protocol
from p1_c.iteration1_provenance import preflight_iteration_model
from utils.p1_c_iteration1_run_paths import IterationRunDir, IterationVariantSeedRunRoot
from utils.p1_c_strict_run_paths import DATASET_ROOT


POLARITY_GRAPHS = DATASET_ROOT / "graphs_polarity_v1"
_original_read = protocol.implementation._read_json


def _mapped_read(path: Path):
    source = Path(path)
    if source.parent.name == "graphs":
        source = POLARITY_GRAPHS / source.name
    return _original_read(source)


protocol.preflight_candidate_model = preflight_iteration_model
protocol.strict_run_dir = lambda variant, seed: IterationRunDir(variant, seed)
protocol.StrictVariantSeedRunRoot = IterationVariantSeedRunRoot
protocol.implementation._read_json = _mapped_read


if __name__ == "__main__":
    raise SystemExit(protocol.main())

