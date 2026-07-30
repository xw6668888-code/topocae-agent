from __future__ import annotations

from pathlib import Path

from utils.p1_c_run_paths import p1_c_run_dir


PACKAGE_NAME = "candidate_training_v2_boundary_context"
MODEL_NAME = "candidate_model_v2_boundary_context"
SELECTION_NAME = "global_decode_v2_boundary_context_selection.json"
PIPELINE_NAME = "synthetic_pipeline_v2_boundary_context"


class CandidateIterationRunDir:
    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        self.path = p1_c_run_dir(self.seed)

    def __truediv__(self, value: object) -> Path:
        name = str(value)
        mapping = {
            "candidate_training_v1": PACKAGE_NAME,
            "candidate_model_v1": MODEL_NAME,
            "global_decode_v2_selection.json": SELECTION_NAME,
            "synthetic_pipeline_v1": PIPELINE_NAME,
        }
        return self.path / mapping.get(name, name)

    def resolve(self) -> Path:
        return self.path.resolve()


class CandidateIterationRunRoot:
    _prefix = "same_instance_seed_"

    def __truediv__(self, value: object) -> CandidateIterationRunDir:
        name = str(value)
        if not name.startswith(self._prefix):
            raise ValueError(f"unexpected legacy run directory: {name}")
        return CandidateIterationRunDir(int(name[len(self._prefix) :]))
