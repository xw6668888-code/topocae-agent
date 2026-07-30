from __future__ import annotations

from pathlib import Path

import evaluate_p1_c_synthetic_boundary_context_v5 as protocol
from utils.p1_c_candidate_iteration_paths import CandidateIterationRunDir


class EvaluationRunDir(CandidateIterationRunDir):
    def __truediv__(self, value: object) -> Path:
        if str(value) == "candidate_training_v1":
            return self.path / "candidate_training_v1"
        return super().__truediv__(value)


class EvaluationRunRoot:
    _prefix = "same_instance_seed_"

    def __truediv__(self, value: object) -> EvaluationRunDir:
        name = str(value)
        if not name.startswith(self._prefix):
            raise ValueError(f"unexpected legacy run directory: {name}")
        return EvaluationRunDir(int(name[len(self._prefix) :]))


protocol.implementation.RUN_ROOT = EvaluationRunRoot()


def main() -> int:
    return protocol.main()


if __name__ == "__main__":
    raise SystemExit(main())
