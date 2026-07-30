from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
P1_C_RUN_ROOT = ROOT / "reports" / "p1_c_training_runs_v1"
P1_C_VARIANT = "iteration2_full_cosine_schedule"


def p1_c_run_dir(seed: int) -> Path:
    return P1_C_RUN_ROOT / f"{P1_C_VARIANT}_seed_{int(seed)}"


class LegacySeedRunRoot:
    """Route the frozen P1-B CLI's seed directory convention to P1-C runs."""

    _prefix = "same_instance_seed_"

    def __truediv__(self, value: object) -> Path:
        name = str(value)
        if not name.startswith(self._prefix):
            raise ValueError(f"unexpected legacy run directory: {name}")
        return p1_c_run_dir(int(name[len(self._prefix) :]))
