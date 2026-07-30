from __future__ import annotations

"""Three-tier routing for frozen GNN, iteration-1 model, iteration-2 outputs."""

from dataclasses import dataclass
from pathlib import Path

from utils.p1_c_iteration1_run_paths import iteration_run_dir
from utils.p1_c_strict_run_paths import ROOT, strict_run_dir, validate_variant_seed


ITERATION2_ROOT = ROOT / "reports" / "p1_c_stage6_iteration2_runs_v1"
GNN_FILES = {"best.pt", "config.json", "history.json", "history.csv", "metrics.json", "predictions.pt", "same_instance_threshold_selection.json", "environment.json"}
ITERATION1_DIRS = {"candidate_training_v1", "candidate_model_v1"}


@dataclass(frozen=True)
class Iteration2RunDir:
    variant: str
    seed: int

    @property
    def output(self) -> Path:
        return ITERATION2_ROOT / f"{self.variant}_seed_{self.seed}"

    def __truediv__(self, child: object) -> Path:
        text = str(child)
        if text in GNN_FILES:
            return strict_run_dir(self.variant, self.seed) / text
        if text in ITERATION1_DIRS:
            return iteration_run_dir(self.variant, self.seed) / text
        return self.output / text

    def resolve(self) -> Path:
        return self.output.resolve()


@dataclass(frozen=True)
class Iteration2VariantSeedRunRoot:
    variant: str

    def __post_init__(self) -> None:
        if self.variant not in {"clean_5000", "clean_7200"}:
            raise ValueError(f"iteration-2 variant not approved: {self.variant}")

    def __truediv__(self, child: object) -> Iteration2RunDir:
        text = str(child)
        prefix = "same_instance_seed_"
        if not text.startswith(prefix):
            raise ValueError(f"unexpected iteration-2 run child: {text}")
        seed = int(text[len(prefix):])
        validate_variant_seed(self.variant, seed)
        return Iteration2RunDir(self.variant, seed)


def iteration2_run_dir(variant: str, seed: int) -> Path:
    validate_variant_seed(variant, seed)
    if variant not in {"clean_5000", "clean_7200"}:
        raise ValueError(f"iteration-2 variant not approved: {variant}")
    return ITERATION2_ROOT / f"{variant}_seed_{seed}"

