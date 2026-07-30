from __future__ import annotations

"""Route unchanged P1-B stage code across frozen GNN and iteration-1 outputs."""

from dataclasses import dataclass
from pathlib import Path

from utils.p1_c_strict_run_paths import ROOT, strict_run_dir, validate_variant_seed


ITERATION_ROOT = ROOT / "reports" / "p1_c_stage6_iteration1_runs_v1"
SOURCE_ONLY_FILES = {
    "best.pt",
    "config.json",
    "history.json",
    "history.csv",
    "metrics.json",
    "predictions.pt",
    "same_instance_threshold_selection.json",
    "environment.json",
}


@dataclass(frozen=True)
class IterationRunDir:
    variant: str
    seed: int

    @property
    def source(self) -> Path:
        return strict_run_dir(self.variant, self.seed)

    @property
    def output(self) -> Path:
        return ITERATION_ROOT / f"{self.variant}_seed_{self.seed}"

    def __truediv__(self, child: object) -> Path:
        text = str(child)
        return (self.source if text in SOURCE_ONLY_FILES else self.output) / text

    def resolve(self) -> Path:
        return self.output.resolve()


@dataclass(frozen=True)
class IterationVariantSeedRunRoot:
    variant: str

    def __post_init__(self) -> None:
        if self.variant not in {"clean_5000", "clean_7200"}:
            raise ValueError(f"iteration-1 variant not approved: {self.variant}")

    def __truediv__(self, child: object) -> IterationRunDir:
        text = str(child)
        prefix = "same_instance_seed_"
        if not text.startswith(prefix):
            raise ValueError(f"unexpected iteration run child: {text}")
        seed = int(text[len(prefix):])
        validate_variant_seed(self.variant, seed)
        return IterationRunDir(self.variant, seed)


def iteration_run_dir(variant: str, seed: int) -> Path:
    validate_variant_seed(variant, seed)
    if variant not in {"clean_5000", "clean_7200"}:
        raise ValueError(f"iteration-1 variant not approved: {variant}")
    return ITERATION_ROOT / f"{variant}_seed_{seed}"

