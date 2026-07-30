from __future__ import annotations

"""Fail-closed path mapping for the approved P1-C strict training matrix."""

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "reports" / "p1_c_strict_training_runs_v2"
DATASET_ROOT = ROOT / "output_generation_p1_c_clean_context_7200_strict_concept_r2"

VARIANT_SPECS = {
    "clean_5000": {
        "seeds": (20260718, 20260719, 20260720),
        "package": "clean_5000",
        "candidate_instance_balanced_loss": True,
        "canonical_patch": True,
    },
    "clean_7200": {
        "seeds": (20260718, 20260719, 20260720),
        "package": "clean_7200",
        "candidate_instance_balanced_loss": True,
        "canonical_patch": True,
    },
    "single_only": {
        "seeds": (20260718,),
        "package": "single_only",
        "candidate_instance_balanced_loss": True,
        "canonical_patch": True,
    },
    "no_instance_balanced_loss": {
        "seeds": (20260718,),
        "package": "clean_7200",
        "candidate_instance_balanced_loss": False,
        "canonical_patch": True,
    },
    "no_canonical_patch": {
        "seeds": (20260718,),
        "package": "clean_7200",
        "candidate_instance_balanced_loss": True,
        "canonical_patch": False,
    },
}


def validate_variant_seed(variant: str, seed: int) -> None:
    if variant not in VARIANT_SPECS:
        raise ValueError(f"unapproved strict variant: {variant}")
    if seed not in VARIANT_SPECS[variant]["seeds"]:
        raise ValueError(f"unapproved seed {seed} for strict variant {variant}")


def strict_run_dir(variant: str, seed: int) -> Path:
    validate_variant_seed(variant, seed)
    return RUN_ROOT / f"{variant}_seed_{seed}"


def training_metadata_path(variant: str) -> Path:
    if variant not in VARIANT_SPECS:
        raise ValueError(f"unapproved strict variant: {variant}")
    return DATASET_ROOT / "training_packages" / str(VARIANT_SPECS[variant]["package"]) / "metadata.json"


def candidate_model_name(variant: str) -> str:
    if variant not in VARIANT_SPECS:
        raise ValueError(f"unapproved strict variant: {variant}")
    return "candidate_model_no_canonical_patch_v1" if not VARIANT_SPECS[variant]["canonical_patch"] else "candidate_model_v1"


@dataclass(frozen=True)
class StrictVariantSeedRunRoot:
    """Adapter for unchanged P1-B code that appends ``same_instance_seed_*``."""

    variant: str

    def __post_init__(self) -> None:
        if self.variant not in VARIANT_SPECS:
            raise ValueError(f"unapproved strict variant: {self.variant}")

    def __truediv__(self, child: object) -> Path:
        text = str(child)
        prefix = "same_instance_seed_"
        if not text.startswith(prefix):
            raise ValueError(f"unexpected run child: {text}")
        try:
            seed = int(text[len(prefix):])
        except ValueError as exc:
            raise ValueError(f"invalid strict seed path: {text}") from exc
        return strict_run_dir(self.variant, seed)

