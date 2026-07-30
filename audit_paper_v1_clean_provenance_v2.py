from __future__ import annotations

import json
from pathlib import Path

import audit_paper_v1_provenance as base


REPO_ROOT = Path(__file__).resolve().parent
EXCLUSION_PATH = (
    REPO_ROOT / "configs" / "paper_v1_clean_candidate" / "quality_exclusions.json"
)
ORIGINAL_LOAD_SPLIT_IDS = base.load_split_ids


def load_effective_split_ids(dataset_config: dict) -> dict[str, list[str]]:
    splits = ORIGINAL_LOAD_SPLIT_IDS(dataset_config)
    exclusions = json.loads(EXCLUSION_PATH.read_text(encoding="utf-8"))["exclusions"]
    return {
        split_name: [
            sample_id
            for sample_id in sample_ids
            if sample_id not in exclusions.get(split_name, {})
        ]
        for split_name, sample_ids in splits.items()
    }


if __name__ == "__main__":
    base.load_split_ids = load_effective_split_ids
    base.main()
