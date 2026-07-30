from __future__ import annotations

"""P1-B semantic-overlay entry point for Holdout-v2 generation."""

import generate_p1_b_synthetic_holdout_v2 as base
from generate_p1_b1_slot_pilot import load_p1_b_semantic_index


_base_generation_resources = base.generation_resources


def _p1_b_generation_resources():
    resources = _base_generation_resources()
    resources["semantic_subtype_index"] = load_p1_b_semantic_index()
    resources["semantic_label_map"] = {
        int(key): value for key, value in resources["semantic_subtype_index"]["id_to_key"].items()
    }
    return resources


base.generation_resources = _p1_b_generation_resources


if __name__ == "__main__":
    raise SystemExit(base.main())
