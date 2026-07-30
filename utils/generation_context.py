from __future__ import annotations

from copy import deepcopy


DEFAULT_STOCK_PROFILE_CONFIG = {
    "enabled": True,
    "weights": {
        "block": 0.22,
        "plate": 0.20,
        "beam": 0.20,
        "cylinder": 0.18,
        "l_plate": 0.0,
        "composite": 0.20,
    },
    "composite_component_count_range": [2, 3],
}


DEFAULT_FEATURE_ORDERING_CONFIG = {
    "enabled": True,
}


DEFAULT_FEATURE_ORIENTATION_CONFIG = {
    "hole_axes": ["z", "x", "y"],
}


DEFAULT_FEATURE_DENSITY_CONFIG = {
    "enabled": True,
    "max_total_instances_per_sample": 8,
    "max_instances_per_generator": 1,
}


DEFAULT_TRANSITION_FEATURE_CONFIG = {
    "min_edge_length": 6.0,
    "min_size": 0.8,
    "max_size": 3.0,
    "midpoint_clearance_factor": 5.0,
}


DEFAULT_RIB_FEATURE_CONFIG = {
    "placement_modes": ["corner"],
    "corner_probability": 1.0,
}


FEATURE_STAGE_BY_GENERATOR = {
    "RectPocketGenerator": 10,
    "CircularPocketGenerator": 10,
    "SteppedHoleGenerator": 20,
    "ThroughHoleGenerator": 20,
    "BlindHoleGenerator": 20,
    "CounterboreHoleGenerator": 20,
    "CountersinkHoleGenerator": 20,
    "ObroundHoleGenerator": 20,
    "BossFeatureGenerator": 30,
    "PadGenerator": 30,
    "SlotRectGenerator": 40,
    "USlotGenerator": 40,
    "SweptSlotGenerator": 40,
    "RibGenerator": 50,
    "WebGenerator": 50,
    "FilletGenerator": 90,
    "ChamferGenerator": 90,
}


def deep_merge(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def build_generation_context(generation_config: dict | None) -> dict:
    generation_config = generation_config or {}
    return {
        "stock_profiles": deep_merge(
            DEFAULT_STOCK_PROFILE_CONFIG,
            generation_config.get("stock_profiles", {}),
        ),
        "feature_ordering": deep_merge(
            DEFAULT_FEATURE_ORDERING_CONFIG,
            generation_config.get("feature_ordering", {}),
        ),
        "feature_orientation": deep_merge(
            DEFAULT_FEATURE_ORIENTATION_CONFIG,
            generation_config.get("feature_orientation", {}),
        ),
        "feature_density": deep_merge(
            DEFAULT_FEATURE_DENSITY_CONFIG,
            generation_config.get("feature_density", {}),
        ),
        "transition_features": deep_merge(
            DEFAULT_TRANSITION_FEATURE_CONFIG,
            generation_config.get("transition_features", {}),
        ),
        "rib_features": deep_merge(
            DEFAULT_RIB_FEATURE_CONFIG,
            generation_config.get("rib_features", {}),
        ),
    }


def order_generator_names(generator_names: list[str], enabled: bool = True) -> list[str]:
    if not enabled:
        return list(generator_names)
    indexed_names = list(enumerate(generator_names))
    indexed_names.sort(key=lambda item: (FEATURE_STAGE_BY_GENERATOR.get(item[1], 60), item[0]))
    return [name for _index, name in indexed_names]


def cap_requested_instances(
    requested_instances: int,
    remaining_budget: int,
    max_instances_per_generator: int | None = None,
) -> int:
    if remaining_budget <= 0:
        return 0
    capped = min(int(requested_instances), int(remaining_budget))
    if max_instances_per_generator is not None:
        capped = min(capped, int(max_instances_per_generator))
    return max(0, capped)
