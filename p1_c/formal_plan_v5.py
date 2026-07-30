from __future__ import annotations

from collections import Counter
from typing import Iterable

from p1_c import formal_plan as core
from p1_c import formal_plan_v4 as revision4  # activates constrained Latin streams
from p1_c.all_features import HOST_FAMILIES, MACRO_CLASSES
from p1_c.rib_web_v3 import RIB_HOST_FAMILIES, WEB_HOST_FAMILIES


PLAN_REVISION = "p1_c_clean_context_strict_concept_plan_r5"
SEED_NAMESPACE = 202608100000
FORBIDDEN_CLASS_PAIRS = revision4.FORBIDDEN_CLASS_PAIRS
CONTEXT_QUOTAS = core.CONTEXT_QUOTAS
PlannedSample = core.PlannedSample
_BASE_AUDIT_PLAN = core.audit_plan


def _supported_host_family(wall_macro: str, index: int) -> str:
    if wall_macro == "rib":
        families = RIB_HOST_FAMILIES
    elif wall_macro == "web":
        families = WEB_HOST_FAMILIES
    else:
        raise KeyError(wall_macro)
    return families[index % len(families)]


def build_plan() -> list[PlannedSample]:
    secondary_streams = {
        primary_index: revision4._secondary_stream(primary_index)
        for primary_index in range(len(MACRO_CLASSES))
    }
    plan: list[PlannedSample] = []
    for primary_index, primary in enumerate(MACRO_CLASSES):
        stream = secondary_streams[primary_index]
        occurrence_cursor = 0
        class_sample_index = 0
        for context in ("target_only", "pair", "triple"):
            for context_index in range(CONTEXT_QUOTAS[context]):
                if context == "target_only":
                    secondary_features: tuple[str, ...] = ()
                elif context == "pair":
                    secondary_features = (stream[occurrence_cursor],)
                    occurrence_cursor += 1
                else:
                    secondary_features = (
                        stream[occurrence_cursor],
                        stream[occurrence_cursor + 1],
                    )
                    occurrence_cursor += 2
                if len(set(secondary_features)) != len(secondary_features):
                    raise AssertionError(
                        f"duplicate triple secondary for {primary}: {secondary_features}"
                    )
                macros = (primary, *secondary_features)
                supported_walls = [macro for macro in macros if macro in {"rib", "web"}]
                if len(supported_walls) > 1:
                    raise AssertionError(f"forbidden supported-wall pair in plan: {macros}")
                if supported_walls:
                    host_family = _supported_host_family(
                        supported_walls[0], primary_index + class_sample_index
                    )
                    host_family_kind = "supported_wall_strict_concept"
                else:
                    host_family = HOST_FAMILIES[
                        (primary_index + class_sample_index) % len(HOST_FAMILIES)
                    ]
                    host_family_kind = "general"
                sample_id = (
                    f"p1c_strict_{primary_index + 1:02d}_{primary}_"
                    f"{class_sample_index:04d}_{context}"
                )
                split = core._split_for(context, context_index)
                seed = SEED_NAMESPACE + primary_index * 10000 + class_sample_index
                plan.append(
                    PlannedSample(
                        sample_id=sample_id,
                        primary_target=primary,
                        primary_index=primary_index,
                        class_sample_index=class_sample_index,
                        context=context,
                        context_index=context_index,
                        secondary_features=secondary_features,
                        split=split,
                        seed=seed,
                        host_family=host_family,
                        host_family_kind=host_family_kind,
                        rotation_axis=core.ROTATION_AXES[
                            (primary_index * 3 + class_sample_index) % len(core.ROTATION_AXES)
                        ],
                        rotation_angle_radians=core.ROTATION_ANGLES[
                            (primary_index + class_sample_index) % len(core.ROTATION_ANGLES)
                        ],
                        uniform_scale=core.SCALE_FACTORS[
                            (primary_index * 2 + class_sample_index) % len(core.SCALE_FACTORS)
                        ],
                        base_geometry_family_id=(
                            f"p1c-strict-base::{primary_index:02d}::{class_sample_index:04d}"
                        ),
                        parametric_family_id=(
                            f"p1c-strict-param::{primary}::{context}::{host_family}::"
                            f"{class_sample_index:04d}"
                        ),
                    )
                )
                class_sample_index += 1
        if occurrence_cursor != 200:
            raise AssertionError(
                f"secondary occurrence cursor mismatch for {primary}: {occurrence_cursor}"
            )
    audit = audit_plan(plan)
    if not audit["passed"]:
        raise RuntimeError(f"strict P1-C formal plan failed: {audit['failures']}")
    return plan


def audit_plan(plan: Iterable[PlannedSample]) -> dict[str, object]:
    records = list(plan)
    payload = dict(_BASE_AUDIT_PLAN(records))
    failures = list(payload["failures"])
    incompatible: list[dict[str, str]] = []
    source_macro_counts: Counter[str] = Counter()
    for record in records:
        walls = [
            macro
            for macro in (record.primary_target, *record.secondary_features)
            if macro in {"rib", "web"}
        ]
        if not walls:
            continue
        wall_macro = walls[0]
        source_macro_counts[wall_macro] += 1
        allowed = RIB_HOST_FAMILIES if wall_macro == "rib" else WEB_HOST_FAMILIES
        if record.host_family not in allowed:
            incompatible.append(
                {
                    "sample_id": record.sample_id,
                    "wall_macro": wall_macro,
                    "host_family": record.host_family,
                }
            )
    if incompatible:
        failures.append(f"concept-incompatible supported-wall hosts: {len(incompatible)}")
    if int(payload["directed_pair_max"]) > 13:
        failures.append(f"directed pair max exceeds frozen cap 13: {payload['directed_pair_max']}")
    payload.update(
        {
            "schema_version": "topocae_p1_c_clean_context_strict_plan_audit_1",
            "plan_revision": PLAN_REVISION,
            "seed_namespace": SEED_NAMESPACE,
            "passed": not failures,
            "failures": failures,
            "supported_wall_source_macro_counts": dict(source_macro_counts),
            "concept_incompatible_supported_wall_count": len(incompatible),
            "concept_incompatible_examples": incompatible[:20],
            "rib_host_families": list(RIB_HOST_FAMILIES),
            "web_host_families": list(WEB_HOST_FAMILIES),
            "semantic_relabel_permitted": False,
        }
    )
    return payload


compatibility_matrix = revision4.compatibility_matrix

# The base writer resolves these module globals at call time. Patch them only after
# capturing the base audit function so the frozen manifest uses this strict plan.
core.PLAN_REVISION = PLAN_REVISION
core.SEED_NAMESPACE = SEED_NAMESPACE
core.build_plan = build_plan
core.audit_plan = audit_plan
core.compatibility_matrix = compatibility_matrix

load_plan = core.load_plan
sha256_file = core.sha256_file
write_plan = core.write_plan


__all__ = [
    "CONTEXT_QUOTAS",
    "FORBIDDEN_CLASS_PAIRS",
    "PLAN_REVISION",
    "SEED_NAMESPACE",
    "PlannedSample",
    "audit_plan",
    "build_plan",
    "compatibility_matrix",
    "load_plan",
    "sha256_file",
    "write_plan",
]
