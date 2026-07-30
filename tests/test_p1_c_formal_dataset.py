from __future__ import annotations

from collections import Counter

from p1_c.formal_generator_v4 import build_planned_result, transformed_shape_and_audit
from p1_c.formal_plan_v4 import FORBIDDEN_CLASS_PAIRS, audit_plan, build_plan


def test_formal_plan_has_exact_clean_context_quotas() -> None:
    plan = build_plan()
    audit = audit_plan(plan)
    assert audit["passed"]
    assert audit["sample_count"] == 7200
    assert audit["split_counts"] == {"train": 5760, "val": 720, "development_test": 720}
    assert set(audit["secondary_counts"].values()) == {200}
    assert audit["directed_pair_max"] <= 16
    assert all(max(counts.values()) <= 140 for counts in audit["host_family_counts_by_primary"].values())


def test_formal_plan_has_no_forbidden_pair_in_pair_or_triple() -> None:
    for sample in build_plan():
        macros = (sample.primary_target, *sample.secondary_features)
        for left_index, left in enumerate(macros):
            for right in macros[left_index + 1 :]:
                assert frozenset((left, right)) not in FORBIDDEN_CLASS_PAIRS


def test_formal_generator_supports_general_and_supported_wall_contexts() -> None:
    plan = build_plan()
    selected = [
        next(sample for sample in plan if sample.primary_target == "slot_u" and sample.context == "triple" and "rib" not in sample.secondary_features and "web" not in sample.secondary_features),
        next(sample for sample in plan if sample.primary_target == "through_hole_round" and "rib" in sample.secondary_features),
        next(sample for sample in plan if sample.primary_target == "rib" and sample.context == "triple"),
        next(sample for sample in plan if sample.primary_target == "web" and sample.context == "target_only" and sample.host_family == "circular_cavity"),
    ]
    for sample in selected:
        result = build_planned_result(sample)
        expected_macros = (sample.primary_target, *sample.secondary_features)
        assert result.cleanup["passed"]
        assert {tag.instance_id for tag in result.face_tags} == set(range(1, len(expected_macros) + 1))
        assert Counter(tag.macro_feature for tag in result.face_tags).keys() == set(expected_macros)


def test_formal_rotation_scale_preserves_clean_topology_and_face_order() -> None:
    sample = next(
        item
        for item in build_plan()
        if item.primary_target == "slot_o" and item.context == "triple"
    )
    result = build_planned_result(sample)
    _shape, audit = transformed_shape_and_audit(
        result,
        scale=sample.uniform_scale,
        rotation_axis=sample.rotation_axis,
        rotation_angle_radians=sample.rotation_angle_radians,
    )
    assert audit["passed"]
    assert audit["checks"]["face_count_preserved"]
    assert audit["checks"]["face_order_area_correspondence"]
    assert audit["cleanup_audit"]["removable_same_domain_face_count"] == 0
    assert audit["cleanup_audit"]["removable_same_domain_edge_count"] == 0
