"""Fail-closed semantic construction contracts for every strict P1-C stage."""

from __future__ import annotations

from typing import Any


SLOT_CLASSES = frozenset(("slot_u", "slot_rect", "slot_v", "slot_o"))
REQUIRED_SLOT_CONSTRUCTION_METHOD = "p1_c_continuous_closed_profile_sweep_cut"
REQUIRED_SLOT_GUIDE_FAMILY = "straight_surface_guide"


def audit_generation_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    failures: list[str] = []
    slot_records: list[dict[str, Any]] = []
    for feature in parameters.get("features", []):
        macro = str(feature.get("macro_feature", ""))
        if macro not in SLOT_CLASSES:
            continue
        feature_parameters = feature.get("parameters", {})
        prefix = f"slot::{macro}::instance_{feature.get('instance_id')}"
        slot_checks = {
            f"{prefix}::continuous_sweep_method": (
                feature_parameters.get("construction_method")
                == REQUIRED_SLOT_CONSTRUCTION_METHOD
            ),
            f"{prefix}::guide_on_existing_planar_face": (
                feature_parameters.get("guide_on_existing_planar_face") is True
            ),
            f"{prefix}::straight_surface_guide": (
                feature_parameters.get("guide_family") == REQUIRED_SLOT_GUIDE_FAMILY
            ),
            f"{prefix}::single_analytic_guide_edge": (
                int(feature_parameters.get("guide_edge_count", -1)) == 1
                and feature_parameters.get("guide_polyline") is False
            ),
            f"{prefix}::closed_profile": feature_parameters.get("profile_closed") is True,
            f"{prefix}::path_ratio_ge_4": (
                float(feature_parameters.get("path_length_profile_width_ratio", -1.0)) >= 4.0
            ),
            f"{prefix}::tool_cleanup_idempotent": bool(
                feature_parameters.get("tool_cleanup", {}).get("passed", False)
            ),
        }
        checks.update(slot_checks)
        slot_records.append(
            {
                "macro_feature": macro,
                "instance_id": feature.get("instance_id"),
                "construction_method": feature_parameters.get("construction_method"),
                "guide_family": feature_parameters.get("guide_family"),
                "guide_on_existing_planar_face": feature_parameters.get(
                    "guide_on_existing_planar_face"
                ),
                "path_length_profile_width_ratio": feature_parameters.get(
                    "path_length_profile_width_ratio"
                ),
            }
        )

    expected_macros = {
        str(parameters.get("primary_target", "")),
        *(str(value) for value in parameters.get("secondary_features", [])),
    }
    expected_slot_count = len(expected_macros & SLOT_CLASSES)
    checks["all_expected_slot_instances_have_construction_records"] = (
        len(slot_records) == expected_slot_count
    )

    semantic_relabel_keys = _find_exact_keys(parameters, "semantic_relabel")
    checks["legacy_semantic_relabel_key_absent"] = not semantic_relabel_keys

    wall_macro = next((macro for macro in ("rib", "web") if macro in expected_macros), None)
    wall_evidence: dict[str, Any] | None = None
    if wall_macro is not None:
        source = _supported_wall_source(parameters)
        wall_checks = {
            "supported_wall_source_present": source is not None,
            "supported_wall_source_macro_matches_target": bool(source)
            and source.get("source_engineering_macro", source.get("macro_feature")) == wall_macro,
            "supported_wall_host_concept_compatible": bool(source)
            and source.get("concept_contract", {}).get("host_family_concept_compatible") is True,
        }
        if wall_macro == "web":
            wall_checks.update(
                {
                    "web_full_height_two_support_connection": bool(source)
                    and source.get("full_height_two_support_connection") is True,
                    "web_exposed_feature_end_faces_zero": bool(source)
                    and int(source.get("exposed_feature_end_face_count", -1)) == 0,
                }
            )
        checks.update(wall_checks)
        wall_evidence = {
            "macro_feature": wall_macro,
            "host_family": None if source is None else source.get("host_family"),
            "source_engineering_macro": None
            if source is None
            else source.get("source_engineering_macro", source.get("macro_feature")),
            "full_height_two_support_connection": None
            if source is None
            else source.get("full_height_two_support_connection"),
            "exposed_feature_end_face_count": None
            if source is None
            else source.get("exposed_feature_end_face_count"),
        }

    failures.extend(name for name, passed in checks.items() if not passed)
    return {
        "schema_version": "topocae_p1_c_strict_generation_contract_audit_1",
        "passed": not failures,
        "checks": checks,
        "failures": failures,
        "slot_contract": {
            "required_construction_method": REQUIRED_SLOT_CONSTRUCTION_METHOD,
            "required_guide_family": REQUIRED_SLOT_GUIDE_FAMILY,
            "records": slot_records,
        },
        "supported_wall_contract": wall_evidence,
        "semantic_relabel_key_paths": semantic_relabel_keys,
    }


def assert_generation_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    audit = audit_generation_parameters(parameters)
    if not audit["passed"]:
        raise ValueError(f"strict generation contract failed: {audit['failures']}")
    return audit


def _supported_wall_source(parameters: dict[str, Any]) -> dict[str, Any] | None:
    if isinstance(parameters.get("supported_wall"), dict):
        adapter = parameters["supported_wall"].get("source_parameters")
        if isinstance(adapter, dict):
            raw = adapter.get("source_parameters")
            return raw if isinstance(raw, dict) else adapter
    raw = parameters.get("source_parameters")
    return raw if isinstance(raw, dict) else None


def _find_exact_keys(value: Any, target: str, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key == target:
                found.append(child_path)
            found.extend(_find_exact_keys(child, target, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_find_exact_keys(child, target, f"{path}[{index}]"))
    return found


__all__ = [
    "REQUIRED_SLOT_CONSTRUCTION_METHOD",
    "REQUIRED_SLOT_GUIDE_FAMILY",
    "SLOT_CLASSES",
    "assert_generation_parameters",
    "audit_generation_parameters",
]
