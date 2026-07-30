from __future__ import annotations

from typing import Any


def normalize_geometry_parameters(macro_feature: str, raw_parameters: dict[str, Any]) -> dict[str, Any]:
    radius = _optional_float(raw_parameters.get("R", raw_parameters.get("radius")))
    depth = _optional_float(raw_parameters.get("Depth", raw_parameters.get("depth")))
    width = _optional_float(raw_parameters.get("width", raw_parameters.get("w")))
    length = _optional_float(raw_parameters.get("length", raw_parameters.get("l")))
    result: dict[str, Any] = {}

    if radius is not None:
        result["radius"] = radius
        result["diameter"] = 2.0 * radius
    if depth is not None:
        result["depth"] = depth
    if width is not None:
        result["width"] = width
    if length is not None:
        result["length"] = length

    if radius is not None and depth is not None and radius > 0.0:
        ratio = depth / (2.0 * radius)
        result["depth_to_diameter_ratio"] = ratio
        if macro_feature in {"blind_hole_round", "pocket_circular"}:
            result["cylindrical_recess_family"] = _classify_cylindrical_recess(ratio)

    if macro_feature in {"slot_rect", "pocket_rect"} and width is not None and length is not None:
        short_side = min(width, length)
        long_side = max(width, length)
        if short_side > 0.0:
            aspect_ratio = long_side / short_side
            result["planform_aspect_ratio"] = aspect_ratio
            result["rectangular_recess_family"] = _classify_rectangular_recess(aspect_ratio)

    for key in (
        "cx",
        "cy",
        "cz",
        "angle",
        "dist",
        "height",
        "thickness",
        "bottom_coord",
        "sweep_path_length",
        "profile_width",
        "profile_depth",
        "top_margin",
    ):
        if key in raw_parameters:
            value = _optional_float(raw_parameters[key])
            if value is not None:
                result[key] = value

    if "axis" in raw_parameters:
        axis = raw_parameters["axis"]
        if isinstance(axis, str):
            result["axis"] = axis
        elif isinstance(axis, (list, tuple)):
            result["axis_vector"] = [
                float(value)
                for value in axis
                if _optional_float(value) is not None
            ]
    if "construction_method" in raw_parameters:
        result["construction_method"] = str(raw_parameters["construction_method"])
    for key in ("profile_shape", "profile_orientation", "path_type", "path_curve_kind", "anchor_face_kind", "path_boundary_kind"):
        if key in raw_parameters:
            result[key] = str(raw_parameters[key])
    for key in ("path_on_face", "path_start_on_boundary", "path_end_on_boundary"):
        if key in raw_parameters:
            result[key] = bool(raw_parameters[key])

    return result


def build_feature_parameter_lookup(feature_registry: list[tuple[Any, list[dict]]], ontology_maps: Any) -> dict[int, dict]:
    lookup: dict[int, dict] = {}
    for generator, params_list in feature_registry:
        for raw_params in params_list:
            raw_macro_key = resolve_raw_macro_key(generator.feature_name, raw_params)
            macro_resolution = ontology_maps.resolve_macro_feature(raw_macro_key)
            macro_feature = macro_resolution.key
            inst_id = int(raw_params["inst"])
            lookup[inst_id] = {
                "source_generator": generator.__class__.__name__,
                "raw_macro_feature": macro_resolution.raw_key,
                "macro_feature": macro_feature,
                "raw_parameters": _json_safe_dict(raw_params),
                "geometry_parameters": normalize_geometry_parameters(macro_feature, raw_params),
            }
    return lookup


def resolve_raw_macro_key(generator_feature_name: str, raw_parameters: dict[str, Any]) -> str:
    if generator_feature_name == "boss_feature":
        boss_type = raw_parameters.get("type")
        if boss_type in {"cylinder", "cone"}:
            return f"{boss_type}_boss"
    if generator_feature_name == "swept_slot":
        macro_key = raw_parameters.get("macro_key")
        if isinstance(macro_key, str):
            return macro_key
    return generator_feature_name


def attach_feature_parameters(graph_data: dict, params_by_instance: dict[int, dict]) -> None:
    for feature_set in graph_data.get("feature_sets", []):
        inst_id = int(feature_set.get("instance_id", 0))
        params = params_by_instance.get(inst_id)
        if not params:
            continue
        feature_set["source_generator"] = params.get("source_generator")
        feature_set["raw_parameters"] = params.get("raw_parameters", {})
        feature_set["geometry_parameters"] = params.get("geometry_parameters", {})


def _classify_cylindrical_recess(depth_to_diameter_ratio: float) -> str:
    if depth_to_diameter_ratio >= 0.8:
        return "deep"
    if depth_to_diameter_ratio <= 0.5:
        return "shallow"
    return "ambiguous"


def _classify_rectangular_recess(planform_aspect_ratio: float) -> str:
    if planform_aspect_ratio >= 4.0:
        return "elongated_slot"
    if planform_aspect_ratio <= 1.8:
        return "compact_pocket"
    return "ambiguous"


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _json_safe_dict(raw_parameters: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in raw_parameters.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
        elif isinstance(value, (list, tuple)):
            safe[key] = list(value)
        else:
            safe[key] = str(value)
    return safe
