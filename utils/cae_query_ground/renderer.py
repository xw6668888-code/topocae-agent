from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

from .constants import ROOT
from .io import read_json, read_yaml


EN_BANK_PATH = ROOT / "configs" / "paper_v1" / "cae_query_phrase_bank.en-US.yaml"
ZH_BANK_PATH = ROOT / "configs" / "paper_v1" / "cae_query_phrase_bank.zh-CN.yaml"
TEMPLATE_SPLIT_MANIFEST = ROOT / "configs" / "paper_v1" / "cae_query_template_split_manifest.json"


def _number(value: Any) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.6f}".rstrip("0").rstrip(".")


def _feature_role(plan: dict[str, Any], bank: dict[str, Any], locale: str) -> tuple[Any, Any]:
    verbal = plan["verbalization"]
    feature = bank["feature_labels"][verbal["feature_type"]]
    role = bank["face_role_labels"][verbal["face_role"]]
    if locale == "en-US":
        return feature, role
    return str(feature), str(role)


def _filter_item(filters: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    for key in ("diameter", "radius", "depth", "area", "length", "width", "thickness"):
        if key in filters:
            return key, filters[key]
    return None


def _target_phrase(plan: dict[str, Any], bank: dict[str, Any], locale: str) -> str:
    verbal = plan["verbalization"]
    feature, role = _feature_role(plan, bank, locale)
    quantifier = verbal.get("quantifier") or "all"
    filters = verbal.get("filters") or {}
    filter_item = _filter_item(filters)
    comparison = verbal.get("comparison_metric")

    if locale == "en-US":
        singular = feature["singular"]
        plural = feature["plural"]
        role_singular = role["singular"]
        role_plural = role["plural"]
        if filter_item:
            metric, condition = filter_item
            quantity = condition["value"]
            metric_label = bank["metric_labels"][metric]
            tolerance = quantity.get("tolerance")
            tolerance_phrase = (
                f" within ±{_number(tolerance)} {quantity['unit']}" if tolerance is not None else ""
            )
            return (
                f"the {role_singular} of the {singular} whose {metric_label} is "
                f"{_number(quantity['value'])} {quantity['unit']}{tolerance_phrase}"
            )
        if quantifier == "exactly_one":
            return f"the {role_singular} of the only {singular}"
        if quantifier == "all":
            return f"the {role_plural} of all {plural}"
        if quantifier == "count":
            return f"the {role_plural} of the {verbal['count']} matching {plural}"
        if quantifier in {"largest", "smallest"}:
            metric_label = bank["metric_labels"][comparison]
            return f"the {role_singular} of the {singular} with the {quantifier} {metric_label}"
        if quantifier == "nearest":
            if filters.get("nearest_to", {}).get("reference_type") == "bounding_box_min":
                return f"the {role_singular} of the {singular} nearest to the part bounding-box minimum corner"
            return f"the {role_singular} of the nearest {singular}"
        raise ValueError(f"Unsupported quantifier: {quantifier}")

    metric_labels = bank["metric_labels"]
    if filter_item:
        metric, condition = filter_item
        quantity = condition["value"]
        tolerance = quantity.get("tolerance")
        tolerance_phrase = f"（容差 ±{_number(tolerance)} {quantity['unit']}）" if tolerance is not None else ""
        return f"{metric_labels[metric]}为 {_number(quantity['value'])} {quantity['unit']}{tolerance_phrase}的{feature}{role}"
    if quantifier == "exactly_one":
        return f"唯一一个{feature}的{role}"
    if quantifier == "all":
        return f"全部{feature}的{role}"
    if quantifier == "count":
        return f"恰好 {verbal['count']} 个匹配{feature}的{role}"
    if quantifier == "largest":
        return f"{metric_labels[comparison]}最大的{feature}{role}"
    if quantifier == "smallest":
        return f"{metric_labels[comparison]}最小的{feature}{role}"
    if quantifier == "nearest":
        if filters.get("nearest_to", {}).get("reference_type") == "bounding_box_min":
            return f"全局坐标系中距离零件包围盒最小角点最近的{feature}{role}"
        return f"距离指定参考最近的{feature}{role}"
    raise ValueError(f"Unsupported quantifier: {quantifier}")


def _direction_en(direction: dict[str, Any]) -> str:
    kind = direction["kind"]
    if kind == "surface_normal":
        return "along the surface normal"
    if kind == "opposite_surface_normal":
        return "opposite to the surface normal"
    vector = direction["vector"]
    labels = {(1.0, 0.0, 0.0): "global +X", (0.0, -1.0, 0.0): "global -Y", (0.0, 0.0, 1.0): "global +Z"}
    return f"in the {labels.get(tuple(float(value) for value in vector), str(vector))} direction"


def _direction_zh(direction: dict[str, Any]) -> str:
    kind = direction["kind"]
    if kind == "surface_normal":
        return "沿表面法向"
    if kind == "opposite_surface_normal":
        return "沿表面法向反向"
    vector = tuple(float(value) for value in direction["vector"])
    labels = {(1.0, 0.0, 0.0): "沿全局 +X 方向", (0.0, -1.0, 0.0): "沿全局 -Y 方向", (0.0, 0.0, 1.0): "沿全局 +Z 方向"}
    return labels.get(vector, f"沿全局向量 {list(vector)}")


def _operation_request(operation: dict[str, Any], target: str, locale: str, *, omit: str | None = None) -> str:
    operation_type = operation["operation_type"]
    parameters = operation.get("parameters", {})
    if locale == "en-US":
        if operation_type == "fixed_support":
            return f"apply a fixed support constraining U1, U2, and U3 to {target}"
        if operation_type == "displacement":
            component = parameters["components"][0]
            quantity = component["value"]
            return (
                f"prescribe a {_number(quantity['value'])} {quantity['unit']} {component['dof']} displacement "
                f"in the global Cartesian frame on {target}"
            )
        if operation_type == "pressure":
            magnitude = parameters.get("magnitude", {"value": 1.0, "unit": "MPa"})
            direction = "" if omit == "direction" else f" {_direction_en(parameters['direction'])}"
            return f"apply a {_number(magnitude['value'])} {magnitude['unit']} pressure{direction} to {target}"
        if operation_type == "surface_traction":
            magnitude = parameters["magnitude"]
            return (
                f"apply a {_number(magnitude['value'])} {magnitude['unit']} surface traction "
                f"{_direction_en(parameters['direction'])} to {target}"
            )
        if operation_type == "local_mesh_size":
            if omit == "mesh_size":
                return f"locally refine {target}"
            size = parameters.get("size", {"value": 0.5, "unit": "mm"})
            return f"set a local mesh size of {_number(size['value'])} {size['unit']} on {target}"
        if operation_type == "material_assignment":
            elastic = parameters["elastic"]
            modulus = elastic["youngs_modulus"]
            return (
                f"assign material {parameters['material_name']} with E={_number(modulus['value'])} {modulus['unit']} "
                f"and Poisson ratio {_number(elastic['poisson_ratio'])} to the whole part"
            )
        if operation_type == "global_mesh_size":
            size = parameters["size"]
            return f"set the whole-part global mesh size to {_number(size['value'])} {size['unit']}"
    else:
        if operation_type == "fixed_support":
            return f"完全固定{target}（约束 U1、U2、U3）"
        if operation_type == "displacement":
            component = parameters["components"][0]
            quantity = component["value"]
            return f"在全局笛卡尔坐标系中对{target}施加 {component['dof']}={_number(quantity['value'])} {quantity['unit']} 的位移"
        if operation_type == "pressure":
            magnitude = parameters.get("magnitude", {"value": 1.0, "unit": "MPa"})
            direction = "" if omit == "direction" else _direction_zh(parameters["direction"])
            return f"对{target}施加 {_number(magnitude['value'])} {magnitude['unit']} {direction}压力".replace("  ", " ")
        if operation_type == "surface_traction":
            magnitude = parameters["magnitude"]
            return f"对{target}施加 {_number(magnitude['value'])} {magnitude['unit']}、{_direction_zh(parameters['direction'])}的表面牵引"
        if operation_type == "local_mesh_size":
            if omit == "mesh_size":
                return f"对{target}进行局部网格加密"
            size = parameters.get("size", {"value": 0.5, "unit": "mm"})
            return f"将{target}的局部网格尺寸设为 {_number(size['value'])} {size['unit']}"
        if operation_type == "material_assignment":
            elastic = parameters["elastic"]
            modulus = elastic["youngs_modulus"]
            return (
                f"为整个零件赋予材料 {parameters['material_name']}（弹性模量 {_number(modulus['value'])} {modulus['unit']}，"
                f"泊松比 {_number(elastic['poisson_ratio'])}）"
            )
        if operation_type == "global_mesh_size":
            size = parameters["size"]
            return f"将整个零件的全局网格尺寸设为 {_number(size['value'])} {size['unit']}"
    raise ValueError(f"Unsupported operation for renderer: {operation_type}")


def _request(plan: dict[str, Any], bank: dict[str, Any], locale: str) -> str:
    verbal = plan["verbalization"]
    kind = verbal["utterance_kind"]
    target = _target_phrase(plan, bank, locale)
    operations = verbal["operations"]

    if kind == "multi_operation":
        requests = []
        for operation in operations:
            operation_target = target if operation["operation_type"] not in {"material_assignment", "global_mesh_size"} else ""
            requests.append(_operation_request(operation, operation_target, locale).strip())
        conjunction = ", and " if locale == "en-US" else "，并"
        if locale == "en-US":
            return ", ".join(requests[:-1]) + conjunction + requests[-1]
        return "，".join(requests[:-1]) + conjunction + requests[-1]

    if kind == "missing_direction":
        pseudo = {
            "operation_type": "pressure",
            "parameters": {"magnitude": {"value": operations[0].get("magnitude", 1.0), "unit": operations[0].get("unit", "MPa")}},
        }
        return _operation_request(pseudo, target, locale, omit="direction")
    if kind == "missing_mesh_size":
        return _operation_request({"operation_type": "local_mesh_size", "parameters": {}}, target, locale, omit="mesh_size")
    if kind == "missing_reference":
        return _operation_request(
            {"operation_type": "local_mesh_size", "parameters": {"size": {"value": 0.5, "unit": "mm"}}},
            target,
            locale,
        )
    if kind == "missing_attribute":
        if locale == "en-US":
            return f"locally refine {target} selected by an unspecified geometric attribute"
        return f"按尚未说明的几何属性筛选并局部加密{target}"
    if kind == "unsupported_negation":
        if locale == "en-US":
            negative_target = f"{target}, except the one with the smallest area"
        else:
            negative_target = f"{target}，但排除面积最小的一个"
        return _operation_request(
            {"operation_type": "local_mesh_size", "parameters": {"size": {"value": 0.5, "unit": "mm"}}},
            negative_target,
            locale,
        )
    if kind == "executable":
        return _operation_request(operations[0], target, locale)
    raise ValueError(f"Unknown utterance kind: {kind}")


def load_phrase_banks() -> tuple[dict[str, Any], dict[str, Any]]:
    en = read_yaml(EN_BANK_PATH)
    zh = read_yaml(ZH_BANK_PATH)
    if en.get("language") != "en-US" or zh.get("language") != "zh-CN":
        raise ValueError("Independent language banks are malformed")
    return en, zh


def render_query(plan: dict[str, Any], variant: str) -> dict[str, Any]:
    en, zh = load_phrase_banks()
    source_split = plan["source_record"]["split"]
    dataset_split = {"train": "train", "val": "val", "development_test": "test"}[source_split]
    if variant in {"canonical", "en-1", "en-2"}:
        locale = "en-US"
        bank = en
    elif variant == "zh-1":
        locale = "zh-CN"
        bank = zh
    else:
        raise ValueError(f"Unknown rendering variant: {variant}")
    template = bank["split_templates"][dataset_split][variant]
    request = _request(plan, bank, locale)
    text = template["pattern"].format(request=request).strip()
    text = re.sub(r"\s+([.,。])", r"\1", text)
    rendering_id = f"R.{plan['canonical_semantics_sha256'][:20]}.{variant}"
    return {
        "text": text,
        "language": locale,
        "template_family_id": template["family_id"],
        "template_id": template["template_id"],
        "rendering_id": rendering_id,
        "rendering_variant": variant,
        "difficulty_level": plan["level"],
    }


def render_all(plan: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    canonical = render_query(plan, "canonical")
    renderings = [render_query(plan, variant) for variant in ("en-1", "en-2", "zh-1")]
    return canonical, renderings


def normalize_exact_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def audit_template_manifest() -> dict[str, Any]:
    manifest = read_json(TEMPLATE_SPLIT_MANIFEST)
    by_split = {split: set(entry["families"]) for split, entry in manifest["splits"].items()}
    collisions = {}
    splits = sorted(by_split)
    for index, left in enumerate(splits):
        for right in splits[index + 1:]:
            overlap = sorted(by_split[left] & by_split[right])
            if overlap:
                collisions[f"{left}:{right}"] = overlap
    return {
        "manifest": str(TEMPLATE_SPLIT_MANIFEST),
        "family_counts": {split: len(values) for split, values in sorted(by_split.items())},
        "cross_split_collisions": collisions,
        "automatic_pass": not collisions,
    }
