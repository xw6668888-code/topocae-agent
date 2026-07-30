from __future__ import annotations

import re
from typing import Any

from .constants import ROOT
from .io import read_json, read_yaml
from . import renderer as base


EXTENSION_PATH = ROOT / "configs" / "paper_v1" / "cae_query_reviewed_phrase_extensions.yaml"
FORMAL_TEMPLATE_MANIFEST = ROOT / "configs" / "paper_v1" / "cae_query_template_split_manifest.formal.json"


def _formal_operation_request(
    operation: dict[str, Any], target: str, locale: str, *, omit: str | None = None
) -> str:
    kind = operation["operation_type"]
    parameters = operation.get("parameters", {})
    if locale == "en-US":
        if kind == "fixed_support":
            return f"fully constrain U1, U2, and U3 on {target}"
        if kind == "displacement":
            component = parameters["components"][0]
            value = component["value"]
            return (
                f"apply a {base._number(value['value'])} {value['unit']} {component['dof']} displacement "
                f"in the global Cartesian frame to {target}"
            )
        if kind == "pressure":
            magnitude = parameters.get("magnitude", {"value": 1.0, "unit": "MPa"})
            if omit == "direction":
                return f"apply a {base._number(magnitude['value'])} {magnitude['unit']} pressure to {target}"
            return (
                f"apply a {base._number(magnitude['value'])} {magnitude['unit']} pressure "
                f"{base._direction_en(parameters['direction'])} to {target}"
            )
        if kind == "surface_traction":
            magnitude = parameters["magnitude"]
            return (
                f"apply a {base._number(magnitude['value'])} {magnitude['unit']} surface traction "
                f"{base._direction_en(parameters['direction'])} to {target}"
            )
        if kind == "local_mesh_size":
            if omit == "mesh_size":
                return f"locally refine {target}"
            size = parameters["size"]
            return f"locally refine {target} to a mesh size of {base._number(size['value'])} {size['unit']}"
        if kind == "material_assignment":
            elastic = parameters["elastic"]
            modulus = elastic["youngs_modulus"]
            return (
                f"use material {parameters['material_name']} (E={base._number(modulus['value'])} {modulus['unit']}, "
                f"Poisson ratio {base._number(elastic['poisson_ratio'])}) for the whole part"
            )
        if kind == "global_mesh_size":
            size = parameters["size"]
            return f"use a whole-part global mesh size of {base._number(size['value'])} {size['unit']}"
    else:
        if kind == "fixed_support":
            return f"完全固定{target}（约束 U1、U2、U3）"
        if kind == "displacement":
            component = parameters["components"][0]
            value = component["value"]
            return f"在全局笛卡尔坐标系中对{target}施加 {component['dof']}={base._number(value['value'])} {value['unit']} 的位移"
        if kind == "pressure":
            magnitude = parameters.get("magnitude", {"value": 1.0, "unit": "MPa"})
            if omit == "direction":
                return f"对{target}施加 {base._number(magnitude['value'])} {magnitude['unit']} 压力"
            return (
                f"对{target}施加 {base._number(magnitude['value'])} {magnitude['unit']}、"
                f"{base._direction_zh(parameters['direction'])}的压力"
            )
        if kind == "surface_traction":
            magnitude = parameters["magnitude"]
            return (
                f"对{target}施加 {base._number(magnitude['value'])} {magnitude['unit']}、"
                f"{base._direction_zh(parameters['direction'])}的表面牵引"
            )
        if kind == "local_mesh_size":
            if omit == "mesh_size":
                return f"对{target}进行局部网格加密"
            size = parameters["size"]
            return f"将{target}局部加密至 {base._number(size['value'])} {size['unit']} 网格"
        if kind == "material_assignment":
            elastic = parameters["elastic"]
            modulus = elastic["youngs_modulus"]
            return (
                f"为整个零件赋予材料 {parameters['material_name']}（弹性模量 {base._number(modulus['value'])} {modulus['unit']}，"
                f"泊松比 {base._number(elastic['poisson_ratio'])}）"
            )
        if kind == "global_mesh_size":
            size = parameters["size"]
            return f"将整个零件的全局网格尺寸设为 {base._number(size['value'])} {size['unit']}"
    raise ValueError(f"Unsupported formal rendering operation: {kind}")


def _request(plan: dict[str, Any], bank: dict[str, Any], locale: str) -> str:
    verbal = plan["verbalization"]
    kind = verbal["utterance_kind"]
    target = base._target_phrase(plan, bank, locale)
    operations = verbal["operations"]
    if kind == "multi_operation":
        clauses = [
            _formal_operation_request(
                operation,
                target if operation["operation_type"] not in {"material_assignment", "global_mesh_size"} else "",
                locale,
            ).strip()
            for operation in operations
        ]
        if locale == "en-US":
            return ", ".join(clauses[:-1]) + ", and " + clauses[-1]
        return "，".join(clauses[:-1]) + "，并" + clauses[-1]
    if kind == "missing_direction":
        operation = {
            "operation_type": "pressure",
            "parameters": {
                "magnitude": {
                    "value": operations[0].get("magnitude", 1.0),
                    "unit": operations[0].get("unit", "MPa"),
                }
            },
        }
        return _formal_operation_request(operation, target, locale, omit="direction")
    if kind == "missing_mesh_size":
        return _formal_operation_request(
            {"operation_type": "local_mesh_size", "parameters": {}},
            target,
            locale,
            omit="mesh_size",
        )
    if kind == "missing_reference":
        return _formal_operation_request(
            {
                "operation_type": "local_mesh_size",
                "parameters": {"size": {"value": 0.5, "unit": "mm"}},
            },
            target,
            locale,
        )
    if kind == "missing_attribute":
        return (
            f"locally refine {target} selected by an unspecified geometric attribute"
            if locale == "en-US"
            else f"按尚未说明的几何属性筛选并局部加密{target}"
        )
    if kind == "unsupported_negation":
        negative = (
            f"{target}, except the one with the smallest area"
            if locale == "en-US"
            else f"{target}，但排除面积最小的一个"
        )
        size = operations[0].get("size", 0.5)
        return _formal_operation_request(
            {
                "operation_type": "local_mesh_size",
                "parameters": {"size": {"value": size, "unit": operations[0].get("unit", "mm")}},
            },
            negative,
            locale,
        )
    if kind == "executable":
        return _formal_operation_request(operations[0], target, locale)
    raise ValueError(f"Unknown formal utterance kind: {kind}")


def render_formal_query(plan: dict[str, Any], variant: str) -> dict[str, Any]:
    en, zh = base.load_phrase_banks()
    extension = read_yaml(EXTENSION_PATH)
    dataset_split = {"train": "train", "val": "val", "development_test": "test"}[
        plan["source_record"]["split"]
    ]
    if variant in {"canonical", "en-1", "en-2"}:
        locale, bank = "en-US", en
    elif variant == "zh-1":
        locale, bank = "zh-CN", zh
    else:
        raise ValueError(variant)
    templates = extension["templates"][dataset_split][variant]
    template = templates[0]
    text = template["pattern"].format(request=_request(plan, bank, locale)).strip()
    text = re.sub(r"\s+([.,。])", r"\1", text)
    return {
        "text": text,
        "language": locale,
        "template_family_id": template["family_id"],
        "template_id": template["template_id"],
        "rendering_id": f"RF.{plan['canonical_semantics_sha256'][:20]}.{variant}",
        "rendering_variant": variant,
        "difficulty_level": plan["level"],
    }


def render_formal_all(plan: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    canonical = render_formal_query(plan, "canonical")
    return canonical, [
        render_formal_query(plan, variant) for variant in ("en-1", "en-2", "zh-1")
    ]


def audit_formal_template_manifest() -> dict[str, Any]:
    manifest = read_json(FORMAL_TEMPLATE_MANIFEST)
    families = {
        split: set(entry["families"]) for split, entry in manifest["splits"].items()
    }
    collisions: dict[str, list[str]] = {}
    splits = sorted(families)
    for index, left in enumerate(splits):
        for right in splits[index + 1 :]:
            overlap = sorted(families[left] & families[right])
            if overlap:
                collisions[f"{left}:{right}"] = overlap
    return {
        "manifest": str(FORMAL_TEMPLATE_MANIFEST.resolve()),
        "family_counts": {split: len(values) for split, values in sorted(families.items())},
        "cross_split_collisions": collisions,
        "automatic_pass": not collisions,
    }


normalize_exact_text = base.normalize_exact_text

