from __future__ import annotations

import re
from typing import Any

from . import renderer as base
from . import formal_renderer as v1
from .io import read_yaml


def _request(plan: dict[str, Any], bank: dict[str, Any], locale: str) -> str:
    verbal = plan["verbalization"]
    if verbal["utterance_kind"] != "unsupported_negation":
        return v1._request(plan, bank, locale)
    target = base._target_phrase(plan, bank, locale)
    operation = verbal["operations"][0]
    size = operation.get("size", 0.5)
    unit = operation.get("unit", "mm")
    if locale == "en-US":
        return (
            f"set the local mesh size of {target}, excluding the one with the smallest area, "
            f"to {base._number(size)} {unit}"
        )
    return f"将{target}（不包括面积最小者）的局部网格尺寸设为 {base._number(size)} {unit}"


def render_formal_query(plan: dict[str, Any], variant: str) -> dict[str, Any]:
    en, zh = base.load_phrase_banks()
    extension = read_yaml(v1.EXTENSION_PATH)
    dataset_split = {"train": "train", "val": "val", "development_test": "test"}[
        plan["source_record"]["split"]
    ]
    if variant in {"canonical", "en-1", "en-2"}:
        locale, bank = "en-US", en
    elif variant == "zh-1":
        locale, bank = "zh-CN", zh
    else:
        raise ValueError(variant)
    template = extension["templates"][dataset_split][variant][0]
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


audit_formal_template_manifest = v1.audit_formal_template_manifest
normalize_exact_text = v1.normalize_exact_text

