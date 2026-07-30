"""Build the P2-A1 ontology/capability audit without touching geometry data."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


SCHEMA_VERSION = "topocae_p2_a1_ontology_audit_1"
CANONICAL_FEATURES = (
    "through_hole_round", "blind_hole_round", "stepped_hole_round",
    "counterbore_hole", "countersink_hole", "obround_hole",
    "slot_rect", "slot_u", "slot_v", "slot_o",
    "pocket_rect", "pocket_circular", "cylinder_boss",
    "rectangular_boss", "rib", "fillet", "chamfer",
)
FAMILIES = {
    "through_hole_round": "hole", "blind_hole_round": "hole",
    "stepped_hole_round": "hole", "counterbore_hole": "hole",
    "countersink_hole": "hole", "obround_hole": "hole",
    "slot_rect": "slot", "slot_u": "slot", "slot_v": "slot", "slot_o": "slot",
    "pocket_rect": "pocket", "pocket_circular": "pocket",
    "cylinder_boss": "protrusion", "rectangular_boss": "protrusion",
    "rib": "stiffener", "fillet": "transition", "chamfer": "transition",
}
EXPLICIT_ALIASES = {
    "through_hole_round": ("through_hole",),
    "blind_hole_round": ("blind_hole",),
    "stepped_hole_round": ("stepped_hole",),
    "rectangular_boss": ("pad", "rectangle_boss"),
    "rib": ("web",),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping in {path}")
    return value


def _master_plan_features(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"### 2\.2 当前目标特征体系(?P<body>.*?)(?:\n### 2\.3)", text, re.S)
    if not match:
        raise ValueError("Could not locate master-plan section 2.2")
    return re.findall(r"`([a-z][a-z0-9_]*)`", match.group("body"))


def _canonicalize(name: str) -> str:
    return "rib" if name == "web" else name


def build_audit(root: Path) -> dict[str, Any]:
    master_path = root / "TOPOCAE_AGENT_RESEARCH_TO_PAPER_MASTER_PLAN.zh-CN.md"
    macro_path = root / "CAE_feature_ontology.yaml"
    semantic_path = root / "face_semantic_subtype_ontology.yaml"
    capability_path = root / "reports" / "p1_d_lite_v1" / "freeze" / "capability_contract.json"

    macro_doc = _read_yaml(macro_path)
    semantic_doc = _read_yaml(semantic_path)
    capability = json.loads(capability_path.read_text(encoding="utf-8"))
    macro_labels = macro_doc["layers"]["macro_feature"]["labels"]
    semantic_labels = semantic_doc["layers"]["face_semantic_subtype"]["labels"]

    master_raw = _master_plan_features(master_path)
    master_canonical = sorted({_canonicalize(name) for name in master_raw})
    enabled_macros = [label["key"] for label in macro_labels if bool(label.get("enabled"))]
    enabled_macro_canonical = sorted({_canonicalize(name) for name in enabled_macros})
    capability_features = list(capability["target_classes"])

    subtype_by_parent: dict[str, list[str]] = defaultdict(list)
    for label in semantic_labels:
        if bool(label.get("enabled")):
            subtype_by_parent[_canonicalize(str(label["parent_macro_feature"]))].append(
                str(label["key"])
            )

    macro_aliases: dict[str, set[str]] = defaultdict(set)
    for label in macro_labels:
        canonical = _canonicalize(str(label["key"]))
        if canonical not in CANONICAL_FEATURES:
            continue
        macro_aliases[canonical].update(
            str(alias) for alias in label.get("current_code_aliases", []) if alias != canonical
        )
        if label["key"] == "web":
            macro_aliases["rib"].add("web")
    for canonical, aliases in EXPLICIT_ALIASES.items():
        macro_aliases[canonical].update(aliases)

    target_set = set(CANONICAL_FEATURES)
    capability_set = set(capability_features)
    master_set = set(master_canonical)
    ontology_set = set(enabled_macro_canonical)
    categories = []
    for canonical in CANONICAL_FEATURES:
        categories.append({
            "canonical_name": canonical,
            "family": FAMILIES[canonical],
            "aliases": sorted(macro_aliases[canonical]),
            "present_in_master_plan": canonical in master_set,
            "present_in_enabled_macro_ontology": canonical in ontology_set,
            "present_in_capability_contract": canonical in capability_set,
            "semantic_subtypes": sorted(subtype_by_parent.get(canonical, [])),
            "semantic_subtype_status": "available" if subtype_by_parent.get(canonical) else "policy_gap",
        })

    policy_gaps = [
        {"code": "G_CAPABILITY_CLASS_MISSING", "feature": "pocket_circular",
         "detail": "P2 target and ontology class exist, but the frozen capability contract omits it."},
        {"code": "G_ALIAS_ONLY_CLASS", "feature": "rib",
         "detail": "The master plan and legacy ontology list web separately; P2 canonicalizes web to the rib alias."},
        {"code": "G_SLOT_CLOSED_END_NOT_DISTINCT", "feature": "slot_rect",
         "detail": "The ontology has side and bottom subtypes but no distinct closed-end subtype."},
        {"code": "G_SLOT_CLOSED_END_NOT_DISTINCT", "feature": "slot_v",
         "detail": "The ontology has side and bottom subtypes but no distinct closed-end subtype."},
        {"code": "G_RIB_TOP_END_NOT_DISTINCT", "feature": "rib",
         "detail": "rib/web top-plane subtypes do not distinguish a top face from an end face."},
        {"code": "G_COMPOSITE_REGION_REQUIRED", "feature": "multiple",
         "detail": "Whole feature regions and complete cavity/slot inner walls require a union of existing subtypes."},
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "P2-A1 CAE-IR ontology and frozen Recognizer-Dev contract",
        "canonical_target_features": list(CANONICAL_FEATURES),
        "counts": {
            "p2_canonical_target": len(CANONICAL_FEATURES),
            "master_plan_raw": len(master_raw),
            "master_plan_after_web_alias": len(master_canonical),
            "enabled_macro_ontology_raw_including_stock_and_extras": len(enabled_macros),
            "enabled_macro_ontology_canonical_target_intersection": len(target_set & ontology_set),
            "capability_contract": len(capability_features),
            "semantic_parent_coverage": sum(1 for feature in CANONICAL_FEATURES if subtype_by_parent.get(feature)),
        },
        "set_differences": {
            "master_plan_extra_after_canonicalization": sorted(master_set - target_set),
            "master_plan_missing": sorted(target_set - master_set),
            "enabled_macro_ontology_outside_p2": sorted(set(enabled_macros) - target_set - {"web"}),
            "enabled_macro_ontology_missing": sorted(target_set - ontology_set),
            "capability_contract_missing": sorted(target_set - capability_set),
            "capability_contract_extra": sorted(capability_set - target_set),
        },
        "alias_policy": {"web": {"canonical": "rib", "mode": "compatibility_alias_only",
                                    "independent_paper_output": False}},
        "categories": categories,
        "policy_gaps": policy_gaps,
        "prototype_audit": {
            "root": r"W:\python\FEA\test",
            "reusable_concepts": [
                "operations list", "material name and elastic properties",
                "global and local mesh size", "boundary-condition translational degrees of freedom",
                "pressure magnitude", "feature/role concepts such as hole wall and boss top",
            ],
            "rejected_legacy_patterns": [
                "raw Abaqus face indices in semantic intent", "unitless physical magnitudes",
                "silent Steel/Aluminum material defaults", "keyword/prompt routing",
                "embedded credential or remote model invocation",
                "hard-coded model, part, instance, and job names",
            ],
            "reuse_policy": "Reuse concepts only; do not copy the keyword parser, prompt, credentials, or face-ID action format.",
        },
        "source_sha256": {
            str(path.relative_to(root)).replace("\\", "/"): _sha256(path)
            for path in (master_path, macro_path, semantic_path, capability_path)
        },
    }


def build_alias_document(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "topocae_cae_feature_aliases_1",
        "canonical_feature_count": len(CANONICAL_FEATURES),
        "canonical_features": [
            {"canonical_name": row["canonical_name"], "family": row["family"], "aliases": row["aliases"]}
            for row in audit["categories"]
        ],
        "compatibility_rules": {
            "web": {"canonical": "rib", "accept_as_input_alias": True,
                    "emit_as_canonical_output": False,
                    "reason": "P2-A1 paper taxonomy has 17 targets and treats web as a legacy rib synonym."}
        },
    }


def render_markdown(audit: dict[str, Any]) -> str:
    counts = audit["counts"]
    lines = [
        "# P2-A1 Ontology 与冻结接口审计", "",
        f"Schema：`{audit['schema_version']}`", "", "## 1. 数量审计", "",
        "| 来源 | 原始/规范化数量 | 与 P2 目标的关系 |", "|---|---:|---|",
        f"| P2-A1 任务书 | {counts['p2_canonical_target']} | canonical 基准 |",
        f"| 主计划 2.2 | {counts['master_plan_raw']} / {counts['master_plan_after_web_alias']} | `web -> rib` 后与 P2 一致 |",
        f"| 启用 macro ontology | {counts['enabled_macro_ontology_raw_including_stock_and_extras']} | 含 stock、web 和 P2 外 slot_dovetail |",
        f"| capability contract | {counts['capability_contract']} | 缺少 `pocket_circular` |",
        f"| semantic parent coverage | {counts['semantic_parent_coverage']} | 17 个 canonical parent 均有至少一个 subtype |",
        "", "## 2. 规范化类别、alias 与 subtype", "",
        "| canonical | family | aliases | capability | semantic subtypes |",
        "|---|---|---|---|---|",
    ]
    for row in audit["categories"]:
        aliases = ", ".join(f"`{value}`" for value in row["aliases"]) or "—"
        subtypes = ", ".join(f"`{value}`" for value in row["semantic_subtypes"]) or "—"
        capability = "yes" if row["present_in_capability_contract"] else "**gap**"
        lines.append(f"| `{row['canonical_name']}` | {row['family']} | {aliases} | {capability} | {subtypes} |")
    lines.extend(["", "`web` 只作为 `rib` 的输入兼容 alias；CAE-IR 永远输出 `rib`，论文统计不把 `web` 作为第 18 类。",
                  "", "## 3. Policy gaps", ""])
    for gap in audit["policy_gaps"]:
        lines.append(f"- `{gap['code']}` / `{gap['feature']}`：{gap['detail']}")
    lines.extend([
        "", "可组合已有 subtype 的角色标为 `composite`；缺少可区分 subtype 的精确角色标为 `unsupported_no_distinct_subtype`，不得制造新标签。",
        "", "## 4. 旧原型接口审计", "",
        "可复用概念：operations 列表、材料与弹性参数、全局/局部网格尺寸、平移自由度、压力大小，以及 hole wall/boss top 等目标语义。",
        "", "拒绝复用：raw face index、无单位数值、材料静默默认、关键词/prompt 路由、远程模型调用或凭据、硬编码模型身份。P2-A1 从强类型 schema 和稳定错误码重新实现。",
        "", "## 5. 冻结 Recognizer 约束", "",
        "- `pocket_circular` 是 ontology/P2 目标，但不在 capability contract；predicted path 必须视为 coverage gap。",
        "- slot 与 rib/web predicted recall 为 0；即使 policy 定义角色，也必须在 predicted envelope 中 abstain 或人工确认。",
        "- transition 能力 limited；所有下游实验必须保留 coverage 与 abstention。",
    ])
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--aliases-output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for output in (args.json_output, args.markdown_output, args.aliases_output):
        if output.exists() and not args.force:
            raise FileExistsError(f"Refusing to overwrite existing output: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
    audit = build_audit(args.root.resolve())
    args.json_output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown_output.write_text(render_markdown(audit), encoding="utf-8")
    args.aliases_output.write_text(yaml.safe_dump(build_alias_document(audit), allow_unicode=True, sort_keys=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
