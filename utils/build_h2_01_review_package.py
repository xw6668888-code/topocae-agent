"""Create the H2-01 human review CSVs after automated P2-A1 tests pass."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from utils.cae_intent import load_feature_role_policy


SURFACE_OPERATIONS = ["fixed_support", "displacement", "pressure", "surface_traction", "local_mesh_size"]


def runtime_note(feature: str, family: str) -> str:
    if family == "slot" or feature == "rib":
        return "Frozen predicted path must abstain or request human confirmation."
    if feature == "pocket_circular":
        return "Class is absent from the frozen capability contract; predicted coverage is a declared gap."
    if family == "transition":
        return "Frozen transition recognition is limited; preserve coverage and abstention."
    if family == "hole":
        return "Hole family is supported only with abstention in the controlled domain."
    return "Class exists in policy; frozen contract does not report a separate calibrated family recall."


def role_rows(policy: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    serial = 1
    for feature, feature_policy in policy["features"].items():
        for role in feature_policy["selectable_roles"]:
            for operation_type in SURFACE_OPERATIONS:
                compatible = operation_type in role["allowed_operations"] and not role["selection_status"].startswith("unsupported")
                rows.append({
                    "review_id": f"RP-{serial:04d}",
                    "feature_type": feature,
                    "family": feature_policy["family"],
                    "face_role": role["role"],
                    "selection_status": role["selection_status"],
                    "semantic_subtype_candidates": ";".join(role["semantic_subtype_candidates"]),
                    "region_kind": role["region_kind"],
                    "operation_type": operation_type,
                    "proposed_compatible": "yes" if compatible else "no",
                    "requires_complete_instance": "yes" if role["requires_complete_instance"] else "no",
                    "may_include_transition": "yes" if feature_policy["may_include_transition"] else "no",
                    "engineering_explanation": role["selection_notes"],
                    "recognizer_runtime_note": runtime_note(feature, feature_policy["family"]),
                    "review_decision": "",
                    "corrected": "",
                    "reviewer_note": "",
                })
                serial += 1
    for operation_type, explanation in [
        ("material_assignment", "Material assignment uses whole_part scope; it is not a geometric face role."),
        ("global_mesh_size", "Global mesh size uses whole_part scope; local_mesh_size is the feature-targeted operation."),
    ]:
        rows.append({
            "review_id": f"RP-{serial:04d}",
            "feature_type": "__whole_part__",
            "family": "whole_part_scope",
            "face_role": "whole_part",
            "selection_status": "direct",
            "semantic_subtype_candidates": "",
            "region_kind": "whole_part",
            "operation_type": operation_type,
            "proposed_compatible": "yes",
            "requires_complete_instance": "yes",
            "may_include_transition": "no",
            "engineering_explanation": explanation,
            "recognizer_runtime_note": "No feature-object selection is involved.",
            "review_decision": "",
            "corrected": "",
            "reviewer_note": "",
        })
        serial += 1
    return rows


def requirement_rows() -> list[dict[str, str]]:
    raw = [
        ("将所有圆形通孔的孔壁完全固定。", "VALID", "fixed_support", "through_hole_round", "hole_wall", "all", {"constrained_dofs": ["U1", "U2", "U3"]}, "", "", "通孔壁作为约束面集，约束三个平移自由度。"),
        ("令唯一盲孔的孔底沿全局 X 方向位移 0.20 mm。", "VALID", "displacement", "blind_hole_round", "hole_bottom", "exactly_one", {"dof": "U1", "value": {"value": 0.2, "unit": "mm"}, "frame": "global_cartesian"}, "", "", "位移分量、方向和单位完整。"),
        ("在直径最大的阶梯孔大孔壁上施加 3 MPa、沿表面法向的压力。", "VALID", "pressure", "stepped_hole_round", "large_hole_wall", "largest", {"magnitude": {"value": 3, "unit": "MPa"}, "direction": "surface_normal", "comparison_metric": "diameter"}, "", "", "最大值比较指标和压力方向均明确。"),
        ("对沉孔的台阶面施加 1.5 MPa 的全局 Z 正向均布表面牵引。", "VALID", "surface_traction", "counterbore_hole", "step_face", "all", {"magnitude": {"value": 1.5, "unit": "MPa"}, "direction": {"frame": "global_cartesian", "vector": [0, 0, 1]}}, "", "", "牵引使用应力维度并给出笛卡尔方向。"),
        ("在全部锥沉孔的锥面施加 2 MPa、沿表面外法向反向的压力。", "VALID", "pressure", "countersink_hole", "countersink_conical_face", "all", {"magnitude": {"value": 2, "unit": "MPa"}, "direction": "opposite_surface_normal"}, "", "", "压力的法向取向已显式给出。"),
        ("把所有长圆孔的完整孔壁局部网格尺寸设为 0.8 mm。", "VALID", "local_mesh_size", "obround_hole", "hole_wall", "all", {"size": {"value": 0.8, "unit": "mm"}}, "", "", "长圆孔壁由直壁和圆端壁组合。"),
        ("固定全部矩形槽的底面。", "VALID", "fixed_support", "slot_rect", "bottom_face", "all", {"constrained_dofs": ["U1", "U2", "U3"]}, "", "", "只选择有独立 subtype 的槽底。"),
        ("在每个 U 形槽的圆弧封闭端面上施加 0.6 MPa 法向压力。", "VALID", "pressure", "slot_u", "closed_end_face", "all", {"magnitude": {"value": 0.6, "unit": "MPa"}, "direction": "surface_normal"}, "", "", "U 槽封闭端有直接 cylinder subtype。"),
        ("对所有 V 形槽的侧壁施加全局 Y 负向 0.9 MPa 表面牵引。", "VALID", "surface_traction", "slot_v", "side_wall", "all", {"magnitude": {"value": 0.9, "unit": "MPa"}, "direction": {"frame": "global_cartesian", "vector": [0, -1, 0]}}, "", "", "V 槽侧壁角色可直接选择。"),
        ("把全部 O 形槽的完整内壁局部网格尺寸设为 0.5 mm。", "VALID", "local_mesh_size", "slot_o", "complete_slot_inner_wall", "all", {"size": {"value": 0.5, "unit": "mm"}}, "", "", "完整内壁是 side 与 bottom subtype 的组合。"),
        ("在面积最大的矩形口袋底面施加 4 MPa 法向压力。", "VALID", "pressure", "pocket_rect", "bottom_face", "largest", {"magnitude": {"value": 4, "unit": "MPa"}, "direction": "surface_normal", "comparison_metric": "area"}, "", "", "口袋底面与最大面积筛选明确。"),
        ("固定所有圆形口袋的侧壁。", "VALID", "fixed_support", "pocket_circular", "side_wall", "all", {"constrained_dofs": ["U1", "U2", "U3"]}, "", "", "语义合法；predicted path 的 capability gap 由 envelope 处理。"),
        ("在唯一圆柱凸台顶面施加 5 MPa、沿表面法向反向的压力。", "VALID", "pressure", "cylinder_boss", "top_face", "exactly_one", {"magnitude": {"value": 5, "unit": "MPa"}, "direction": "opposite_surface_normal"}, "", "", "凸台顶面和法向取向均明确。"),
        ("将所有矩形凸台的侧壁沿全局 Z 方向位移 0.1 mm。", "VALID", "displacement", "rectangular_boss", "side_wall", "all", {"dof": "U3", "value": {"value": 0.1, "unit": "mm"}, "frame": "global_cartesian"}, "", "", "位移自由度和项目单位完整。"),
        ("对全部 rib 侧面施加全局 X 正向 1 MPa 表面牵引。", "VALID", "surface_traction", "rib", "side_face", "all", {"magnitude": {"value": 1, "unit": "MPa"}, "direction": {"frame": "global_cartesian", "vector": [1, 0, 0]}}, "", "", "web 输入 alias 统一输出 rib；predicted path 仍须 abstain。"),
        ("把半径最小的圆角 blend face 局部网格尺寸设为 0.25 mm。", "VALID", "local_mesh_size", "fillet", "blend_face", "smallest", {"size": {"value": 0.25, "unit": "mm"}, "comparison_metric": "radius"}, "", "", "圆角角色与比较指标明确。"),
        ("将所有倒角 bevel face 完全固定。", "VALID", "fixed_support", "chamfer", "bevel_face", "all", {"constrained_dofs": ["U1", "U2", "U3"]}, "", "", "倒角面作为几何选择，是否物理合理由人工审查。"),
        ("把整个零件赋予材料 Mat-A：弹性模量 210 GPa，泊松比 0.30。", "VALID", "material_assignment", "__whole_part__", "whole_part", "all", {"material_name": "Mat-A", "youngs_modulus": {"value": 210, "unit": "GPa"}, "poisson_ratio": 0.3}, "", "", "材料没有静默默认，属性和值完整。"),
        ("将整个零件的全局网格尺寸设为 4 mm。", "VALID", "global_mesh_size", "__whole_part__", "whole_part", "all", {"size": {"value": 4, "unit": "mm"}}, "", "", "全局网格使用 whole_part scope。"),
        ("材料采用 Mat-B（70 GPa，0.33），全局网格 3 mm，并固定全部通孔孔壁。", "VALID", "material_assignment+global_mesh_size+fixed_support", "through_hole_round", "hole_wall", "all", {"operation_count": 3}, "", "", "一个 intent 内包含三个有唯一 operation_id 的独立操作。"),
        ("固定恰好两个直径为 10±0.1 mm 的通孔孔壁。", "VALID", "fixed_support", "through_hole_round", "hole_wall", "count", {"count": 2, "diameter": {"value": 10, "unit": "mm", "tolerance": 0.1}}, "", "", "count 与直径筛选均明确。"),
        ("在面积最大的圆形口袋底面施加 2.2 MPa 法向压力。", "VALID", "pressure", "pocket_circular", "bottom_face", "largest", {"comparison_metric": "area", "magnitude": {"value": 2.2, "unit": "MPa"}, "direction": "surface_normal"}, "", "", "最大面积比较不需要 raw area truth。"),
        ("对距离零件质心最近的圆柱凸台顶面施加全局 X 方向 0.5 MPa 牵引。", "VALID", "surface_traction", "cylinder_boss", "top_face", "nearest", {"reference": "part_centroid", "reference_frame": "global_cartesian", "magnitude": {"value": 0.5, "unit": "MPa"}}, "", "", "nearest reference 与 frame 都是语义引用。"),
        ("给直径 8 到 12 mm 的所有盲孔孔底施加 1 MPa 法向压力。", "VALID", "pressure", "blind_hole_round", "hole_bottom", "all", {"diameter_between": [{"value": 8, "unit": "mm"}, {"value": 12, "unit": "mm"}], "magnitude": {"value": 1, "unit": "MPa"}, "direction": "surface_normal"}, "", "", "范围筛选和压力参数完整。"),
        ("将宽度小于 6 mm 的所有矩形槽完整区域局部网格尺寸设为 0.4 mm。", "VALID", "local_mesh_size", "slot_rect", "feature_region", "all", {"width_lt": {"value": 6, "unit": "mm"}, "size": {"value": 0.4, "unit": "mm"}}, "", "", "宽度和局部网格尺寸均带单位。"),
        ("固定全部 rib 的完整 feature region。", "VALID", "fixed_support", "rib", "feature_region", "all", {"constrained_dofs": ["U1", "U2", "U3"]}, "", "", "完整 rib region 使用已有 rib/web subtypes 的组合。"),
        ("把最小半径圆角的 blend face 局部网格尺寸设为 0.2 mm。", "VALID", "local_mesh_size", "fillet", "blend_face", "smallest", {"comparison_metric": "radius", "size": {"value": 0.2, "unit": "mm"}}, "", "", "比较量和目标角色都明确。"),
        ("在唯一倒角 bevel face 上施加 0.3 MPa 法向压力。", "VALID", "pressure", "chamfer", "bevel_face", "exactly_one", {"magnitude": {"value": 0.3, "unit": "MPa"}, "direction": "surface_normal"}, "", "", "exactly_one 不使用 count 字段。"),
        ("给盲孔加压力。", "NEEDS_CLARIFICATION", "pressure", "blind_hole_round", "feature_region", "all", {}, "M_FACE_ROLE;M_MAGNITUDE;M_UNIT;M_DIRECTION", "", "请说明盲孔的具体面角色、压力大小、单位以及 surface_normal 或 opposite_surface_normal。", "缺失信息不能使用默认值。"),
        ("把孔固定住。", "NEEDS_CLARIFICATION", "fixed_support", "", "", "all", {}, "", "A_FEATURE_TYPE;A_FACE_ROLE", "请说明是哪一种孔特征，以及固定孔壁、孔底还是完整孔区域。", "“孔”不足以唯一映射到 6 类孔和面角色。"),
        ("全局网格尺寸设为 2。", "NEEDS_CLARIFICATION", "global_mesh_size", "__whole_part__", "whole_part", "all", {"value": 2}, "M_UNIT", "", "网格尺寸 2 使用什么长度单位？", "Abaqus 不提供自动单位换算，单位必须显式。"),
        ("材料用钢。", "NEEDS_CLARIFICATION", "material_assignment", "__whole_part__", "whole_part", "all", {"material_name": "Steel"}, "M_MATERIAL_PROPERTY", "A_MATERIAL", "请确认要使用的弹性模量、单位和泊松比；系统不能套用钢的默认值。", "材料名称不能替代已确认的工程属性。"),
        ("移动圆柱凸台顶面。", "NEEDS_CLARIFICATION", "displacement", "cylinder_boss", "top_face", "all", {}, "M_DISPLACEMENT_COMPONENT;M_UNIT", "A_DIRECTION_ORIENTATION", "请给出位移分量、数值、单位和参考坐标系。", "位移不能从“移动”一词推断。"),
        ("给左边的槽侧壁施加 1 MPa 载荷。", "NEEDS_CLARIFICATION", "", "", "side_wall", "all", {"magnitude": {"value": 1, "unit": "MPa"}}, "M_REFERENCE;M_DIRECTION", "A_FEATURE_TYPE;A_SPATIAL_REFERENCE", "请说明槽类型、左侧参考坐标系，以及载荷是压力还是表面牵引及其方向。", "空间关系和操作类型均不唯一。"),
        ("在沉孔台阶面施加 10 MPa 压力。", "NEEDS_CLARIFICATION", "pressure", "counterbore_hole", "step_face", "all", {"magnitude": {"value": 10, "unit": "MPa"}}, "M_DIRECTION", "A_DIRECTION_ORIENTATION", "压力沿 surface_normal 还是 opposite_surface_normal？", "正负号与 inward/outward 不得凭空推断。"),
        ("固定两个通孔。", "NEEDS_CLARIFICATION", "fixed_support", "through_hole_round", "", "count", {"count": 2}, "M_FACE_ROLE", "", "要固定通孔的孔壁还是其他经确认的角色？", "数量明确，但目标面角色缺失。"),
        ("局部加密最小的圆角。", "NEEDS_CLARIFICATION", "local_mesh_size", "fillet", "blend_face", "smallest", {"comparison_metric": "radius"}, "M_MESH_SIZE;M_UNIT", "", "请给出局部网格尺寸及其长度单位。", "目标明确但操作参数不完整。"),
        ("对沉孔施加载荷。", "NEEDS_CLARIFICATION", "", "counterbore_hole", "feature_region", "all", {}, "M_MAGNITUDE;M_UNIT;M_DIRECTION", "A_FACE_ROLE", "请说明沉孔面角色，并选择压力或表面牵引及其大小、单位和方向。", "“载荷”与完整沉孔区域都存在工程歧义。"),
        ("固定 rib。", "NEEDS_CLARIFICATION", "fixed_support", "rib", "", "all", {}, "M_FACE_ROLE", "A_FACE_ROLE", "要固定完整 rib region、侧面还是 top_or_end_face？", "不能把 support 当成 rib 的固有标签。"),
        ("给口袋底面压力 5。", "NEEDS_CLARIFICATION", "pressure", "", "bottom_face", "all", {"value": 5}, "M_FEATURE_TYPE;M_UNIT;M_DIRECTION", "A_FEATURE_TYPE;A_DIRECTION_ORIENTATION", "请说明矩形或圆形口袋、压力单位，以及法向取向。", "特征类型、单位和方向均未闭合。"),
    ]
    rows = []
    for index, item in enumerate(raw, start=1):
        if len(item) == 10:
            text, status, op, feature, role, quantifier, params, missing, ambiguities, rationale = item
            question = ""
        else:
            text, status, op, feature, role, quantifier, params, missing, ambiguities, question, rationale = item
        rows.append({
            "review_id": f"RQ-{index:03d}",
            "requirement_text": text,
            "source_language": "zh-CN",
            "expected_status": status,
            "analysis_type": "static_structural",
            "operation_type": op,
            "feature_type": feature,
            "face_role": role,
            "quantifier": quantifier,
            "typed_parameters_json": json.dumps(params, ensure_ascii=False, sort_keys=True),
            "missing_information": missing,
            "ambiguities": ambiguities,
            "clarification_question": question,
            "engineering_rationale": rationale,
            "review_decision": "",
            "corrected": "",
            "reviewer_note": "",
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def readme(role_count: int, requirement_count: int, ambiguous_count: int) -> str:
    return f"""# H2-01 CAE 工程语义人工审查包

状态：`NOT_COMPLETED`  
Role policy 行数：{role_count}  
Requirement 行数：{requirement_count}（其中故意不完整或有歧义：{ambiguous_count}）

## 为什么必须人工审核

Schema、单位和错误关闭行为可以自动验证，但“某类面是否适合作为某种结构静力操作的候选区域”、需求是否符合工程师表达习惯，以及缺失时应该追问什么，不能由自动测试替代。此 Gate 只审工程语义，不要求填写 face ID，也不执行 Abaqus。

## 文件

1. `feature_role_policy_review.csv`：每行一个 feature-role-operation 组合；含 direct/composite/unsupported 状态、semantic subtype 依据和 proposed compatibility。
2. `cae_requirement_review_40.csv`：40 条已预填候选需求；覆盖全部 feature family 与 7 种 operation，12 条故意不完整或有歧义。

## 具体操作

1. 用 Excel、LibreOffice Calc 或支持 UTF-8 CSV 的编辑器打开两个 CSV。不要改表头、`review_id` 或行顺序。
2. 在 `review_decision` 列逐行填写且只能填写：`approve`、`edit` 或 `reject`。
3. 选择 `approve`：保留预填语义；`corrected` 可留空。
4. 选择 `edit`：必须在 `corrected` 写一个 JSON object，只列要修改的字段。例如 role 行可写 `{{"proposed_compatible":"no","engineering_explanation":"原因"}}`；requirement 行可写 `{{"requirement_text":"修订文本","expected_status":"VALID","face_role":"hole_wall"}}`。
5. 选择 `reject`：在 `reviewer_note` 写明拒绝原因；不要仅写“错了”。
6. `feature_role_policy_review.csv` 的 {role_count} 行必须全部审核。可先筛选 `proposed_compatible=no`、`selection_status=unsupported_no_distinct_subtype` 和 `recognizer_runtime_note` 做重点检查，再批量填充其余同类行。
7. `cae_requirement_review_40.csv` 至少审核 30 行；建议审核全部 40 行。重点检查 `expected_status=NEEDS_CLARIFICATION` 的 12 行是否确实需要追问、问题是否合理。
8. 保存为原文件名的 UTF-8 CSV。不要另存为 `.xlsx`，不要删除未审行。
9. 保存后在本 Goal 中回复“已完成 H2-01，请继续”。

## 继续前的自动拒绝条件

- role policy 存在任何空 `review_decision`；
- requirement 已审少于 30 行；
- 决定不是 `approve | edit | reject`；
- `edit` 但 `corrected` 为空、不是 JSON object 或与基础字段矛盾；
- `reject` 没有 `reviewer_note`；
- 修改引入 raw face ID、sample/part/file identity、无单位物理量或 geometry source；
- 修改试图进入 LLM、Pointer、Abaqus 或 P3。

## Gate 通过标准

只有用户保存审核结果并明确通知继续后，Agent 才能解析反馈、更新 schema examples/policy/fixtures、重跑测试并冻结 P2。当前 README 与空决定列不构成 H2-01 通过；未经用户审核不得自动批准。
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    role_path = args.output_dir / "feature_role_policy_review.csv"
    requirement_path = args.output_dir / "cae_requirement_review_40.csv"
    readme_path = args.output_dir / "README_H2_01.zh-CN.md"
    for output in (role_path, requirement_path, readme_path):
        if output.exists() and not args.force:
            raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    roles = role_rows(load_feature_role_policy())
    requirements = requirement_rows()
    if len(requirements) != 40:
        raise AssertionError(f"Expected 40 requirements, got {len(requirements)}")
    clarification_count = sum(row["expected_status"] == "NEEDS_CLARIFICATION" for row in requirements)
    if clarification_count < 10:
        raise AssertionError("At least 10 incomplete or ambiguous requirements are required")
    if not {"fixed_support", "displacement", "pressure", "surface_traction", "material_assignment", "global_mesh_size", "local_mesh_size"}.issubset(
        {part for row in requirements for part in row["operation_type"].split("+")}
    ):
        raise AssertionError("Requirement operation coverage is incomplete")
    write_csv(role_path, roles)
    write_csv(requirement_path, requirements)
    readme_path.write_text(readme(len(roles), len(requirements), clarification_count), encoding="utf-8")
    print(json.dumps({"role_rows": len(roles), "requirement_rows": len(requirements), "clarification_rows": clarification_count}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
