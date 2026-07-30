"""Bilingual deterministic keyword/rule baseline developed on train/val."""

from __future__ import annotations

import copy
import re
import unicodedata
from typing import Any, Mapping, Sequence

from utils.cae_intent import validate_intent_result


RESULT_VERSION = "cae-intent-result/1.0.0"
INTENT_VERSION = "cae-intent/1.0.0"
UNIT_SYSTEM = {
    "length": "mm",
    "area": "mm2",
    "force": "N",
    "pressure": "MPa",
    "displacement": "mm",
    "angle": "deg",
}

FEATURE_TERMS = {
    "through_hole_round": (
        "round through holes",
        "round through hole",
        "through holes",
        "through hole",
        "圆形通孔",
        "通孔",
    ),
    "blind_hole_round": (
        "round blind holes",
        "round blind hole",
        "blind holes",
        "blind hole",
        "圆形盲孔",
        "盲孔",
    ),
    "stepped_hole_round": (
        "stepped round holes",
        "stepped round hole",
        "stepped holes",
        "stepped hole",
        "圆形阶梯孔",
        "阶梯孔",
    ),
    "counterbore_hole": (
        "counterbore holes",
        "counterbore hole",
        "counterbores",
        "counterbore",
        "沉孔",
    ),
    "countersink_hole": (
        "countersink holes",
        "countersink hole",
        "countersinks",
        "countersink",
        "锥沉孔",
    ),
    "obround_hole": (
        "obround holes",
        "obround hole",
        "长圆孔",
    ),
    "slot_rect": (
        "rectangular slots",
        "rectangular slot",
        "矩形槽",
    ),
    "slot_u": (
        "u-shaped slots",
        "u-shaped slot",
        "u shaped slots",
        "u shaped slot",
        "u 形槽",
        "u形槽",
    ),
    "slot_v": (
        "v-shaped slots",
        "v-shaped slot",
        "v shaped slots",
        "v shaped slot",
        "v 形槽",
        "v形槽",
    ),
    "slot_o": (
        "o-shaped slots",
        "o-shaped slot",
        "o shaped slots",
        "o shaped slot",
        "o 形槽",
        "o形槽",
    ),
    "pocket_rect": (
        "rectangular pockets",
        "rectangular pocket",
        "矩形口袋",
    ),
    "pocket_circular": (
        "circular pockets",
        "circular pocket",
        "round pockets",
        "round pocket",
        "圆形口袋",
    ),
    "cylinder_boss": (
        "cylindrical bosses",
        "cylindrical boss",
        "cylinder bosses",
        "cylinder boss",
        "圆柱凸台",
    ),
    "rectangular_boss": (
        "rectangular bosses",
        "rectangular boss",
        "矩形凸台",
    ),
    "rib": (
        "stiffening ribs",
        "stiffening rib",
        "ribs",
        "rib",
        "webs",
        "web",
        "加强筋",
        "筋板",
    ),
    "fillet": ("fillets", "fillet", "圆角"),
    "chamfer": ("chamfers", "chamfer", "倒角"),
}

ROLE_TERMS = {
    "feature_region": (
        "complete feature regions",
        "complete feature region",
        "as a complete feature region",
        "完整特征区域",
    ),
    "through_small_hole_wall": (
        "through small-hole walls",
        "through small-hole wall",
        "through small hole walls",
        "through small hole wall",
        "贯通小孔壁",
    ),
    "countersink_conical_face": (
        "conical countersink faces",
        "conical countersink face",
        "锥沉面",
    ),
    "complete_slot_inner_wall": (
        "complete slot inner walls",
        "complete slot inner wall",
        "完整槽内壁",
    ),
    "bottom_or_root_face": (
        "bottom or root faces",
        "bottom or root face",
        "底面或根部面",
    ),
    "top_or_end_face": (
        "top or end faces",
        "top or end face",
        "顶面或端面",
    ),
    "large_hole_wall": (
        "large-hole walls",
        "large-hole wall",
        "large hole walls",
        "large hole wall",
        "大孔壁",
    ),
    "small_hole_wall": (
        "small-hole walls",
        "small-hole wall",
        "small hole walls",
        "small hole wall",
        "小孔壁",
    ),
    "through_hole_wall": (
        "through-hole walls",
        "through-hole wall",
        "through hole walls",
        "through hole wall",
        "通孔壁",
    ),
    "straight_side_wall": (
        "straight side walls",
        "straight side wall",
        "直侧壁",
    ),
    "rounded_end_wall": (
        "rounded end walls",
        "rounded end wall",
        "圆弧端壁",
    ),
    "closed_end_face": (
        "closed end faces",
        "closed end face",
        "封闭端面",
    ),
    "hole_bottom": ("hole bottoms", "hole bottom", "孔底"),
    "hole_wall": ("hole walls", "hole wall", "孔壁"),
    "step_face": ("step faces", "step face", "台阶面"),
    "side_wall": ("side walls", "side wall", "侧壁"),
    "bottom_face": ("bottom faces", "bottom face", "底面"),
    "top_face": ("top faces", "top face", "顶面"),
    "side_face": ("side faces", "side face", "侧面"),
    "blend_face": ("blend faces", "blend face", "过渡曲面"),
    "bevel_face": ("bevel faces", "bevel face", "斜面"),
}

METRIC_TERMS = {
    "diameter": ("diameter", "直径"),
    "radius": ("radius", "半径"),
    "depth": ("depth", "深度"),
    "area": ("area", "面积"),
    "length": ("length", "长度"),
    "width": ("width", "宽度"),
    "thickness": ("thickness", "厚度"),
}

UNIT_DIMENSION = {
    "um": "length",
    "mm": "length",
    "cm": "length",
    "m": "length",
    "mm2": "area",
    "cm2": "area",
    "m2": "area",
    "Pa": "pressure",
    "kPa": "pressure",
    "MPa": "pressure",
    "GPa": "pressure",
    "N/mm2": "pressure",
}
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
UNIT = r"(?:N/mm2|GPa|MPa|kPa|Pa|mm2|cm2|m2|um|mm|cm|m)"
QUANTITY_RE = re.compile(
    rf"(?P<value>{NUMBER})\s*(?P<unit>{UNIT})(?![A-Za-z0-9/])",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).casefold()
    value = value.replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", value).strip()


def _find_term(text: str, mapping: Mapping[str, Sequence[str]]) -> str | None:
    matches: list[tuple[int, int, str]] = []
    for canonical, terms in mapping.items():
        for term in terms:
            normalized = _normalize(term)
            index = text.find(normalized)
            if index >= 0:
                matches.append((len(normalized), -index, canonical))
    if not matches:
        return None
    return max(matches)[2]


def _invalid_result(
    *,
    error_code: str = "E_SCHEMA_VALIDATION",
    field_path: str = "/",
) -> dict[str, Any]:
    return {
        "schema_version": RESULT_VERSION,
        "status": "INVALID",
        "error_code": error_code,
        "field_path": field_path,
        "missing_information": [],
        "ambiguities": [],
        "clarification_questions": [],
    }


def _missing_result(
    *,
    code: str,
    field_path: str,
    description: str,
) -> dict[str, Any]:
    return {
        "schema_version": RESULT_VERSION,
        "status": "NEEDS_CLARIFICATION",
        "error_code": "E_MISSING_INFORMATION",
        "field_path": field_path,
        "missing_information": [
            {
                "code": code,
                "field_path": field_path,
                "description": description,
            }
        ],
        "ambiguities": [],
        "clarification_questions": [
            {
                "question_id": "q1",
                "field_path": field_path,
                "question": f"Please clarify: {description}",
            }
        ],
    }


def _quantity_matches(text: str) -> list[dict[str, Any]]:
    result = []
    for match in QUANTITY_RE.finditer(text):
        unit_token = match.group("unit")
        canonical_unit = next(
            unit for unit in UNIT_DIMENSION if unit.casefold() == unit_token.casefold()
        )
        result.append(
            {
                "value": float(match.group("value")),
                "unit": canonical_unit,
                "start": match.start(),
                "end": match.end(),
            }
        )
    return result


def _nearest_quantity(
    text: str,
    anchors: Sequence[str],
    *,
    allowed_units: set[str],
) -> dict[str, Any] | None:
    positions = [
        text.find(anchor) for anchor in anchors if text.find(anchor) >= 0
    ]
    if not positions:
        return None
    candidates = [
        item for item in _quantity_matches(text) if item["unit"] in allowed_units
    ]
    if not candidates:
        return None
    anchor = min(positions)
    return min(
        candidates,
        key=lambda item: min(
            abs(item["start"] - position) for position in positions
        ),
    )


def _typed_quantity(item: Mapping[str, Any], dimension: str) -> dict[str, Any]:
    return {
        "dimension": dimension,
        "value": float(item["value"]),
        "unit": str(item["unit"]),
    }


def _parse_feature_target(text: str) -> dict[str, Any] | None:
    feature_type = _find_term(text, FEATURE_TERMS)
    face_role = _find_term(text, ROLE_TERMS)
    if feature_type is None or face_role is None:
        return None

    filters: dict[str, Any] = {}
    comparison_metric = None
    count = None
    if "nearest" in text or "最近" in text or "最靠近" in text:
        quantifier = "nearest"
        if "part centroid" in text or "零件质心" in text:
            filters["nearest_to"] = {"reference_type": "part_centroid"}
        else:
            return {
                "_clarification": _missing_result(
                    code="M_REFERENCE",
                    field_path="/operations/0/target/filters/nearest_to",
                    description=(
                        "A nearest-target request requires an explicit spatial "
                        "reference."
                    ),
                )
            }
    elif "largest" in text or "最大" in text:
        quantifier = "largest"
        comparison_metric = _find_term(text, METRIC_TERMS)
    elif "smallest" in text or "最小" in text:
        quantifier = "smallest"
        comparison_metric = _find_term(text, METRIC_TERMS)
    elif re.search(r"\bonly\b", text) or "唯一" in text:
        quantifier = "exactly_one"
    else:
        feature_position = min(
            (
                text.find(_normalize(term))
                for term in FEATURE_TERMS[feature_type]
                if text.find(_normalize(term)) >= 0
            ),
            default=-1,
        )
        prefix = text[max(0, feature_position - 40):feature_position]
        count_match = re.search(r"\b(\d+)\s+(?:matching\s+)?$", prefix)
        if count_match is None:
            count_match = re.search(r"(\d+)\s*个[^0-9]{0,12}$", prefix)
        if count_match is not None:
            quantifier = "count"
            count = int(count_match.group(1))
        elif (
            re.search(r"\ball\b", text)
            or re.search(r"\bevery\b", text)
            or "全部" in text
            or "所有" in text
        ):
            quantifier = "all"
        else:
            quantifier = "exactly_one"

    for metric, terms in METRIC_TERMS.items():
        for term in terms:
            metric_term = _normalize(term)
            index = text.find(metric_term)
            if index < 0:
                continue
            tail = text[index + len(metric_term): index + len(metric_term) + 45]
            match = QUANTITY_RE.search(tail)
            if match is None:
                continue
            unit_token = match.group("unit")
            unit = next(
                item
                for item in UNIT_DIMENSION
                if item.casefold() == unit_token.casefold()
            )
            dimension = "area" if metric == "area" else "length"
            filters[metric] = {
                "operator": "eq",
                "value": {
                    "dimension": dimension,
                    "value": float(match.group("value")),
                    "unit": unit,
                },
            }
            break

    target: dict[str, Any] = {
        "feature_type": feature_type,
        "quantifier": quantifier,
        "face_role": face_role,
        "filters": filters,
    }
    if count is not None:
        target["count"] = count
    if comparison_metric is not None:
        target["comparison_metric"] = comparison_metric
    if quantifier == "nearest":
        target["reference_frame"] = {"kind": "part_local_cartesian"}
    return target


def _operation_positions(text: str) -> list[tuple[int, str]]:
    patterns = {
        "material_assignment": (
            "assign material",
            "use material",
            "赋予材料",
            "材料 ",
        ),
        "global_mesh_size": (
            "global mesh",
            "全局网格",
        ),
        "fixed_support": (
            "fixed support",
            "fully constrain",
            "completely fix",
            "完全固定",
        ),
        "displacement": ("displacement", "位移"),
        "surface_traction": ("surface traction", "表面牵引"),
        "pressure": ("pressure", "压力"),
        "local_mesh_size": (
            "local mesh",
            "locally refine",
            "local refinement",
            "局部网格",
            "局部加密",
        ),
    }
    found: list[tuple[int, str]] = []
    for operation, terms in patterns.items():
        positions = [text.find(term) for term in terms if text.find(term) >= 0]
        if positions:
            found.append((min(positions), operation))
    # "mesh" clauses can contain both locally-refine and local-mesh phrases.
    return sorted(set(found), key=lambda item: (item[0], item[1]))


def _parse_direction(text: str, operation_type: str) -> dict[str, Any] | None:
    if operation_type == "pressure":
        if "opposite" in text or "反向" in text:
            return {"kind": "opposite_surface_normal"}
        if "surface normal" in text or "表面法向" in text:
            return {"kind": "surface_normal"}
        return None
    vectors = {
        "+x": [1.0, 0.0, 0.0],
        "-y": [0.0, -1.0, 0.0],
        "+z": [0.0, 0.0, 1.0],
    }
    compact = text.replace(" ", "")
    for token, vector in vectors.items():
        if token in compact:
            return {"kind": "global_cartesian", "vector": vector}
    return None


def _parse_material_parameters(text: str) -> dict[str, Any] | None:
    name_match = re.search(
        r"(?:material|材料)\s+([a-z0-9][a-z0-9_-]*)",
        text,
        re.IGNORECASE,
    )
    modulus_match = re.search(
        rf"(?:\be\s*=|young'?s modulus|elastic modulus|弹性模量)\s*({NUMBER})\s*({UNIT})",
        text,
        re.IGNORECASE,
    )
    poisson_match = re.search(
        rf"(?:poisson(?: ratio)?|泊松比)\s*({NUMBER})",
        text,
        re.IGNORECASE,
    )
    if not (name_match and modulus_match and poisson_match):
        return None
    unit_token = modulus_match.group(2)
    unit = next(
        item for item in UNIT_DIMENSION if item.casefold() == unit_token.casefold()
    )
    return {
        "material_name": name_match.group(1),
        "elastic": {
            "youngs_modulus": {
                "dimension": "pressure",
                "value": float(modulus_match.group(1)),
                "unit": unit,
            },
            "poisson_ratio": float(poisson_match.group(1)),
        },
    }


def _parse_operation(
    operation_type: str,
    text: str,
    target: Mapping[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    length_units = {"um", "mm", "cm", "m"}
    pressure_units = {"Pa", "kPa", "MPa", "GPa", "N/mm2"}
    if operation_type == "material_assignment":
        parameters = _parse_material_parameters(text)
        if parameters is None:
            return None, _missing_result(
                code="M_MATERIAL_PROPERTY",
                field_path="/operations/0/parameters/elastic",
                description="Material name and elastic properties are required.",
            )
        return {
            "operation_type": operation_type,
            "target": {"scope": "whole_part"},
            "parameters": parameters,
        }, None
    if operation_type == "global_mesh_size":
        quantity = _nearest_quantity(
            text,
            ("global mesh", "全局网格"),
            allowed_units=length_units,
        )
        if quantity is None:
            return None, _missing_result(
                code="M_MESH_SIZE",
                field_path="/operations/0/parameters/size",
                description="A global mesh size is required.",
            )
        return {
            "operation_type": operation_type,
            "target": {"scope": "whole_part"},
            "parameters": {"size": _typed_quantity(quantity, "length")},
        }, None
    if target is None:
        return None, _missing_result(
            code="M_TARGET",
            field_path="/operations/0/target",
            description="A feature target and face role are required.",
        )
    if operation_type == "fixed_support":
        parameters = {"constrained_dofs": ["U1", "U2", "U3"]}
    elif operation_type == "displacement":
        quantity = _nearest_quantity(
            text,
            ("displacement", "位移"),
            allowed_units=length_units,
        )
        dof_match = re.search(r"\b(U[123])\b", text, re.IGNORECASE)
        if quantity is None or dof_match is None:
            return None, _missing_result(
                code="M_DISPLACEMENT_COMPONENT",
                field_path="/operations/0/parameters/components",
                description="A displacement DOF and value are required.",
            )
        parameters = {
            "components": [
                {
                    "dof": dof_match.group(1).upper(),
                    "value": _typed_quantity(quantity, "displacement"),
                }
            ],
            "reference_frame": {"kind": "global_cartesian"},
        }
    elif operation_type in {"pressure", "surface_traction"}:
        anchor = ("surface traction", "表面牵引") if operation_type == "surface_traction" else ("pressure", "压力")
        quantity = _nearest_quantity(
            text,
            anchor,
            allowed_units=pressure_units,
        )
        if quantity is None:
            return None, _missing_result(
                code="M_MAGNITUDE",
                field_path="/operations/0/parameters/magnitude",
                description="A load magnitude is required.",
            )
        direction = _parse_direction(text, operation_type)
        if direction is None:
            return None, _missing_result(
                code="M_DIRECTION",
                field_path="/operations/0/parameters/direction",
                description="An explicit load direction is required.",
            )
        parameters = {
            "magnitude": _typed_quantity(quantity, "pressure"),
            "direction": direction,
        }
    elif operation_type == "local_mesh_size":
        quantity = _nearest_quantity(
            text,
            ("local mesh", "locally refine", "局部网格", "局部加密"),
            allowed_units=length_units,
        )
        if quantity is None:
            return None, _missing_result(
                code="M_MESH_SIZE",
                field_path="/operations/0/parameters/size",
                description="A local mesh size is required.",
            )
        parameters = {"size": _typed_quantity(quantity, "length")}
    else:
        return None, _invalid_result(error_code="E_UNSUPPORTED_OPERATION")
    return {
        "operation_type": operation_type,
        "target": copy.deepcopy(dict(target)),
        "parameters": parameters,
    }, None


def parse_keyword_query(text: str, language: str) -> dict[str, Any]:
    """Parse only natural-language text and language into a P2 result."""

    if not isinstance(text, str) or not text.strip():
        return _invalid_result(error_code="E_JSON_TYPE")
    if language not in {"en-US", "zh-CN"}:
        return _invalid_result(error_code="E_INVALID_ENUM")
    normalized = _normalize(text)
    if (
        "except" in normalized
        or "excluding" in normalized
        or "排除" in normalized
    ):
        result = _invalid_result(
            error_code="E_SCHEMA_VALIDATION",
            field_path="/operations/0/target",
        )
        return result
    target = _parse_feature_target(normalized)
    if isinstance(target, Mapping) and "_clarification" in target:
        return copy.deepcopy(target["_clarification"])
    positions = _operation_positions(normalized)
    if not positions:
        return _invalid_result(error_code="E_UNSUPPORTED_OPERATION")
    operations = []
    for _, operation_type in positions:
        operation, nonvalid = _parse_operation(
            operation_type,
            normalized,
            target if isinstance(target, Mapping) else None,
        )
        if nonvalid is not None:
            return nonvalid
        if operation is not None:
            operations.append(operation)
    for index, operation in enumerate(operations, start=1):
        operation["operation_id"] = f"op{index}"
    intent_result = {
        "schema_version": RESULT_VERSION,
        "status": "VALID",
        "intent": {
            "schema_version": INTENT_VERSION,
            "analysis_type": "static_structural",
            "operations": operations,
            "source_language": language,
            "assumptions": [],
            "unit_system": copy.deepcopy(UNIT_SYSTEM),
            "normalization_audit": [],
        },
    }
    validation = validate_intent_result(intent_result)
    if validation["valid"]:
        return intent_result
    first = validation["errors"][0]
    result = _invalid_result(
        error_code=(
            str(first["error_code"])
            if str(first["error_code"])
            in {
                "E_SCHEMA_VALIDATION",
                "E_ROLE_UNKNOWN",
                "E_ROLE_OPERATION_INCOMPATIBLE",
                "E_UNIT_UNSUPPORTED",
                "E_UNIT_DIMENSION_MISMATCH",
                "E_QUANTIFIER_COUNT_CONFLICT",
                "E_REFERENCE_FRAME_REQUIRED",
                "E_REFERENCE_FRAME_UNEXPECTED",
                "E_PRESSURE_DIRECTION_INVALID",
                "E_DISPLACEMENT_DOF_DUPLICATE",
                "E_FILTER_RANGE_INVALID",
            }
            else "E_SCHEMA_VALIDATION"
        ),
        field_path=str(first.get("field_path", "/")),
    )
    return result


__all__ = ["parse_keyword_query"]
