"""Deterministic bilingual text evidence for P4-R3A."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Mapping, Pattern


EVIDENCE_VERSION = "cae-text-evidence/1.0.0"
SUPPORTED_LANGUAGES = frozenset({"en-US", "zh-CN"})


class EvidenceInputError(ValueError):
    """The public extractor input is invalid."""


@dataclass(frozen=True)
class _Rule:
    rule_id: str
    normalized: Any
    pattern: Pattern[str]


def _rule(rule_id: str, normalized: Any, pattern: str, flags: int = 0) -> _Rule:
    return _Rule(rule_id, normalized, re.compile(pattern, flags))


_ACTIONS = {
    "en-US": (
        _rule("P4R3A_ACTION_LOCAL_MESH_EN", "local_mesh_size", r"\b(?:(?:set\s+)?local\s+mesh(?:\s+size)?|locally\s+refine|local\s+refinement)\b", re.I),
        _rule("P4R3A_ACTION_GLOBAL_MESH_EN", "global_mesh_size", r"\b(?:set\s+)?global\s+mesh(?:\s+size)?\b", re.I),
        _rule("P4R3A_ACTION_MATERIAL_EN", "material_assignment", r"\b(?:assign(?:\s+the)?\s+(?:material\s+)?|material\s+assignment|use\s+material)\b", re.I),
        _rule("P4R3A_ACTION_TRACTION_EN", "surface_traction", r"\b(?:apply\s+)?surface\s+traction\b", re.I),
        _rule("P4R3A_ACTION_DISPLACEMENT_EN", "displacement", r"\b(?:apply|prescribe|set)?\s*(?:a\s+)?displacement\b", re.I),
        _rule("P4R3A_ACTION_FIXED_EN", "fixed_support", r"\b(?:fix|fixed\s+support|fully\s+constrain)\b", re.I),
        _rule("P4R3A_ACTION_PRESSURE_EN", "pressure", r"\b(?:apply\s+)?(?:a\s+)?pressure\b", re.I),
    ),
    "zh-CN": (
        _rule("P4R3A_ACTION_LOCAL_MESH_ZH", "local_mesh_size", r"(?:\u8bbe\u7f6e|\u65bd\u52a0|\u91c7\u7528)?\u5c40\u90e8\u7f51\u683c(?:\u5c3a\u5bf8)?|\u5c40\u90e8\u52a0\u5bc6"),
        _rule("P4R3A_ACTION_GLOBAL_MESH_ZH", "global_mesh_size", r"(?:\u8bbe\u7f6e|\u65bd\u52a0|\u91c7\u7528)?\u5168\u5c40\u7f51\u683c(?:\u5c3a\u5bf8)?"),
        _rule("P4R3A_ACTION_MATERIAL_ZH", "material_assignment", r"(?:\u6307\u5b9a|\u8d4b\u4e88|\u5206\u914d)(?:\u6750\u6599)?|\u6750\u6599\u8d4b\u503c"),
        _rule("P4R3A_ACTION_TRACTION_ZH", "surface_traction", r"(?:\u65bd\u52a0)?\u8868\u9762\u7275\u5f15"),
        _rule("P4R3A_ACTION_DISPLACEMENT_ZH", "displacement", r"(?:\u65bd\u52a0|\u6307\u5b9a|\u8bbe\u7f6e)?\u4f4d\u79fb"),
        _rule("P4R3A_ACTION_FIXED_ZH", "fixed_support", r"\u56fa\u5b9a\u652f\u6491|\u5b8c\u5168\u7ea6\u675f|\u56fa\u5b9a"),
        _rule("P4R3A_ACTION_PRESSURE_ZH", "pressure", r"(?:\u65bd\u52a0)?\u538b\u529b"),
    ),
}

_FEATURES = {
    "en-US": (
        _rule("P4R3A_FEATURE_COUNTERBORE_EN", "counterbore_hole", r"\bcounterbore(?:d)?\s+holes?\b", re.I),
        _rule("P4R3A_FEATURE_COUNTERSINK_EN", "countersink_hole", r"\bcountersink(?:ed)?\s+holes?\b|\bcountersinks?\b", re.I),
        _rule("P4R3A_FEATURE_BLIND_HOLE_EN", "blind_hole_round", r"\b(?:round\s+)?blind\s+(?:round\s+)?holes?\b", re.I),
        _rule("P4R3A_FEATURE_STEPPED_HOLE_EN", "stepped_hole_round", r"\bstepped\s+(?:round\s+)?holes?\b", re.I),
        _rule("P4R3A_FEATURE_OBROUND_HOLE_EN", "obround_hole", r"\bobround\s+holes?\b", re.I),
        _rule("P4R3A_FEATURE_THROUGH_HOLE_EN", "through_hole_round", r"\b(?:(?:round\s+)?through|round)?\s*holes?\b", re.I),
        _rule("P4R3A_FEATURE_SLOT_U_EN", "slot_u", r"\bU[-\s]?shaped\s+slots?\b", re.I),
        _rule("P4R3A_FEATURE_SLOT_V_EN", "slot_v", r"\bV[-\s]?shaped\s+slots?\b", re.I),
        _rule("P4R3A_FEATURE_SLOT_O_EN", "slot_o", r"\bO[-\s]?shaped\s+slots?\b", re.I),
        _rule("P4R3A_FEATURE_SLOT_EN", "slot_rect", r"\b(?:rectangular\s+)?slots?\b", re.I),
        _rule("P4R3A_FEATURE_POCKET_RECT_EN", "pocket_rect", r"\brectangular\s+pockets?\b", re.I),
        _rule("P4R3A_FEATURE_POCKET_CIRCULAR_EN", "pocket_circular", r"\bcircular\s+pockets?\b", re.I),
        _rule("P4R3A_FEATURE_CYLINDER_BOSS_EN", "cylinder_boss", r"\b(?:cylindrical|round)\s+boss(?:es)?\b", re.I),
        _rule("P4R3A_FEATURE_RECT_BOSS_EN", "rectangular_boss", r"\brectangular\s+boss(?:es)?\b", re.I),
        _rule("P4R3A_FEATURE_RIB_EN", "rib", r"\b(?:ribs?|webs?)\b", re.I),
        _rule("P4R3A_FEATURE_FILLET_EN", "fillet", r"\bfillets?\b", re.I),
        _rule("P4R3A_FEATURE_CHAMFER_EN", "chamfer", r"\bchamfers?\b", re.I),
    ),
    "zh-CN": (
        _rule("P4R3A_FEATURE_COUNTERSINK_ZH", "countersink_hole", r"\u9525\u6c89\u5b54|\u57cb\u5934\u5b54|\u9525\u5f62\u6c89\u5934\u5b54"),
        _rule("P4R3A_FEATURE_COUNTERBORE_ZH", "counterbore_hole", r"(?<!\u9525)\u6c89\u5b54|\u67f1\u5f62\u6c89\u5934\u5b54"),
        _rule("P4R3A_FEATURE_BLIND_HOLE_ZH", "blind_hole_round", r"\u76f2\u5b54"),
        _rule("P4R3A_FEATURE_STEPPED_HOLE_ZH", "stepped_hole_round", r"\u9636\u68af\u5b54"),
        _rule("P4R3A_FEATURE_OBROUND_HOLE_ZH", "obround_hole", r"\u957f\u5706\u5b54|\u8170\u5f62\u5b54"),
        _rule("P4R3A_FEATURE_THROUGH_HOLE_ZH", "through_hole_round", r"\u901a\u5b54|\u5706\u5b54|\u5b54"),
        _rule("P4R3A_FEATURE_SLOT_U_ZH", "slot_u", r"U\s*\u5f62\u69fd"),
        _rule("P4R3A_FEATURE_SLOT_V_ZH", "slot_v", r"V\s*\u5f62\u69fd"),
        _rule("P4R3A_FEATURE_SLOT_O_ZH", "slot_o", r"O\s*\u5f62\u69fd"),
        _rule("P4R3A_FEATURE_SLOT_ZH", "slot_rect", r"\u77e9\u5f62\u69fd|\u69fd"),
        _rule("P4R3A_FEATURE_POCKET_RECT_ZH", "pocket_rect", r"\u77e9\u5f62\u8154|\u77e9\u5f62\u53e3\u888b"),
        _rule("P4R3A_FEATURE_POCKET_CIRCULAR_ZH", "pocket_circular", r"\u5706\u5f62\u8154|\u5706\u5f62\u53e3\u888b"),
        _rule("P4R3A_FEATURE_CYLINDER_BOSS_ZH", "cylinder_boss", r"\u5706\u67f1\u51f8\u53f0"),
        _rule("P4R3A_FEATURE_RECT_BOSS_ZH", "rectangular_boss", r"\u77e9\u5f62\u51f8\u53f0"),
        _rule("P4R3A_FEATURE_RIB_ZH", "rib", r"\u52a0\u5f3a\u7b4b|\u808b\u677f|\u8179\u677f"),
        _rule("P4R3A_FEATURE_FILLET_ZH", "fillet", r"\u5706\u89d2"),
        _rule("P4R3A_FEATURE_CHAMFER_ZH", "chamfer", r"\u5012\u89d2"),
    ),
}

_ROLES = {
    "en-US": (
        _rule("P4R3A_ROLE_THROUGH_SMALL_WALL_EN", "through_small_hole_wall", r"\bthrough\s+small[-\s]hole\s+walls?\b", re.I),
        _rule("P4R3A_ROLE_THROUGH_WALL_EN", "through_hole_wall", r"\bthrough[-\s]hole\s+walls?\b", re.I),
        _rule("P4R3A_ROLE_LARGE_WALL_EN", "large_hole_wall", r"\blarge[-\s]hole\s+walls?\b", re.I),
        _rule("P4R3A_ROLE_SMALL_WALL_EN", "small_hole_wall", r"\bsmall[-\s]hole\s+walls?\b", re.I),
        _rule("P4R3A_ROLE_SLOT_INNER_EN", "complete_slot_inner_wall", r"\bcomplete\s+slot\s+inner\s+walls?\b", re.I),
        _rule("P4R3A_ROLE_HOLE_BOTTOM_EN", "hole_bottom", r"\bhole\s+bottoms?\b", re.I),
        _rule("P4R3A_ROLE_STEP_FACE_EN", "step_face", r"\bstep\s+faces?\b", re.I),
        _rule("P4R3A_ROLE_COUNTERSINK_FACE_EN", "countersink_conical_face", r"\b(?:conical\s+countersink|countersink\s+conical)\s+faces?\b", re.I),
        _rule("P4R3A_ROLE_INNER_WALL_EN", "hole_wall", r"\b(?:inner|bore|hole)\s+walls?\b", re.I),
        _rule("P4R3A_ROLE_BOTTOM_FACE_EN", "bottom_face", r"\bbottom\s+faces?\b|\bfloors?\b", re.I),
        _rule("P4R3A_ROLE_OPENING_FACE_EN", "opening_face", r"\bopening\s+faces?\b", re.I),
        _rule("P4R3A_ROLE_TOP_FACE_EN", "top_face", r"\btop\s+faces?\b", re.I),
        _rule("P4R3A_ROLE_END_FACE_EN", "end_face", r"\bend\s+faces?\b", re.I),
        _rule("P4R3A_ROLE_SIDE_WALL_EN", "side_wall", r"\bside\s+walls?\b", re.I),
        _rule("P4R3A_ROLE_SIDE_FACE_EN", "side_face", r"\bside\s+faces?\b", re.I),
        _rule("P4R3A_ROLE_BLEND_FACE_EN", "blend_face", r"\bblend\s+faces?\b", re.I),
        _rule("P4R3A_ROLE_BEVEL_FACE_EN", "bevel_face", r"\bbevel\s+faces?\b", re.I),
        _rule("P4R3A_ROLE_FEATURE_REGION_EN", "feature_region", r"\bcomplete\s+(?:feature\s+)?regions?\b", re.I),
    ),
    "zh-CN": (
        _rule("P4R3A_ROLE_THROUGH_SMALL_WALL_ZH", "through_small_hole_wall", r"\u901a\u5c0f\u5b54\u58c1"),
        _rule("P4R3A_ROLE_THROUGH_WALL_ZH", "through_hole_wall", r"\u901a\u5b54\u58c1"),
        _rule("P4R3A_ROLE_LARGE_WALL_ZH", "large_hole_wall", r"\u5927\u5b54\u58c1"),
        _rule("P4R3A_ROLE_SMALL_WALL_ZH", "small_hole_wall", r"\u5c0f\u5b54\u58c1"),
        _rule("P4R3A_ROLE_SLOT_INNER_ZH", "complete_slot_inner_wall", r"\u5b8c\u6574\u69fd\u5185\u58c1"),
        _rule("P4R3A_ROLE_HOLE_BOTTOM_ZH", "hole_bottom", r"\u5b54\u5e95"),
        _rule("P4R3A_ROLE_STEP_FACE_ZH", "step_face", r"\u53f0\u9636\u9762"),
        _rule("P4R3A_ROLE_COUNTERSINK_FACE_ZH", "countersink_conical_face", r"\u9525\u6c89\u9762|\u57cb\u5934\u9525\u9762"),
        _rule("P4R3A_ROLE_INNER_WALL_ZH", "hole_wall", r"\u5185\u58c1|\u5b54\u58c1"),
        _rule("P4R3A_ROLE_BOTTOM_FACE_ZH", "bottom_face", r"\u5e95\u9762"),
        _rule("P4R3A_ROLE_OPENING_FACE_ZH", "opening_face", r"\u5f00\u53e3\u9762"),
        _rule("P4R3A_ROLE_TOP_FACE_ZH", "top_face", r"\u9876\u9762"),
        _rule("P4R3A_ROLE_END_FACE_ZH", "end_face", r"\u7aef\u9762"),
        _rule("P4R3A_ROLE_SIDE_WALL_ZH", "side_wall", r"\u4fa7\u58c1"),
        _rule("P4R3A_ROLE_SIDE_FACE_ZH", "side_face", r"\u4fa7\u9762"),
        _rule("P4R3A_ROLE_BLEND_FACE_ZH", "blend_face", r"\u5706\u89d2\u9762|\u8fc7\u6e21\u9762"),
        _rule("P4R3A_ROLE_BEVEL_FACE_ZH", "bevel_face", r"\u659c\u9762|\u5012\u89d2\u9762"),
        _rule("P4R3A_ROLE_FEATURE_REGION_ZH", "feature_region", r"\u5b8c\u6574\u7279\u5f81\u533a\u57df|\u6574\u4e2a\u7279\u5f81"),
    ),
}

_UNIT_PATTERN = r"GPa|MPa|kPa|Pa|mm(?:\^?2)|cm(?:\^?2)|m(?:\^?2)|\u03bcm|um|mm|cm|\u6beb\u7c73|\u5398\u7c73|\u5fae\u7c73|\u7c73|m"
_QUANTITY_RE = re.compile(rf"(?P<value>[+-]?\d+(?:\.\d+)?)\s*(?P<unit>{_UNIT_PATTERN})(?![A-Za-z0-9])", re.I)
_UNITS = {
    "gpa": ("pressure", 1000.0, "MPa"), "mpa": ("pressure", 1.0, "MPa"),
    "kpa": ("pressure", 0.001, "MPa"), "pa": ("pressure", 0.000001, "MPa"),
    "mm": ("length", 1.0, "mm"), "\u6beb\u7c73": ("length", 1.0, "mm"),
    "cm": ("length", 10.0, "mm"), "\u5398\u7c73": ("length", 10.0, "mm"),
    "m": ("length", 1000.0, "mm"), "\u7c73": ("length", 1000.0, "mm"),
    "um": ("length", 0.001, "mm"), "\u03bcm": ("length", 0.001, "mm"), "\u5fae\u7c73": ("length", 0.001, "mm"),
    "mm2": ("area", 1.0, "mm2"), "cm2": ("area", 100.0, "mm2"), "m2": ("area", 1000000.0, "mm2"),
}
_EN_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
_ZH_NUMBERS = {"\u4e00": 1, "\u4e8c": 2, "\u4e24": 2, "\u4e09": 3, "\u56db": 4, "\u4e94": 5, "\u516d": 6, "\u4e03": 7, "\u516b": 8, "\u4e5d": 9}


def _unit_key(unit: str) -> str:
    return unit.lower().replace("^", "")


def _quantity(value: str, unit: str) -> dict[str, Any]:
    dimension, multiplier, target = _UNITS[_unit_key(unit)]
    return {"dimension": dimension, "value": float(value) * multiplier, "unit": target, "source_unit": unit}


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _zh_count(raw: str) -> int:
    if raw.isdigit():
        return int(raw)
    if raw == "\u5341":
        return 10
    if "\u5341" in raw:
        left, right = raw.split("\u5341", 1)
        return (10 if not left else _ZH_NUMBERS[left] * 10) + (0 if not right else _ZH_NUMBERS[right])
    if all(character in _ZH_NUMBERS for character in raw):
        return int("".join(str(_ZH_NUMBERS[character]) for character in raw))
    raise ValueError("EVIDENCE_E_CHINESE_NUMBER")


class CAETextEvidenceExtractor:
    """Extract evidence using only text and language."""

    def extract(self, value: Mapping[str, Any]) -> dict[str, Any]:
        clean = self._validate(value)
        text, language = clean["text"], clean["language"]
        clauses = self._clauses(text, language)
        items: list[dict[str, Any]] = []
        for clause in clauses:
            self._add(items, "clause_boundary", clause["span"]["start"], clause["span"]["end"], text, {"index": clause["index"]}, "P4R3A_CLAUSE_BOUNDARY", clause["index"])
        self._apply(items, "operation_cue", _ACTIONS[language], text, clauses)
        self._apply(items, "feature_mention", _FEATURES[language], text, clauses)
        self._apply(items, "face_role_mention", _ROLES[language], text, clauses)
        self._selectors(items, text, language, clauses)
        self._quantities(items, text, clauses)
        self._parameters(items, text, clauses)
        self._spatial(items, text, clauses)
        self._exclusions(items, text, clauses)
        self._missing(items, text, clauses)
        items.sort(key=lambda row: (row["span"]["start"], row["span"]["end"], row["category"], row["rule_id"]))
        return {"contract_version": EVIDENCE_VERSION, "language": language, "text_length": len(text), "clauses": clauses, "evidence": items}

    @staticmethod
    def _validate(value: Mapping[str, Any]) -> dict[str, str]:
        if not isinstance(value, Mapping) or set(value) != {"text", "language"}:
            raise EvidenceInputError("EVIDENCE_E_INPUT_FIELDS")
        if not isinstance(value["text"], str) or not value["text"].strip():
            raise EvidenceInputError("EVIDENCE_E_TEXT")
        if value["language"] not in SUPPORTED_LANGUAGES:
            raise EvidenceInputError("EVIDENCE_E_LANGUAGE")
        return {"text": value["text"], "language": value["language"]}

    def _clauses(self, text: str, language: str) -> list[dict[str, Any]]:
        base, cursor = [], 0
        for match in re.finditer(r"[;\u3002\uff1b\uff01\uff1f!?\n]+", text):
            span = _trim(text, cursor, match.start())
            if span[0] < span[1]:
                base.append(span)
            cursor = match.end()
        span = _trim(text, cursor, len(text))
        if span[0] < span[1]:
            base.append(span)
        result = []
        for start, end in base:
            anchors = sorted({m.start() for rule in _ACTIONS[language] for m in rule.pattern.finditer(text, start, end)})
            cuts = [start, *(anchors[1:] if len(anchors) > 1 else ()), end]
            for left, right in zip(cuts, cuts[1:]):
                left, right = _trim(text, left, right)
                prefix = re.match(r"^(?:and\s+then|then|and)\s+", text[left:right], re.I) if language == "en-US" else re.match(r"^(?:\u7136\u540e|\u5e76\u4e14|\u540c\u65f6|\u518d)\s*", text[left:right])
                if prefix:
                    left += prefix.end()
                if left < right:
                    result.append((left, right))
        return [{"index": index, "span": {"start": start, "end": end}} for index, (start, end) in enumerate(result)]

    @staticmethod
    def _index(clauses: Iterable[Mapping[str, Any]], position: int) -> int:
        for clause in clauses:
            if clause["span"]["start"] <= position < clause["span"]["end"]:
                return int(clause["index"])
        return 0

    @staticmethod
    def _add(items: list[dict[str, Any]], category: str, start: int, end: int, text: str, normalized: Any, rule_id: str, clause_index: int) -> None:
        items.append({"category": category, "span": {"start": start, "end": end}, "text": text[start:end], "normalized": normalized, "rule_id": rule_id, "clause_index": clause_index})

    def _apply(self, items: list[dict[str, Any]], category: str, rules: Iterable[_Rule], text: str, clauses: list[dict[str, Any]]) -> None:
        occupied = []
        for rule in rules:
            for match in rule.pattern.finditer(text):
                if any(match.start() < right and left < match.end() for left, right in occupied):
                    continue
                occupied.append(match.span())
                self._add(items, category, match.start(), match.end(), text, rule.normalized, rule.rule_id, self._index(clauses, match.start()))

    def _selectors(self, items: list[dict[str, Any]], text: str, language: str, clauses: list[dict[str, Any]]) -> None:
        specs = (
            (("P4R3A_SELECTOR_EXACTLY_ONE_EN", {"selector": "exactly_one"}, r"\b(?:exactly\s+one|a\s+single|single|the\s+only|only)\b"), ("P4R3A_SELECTOR_ALL_EN", {"selector": "all"}, r"\b(?:all|every|each)\b"), ("P4R3A_SELECTOR_LARGEST_EN", {"selector": "largest", "compare_by": "area"}, r"\blargest\b"), ("P4R3A_SELECTOR_SMALLEST_EN", {"selector": "smallest", "compare_by": "area"}, r"\bsmallest\b"), ("P4R3A_SELECTOR_NEAREST_EN", {"selector": "nearest"}, r"\bnearest\b"))
            if language == "en-US" else
            (("P4R3A_SELECTOR_EXACTLY_ONE_ZH", {"selector": "exactly_one"}, r"\u6070\u597d\u4e00\u4e2a|\u4ec5\u4e00\u4e2a|\u552f\u4e00\u4e00\u4e2a|\u552f\u4e00|\u5355\u4e2a"), ("P4R3A_SELECTOR_ALL_ZH", {"selector": "all"}, r"\u6240\u6709|\u5168\u90e8|\u6bcf\u4e2a"), ("P4R3A_SELECTOR_LARGEST_ZH", {"selector": "largest", "compare_by": "area"}, r"\u6700\u5927"), ("P4R3A_SELECTOR_SMALLEST_ZH", {"selector": "smallest", "compare_by": "area"}, r"\u6700\u5c0f"), ("P4R3A_SELECTOR_NEAREST_ZH", {"selector": "nearest"}, r"\u6700\u8fd1"))
        )
        for rule_id, normalized, pattern in specs:
            for match in re.finditer(pattern, text, re.I):
                self._add(items, "selector", match.start(), match.end(), text, normalized, rule_id, self._index(clauses, match.start()))
        pattern = re.compile(r"\b(?P<count>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:(?:largest|smallest|nearest)\s+)?(?:round\s+)?(?:holes?|slots?|pockets?|boss(?:es)?|ribs?|fillets?|chamfers?)\b", re.I) if language == "en-US" else re.compile(r"(?P<count>\d+|[\u4e00\u4e8c\u4e24\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341]+)\u4e2a(?:\u5706\u5b54|\u901a\u5b54|\u5b54|\u69fd|\u8154|\u51f8\u53f0|\u52a0\u5f3a\u7b4b|\u808b\u677f|\u5706\u89d2|\u5012\u89d2)")
        for match in pattern.finditer(text):
            raw = match.group("count")
            count = (int(raw) if raw.isdigit() else _EN_NUMBERS[raw.lower()]) if language == "en-US" else _zh_count(raw)
            prefix = text[max(0, match.start() - 12):match.start()].lower()
            if count == 1 and ("exactly" in prefix or "only" in prefix or "\u6070\u597d" in prefix or "\u4ec5" in prefix or "\u552f\u4e00" in prefix):
                continue
            self._add(items, "selector", match.start("count"), match.end("count"), text, {"selector": "count", "count": count}, "P4R3A_SELECTOR_COUNT_EN" if language == "en-US" else "P4R3A_SELECTOR_COUNT_ZH", self._index(clauses, match.start()))

    def _quantities(self, items: list[dict[str, Any]], text: str, clauses: list[dict[str, Any]]) -> None:
        for match in _QUANTITY_RE.finditer(text):
            self._add(items, "quantity", match.start(), match.end(), text, _quantity(match.group("value"), match.group("unit")), "P4R3A_QUANTITY_UNIT_DICTIONARY", self._index(clauses, match.start()))

    def _parameters(self, items: list[dict[str, Any]], text: str, clauses: list[dict[str, Any]]) -> None:
        component = re.compile(rf"\b(?P<dof>U[123])\s*=\s*(?P<value>[+-]?\d+(?:\.\d+)?)\s*(?P<unit>{_UNIT_PATTERN})(?![A-Za-z0-9])", re.I)
        for match in component.finditer(text):
            value = _quantity(match.group("value"), match.group("unit"))
            self._add(items, "displacement_component", match.start(), match.end(), text, {"dof": match.group("dof").upper(), "value": value["value"], "unit": value["unit"]}, "P4R3A_DISPLACEMENT_COMPONENT", self._index(clauses, match.start()))
        for match in re.finditer(r"\bU[123]\b", text, re.I):
            self._add(items, "degree_of_freedom", match.start(), match.end(), text, match.group().upper(), "P4R3A_DOF_EXPLICIT", self._index(clauses, match.start()))
        for pattern in (re.compile(r"\b(?:E|Young'?s?\s+modulus)\s*=\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>GPa|MPa|kPa|Pa)\b", re.I), re.compile(r"(?:\u5f39\u6027\u6a21\u91cf|\u6768\u6c0f\u6a21\u91cf)\s*(?:\u4e3a|=)?\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>GPa|MPa|kPa|Pa)", re.I)):
            for match in pattern.finditer(text):
                value = _quantity(match.group("value"), match.group("unit"))
                self._add(items, "youngs_modulus", match.start(), match.end(), text, {"value": value["value"], "unit": value["unit"]}, "P4R3A_MATERIAL_YOUNGS", self._index(clauses, match.start()))
        for match in re.finditer(r"(?:Poisson(?:'?s)?\s+ratio|\u6cca\u677e\u6bd4)\s*(?:\u4e3a|=)?\s*(?P<value>0(?:\.\d+)?|1(?:\.0+)?)", text, re.I):
            self._add(items, "poisson_ratio", match.start(), match.end(), text, float(match.group("value")), "P4R3A_MATERIAL_POISSON", self._index(clauses, match.start()))
        for pattern in (re.compile(r"\b(?:assign|use\s+material)\s+(?P<name>[A-Za-z][A-Za-z0-9_-]*)\s+(?:with|\()", re.I), re.compile(r"(?:\u6307\u5b9a|\u8d4b\u4e88|\u5206\u914d)(?:\u6750\u6599\s*)?(?P<name>[\u4e00-\u9fffA-Za-z0-9_-]+?)(?:\u6750\u6599)?(?:\uff0c|,|\uff08|\(|\u5176|\u5f39\u6027\u6a21\u91cf|\u6768\u6c0f\u6a21\u91cf)", re.I)):
            for match in pattern.finditer(text):
                self._add(items, "material_name", match.start("name"), match.end("name"), text, match.group("name").lower(), "P4R3A_MATERIAL_NAME", self._index(clauses, match.start()))
        self._simple_patterns(items, text, clauses, "coordinate_frame", (("P4R3A_FRAME_GLOBAL", "global", r"\bglobal\s+(?:coordinate\s+)?frame\b|\u5168\u5c40\u5750\u6807\u7cfb"), ("P4R3A_FRAME_LOCAL", "local", r"\blocal\s+(?:coordinate\s+)?frame\b|\u5c40\u90e8\u5750\u6807\u7cfb"), ("P4R3A_FRAME_CYLINDRICAL", "cylindrical", r"\bcylindrical\s+(?:coordinate\s+)?frame\b|\u67f1\u5750\u6807\u7cfb")))
        self._simple_patterns(items, text, clauses, "direction", (("P4R3A_DIRECTION_OPPOSITE_NORMAL", {"kind": "opposite_normal"}, r"\bopposite\s+to\s+(?:the\s+)?surface\s+normal\b|\u6cd5\u5411\u53cd\u5411"), ("P4R3A_DIRECTION_NORMAL", {"kind": "normal"}, r"\bnormal(?:ly)?(?:\s+to)?\b|\u6cd5\u5411"), ("P4R3A_DIRECTION_POS_X", {"kind": "vector", "vector": [1.0, 0.0, 0.0]}, r"(?:positive|\+)\s*X(?:\s+direction)?|\u6cbf\u6b63X\u65b9\u5411"), ("P4R3A_DIRECTION_NEG_X", {"kind": "vector", "vector": [-1.0, 0.0, 0.0]}, r"(?:negative|-)\s*X(?:\s+direction)?|\u6cbf\u8d1fX\u65b9\u5411"), ("P4R3A_DIRECTION_POS_Y", {"kind": "vector", "vector": [0.0, 1.0, 0.0]}, r"(?:positive|\+)\s*Y(?:\s+direction)?|\u6cbf\u6b63Y\u65b9\u5411"), ("P4R3A_DIRECTION_NEG_Y", {"kind": "vector", "vector": [0.0, -1.0, 0.0]}, r"(?:negative|-)\s*Y(?:\s+direction)?|\u6cbf\u8d1fY\u65b9\u5411"), ("P4R3A_DIRECTION_POS_Z", {"kind": "vector", "vector": [0.0, 0.0, 1.0]}, r"(?:positive|\+)\s*Z(?:\s+direction)?|\u6cbf\u6b63Z\u65b9\u5411"), ("P4R3A_DIRECTION_NEG_Z", {"kind": "vector", "vector": [0.0, 0.0, -1.0]}, r"(?:negative|-)\s*Z(?:\s+direction)?|\u6cbf\u8d1fZ\u65b9\u5411")))
        actions = self._actions(items)
        for item in list(items):
            if item["category"] != "quantity":
                continue
            action = actions.get(item["clause_index"])
            dimension = item["normalized"]["dimension"]
            normalized = {key: item["normalized"][key] for key in ("value", "unit")}
            if action in {"global_mesh_size", "local_mesh_size"} and dimension == "length":
                self._add(items, "mesh_size", item["span"]["start"], item["span"]["end"], text, normalized, "P4R3A_PARAMETER_MESH_SIZE", item["clause_index"])
            if action in {"pressure", "surface_traction"} and dimension == "pressure":
                self._add(items, "load_magnitude", item["span"]["start"], item["span"]["end"], text, normalized, "P4R3A_PARAMETER_LOAD_MAGNITUDE", item["clause_index"])

    def _simple_patterns(self, items: list[dict[str, Any]], text: str, clauses: list[dict[str, Any]], category: str, specs: Iterable[tuple[str, Any, str]]) -> None:
        occupied: list[tuple[int, int]] = []
        for rule_id, normalized, pattern in specs:
            for match in re.finditer(pattern, text, re.I):
                if any(match.start() < right and left < match.end() for left, right in occupied):
                    continue
                occupied.append(match.span())
                self._add(items, category, match.start(), match.end(), text, normalized, rule_id, self._index(clauses, match.start()))

    def _spatial(self, items: list[dict[str, Any]], text: str, clauses: list[dict[str, Any]]) -> None:
        for match in re.finditer(r"(?:global\s+)?(?P<axis>[XYZ])\s+axis|\u5168\u5c40(?P<zh_axis>[XYZ])\u8f74", text, re.I):
            axis = (match.group("axis") or match.group("zh_axis")).upper()
            self._add(items, "spatial_reference", match.start(), match.end(), text, {"kind": "axis", "axis": axis}, "P4R3A_SPATIAL_REFERENCE_AXIS", self._index(clauses, match.start()))
        for match in re.finditer(r"\b(?:global\s+)?origin\b|\u5168\u5c40\u539f\u70b9|\u539f\u70b9", text, re.I):
            self._add(items, "spatial_reference", match.start(), match.end(), text, {"kind": "origin"}, "P4R3A_SPATIAL_REFERENCE_ORIGIN", self._index(clauses, match.start()))

    def _exclusions(self, items: list[dict[str, Any]], text: str, clauses: list[dict[str, Any]]) -> None:
        differences = list(re.finditer(r"\b(?:except|excluding|other\s+than)\b|\u9664\u4e86|\u6392\u9664|\u4e0d\u5305\u62ec", text, re.I))
        for match in differences:
            self._add(items, "exclusion", match.start(), match.end(), text, {"kind": "set_difference"}, "P4R3A_EXCLUSION_SET_DIFFERENCE", self._index(clauses, match.start()))
        complex_matches = list(re.finditer(r"\bbut\s+not\b|\u4f46\u662f?\u4e0d|\u4f46\u4e0d", text, re.I))
        if complex_matches or len(differences) > 1:
            match = complex_matches[0] if complex_matches else differences[1]
            self._add(items, "exclusion", match.start(), match.end(), text, {"kind": "complex_exclusion"}, "P4R3A_EXCLUSION_COMPLEX", self._index(clauses, match.start()))

    def _missing(self, items: list[dict[str, Any]], text: str, clauses: list[dict[str, Any]]) -> None:
        required = {"pressure": (("load_magnitude", "magnitude"), ("direction", "direction")), "surface_traction": (("load_magnitude", "magnitude"), ("direction", "direction")), "displacement": (("displacement_component", "displacements"), ("coordinate_frame", "load_frame")), "material_assignment": (("material_name", "material.name"), ("youngs_modulus", "material.youngs_modulus"), ("poisson_ratio", "material.poisson_ratio")), "global_mesh_size": (("mesh_size", "mesh_size"),), "local_mesh_size": (("mesh_size", "mesh_size"),)}
        feature_actions = {"fixed_support", "displacement", "pressure", "surface_traction", "local_mesh_size"}
        actions = self._actions(items)
        present: dict[int, set[str]] = {}
        for item in items:
            present.setdefault(item["clause_index"], set()).add(item["category"])
        for clause in clauses:
            index, action = clause["index"], actions.get(clause["index"])
            if action is None:
                continue
            needs = list(required.get(action, ()))
            if action in feature_actions:
                needs.extend((("feature_mention", "feature"), ("face_role_mention", "face_role"), ("selector", "selector")))
            if any(item["category"] == "selector" and item["clause_index"] == index and item["normalized"].get("selector") == "nearest" for item in items):
                needs.append(("spatial_reference", "nearest_to"))
            for category, parameter in needs:
                if category not in present.get(index, set()):
                    self._add(items, "missing_required_parameter", clause["span"]["start"], clause["span"]["end"], text, {"action": action, "parameter": parameter}, "P4R3A_MISSING_REQUIRED_PARAMETER", index)

    @staticmethod
    def _actions(items: Iterable[Mapping[str, Any]]) -> dict[int, str]:
        result = {}
        for item in items:
            if item["category"] == "operation_cue":
                result.setdefault(int(item["clause_index"]), str(item["normalized"]))
        return result


__all__ = ["CAETextEvidenceExtractor", "EVIDENCE_VERSION", "EvidenceInputError"]
