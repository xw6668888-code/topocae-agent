from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from p1_c.all_features import HOST_FAMILIES, MACRO_CLASSES
from p1_c.supported_wall_families import SUPPORTED_WALL_HOST_FAMILIES


PLAN_SCHEMA = "topocae_p1_c_clean_context_plan_1"
PLAN_REVISION = "p1_c_clean_context_latin_plan_r1"
SEED_NAMESPACE = 202607700000
CONTEXT_QUOTAS = {"target_only": 240, "pair": 120, "triple": 40}
SPLIT_BY_CONTEXT_INDEX = {
    "target_only": (("train", 192), ("val", 24), ("development_test", 24)),
    "pair": (("train", 96), ("val", 12), ("development_test", 12)),
    "triple": (("train", 32), ("val", 4), ("development_test", 4)),
}
SCALE_FACTORS = (0.72, 0.90, 1.00, 1.25, 1.60)
ROTATION_AXES = (
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (1.0, 2.0, 3.0),
    (2.0, 3.0, 1.0),
    (3.0, 1.0, 2.0),
)
ROTATION_ANGLES = (0.0, 0.31, 0.57, 0.83, 1.11, 1.37, 1.73, 2.09)
_RIB_INDEX = MACRO_CLASSES.index("rib")
_WEB_INDEX = MACRO_CLASSES.index("web")


@dataclass(frozen=True)
class PlannedSample:
    sample_id: str
    primary_target: str
    primary_index: int
    class_sample_index: int
    context: str
    context_index: int
    secondary_features: tuple[str, ...]
    split: str
    seed: int
    host_family: str
    host_family_kind: str
    rotation_axis: tuple[float, float, float]
    rotation_angle_radians: float
    uniform_scale: float
    base_geometry_family_id: str
    parametric_family_id: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["secondary_features"] = list(self.secondary_features)
        payload["rotation_axis"] = list(self.rotation_axis)
        return payload


def compatibility_matrix() -> dict[str, list[str]]:
    matrix: dict[str, list[str]] = {}
    for primary in MACRO_CLASSES:
        compatible = [secondary for secondary in MACRO_CLASSES if secondary != primary]
        if primary == "rib":
            compatible.remove("web")
        elif primary == "web":
            compatible.remove("rib")
        matrix[primary] = compatible
    return matrix


def build_plan() -> list[PlannedSample]:
    secondary_streams = {
        primary_index: _secondary_stream(primary_index)
        for primary_index in range(len(MACRO_CLASSES))
    }
    plan: list[PlannedSample] = []
    for primary_index, primary in enumerate(MACRO_CLASSES):
        stream = secondary_streams[primary_index]
        occurrence_cursor = 0
        class_sample_index = 0
        for context in ("target_only", "pair", "triple"):
            for context_index in range(CONTEXT_QUOTAS[context]):
                if context == "target_only":
                    secondary_features: tuple[str, ...] = ()
                elif context == "pair":
                    secondary_features = (stream[occurrence_cursor],)
                    occurrence_cursor += 1
                else:
                    secondary_features = (stream[occurrence_cursor], stream[occurrence_cursor + 1])
                    occurrence_cursor += 2
                if len(set(secondary_features)) != len(secondary_features):
                    raise AssertionError(f"duplicate triple secondary for {primary}: {secondary_features}")
                contains_supported_wall = any(
                    macro in {"rib", "web"} for macro in (primary, *secondary_features)
                )
                if contains_supported_wall:
                    host_family = SUPPORTED_WALL_HOST_FAMILIES[(primary_index + class_sample_index) % 4]
                    host_family_kind = "supported_wall"
                else:
                    host_family = HOST_FAMILIES[(primary_index + class_sample_index) % 4]
                    host_family_kind = "general"
                sample_id = f"p1c_{primary_index + 1:02d}_{primary}_{class_sample_index:04d}_{context}"
                split = _split_for(context, context_index)
                seed = SEED_NAMESPACE + primary_index * 10000 + class_sample_index
                plan.append(
                    PlannedSample(
                        sample_id=sample_id,
                        primary_target=primary,
                        primary_index=primary_index,
                        class_sample_index=class_sample_index,
                        context=context,
                        context_index=context_index,
                        secondary_features=secondary_features,
                        split=split,
                        seed=seed,
                        host_family=host_family,
                        host_family_kind=host_family_kind,
                        rotation_axis=ROTATION_AXES[(primary_index * 3 + class_sample_index) % len(ROTATION_AXES)],
                        rotation_angle_radians=ROTATION_ANGLES[(primary_index + class_sample_index) % len(ROTATION_ANGLES)],
                        uniform_scale=SCALE_FACTORS[(primary_index * 2 + class_sample_index) % len(SCALE_FACTORS)],
                        base_geometry_family_id=f"p1c-base::{primary_index:02d}::{class_sample_index:04d}",
                        parametric_family_id=f"p1c-param::{primary}::{context}::{host_family}::{class_sample_index:04d}",
                    )
                )
                class_sample_index += 1
        if occurrence_cursor != 200:
            raise AssertionError(f"secondary occurrence cursor mismatch for {primary}: {occurrence_cursor}")
    audit = audit_plan(plan)
    if not audit["passed"]:
        raise RuntimeError(f"P1-C formal plan failed its own audit: {audit['failures']}")
    return plan


def audit_plan(plan: Iterable[PlannedSample]) -> dict[str, Any]:
    records = list(plan)
    failures: list[str] = []
    primary_counts = Counter(record.primary_target for record in records)
    context_by_primary: dict[str, Counter[str]] = defaultdict(Counter)
    secondary_counts: Counter[str] = Counter()
    directed_counts: Counter[tuple[str, str]] = Counter()
    host_by_primary: dict[str, Counter[str]] = defaultdict(Counter)
    split_counts: Counter[str] = Counter()
    split_by_primary_context: dict[str, Counter[str]] = defaultdict(Counter)
    matrix = compatibility_matrix()
    ids: set[str] = set()
    seeds: set[int] = set()
    for record in records:
        if record.sample_id in ids:
            failures.append(f"duplicate sample id: {record.sample_id}")
        if record.seed in seeds:
            failures.append(f"duplicate seed: {record.seed}")
        ids.add(record.sample_id)
        seeds.add(record.seed)
        context_by_primary[record.primary_target][record.context] += 1
        host_by_primary[record.primary_target][record.host_family] += 1
        split_counts[record.split] += 1
        split_by_primary_context[f"{record.primary_target}::{record.context}"][record.split] += 1
        expected_secondary_count = {"target_only": 0, "pair": 1, "triple": 2}[record.context]
        if len(record.secondary_features) != expected_secondary_count:
            failures.append(f"context arity mismatch: {record.sample_id}")
        if len(set(record.secondary_features)) != len(record.secondary_features):
            failures.append(f"duplicate secondary in one sample: {record.sample_id}")
        for secondary in record.secondary_features:
            secondary_counts[secondary] += 1
            directed_counts[(record.primary_target, secondary)] += 1
            if secondary not in matrix[record.primary_target]:
                failures.append(f"incompatible pair: {record.primary_target}->{secondary}")
    if len(records) != 7200:
        failures.append(f"sample count {len(records)} != 7200")
    for primary in MACRO_CLASSES:
        if primary_counts[primary] != 400:
            failures.append(f"primary quota {primary}={primary_counts[primary]}")
        for context, quota in CONTEXT_QUOTAS.items():
            if context_by_primary[primary][context] != quota:
                failures.append(f"context quota {primary}/{context}={context_by_primary[primary][context]}")
        if len(host_by_primary[primary]) < 4:
            failures.append(f"host family variety {primary}={len(host_by_primary[primary])}")
        if host_by_primary[primary] and max(host_by_primary[primary].values()) > 140:
            failures.append(f"host family exceeds 35% for {primary}: {dict(host_by_primary[primary])}")
        distinct_secondaries = sum(directed_counts[(primary, secondary)] > 0 for secondary in MACRO_CLASSES)
        if distinct_secondaries < 10:
            failures.append(f"secondary variety {primary}={distinct_secondaries}")
    for secondary in MACRO_CLASSES:
        if secondary_counts[secondary] != 200:
            failures.append(f"secondary quota {secondary}={secondary_counts[secondary]}")
    over_pairs = {
        f"{primary}->{secondary}": count
        for (primary, secondary), count in directed_counts.items()
        if count > 16
    }
    if over_pairs:
        failures.append(f"directed pair quota exceeded: {over_pairs}")
    expected_splits = {"train": 5760, "val": 720, "development_test": 720}
    if dict(split_counts) != expected_splits:
        failures.append(f"split quota mismatch: {dict(split_counts)}")
    return {
        "schema_version": "topocae_p1_c_clean_context_plan_audit_1",
        "passed": not failures,
        "failures": failures,
        "sample_count": len(records),
        "primary_counts": {macro: primary_counts[macro] for macro in MACRO_CLASSES},
        "context_counts_by_primary": {
            macro: dict(context_by_primary[macro]) for macro in MACRO_CLASSES
        },
        "secondary_counts": {macro: secondary_counts[macro] for macro in MACRO_CLASSES},
        "directed_pair_max": max(directed_counts.values(), default=0),
        "directed_pair_counts": {
            f"{primary}->{secondary}": directed_counts[(primary, secondary)]
            for primary in MACRO_CLASSES
            for secondary in MACRO_CLASSES
            if directed_counts[(primary, secondary)]
        },
        "host_family_counts_by_primary": {
            macro: dict(host_by_primary[macro]) for macro in MACRO_CLASSES
        },
        "split_counts": dict(split_counts),
        "split_counts_by_primary_context": {
            key: dict(value) for key, value in sorted(split_by_primary_context.items())
        },
    }


def write_plan(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    manifests = output / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    plan = build_plan()
    plan_path = manifests / "generation_plan.jsonl"
    with plan_path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in plan:
            stream.write(json.dumps(record.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n")
    matrix = {
        "schema_version": "topocae_p1_c_compatibility_matrix_1",
        "revision": PLAN_REVISION,
        "compatibility": compatibility_matrix(),
        "contracts": {
            "no_self_pair": True,
            "rib_web_mutual_pair_excluded": True,
            "matrix_written_before_generation": True,
        },
    }
    matrix_path = manifests / "compatibility_matrix.json"
    matrix_path.write_text(json.dumps(matrix, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit = audit_plan(plan)
    audit_path = manifests / "generation_plan_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    freeze = {
        "schema_version": PLAN_SCHEMA,
        "revision": PLAN_REVISION,
        "seed_namespace": SEED_NAMESPACE,
        "sample_count": len(plan),
        "generation_plan_sha256": sha256_file(plan_path),
        "compatibility_matrix_sha256": sha256_file(matrix_path),
        "generation_plan_audit_sha256": sha256_file(audit_path),
        "model_results_seen": False,
    }
    freeze_path = manifests / "generation_plan_freeze.json"
    freeze_path.write_text(json.dumps(freeze, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return freeze


def load_plan(path: Path) -> list[PlannedSample]:
    records: list[PlannedSample] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            payload = json.loads(line)
            payload["secondary_features"] = tuple(payload["secondary_features"])
            payload["rotation_axis"] = tuple(float(value) for value in payload["rotation_axis"])
            records.append(PlannedSample(**payload))
    return records


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _secondary_stream(primary_index: int) -> list[str]:
    stream: list[str] = []
    shift_one_helpers = [index for index in range(18) if index not in {10, 11, 12}]
    shift_last_helpers = [index for index in range(18) if index not in {11, 12, 13}]
    shift_one_cursor = 0
    shift_last_cursor = 0
    for occurrence in range(200):
        shift = occurrence % 17 + 1
        mapping = [(row + shift) % 18 for row in range(18)]
        if shift == 1:
            helper = shift_one_helpers[shift_one_cursor % len(shift_one_helpers)]
            shift_one_cursor += 1
            mapping[_RIB_INDEX] = (helper + 1) % 18
            mapping[helper] = _WEB_INDEX
        elif shift == 17:
            helper = shift_last_helpers[shift_last_cursor % len(shift_last_helpers)]
            shift_last_cursor += 1
            mapping[_WEB_INDEX] = (helper - 1) % 18
            mapping[helper] = _RIB_INDEX
        if len(set(mapping)) != 18:
            raise AssertionError(f"secondary mapping is not a permutation at occurrence {occurrence}")
        if any(row == secondary for row, secondary in enumerate(mapping)):
            raise AssertionError(f"secondary mapping contains a self-pair at occurrence {occurrence}")
        if mapping[_RIB_INDEX] == _WEB_INDEX or mapping[_WEB_INDEX] == _RIB_INDEX:
            raise AssertionError(f"secondary mapping contains forbidden rib/web pair at occurrence {occurrence}")
        stream.append(MACRO_CLASSES[mapping[primary_index]])
    return stream


def _split_for(context: str, context_index: int) -> str:
    cursor = 0
    for split, count in SPLIT_BY_CONTEXT_INDEX[context]:
        if cursor <= context_index < cursor + count:
            return split
        cursor += count
    raise IndexError((context, context_index))


__all__ = [
    "CONTEXT_QUOTAS",
    "PLAN_REVISION",
    "PlannedSample",
    "audit_plan",
    "build_plan",
    "compatibility_matrix",
    "load_plan",
    "sha256_file",
    "write_plan",
]
