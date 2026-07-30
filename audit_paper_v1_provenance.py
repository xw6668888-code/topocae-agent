from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import joblib


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_CONFIG = REPO_ROOT / "configs" / "paper_v1_clean_candidate" / "datasets.json"
DEFAULT_MODEL_CONFIG = REPO_ROOT / "configs" / "paper_v1_clean_candidate" / "models.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "paper_v1_p1"
SPLIT_NAMES = ("train", "val", "test")


class ExcludedArtifactError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit paper_v1 model provenance and the formal Synthetic split."
    )
    parser.add_argument("--dataset-config", type=Path, default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--validate-test-steps",
        action="store_true",
        help="Read every formal Synthetic-Test STEP with the established OCC solid validator.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def excluded_tokens(dataset_config: dict) -> tuple[str, ...]:
    return tuple(str(item).casefold() for item in dataset_config.get("excluded_tokens", []))


def assert_allowed_path(
    path: Path,
    dataset_config: dict,
    *,
    allow_legacy_metadata: bool = False,
) -> Path:
    resolved = path.resolve()
    lowered = str(resolved).replace("\\", "/").casefold()
    blocked = [token for token in excluded_tokens(dataset_config) if token in lowered]
    if not blocked:
        return resolved
    if allow_legacy_metadata:
        exceptions = {
            str(resolve_repo_path(value)).replace("\\", "/").casefold()
            for value in dataset_config.get("guard", {}).get("legacy_metadata_exception", [])
        }
        if lowered in exceptions:
            return resolved
    raise ExcludedArtifactError(
        f"P1 audit rejected excluded path before read: {resolved} ({blocked[0]})"
    )


def path_has_excluded_token(value: Any, dataset_config: dict) -> bool:
    lowered = str(value).replace("\\", "/").casefold()
    return any(token in lowered for token in excluded_tokens(dataset_config))


def load_split_ids(dataset_config: dict) -> dict[str, list[str]]:
    formal = dataset_config["formal_synthetic"]
    splits: dict[str, list[str]] = {}
    for split_name in SPLIT_NAMES:
        path = assert_allowed_path(resolve_repo_path(formal["splits"][split_name]), dataset_config)
        values = load_json(path)
        if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
            raise ValueError(f"Invalid split manifest: {path}")
        splits[split_name] = values
    return splits


def load_jsonl_sample_ids(path: Path) -> set[str]:
    sample_ids: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as stream:
        for line in stream:
            if not line.strip():
                continue
            value = json.loads(line)
            sample_id = value.get("sample_id")
            if sample_id is not None:
                sample_ids.add(str(sample_id))
    return sample_ids


def split_overlap_matrix(
    left: dict[str, Iterable[str]], right: dict[str, Iterable[str]]
) -> dict[str, int]:
    left_sets = {name: set(values) for name, values in left.items()}
    right_sets = {name: set(values) for name, values in right.items()}
    return {
        f"{left_name}_vs_{right_name}": len(left_ids & right_ids)
        for left_name, left_ids in left_sets.items()
        for right_name, right_ids in right_sets.items()
    }


def checkpoint_config_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.parent / "config.json"


def artifact_row(
    *,
    artifact: str,
    artifact_path: Path,
    training_dataset: str,
    source_graph_roots: list[str],
    truth_source: str,
    split: str,
    seed: int | str | None,
    contains_real_data: bool | str,
    contains_s3_derived_data: bool | str,
    allowed: bool | str,
    reason: str,
    evidence: list[str],
) -> dict:
    return {
        "artifact": artifact,
        "artifact_path": str(artifact_path.resolve()),
        "artifact_hash": sha256_file(artifact_path) if artifact_path.is_file() else None,
        "training_dataset": training_dataset,
        "source_graph_roots": source_graph_roots,
        "truth_source": truth_source,
        "split": split,
        "seed": seed,
        "contains_real_data": contains_real_data,
        "contains_s3_derived_data": contains_s3_derived_data,
        "allowed_in_clean_paper_baseline": allowed,
        "decision_reason": reason,
        "evidence": evidence,
        "status": "present" if artifact_path.is_file() else "missing",
    }


def audit_p0_artifacts(
    dataset_config: dict, model_config: dict, formal_splits: dict[str, list[str]]
) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    overlap_evidence: dict[str, Any] = {}

    gnn_path = assert_allowed_path(
        resolve_repo_path(model_config["frozen_gnn"]["path"]), dataset_config
    )
    source_checkpoint = assert_allowed_path(
        resolve_repo_path(model_config["frozen_gnn"]["source_checkpoint"]), dataset_config
    )
    gnn_config = load_json(checkpoint_config_path(gnn_path))
    source_config = load_json(checkpoint_config_path(source_checkpoint))
    rows.append(
        artifact_row(
            artifact="frozen_gnn",
            artifact_path=gnn_path,
            training_dataset="output_generation_v5_1_augmented_full_5400_clean",
            source_graph_roots=[
                str(resolve_repo_path(dataset_config["formal_synthetic"]["graph_root"]))
            ],
            truth_source="programmatic Synthetic graph labels and feature sets",
            split="formal Synthetic-Train; checkpoint selection on formal Synthetic-Val",
            seed=source_config.get("seed", gnn_config.get("seed", "unknown")),
            contains_real_data=False,
            contains_s3_derived_data=False,
            allowed=True,
            reason=(
                "The 24D frozen artifact is a deterministic compatibility expansion of the "
                "v5.1 checkpoint; its metrics file records best_epoch=0 and no additional "
                "Real/S3 fitting. The source checkpoint used the formal Synthetic split."
            ),
            evidence=[
                str(checkpoint_config_path(gnn_path).resolve()),
                str(checkpoint_config_path(source_checkpoint).resolve()),
                str(resolve_repo_path(dataset_config["formal_synthetic"]["dataset_meta"])),
            ],
        )
    )

    legacy = model_config["legacy_components"]
    candidate_path = assert_allowed_path(
        resolve_repo_path(legacy["candidate_classifier"]), dataset_config
    )
    candidate_bundle = joblib.load(candidate_path)
    candidate_metadata = candidate_bundle.get("metadata", {})
    candidate_dataset_dir = candidate_path.parents[2]
    candidate_splits = {
        split_name: load_jsonl_sample_ids(candidate_dataset_dir / f"{split_name}.jsonl")
        for split_name in SPLIT_NAMES
    }
    candidate_overlap = split_overlap_matrix(candidate_splits, formal_splits)
    overlap_evidence["legacy_candidate_vs_formal_gnn"] = candidate_overlap
    leaked_test_count = candidate_overlap.get("train_vs_test", 0)
    rows.append(
        artifact_row(
            artifact="legacy_candidate_topology_classifier",
            artifact_path=candidate_path,
            training_dataset=str(candidate_dataset_dir.resolve()),
            source_graph_roots=[str(candidate_metadata.get("graph_dir", "unknown"))],
            truth_source="Synthetic graph feature sets plus GNN/rule proposals",
            split="independent candidate split, seed 20260625",
            seed=candidate_bundle.get("config", {}).get("seed", "unknown"),
            contains_real_data=False,
            contains_s3_derived_data=False,
            allowed=False,
            reason=(
                f"Synthetic-only, but candidate training overlaps {leaked_test_count} formal "
                "GNN Synthetic-Test base samples because it used an independent split. It must "
                "be rebuilt on the formal split."
            ),
            evidence=[
                str((candidate_dataset_dir / "metadata.json").resolve()),
                "reports/paper_v1_p1/model_data_provenance.json#split_overlap_evidence",
            ],
        )
    )

    proposer_path = assert_allowed_path(
        resolve_repo_path(legacy["rib_candidate_proposer"]), dataset_config
    )
    proposer_bundle = joblib.load(proposer_path)
    rows.append(
        artifact_row(
            artifact="legacy_rib_candidate_proposer",
            artifact_path=proposer_path,
            training_dataset=str(proposer_path.parents[2].resolve()),
            source_graph_roots=["unknown from model metadata"],
            truth_source="S3-derived colored rib review, established by P0 supervisor audit",
            split="legacy full-train layout; val/test files empty",
            seed=proposer_bundle.get("config", {}).get("seed", "unknown"),
            contains_real_data=True,
            contains_s3_derived_data=True,
            allowed=False,
            reason=(
                "The P0 supervisor report records direct S3 graph and colored-truth references "
                "in the training records. Those records are not reopened by this audit."
            ),
            evidence=[str((REPO_ROOT / "reports" / "p0_supervisor_gate_review_full.zh-CN.md").resolve())],
        )
    )

    boundary_path = assert_allowed_path(
        resolve_repo_path(legacy["rib_boundary_classifier"]), dataset_config
    )
    boundary_bundle = joblib.load(boundary_path)
    boundary_metadata_path = boundary_path.parents[2] / "metadata.json"
    boundary_metadata_path = assert_allowed_path(
        boundary_metadata_path, dataset_config, allow_legacy_metadata=True
    )
    boundary_metadata = load_json(boundary_metadata_path)
    rows.append(
        artifact_row(
            artifact="legacy_rib_boundary_classifier",
            artifact_path=boundary_path,
            training_dataset=str(boundary_path.parents[2].resolve()),
            source_graph_roots=[str(boundary_metadata.get("graph_dir", "unknown"))],
            truth_source="Synthetic truth plus augmented S3-derived colored rib truth",
            split="legacy independent split, seed 20260629",
            seed=boundary_bundle.get("config", {}).get("seed", "unknown"),
            contains_real_data=True,
            contains_s3_derived_data=True,
            allowed=False,
            reason=(
                "Model metadata records 153 colored S3 records expanded to 3060 training records."
            ),
            evidence=[str(boundary_metadata_path.resolve())],
        )
    )
    return rows, overlap_evidence


def audit_clean_artifacts(dataset_config: dict, model_config: dict) -> list[dict]:
    rows: list[dict] = []
    clean = model_config.get("clean_components", {})
    for component_name, relative_path in clean.items():
        path = assert_allowed_path(resolve_repo_path(relative_path), dataset_config)
        if not path.is_file():
            rows.append(
                artifact_row(
                    artifact=f"clean_{component_name}",
                    artifact_path=path,
                    training_dataset="unknown",
                    source_graph_roots=["unknown"],
                    truth_source="unknown",
                    split="unknown",
                    seed="unknown",
                    contains_real_data="unknown",
                    contains_s3_derived_data="unknown",
                    allowed="unknown",
                    reason="Clean artifact has not been generated yet.",
                    evidence=[],
                )
            )
            continue
        bundle = joblib.load(path)
        metadata = bundle.get("metadata", {})
        provenance = metadata.get("paper_clean_provenance", {})
        real_status = provenance.get("contains_real_data", "unknown")
        s3_status = provenance.get("contains_s3_derived_data", "unknown")
        formal_split = provenance.get("formal_split_enforced", False)
        allowed = bool(real_status is False and s3_status is False and formal_split)
        rows.append(
            artifact_row(
                artifact=f"clean_{component_name}",
                artifact_path=path,
                training_dataset=str(bundle.get("config", {}).get("dataset_dir", path.parents[2])),
                source_graph_roots=[str(item) for item in provenance.get("source_graph_roots", ["unknown"])],
                truth_source=str(provenance.get("truth_source", "unknown")),
                split=str(provenance.get("split", "unknown")),
                seed=bundle.get("config", {}).get("seed", "unknown"),
                contains_real_data=real_status,
                contains_s3_derived_data=s3_status,
                allowed=allowed,
                reason=(
                    "Clean metadata proves formal-split Synthetic-only training."
                    if allowed
                    else "Clean provenance is incomplete or not compliant."
                ),
                evidence=[str(path.resolve())],
            )
        )
    return rows


def get_nested(mapping: dict, *keys: str, default: Any = None) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    return default if value is None else value


def finite_number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def node_attributes(node: dict) -> dict:
    return node.get("attributes", {}) if isinstance(node.get("attributes"), dict) else {}


def descriptor(node: dict, name: str) -> float:
    attrs = node_attributes(node)
    descriptors = attrs.get("shape_descriptors", {})
    specific = attrs.get("surface_specific", {})
    for source in (attrs, descriptors, specific, node):
        if isinstance(source, dict) and name in source:
            return finite_number(source.get(name))
    return 0.0


def graph_scale(graph: dict) -> float:
    values: list[float] = []
    dimensions = get_nested(graph, "stock_profile", "dimensions", default={})
    if isinstance(dimensions, dict):
        values.extend(finite_number(value) for value in dimensions.values())
    for node in graph.get("nodes", []) or []:
        values.extend(
            descriptor(node, name)
            for name in ("bbox_longest_extent", "bbox_middle_extent", "bbox_shortest_extent")
        )
    return max([value for value in values if value > 1e-12] or [1.0])


def node_geometry_token(node: dict, *, scale: float, ndigits: int) -> tuple:
    area_scale = scale * scale
    area = descriptor(node, "area") / area_scale
    extents = sorted(
        descriptor(node, name) / scale
        for name in ("bbox_longest_extent", "bbox_middle_extent", "bbox_shortest_extent")
    )
    spans = sorted(
        descriptor(node, name) / scale for name in ("uv_long_span", "uv_short_span")
    )
    return (
        int(node.get("primitive_surface_id", -1)),
        round(area, ndigits),
        round(descriptor(node, "radius") / scale, ndigits),
        round(descriptor(node, "semi_angle"), ndigits),
        *(round(value, ndigits) for value in extents),
        *(round(value, ndigits) for value in spans),
        int(round(descriptor(node, "vertex_count"))),
        int(round(descriptor(node, "edge_count"))),
    )


def edge_attribute_records(graph: dict) -> list[dict]:
    edges = graph.get("edges", []) or []
    attrs = graph.get("edge_attributes", []) or []
    records: list[dict] = []
    for index, edge in enumerate(edges):
        if not isinstance(edge, list) or len(edge) != 2:
            continue
        record = dict(attrs[index]) if index < len(attrs) and isinstance(attrs[index], dict) else {}
        record.setdefault("source_face", int(edge[0]))
        record.setdefault("target_face", int(edge[1]))
        records.append(record)
    return records


def edge_geometry_token(edge: dict, *, scale: float, ndigits: int) -> tuple:
    return (
        str(edge.get("curve_type", edge.get("curve_type_id", "unknown"))),
        round(finite_number(edge.get("length")) / scale, ndigits),
        str(edge.get("convexity", edge.get("convexity_id", "unknown"))),
        round(abs(finite_number(edge.get("dihedral_angle"))), ndigits),
        round(finite_number(edge.get("circle_radius")) / scale, ndigits),
    )


def rotation_invariant_graph_signature(graph: dict, *, ndigits: int) -> str:
    nodes = graph.get("nodes", []) or []
    scale = graph_scale(graph)
    labels = {
        int(node.get("idx", index)): stable_hash(node_geometry_token(node, scale=scale, ndigits=ndigits))
        for index, node in enumerate(nodes)
    }
    adjacency: dict[int, list[tuple[str, int]]] = defaultdict(list)
    edge_tokens: list[tuple] = []
    for edge in edge_attribute_records(graph):
        source = int(edge.get("source_face", -1))
        target = int(edge.get("target_face", -1))
        if source not in labels or target not in labels:
            continue
        token = edge_geometry_token(edge, scale=scale, ndigits=ndigits)
        edge_hash = stable_hash(token)
        adjacency[source].append((edge_hash, target))
        adjacency[target].append((edge_hash, source))
        edge_tokens.append(token)
    for _ in range(3):
        labels = {
            node_idx: stable_hash(
                [label, sorted((edge_hash, labels[neighbor]) for edge_hash, neighbor in adjacency[node_idx])]
            )
            for node_idx, label in labels.items()
        }
    stock_dimensions = get_nested(graph, "stock_profile", "dimensions", default={})
    normalized_dimensions = []
    if isinstance(stock_dimensions, dict):
        normalized_dimensions = sorted(
            round(finite_number(value) / scale, ndigits)
            for value in stock_dimensions.values()
            if finite_number(value) > 0
        )
    return stable_hash(
        {
            "node_labels": sorted(labels.values()),
            "edge_tokens": sorted(edge_tokens),
            "stock_dimensions": normalized_dimensions,
        }
    )


def base_geometry_key(graph: dict, sample_id: str) -> str:
    source = graph.get("merged_dataset_source", {})
    if isinstance(source, dict):
        source_root = source.get("source_root")
        source_sample = source.get("source_sample_id")
        if source_root and source_sample:
            return stable_hash([str(source_root).replace("\\", "/"), str(source_sample)])
    for key in ("base_sample_id", "rotation_source_sample_id", "parent_sample_id", "derived_from"):
        if graph.get(key):
            return stable_hash([key, str(graph[key])])
    return stable_hash(["sample", sample_id])


def explicit_rotation_key(graph: dict) -> str | None:
    candidates = (
        "rotation_source_sample_id",
        "base_sample_id",
        "parent_sample_id",
        "derived_from",
    )
    for key in candidates:
        value = graph.get(key)
        if value:
            return f"{key}:{value}"
    context = graph.get("generation_context", {})
    if isinstance(context, dict):
        for key in candidates:
            value = context.get(key)
            if value:
                return f"generation_context.{key}:{value}"
    return None


@dataclass
class AuditAccumulator:
    sample_count: int = 0
    face_count: int = 0
    feature_instance_count: int = 0
    graph_present_count: int = 0
    step_present_count: int = 0
    embedded_label_count: int = 0
    recorded_solid_valid_count: int = 0
    recorded_solid_invalid_count: int = 0
    empty_graph_count: int = 0
    corrupt_graph_count: int = 0
    invalid_instance_count: int = 0
    invalid_label_count: int = 0
    graph_step_face_mismatch_count: int = 0


def cross_split_collisions(
    values: dict[str, list[tuple[str, str]]]
) -> tuple[int, list[dict]]:
    by_key: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for split_name, items in values.items():
        for sample_id, key in items:
            by_key[key].append((split_name, sample_id))
    collisions = []
    for key, items in by_key.items():
        involved = sorted({split_name for split_name, _ in items})
        if len(involved) > 1:
            collisions.append(
                {
                    "signature": key,
                    "splits": involved,
                    "samples": [f"{split_name}:{sample_id}" for split_name, sample_id in sorted(items)],
                }
            )
    return len(collisions), collisions[:100]


def audit_dataset(
    dataset_config: dict,
    formal_splits: dict[str, list[str]],
    *,
    validate_test_steps: bool,
) -> dict:
    formal = dataset_config["formal_synthetic"]
    graph_root = assert_allowed_path(resolve_repo_path(formal["graph_root"]), dataset_config)
    step_root = assert_allowed_path(resolve_repo_path(formal["step_root"]), dataset_config)
    source_config_path = checkpoint_config_path(
        resolve_repo_path(
            "output_generation_v5_1_augmented_full_5400_clean/model_runs/edge_aware_v5_1_augmented_70ep/best.pt"
        )
    )
    checkpoint_label_spaces = load_json(source_config_path)["label_spaces"]
    allowed_macro_ids = {int(value) for value in checkpoint_label_spaces["macro_feature_id_to_key"]}
    allowed_semantic_ids = {int(value) for value in checkpoint_label_spaces["semantic_subtype_id_to_key"]}
    allowed_primitive_ids = {int(value) for value in checkpoint_label_spaces["primitive_surface_id_to_key"]}

    split_sets = {name: set(values) for name, values in formal_splits.items()}
    duplicate_ids = {
        f"{left}_vs_{right}": sorted(split_sets[left] & split_sets[right])
        for index, left in enumerate(SPLIT_NAMES)
        for right in SPLIT_NAMES[index + 1 :]
    }
    statistics: dict[str, Any] = {}
    base_keys: dict[str, list[tuple[str, str]]] = defaultdict(list)
    exact_signatures: dict[str, list[tuple[str, str]]] = defaultdict(list)
    near_signatures: dict[str, list[tuple[str, str]]] = defaultdict(list)
    rotation_keys: dict[str, list[tuple[str, str]]] = defaultdict(list)
    corruption_examples: list[dict] = []
    invalid_instance_examples: list[dict] = []
    invalid_label_examples: list[dict] = []
    first_graph_label_maps: dict | None = None

    for split_name in SPLIT_NAMES:
        accumulator = AuditAccumulator(sample_count=len(formal_splits[split_name]))
        face_macros: Counter[str] = Counter()
        instance_macros: Counter[str] = Counter()
        stock_profiles: Counter[str] = Counter()
        primitive_faces: Counter[str] = Counter()
        semantic_faces: Counter[str] = Counter()
        sample_macro_coverage: Counter[str] = Counter()
        for sample_id in formal_splits[split_name]:
            graph_path = assert_allowed_path(graph_root / f"{sample_id}.json", dataset_config)
            step_path = assert_allowed_path(step_root / f"{sample_id}.step", dataset_config)
            if graph_path.is_file():
                accumulator.graph_present_count += 1
            else:
                corruption_examples.append({"split": split_name, "sample_id": sample_id, "issue": "missing_graph"})
                continue
            if step_path.is_file():
                accumulator.step_present_count += 1
            try:
                graph = load_json(graph_path)
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                accumulator.corrupt_graph_count += 1
                corruption_examples.append(
                    {"split": split_name, "sample_id": sample_id, "issue": "corrupt_graph", "detail": str(error)}
                )
                continue
            nodes = graph.get("nodes", [])
            feature_sets = graph.get("feature_sets", [])
            if not isinstance(nodes, list) or not nodes:
                accumulator.empty_graph_count += 1
                corruption_examples.append({"split": split_name, "sample_id": sample_id, "issue": "empty_graph"})
                continue
            if first_graph_label_maps is None:
                first_graph_label_maps = graph.get("label_maps", {})
            accumulator.face_count += len(nodes)
            accumulator.feature_instance_count += len(feature_sets or [])
            accumulator.embedded_label_count += len(nodes)
            node_by_idx = {int(node.get("idx", index)): node for index, node in enumerate(nodes)}
            sample_macros: set[str] = set()
            for node in nodes:
                macro_id = int(node.get("macro_feature_id", -1))
                semantic_id = int(node.get("semantic_label", node.get("semantic_subtype_id", -1)))
                primitive_id = int(node.get("primitive_surface_id", -1))
                macro_name = str(node.get("macro_feature", f"id_{macro_id}"))
                semantic_name = str(node.get("semantic_subtype", f"id_{semantic_id}"))
                primitive_name = str(node.get("primitive_surface", f"id_{primitive_id}"))
                face_macros[macro_name] += 1
                semantic_faces[semantic_name] += 1
                primitive_faces[primitive_name] += 1
                if macro_id > 0:
                    sample_macros.add(macro_name)
                if (
                    macro_id not in allowed_macro_ids
                    or semantic_id not in allowed_semantic_ids
                    or primitive_id not in allowed_primitive_ids
                ):
                    accumulator.invalid_label_count += 1
                    if len(invalid_label_examples) < 100:
                        invalid_label_examples.append(
                            {
                                "split": split_name,
                                "sample_id": sample_id,
                                "face_idx": node.get("idx"),
                                "macro_id": macro_id,
                                "semantic_id": semantic_id,
                                "primitive_id": primitive_id,
                            }
                        )
            sample_macro_coverage.update(sample_macros)
            for feature_set in feature_sets or []:
                macro_name = str(feature_set.get("macro_feature", "unknown"))
                instance_macros[macro_name] += 1
                indices = [int(index) for index in feature_set.get("node_indices", [])]
                invalid_reasons = []
                if not indices:
                    invalid_reasons.append("empty_instance")
                missing = sorted(set(indices) - set(node_by_idx))
                if missing:
                    invalid_reasons.append("instance_face_missing")
                for face_idx in indices:
                    node = node_by_idx.get(face_idx)
                    if node is None:
                        continue
                    if int(node.get("macro_feature_id", -1)) != int(feature_set.get("macro_feature_id", -2)):
                        invalid_reasons.append("macro_mismatch")
                        break
                    if node.get("instance_id") != feature_set.get("instance_id"):
                        invalid_reasons.append("instance_id_mismatch")
                        break
                if invalid_reasons:
                    accumulator.invalid_instance_count += 1
                    if len(invalid_instance_examples) < 100:
                        invalid_instance_examples.append(
                            {
                                "split": split_name,
                                "sample_id": sample_id,
                                "instance_id": feature_set.get("instance_id"),
                                "issues": sorted(set(invalid_reasons)),
                            }
                        )
            quality = graph.get("generation_quality", {})
            step_validation = quality.get("step_write_validation", {}) if isinstance(quality, dict) else {}
            recorded_valid = bool(step_validation.get("valid")) and not step_validation.get("issues")
            if recorded_valid:
                accumulator.recorded_solid_valid_count += 1
            else:
                accumulator.recorded_solid_invalid_count += 1
            recorded_faces = step_validation.get("face_count")
            if recorded_faces is not None and int(recorded_faces) != len(nodes):
                accumulator.graph_step_face_mismatch_count += 1
            stock_profiles[str(get_nested(graph, "stock_profile", "stock_type", default="unknown"))] += 1
            base_keys[split_name].append((sample_id, base_geometry_key(graph, sample_id)))
            exact_signatures[split_name].append(
                (sample_id, rotation_invariant_graph_signature(graph, ndigits=6))
            )
            near_signatures[split_name].append(
                (sample_id, rotation_invariant_graph_signature(graph, ndigits=3))
            )
            rotation_key = explicit_rotation_key(graph)
            if rotation_key:
                rotation_keys[split_name].append((sample_id, rotation_key))

        expected_macros = {
            str(value)
            for key, value in checkpoint_label_spaces["macro_feature_id_to_key"].items()
            if int(key) in {1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 14, 20, 21, 24, 25, 26, 33, 35}
        }
        statistics[split_name] = {
            **accumulator.__dict__,
            "face_macro_distribution": dict(sorted(face_macros.items())),
            "feature_instance_distribution": dict(sorted(instance_macros.items())),
            "primitive_surface_distribution": dict(sorted(primitive_faces.items())),
            "semantic_subtype_distribution": dict(sorted(semantic_faces.items())),
            "samples_per_macro": dict(sorted(sample_macro_coverage.items())),
            "stock_profile_distribution": dict(sorted(stock_profiles.items())),
            "missing_target_macro_categories": sorted(expected_macros - set(instance_macros)),
        }

    base_collision_count, base_collision_examples = cross_split_collisions(base_keys)
    exact_collision_count, exact_collision_examples = cross_split_collisions(exact_signatures)
    near_collision_count, near_collision_examples = cross_split_collisions(near_signatures)
    rotation_collision_count, rotation_collision_examples = cross_split_collisions(rotation_keys)

    direct_step_validation: dict[str, Any] = {
        "requested": validate_test_steps,
        "status": "not_run",
        "step_count": 0,
        "issue_counts": {},
        "blocking_issue_count": None,
        "samples": [],
    }
    if validate_test_steps:
        from utils.step_solid_validation import inspect_step_file

        reports = []
        issue_counts: Counter[str] = Counter()
        for sample_id in formal_splits["test"]:
            step_path = assert_allowed_path(step_root / f"{sample_id}.step", dataset_config)
            report = inspect_step_file(step_path)
            report["sample_id"] = sample_id
            reports.append(report)
            issue_counts.update(report.get("issues", []))
        blocking = sum(
            count
            for issue, count in issue_counts.items()
            if issue in {"step_read_failed", "no_solid", "not_single_solid_body", "invalid_topology"}
        )
        direct_step_validation = {
            "requested": True,
            "status": "pass" if blocking == 0 else "fail",
            "step_count": len(reports),
            "issue_counts": dict(sorted(issue_counts.items())),
            "blocking_issue_count": blocking,
            "samples": reports,
        }

    mapping_hash = stable_hash(first_graph_label_maps or {})
    source_label_hash = stable_hash(checkpoint_label_spaces)
    gate_failures = []
    if any(duplicate_ids.values()):
        gate_failures.append("sample_id_cross_split_overlap")
    if base_collision_count:
        gate_failures.append("base_geometry_cross_split_overlap")
    if exact_collision_count:
        gate_failures.append("rotation_invariant_exact_graph_cross_split_overlap")
    if near_collision_count:
        gate_failures.append("rotation_invariant_near_duplicate_cross_split_overlap")
    if rotation_collision_count:
        gate_failures.append("explicit_rotation_family_cross_split_overlap")
    for split_name, split_stats in statistics.items():
        for key in (
            "corrupt_graph_count",
            "empty_graph_count",
            "invalid_instance_count",
            "invalid_label_count",
            "recorded_solid_invalid_count",
            "graph_step_face_mismatch_count",
        ):
            if split_stats[key]:
                gate_failures.append(f"{split_name}:{key}")
        if split_stats["graph_present_count"] != split_stats["sample_count"]:
            gate_failures.append(f"{split_name}:missing_graph")
        if split_stats["step_present_count"] != split_stats["sample_count"]:
            gate_failures.append(f"{split_name}:missing_step")
    if validate_test_steps and direct_step_validation["status"] != "pass":
        gate_failures.append("synthetic_test_direct_step_validation_failed")

    return {
        "schema": "topocae_paper_v1_dataset_audit_v1",
        "phase": "P1-A1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(resolve_repo_path(formal["root"])),
        "split_seed": formal.get("split_seed", "unknown"),
        "per_sample_generation_seed_status": "unknown_not_recorded",
        "split_statistics": statistics,
        "pairing": {
            "contract": "one embedded-label graph JSON and one STEP per split sample ID",
            "labels_embedded_in_graph": True,
        },
        "label_schema": {
            "checkpoint_label_spaces_hash": source_label_hash,
            "first_graph_label_maps_hash": mapping_hash,
            "invalid_label_count": sum(item["invalid_label_count"] for item in statistics.values()),
            "note": "Hashes cover different container layouts; validity is checked by every numeric ID against the checkpoint spaces.",
        },
        "leakage": {
            "sample_id_overlap": {key: len(value) for key, value in duplicate_ids.items()},
            "base_geometry_cross_split_collision_count": base_collision_count,
            "base_geometry_collision_examples": base_collision_examples,
            "rotation_invariant_exact_signature_cross_split_collision_count": exact_collision_count,
            "rotation_invariant_exact_collision_examples": exact_collision_examples,
            "rotation_invariant_near_signature_cross_split_collision_count": near_collision_count,
            "rotation_invariant_near_collision_examples": near_collision_examples,
            "explicit_rotation_family_cross_split_collision_count": rotation_collision_count,
            "explicit_rotation_family_collision_examples": rotation_collision_examples,
            "signature_method": (
                "three-round Weisfeiler-Lehman hash over rotation-invariant primitive, normalized "
                "face geometry, edge curve/length/convexity/dihedral, and normalized stock dimensions; "
                "6 digits exact and 3 digits near-duplicate screen"
            ),
        },
        "solid_validity": {
            "recorded_generation_gate": {
                split: {
                    "valid": stats["recorded_solid_valid_count"],
                    "invalid": stats["recorded_solid_invalid_count"],
                }
                for split, stats in statistics.items()
            },
            "direct_synthetic_test_validation": direct_step_validation,
        },
        "data_quality_examples": {
            "corruption": corruption_examples[:100],
            "invalid_instances": invalid_instance_examples,
            "invalid_labels": invalid_label_examples,
        },
        "gate": {
            "status": "pass" if not gate_failures else "fail",
            "failures": sorted(set(gate_failures)),
        },
    }


def write_provenance_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "artifact",
        "artifact_path",
        "artifact_hash",
        "training_dataset",
        "source_graph_roots",
        "truth_source",
        "split",
        "seed",
        "contains_real_data",
        "contains_s3_derived_data",
        "allowed_in_clean_paper_baseline",
        "decision_reason",
        "status",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            encoded = dict(row)
            encoded["source_graph_roots"] = " | ".join(row.get("source_graph_roots", []))
            writer.writerow({name: encoded.get(name) for name in fieldnames})


def write_dataset_markdown(path: Path, report: dict) -> None:
    lines = [
        "# paper_v1 Phase 1 数据集审计",
        "",
        f"- 数据根：`{report['dataset_root']}`",
        f"- split seed：`{report['split_seed']}`",
        f"- Gate：`{report['gate']['status']}`",
        f"- 单样本生成 seed：`{report['per_sample_generation_seed_status']}`（来源未记录时不猜测）",
        "",
        "## Split 统计",
        "",
        "| Split | 样本 | Face | Feature instance | STEP 配对 | 记录 solid 有效 | 非法实例 | 非法标签 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split_name in SPLIT_NAMES:
        stats = report["split_statistics"][split_name]
        lines.append(
            f"| {split_name} | {stats['sample_count']} | {stats['face_count']} | "
            f"{stats['feature_instance_count']} | {stats['step_present_count']} | "
            f"{stats['recorded_solid_valid_count']} | {stats['invalid_instance_count']} | "
            f"{stats['invalid_label_count']} |"
        )
    leakage = report["leakage"]
    lines.extend(
        [
            "",
            "## 泄漏检查",
            "",
            f"- sample ID 跨 split：`{sum(leakage['sample_id_overlap'].values())}`",
            f"- base geometry 跨 split：`{leakage['base_geometry_cross_split_collision_count']}`",
            f"- rotation-invariant exact signature 跨 split：`{leakage['rotation_invariant_exact_signature_cross_split_collision_count']}`",
            f"- rotation-invariant near signature 跨 split：`{leakage['rotation_invariant_near_signature_cross_split_collision_count']}`",
            f"- 显式 rotation family 跨 split：`{leakage['explicit_rotation_family_cross_split_collision_count']}`",
            f"- 方法：{leakage['signature_method']}。",
            "",
            "## Solid validity",
            "",
        ]
    )
    direct = report["solid_validity"]["direct_synthetic_test_validation"]
    if direct["requested"]:
        lines.append(
            f"正式 Synthetic-Test 共 `{direct['step_count']}` 个 STEP，OCC 直接检查结果："
            f"`{direct['status']}`；blocking issues=`{direct['blocking_issue_count']}`。"
        )
    else:
        lines.append("正式 Synthetic-Test 的 OCC 直接检查尚未运行；当前只有生成时质量门记录。")
    lines.extend(
        [
            "",
            "## 类别分布",
            "",
            "完整 face、instance、semantic、primitive、stock-profile 分布保存在机器可读审计 JSON 中。",
            "",
            "## 结论与边界",
            "",
            "- 标签直接嵌入 graph；审计逐 face 校验 primitive/macro/semantic ID 是否属于冻结 checkpoint schema。",
            "- 近重复检查不是文件名规则，而是使用 generation provenance 与 rotation-invariant graph signature。",
            "- 未读取任何被排除的 S3 几何、真值或预测。",
            "- Real-Dev 不参与本数据 split、训练或阈值选择。",
            "",
            f"机器可读报告：`{str((path.parent / 'paper_v1_p1' / 'dataset_audit.json').resolve())}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_config = load_json(args.dataset_config.resolve())
    model_config = load_json(args.model_config.resolve())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    formal_splits = load_split_ids(dataset_config)
    p0_rows, overlap_evidence = audit_p0_artifacts(dataset_config, model_config, formal_splits)
    clean_rows = audit_clean_artifacts(dataset_config, model_config)
    provenance = {
        "schema": "topocae_paper_v1_model_data_provenance_v1",
        "phase": "P1-A1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows": p0_rows + clean_rows,
        "split_overlap_evidence": overlap_evidence,
        "excluded_artifact_policy": {
            "tokens": list(excluded_tokens(dataset_config)),
            "legacy_metadata_only": dataset_config.get("guard", {}).get("legacy_metadata_exception", []),
            "s3_geometry_truth_prediction_read": False,
        },
    }
    write_json(output_dir / "model_data_provenance.json", provenance)
    write_provenance_csv(output_dir / "model_data_provenance.csv", provenance["rows"])

    dataset_report = audit_dataset(
        dataset_config,
        formal_splits,
        validate_test_steps=args.validate_test_steps,
    )
    write_json(output_dir / "dataset_audit.json", dataset_report)
    write_dataset_markdown(REPO_ROOT / "reports" / "paper_v1_dataset_audit.md", dataset_report)
    print(
        json.dumps(
            {
                "provenance_rows": len(provenance["rows"]),
                "dataset_gate": dataset_report["gate"],
                "dataset_report": str((output_dir / "dataset_audit.json").resolve()),
                "provenance_report": str((output_dir / "model_data_provenance.json").resolve()),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if dataset_report["gate"]["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
