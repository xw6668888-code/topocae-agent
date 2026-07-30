from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

from utils.instance_candidate_assembly import assemble_instance_candidates
from utils.instance_candidate_features import extract_instance_candidate_features
from utils.object_candidate_proposer import generate_hybrid_object_proposals


DEFAULT_TARGET_MACROS = {
    "rib",
    "web",
    "slot_o",
    "slot_u",
    "slot_rect",
    "slot_v",
    "fillet",
    "chamfer",
    "obround_hole",
    "counterbore_hole",
    "countersink_hole",
    "cylinder_boss",
    "rectangular_boss",
    "blind_hole_round",
    "pocket_rect",
    "pocket_circular",
    "through_hole_round",
}
REJECT_LABEL = "reject"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build proposal records for an instance candidate classifier."
    )
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--target-macros", default=",".join(sorted(DEFAULT_TARGET_MACROS)))
    parser.add_argument("--min-iou", type=float, default=0.50)
    parser.add_argument("--min-candidate-coverage", type=float, default=0.70)
    parser.add_argument("--min-truth-coverage", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--proposal-mode", choices=("same_instance", "hybrid"), default="same_instance")
    parser.add_argument("--source-macro-positive-clones", default="")
    parser.add_argument("--assembly-reason-positive-clones", default="")
    parser.add_argument("--reject-downweight-source-macros", default="")
    parser.add_argument("--default-sample-weight", type=float, default=1.0)
    parser.add_argument("--positive-nontruth-weight", type=float, default=3.0)
    parser.add_argument("--positive-clone-weight", type=float, default=5.0)
    parser.add_argument("--downweighted-reject-weight", type=float, default=0.35)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def parse_macro_set(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def parse_source_macro_pairs(value: str) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"source/macro item must use source:macro format, got {item!r}")
        source, macro = (part.strip() for part in item.split(":", 1))
        if not source or not macro:
            raise ValueError(f"source/macro item must use source:macro format, got {item!r}")
        pairs.add((source, macro))
    return pairs


def parse_assembly_reason_macro_pairs(value: str) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"assembly reason/macro item must use reason:macro format, got {item!r}")
        reason, macro = (part.strip() for part in item.split(":", 1))
        if not reason or not macro:
            raise ValueError(f"assembly reason/macro item must use reason:macro format, got {item!r}")
        pairs.add((reason, macro))
    return pairs


def collect_json_files(root: Path, recursive: bool) -> list[Path]:
    globber = root.rglob if recursive else root.glob
    return sorted(path for path in globber("*.json") if path.is_file())


def prediction_path_for_graph(graph_root: Path, graph_path: Path, prediction_root: Path) -> Path:
    try:
        relative = graph_path.relative_to(graph_root)
    except ValueError:
        relative = Path(graph_path.name)
    return prediction_root / relative


def build_truth_instances(graph_data: dict[str, Any], target_macros: set[str]) -> list[dict[str, Any]]:
    truth_instances: list[dict[str, Any]] = []
    for feature_set in graph_data.get("feature_sets", []) or []:
        if not isinstance(feature_set, dict):
            continue
        macro = feature_set.get("macro_feature") or feature_set.get("macro")
        if macro not in target_macros:
            continue
        raw_indices = (
            feature_set.get("node_indices")
            or feature_set.get("face_indices")
            or feature_set.get("faces")
            or []
        )
        face_indices = sorted({int(idx) for idx in raw_indices if idx is not None})
        if not face_indices:
            continue
        truth_instances.append(
            {
                "macro_feature": str(macro),
                "face_indices": face_indices,
                "instance_id": feature_set.get("instance_id"),
            }
        )
    return truth_instances


def build_source_macro_positive_clone_proposals(
    truth_instances: list[dict[str, Any]],
    source_macro_pairs: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    proposals: list[dict[str, Any]] = []
    for instance in truth_instances:
        macro = str(instance["macro_feature"])
        face_indices = list(instance["face_indices"])
        for source, pair_macro in sorted(source_macro_pairs):
            if pair_macro != macro:
                continue
            proposals.append(
                {
                    "source": source,
                    "proposal_sources": [source],
                    "proposal_macro": macro,
                    "face_indices": face_indices,
                    "augmentation": "source_macro_positive_clone",
                    "assembly_reasons": [f"source_macro_positive_clone:{source}:{macro}"],
                }
            )
    return proposals


def build_assembly_reason_positive_clone_proposals(
    truth_instances: list[dict[str, Any]],
    reason_macro_pairs: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    proposals: list[dict[str, Any]] = []
    for instance in truth_instances:
        macro = str(instance["macro_feature"])
        face_indices = list(instance["face_indices"])
        for reason, pair_macro in sorted(reason_macro_pairs):
            if pair_macro != macro:
                continue
            proposals.append(
                {
                    "source": "topology_rule",
                    "proposal_sources": ["topology_rule"],
                    "proposal_macro": macro,
                    "face_indices": face_indices,
                    "augmentation": "assembly_reason_positive_clone",
                    "assembly_reasons": [reason],
                }
            )
    return proposals


def build_proposals(
    graph_data: dict[str, Any],
    prediction_data: dict[str, Any] | None,
    truth_instances: list[dict[str, Any]],
    target_macros: set[str],
    *,
    proposal_mode: str = "same_instance",
    source_macro_positive_clone_pairs: set[tuple[str, str]] | None = None,
    assembly_reason_positive_clone_pairs: set[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    proposals: list[dict[str, Any]] = []
    for instance in truth_instances:
        proposals.append(
            {
                "source": "truth",
                "proposal_macro": instance["macro_feature"],
                "face_indices": list(instance["face_indices"]),
            }
        )

    clone_pairs = source_macro_positive_clone_pairs or set()
    proposals.extend(build_source_macro_positive_clone_proposals(truth_instances, clone_pairs))

    proposals.extend(
        build_assembly_reason_positive_clone_proposals(
            truth_instances,
            assembly_reason_positive_clone_pairs or set(),
        )
    )

    if prediction_data:
        same_instance_candidates = list(
            assemble_instance_candidates(
                graph_data,
                prediction_data,
                target_macros=target_macros,
            )
        )
        if proposal_mode == "hybrid":
            candidate_iterable = generate_hybrid_object_proposals(
                graph_data,
                prediction_data,
                same_instance_proposals=same_instance_candidates,
            )
        else:
            candidate_iterable = same_instance_candidates

        for candidate in candidate_iterable:
            proposal_macro = str(candidate.get("proposal_macro", "unknown_feature"))
            if proposal_macro not in target_macros:
                continue
            proposal = dict(candidate)
            if proposal_mode == "hybrid":
                proposal["source"] = str(proposal.get("source", "hybrid_candidate"))
            else:
                proposal["source"] = "assembled_instance_candidate"
                proposal["assembly_source"] = candidate.get("source")
            proposal["proposal_macro"] = proposal_macro
            proposals.append(proposal)

    return proposals


def label_candidate(
    face_indices: list[int],
    truth_instances: list[dict[str, Any]],
    *,
    source: str,
    proposal_macro: str,
    min_iou: float,
    min_candidate_coverage: float,
    min_truth_coverage: float,
) -> tuple[str, dict[str, Any]]:
    candidate_set = set(face_indices)
    if source == "truth":
        return proposal_macro, {
            "truth_macro": proposal_macro,
            "truth_face_indices": sorted(candidate_set),
            "iou": 1.0,
            "candidate_coverage": 1.0,
            "truth_coverage": 1.0,
        }

    best_match = {
        "truth_macro": None,
        "truth_face_indices": [],
        "iou": 0.0,
        "candidate_coverage": 0.0,
        "truth_coverage": 0.0,
    }
    for truth in truth_instances:
        truth_set = set(truth["face_indices"])
        if not candidate_set or not truth_set:
            continue
        intersection = len(candidate_set & truth_set)
        union = len(candidate_set | truth_set)
        if union == 0:
            continue
        iou = intersection / union
        candidate_coverage = intersection / len(candidate_set)
        truth_coverage = intersection / len(truth_set)
        score = (iou, candidate_coverage, truth_coverage)
        best_score = (
            best_match["iou"],
            best_match["candidate_coverage"],
            best_match["truth_coverage"],
        )
        if score > best_score:
            best_match = {
                "truth_macro": truth["macro_feature"],
                "truth_face_indices": sorted(truth_set),
                "iou": iou,
                "candidate_coverage": candidate_coverage,
                "truth_coverage": truth_coverage,
            }

    if (
        best_match["truth_macro"] is not None
        and best_match["iou"] >= min_iou
        and best_match["candidate_coverage"] >= min_candidate_coverage
        and best_match["truth_coverage"] >= min_truth_coverage
    ):
        return str(best_match["truth_macro"]), best_match
    return REJECT_LABEL, best_match


def assign_candidate_sample_weight(
    *,
    source: str,
    proposal_macro: str,
    label: str,
    augmentation: str | None,
    source_macro_positive_clone_pairs: set[tuple[str, str]],
    reject_downweight_source_macro_pairs: set[tuple[str, str]],
    default_sample_weight: float,
    positive_nontruth_weight: float,
    positive_clone_weight: float,
    downweighted_reject_weight: float,
) -> tuple[float, str]:
    source_macro = (source, proposal_macro)
    if label == REJECT_LABEL and source_macro in reject_downweight_source_macro_pairs:
        return float(downweighted_reject_weight), "downweighted_reject"
    if label != REJECT_LABEL and augmentation == "assembly_reason_positive_clone":
        return float(positive_clone_weight), "positive_assembly_reason_clone"
    if label != REJECT_LABEL and augmentation == "source_macro_positive_clone":
        return float(positive_clone_weight), "positive_source_macro_clone"
    if label != REJECT_LABEL and source != "truth":
        if source_macro in source_macro_positive_clone_pairs:
            return float(positive_clone_weight), "positive_source_macro_pair"
        return float(positive_nontruth_weight), "positive_nontruth"
    return float(default_sample_weight), "default"


def build_candidate_records_for_graph(
    *,
    sample_id: str,
    graph_data: dict[str, Any],
    prediction_data: dict[str, Any] | None,
    graph_path: Path,
    prediction_path: Path | None,
    target_macros: set[str],
    min_iou: float,
    min_candidate_coverage: float,
    min_truth_coverage: float,
    proposal_mode: str = "same_instance",
    source_macro_positive_clone_pairs: set[tuple[str, str]] | None = None,
    assembly_reason_positive_clone_pairs: set[tuple[str, str]] | None = None,
    reject_downweight_source_macro_pairs: set[tuple[str, str]] | None = None,
    default_sample_weight: float = 1.0,
    positive_nontruth_weight: float = 3.0,
    positive_clone_weight: float = 5.0,
    downweighted_reject_weight: float = 0.35,
) -> list[dict[str, Any]]:
    truth_instances = build_truth_instances(graph_data, target_macros)
    proposals = build_proposals(
        graph_data,
        prediction_data,
        truth_instances,
        target_macros,
        proposal_mode=proposal_mode,
        source_macro_positive_clone_pairs=source_macro_positive_clone_pairs or set(),
        assembly_reason_positive_clone_pairs=assembly_reason_positive_clone_pairs or set(),
    )
    prediction_faces = prediction_data.get("faces", []) if prediction_data else []

    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str, tuple[int, ...], str]] = set()
    for proposal in proposals:
        face_indices = sorted({int(idx) for idx in proposal.get("face_indices", [])})
        if not face_indices:
            continue
        proposal_macro = str(proposal.get("proposal_macro", "unknown_feature"))
        source = str(proposal.get("source", "unknown"))
        augmentation = str(proposal.get("augmentation") or "")
        key = (source, proposal_macro, tuple(face_indices), augmentation)
        if key in seen:
            continue
        seen.add(key)

        label, match = label_candidate(
            face_indices,
            truth_instances,
            source=source,
            proposal_macro=proposal_macro,
            min_iou=min_iou,
            min_candidate_coverage=min_candidate_coverage,
            min_truth_coverage=min_truth_coverage,
        )
        augmentation = proposal.get("augmentation")
        sample_weight, rebalance_action = assign_candidate_sample_weight(
            source=source,
            proposal_macro=proposal_macro,
            label=label,
            augmentation=str(augmentation) if augmentation is not None else None,
            source_macro_positive_clone_pairs=source_macro_positive_clone_pairs or set(),
            reject_downweight_source_macro_pairs=reject_downweight_source_macro_pairs or set(),
            default_sample_weight=default_sample_weight,
            positive_nontruth_weight=positive_nontruth_weight,
            positive_clone_weight=positive_clone_weight,
            downweighted_reject_weight=downweighted_reject_weight,
        )
        features = extract_instance_candidate_features(
            graph_data,
            proposal,
            prediction_faces=prediction_faces,
            candidate_macro=proposal_macro,
        )
        record = {
            "sample_id": sample_id,
            "graph_path": str(graph_path),
            "prediction_path": str(prediction_path) if prediction_path else None,
            "source": source,
            "proposal_macro": proposal_macro,
            "face_indices": face_indices,
            "label": label,
            "match": match,
            "features": features,
            "sample_weight": sample_weight,
            "rebalance_action": rebalance_action,
        }
        if proposal.get("assembly_source") is not None:
            record["assembly_source"] = proposal["assembly_source"]
        if proposal.get("proposal_sources") is not None:
            record["proposal_sources"] = list(proposal["proposal_sources"])
        if proposal.get("assembly_reasons") is not None:
            record["assembly_reasons"] = list(proposal["assembly_reasons"])
        if augmentation is not None:
            record["augmentation"] = str(augmentation)
        records.append(record)
    return records


def split_records(
    records: list[dict[str, Any]],
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> dict[str, list[dict[str, Any]]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must satisfy 0 < train_ratio < 1")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must satisfy 0 <= val_ratio < 1")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1")

    shuffled = list(records)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    train_count = int(len(shuffled) * train_ratio)
    val_count = int(len(shuffled) * val_ratio)
    test_count = len(shuffled) - train_count - val_count
    if train_count <= 0 or test_count <= 0:
        raise ValueError("split requires non-empty train and test records; val may be empty")
    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
        "test": shuffled[train_count + val_count :],
    }

def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")



def _finite_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _record_features(record: dict[str, Any]) -> dict[str, Any]:
    features = record.get("features", {})
    return features if isinstance(features, dict) else {}


def summarize_sample_weights(records: list[dict[str, Any]]) -> dict[str, float]:
    weights = [_finite_float(record.get("sample_weight", 1.0)) for record in records]
    if not weights:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": float(min(weights)),
        "max": float(max(weights)),
        "mean": float(sum(weights) / len(weights)),
    }


def _validate_dataset_inputs(records: list[dict[str, Any]], feature_schema: list[str]) -> None:
    record_count = len(records)
    if record_count <= 0:
        raise ValueError("record_count must be > 0")
    if not feature_schema:
        raise ValueError("feature_schema must be non-empty")

    labels = [record.get("label") for record in records]
    if REJECT_LABEL not in labels:
        raise ValueError("dataset must contain at least one reject label")
    if not any(label and label != REJECT_LABEL for label in labels):
        raise ValueError("dataset must contain at least one non-reject label")
def write_dataset_outputs(
    records: list[dict[str, Any]],
    output_dir: Path,
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    target_macros: set[str],
    skipped_prediction_count: int,
    proposal_mode: str = "same_instance",
) -> dict[str, Any]:
    feature_schema = sorted({key for record in records for key in _record_features(record)})
    _validate_dataset_inputs(records, feature_schema)
    labels = sorted({record["label"] for record in records if record["label"] != REJECT_LABEL})
    label_to_id = {REJECT_LABEL: 0}
    for label in labels:
        label_to_id[label] = len(label_to_id)
    enriched_records = [_enrich_record(record, feature_schema, label_to_id) for record in records]
    splits = split_records(
        enriched_records,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "records.jsonl", enriched_records)
    for split_name in ("train", "val", "test"):
        write_jsonl(output_dir / f"{split_name}.jsonl", splits[split_name])

    feature_schema_data = {
        "schema": "instance_candidate_feature_schema_v1",
        "features": feature_schema,
    }
    write_json(output_dir / "feature_schema.json", feature_schema_data)
    metadata = {
        "schema": "instance_candidate_classifier_metadata_v1",
        "feature_schema": feature_schema,
        "label_to_id": label_to_id,
        "id_to_label": {str(class_id): label for label, class_id in label_to_id.items()},
    }
    write_json(output_dir / "metadata.json", metadata)

    label_counts = Counter(record["label"] for record in enriched_records)
    source_counts = Counter(record["source"] for record in enriched_records)
    rebalance_action_counts = Counter(record.get("rebalance_action", "default") for record in enriched_records)
    source_macro_positive_clone_count = sum(
        1 for record in enriched_records
        if record.get("augmentation") == "source_macro_positive_clone"
    )
    assembly_reason_positive_clone_count = sum(
        1 for record in enriched_records
        if record.get("augmentation") == "assembly_reason_positive_clone"
    )
    summary = {
        "schema": "instance_candidate_dataset_v1",
        "record_count": len(enriched_records),
        "split_counts": {name: len(items) for name, items in splits.items()},
        "label_counts": dict(sorted(label_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "label_to_id": label_to_id,
        "target_macros": sorted(target_macros),
        "feature_schema_path": str(output_dir / "feature_schema.json"),
        "skipped_prediction_count": skipped_prediction_count,
        "proposal_mode": proposal_mode,
        "rebalance_action_counts": dict(sorted(rebalance_action_counts.items())),
        "source_macro_positive_clone_count": int(source_macro_positive_clone_count),
        "assembly_reason_positive_clone_count": int(assembly_reason_positive_clone_count),
        "sample_weight_stats": summarize_sample_weights(enriched_records),
    }
    write_json(output_dir / "dataset_summary.json", summary)
    return summary


def _enrich_record(
    record: dict[str, Any],
    feature_schema: list[str],
    label_to_id: dict[str, int],
) -> dict[str, Any]:
    enriched = dict(record)
    raw_features = _record_features(record)
    features = {
        feature_name: _finite_float(raw_features.get(feature_name, 0.0))
        for feature_name in feature_schema
    }
    enriched["features"] = features
    enriched["label_id"] = int(label_to_id[record["label"]])
    enriched["feature_vector"] = [features[feature_name] for feature_name in feature_schema]
    return enriched

def build_dataset_from_paths(
    *,
    graph_dir: Path,
    prediction_dir: Path,
    output_dir: Path,
    recursive: bool,
    target_macros: set[str],
    min_iou: float,
    min_candidate_coverage: float,
    min_truth_coverage: float,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    proposal_mode: str = "same_instance",
    source_macro_positive_clone_pairs: set[tuple[str, str]] | None = None,
    assembly_reason_positive_clone_pairs: set[tuple[str, str]] | None = None,
    reject_downweight_source_macro_pairs: set[tuple[str, str]] | None = None,
    default_sample_weight: float = 1.0,
    positive_nontruth_weight: float = 3.0,
    positive_clone_weight: float = 5.0,
    downweighted_reject_weight: float = 0.35,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    skipped_predictions = 0
    for graph_path in collect_json_files(graph_dir, recursive):
        graph_data = load_json(graph_path)
        prediction_path = prediction_path_for_graph(graph_dir, graph_path, prediction_dir)
        prediction_data = None
        if prediction_path.exists():
            prediction_data = load_json(prediction_path)
        else:
            skipped_predictions += 1
        sample_id = str(graph_data.get("sample_id") or graph_path.stem)
        records.extend(
            build_candidate_records_for_graph(
                sample_id=sample_id,
                graph_data=graph_data,
                prediction_data=prediction_data,
                graph_path=graph_path,
                prediction_path=prediction_path if prediction_data else None,
                target_macros=target_macros,
                min_iou=min_iou,
                min_candidate_coverage=min_candidate_coverage,
                min_truth_coverage=min_truth_coverage,
                proposal_mode=proposal_mode,
                source_macro_positive_clone_pairs=source_macro_positive_clone_pairs or set(),
                assembly_reason_positive_clone_pairs=assembly_reason_positive_clone_pairs or set(),
                reject_downweight_source_macro_pairs=reject_downweight_source_macro_pairs or set(),
                default_sample_weight=default_sample_weight,
                positive_nontruth_weight=positive_nontruth_weight,
                positive_clone_weight=positive_clone_weight,
                downweighted_reject_weight=downweighted_reject_weight,
            )
        )

    summary = write_dataset_outputs(
        records,
        output_dir,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        target_macros=target_macros,
        skipped_prediction_count=skipped_predictions,
        proposal_mode=proposal_mode,
    )
    summary["graph_dir"] = str(graph_dir)
    summary["prediction_dir"] = str(prediction_dir)
    write_json(output_dir / "dataset_summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = build_dataset_from_paths(
        graph_dir=args.graph_dir,
        prediction_dir=args.prediction_dir,
        output_dir=args.output_dir,
        recursive=args.recursive,
        target_macros=parse_macro_set(args.target_macros),
        min_iou=args.min_iou,
        min_candidate_coverage=args.min_candidate_coverage,
        min_truth_coverage=args.min_truth_coverage,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        proposal_mode=args.proposal_mode,
        source_macro_positive_clone_pairs=parse_source_macro_pairs(args.source_macro_positive_clones),
        assembly_reason_positive_clone_pairs=parse_assembly_reason_macro_pairs(args.assembly_reason_positive_clones),
        reject_downweight_source_macro_pairs=parse_source_macro_pairs(args.reject_downweight_source_macros),
        default_sample_weight=args.default_sample_weight,
        positive_nontruth_weight=args.positive_nontruth_weight,
        positive_clone_weight=args.positive_clone_weight,
        downweighted_reject_weight=args.downweighted_reject_weight,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "record_count": summary["record_count"],
                "split_counts": summary["split_counts"],
                "label_counts": summary["label_counts"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()



