from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from utils.candidate_topology_features import extract_candidate_topology_features
from utils.topology_postprocess import (
    connected_components_for_prediction,
    find_blind_hole_candidates,
    find_counterbore_candidates,
    find_countersink_candidates,
    find_cylinder_boss_candidates,
    find_fillet_stock_candidates,
    find_o_slot_candidates,
    find_rib_stock_candidates,
    find_u_slot_candidates,
)


DEFAULT_TARGET_MACROS = {
    "rib",
    "web",
    "slot_o",
    "slot_u",
    "slot_rect",
    "slot_v",
    "fillet",
    "obround_hole",
    "counterbore_hole",
    "countersink_hole",
    "cylinder_boss",
    "blind_hole_round",
    "pocket_circular",
    "through_hole_round",
}
REJECT_LABEL = "reject"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build candidate-level topology classifier data from labeled graphs and GNN predictions."
    )
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--target-macros", default=",".join(sorted(DEFAULT_TARGET_MACROS)))
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--min-iou", type=float, default=0.6)
    parser.add_argument("--min-both-coverage", type=float, default=0.75)
    parser.add_argument("--same-instance-threshold", type=float, default=0.5)
    parser.add_argument("--max-records-per-label", type=int, default=0)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")


def collect_json_files(root: Path, recursive: bool) -> list[Path]:
    globber = root.rglob if recursive else root.glob
    return sorted(path for path in globber("*.json") if path.is_file())


def parse_macro_set(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def build_candidate_records_for_graph(
    *,
    sample_id: str,
    graph_data: dict,
    prediction_data: dict | None,
    graph_path: Path,
    prediction_path: Path | None,
    target_macros: set[str],
    min_iou: float = 0.6,
    min_both_coverage: float = 0.75,
    same_instance_threshold: float = 0.5,
) -> list[dict]:
    truth_instances = build_truth_instances(graph_data, target_macros)
    proposals = []
    proposals.extend(
        {
            "source": "truth",
            "proposal_macro": instance["macro_feature"],
            "face_indices": instance["face_indices"],
        }
        for instance in truth_instances
    )
    if prediction_data:
        proposals.extend(
            build_prediction_component_proposals(
                graph_data,
                prediction_data,
                target_macros,
                same_instance_threshold=same_instance_threshold,
            )
        )
        proposals.extend(build_rule_proposals(graph_data, prediction_data, target_macros))

    records = []
    seen = set()
    for proposal in proposals:
        face_indices = sorted({int(idx) for idx in proposal["face_indices"]})
        if not face_indices:
            continue
        proposal_macro = proposal["proposal_macro"]
        key = (proposal_macro, tuple(face_indices))
        if key in seen:
            continue
        seen.add(key)
        label, match = label_candidate(
            face_indices,
            truth_instances,
            source=proposal["source"],
            proposal_macro=proposal_macro,
            min_iou=min_iou,
            min_both_coverage=min_both_coverage,
        )
        prediction_faces = prediction_data.get("faces", []) if prediction_data else []
        features = extract_candidate_topology_features(
            graph_data,
            face_indices,
            prediction_faces=prediction_faces,
            candidate_macro=proposal_macro,
        )
        features.update(proposal_macro_features(proposal_macro, target_macros))
        records.append({
            "sample_id": sample_id,
            "graph_path": str(graph_path),
            "prediction_path": str(prediction_path) if prediction_path else None,
            "source": proposal["source"],
            "proposal_macro": proposal_macro,
            "face_indices": face_indices,
            "label": label,
            "match": match,
            "features": features,
        })
    return records


def build_truth_instances(graph_data: dict, target_macros: set[str]) -> list[dict]:
    truth_instances = []
    for feature_set in graph_data.get("feature_sets", []) or []:
        macro_feature = feature_set.get("macro_feature")
        if macro_feature not in target_macros:
            continue
        face_indices = sorted(int(idx) for idx in feature_set.get("node_indices", []) if idx is not None)
        if not face_indices:
            continue
        truth_instances.append({
            "macro_feature": macro_feature,
            "face_indices": face_indices,
            "instance_id": feature_set.get("instance_id"),
        })
    return truth_instances


def build_prediction_component_proposals(
    graph_data: dict,
    prediction_data: dict,
    target_macros: set[str],
    *,
    same_instance_threshold: float,
) -> list[dict]:
    graph_for_components = dict(graph_data)
    graph_for_components["prediction_edge_same_instance"] = prediction_data.get("edge_same_instance", [])
    components = connected_components_for_prediction(
        prediction_data.get("faces", []),
        graph_for_components,
        same_instance_threshold=same_instance_threshold,
    )
    proposals = []
    for component in components:
        macro_feature = component.get("macro_feature")
        if macro_feature not in target_macros:
            continue
        proposals.append({
            "source": "prediction_component",
            "proposal_macro": macro_feature,
            "face_indices": component.get("face_indices", []),
        })
    return proposals


def build_rule_proposals(graph_data: dict, prediction_data: dict, target_macros: set[str]) -> list[dict]:
    faces = prediction_data.get("faces", [])
    repaired_faces: set[int] = set()
    finder_specs = [
        ("countersink_hole", find_countersink_candidates),
        ("counterbore_hole", find_counterbore_candidates),
        ("cylinder_boss", find_cylinder_boss_candidates),
        ("blind_hole_round", find_blind_hole_candidates),
        ("slot_u", find_u_slot_candidates),
        ("slot_o", find_o_slot_candidates),
        ("rib", find_rib_stock_candidates),
        ("fillet", find_fillet_stock_candidates),
    ]
    proposals = []
    for macro_feature, finder in finder_specs:
        if macro_feature not in target_macros:
            continue
        for face_indices in finder(faces, graph_data, repaired_faces):
            proposals.append({
                "source": "topology_rule",
                "proposal_macro": macro_feature,
                "face_indices": face_indices,
            })
    return proposals


def label_candidate(
    face_indices: list[int],
    truth_instances: list[dict],
    *,
    source: str,
    proposal_macro: str,
    min_iou: float,
    min_both_coverage: float,
) -> tuple[str, dict]:
    candidate_set = set(face_indices)
    if source == "truth":
        return proposal_macro, {
            "truth_macro": proposal_macro,
            "iou": 1.0,
            "candidate_coverage": 1.0,
            "truth_coverage": 1.0,
        }

    best = {
        "truth_macro": None,
        "iou": 0.0,
        "candidate_coverage": 0.0,
        "truth_coverage": 0.0,
    }
    for instance in truth_instances:
        truth_set = set(instance["face_indices"])
        if not truth_set:
            continue
        intersection = len(candidate_set & truth_set)
        if intersection == 0:
            continue
        union = len(candidate_set | truth_set)
        iou = intersection / union if union else 0.0
        candidate_coverage = intersection / len(candidate_set)
        truth_coverage = intersection / len(truth_set)
        if (iou, truth_coverage, candidate_coverage) > (
            best["iou"],
            best["truth_coverage"],
            best["candidate_coverage"],
        ):
            best = {
                "truth_macro": instance["macro_feature"],
                "iou": iou,
                "candidate_coverage": candidate_coverage,
                "truth_coverage": truth_coverage,
            }

    if best["iou"] >= min_iou or (
        best["candidate_coverage"] >= min_both_coverage and best["truth_coverage"] >= min_both_coverage
    ):
        return str(best["truth_macro"]), best
    return REJECT_LABEL, best


def proposal_macro_features(proposal_macro: str, target_macros: set[str]) -> dict[str, float]:
    return {
        f"proposal_macro_{macro}": 1.0 if proposal_macro == macro else 0.0
        for macro in sorted(target_macros)
    }


def split_candidate_records(
    records: list[dict],
    *,
    seed: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> dict[str, list[dict]]:
    splits = {"train": [], "val": [], "test": []}
    for record in records:
        bucket = stable_unit_interval(f"{seed}:{record['sample_id']}")
        if bucket < train_ratio:
            splits["train"].append(record)
        elif bucket < train_ratio + val_ratio:
            splits["val"].append(record)
        else:
            splits["test"].append(record)
    return splits


def stable_unit_interval(value: str) -> float:
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16 ** 12)


def limit_records_per_label(records: list[dict], max_records_per_label: int, seed: int) -> list[dict]:
    if max_records_per_label <= 0:
        return records
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_label[record["label"]].append(record)
    limited = []
    for label, label_records in sorted(by_label.items()):
        rng.shuffle(label_records)
        limited.extend(label_records[:max_records_per_label])
    limited.sort(key=lambda item: (item["sample_id"], item["source"], item["proposal_macro"], item["face_indices"]))
    return limited


def write_candidate_dataset(records: list[dict], splits: dict[str, list[dict]], output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_schema = sorted({key for record in records for key in record.get("features", {}).keys()})
    labels = sorted({record["label"] for record in records if record["label"] != REJECT_LABEL})
    label_to_id = {REJECT_LABEL: 0}
    for label in labels:
        label_to_id[label] = len(label_to_id)
    id_to_label = {str(label_id): label for label, label_id in label_to_id.items()}

    split_counts = {}
    label_counts = Counter(record["label"] for record in records)
    for split_name in ["train", "val", "test"]:
        split_records = []
        for record in splits.get(split_name, []):
            enriched = dict(record)
            enriched["label_id"] = int(label_to_id[record["label"]])
            enriched["feature_schema"] = feature_schema
            enriched["feature_vector"] = [
                float(record.get("features", {}).get(feature_name, 0.0))
                for feature_name in feature_schema
            ]
            split_records.append(enriched)
        split_counts[split_name] = len(split_records)
        with (output_dir / f"{split_name}.jsonl").open("w", encoding="utf-8") as fp:
            for record in split_records:
                fp.write(json.dumps(record, ensure_ascii=False) + "\n")

    metadata = {
        "schema": "candidate_topology_classifier_dataset_v1",
        "feature_schema": feature_schema,
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "record_count": len(records),
        "split_counts": split_counts,
        "label_counts": dict(sorted(label_counts.items())),
    }
    write_json(output_dir / "metadata.json", metadata)
    return metadata


def prediction_path_for_graph(graph_root: Path, graph_path: Path, prediction_root: Path) -> Path:
    try:
        relative = graph_path.relative_to(graph_root)
    except ValueError:
        relative = Path(graph_path.name)
    return prediction_root / relative


def build_dataset_from_paths(
    *,
    graph_dir: Path,
    prediction_dir: Path,
    output_dir: Path,
    recursive: bool,
    target_macros: set[str],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    min_iou: float,
    min_both_coverage: float,
    same_instance_threshold: float,
    max_records_per_label: int,
) -> dict:
    records = []
    skipped_predictions = 0
    for graph_path in collect_json_files(graph_dir, recursive):
        graph_data = load_json(graph_path)
        prediction_path = prediction_path_for_graph(graph_dir, graph_path, prediction_dir)
        prediction_data = None
        if prediction_path.exists():
            candidate_prediction = load_json(prediction_path)
            if isinstance(candidate_prediction.get("faces"), list):
                prediction_data = candidate_prediction
            else:
                skipped_predictions += 1
        else:
            skipped_predictions += 1
        sample_id = graph_path.stem
        records.extend(
            build_candidate_records_for_graph(
                sample_id=sample_id,
                graph_data=graph_data,
                prediction_data=prediction_data,
                graph_path=graph_path,
                prediction_path=prediction_path if prediction_data else None,
                target_macros=target_macros,
                min_iou=min_iou,
                min_both_coverage=min_both_coverage,
                same_instance_threshold=same_instance_threshold,
            )
        )

    records = limit_records_per_label(records, max_records_per_label, seed)
    splits = split_candidate_records(records, seed=seed, train_ratio=train_ratio, val_ratio=val_ratio)
    metadata = write_candidate_dataset(records, splits, output_dir)
    metadata["graph_dir"] = str(graph_dir)
    metadata["prediction_dir"] = str(prediction_dir)
    metadata["target_macros"] = sorted(target_macros)
    metadata["skipped_prediction_count"] = skipped_predictions
    write_json(output_dir / "metadata.json", metadata)
    return metadata


def main() -> None:
    args = parse_args()
    metadata = build_dataset_from_paths(
        graph_dir=args.graph_dir,
        prediction_dir=args.prediction_dir,
        output_dir=args.output_dir,
        recursive=args.recursive,
        target_macros=parse_macro_set(args.target_macros),
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        min_iou=args.min_iou,
        min_both_coverage=args.min_both_coverage,
        same_instance_threshold=args.same_instance_threshold,
        max_records_per_label=args.max_records_per_label,
    )
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "record_count": metadata["record_count"],
        "split_counts": metadata["split_counts"],
        "label_counts": metadata["label_counts"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
