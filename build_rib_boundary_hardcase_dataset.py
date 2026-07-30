from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_rib_boundary_classifier_dataset import (
    NOT_RIB_MEMBER_LABEL,
    RIB_MEMBER_LABEL,
    adjacent_faces,
    build_rib_truth_instances,
    candidate_to_seed_edges,
    collect_json_files,
    limit_records_per_label,
    load_json,
    make_record,
    split_rib_boundary_records,
    write_json,
    write_rib_boundary_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Augment a rib-boundary classifier dataset with harder topology boundary cases."
    )
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-dataset-dir", type=Path, default=None)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-contaminants", type=int, default=2)
    parser.add_argument("--max-records-per-label", type=int, default=0)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fp:
        return [json.loads(line) for line in fp if line.strip()]


def load_base_records(dataset_dir: Path | None) -> list[dict]:
    if dataset_dir is None:
        return []
    records: list[dict] = []
    for split_name in ["train", "val", "test"]:
        for record in load_jsonl(dataset_dir / f"{split_name}.jsonl"):
            cleaned = dict(record)
            for key in ["label_id", "feature_schema", "feature_vector"]:
                cleaned.pop(key, None)
            records.append(cleaned)
    return deduplicate_records(records)


def deduplicate_records(records: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for record in records:
        key = (
            record.get("sample_id"),
            record.get("source"),
            tuple(int(idx) for idx in record.get("seed_face_indices", [])),
            int(record.get("candidate_face_idx", -1)),
            record.get("label"),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)
    unique.sort(
        key=lambda item: (
            str(item.get("sample_id", "")),
            str(item.get("source", "")),
            list(item.get("seed_face_indices", [])),
            int(item.get("candidate_face_idx", -1)),
        )
    )
    return unique


def build_hardcase_records_for_graph(
    *,
    sample_id: str,
    graph_data: dict,
    graph_path: Path,
    max_contaminants: int = 2,
) -> list[dict]:
    records: list[dict] = []
    seen = set()
    for instance in build_rib_truth_instances(graph_data):
        rib_faces = sorted(instance["face_indices"])
        if len(rib_faces) < 2:
            continue
        rib_set = set(rib_faces)
        records.extend(
            build_fragment_positive_records(
                sample_id=sample_id,
                graph_data=graph_data,
                graph_path=graph_path,
                instance=instance,
                seen=seen,
            )
        )
        records.extend(
            build_contaminated_negative_records(
                sample_id=sample_id,
                graph_data=graph_data,
                graph_path=graph_path,
                instance=instance,
                rib_set=rib_set,
                max_contaminants=max_contaminants,
                seen=seen,
            )
        )
    records.sort(key=lambda item: (item["sample_id"], item["source"], item["seed_face_indices"], item["candidate_face_idx"]))
    return records


def build_fragment_positive_records(
    *,
    sample_id: str,
    graph_data: dict,
    graph_path: Path,
    instance: dict,
    seen: set,
) -> list[dict]:
    rib_faces = sorted(instance["face_indices"])
    rib_set = set(rib_faces)
    records = []
    for candidate_face_idx in rib_faces:
        seed_faces = sorted(
            face_idx
            for face_idx in rib_faces
            if face_idx != candidate_face_idx
            and candidate_to_seed_edges(graph_data, candidate_face_idx, {face_idx})
        )
        if len(seed_faces) > 2:
            seed_faces = seed_faces[:2]
        if not seed_faces:
            seed_faces = sorted(rib_set - {candidate_face_idx})[:2]
        if not seed_faces:
            continue
        key = ("hard_fragment_positive", tuple(seed_faces), candidate_face_idx)
        if key in seen:
            continue
        seen.add(key)
        records.append(
            make_hard_record(
                sample_id=sample_id,
                graph_path=graph_path,
                source="hard_fragment_positive",
                label=RIB_MEMBER_LABEL,
                seed_face_indices=seed_faces,
                candidate_face_idx=candidate_face_idx,
                graph_data=graph_data,
                instance=instance,
            )
        )
    return records


def build_contaminated_negative_records(
    *,
    sample_id: str,
    graph_data: dict,
    graph_path: Path,
    instance: dict,
    rib_set: set[int],
    max_contaminants: int,
    seen: set,
) -> list[dict]:
    non_rib_neighbors = sorted(adjacent_faces(graph_data, rib_set) - rib_set)
    if not non_rib_neighbors:
        return []
    records = []
    for candidate_face_idx in non_rib_neighbors:
        contaminants = [idx for idx in non_rib_neighbors if idx != candidate_face_idx]
        seed_faces = sorted(rib_set | set(contaminants[:max(0, max_contaminants)]))
        if not candidate_to_seed_edges(graph_data, candidate_face_idx, set(seed_faces)):
            continue
        key = ("hard_contaminated_negative", tuple(seed_faces), candidate_face_idx)
        if key in seen:
            continue
        seen.add(key)
        records.append(
            make_hard_record(
                sample_id=sample_id,
                graph_path=graph_path,
                source="hard_contaminated_negative",
                label=NOT_RIB_MEMBER_LABEL,
                seed_face_indices=seed_faces,
                candidate_face_idx=candidate_face_idx,
                graph_data=graph_data,
                instance=instance,
            )
        )
    return records


def make_hard_record(
    *,
    sample_id: str,
    graph_path: Path,
    source: str,
    label: str,
    seed_face_indices: list[int],
    candidate_face_idx: int,
    graph_data: dict,
    instance: dict,
) -> dict:
    record = make_record(
        sample_id=sample_id,
        graph_path=graph_path,
        source=source,
        label=label,
        seed_face_indices=seed_face_indices,
        candidate_face_idx=candidate_face_idx,
        graph_data=graph_data,
        instance=instance,
    )
    record["hardcase_family"] = source
    return record


def build_hardcase_records_from_paths(
    *,
    graph_dir: Path,
    recursive: bool,
    max_contaminants: int,
) -> tuple[list[dict], dict]:
    records = []
    graph_count = 0
    rib_graph_count = 0
    for graph_path in collect_json_files(graph_dir, recursive):
        graph_data = load_json(graph_path)
        graph_count += 1
        graph_records = build_hardcase_records_for_graph(
            sample_id=graph_path.stem,
            graph_data=graph_data,
            graph_path=graph_path,
            max_contaminants=max_contaminants,
        )
        if graph_records:
            rib_graph_count += 1
        records.extend(graph_records)
    return records, {"graph_count": graph_count, "rib_graph_count": rib_graph_count}


def build_augmented_dataset(
    *,
    graph_dir: Path,
    output_dir: Path,
    base_dataset_dir: Path | None,
    recursive: bool,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    max_contaminants: int,
    max_records_per_label: int,
) -> dict:
    base_records = load_base_records(base_dataset_dir)
    hard_records, stats = build_hardcase_records_from_paths(
        graph_dir=graph_dir,
        recursive=recursive,
        max_contaminants=max_contaminants,
    )
    records = deduplicate_records(base_records + hard_records)
    records = limit_records_per_label(records, max_records_per_label, seed)
    splits = split_rib_boundary_records(records, seed=seed, train_ratio=train_ratio, val_ratio=val_ratio)
    metadata = write_rib_boundary_dataset(records, splits, output_dir)
    metadata.update(
        {
            "schema": "rib_boundary_hardcase_classifier_dataset_v1",
            "graph_dir": str(graph_dir),
            "base_dataset_dir": str(base_dataset_dir) if base_dataset_dir else None,
            "base_record_count": len(base_records),
            "hardcase_record_count": len(hard_records),
            "max_contaminants": max_contaminants,
            **stats,
        }
    )
    write_json(output_dir / "metadata.json", metadata)
    return metadata


def main() -> None:
    args = parse_args()
    metadata = build_augmented_dataset(
        graph_dir=args.graph_dir,
        output_dir=args.output_dir,
        base_dataset_dir=args.base_dataset_dir,
        recursive=args.recursive,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        max_contaminants=args.max_contaminants,
        max_records_per_label=args.max_records_per_label,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "record_count": metadata["record_count"],
                "base_record_count": metadata["base_record_count"],
                "hardcase_record_count": metadata["hardcase_record_count"],
                "label_counts": metadata["label_counts"],
                "split_counts": metadata["split_counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
