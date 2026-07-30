from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from utils.rib_candidate_features import extract_rib_candidate_features
from utils.rib_candidate_hardcases import (
    build_colored_rib_hardcase_records,
    merge_rib_candidate_records,
)


RIB_CANDIDATE_LABEL = "rib_candidate"
NOT_RIB_CANDIDATE_LABEL = "not_rib_candidate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a face-level rib candidate proposer dataset from reviewed colored rib truth.")
    parser.add_argument("--review-template-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--root-dir", type=Path, default=Path("."))
    parser.add_argument("--colored-diagnostics-json", type=Path, default=None)
    parser.add_argument("--prediction-dir", type=Path, default=None)
    parser.add_argument("--hardcase-positive-weight", type=float, default=5.0)
    parser.add_argument("--hardcase-negative-weight", type=float, default=3.0)
    parser.add_argument("--max-hardcase-negatives-per-sample", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_records_from_review_template(template: dict, root_dir: Path) -> list[dict]:
    records = []
    for sample in template.get("samples", []) or []:
        sample_id = str(sample.get("sample_id") or Path(str(sample.get("graph_path", "sample"))).stem)
        graph_path = resolve_path(root_dir, sample.get("graph_path"))
        if graph_path is None or not graph_path.exists():
            continue
        graph_data = load_json(graph_path)
        prediction_by_idx = {
            int(candidate.get("face_idx")): candidate.get("current_prediction", {}) or {}
            for candidate in sample.get("candidate_faces", []) or []
            if candidate.get("face_idx") is not None
        }
        for candidate in sample.get("candidate_faces", []) or []:
            review = candidate.get("review", {}) or {}
            value = review.get("is_rib_face")
            if value is None:
                continue
            face_idx = int(candidate["face_idx"])
            label = RIB_CANDIDATE_LABEL if bool(value) else NOT_RIB_CANDIDATE_LABEL
            records.append(
                {
                    "sample_id": sample_id,
                    "graph_path": str(graph_path),
                    "source": "colored_rib_truth",
                    "label": label,
                    "candidate_face_idx": face_idx,
                    "rib_instance_id": review.get("rib_instance_id", ""),
                    "features": extract_rib_candidate_features(
                        graph_data,
                        face_idx,
                        prediction_face=prediction_by_idx.get(face_idx),
                    ),
                }
            )
    return records


def resolve_path(root_dir: Path, value) -> Path | None:
    if value is None:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (root_dir / path).resolve()


def split_rib_candidate_records(
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
    return ensure_nonempty_training_split(splits)


def ensure_nonempty_training_split(splits: dict[str, list[dict]]) -> dict[str, list[dict]]:
    if splits["train"]:
        return splits
    for split_name in ["val", "test"]:
        if splits[split_name]:
            splits["train"].append(splits[split_name].pop(0))
            return splits
    return splits


def stable_unit_interval(value: str) -> float:
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def write_rib_candidate_dataset(records: list[dict], splits: dict[str, list[dict]], output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_schema = sorted({name for record in records for name in record.get("features", {}).keys()})
    label_to_id = {NOT_RIB_CANDIDATE_LABEL: 0, RIB_CANDIDATE_LABEL: 1}
    id_to_label = {str(value): key for key, value in label_to_id.items()}
    metadata = {
        "schema": "rib_candidate_proposer_dataset_v1",
        "record_count": len(records),
        "feature_schema": feature_schema,
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=4, ensure_ascii=False), encoding="utf-8")
    for split_name, split_records in splits.items():
        with (output_dir / f"{split_name}.jsonl").open("w", encoding="utf-8") as fp:
            for record in split_records:
                encoded = dict(record)
                encoded["label_id"] = label_to_id[record["label"]]
                encoded["feature_schema"] = feature_schema
                encoded["feature_vector"] = [float(record.get("features", {}).get(name, 0.0)) for name in feature_schema]
                fp.write(json.dumps(encoded, ensure_ascii=False) + "\n")
    return metadata


def main() -> None:
    args = parse_args()
    template = load_json(args.review_template_json)
    records = build_records_from_review_template(template, args.root_dir)
    if args.colored_diagnostics_json is not None:
        if args.prediction_dir is None:
            raise ValueError("--prediction-dir is required when --colored-diagnostics-json is used")
        colored_diagnostics = load_json(args.colored_diagnostics_json)
        hard_records = build_colored_rib_hardcase_records(
            review_template=template,
            colored_diagnostics=colored_diagnostics,
            root_dir=args.root_dir,
            prediction_dir=args.prediction_dir,
            positive_weight=args.hardcase_positive_weight,
            negative_weight=args.hardcase_negative_weight,
            max_negatives_per_sample=args.max_hardcase_negatives_per_sample,
        )
        records = merge_rib_candidate_records(records, hard_records)
    if not records:
        raise RuntimeError(f"No reviewed rib candidate records found in {args.review_template_json}")
    splits = split_rib_candidate_records(
        records,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    metadata = write_rib_candidate_dataset(records, splits, args.output_dir)
    print(json.dumps(metadata, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()

