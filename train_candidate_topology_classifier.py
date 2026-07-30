from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a candidate-level topology-aware classifier.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", default="candidate_topology_classifier")
    parser.add_argument("--model-kind", choices=["random_forest", "mlp"], default="random_forest")
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=18)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden-layer-sizes", default="128,64")
    parser.add_argument("--hard-source-weight", type=float, default=4.0)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fp:
        return [json.loads(line) for line in fp if line.strip()]


def records_to_arrays(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    if not records:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    features = np.asarray([record["feature_vector"] for record in records], dtype=np.float32)
    labels = np.asarray([int(record["label_id"]) for record in records], dtype=np.int64)
    return features, labels


def parse_hidden_layer_sizes(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def build_estimator(
    *,
    model_kind: str,
    seed: int,
    n_estimators: int,
    max_depth: int | None,
    epochs: int,
    hidden_layer_sizes: tuple[int, ...],
):
    if model_kind == "random_forest":
        return Pipeline([
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=seed,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                ),
            ),
        ])
    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                activation="relu",
                solver="adam",
                alpha=1e-4,
                batch_size="auto",
                learning_rate_init=1e-3,
                max_iter=epochs,
                random_state=seed,
                early_stopping=True,
                n_iter_no_change=20,
            ),
        ),
    ])


def evaluate_split(estimator, records: list[dict], id_to_label: dict[str, str]) -> dict:
    if not records:
        return {
            "record_count": 0,
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "per_class": {},
            "confusion_matrix": [],
        }
    features, labels = records_to_arrays(records)
    predictions = estimator.predict(features)
    class_ids = sorted(int(class_id) for class_id in id_to_label.keys())
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=class_ids,
        zero_division=0,
    )
    per_class = {}
    for idx, class_id in enumerate(class_ids):
        label = id_to_label[str(class_id)]
        per_class[label] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }
    return {
        "record_count": len(records),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=class_ids, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, labels=class_ids, average="weighted", zero_division=0)),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(labels, predictions, labels=class_ids).tolist(),
    }


def evaluate_by_source(estimator, records: list[dict], id_to_label: dict[str, str]) -> dict:
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(record.get("source", "unknown"), []).append(record)
    return {
        source: evaluate_split(estimator, source_records, id_to_label)
        for source, source_records in sorted(grouped.items())
    }


def sample_weights_for_records(records: list[dict], hard_source_weight: float) -> np.ndarray:
    weights = []
    for record in records:
        if "sample_weight" in record:
            weights.append(float(record["sample_weight"]))
            continue
        source = record.get("source", "truth")
        weights.append(float(hard_source_weight if source != "truth" else 1.0))
    return np.asarray(weights, dtype=np.float32)


def feature_importance(estimator, feature_schema: list[str]) -> list[dict]:
    classifier = estimator.named_steps.get("classifier") if hasattr(estimator, "named_steps") else estimator
    importances = getattr(classifier, "feature_importances_", None)
    if importances is None:
        return []
    pairs = [
        {"feature": feature_schema[index], "importance": float(value)}
        for index, value in enumerate(importances)
    ]
    pairs.sort(key=lambda item: item["importance"], reverse=True)
    return pairs


def write_markdown_report(path: Path, metrics: dict, top_features: list[dict]) -> None:
    lines = ["# Candidate Topology Classifier Report", ""]
    for split_name in ["train", "val", "test"]:
        split_metrics = metrics.get(split_name, {})
        lines.append(f"## {split_name}")
        lines.append("")
        lines.append(f"- records: `{split_metrics.get('record_count', 0)}`")
        lines.append(f"- accuracy: `{split_metrics.get('accuracy', 0.0):.4f}`")
        lines.append(f"- macro-F1: `{split_metrics.get('macro_f1', 0.0):.4f}`")
        lines.append(f"- weighted-F1: `{split_metrics.get('weighted_f1', 0.0):.4f}`")
        lines.append("")
    if top_features:
        lines.append("## Top Features")
        lines.append("")
        for item in top_features[:30]:
            lines.append(f"- `{item['feature']}`: `{item['importance']:.6f}`")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def train_candidate_classifier(
    *,
    dataset_dir: Path,
    output_dir: Path,
    run_name: str,
    model_kind: str = "random_forest",
    seed: int = 20260625,
    n_estimators: int = 300,
    max_depth: int | None = 18,
    epochs: int = 200,
    hidden_layer_sizes: tuple[int, ...] = (128, 64),
    hard_source_weight: float = 4.0,
) -> dict:
    metadata = load_json(dataset_dir / "metadata.json")
    id_to_label = {str(key): value for key, value in metadata["id_to_label"].items()}
    feature_schema = list(metadata["feature_schema"])
    train_records = load_jsonl(dataset_dir / "train.jsonl")
    val_records = load_jsonl(dataset_dir / "val.jsonl")
    test_records = load_jsonl(dataset_dir / "test.jsonl")
    if not train_records:
        raise ValueError(f"No training records found in {dataset_dir / 'train.jsonl'}")

    train_features, train_labels = records_to_arrays(train_records)
    estimator = build_estimator(
        model_kind=model_kind,
        seed=seed,
        n_estimators=n_estimators,
        max_depth=max_depth,
        epochs=epochs,
        hidden_layer_sizes=hidden_layer_sizes,
    )
    sample_weights = sample_weights_for_records(train_records, hard_source_weight)
    try:
        estimator.fit(train_features, train_labels, classifier__sample_weight=sample_weights)
    except TypeError:
        estimator.fit(train_features, train_labels)

    metrics = {
        "train": evaluate_split(estimator, train_records, id_to_label),
        "val": evaluate_split(estimator, val_records, id_to_label),
        "test": evaluate_split(estimator, test_records, id_to_label),
        "val_by_source": evaluate_by_source(estimator, val_records, id_to_label),
        "test_by_source": evaluate_by_source(estimator, test_records, id_to_label),
    }
    top_features = feature_importance(estimator, feature_schema)
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "estimator": estimator,
            "metadata": metadata,
            "config": {
                "model_kind": model_kind,
                "seed": seed,
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "epochs": epochs,
                "hidden_layer_sizes": hidden_layer_sizes,
                "hard_source_weight": hard_source_weight,
            },
        },
        run_dir / "model.joblib",
    )
    result = {
        "dataset_dir": str(dataset_dir),
        "run_dir": str(run_dir),
        "metrics": metrics,
        "top_features": top_features,
    }
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=4, ensure_ascii=False), encoding="utf-8")
    write_markdown_report(run_dir / "report.md", metrics, top_features)
    return result


def main() -> None:
    args = parse_args()
    result = train_candidate_classifier(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        run_name=args.run_name,
        model_kind=args.model_kind,
        seed=args.seed,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth if args.max_depth > 0 else None,
        epochs=args.epochs,
        hidden_layer_sizes=parse_hidden_layer_sizes(args.hidden_layer_sizes),
        hard_source_weight=args.hard_source_weight,
    )
    print(json.dumps({
        "run_dir": result["run_dir"],
        "test_accuracy": result["metrics"]["test"]["accuracy"],
        "test_macro_f1": result["metrics"]["test"]["macro_f1"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

