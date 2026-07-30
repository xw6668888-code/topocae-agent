from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GNN prediction JSON files against labeled graph JSON files.")
    parser.add_argument("--prediction-path", type=Path, required=True)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--recursive", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")


def collect_prediction_files(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        return [path]
    globber = path.rglob if recursive else path.glob
    return sorted(
        item for item in globber("*.json")
        if item.is_file() and item.name != "prediction_manifest.json"
    )


def resolve_graph_path(prediction_file: Path, prediction: dict, graph_dir: Path) -> Path:
    source_graph_path = prediction.get("source_graph_path")
    if source_graph_path and Path(source_graph_path).exists():
        return Path(source_graph_path)
    return graph_dir / prediction_file.name


def evaluate_prediction(prediction: dict, graph: dict) -> dict:
    pred_faces = {int(face["idx"]): face for face in prediction.get("faces", [])}
    graph_nodes = {int(node["idx"]): node for node in graph.get("nodes", [])}
    shared_face_indices = sorted(set(pred_faces) & set(graph_nodes))

    macro_pairs = []
    semantic_pairs = []
    for face_idx in shared_face_indices:
        pred_face = pred_faces[face_idx]
        graph_node = graph_nodes[face_idx]
        macro_pairs.append((
            int(pred_face.get("macro_feature_id", -1)),
            int(graph_node.get("macro_feature_id", -1)),
        ))
        semantic_pairs.append((
            int(pred_face.get("semantic_subtype_id", -1)),
            int(graph_node.get("semantic_label", -1)),
        ))

    macro_id_to_key = graph.get("label_maps", {}).get("macro_feature", {}).get("id_to_key", {})
    semantic_id_to_key = graph.get("label_maps", {}).get("face_semantic_subtype", {}).get("id_to_key", {})

    return {
        "sample_id": prediction.get("sample_id"),
        "face_count": len(shared_face_indices),
        "missing_prediction_faces": sorted(set(graph_nodes) - set(pred_faces)),
        "extra_prediction_faces": sorted(set(pred_faces) - set(graph_nodes)),
        "macro": compute_classification_metrics(macro_pairs, macro_id_to_key),
        "semantic": compute_classification_metrics(semantic_pairs, semantic_id_to_key),
    }


def compute_classification_metrics(pred_ref_pairs: list[tuple[int, int]], id_to_key: dict) -> dict:
    total = len(pred_ref_pairs)
    correct = sum(1 for pred, ref in pred_ref_pairs if pred == ref)
    class_ids = sorted({label_id for pair in pred_ref_pairs for label_id in pair if label_id >= 0})
    per_class = {}
    f1_scores = []

    for class_id in class_ids:
        tp = sum(1 for pred, ref in pred_ref_pairs if pred == class_id and ref == class_id)
        fp = sum(1 for pred, ref in pred_ref_pairs if pred == class_id and ref != class_id)
        fn = sum(1 for pred, ref in pred_ref_pairs if pred != class_id and ref == class_id)
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(2.0 * precision * recall, precision + recall)
        f1_scores.append(f1)
        per_class[label_name(class_id, id_to_key)] = {
            "id": class_id,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return {
        "accuracy": safe_divide(correct, total),
        "macro_f1": sum(f1_scores) / max(len(f1_scores), 1),
        "active_class_count": len(class_ids),
        "per_class": per_class,
    }


def aggregate_reports(sample_reports: list[dict]) -> dict:
    sample_count = len(sample_reports)
    face_count = sum(int(report.get("face_count", 0)) for report in sample_reports)
    return {
        "sample_count": sample_count,
        "face_count": face_count,
        "macro": aggregate_task_reports(sample_reports, "macro"),
        "semantic": aggregate_task_reports(sample_reports, "semantic"),
        "samples": sample_reports,
    }


def aggregate_task_reports(sample_reports: list[dict], task_name: str) -> dict:
    per_class_counts: dict[str, dict] = {}
    for report in sample_reports:
        for class_name, class_metrics in report.get(task_name, {}).get("per_class", {}).items():
            if class_name not in per_class_counts:
                per_class_counts[class_name] = {
                    "id": int(class_metrics.get("id", -1)),
                    "tp": 0,
                    "fp": 0,
                    "fn": 0,
                }
            per_class_counts[class_name]["tp"] += int(class_metrics.get("tp", 0))
            per_class_counts[class_name]["fp"] += int(class_metrics.get("fp", 0))
            per_class_counts[class_name]["fn"] += int(class_metrics.get("fn", 0))

    total_correct = sum(item["tp"] for item in per_class_counts.values())
    total_support = sum(item["tp"] + item["fn"] for item in per_class_counts.values())
    per_class = {}
    f1_scores = []
    weighted_f1_sum = 0.0
    for class_name, counts in sorted(per_class_counts.items(), key=lambda item: (item[1]["id"], item[0])):
        precision = safe_divide(counts["tp"], counts["tp"] + counts["fp"])
        recall = safe_divide(counts["tp"], counts["tp"] + counts["fn"])
        f1 = safe_divide(2.0 * precision * recall, precision + recall)
        support = counts["tp"] + counts["fn"]
        f1_scores.append(f1)
        weighted_f1_sum += f1 * support
        per_class[class_name] = {
            **counts,
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return {
        "mean_sample_accuracy": mean(report[task_name]["accuracy"] for report in sample_reports),
        "mean_sample_macro_f1": mean(report[task_name]["macro_f1"] for report in sample_reports),
        "global_accuracy": safe_divide(total_correct, total_support),
        "macro_f1": mean(f1_scores),
        "weighted_f1": safe_divide(weighted_f1_sum, total_support),
        "active_class_count": len(per_class),
        "per_class": per_class,
    }


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def label_name(label_id: int, id_to_key: dict) -> str:
    return id_to_key.get(str(label_id), f"id_{label_id}")


def main() -> None:
    args = parse_args()
    prediction_files = collect_prediction_files(args.prediction_path, args.recursive)
    if not prediction_files:
        raise RuntimeError(f"No prediction JSON files found under: {args.prediction_path}")

    sample_reports = []
    for prediction_file in prediction_files:
        prediction = load_json(prediction_file)
        graph_path = resolve_graph_path(prediction_file, prediction, args.graph_dir)
        graph = load_json(graph_path)
        report = evaluate_prediction(prediction, graph)
        report["prediction_path"] = str(prediction_file.resolve())
        report["graph_path"] = str(graph_path.resolve())
        sample_reports.append(report)

    output = aggregate_reports(sample_reports)
    write_json(args.output_path, output)
    print(f"prediction evaluation written: {args.output_path}")


if __name__ == "__main__":
    main()
