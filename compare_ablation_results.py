from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two CAE STEP recognition ablation experiment outputs.")
    parser.add_argument("--single-dataset-root", type=Path, required=True)
    parser.add_argument("--single-run-name", required=True)
    parser.add_argument("--multi-dataset-root", type=Path, required=True)
    parser.add_argument("--multi-run-name", required=True)
    parser.add_argument("--real-report-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def synthetic_summary(dataset_root: Path, run_name: str, eval_name: str) -> dict[str, Any]:
    metrics = load_json(dataset_root / "model_runs" / run_name / "metrics.json")
    prediction_eval = load_json(dataset_root / "reports" / eval_name)
    generation = load_json(dataset_root / "reports" / "ablation_generation_report.json")
    topology = load_json(dataset_root / "reports" / "topology_rule_audit.json")
    final_test = metrics["final_test_metrics"]
    return {
        "dataset_root": str(dataset_root.resolve()),
        "run_name": run_name,
        "sample_count": generation["sample_count"],
        "macro_counts": generation["macro_counts"],
        "reject_counts": generation["reject_counts"],
        "topology_invalid_feature_set_count": topology.get("invalid_feature_set_count", 0),
        "training_best_epoch": metrics["best_epoch"],
        "test_macro_f1": final_test["macro_macro_f1_active"],
        "test_semantic_f1": final_test["semantic_macro_f1_active"],
        "test_edge_f1": final_test["edge_micro_f1"],
        "prediction_macro_global_accuracy": prediction_eval["macro"]["global_accuracy"],
        "prediction_macro_f1": prediction_eval["macro"]["macro_f1"],
        "prediction_semantic_global_accuracy": prediction_eval["semantic"]["global_accuracy"],
        "prediction_semantic_f1": prediction_eval["semantic"]["macro_f1"],
    }


def real_summary(real_report_root: Path, prefix: str) -> dict[str, Any]:
    compact = load_json(real_report_root / "reports" / f"{prefix}_compact_review.json")
    rotation = load_json(real_report_root / "reports" / f"{prefix}_rotation_consistency.json")
    samples = {}
    total_counts = Counter()
    for sample in compact.get("samples", []):
        counts = dict(sample.get("macro_counts", {}))
        total_counts.update(counts)
        samples[sample["sample_id"]] = {
            "feature_object_count": sample.get("feature_object_count", 0),
            "macro_counts": counts,
        }
    return {
        "sample_count": compact.get("sample_count", 0),
        "total_macro_counts": dict(sorted(total_counts.items())),
        "rotation_overall_consistent_group_count": rotation.get("overall_consistent_group_count", 0),
        "rotation_group_count": rotation.get("group_count", 0),
        "rotation_groups": [
            {
                "group_id": group["group_id"],
                "overall_consistent": group["overall_consistent"],
                "face_macro_consistent": group["face_macro_consistent"],
                "feature_object_macro_consistent": group["feature_object_macro_consistent"],
                "feature_object_signature_consistent": group["feature_object_signature_consistent"],
            }
            for group in rotation.get("groups", [])
        ],
        "samples": samples,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    single_eval = "edge_aware_v4_single_feature_80ep_topology_post_prediction_eval.json"
    multi_eval = "edge_aware_v4_nonintersecting_multi_80ep_topology_post_prediction_eval.json"
    return {
        "schema": "cae_ablation_comparison_v1",
        "single_feature": {
            "synthetic": synthetic_summary(args.single_dataset_root, args.single_run_name, single_eval),
            "real_step_s": real_summary(args.real_report_root, "single_feature"),
        },
        "nonintersecting_multi": {
            "synthetic": synthetic_summary(args.multi_dataset_root, args.multi_run_name, multi_eval),
            "real_step_s": real_summary(args.real_report_root, "nonintersecting_multi"),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    single = report["single_feature"]
    multi = report["nonintersecting_multi"]
    lines = [
        "# Ablation Dataset Comparison",
        "",
        "## Synthetic Metrics",
        "",
        "| Scheme | Samples | Test Macro F1 | Test Semantic F1 | Eval Macro Acc | Eval Macro F1 | Eval Semantic Acc | Eval Semantic F1 | Topology Invalid |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        metric_row("single_feature", single["synthetic"]),
        metric_row("nonintersecting_multi", multi["synthetic"]),
        "",
        "## Real STEP Rotation Consistency",
        "",
        "| Scheme | Consistent Groups | Total Groups | Total Predicted Macro Counts |",
        "|---|---:|---:|---|",
        real_row("single_feature", single["real_step_s"]),
        real_row("nonintersecting_multi", multi["real_step_s"]),
        "",
        "## Per-Part Predicted Feature Counts",
        "",
        "| Part | Single Feature Model | Nonintersecting Multi Model |",
        "|---|---|---|",
    ]
    part_ids = sorted(set(single["real_step_s"]["samples"]) | set(multi["real_step_s"]["samples"]))
    for part_id in part_ids:
        single_counts = single["real_step_s"]["samples"].get(part_id, {}).get("macro_counts", {})
        multi_counts = multi["real_step_s"]["samples"].get(part_id, {}).get("macro_counts", {})
        lines.append(f"| `{part_id}` | `{single_counts}` | `{multi_counts}` |")
    lines.append("")
    return "\n".join(lines)


def metric_row(label: str, metrics: dict[str, Any]) -> str:
    return (
        f"| `{label}` | {metrics['sample_count']} | {fmt(metrics['test_macro_f1'])} | "
        f"{fmt(metrics['test_semantic_f1'])} | {fmt(metrics['prediction_macro_global_accuracy'])} | "
        f"{fmt(metrics['prediction_macro_f1'])} | {fmt(metrics['prediction_semantic_global_accuracy'])} | "
        f"{fmt(metrics['prediction_semantic_f1'])} | {metrics['topology_invalid_feature_set_count']} |"
    )


def real_row(label: str, summary: dict[str, Any]) -> str:
    return (
        f"| `{label}` | {summary['rotation_overall_consistent_group_count']} | "
        f"{summary['rotation_group_count']} | `{summary['total_macro_counts']}` |"
    )


def fmt(value: float) -> str:
    return f"{float(value):.4f}"


def main() -> None:
    args = parse_args()
    report = build_report(args)
    write_json(args.output_json, report)
    write_text(args.output_md, render_markdown(report))
    print(f"comparison json written: {args.output_json}")
    print(f"comparison markdown written: {args.output_md}")


if __name__ == "__main__":
    main()
