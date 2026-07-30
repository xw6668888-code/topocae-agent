from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a self-contained HTML report for CAE STEP recognition experiments.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--prediction-eval-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def generate_report(dataset_root: Path, run_dir: Path, prediction_eval_path: Path, output_path: Path) -> None:
    dataset_meta = load_json(dataset_root / "dataset_meta.json")
    dataset_report = load_json(dataset_root / "reports" / "dataset_report.json")
    metrics = load_json(run_dir / "metrics.json")
    prediction_eval = load_json(prediction_eval_path)
    geometry_report = optional_json(dataset_root / "reports" / "feature_geometry_validation.json")

    html_text = render_report(
        dataset_root=dataset_root,
        run_dir=run_dir,
        prediction_eval_path=prediction_eval_path,
        dataset_meta=dataset_meta,
        dataset_report=dataset_report,
        metrics=metrics,
        prediction_eval=prediction_eval,
        geometry_report=geometry_report,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")


def optional_json(path: Path) -> dict:
    return load_json(path) if path.exists() else {}


def render_report(
    dataset_root: Path,
    run_dir: Path,
    prediction_eval_path: Path,
    dataset_meta: dict,
    dataset_report: dict,
    metrics: dict,
    prediction_eval: dict,
    geometry_report: dict,
) -> str:
    macro = prediction_eval.get("macro", {})
    semantic = prediction_eval.get("semantic", {})
    history = metrics.get("history", [])
    feature_distribution = aggregate_feature_distribution(dataset_report)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>CAE STEP Recognition Experiment Report</title>
  <style>
    :root {{
      --ink: #172026;
      --muted: #5b6670;
      --line: #d8dee4;
      --bg: #f7f9fb;
      --panel: #ffffff;
      --accent: #2a6f97;
      --accent2: #7a4e9d;
      --good: #237a57;
    }}
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      color: var(--ink);
      background: var(--bg);
      line-height: 1.45;
    }}
    header {{
      padding: 28px 40px 22px;
      background: #ffffff;
      border-bottom: 1px solid var(--line);
    }}
    h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }}
    h2 {{ margin: 0 0 16px; font-size: 19px; letter-spacing: 0; }}
    h3 {{ margin: 18px 0 10px; font-size: 15px; letter-spacing: 0; }}
    main {{ padding: 24px 40px 42px; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }}
    .two {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 16px;
      margin-bottom: 18px;
    }}
    .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 14px;
    }}
    .metric .label {{ color: var(--muted); font-size: 12px; }}
    .metric .value {{ font-size: 24px; font-weight: 700; margin-top: 4px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 8px 9px; border-bottom: 1px solid var(--line); text-align: left; }}
    th {{ color: var(--muted); font-weight: 700; }}
    code {{ background: #eef2f5; padding: 1px 4px; border-radius: 4px; }}
    .small {{ color: var(--muted); font-size: 12px; }}
    .note {{ color: var(--muted); margin-top: 8px; }}
    svg {{ max-width: 100%; height: auto; }}
    @media (max-width: 900px) {{ .grid, .two {{ grid-template-columns: 1fr; }} main, header {{ padding-left: 18px; padding-right: 18px; }} }}
  </style>
</head>
<body>
<header>
  <h1>CAE STEP Recognition Experiment Report</h1>
  <div class="small">Dataset: <code>{escape(str(dataset_root))}</code> | Run: <code>{escape(str(run_dir))}</code></div>
</header>
<main>
  <section class="grid">
    {metric_card("Samples", dataset_meta.get("actual_samples_written", dataset_report.get("total_samples", 0)))}
    {metric_card("Faces Evaluated", prediction_eval.get("face_count", 0))}
    {metric_card("Macro Feature F1", fmt_float(macro.get("macro_f1", 0.0)))}
    {metric_card("Semantic Subtype F1", fmt_float(semantic.get("macro_f1", 0.0)))}
  </section>

  <section class="panel">
    <h2>Paper-Ready Metrics</h2>
    <table>
      <thead><tr><th>Task</th><th>Global Accuracy</th><th>Macro-F1</th><th>Weighted-F1</th><th>Active Classes</th></tr></thead>
      <tbody>
        {task_metric_row("Macro Feature", macro)}
        {task_metric_row("Semantic Subtype", semantic)}
      </tbody>
    </table>
    <div class="note">Use Global Accuracy, Macro-F1, Weighted-F1, and per-class precision/recall/F1 as the main recognition metrics in the paper.</div>
  </section>

  <section class="two">
    <div class="panel">
      <h2>Training Curves</h2>
      {line_chart(history)}
    </div>
    <div class="panel">
      <h2>Feature Instance Distribution</h2>
      {bar_chart(feature_distribution, width=640, height=300)}
    </div>
  </section>

  <section class="two">
    <div class="panel">
      <h2>Macro Classes to Inspect</h2>
      {class_review_table(macro.get("per_class", {}), limit=8)}
    </div>
    <div class="panel">
      <h2>Semantic Classes to Inspect</h2>
      {class_review_table(semantic.get("per_class", {}), limit=8)}
    </div>
  </section>

  <section class="panel">
    <h2>Macro Feature Per-Class Results</h2>
    {per_class_table(macro.get("per_class", {}), limit=40)}
  </section>

  <section class="panel">
    <h2>Semantic Subtype Per-Class Results</h2>
    {per_class_table(semantic.get("per_class", {}), limit=60)}
  </section>

  <section class="panel">
    <h2>Dataset Quality Checks</h2>
    {quality_table(dataset_report, geometry_report)}
  </section>

  <section class="panel">
    <h2>Artifacts</h2>
    <table>
      <tbody>
        <tr><th>Dataset report</th><td><code>{escape(str(dataset_root / "reports" / "dataset_report.json"))}</code></td></tr>
        <tr><th>Training metrics</th><td><code>{escape(str(run_dir / "metrics.json"))}</code></td></tr>
        <tr><th>Prediction evaluation</th><td><code>{escape(str(prediction_eval_path))}</code></td></tr>
        <tr><th>Best checkpoint</th><td><code>{escape(str(run_dir / "best.pt"))}</code></td></tr>
      </tbody>
    </table>
  </section>
</main>
</body>
</html>
"""


def aggregate_feature_distribution(dataset_report: dict) -> dict[str, int]:
    total: dict[str, int] = {}
    for split in dataset_report.get("splits", {}).values():
        for key, value in split.get("feature_instance_distribution", {}).items():
            total[key] = total.get(key, 0) + int(value)
    return dict(sorted(total.items(), key=lambda item: (-item[1], item[0])))


def metric_card(label: str, value) -> str:
    return f'<div class="metric"><div class="label">{escape(label)}</div><div class="value">{escape(str(value))}</div></div>'


def task_metric_row(label: str, metrics: dict) -> str:
    return (
        f"<tr><td>{escape(label)}</td>"
        f"<td>{fmt_float(metrics.get('global_accuracy', 0.0))}</td>"
        f"<td>{fmt_float(metrics.get('macro_f1', 0.0))}</td>"
        f"<td>{fmt_float(metrics.get('weighted_f1', 0.0))}</td>"
        f"<td>{int(metrics.get('active_class_count', 0))}</td></tr>"
    )


def per_class_table(per_class: dict, limit: int) -> str:
    rows = []
    for name, stats in sorted(per_class.items(), key=lambda item: (-item[1].get("support", 0), item[0]))[:limit]:
        rows.append(
            f"<tr><td>{escape(name)}</td><td>{int(stats.get('support', 0))}</td>"
            f"<td>{fmt_float(stats.get('precision', 0.0))}</td>"
            f"<td>{fmt_float(stats.get('recall', 0.0))}</td>"
            f"<td>{fmt_float(stats.get('f1', 0.0))}</td></tr>"
        )
    if not rows:
        rows.append('<tr><td colspan="5">No per-class metrics available.</td></tr>')
    return "<table><thead><tr><th>Class</th><th>Support</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def class_review_table(per_class: dict, limit: int) -> str:
    rows = []
    ranked = sorted(
        ((name, stats) for name, stats in per_class.items() if int(stats.get("support", 0)) > 0),
        key=lambda item: (item[1].get("f1", 0.0), -item[1].get("support", 0), item[0]),
    )
    for name, stats in ranked[:limit]:
        rows.append(
            f"<tr><td>{escape(name)}</td><td>{int(stats.get('support', 0))}</td>"
            f"<td>{fmt_float(stats.get('precision', 0.0))}</td>"
            f"<td>{fmt_float(stats.get('recall', 0.0))}</td>"
            f"<td>{fmt_float(stats.get('f1', 0.0))}</td></tr>"
        )
    if not rows:
        rows.append('<tr><td colspan="5">No class review metrics available.</td></tr>')
    return "<table><thead><tr><th>Class</th><th>Support</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def quality_table(dataset_report: dict, geometry_report: dict) -> str:
    splits = dataset_report.get("splits", {})
    split_rows = "".join(
        f"<tr><td>{escape(name)}</td><td>{split.get('sample_count', 0)}</td><td>{split.get('node_count', 0)}</td><td>{split.get('feature_set_count', 0)}</td></tr>"
        for name, split in splits.items()
    )
    geometry_status = geometry_report.get("status", "not_run")
    geometry_checked = geometry_report.get("checked_feature_count", 0)
    geometry_issues = geometry_report.get("issue_count", 0)
    return (
        "<h3>Splits</h3>"
        "<table><thead><tr><th>Split</th><th>Samples</th><th>Faces</th><th>Feature Instances</th></tr></thead><tbody>"
        + split_rows
        + "</tbody></table>"
        "<h3>Geometry Validation</h3>"
        f"<table><tbody><tr><th>Status</th><td>{escape(str(geometry_status))}</td></tr>"
        f"<tr><th>Checked Features</th><td>{geometry_checked}</td></tr>"
        f"<tr><th>Issues</th><td>{geometry_issues}</td></tr></tbody></table>"
    )


def line_chart(history: list[dict], width: int = 640, height: int = 300) -> str:
    if not history:
        return "<p>No training history available.</p>"
    series = {
        "val macro F1": [item.get("val", {}).get("macro_macro_f1_active", 0.0) for item in history],
        "val semantic F1": [item.get("val", {}).get("semantic_macro_f1_active", 0.0) for item in history],
        "val score": [item.get("val", {}).get("selection_score", 0.0) for item in history],
    }
    max_value = max([1e-9] + [value for values in series.values() for value in values])
    colors = ["#2a6f97", "#7a4e9d", "#237a57"]
    paths = []
    for idx, (label, values) in enumerate(series.items()):
        points = []
        for i, value in enumerate(values):
            x = 48 + safe_divide(i, max(len(values) - 1, 1)) * (width - 78)
            y = height - 36 - safe_divide(value, max_value) * (height - 68)
            points.append(f"{x:.1f},{y:.1f}")
        paths.append(f'<polyline fill="none" stroke="{colors[idx]}" stroke-width="2.5" points="{" ".join(points)}"><title>{escape(label)}</title></polyline>')
    legend = "".join(
        f'<text x="54" y="{22 + i * 17}" fill="{colors[i]}" font-size="12">{escape(label)}</text>'
        for i, label in enumerate(series)
    )
    return f'<svg viewBox="0 0 {width} {height}" role="img"><rect x="0" y="0" width="{width}" height="{height}" fill="#fff"/><line x1="48" y1="{height-36}" x2="{width-24}" y2="{height-36}" stroke="#d8dee4"/><line x1="48" y1="24" x2="48" y2="{height-36}" stroke="#d8dee4"/>{"".join(paths)}{legend}</svg>'


def bar_chart(values: dict[str, int], width: int = 640, height: int = 300) -> str:
    if not values:
        return "<p>No distribution data available.</p>"
    items = list(values.items())[:16]
    max_value = max(value for _, value in items)
    bar_height = max(10, (height - 42) / max(len(items), 1) - 4)
    rows = []
    for idx, (label, value) in enumerate(items):
        y = 24 + idx * (bar_height + 4)
        bar_width = safe_divide(value, max_value) * (width - 230)
        rows.append(
            f'<text x="8" y="{y + bar_height * 0.72:.1f}" font-size="11">{escape(label)}</text>'
            f'<rect x="170" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="#2a6f97"/>'
            f'<text x="{176 + bar_width:.1f}" y="{y + bar_height * 0.72:.1f}" font-size="11">{value}</text>'
        )
    return f'<svg viewBox="0 0 {width} {height}" role="img"><rect x="0" y="0" width="{width}" height="{height}" fill="#fff"/>{"".join(rows)}</svg>'


def fmt_float(value) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "0.0000"


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def escape(value: str) -> str:
    return html.escape(value, quote=True)


def main() -> None:
    args = parse_args()
    generate_report(
        dataset_root=args.dataset_root,
        run_dir=args.run_dir,
        prediction_eval_path=args.prediction_eval_path,
        output_path=args.output_path,
    )
    print(f"experiment report written: {args.output_path}")


if __name__ == "__main__":
    main()
