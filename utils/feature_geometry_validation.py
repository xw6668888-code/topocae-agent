from __future__ import annotations

from pathlib import Path
import argparse
import json


def validate_hole_pocket_separation(
    graph_data: dict,
    blind_hole_min_ratio: float = 0.8,
    circular_pocket_max_ratio: float = 0.5,
) -> dict:
    issues = []
    checked = 0

    for feature_set in graph_data.get("feature_sets", []):
        macro_feature = feature_set.get("macro_feature")
        if macro_feature not in {"blind_hole_round", "pocket_circular"}:
            continue

        checked += 1
        ratio = feature_set.get("geometry_parameters", {}).get("depth_to_diameter_ratio")
        if ratio is None:
            issues.append({
                "code": "missing_depth_to_diameter_ratio",
                "instance_id": feature_set.get("instance_id"),
                "macro_feature": macro_feature,
                "message": "Feature set is missing depth_to_diameter_ratio.",
            })
            continue

        ratio = float(ratio)
        if macro_feature == "blind_hole_round" and ratio < blind_hole_min_ratio:
            issues.append({
                "code": "blind_hole_too_shallow",
                "instance_id": feature_set.get("instance_id"),
                "macro_feature": macro_feature,
                "depth_to_diameter_ratio": ratio,
                "threshold": blind_hole_min_ratio,
                "message": "Blind hole is too shallow relative to diameter and may be confused with a circular pocket.",
            })
        elif macro_feature == "pocket_circular" and ratio > circular_pocket_max_ratio:
            issues.append({
                "code": "circular_pocket_too_deep",
                "instance_id": feature_set.get("instance_id"),
                "macro_feature": macro_feature,
                "depth_to_diameter_ratio": ratio,
                "threshold": circular_pocket_max_ratio,
                "message": "Circular pocket is too deep relative to diameter and may be confused with a blind hole.",
            })

    return {
        "validator": "hole_pocket_depth_diameter_separation_v1",
        "checked_feature_count": checked,
        "issue_count": len(issues),
        "status": "ok" if not issues else "issues_found",
        "thresholds": {
            "blind_hole_min_depth_to_diameter_ratio": blind_hole_min_ratio,
            "circular_pocket_max_depth_to_diameter_ratio": circular_pocket_max_ratio,
        },
        "issues": issues,
    }


def validate_rectangular_recess_separation(
    graph_data: dict,
    slot_min_aspect_ratio: float = 4.0,
    pocket_max_aspect_ratio: float = 1.8,
) -> dict:
    issues = []
    checked = 0

    for feature_set in graph_data.get("feature_sets", []):
        macro_feature = feature_set.get("macro_feature")
        if macro_feature not in {"slot_rect", "pocket_rect"}:
            continue

        checked += 1
        ratio = feature_set.get("geometry_parameters", {}).get("planform_aspect_ratio")
        if ratio is None:
            issues.append({
                "code": "missing_planform_aspect_ratio",
                "instance_id": feature_set.get("instance_id"),
                "macro_feature": macro_feature,
                "message": "Feature set is missing planform_aspect_ratio.",
            })
            continue

        ratio = float(ratio)
        if macro_feature == "slot_rect" and ratio < slot_min_aspect_ratio:
            issues.append({
                "code": "slot_rect_not_elongated_enough",
                "instance_id": feature_set.get("instance_id"),
                "macro_feature": macro_feature,
                "planform_aspect_ratio": ratio,
                "threshold": slot_min_aspect_ratio,
                "message": "Rectangular slot is not elongated enough and may be confused with a rectangular pocket.",
            })
        elif macro_feature == "pocket_rect" and ratio > pocket_max_aspect_ratio:
            issues.append({
                "code": "pocket_rect_too_elongated",
                "instance_id": feature_set.get("instance_id"),
                "macro_feature": macro_feature,
                "planform_aspect_ratio": ratio,
                "threshold": pocket_max_aspect_ratio,
                "message": "Rectangular pocket is too elongated and may be confused with a rectangular slot.",
            })

    return {
        "validator": "rectangular_recess_planform_aspect_separation_v1",
        "checked_feature_count": checked,
        "issue_count": len(issues),
        "status": "ok" if not issues else "issues_found",
        "thresholds": {
            "slot_rect_min_planform_aspect_ratio": slot_min_aspect_ratio,
            "pocket_rect_max_planform_aspect_ratio": pocket_max_aspect_ratio,
        },
        "issues": issues,
    }


def validate_feature_geometry_separation(
    graph_data: dict,
    blind_hole_min_ratio: float = 0.8,
    circular_pocket_max_ratio: float = 0.5,
    slot_min_aspect_ratio: float = 4.0,
    pocket_max_aspect_ratio: float = 1.8,
) -> dict:
    reports = [
        validate_hole_pocket_separation(
            graph_data,
            blind_hole_min_ratio=blind_hole_min_ratio,
            circular_pocket_max_ratio=circular_pocket_max_ratio,
        ),
        validate_rectangular_recess_separation(
            graph_data,
            slot_min_aspect_ratio=slot_min_aspect_ratio,
            pocket_max_aspect_ratio=pocket_max_aspect_ratio,
        ),
    ]
    issues = [issue for report in reports for issue in report["issues"]]
    checked = sum(report["checked_feature_count"] for report in reports)
    return {
        "validator": "feature_geometry_separation_v1",
        "checked_feature_count": checked,
        "issue_count": len(issues),
        "status": "ok" if not issues else "issues_found",
        "validators": reports,
        "issues": issues,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate generated feature geometry parameter separability.")
    parser.add_argument("--graphs-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--blind-hole-min-ratio", type=float, default=0.8)
    parser.add_argument("--circular-pocket-max-ratio", type=float, default=0.5)
    parser.add_argument("--slot-min-aspect-ratio", type=float, default=4.0)
    parser.add_argument("--pocket-max-aspect-ratio", type=float, default=1.8)
    return parser


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")


def validate_graph_directory(
    graphs_dir: Path,
    blind_hole_min_ratio: float = 0.8,
    circular_pocket_max_ratio: float = 0.5,
    slot_min_aspect_ratio: float = 4.0,
    pocket_max_aspect_ratio: float = 1.8,
) -> dict:
    sample_reports = []
    total_issues = 0
    total_checked = 0
    for graph_path in sorted(graphs_dir.glob("*.json")):
        graph_data = load_json(graph_path)
        report = validate_feature_geometry_separation(
            graph_data,
            blind_hole_min_ratio=blind_hole_min_ratio,
            circular_pocket_max_ratio=circular_pocket_max_ratio,
            slot_min_aspect_ratio=slot_min_aspect_ratio,
            pocket_max_aspect_ratio=pocket_max_aspect_ratio,
        )
        total_issues += report["issue_count"]
        total_checked += report["checked_feature_count"]
        sample_reports.append({
            "graph_path": str(graph_path),
            **report,
        })

    return {
        "validator": "feature_geometry_dataset_validation_v1",
        "sample_count": len(sample_reports),
        "checked_feature_count": total_checked,
        "issue_count": total_issues,
        "status": "ok" if total_issues == 0 else "issues_found",
        "samples": sample_reports,
    }


def main() -> None:
    args = build_parser().parse_args()
    report = validate_graph_directory(
        args.graphs_dir,
        blind_hole_min_ratio=args.blind_hole_min_ratio,
        circular_pocket_max_ratio=args.circular_pocket_max_ratio,
        slot_min_aspect_ratio=args.slot_min_aspect_ratio,
        pocket_max_aspect_ratio=args.pocket_max_aspect_ratio,
    )
    write_json(args.output_path, report)
    print(f"feature geometry validation written: {args.output_path}")
    if report["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
