from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

from generate_area_truth_review import feature_area, within_tolerance
from utils.area_truth_normalization import normalize_truth_macro

UNIT_IN2_TO_MM2 = 25.4 * 25.4


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def collect_json_files(path: Path, recursive: bool, ignored_names: set[str]) -> list[Path]:
    if path.is_file():
        return [path]
    globber = path.rglob if recursive else path.glob
    return sorted(p for p in globber("*.json") if p.name not in ignored_names)


def normalize_sample_id(path: Path, doc: dict[str, Any] | None = None) -> str:
    if doc:
        for key in ("sample_id", "part_id", "name"):
            value = doc.get(key)
            if value:
                return str(value)
    return path.stem


def node_area(node: dict[str, Any]) -> float | None:
    attributes = node.get("attributes", {})
    physical = attributes.get("physical", {})
    for value in (physical.get("area"), attributes.get("area"), node.get("area")):
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def graph_face_areas(graph_doc: dict[str, Any]) -> list[float]:
    areas: list[float] = []
    for node in graph_doc.get("nodes", []):
        area = node_area(node)
        if area is not None and math.isfinite(area) and area > 0.0:
            areas.append(area)
    return areas


def load_graphs(path: Path, recursive: bool = False) -> dict[str, dict[str, Any]]:
    graphs: dict[str, dict[str, Any]] = {}
    for json_path in collect_json_files(path, recursive, {"inference_graph_manifest.json"}):
        doc = load_json(json_path)
        if "nodes" not in doc:
            continue
        graphs[normalize_sample_id(json_path, doc)] = doc
    return graphs


def summarized_feature_object(feature_object: dict[str, Any], index: int) -> dict[str, Any]:
    face_indices = list(feature_object.get("face_indices", []))
    return {
        "index": index,
        "feature_object_id": feature_object.get("feature_object_id", f"feature_{index:03d}"),
        "macro": normalize_truth_macro(str(feature_object.get("macro_feature", feature_object.get("macro", "unknown")))),
        "face_count": int(feature_object.get("face_count", len(face_indices))),
        "face_indices": face_indices,
        "area": feature_area(feature_object),
    }


def load_feature_objects(path: Path, recursive: bool = False) -> dict[str, list[dict[str, Any]]]:
    by_sample: dict[str, list[dict[str, Any]]] = {}
    for json_path in collect_json_files(path, recursive, {"feature_object_manifest.json"}):
        doc = load_json(json_path)
        sample_id = normalize_sample_id(json_path, doc)
        by_sample[sample_id] = [
            summarized_feature_object(feature_object, index)
            for index, feature_object in enumerate(doc.get("feature_objects", []))
        ]
    return by_sample


def load_truth(path: Path) -> tuple[dict[str, list[dict[str, Any]]], str]:
    truth_doc = load_json(path)
    truth_source_name = Path(str(truth_doc.get("source_xlsx", ""))).name
    samples = truth_doc.get("samples", {})
    normalized: dict[str, list[dict[str, Any]]] = {}
    for sample_id, entries in samples.items():
        normalized_entries = []
        for index, entry in enumerate(entries):
            area = float(entry["area"])
            face_count = int(entry["face_count"])
            normalized_entries.append(
                {
                    "index": index,
                    "sample": sample_id,
                    "macro": normalize_truth_macro(
                        str(entry["macro"]),
                        sample_id=sample_id,
                        area=area,
                        face_count=face_count,
                        truth_source_name=truth_source_name,
                    ),
                    "raw_macro": str(entry["macro"]),
                    "area": area,
                    "face_count": face_count,
                }
            )
        normalized[str(sample_id)] = normalized_entries
    return normalized, truth_source_name


def area_error(predicted_area: float, truth_area: float) -> dict[str, float]:
    absolute = abs(predicted_area - truth_area)
    relative = absolute / max(abs(truth_area), 1e-9)
    return {"absolute": absolute, "relative": relative}


def best_area_candidate(
    truth_area: float,
    candidates: list[dict[str, Any]],
    abs_tol: float,
    rel_tol: float,
) -> dict[str, Any] | None:
    ranked: list[tuple[int, float, float, dict[str, Any]]] = []
    for candidate in candidates:
        candidate_area = float(candidate["area"])
        error = area_error(candidate_area, truth_area)
        is_match = within_tolerance(candidate_area, truth_area, abs_tol, rel_tol)
        ranked.append((0 if is_match else 1, error["relative"], error["absolute"], {**candidate, "error": error}))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (item[0], item[1], item[2]))
    return ranked[0][3]


def nearest_unit_ratio(truth_area: float, observed_areas: list[float]) -> float | None:
    if truth_area <= 0.0:
        return None
    ratios = [area / truth_area for area in observed_areas if area > 0.0]
    if not ratios:
        return None
    return min(ratios, key=lambda ratio: abs(math.log(max(ratio, 1e-12) / UNIT_IN2_TO_MM2)))


def evaluate_truth_plausibility(
    truth_object: dict[str, Any],
    face_areas: list[float],
    feature_objects: list[dict[str, Any]],
    rel_tol: float = 0.02,
    abs_tol: float = 5.0,
) -> dict[str, Any]:
    truth_area = float(truth_object["area"])
    macro = normalize_truth_macro(str(truth_object["macro"]))
    face_count = int(truth_object.get("face_count", 0))

    object_candidates = [
        {
            "kind": "feature_object",
            "macro": normalize_truth_macro(str(feature_object.get("macro", feature_object.get("macro_feature", "unknown")))),
            "face_count": int(feature_object.get("face_count", len(feature_object.get("face_indices", [])))),
            "area": float(feature_object["area"]),
            "feature_object_id": feature_object.get("feature_object_id"),
            "face_indices": list(feature_object.get("face_indices", [])),
        }
        for feature_object in feature_objects
        if normalize_truth_macro(str(feature_object.get("macro", feature_object.get("macro_feature", "unknown")))) == macro
        and int(feature_object.get("face_count", len(feature_object.get("face_indices", [])))) == face_count
    ]
    best_object = best_area_candidate(truth_area, object_candidates, abs_tol, rel_tol)
    object_match = bool(best_object and best_object["error"]["absolute"] <= abs_tol or best_object and best_object["error"]["relative"] <= rel_tol)

    face_candidates: list[dict[str, Any]] = []
    if face_count == 1:
        face_candidates = [
            {"kind": "graph_face_area", "area": float(area), "face_index": index}
            for index, area in enumerate(face_areas)
        ]
    best_face = best_area_candidate(truth_area, face_candidates, abs_tol, rel_tol)
    face_match = bool(best_face and best_face["error"]["absolute"] <= abs_tol or best_face and best_face["error"]["relative"] <= rel_tol)

    observed_areas = [float(area) for area in face_areas]
    observed_areas.extend(float(feature_object["area"]) for feature_object in feature_objects if "area" in feature_object)
    ratio = nearest_unit_ratio(truth_area, observed_areas)

    status = "plausible" if object_match or face_match else "suspicious"
    reasons: list[str] = []
    if object_match:
        reasons.append("matching_feature_object_area")
    if face_match:
        reasons.append("matching_graph_face_area")
    if not reasons:
        reasons.append("no_nearby_area_in_graph_or_feature_objects")
    if ratio is not None and 400.0 <= ratio <= 900.0:
        reasons.append("nearest_area_ratio_close_to_in2_to_mm2")

    return {
        "truth": truth_object,
        "status": status,
        "reasons": reasons,
        "best_feature_object": best_object,
        "best_graph_face": best_face,
        "nearest_unit_ratio": ratio,
    }


def sample_unit_ratio(plausibility_results: list[dict[str, Any]]) -> float | None:
    suspicious_ratios = [
        float(result["nearest_unit_ratio"])
        for result in plausibility_results
        if result["status"] == "suspicious"
        and result.get("nearest_unit_ratio") is not None
        and 400.0 <= float(result["nearest_unit_ratio"]) <= 900.0
    ]
    if not suspicious_ratios:
        return None
    return median(suspicious_ratios)


def build_sanity_report(
    truth_by_sample: dict[str, list[dict[str, Any]]],
    graph_by_sample: dict[str, dict[str, Any] | list[Any]],
    feature_objects_by_sample: dict[str, list[dict[str, Any]]],
    rel_tol: float = 0.02,
    abs_tol: float = 5.0,
) -> dict[str, Any]:
    sample_reports: dict[str, dict[str, Any]] = {}
    overall_truth = 0
    overall_plausible = 0
    overall_suspicious = 0
    flag_counter: Counter[str] = Counter()

    all_sample_ids = sorted(set(truth_by_sample) | set(graph_by_sample) | set(feature_objects_by_sample))
    for sample_id in all_sample_ids:
        truth_entries = truth_by_sample.get(sample_id, [])
        graph_doc = graph_by_sample.get(sample_id)
        feature_objects = feature_objects_by_sample.get(sample_id, [])
        flags: list[str] = []

        if graph_doc is None:
            face_areas = []
            flags.append("missing_graph")
        elif isinstance(graph_doc, dict):
            face_areas = graph_face_areas(graph_doc)
            if not face_areas:
                flags.append("graph_has_no_face_areas")
        else:
            face_areas = []
            flags.append("invalid_graph_doc")

        if not truth_entries:
            flags.append("missing_truth")
        if not feature_objects:
            flags.append("missing_feature_objects")

        plausibility = [
            evaluate_truth_plausibility(entry, face_areas, feature_objects, rel_tol=rel_tol, abs_tol=abs_tol)
            for entry in truth_entries
        ]
        plausible_count = sum(1 for item in plausibility if item["status"] == "plausible")
        suspicious_count = sum(1 for item in plausibility if item["status"] == "suspicious")
        unit_ratio = sample_unit_ratio(plausibility)
        if unit_ratio is not None:
            flags.append("possible_unit_mismatch")
        if truth_entries and suspicious_count / max(len(truth_entries), 1) >= 0.5:
            flags.append("many_suspicious_truth_areas")

        for flag in flags:
            flag_counter[flag] += 1
        overall_truth += len(truth_entries)
        overall_plausible += plausible_count
        overall_suspicious += suspicious_count

        sample_reports[sample_id] = {
            "sample_id": sample_id,
            "truth_count": len(truth_entries),
            "graph_face_count": len(face_areas),
            "feature_object_count": len(feature_objects),
            "plausible_truth_count": plausible_count,
            "suspicious_truth_count": suspicious_count,
            "possible_unit_ratio": unit_ratio,
            "flags": flags,
            "truth_macro_counts": dict(Counter(entry["macro"] for entry in truth_entries)),
            "feature_object_macro_counts": dict(Counter(obj["macro"] for obj in feature_objects)),
            "plausibility": plausibility,
        }

    return {
        "schema": "real_part_sanity_report_v1",
        "absolute_area_tolerance": abs_tol,
        "relative_area_tolerance": rel_tol,
        "sample_count": len(sample_reports),
        "truth_count": overall_truth,
        "plausible_truth_count": overall_plausible,
        "suspicious_truth_count": overall_suspicious,
        "flag_counts": dict(flag_counter),
        "samples": sample_reports,
    }


def render_markdown(report: dict[str, Any], max_suspicious_per_sample: int = 6) -> str:
    lines = [
        "# Real-Part Evaluation Sanity Report",
        "",
        f"- Samples: `{report['sample_count']}`",
        f"- Truth objects: `{report['truth_count']}`",
        f"- Plausible truth objects: `{report['plausible_truth_count']}`",
        f"- Suspicious truth objects: `{report['suspicious_truth_count']}`",
        f"- Area tolerance: absolute `{report['absolute_area_tolerance']}`, relative `{report['relative_area_tolerance']:.4f}`",
        "",
        "## Flag Counts",
        "",
    ]
    if report["flag_counts"]:
        for flag, count in sorted(report["flag_counts"].items()):
            lines.append(f"- `{flag}`: `{count}`")
    else:
        lines.append("- No sample-level flags.")

    lines.extend(
        [
            "",
            "## Sample Summary",
            "",
            "| Sample | Truth | Graph faces | Pred objects | Plausible | Suspicious | Unit ratio | Flags |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for sample in report["samples"].values():
        unit_ratio = sample["possible_unit_ratio"]
        unit_text = "" if unit_ratio is None else f"{unit_ratio:.2f}"
        flag_text = ", ".join(f"`{flag}`" for flag in sample["flags"]) if sample["flags"] else ""
        lines.append(
            f"| `{sample['sample_id']}` | {sample['truth_count']} | {sample['graph_face_count']} | "
            f"{sample['feature_object_count']} | {sample['plausible_truth_count']} | "
            f"{sample['suspicious_truth_count']} | {unit_text} | {flag_text} |"
        )

    lines.extend(["", "## Suspicious Truth Details", ""])
    any_suspicious = False
    for sample in report["samples"].values():
        suspicious = [item for item in sample["plausibility"] if item["status"] == "suspicious"]
        if not suspicious:
            continue
        any_suspicious = True
        lines.extend(
            [
                f"### {sample['sample_id']}",
                "",
                "| Truth index | Macro | Face count | Area | Reasons | Nearest feature object | Nearest graph face | Unit ratio |",
                "|---:|---|---:|---:|---|---|---|---:|",
            ]
        )
        for item in suspicious[:max_suspicious_per_sample]:
            truth = item["truth"]
            best_object = item.get("best_feature_object")
            best_face = item.get("best_graph_face")
            object_text = ""
            if best_object:
                object_text = (
                    f"{best_object.get('macro', '')}/faces={best_object.get('face_count', '')}/"
                    f"area={float(best_object['area']):.4f}"
                )
            face_text = ""
            if best_face:
                face_text = f"face={best_face.get('face_index')}/area={float(best_face['area']):.4f}"
            ratio = item.get("nearest_unit_ratio")
            ratio_text = "" if ratio is None else f"{float(ratio):.2f}"
            lines.append(
                f"| {truth['index']} | `{truth['macro']}` | {truth['face_count']} | "
                f"{float(truth['area']):.4f} | {', '.join(item['reasons'])} | "
                f"{object_text} | {face_text} | {ratio_text} |"
            )
        if len(suspicious) > max_suspicious_per_sample:
            lines.append(f"|  |  |  |  | ... {len(suspicious) - max_suspicious_per_sample} more suspicious rows omitted here. |  |  |  |")
        lines.append("")
    if not any_suspicious:
        lines.append("No suspicious truth rows detected.")

    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether real-part area truth rows are evaluable against graphs and feature objects.")
    parser.add_argument("--graph-dir", required=True, type=Path)
    parser.add_argument("--truth-json", required=True, type=Path)
    parser.add_argument("--feature-objects-path", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--absolute-area-tolerance", type=float, default=5.0)
    parser.add_argument("--relative-area-tolerance", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    truth_by_sample, truth_source_name = load_truth(args.truth_json)
    report = build_sanity_report(
        truth_by_sample,
        load_graphs(args.graph_dir, recursive=args.recursive),
        load_feature_objects(args.feature_objects_path, recursive=args.recursive),
        rel_tol=args.relative_area_tolerance,
        abs_tol=args.absolute_area_tolerance,
    )
    report.update(
        {
            "graph_dir": str(args.graph_dir.resolve()),
            "truth_json": str(args.truth_json.resolve()),
            "truth_source_name": truth_source_name,
            "feature_objects_path": str(args.feature_objects_path.resolve()),
        }
    )
    write_json(args.output_json, report)
    write_text(args.output_md, render_markdown(report))


if __name__ == "__main__":
    main()
