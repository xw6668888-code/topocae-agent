from __future__ import annotations

import argparse
import json
from pathlib import Path


STEP_SUFFIXES = {".step", ".stp"}


def orient_shape_for_positive_volume(shape):
    volume = signed_solid_volume(shape)
    if volume is not None and volume < 0.0:
        return shape.Reversed()
    return shape


def write_step_with_positive_volume(shape, output_path: Path) -> None:
    from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_ManifoldSolidBrep, STEPControl_Writer
    from OCC.Core.TopAbs import TopAbs_SOLID

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_shape = solid_body_for_step_export(shape)
    transfer_mode = STEPControl_ManifoldSolidBrep if export_shape.ShapeType() == TopAbs_SOLID else STEPControl_AsIs

    writer = STEPControl_Writer()
    writer.Transfer(orient_shape_for_positive_volume(export_shape), transfer_mode)
    writer.Write(str(output_path))

    report = inspect_step_file(output_path)
    if "negative_solid_volume" not in report["issues"]:
        return

    repaired_shape = read_step_shape(output_path)
    repaired_export_shape = solid_body_for_step_export(repaired_shape)
    repaired_transfer_mode = STEPControl_ManifoldSolidBrep if repaired_export_shape.ShapeType() == TopAbs_SOLID else STEPControl_AsIs
    repair_writer = STEPControl_Writer()
    repair_writer.Transfer(orient_shape_for_positive_volume(repaired_export_shape), repaired_transfer_mode)
    repair_writer.Write(str(output_path))


def read_step_shape(path: Path):
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.STEPControl import STEPControl_Reader

    reader = STEPControl_Reader()
    status = reader.ReadFile(str(path))
    if status != IFSelect_RetDone:
        raise ValueError(f"Could not read STEP file: {path}")
    reader.TransferRoots()
    return reader.OneShape()


def signed_solid_volume(shape) -> float | None:
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GProp import GProp_GProps
    from OCC.Extend.TopologyUtils import TopologyExplorer

    shape_props = GProp_GProps()
    brepgprop.VolumeProperties(shape, shape_props)
    shape_volume = shape_props.Mass()
    if abs(shape_volume) > 1e-9:
        return shape_volume

    solids = list(TopologyExplorer(shape).solids())
    if not solids:
        return None

    total_volume = 0.0
    for solid in solids:
        props = GProp_GProps()
        brepgprop.VolumeProperties(solid, props)
        total_volume += props.Mass()
    return total_volume


def solid_body_for_step_export(shape):
    from OCC.Extend.TopologyUtils import TopologyExplorer

    solids = list(TopologyExplorer(shape).solids())
    if len(solids) == 1:
        return orient_shape_for_positive_volume(solids[0])
    return orient_shape_for_positive_volume(shape)


def inspect_step_file(path: Path) -> dict:
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Extend.TopologyUtils import TopologyExplorer

    reader = STEPControl_Reader()
    status = reader.ReadFile(str(path))
    if status != IFSelect_RetDone:
        return {
            "step_path": str(path),
            "readable": False,
            "solid_count": 0,
            "shell_count": 0,
            "face_count": 0,
            "edge_count": 0,
            "signed_volume": None,
            "valid": False,
            "issues": ["step_read_failed"],
        }

    reader.TransferRoots()
    shape = reader.OneShape()
    topo = TopologyExplorer(shape)
    top_shape_type = shape.ShapeType()
    solid_count = len(list(topo.solids()))
    shell_count = len(list(topo.shells()))
    face_count = len(list(topo.faces()))
    edge_count = len(list(topo.edges()))
    signed_volume = signed_solid_volume(shape)
    valid = BRepCheck_Analyzer(shape).IsValid()

    issues = []
    if solid_count == 0:
        issues.append("no_solid")
    if solid_count != 1:
        issues.append("not_single_solid_body")
    if not valid:
        issues.append("invalid_topology")
    if signed_volume is not None and signed_volume < 0.0:
        issues.append("negative_solid_volume")

    return {
        "step_path": str(path),
        "readable": True,
        "top_shape_type": top_shape_type,
        "solid_count": solid_count,
        "shell_count": shell_count,
        "face_count": face_count,
        "edge_count": edge_count,
        "signed_volume": signed_volume,
        "valid": valid,
        "issues": issues,
    }


def validate_step_directory(steps_dir: Path, recursive: bool = False) -> dict:
    globber = steps_dir.rglob if recursive else steps_dir.glob
    step_paths = sorted(
        path for path in globber("*")
        if path.is_file() and path.suffix.lower() in STEP_SUFFIXES
    )
    reports = [inspect_step_file(path) for path in step_paths]
    issue_counts: dict[str, int] = {}
    for report in reports:
        for issue in report["issues"]:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1

    blocking_issue_count = sum(
        count for issue, count in issue_counts.items()
        if issue in {"step_read_failed", "no_solid", "not_single_solid_body", "invalid_topology"}
    )
    return {
        "validator": "step_solid_topology_validation_v1",
        "step_count": len(reports),
        "issue_counts": issue_counts,
        "blocking_issue_count": blocking_issue_count,
        "status": "ok" if blocking_issue_count == 0 else "issues_found",
        "samples": reports,
    }


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate STEP files as readable solid topology.")
    parser.add_argument("--steps-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--fail-on-negative-volume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = validate_step_directory(args.steps_dir, recursive=args.recursive)
    write_json(args.output_path, report)
    print(f"STEP solid validation written: {args.output_path}")

    has_negative_volume = report["issue_counts"].get("negative_solid_volume", 0) > 0
    if report["status"] != "ok" or (args.fail_on_negative_volume and has_negative_volume):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
