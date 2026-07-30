from __future__ import annotations

import argparse
import json
from pathlib import Path

from OCC.Extend.DataExchange import read_step_file

from core.AAGExtractor import extract_aag_graph
from utils.ontology import export_layer_label_map, load_layer_index


STEP_SUFFIXES = {".step", ".stp"}


def parse_args() -> argparse.Namespace:
    current_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Convert STEP files into unlabeled inference graphs for CAE feature recognition."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="Input STEP file or directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where inference graph JSON files will be written.",
    )
    parser.add_argument(
        "--ontology",
        type=Path,
        default=current_dir / "CAE_feature_ontology.yaml",
        help="Ontology file used to resolve primitive_surface ids.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan --input-path when it is a directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing graph JSON files in the output directory.",
    )
    return parser.parse_args()


def collect_step_files(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in STEP_SUFFIXES:
            raise ValueError(f"Unsupported STEP suffix: {input_path.suffix}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    globber = input_path.rglob if recursive else input_path.glob
    return sorted(
        path
        for path in globber("*")
        if path.is_file() and path.suffix.lower() in STEP_SUFFIXES
    )


def target_graph_path(step_path: Path, input_root: Path, output_dir: Path) -> Path:
    if input_root.is_file():
        return output_dir / f"{input_root.stem}.json"
    rel_path = step_path.relative_to(input_root)
    return (output_dir / rel_path).with_suffix(".json")


def convert_step_to_graph(
    step_path: Path,
    primitive_surface_index: dict,
    primitive_label_map: dict,
) -> dict:
    shape = read_step_file(str(step_path))
    graph_data = extract_aag_graph(
        shape,
        face_infos=None,
        primitive_surface_index=primitive_surface_index,
        include_label_fields=False,
    )
    graph_data["graph_mode"] = "inference_unlabeled"
    graph_data["source_step_path"] = str(step_path.resolve())
    graph_data["feature_sets"] = []
    graph_data["label_maps"] = {
        "primitive_surface": primitive_label_map["primitive_surface"],
    }
    return graph_data


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")


def build_manifest(entries: list[dict], output_dir: Path, primitive_label_map: dict) -> None:
    manifest = {
        "graph_mode": "inference_unlabeled",
        "sample_count": len(entries),
        "primitive_surface_label_map_path": str((output_dir / "primitive_surface_label_map.json").resolve()),
        "samples": entries,
    }
    write_json(output_dir / "inference_graph_manifest.json", manifest)
    write_json(output_dir / "primitive_surface_label_map.json", primitive_label_map)


def main() -> None:
    args = parse_args()

    input_path = args.input_path.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    primitive_surface_index = load_layer_index(args.ontology, "primitive_surface")
    primitive_label_map = export_layer_label_map(primitive_surface_index)

    step_files = collect_step_files(input_path, args.recursive)
    if not step_files:
        raise RuntimeError(f"No STEP files found under: {input_path}")

    manifest_entries = []
    for step_path in step_files:
        output_path = target_graph_path(step_path, input_path, output_dir)
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to replace it.")

        graph_data = convert_step_to_graph(
            step_path=step_path,
            primitive_surface_index=primitive_surface_index,
            primitive_label_map=primitive_label_map,
        )
        write_json(output_path, graph_data)
        manifest_entries.append({
            "sample_id": output_path.stem,
            "source_step_path": str(step_path.resolve()),
            "graph_path": str(output_path.resolve()),
            "node_count": len(graph_data.get("nodes", [])),
            "edge_count": len(graph_data.get("edges", [])),
        })
        print(f"inference graph written: {output_path}")

    build_manifest(manifest_entries, output_dir, primitive_label_map)


if __name__ == "__main__":
    main()
