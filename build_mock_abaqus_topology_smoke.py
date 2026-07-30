from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


PRIMITIVE_TO_TOPOLOGY_TYPE = {
    "plane": "Planar",
    "cylinder": "Cylindrical/Curved",
    "cone": "Conical/Curved",
    "sphere": "Spherical/Curved",
    "torus": "Toroidal/Curved",
    "bspline_surface": "Freeform/Curved",
    "surface_of_revolution": "Curved",
    "surface_of_extrusion": "Curved",
    "offset_surface": "Curved",
    "other_surface": "Curved",
    "unknown_surface": "Unknown",
}


def parse_args() -> argparse.Namespace:
    current_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Build a shuffled mock Abaqus topology snapshot from inference graph json for smoke testing."
    )
    parser.add_argument("--graph-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--part-name", default="MockPart")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")


def build_bbox(nodes: list[dict]) -> dict:
    centers = [node["attributes"]["physical"]["center_of_mass"] for node in nodes]
    xs = [center[0] for center in centers]
    ys = [center[1] for center in centers]
    zs = [center[2] for center in centers]
    return {
        "x_min": min(xs), "x_max": max(xs),
        "y_min": min(ys), "y_max": max(ys),
        "z_min": min(zs), "z_max": max(zs),
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    graph = load_json(args.graph_path)
    nodes = graph["nodes"]

    old_ids = [node["idx"] for node in nodes]
    new_ids = old_ids[:]
    rng.shuffle(new_ids)
    id_map = {old_id: new_id for old_id, new_id in zip(old_ids, new_ids)}

    mock_faces = []
    for node in nodes:
        attrs = node["attributes"]
        specific = attrs.get("specific", {})
        mock_faces.append({
            "id": id_map[node["idx"]],
            "type": PRIMITIVE_TO_TOPOLOGY_TYPE.get(node.get("primitive_surface", "unknown_surface"), "Unknown"),
            "area": attrs["physical"].get("area", 0.0),
            "center": attrs["physical"].get("center_of_mass", [0.0, 0.0, 0.0]),
            "normal": specific.get("normal", [0.0, 0.0, 0.0]),
            "axis_vector": specific.get("axis", [0.0, 0.0, 0.0]),
            "radius": specific.get("radius", 0.0),
            "depth": 0.0,
            "edge_stats": {"lines": 0, "curves": 0},
        })
    mock_faces.sort(key=lambda item: item["id"])

    neighbors = {str(id_map[node["idx"]]): [] for node in nodes}
    topology_edges = []
    seen_pairs = set()
    for edge_id, edge in enumerate(graph.get("edges", [])):
        if len(edge) != 2:
            continue
        src, dst = edge
        new_src = id_map[src]
        new_dst = id_map[dst]
        if new_dst not in neighbors[str(new_src)]:
            neighbors[str(new_src)].append(new_dst)
        pair = tuple(sorted((new_src, new_dst)))
        if pair not in seen_pairs and new_src != new_dst:
            topology_edges.append({"edge_id": len(topology_edges), "face_pair": [pair[0], pair[1]]})
            seen_pairs.add(pair)

    output = {
        "part_name": args.part_name,
        "source_graph_path": str(args.graph_path.resolve()),
        "mock_topology": True,
        "id_permutation_seed": args.seed,
        "bbox": build_bbox(nodes),
        "neighbors": neighbors,
        "topology_edges": topology_edges,
        "faces": mock_faces,
    }
    write_json(args.output_path, output)
    print(f"mock topology written: {args.output_path}")


if __name__ == "__main__":
    main()
