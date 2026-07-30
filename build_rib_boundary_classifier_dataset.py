from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


RIB_MEMBER_LABEL = "rib_member"
NOT_RIB_MEMBER_LABEL = "not_rib_member"
RIB_MACROS = {"rib", "web", "rib_web_candidate"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a candidate-face rib boundary classifier dataset from labeled B-Rep graphs."
    )
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-records-per-label", type=int, default=0)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")


def collect_json_files(root: Path, recursive: bool) -> list[Path]:
    globber = root.rglob if recursive else root.glob
    return sorted(path for path in globber("*.json") if path.is_file())


def build_rib_boundary_records_for_graph(
    *,
    sample_id: str,
    graph_data: dict,
    graph_path: Path,
) -> list[dict]:
    truth_instances = build_rib_truth_instances(graph_data)
    records: list[dict] = []
    seen: set[tuple[str, tuple[int, ...], int]] = set()

    for instance in truth_instances:
        face_indices = sorted(instance["face_indices"])
        face_set = set(face_indices)
        if len(face_indices) >= 2:
            for candidate_face_idx in face_indices:
                seed_faces = sorted(face_set - {candidate_face_idx})
                key = (RIB_MEMBER_LABEL, tuple(seed_faces), candidate_face_idx)
                if key in seen:
                    continue
                seen.add(key)
                records.append(
                    make_record(
                        sample_id=sample_id,
                        graph_path=graph_path,
                        source="truth_leave_one_out",
                        label=RIB_MEMBER_LABEL,
                        seed_face_indices=seed_faces,
                        candidate_face_idx=candidate_face_idx,
                        graph_data=graph_data,
                        instance=instance,
                    )
                )

        negative_seed_sets = []
        if len(face_indices) >= 2:
            for dropped_face_idx in face_indices:
                seed_faces = sorted(face_set - {dropped_face_idx})
                if seed_faces:
                    negative_seed_sets.append(seed_faces)
        else:
            negative_seed_sets.append(face_indices)

        for candidate_face_idx in sorted(adjacent_faces(graph_data, face_set) - face_set):
            for seed_faces in negative_seed_sets:
                if not candidate_to_seed_edges(graph_data, candidate_face_idx, set(seed_faces)):
                    continue
                key = (NOT_RIB_MEMBER_LABEL, tuple(seed_faces), candidate_face_idx)
                if key in seen:
                    continue
                seen.add(key)
                records.append(
                    make_record(
                        sample_id=sample_id,
                        graph_path=graph_path,
                        source="adjacent_truth_negative",
                        label=NOT_RIB_MEMBER_LABEL,
                        seed_face_indices=seed_faces,
                        candidate_face_idx=candidate_face_idx,
                        graph_data=graph_data,
                        instance=instance,
                    )
                )

    records.sort(key=lambda item: (item["sample_id"], item["source"], item["seed_face_indices"], item["candidate_face_idx"]))
    return records


def build_rib_truth_instances(graph_data: dict) -> list[dict]:
    instances = []
    for feature_set in graph_data.get("feature_sets", []) or []:
        macro_feature = feature_set.get("macro_feature")
        if macro_feature not in RIB_MACROS:
            continue
        face_indices = sorted(int(idx) for idx in feature_set.get("node_indices", []) if idx is not None)
        if not face_indices:
            continue
        instances.append(
            {
                "macro_feature": macro_feature,
                "instance_id": feature_set.get("instance_id"),
                "face_indices": face_indices,
            }
        )
    return instances


def make_record(
    *,
    sample_id: str,
    graph_path: Path,
    source: str,
    label: str,
    seed_face_indices: list[int],
    candidate_face_idx: int,
    graph_data: dict,
    instance: dict,
) -> dict:
    return {
        "sample_id": sample_id,
        "graph_path": str(graph_path),
        "source": source,
        "label": label,
        "truth_macro": instance.get("macro_feature"),
        "truth_instance_id": instance.get("instance_id"),
        "seed_face_indices": seed_face_indices,
        "candidate_face_idx": int(candidate_face_idx),
        "features": extract_rib_boundary_features(graph_data, seed_face_indices, int(candidate_face_idx)),
    }


def extract_rib_boundary_features(graph_data: dict, seed_face_indices: list[int], candidate_face_idx: int) -> dict[str, float]:
    node_by_idx = node_index(graph_data)
    seed_set = {int(idx) for idx in seed_face_indices}
    candidate = node_by_idx.get(int(candidate_face_idx), {})
    seed_nodes = [node_by_idx[idx] for idx in sorted(seed_set) if idx in node_by_idx]
    model_area = sum(max(face_area(node), 0.0) for node in node_by_idx.values())
    seed_area = sum(face_area(node) for node in seed_nodes)
    candidate_area = face_area(candidate)
    relation_edges = candidate_to_seed_edges(graph_data, int(candidate_face_idx), seed_set)

    candidate_primitive = str(candidate.get("primitive_surface") or "").lower()
    candidate_descriptors = shape_descriptors(candidate)
    seed_plane_normals = [
        vector
        for vector in (normal_vector(node) for node in seed_nodes)
        if vector is not None
    ]
    candidate_normal = normal_vector(candidate)
    parallel_count = 0
    orthogonal_count = 0
    if candidate_normal is not None:
        for seed_normal in seed_plane_normals:
            dot = abs(dot_product(candidate_normal, seed_normal))
            if dot >= 0.92:
                parallel_count += 1
            if dot <= 0.18:
                orthogonal_count += 1

    feature = {
        "seed_face_count": float(len(seed_set)),
        "seed_area": float(seed_area),
        "seed_area_ratio": safe_div(seed_area, model_area),
        "candidate_area": float(candidate_area),
        "candidate_area_ratio": safe_div(candidate_area, model_area),
        "candidate_to_seed_area_ratio": safe_div(candidate_area, seed_area),
        "candidate_model_area_ratio": safe_div(candidate_area, model_area),
        "candidate_is_plane": 1.0 if candidate_primitive == "plane" else 0.0,
        "candidate_is_cylinder": 1.0 if candidate_primitive == "cylinder" else 0.0,
        "candidate_is_cone": 1.0 if candidate_primitive == "cone" else 0.0,
        "candidate_is_curved": 1.0 if candidate_primitive in {"cylinder", "cone", "sphere", "torus"} else 0.0,
        "candidate_projected_aspect_ratio": float(candidate_descriptors.get("projected_aspect_ratio", 0.0) or 0.0),
        "candidate_bbox_aspect_ratio": float(candidate_descriptors.get("bbox_aspect_ratio", 0.0) or 0.0),
        "candidate_bbox_longest_extent": float(candidate_descriptors.get("bbox_longest_extent", 0.0) or 0.0),
        "candidate_bbox_shortest_extent": float(candidate_descriptors.get("bbox_shortest_extent", 0.0) or 0.0),
        "candidate_neighbor_count": float(len(adjacent_faces(graph_data, {int(candidate_face_idx)}))),
        "candidate_to_seed_edge_count": float(len(relation_edges)),
        "candidate_to_seed_line_edge_count": float(count_edges(relation_edges, "curve_type", "line")),
        "candidate_to_seed_circle_edge_count": float(count_edges(relation_edges, "curve_type", "circle")),
        "candidate_to_seed_convex_edge_count": float(count_edges(relation_edges, "convexity", "convex")),
        "candidate_to_seed_concave_edge_count": float(count_edges(relation_edges, "convexity", "concave")),
        "candidate_to_seed_smooth_edge_count": float(count_edges(relation_edges, "convexity", "smooth")),
        "candidate_to_seed_total_edge_length": float(sum(edge_length(edge) for edge in relation_edges)),
        "candidate_to_seed_max_edge_length": float(max((edge_length(edge) for edge in relation_edges), default=0.0)),
        "candidate_to_seed_max_line_length": float(max((edge_length(edge) for edge in relation_edges if edge_curve(edge) == "line"), default=0.0)),
        "candidate_parallel_seed_plane_count": float(parallel_count),
        "candidate_orthogonal_seed_plane_count": float(orthogonal_count),
        "candidate_has_seed_contact": 1.0 if relation_edges else 0.0,
    }
    feature["candidate_to_seed_edge_length_ratio"] = safe_div(
        feature["candidate_to_seed_total_edge_length"],
        feature["candidate_bbox_longest_extent"],
    )
    return feature


def node_index(graph_data: dict) -> dict[int, dict]:
    indexed = {}
    for idx, node in enumerate(graph_data.get("nodes", []) or []):
        node_idx = int(node.get("idx", idx))
        indexed[node_idx] = node
    return indexed


def face_area(node: dict) -> float:
    physical = node.get("attributes", {}).get("physical", {}) if isinstance(node, dict) else {}
    if "area" in physical:
        return float(physical.get("area") or 0.0)
    descriptors = shape_descriptors(node)
    return float(descriptors.get("area", 0.0) or 0.0)


def shape_descriptors(node: dict) -> dict:
    if not isinstance(node, dict):
        return {}
    return node.get("attributes", {}).get("shape_descriptors", {}) or {}


def normal_vector(node: dict) -> list[float] | None:
    specific = node.get("attributes", {}).get("specific", {}) if isinstance(node, dict) else {}
    value = specific.get("normal")
    if not isinstance(value, list) or len(value) != 3:
        return None
    return [float(item) for item in value]


def dot_product(left: list[float], right: list[float]) -> float:
    return sum(l * r for l, r in zip(left, right))


def adjacent_faces(graph_data: dict, face_indices: set[int]) -> set[int]:
    neighbors: set[int] = set()
    for src, dst in iter_edge_pairs(graph_data):
        if src in face_indices and dst not in face_indices:
            neighbors.add(dst)
        if dst in face_indices and src not in face_indices:
            neighbors.add(src)
    return neighbors


def iter_edge_pairs(graph_data: dict):
    for index, edge in enumerate(graph_data.get("edges", []) or []):
        if isinstance(edge, (list, tuple)) and len(edge) >= 2:
            yield int(edge[0]), int(edge[1])
            continue
        attr = edge_attribute_at(graph_data, index)
        if "source_face" in attr and "target_face" in attr:
            yield int(attr["source_face"]), int(attr["target_face"])


def candidate_to_seed_edges(graph_data: dict, candidate_face_idx: int, seed_set: set[int]) -> list[dict]:
    directed = [
        attr
        for attr in iter_edge_attributes(graph_data)
        if int(attr.get("source_face", -1)) == candidate_face_idx and int(attr.get("target_face", -1)) in seed_set
    ]
    if directed:
        return directed
    return [
        attr
        for attr in iter_edge_attributes(graph_data)
        if int(attr.get("target_face", -1)) == candidate_face_idx and int(attr.get("source_face", -1)) in seed_set
    ]


def iter_edge_attributes(graph_data: dict):
    edge_attributes = graph_data.get("edge_attributes", []) or []
    edges = graph_data.get("edges", []) or []
    for index, attr in enumerate(edge_attributes):
        item = dict(attr or {})
        if ("source_face" not in item or "target_face" not in item) and index < len(edges):
            edge = edges[index]
            if isinstance(edge, (list, tuple)) and len(edge) >= 2:
                item.setdefault("source_face", int(edge[0]))
                item.setdefault("target_face", int(edge[1]))
        yield item


def edge_attribute_at(graph_data: dict, index: int) -> dict:
    attrs = graph_data.get("edge_attributes", []) or []
    if 0 <= index < len(attrs):
        return attrs[index] or {}
    return {}


def edge_curve(edge: dict) -> str:
    return str(edge.get("curve_type") or "").lower()


def count_edges(edges: list[dict], key: str, value: str) -> int:
    value = value.lower()
    return sum(1 for edge in edges if str(edge.get(key) or "").lower() == value)


def edge_length(edge: dict) -> float:
    return float(edge.get("length", 0.0) or 0.0)


def safe_div(numerator: float, denominator: float) -> float:
    denominator = float(denominator or 0.0)
    if abs(denominator) < 1e-12:
        return 0.0
    return float(numerator) / denominator


def split_rib_boundary_records(
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
    return splits


def stable_unit_interval(value: str) -> float:
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def limit_records_per_label(records: list[dict], max_records_per_label: int, seed: int) -> list[dict]:
    if max_records_per_label <= 0:
        return records
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_label[record["label"]].append(record)
    limited = []
    for label, label_records in sorted(by_label.items()):
        rng.shuffle(label_records)
        limited.extend(label_records[:max_records_per_label])
    limited.sort(key=lambda item: (item["sample_id"], item["source"], item["seed_face_indices"], item["candidate_face_idx"]))
    return limited


def write_rib_boundary_dataset(records: list[dict], splits: dict[str, list[dict]], output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_schema = sorted({key for record in records for key in record.get("features", {}).keys()})
    label_to_id = {NOT_RIB_MEMBER_LABEL: 0, RIB_MEMBER_LABEL: 1}
    id_to_label = {str(label_id): label for label, label_id in label_to_id.items()}
    split_counts = {}
    label_counts = Counter(record["label"] for record in records)

    for split_name in ["train", "val", "test"]:
        split_records = []
        for record in splits.get(split_name, []):
            enriched = dict(record)
            enriched["label_id"] = int(label_to_id[record["label"]])
            enriched["feature_schema"] = feature_schema
            enriched["feature_vector"] = [
                float(record.get("features", {}).get(feature_name, 0.0))
                for feature_name in feature_schema
            ]
            split_records.append(enriched)
        split_counts[split_name] = len(split_records)
        with (output_dir / f"{split_name}.jsonl").open("w", encoding="utf-8") as fp:
            for record in split_records:
                fp.write(json.dumps(record, ensure_ascii=False) + "\n")

    metadata = {
        "schema": "rib_boundary_classifier_dataset_v1",
        "feature_schema": feature_schema,
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "record_count": len(records),
        "split_counts": split_counts,
        "label_counts": dict(sorted(label_counts.items())),
    }
    write_json(output_dir / "metadata.json", metadata)
    return metadata


def build_dataset_from_paths(
    *,
    graph_dir: Path,
    output_dir: Path,
    recursive: bool,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    max_records_per_label: int,
) -> dict:
    records = []
    graph_count = 0
    rib_graph_count = 0
    for graph_path in collect_json_files(graph_dir, recursive):
        graph_data = load_json(graph_path)
        graph_count += 1
        graph_records = build_rib_boundary_records_for_graph(
            sample_id=graph_path.stem,
            graph_data=graph_data,
            graph_path=graph_path,
        )
        if graph_records:
            rib_graph_count += 1
        records.extend(graph_records)

    records = limit_records_per_label(records, max_records_per_label, seed)
    splits = split_rib_boundary_records(records, seed=seed, train_ratio=train_ratio, val_ratio=val_ratio)
    metadata = write_rib_boundary_dataset(records, splits, output_dir)
    metadata["graph_dir"] = str(graph_dir)
    metadata["graph_count"] = graph_count
    metadata["rib_graph_count"] = rib_graph_count
    write_json(output_dir / "metadata.json", metadata)
    return metadata


def main() -> None:
    args = parse_args()
    metadata = build_dataset_from_paths(
        graph_dir=args.graph_dir,
        output_dir=args.output_dir,
        recursive=args.recursive,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        max_records_per_label=args.max_records_per_label,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "record_count": metadata["record_count"],
                "split_counts": metadata["split_counts"],
                "label_counts": metadata["label_counts"],
                "rib_graph_count": metadata["rib_graph_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

