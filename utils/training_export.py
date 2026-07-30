import argparse
import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path


NODE_FEATURE_MODE_FULL = "full_v3"
NODE_FEATURE_MODE_ROTATION_INVARIANT = "rotation_invariant_v1"
DEFAULT_NODE_FEATURE_MODE = NODE_FEATURE_MODE_FULL


NODE_FEATURE_SCHEMA = [
    "type_onehot_0",
    "type_onehot_1",
    "type_onehot_2",
    "type_onehot_3",
    "type_onehot_4",
    "type_onehot_5",
    "type_onehot_6",
    "type_onehot_7",
    "type_onehot_8",
    "type_onehot_9",
    "type_onehot_10",
    "area",
    "center_of_mass_x",
    "center_of_mass_y",
    "center_of_mass_z",
    "normal_x",
    "normal_y",
    "normal_z",
    "axis_x",
    "axis_y",
    "axis_z",
    "radius",
    "apex_x",
    "apex_y",
    "apex_z",
    "semi_angle",
    "major_radius",
    "minor_radius",
    "bbox_dx",
    "bbox_dy",
    "bbox_dz",
    "bbox_longest_extent",
    "bbox_middle_extent",
    "bbox_shortest_extent",
    "bbox_diagonal",
    "bbox_aspect_ratio",
    "uv_u_span",
    "uv_v_span",
    "uv_aspect_ratio",
    "projected_long_extent",
    "projected_short_extent",
    "projected_aspect_ratio",
    "projected_area_fill_ratio",
    "vertex_count",
    "edge_count",
    "neighbor_count",
    "neighbor_area_mean",
    "neighbor_area_max",
    "neighbor_projected_aspect_mean",
    "neighbor_long_extent_mean",
]


INVARIANT_NODE_FEATURE_SCHEMA = [
    "type_onehot_0",
    "type_onehot_1",
    "type_onehot_2",
    "type_onehot_3",
    "type_onehot_4",
    "type_onehot_5",
    "type_onehot_6",
    "type_onehot_7",
    "type_onehot_8",
    "type_onehot_9",
    "type_onehot_10",
    "area",
    "radius",
    "semi_angle",
    "major_radius",
    "minor_radius",
    "bbox_longest_extent",
    "bbox_middle_extent",
    "bbox_shortest_extent",
    "bbox_diagonal",
    "bbox_aspect_ratio",
    "uv_long_span",
    "uv_short_span",
    "uv_aspect_ratio",
    "projected_long_extent",
    "projected_short_extent",
    "projected_aspect_ratio",
    "projected_area_fill_ratio",
    "vertex_count",
    "edge_count",
    "neighbor_count",
    "neighbor_area_mean",
    "neighbor_area_max",
    "neighbor_projected_aspect_mean",
    "neighbor_long_extent_mean",
]


EDGE_FEATURE_SCHEMA = [
    "curve_type_id",
    "length",
    "is_line",
    "is_circle",
    "is_ellipse",
    "is_spline",
    "circle_radius",
    "convexity_id",
    "dihedral_angle",
    "adjacent_face_normal_angle",
    "source_primitive_surface_id",
    "target_primitive_surface_id",
    "source_is_plane",
    "target_is_plane",
    "source_is_cylinder",
    "target_is_cylinder",
    "adjacent_plane_loop_role_id",
    "adjacent_plane_loop_role_inner",
    "adjacent_plane_loop_role_outer",
    "adjacent_plane_loop_role_unknown",
    "cylindrical_wall_role_hint_id",
    "cylindrical_wall_role_inner_hint",
    "cylindrical_wall_role_outer_hint",
    "cylindrical_wall_role_ambiguous_hint",
]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def round_value(value, ndigits):
    if isinstance(value, float):
        return round(value, ndigits)
    return value


def round_vector(values, ndigits):
    return [round_value(v, ndigits) for v in values]


def round_specific_dict(specific, ndigits):
    rounded = {}
    for key, value in specific.items():
        if isinstance(value, list):
            rounded[key] = round_vector(value, ndigits)
        else:
            rounded[key] = round_value(value, ndigits)
    return rounded


def node_feature_schema_for_mode(node_feature_mode=DEFAULT_NODE_FEATURE_MODE):
    if node_feature_mode in (None, "", NODE_FEATURE_MODE_FULL):
        return NODE_FEATURE_SCHEMA
    if node_feature_mode == NODE_FEATURE_MODE_ROTATION_INVARIANT:
        return INVARIANT_NODE_FEATURE_SCHEMA
    raise ValueError(f"Unsupported node feature mode: {node_feature_mode}")


def build_numeric_feature_vector(node, ndigits):
    attrs = node["attributes"]
    physical = attrs.get("physical", {})
    specific = attrs.get("specific", {})
    shape_descriptors = attrs.get("shape_descriptors", {})

    type_onehot = attrs.get("type_onehot", [0] * 11)
    center_of_mass = physical.get("center_of_mass", [0.0, 0.0, 0.0])
    normal = specific.get("normal", [0.0, 0.0, 0.0])
    axis = specific.get("axis", [0.0, 0.0, 0.0])
    apex = specific.get("apex", [0.0, 0.0, 0.0])

    vector = (
        list(type_onehot)
        + [physical.get("area", 0.0)]
        + list(center_of_mass)
        + list(normal)
        + list(axis)
        + [specific.get("radius", 0.0)]
        + list(apex)
        + [specific.get("semi_angle", 0.0)]
        + [specific.get("major_radius", 0.0)]
        + [specific.get("minor_radius", 0.0)]
        + [shape_descriptors.get("bbox_dx", 0.0)]
        + [shape_descriptors.get("bbox_dy", 0.0)]
        + [shape_descriptors.get("bbox_dz", 0.0)]
        + [shape_descriptors.get("bbox_longest_extent", 0.0)]
        + [shape_descriptors.get("bbox_middle_extent", 0.0)]
        + [shape_descriptors.get("bbox_shortest_extent", 0.0)]
        + [shape_descriptors.get("bbox_diagonal", 0.0)]
        + [shape_descriptors.get("bbox_aspect_ratio", 0.0)]
        + [shape_descriptors.get("uv_u_span", 0.0)]
        + [shape_descriptors.get("uv_v_span", 0.0)]
        + [shape_descriptors.get("uv_aspect_ratio", 0.0)]
        + [shape_descriptors.get("projected_long_extent", 0.0)]
        + [shape_descriptors.get("projected_short_extent", 0.0)]
        + [shape_descriptors.get("projected_aspect_ratio", 0.0)]
        + [shape_descriptors.get("projected_area_fill_ratio", 0.0)]
        + [shape_descriptors.get("vertex_count", 0.0)]
        + [shape_descriptors.get("edge_count", 0.0)]
        + [shape_descriptors.get("neighbor_count", 0.0)]
        + [shape_descriptors.get("neighbor_area_mean", 0.0)]
        + [shape_descriptors.get("neighbor_area_max", 0.0)]
        + [shape_descriptors.get("neighbor_projected_aspect_mean", 0.0)]
        + [shape_descriptors.get("neighbor_long_extent_mean", 0.0)]
    )
    return [round_value(value, ndigits) for value in vector]


def _as_float(value, default=0.0):
    try:
        if value is None:
            return default
        value = float(value)
        if not math.isfinite(value):
            return default
        return value
    except (TypeError, ValueError):
        return default


def _sorted_extents(values):
    return sorted((_as_float(value) for value in values), reverse=True)


def _aspect_ratio(long_extent, short_extent, fallback=0.0):
    long_extent = _as_float(long_extent)
    short_extent = abs(_as_float(short_extent))
    if short_extent > 1e-9:
        return abs(long_extent) / short_extent
    return _as_float(fallback)


def build_rotation_invariant_node_feature_vector(node, ndigits):
    attrs = node["attributes"]
    physical = attrs.get("physical", {})
    specific = attrs.get("specific", {})
    shape_descriptors = attrs.get("shape_descriptors", {})

    type_onehot = attrs.get("type_onehot", [0] * 11)
    bbox_extents = _sorted_extents([
        shape_descriptors.get("bbox_dx", 0.0),
        shape_descriptors.get("bbox_dy", 0.0),
        shape_descriptors.get("bbox_dz", 0.0),
    ])
    if sum(abs(value) for value in bbox_extents) <= 1e-9:
        bbox_extents = [
            _as_float(shape_descriptors.get("bbox_longest_extent", 0.0)),
            _as_float(shape_descriptors.get("bbox_middle_extent", 0.0)),
            _as_float(shape_descriptors.get("bbox_shortest_extent", 0.0)),
        ]

    uv_spans = _sorted_extents([
        shape_descriptors.get("uv_u_span", 0.0),
        shape_descriptors.get("uv_v_span", 0.0),
    ])
    uv_aspect_ratio = _aspect_ratio(
        uv_spans[0],
        uv_spans[1],
        shape_descriptors.get("uv_aspect_ratio", 0.0),
    )

    vector = (
        list(type_onehot)
        + [physical.get("area", 0.0)]
        + [specific.get("radius", 0.0)]
        + [specific.get("semi_angle", 0.0)]
        + [specific.get("major_radius", 0.0)]
        + [specific.get("minor_radius", 0.0)]
        + [bbox_extents[0]]
        + [bbox_extents[1]]
        + [bbox_extents[2]]
        + [shape_descriptors.get("bbox_diagonal", 0.0)]
        + [_aspect_ratio(bbox_extents[0], bbox_extents[-1], shape_descriptors.get("bbox_aspect_ratio", 0.0))]
        + [uv_spans[0]]
        + [uv_spans[1]]
        + [uv_aspect_ratio]
        + [shape_descriptors.get("projected_long_extent", 0.0)]
        + [shape_descriptors.get("projected_short_extent", 0.0)]
        + [shape_descriptors.get("projected_aspect_ratio", 0.0)]
        + [shape_descriptors.get("projected_area_fill_ratio", 0.0)]
        + [shape_descriptors.get("vertex_count", 0.0)]
        + [shape_descriptors.get("edge_count", 0.0)]
        + [shape_descriptors.get("neighbor_count", 0.0)]
        + [shape_descriptors.get("neighbor_area_mean", 0.0)]
        + [shape_descriptors.get("neighbor_area_max", 0.0)]
        + [shape_descriptors.get("neighbor_projected_aspect_mean", 0.0)]
        + [shape_descriptors.get("neighbor_long_extent_mean", 0.0)]
    )
    return [round_value(_as_float(value), ndigits) for value in vector]


def build_node_feature_vector(node, ndigits, node_feature_mode=DEFAULT_NODE_FEATURE_MODE):
    if node_feature_mode in (None, "", NODE_FEATURE_MODE_FULL):
        return build_numeric_feature_vector(node, ndigits)
    if node_feature_mode == NODE_FEATURE_MODE_ROTATION_INVARIANT:
        return build_rotation_invariant_node_feature_vector(node, ndigits)
    raise ValueError(f"Unsupported node feature mode: {node_feature_mode}")


def build_numeric_edge_feature_vector(edge_attr, ndigits):
    edge_attr = edge_attr or {}
    vector = [edge_attr.get(key, 0.0) for key in EDGE_FEATURE_SCHEMA]
    return [round_value(float(value), ndigits) for value in vector]


def build_same_instance_edge_supervision(edges, nodes):
    instance_ids = {int(node["idx"]): int(node.get("instance_id", 0)) for node in nodes}
    labels = []
    mask = []
    for edge in edges:
        if len(edge) != 2:
            labels.append(-1)
            mask.append(0.0)
            continue
        src, dst = int(edge[0]), int(edge[1])
        src_instance = instance_ids.get(src, 0)
        dst_instance = instance_ids.get(dst, 0)
        if src_instance == 0 and dst_instance == 0:
            labels.append(-1)
            mask.append(0.0)
        elif src_instance > 0 and src_instance == dst_instance:
            labels.append(1)
            mask.append(1.0)
        else:
            labels.append(0)
            mask.append(1.0)
    return labels, mask


def build_manifest_record(graph_path, graph_data):
    nodes = graph_data.get("nodes", [])
    feature_sets = graph_data.get("feature_sets", [])
    sample_id = Path(graph_path).stem
    macro_feature_ids = sorted({node.get("macro_feature_id", 0) for node in nodes if node.get("macro_feature_id", 0) > 0})
    macro_features = sorted({node.get("macro_feature", "stock") for node in nodes if node.get("macro_feature", "stock") != "stock"})
    semantic_labels = sorted({node.get("semantic_label", 0) for node in nodes if node.get("semantic_label", 0) > 0})
    derived_macro_feature_ids = sorted({
        derived_id
        for node in nodes
        for derived_id in node.get("derived_macro_feature_ids", [])
        if derived_id > 0
    })
    derived_macro_features = sorted({
        derived_key
        for node in nodes
        for derived_key in node.get("derived_macro_features", [])
        if derived_key
    })
    idealization_tag_ids = sorted({
        tag_id
        for node in nodes
        for tag_id in node.get("idealization_tag_ids", [])
        if tag_id > 0
    })
    idealization_tags = sorted({
        tag_key
        for node in nodes
        for tag_key in node.get("idealization_tags", [])
        if tag_key
    })

    return {
        "sample_id": sample_id,
        "graph_path": str(graph_path),
        "step_path": str(Path(graph_path).parent.parent / "steps" / f"{sample_id}.step"),
        "node_count": len(nodes),
        "edge_count": len(graph_data.get("edges", [])),
        "feature_set_count": len(feature_sets),
        "macro_feature_ids_present": macro_feature_ids,
        "macro_features_present": macro_features,
        "semantic_labels_present": semantic_labels,
        "derived_macro_feature_ids_present": derived_macro_feature_ids,
        "derived_macro_features_present": derived_macro_features,
        "idealization_tag_ids_present": idealization_tag_ids,
        "idealization_tags_present": idealization_tags,
        "has_nonstock": any(node.get("macro_feature_id", 0) > 0 for node in nodes),
    }


def split_samples(manifest_records, split_ratios, seed):
    rng = random.Random(seed if seed is not None else 0)
    records = list(manifest_records)
    rng.shuffle(records)

    macro_frequency = Counter()
    for record in records:
        macro_frequency.update(record["macro_feature_ids_present"])

    def rarity_score(record):
        if not record["macro_feature_ids_present"]:
            return 0.0
        return sum(1.0 / max(macro_frequency[m], 1) for m in record["macro_feature_ids_present"])

    records.sort(key=rarity_score, reverse=True)

    split_names = list(split_ratios.keys())
    total = len(records)
    target_counts = {}
    remaining = total
    for index, split_name in enumerate(split_names):
        if index == len(split_names) - 1:
            target_counts[split_name] = remaining
        else:
            count = int(round(total * split_ratios[split_name]))
            count = min(count, remaining)
            target_counts[split_name] = count
            remaining -= count

    assigned = {name: [] for name in split_names}
    split_macro_presence = {name: Counter() for name in split_names}

    for record in records:
        best_split = None
        best_score = None
        record_macros = record["macro_feature_ids_present"]

        for split_name in split_names:
            split_fill = len(assigned[split_name]) / max(target_counts[split_name], 1)
            missing_macro_bonus = sum(1 for macro_id in record_macros if split_macro_presence[split_name][macro_id] == 0)
            soft_overflow = max(0, len(assigned[split_name]) - target_counts[split_name])
            score = (missing_macro_bonus * 3.0) - split_fill - soft_overflow
            if best_score is None or score > best_score:
                best_score = score
                best_split = split_name

        assigned[best_split].append(record)
        split_macro_presence[best_split].update(record_macros)

    return assigned


def build_gnn_record(sample_record, graph_data, ndigits, node_feature_mode=DEFAULT_NODE_FEATURE_MODE):
    nodes = graph_data["nodes"]
    feature_sets = sorted(graph_data.get("feature_sets", []), key=lambda item: item["instance_id"])
    edges = graph_data.get("edges", [])
    edge_attributes = list(graph_data.get("edge_attributes", []))
    if len(edge_attributes) < len(edges):
        edge_attributes.extend({} for _ in range(len(edges) - len(edge_attributes)))
    elif len(edge_attributes) > len(edges):
        edge_attributes = edge_attributes[:len(edges)]
    edge_same_instance_labels, edge_supervision_mask = build_same_instance_edge_supervision(edges, nodes)
    return {
        "sample_id": sample_record["sample_id"],
        "source_graph_path": sample_record["graph_path"],
        "source_step_path": sample_record["step_path"],
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "edge_index": edges,
        "edge_features": [
            build_numeric_edge_feature_vector(edge_attr, ndigits)
            for edge_attr in edge_attributes
        ],
        "edge_feature_schema": EDGE_FEATURE_SCHEMA,
        "edge_same_instance_labels": edge_same_instance_labels,
        "edge_supervision_mask": edge_supervision_mask,
        "node_feature_mode": node_feature_mode,
        "node_feature_schema": node_feature_schema_for_mode(node_feature_mode),
        "node_features": [
            build_node_feature_vector(node, ndigits, node_feature_mode=node_feature_mode)
            for node in nodes
        ],
        "primitive_surface_ids": [node.get("primitive_surface_id", 0) for node in nodes],
        "macro_feature_ids": [node.get("macro_feature_id", 0) for node in nodes],
        "semantic_subtype_ids": [node.get("semantic_label", 0) for node in nodes],
        "derived_macro_feature_ids_per_node": [node.get("derived_macro_feature_ids", []) for node in nodes],
        "idealization_tag_ids_per_node": [node.get("idealization_tag_ids", []) for node in nodes],
        "instance_ids": [node.get("instance_id", 0) for node in nodes],
        "feature_sets": feature_sets,
        "derived_label_summary": graph_data.get("derived_label_summary", {}),
    }


def build_llm_record(sample_record, graph_data, ndigits):
    semantic_label_map = graph_data["label_maps"]["face_semantic_subtype"]["id_to_key"]

    user_payload = {
        "sample_id": sample_record["sample_id"],
        "task": "Assign a macro_feature and semantic_subtype to each face, then group non-stock faces by instance_id.",
        "nodes": [],
        "edges": graph_data.get("edges", []),
    }
    for node in graph_data["nodes"]:
        attrs = node["attributes"]
        user_payload["nodes"].append({
            "idx": node["idx"],
            "primitive_surface": node.get("primitive_surface", "unknown_surface"),
            "primitive_surface_id": node.get("primitive_surface_id", 0),
            "area": round_value(attrs.get("physical", {}).get("area", 0.0), ndigits),
            "center_of_mass": round_vector(attrs.get("physical", {}).get("center_of_mass", [0.0, 0.0, 0.0]), ndigits),
            "specific": round_specific_dict(attrs.get("specific", {}), ndigits),
        })

    assistant_payload = {
        "sample_id": sample_record["sample_id"],
        "faces": [],
        "feature_instances": [],
    }
    for node in graph_data["nodes"]:
        semantic_id = node.get("semantic_label", 0)
        assistant_payload["faces"].append({
            "idx": node["idx"],
            "macro_feature": node.get("macro_feature", "stock"),
            "macro_feature_id": node.get("macro_feature_id", 0),
            "semantic_subtype": semantic_label_map[str(semantic_id)],
            "semantic_subtype_id": semantic_id,
            "derived_macro_features": node.get("derived_macro_features", []),
            "derived_macro_feature_ids": node.get("derived_macro_feature_ids", []),
            "idealization_tags": node.get("idealization_tags", []),
            "idealization_tag_ids": node.get("idealization_tag_ids", []),
            "instance_id": node.get("instance_id", 0),
        })
    for feature_set in sorted(graph_data.get("feature_sets", []), key=lambda item: item["instance_id"]):
        assistant_payload["feature_instances"].append({
            "instance_id": feature_set["instance_id"],
            "macro_feature": feature_set["macro_feature"],
            "macro_feature_id": feature_set.get("macro_feature_id", 0),
            "face_indices": feature_set["node_indices"],
        })

    return {
        "messages": [
            {
                "role": "system",
                "content": "You label CAD/CAE B-Rep faces. Return strict JSON with per-face labels and feature instance groups.",
            },
            {
                "role": "user",
                "content": json.dumps(user_payload, ensure_ascii=False, separators=(",", ":")),
            },
            {
                "role": "assistant",
                "content": json.dumps(assistant_payload, ensure_ascii=False, separators=(",", ":")),
            },
        ]
    }


def build_report(split_assignments):
    report = {
        "total_samples": sum(len(records) for records in split_assignments.values()),
        "splits": {},
    }
    for split_name, records in split_assignments.items():
        macro_counter = Counter()
        feature_instance_counter = Counter()
        semantic_counter = Counter()
        derived_macro_counter = Counter()
        idealization_counter = Counter()
        node_total = 0
        feature_set_total = 0
        for record in records:
            graph_data = load_json(record["graph_path"])
            node_total += len(graph_data.get("nodes", []))
            feature_set_total += len(graph_data.get("feature_sets", []))
            macro_counter.update(node["macro_feature"] for node in graph_data.get("nodes", []))
            semantic_counter.update(str(node["semantic_label"]) for node in graph_data.get("nodes", []))
            for node in graph_data.get("nodes", []):
                derived_macro_counter.update(node.get("derived_macro_features", []))
                idealization_counter.update(node.get("idealization_tags", []))
            feature_instance_counter.update(feature_set["macro_feature"] for feature_set in graph_data.get("feature_sets", []))
        report["splits"][split_name] = {
            "sample_count": len(records),
            "node_count": node_total,
            "feature_set_count": feature_set_total,
            "macro_feature_distribution": dict(macro_counter),
            "feature_instance_distribution": dict(feature_instance_counter),
            "semantic_subtype_distribution": dict(semantic_counter),
            "derived_macro_feature_distribution": dict(derived_macro_counter),
            "idealization_tag_distribution": dict(idealization_counter),
        }
    return report


def export_training_package(
    output_dir,
    split_ratios,
    seed=None,
    llm_round_ndigits=4,
    gnn_round_ndigits=6,
    node_feature_mode=DEFAULT_NODE_FEATURE_MODE,
):
    output_dir = os.path.abspath(output_dir)
    graphs_dir = os.path.join(output_dir, "graphs")
    manifests_dir = os.path.join(output_dir, "manifests")
    splits_dir = os.path.join(output_dir, "splits")
    reports_dir = os.path.join(output_dir, "reports")
    training_dir = os.path.join(output_dir, "training")
    gnn_dir = os.path.join(training_dir, "gnn")
    llm_dir = os.path.join(training_dir, "llm")
    metadata_dir = os.path.join(training_dir, "metadata")

    graph_files = sorted(Path(graphs_dir).glob("*.json"))
    manifest_records = []
    for graph_path in graph_files:
        graph_data = load_json(graph_path)
        manifest_records.append(build_manifest_record(str(graph_path), graph_data))

    split_assignments = split_samples(manifest_records, split_ratios, seed)

    write_jsonl(os.path.join(manifests_dir, "sample_index.jsonl"), manifest_records)
    for split_name, records in split_assignments.items():
        write_json(os.path.join(splits_dir, f"{split_name}.json"), [record["sample_id"] for record in records])

    for split_name, records in split_assignments.items():
        gnn_rows = []
        llm_rows = []
        for record in records:
            graph_data = load_json(record["graph_path"])
            gnn_rows.append(
                build_gnn_record(
                    record,
                    graph_data,
                    gnn_round_ndigits,
                    node_feature_mode=node_feature_mode,
                )
            )
            llm_rows.append(build_llm_record(record, graph_data, llm_round_ndigits))
        write_jsonl(os.path.join(gnn_dir, f"{split_name}.jsonl"), gnn_rows)
        write_jsonl(os.path.join(llm_dir, f"{split_name}.jsonl"), llm_rows)

    metadata = {
        "node_feature_mode": node_feature_mode,
        "node_feature_schema": node_feature_schema_for_mode(node_feature_mode),
        "edge_feature_schema": EDGE_FEATURE_SCHEMA,
        "split_ratios": split_ratios,
        "random_seed": seed,
        "formats": {
            "gnn": {
                "description": "One graph sample per line for graph neural network training.",
                "target_heads": [
                    "macro_feature_ids",
                    "semantic_subtype_ids",
                    "derived_macro_feature_ids_per_node",
                    "idealization_tag_ids_per_node",
                    "instance_ids",
                ],
            },
            "llm": {
                "description": "OpenAI-style messages JSONL for supervised fine-tuning on face labeling and instance grouping.",
                "target_schema": ["faces", "feature_instances"],
            },
        },
    }
    write_json(os.path.join(metadata_dir, "training_metadata.json"), metadata)
    write_json(os.path.join(reports_dir, "dataset_report.json"), build_report(split_assignments))


def parse_args():
    parser = argparse.ArgumentParser(description="Export training-ready corpora from generated graph json files.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--llm-round-ndigits", type=int, default=4)
    parser.add_argument("--gnn-round-ndigits", type=int, default=6)
    parser.add_argument(
        "--node-feature-mode",
        choices=[NODE_FEATURE_MODE_FULL, NODE_FEATURE_MODE_ROTATION_INVARIANT],
        default=DEFAULT_NODE_FEATURE_MODE,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    split_total = args.train_ratio + args.val_ratio + args.test_ratio
    split_ratios = {
        "train": args.train_ratio / split_total,
        "val": args.val_ratio / split_total,
        "test": args.test_ratio / split_total,
    }
    export_training_package(
        output_dir=args.output_dir,
        split_ratios=split_ratios,
        seed=args.seed,
        llm_round_ndigits=args.llm_round_ndigits,
        gnn_round_ndigits=args.gnn_round_ndigits,
        node_feature_mode=args.node_feature_mode,
    )


if __name__ == "__main__":
    main()
