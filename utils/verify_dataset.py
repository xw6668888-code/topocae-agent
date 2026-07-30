import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

from ontology import load_layer_index


def parse_args():
    current_dir = Path(__file__).resolve().parent
    dataset_root = current_dir.parent / "output"
    return argparse.ArgumentParser(description="Verify generated CAE dataset graphs and label mappings.").parse_args()


def build_parser():
    current_dir = Path(__file__).resolve().parent
    dataset_root = current_dir.parent / "output"
    parser = argparse.ArgumentParser(description="Verify generated CAE dataset graphs and label mappings.")
    parser.add_argument("--graphs-dir", default=str(dataset_root / "graphs"))
    parser.add_argument("--dataset-meta", default=str(dataset_root / "dataset_meta.json"))
    parser.add_argument("--primitive-label-map", default=str(dataset_root / "label_maps" / "primitive_surface_label_map.json"))
    parser.add_argument("--label-map", default=str(dataset_root / "label_maps" / "macro_feature_label_map.json"))
    parser.add_argument("--idealization-label-map", default=str(dataset_root / "label_maps" / "idealization_tag_label_map.json"))
    parser.add_argument("--macro-ontology", default=str(current_dir.parent / "CAE_feature_ontology.yaml"))
    parser.add_argument("--semantic-ontology", default=str(current_dir.parent / "face_semantic_subtype_ontology.yaml"))
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def add_issue(issues, path, message, fail_fast=False):
    issues.append((path, message))
    if fail_fast:
        raise RuntimeError(f"{path}: {message}")


def verify_dataset_meta(meta_path, issues, fail_fast=False):
    if not os.path.exists(meta_path):
        add_issue(issues, meta_path, "dataset_meta.json is missing", fail_fast)
        return None

    meta = load_json(meta_path)
    required_keys = [
        "dataset_name",
        "generated_at_utc",
        "num_samples_requested",
        "macro_feature_ontology_path",
        "semantic_subtype_ontology_path",
        "primitive_surface_label_map_path",
        "idealization_tag_label_map_path",
        "generator_classes",
    ]
    for key in required_keys:
        if key not in meta:
            add_issue(issues, meta_path, f"missing required key '{key}'", fail_fast)
    return meta


def verify_graph(
    graph_path,
    graph_data,
    primitive_external_map,
    macro_external_map,
    semantic_external_map,
    idealization_external_map,
    issues,
    summary,
    fail_fast=False,
):
    nodes = graph_data.get("nodes", [])
    feature_sets = graph_data.get("feature_sets", [])
    embedded_label_maps = graph_data.get("label_maps", {})
    embedded_primitive_map = embedded_label_maps.get("primitive_surface", {})
    embedded_macro_map = embedded_label_maps.get("macro_feature", {})
    embedded_idealization_map = embedded_label_maps.get("idealization_tag", {})

    external_primitive_id_to_key = primitive_external_map["id_to_key"]
    external_macro_id_to_key = macro_external_map["id_to_key"]
    external_macro_key_to_id = macro_external_map["key_to_id"]
    external_semantic_id_to_key = semantic_external_map["id_to_key"]
    external_idealization_id_to_key = idealization_external_map["id_to_key"]

    if not nodes:
        add_issue(issues, graph_path, "graph has no nodes", fail_fast)
        return

    if embedded_macro_map:
        if embedded_macro_map.get("id_to_key") != external_macro_id_to_key:
            add_issue(issues, graph_path, "embedded macro label map does not match ontology label map", fail_fast)
    if embedded_primitive_map:
        if embedded_primitive_map.get("id_to_key") != external_primitive_id_to_key:
            add_issue(issues, graph_path, "embedded primitive label map does not match ontology label map", fail_fast)
    if embedded_idealization_map:
        if embedded_idealization_map.get("id_to_key") != external_idealization_id_to_key:
            add_issue(issues, graph_path, "embedded idealization label map does not match ontology label map", fail_fast)

    stock_macro_id = external_macro_key_to_id.get("stock")
    instance_nodes = defaultdict(list)

    for idx, node in enumerate(nodes):
        summary["node_count"] += 1

        macro_key = node.get("macro_feature")
        macro_id = node.get("macro_feature_id")
        primitive_key = node.get("primitive_surface")
        primitive_id = node.get("primitive_surface_id")
        semantic_id = node.get("semantic_label")
        inst_id = node.get("instance_id")
        derived_macro_ids = node.get("derived_macro_feature_ids", [])
        derived_macro_keys = node.get("derived_macro_features", [])
        idealization_tag_ids = node.get("idealization_tag_ids", [])
        idealization_tags = node.get("idealization_tags", [])

        if macro_key is None or macro_id is None or primitive_key is None or primitive_id is None:
            add_issue(issues, graph_path, f"node {idx} missing primitive/macro feature fields", fail_fast)
            continue

        if str(primitive_id) not in external_primitive_id_to_key:
            add_issue(issues, graph_path, f"node {idx} primitive_surface_id {primitive_id} not in ontology", fail_fast)
        else:
            expected_primitive_key = external_primitive_id_to_key[str(primitive_id)]
            if primitive_key != expected_primitive_key:
                add_issue(
                    issues,
                    graph_path,
                    f"node {idx} primitive_surface '{primitive_key}' does not match primitive_surface_id {primitive_id} -> '{expected_primitive_key}'",
                    fail_fast,
                )

        if str(macro_id) not in external_macro_id_to_key:
            add_issue(issues, graph_path, f"node {idx} macro_feature_id {macro_id} not in ontology", fail_fast)
        else:
            expected_macro_key = external_macro_id_to_key[str(macro_id)]
            if macro_key != expected_macro_key:
                add_issue(
                    issues,
                    graph_path,
                    f"node {idx} macro_feature '{macro_key}' does not match macro_feature_id {macro_id} -> '{expected_macro_key}'",
                    fail_fast,
                )

        if str(semantic_id) not in external_semantic_id_to_key:
            add_issue(issues, graph_path, f"node {idx} semantic_label {semantic_id} not in subtype ontology", fail_fast)

        if len(derived_macro_ids) != len(derived_macro_keys):
            add_issue(issues, graph_path, f"node {idx} derived macro id/key lengths differ", fail_fast)
        for derived_id, derived_key in zip(derived_macro_ids, derived_macro_keys):
            if str(derived_id) not in external_macro_id_to_key:
                add_issue(issues, graph_path, f"node {idx} derived_macro_feature_id {derived_id} not in ontology", fail_fast)
                continue
            expected_derived_key = external_macro_id_to_key[str(derived_id)]
            if derived_key != expected_derived_key:
                add_issue(
                    issues,
                    graph_path,
                    f"node {idx} derived_macro_feature '{derived_key}' does not match derived_macro_feature_id {derived_id} -> '{expected_derived_key}'",
                    fail_fast,
                )

        if len(idealization_tag_ids) != len(idealization_tags):
            add_issue(issues, graph_path, f"node {idx} idealization id/key lengths differ", fail_fast)
        for idealization_id, idealization_key in zip(idealization_tag_ids, idealization_tags):
            if str(idealization_id) not in external_idealization_id_to_key:
                add_issue(issues, graph_path, f"node {idx} idealization_tag_id {idealization_id} not in ontology", fail_fast)
                continue
            expected_idealization_key = external_idealization_id_to_key[str(idealization_id)]
            if idealization_key != expected_idealization_key:
                add_issue(
                    issues,
                    graph_path,
                    f"node {idx} idealization_tag '{idealization_key}' does not match idealization_tag_id {idealization_id} -> '{expected_idealization_key}'",
                    fail_fast,
                )

        if inst_id == 0 and macro_id != stock_macro_id:
            add_issue(issues, graph_path, f"node {idx} has instance_id 0 but non-stock macro_feature_id {macro_id}", fail_fast)
        if inst_id > 0:
            instance_nodes[inst_id].append(idx)

        summary["macro_feature_counter"][macro_key] += 1
        summary["semantic_counter"][semantic_id] += 1
        summary["derived_macro_counter"].update(derived_macro_keys)
        summary["idealization_counter"].update(idealization_tags)

    feature_set_instance_ids = set()
    for fs_idx, feature_set in enumerate(feature_sets):
        summary["feature_set_count"] += 1
        fs_inst_id = feature_set.get("instance_id")
        fs_macro_key = feature_set.get("macro_feature")
        fs_macro_id = feature_set.get("macro_feature_id")
        node_indices = feature_set.get("node_indices", [])

        if fs_inst_id is None or fs_inst_id <= 0:
            add_issue(issues, graph_path, f"feature_set {fs_idx} has invalid instance_id {fs_inst_id}", fail_fast)
            continue

        feature_set_instance_ids.add(fs_inst_id)

        if not node_indices:
            add_issue(issues, graph_path, f"feature_set {fs_idx} is empty", fail_fast)
            continue

        if str(fs_macro_id) not in external_macro_id_to_key:
            add_issue(issues, graph_path, f"feature_set {fs_idx} macro_feature_id {fs_macro_id} not in ontology", fail_fast)
        else:
            expected_macro_key = external_macro_id_to_key[str(fs_macro_id)]
            if fs_macro_key != expected_macro_key:
                add_issue(
                    issues,
                    graph_path,
                    f"feature_set {fs_idx} macro_feature '{fs_macro_key}' does not match macro_feature_id {fs_macro_id} -> '{expected_macro_key}'",
                    fail_fast,
                )

        for node_idx in node_indices:
            if node_idx < 0 or node_idx >= len(nodes):
                add_issue(issues, graph_path, f"feature_set {fs_idx} references invalid node index {node_idx}", fail_fast)
                continue

            node = nodes[node_idx]
            if node.get("instance_id") != fs_inst_id:
                add_issue(
                    issues,
                    graph_path,
                    f"feature_set {fs_idx} node {node_idx} instance_id {node.get('instance_id')} != feature_set instance_id {fs_inst_id}",
                    fail_fast,
                )
            if node.get("macro_feature_id") != fs_macro_id:
                add_issue(
                    issues,
                    graph_path,
                    f"feature_set {fs_idx} node {node_idx} macro_feature_id {node.get('macro_feature_id')} != feature_set macro_feature_id {fs_macro_id}",
                    fail_fast,
                )

    node_instance_ids = set(instance_nodes.keys())
    if node_instance_ids != feature_set_instance_ids:
        add_issue(
            issues,
            graph_path,
            f"instance ids in nodes {sorted(node_instance_ids)} do not match feature_sets {sorted(feature_set_instance_ids)}",
            fail_fast,
        )


def main():
    parser = build_parser()
    args = parser.parse_args()

    issues = []
    summary = {
        "file_count": 0,
        "node_count": 0,
        "feature_set_count": 0,
        "macro_feature_counter": Counter(),
        "semantic_counter": Counter(),
        "derived_macro_counter": Counter(),
        "idealization_counter": Counter(),
    }

    macro_external_map = load_layer_index(args.macro_ontology, "macro_feature")
    primitive_external_map = load_layer_index(args.macro_ontology, "primitive_surface")
    idealization_external_map = load_layer_index(args.macro_ontology, "idealization_tag")
    semantic_external_map = load_layer_index(args.semantic_ontology, "face_semantic_subtype")

    primitive_label_map = load_json(args.primitive_label_map)
    if primitive_label_map.get("primitive_surface", {}).get("id_to_key") != primitive_external_map["id_to_key"]:
        add_issue(issues, args.primitive_label_map, "primitive_surface_label_map.json does not match primitive ontology", args.fail_fast)

    external_label_map = load_json(args.label_map)
    if external_label_map.get("macro_feature", {}).get("id_to_key") != macro_external_map["id_to_key"]:
        add_issue(issues, args.label_map, "macro_feature_label_map.json does not match macro ontology", args.fail_fast)

    idealization_label_map = load_json(args.idealization_label_map)
    if idealization_label_map.get("idealization_tag", {}).get("id_to_key") != idealization_external_map["id_to_key"]:
        add_issue(issues, args.idealization_label_map, "idealization_tag_label_map.json does not match idealization ontology", args.fail_fast)

    verify_dataset_meta(args.dataset_meta, issues, args.fail_fast)

    graph_files = sorted(Path(args.graphs_dir).glob("*.json"))
    if not graph_files:
        add_issue(issues, args.graphs_dir, "no graph json files found", args.fail_fast)

    for graph_path in graph_files:
        summary["file_count"] += 1
        graph_data = load_json(graph_path)
        verify_graph(
            graph_path=str(graph_path),
            graph_data=graph_data,
            primitive_external_map=primitive_external_map,
            macro_external_map=macro_external_map,
            semantic_external_map=semantic_external_map,
            idealization_external_map=idealization_external_map,
            issues=issues,
            summary=summary,
            fail_fast=args.fail_fast,
        )

    print(f"checked_files: {summary['file_count']}")
    print(f"checked_nodes: {summary['node_count']}")
    print(f"checked_feature_sets: {summary['feature_set_count']}")
    print(f"macro_feature_distribution: {dict(summary['macro_feature_counter'])}")
    print(f"semantic_label_distribution: {dict(summary['semantic_counter'])}")
    print(f"derived_macro_feature_distribution: {dict(summary['derived_macro_counter'])}")
    print(f"idealization_tag_distribution: {dict(summary['idealization_counter'])}")

    if issues:
        print("verification_status: FAILED")
        for path, message in issues[:50]:
            print(f"- {path}: {message}")
        if len(issues) > 50:
            print(f"... truncated {len(issues) - 50} additional issues")
        sys.exit(1)

    print("verification_status: OK")


if __name__ == "__main__":
    main()
