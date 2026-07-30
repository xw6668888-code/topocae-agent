from __future__ import annotations

from collections import defaultdict


THIN_WALL_SEMANTIC_KEYS = {
    "rib_side_plane",
    "web_side_plane",
}

THICKNESS_TRANSITION_SEMANTIC_KEYS = {
    "pad_side_plane",
    "pad_top_plane",
}

THICKNESS_TRANSITION_PRIMITIVES = {"plane", "cylinder", "cone"}
THICKNESS_TRANSITION_MACROS = {"rectangular_boss"}


def _build_adjacency(edges, node_count):
    adjacency = defaultdict(set)
    for edge in edges or []:
        if not isinstance(edge, list) or len(edge) != 2:
            continue
        src, dst = edge
        if not isinstance(src, int) or not isinstance(dst, int):
            continue
        if 0 <= src < node_count and 0 <= dst < node_count:
            adjacency[src].add(dst)
            adjacency[dst].add(src)
    return adjacency


def _safe_semantic_key(node, semantic_id_to_key):
    semantic_id = node.get("semantic_label", 0)
    return semantic_id_to_key.get(str(semantic_id))


def annotate_derived_cae_labels(graph_data, ontology_maps, idealization_tag_index):
    label_maps = graph_data.get("label_maps", {})
    semantic_map = label_maps.get("face_semantic_subtype", {})
    semantic_id_to_key = semantic_map.get("id_to_key", {})

    macro_id_by_key = ontology_maps.macro_id_by_key
    idealization_id_by_key = idealization_tag_index["key_to_id"]

    thin_wall_id = macro_id_by_key["thin_wall_region"]
    thickness_transition_id = macro_id_by_key["thickness_transition"]
    midsurface_candidate_id = macro_id_by_key["midsurface_candidate"]
    mesh_sensitive_region_id = idealization_id_by_key["mesh_sensitive_region"]
    shell_candidate_id = idealization_id_by_key["shell_candidate"]

    nodes = graph_data.get("nodes", [])
    adjacency = _build_adjacency(graph_data.get("edges", []), len(nodes))

    thin_wall_nodes = set()
    thickness_transition_nodes = set()
    midsurface_candidate_nodes = set()

    for idx, node in enumerate(nodes):
        semantic_key = _safe_semantic_key(node, semantic_id_to_key)
        if semantic_key in THIN_WALL_SEMANTIC_KEYS:
            thin_wall_nodes.add(idx)
            midsurface_candidate_nodes.add(idx)
        if semantic_key in THICKNESS_TRANSITION_SEMANTIC_KEYS:
            thickness_transition_nodes.add(idx)

    for idx in list(thin_wall_nodes):
        for neighbor_idx in adjacency.get(idx, set()):
            neighbor = nodes[neighbor_idx]
            primitive = neighbor.get("primitive_surface")
            macro_key = neighbor.get("macro_feature")
            area = float(neighbor.get("attributes", {}).get("physical", {}).get("area", 0.0))
            if primitive not in THICKNESS_TRANSITION_PRIMITIVES:
                continue
            if area <= 2500.0 or macro_key in THICKNESS_TRANSITION_MACROS:
                thickness_transition_nodes.add(neighbor_idx)

    for idx, node in enumerate(nodes):
        derived_ids = []
        derived_keys = []
        idealization_ids = []
        idealization_keys = []

        if idx in thin_wall_nodes:
            derived_ids.append(thin_wall_id)
            derived_keys.append("thin_wall_region")
        if idx in thickness_transition_nodes:
            derived_ids.append(thickness_transition_id)
            derived_keys.append("thickness_transition")
            idealization_ids.append(mesh_sensitive_region_id)
            idealization_keys.append("mesh_sensitive_region")
        if idx in midsurface_candidate_nodes:
            derived_ids.append(midsurface_candidate_id)
            derived_keys.append("midsurface_candidate")
            idealization_ids.append(shell_candidate_id)
            idealization_keys.append("shell_candidate")

        node["derived_macro_feature_ids"] = derived_ids
        node["derived_macro_features"] = derived_keys
        node["idealization_tag_ids"] = idealization_ids
        node["idealization_tags"] = idealization_keys

    graph_data["derived_label_summary"] = {
        "thin_wall_region_node_count": len(thin_wall_nodes),
        "thickness_transition_node_count": len(thickness_transition_nodes),
        "midsurface_candidate_node_count": len(midsurface_candidate_nodes),
    }
    return graph_data
