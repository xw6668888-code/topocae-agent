from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Iterable, Mapping, Sequence


def predicted_same_instance_components(
    node_count: int,
    edges: Sequence[Sequence[int]],
    probabilities: Sequence[float],
    threshold: float,
) -> tuple[list[set[int]], dict[int, int]]:
    if node_count < 0:
        raise ValueError("node_count must be non-negative")
    if len(edges) != len(probabilities):
        raise ValueError("edge probability count mismatch")
    adjacency = {index: set() for index in range(node_count)}
    for edge, probability in zip(edges, probabilities):
        if len(edge) != 2:
            raise ValueError("each edge must have two endpoints")
        left, right = int(edge[0]), int(edge[1])
        if min(left, right) < 0 or max(left, right) >= node_count:
            raise ValueError("edge endpoint outside graph")
        if float(probability) >= float(threshold):
            adjacency[left].add(right)
            adjacency[right].add(left)
    components = []
    component_by_node = {}
    unseen = set(range(node_count))
    while unseen:
        start = min(unseen)
        queue = deque([start])
        unseen.remove(start)
        component = set()
        while queue:
            node = queue.popleft()
            component.add(node)
            for neighbor in sorted(adjacency[node]):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
        component_index = len(components)
        for node in component:
            component_by_node[node] = component_index
        components.append(component)
    return components, component_by_node


def _topology_components(faces: set[int], edges: Sequence[Sequence[int]]) -> list[set[int]]:
    adjacency = {face: set() for face in faces}
    for edge in edges:
        left, right = int(edge[0]), int(edge[1])
        if left in faces and right in faces:
            adjacency[left].add(right)
            adjacency[right].add(left)
    result = []
    unseen = set(faces)
    while unseen:
        start = min(unseen)
        queue = deque([start])
        unseen.remove(start)
        component = set()
        while queue:
            node = queue.popleft()
            component.add(node)
            for neighbor in sorted(adjacency[node]):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
        result.append(component)
    return result


def constrain_feature_objects_to_instance_components(
    graph: Mapping[str, Any],
    candidates: Iterable[Mapping[str, Any]],
    edge_same_instance_probabilities: Sequence[float],
    threshold: float,
) -> dict[str, Any]:
    nodes = list(graph.get("nodes", []))
    edges = [list(edge) for edge in graph.get("edges", [])]
    node_count = len(nodes)
    components, component_by_node = predicted_same_instance_components(
        node_count,
        edges,
        edge_same_instance_probabilities,
        threshold,
    )
    objects = []
    rejections = []
    split_events = []
    for candidate_index, candidate in enumerate(candidates):
        raw_faces = [int(value) for value in candidate.get("face_indices", [])]
        invalid = sorted({face for face in raw_faces if face < 0 or face >= node_count})
        valid_faces = sorted({face for face in raw_faces if 0 <= face < node_count})
        if invalid:
            rejections.append(
                {
                    "candidate_index": candidate_index,
                    "reason": "invalid_face_index",
                    "invalid_face_indices": invalid,
                }
            )
        if not valid_faces:
            rejections.append({"candidate_index": candidate_index, "reason": "empty_after_validation"})
            continue
        grouped: dict[int, set[int]] = defaultdict(set)
        for face in valid_faces:
            grouped[component_by_node[face]].add(face)
        parts = []
        for predicted_component, faces in sorted(grouped.items()):
            for topology_part in _topology_components(faces, edges):
                parts.append((predicted_component, topology_part))
        if len(parts) > 1:
            split_events.append(
                {
                    "candidate_index": candidate_index,
                    "input_face_count": len(valid_faces),
                    "output_part_count": len(parts),
                    "reason": "predicted_instance_component_or_topology_boundary",
                }
            )
        for part_index, (predicted_component, faces) in enumerate(parts):
            if not faces:
                rejections.append({"candidate_index": candidate_index, "reason": "empty_split_part"})
                continue
            output = dict(candidate)
            output["face_indices"] = sorted(faces)
            output["source_candidate_index"] = candidate_index
            output["split_part_index"] = part_index
            output["predicted_instance_component"] = predicted_component
            output["topology_valid"] = True
            output["boundary_decision"] = "retained" if len(parts) == 1 else "split"
            objects.append(output)
    return {
        "feature_objects": objects,
        "predicted_components": [sorted(component) for component in components],
        "split_events": split_events,
        "rejections": rejections,
        "contracts": {
            "class_similarity_merge_used": False,
            "dimension_similarity_merge_used": False,
            "spatial_proximity_merge_used": False,
            "sample_id_rule_used": False,
            "fixed_face_rule_used": False,
            "area_constant_rule_used": False,
        },
    }


__all__ = [
    "constrain_feature_objects_to_instance_components",
    "predicted_same_instance_components",
]

