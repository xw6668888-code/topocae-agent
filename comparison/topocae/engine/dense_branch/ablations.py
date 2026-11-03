from __future__ import annotations

from common import SEEDS

import torch
from p1_e6r2_r3.training_common import build_model
from utils.training_export_v2_2 import GRAPH_V2_NODE_FEATURE_SCHEMA, GRAPH_V2_EDGE_FEATURE_SCHEMA


CONFIGS = {
    'full': {'node_mask': [], 'edge_mask': [], 'head': 'sparse', 'semantic_weight': 0.55},
    'no_material': {
        'node_mask': list(range(34, 40)),
        'edge_mask': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19],
        'head': 'sparse', 'semantic_weight': 0.55,
    },
    'no_relative_geometry': {
        'node_mask': list(range(12, 31)), 'edge_mask': [15, 16],
        'head': 'sparse', 'semantic_weight': 0.55,
    },
    'dense_instance': {'node_mask': [], 'edge_mask': [], 'head': 'dense', 'semantic_weight': 0.55},
}


def feature_groups():
    return {
        'schema_version': 'g03_feature_groups_v2', 'index_base': 0,
        'node_schema': list(GRAPH_V2_NODE_FEATURE_SCHEMA),
        'edge_schema': list(GRAPH_V2_EDGE_FEATURE_SCHEMA),
        'configs': {name: {**config,
            'removed_node_names': [GRAPH_V2_NODE_FEATURE_SCHEMA[i] for i in config['node_mask']],
            'removed_edge_names': [GRAPH_V2_EDGE_FEATURE_SCHEMA[i] for i in config['edge_mask']],
            'seeds': list(SEEDS),
        } for name, config in CONFIGS.items()},
        'mask_policy': 'Clone tensors and set selected channels to zero AFTER frozen train quantile clipping and normalization. No refitting; equivalent to frozen clipped-space mean replacement before standardization.',
        'material_duplicate_paths': [
            'node 39: oriented cylindrical radial alignment',
            'node 34:39: wire count and inner/outer loop counts and edge counts; aggregate encodings of the same loop role signal',
            'edge 17:20: explicit material-side evidence',
            'edge 11:15: cylindrical wall role ordinal and all indicators, including ambiguous',
            'edge 7:11: adjacent planar loop role ordinal and all indicators, including unknown',
            'edge 5:7: convexity and signed/oriented dihedral information used to derive material evidence',
        ],
        'geometry_duplicate_paths': [
            'node 12:31: both area normalizations, surface radii, inertia extents, projected/UV extents, neighbor-area normalization, all aspect/fill ratios',
            'edge 15:17: edge length and circle radius normalized by part scale',
        ],
        'retained_in_both_input_ablations': ['node type one-hot 0:11', 'primitive surface embedding IDs', 'semi-angle node 11', 'vertex/edge/neighbor counts 31:34', 'edge curve type ordinal and indicators 0:5', 'original B-Rep adjacency'],
        'scope': 'Network input group effects conditional on retained geometric/topological information. Recovery sees the same unlabelled geometry for every configuration; this is not an end-to-end removal of physical geometry.',
        'not_claimed': ['Individual component benefits within a removed group', 'Perfect statistical independence from all retained correlated geometry', 'Material-property ablation: material side means solid/void orientation, not engineering material constants'],
    }


def masked_inputs(node, edge, config):
    spec = CONFIGS[config]
    if node.ndim != 2 or node.shape[1] != 40 or edge.ndim != 2 or edge.shape[1] != 20:
        raise ValueError('expected frozen 40/20 inputs')
    node, edge = node.clone(), edge.clone()
    node[:, spec['node_mask']] = 0.0
    edge[:, spec['edge_mask']] = 0.0
    return node, edge


def masked_record(record, config):
    node, edge = masked_inputs(record['node_features'], record['edge_features'], config)
    return {**record, 'node_features': node, 'edge_features': edge}


def dense_pairs(node_slices, device=None):
    pairs = []
    for start, end in node_slices:
        idx = torch.arange(start, end, device=device)
        left = idx.repeat_interleave(end - start)
        right = idx.repeat(end - start)
        keep = left != right
        pairs.append(torch.stack((left[keep], right[keep])))
    return torch.cat(pairs, dim=1) if pairs else torch.empty((2, 0), dtype=torch.long, device=device)


def dense_targets(instances, pairs):
    left, right = pairs
    return ((instances[left] > 0) & (instances[left] == instances[right])).float()


def relation_mask(instances, pairs):
    return (instances[pairs[0]] > 0) | (instances[pairs[1]] > 0)


def expanded_relations(adjacency, edge_features, node_count):
    pairs = dense_pairs([(0, node_count)], device=adjacency.device)
    keys = adjacency[0] * node_count + adjacency[1]
    missing = ~torch.isin(pairs[0] * node_count + pairs[1], keys)
    extra = pairs[:, missing]
    return (torch.cat((adjacency, extra), 1),
            torch.cat((edge_features, edge_features.new_zeros((extra.shape[1], 20)))))


def pair_attributes(pair_index, adjacency, edge_features, node_count):
    # Keep the original edge features on adjacent pairs. Nonedges get the
    # frozen-normalized zero vector without creating any trainable parameters.
    keys = adjacency[0] * node_count + adjacency[1]
    pair_keys = pair_index[0] * node_count + pair_index[1]
    order = keys.argsort(stable=True)
    sorted_keys = keys[order]
    out = edge_features.new_zeros((pair_index.shape[1], edge_features.shape[1]))
    if not keys.numel():
        return out
    positions = torch.searchsorted(sorted_keys, pair_keys)
    exists = positions < len(sorted_keys)
    bounded = positions.clamp_max(len(sorted_keys) - 1)
    exists &= sorted_keys[bounded] == pair_keys
    out[exists] = edge_features[order[bounded[exists]]]
    return out


def score_pairs(model, embeddings, pairs, edge_features, chunk_size=32768):
    logits = []
    for start in range(0, pairs.shape[1], chunk_size):
        stop = start + chunk_size
        src, dst = pairs[:, start:stop]
        edge_embedding = model.edge_projection(edge_features[start:stop])
        values = torch.cat((embeddings[src], embeddings[dst], edge_embedding), dim=-1)
        logits.append(model.edge_same_instance_head(values).squeeze(-1))
    return torch.cat(logits) if logits else embeddings.new_empty(0)


def make_model():
    return build_model()


def forward_dense(model, node, adjacency, primitive, features, pairs, pair_features, adjacency_positions=None):
    x = model.input_projection(torch.cat((node, model.primitive_embedding(primitive)), dim=-1))
    for layer in model.layers:
        x = x + layer(x, adjacency, features)
    relation_logits = score_pairs(model, x, pairs, pair_features)
    if adjacency_positions is None:
        adjacency_positions = torch.arange(adjacency.shape[1], device=adjacency.device)
    return {'macro_logits': model.macro_head(x), 'semantic_logits': model.semantic_head(x),
            'node_embeddings': x, 'edge_same_instance_logits': relation_logits[adjacency_positions],
            'relation_logits': relation_logits}
