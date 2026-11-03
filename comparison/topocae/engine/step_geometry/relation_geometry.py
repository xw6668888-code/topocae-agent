"""Geometry-only attributes for additional dense pairs; physical edges are copied.

This is a feature configuration of the existing G03 relation head. It does not
create model parameters, message-passing edges, pair filters or labels.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from step_entry import geometry_only

VERSION = 'g07_nonadj_relation_geometry_20_v1'
SCALE_DEFINITION = 'l2_norm_of_surface_central_moment_rms_extents'
HOP_CAP = 8
FEATURE_NAMES = (
    'nonadjacent_sentinel',
    'log1p_centroid_distance_over_part_scale',
    'absolute_axis_dot',
    'log1p_axial_distance_left_over_part_scale',
    'log1p_axial_distance_right_over_part_scale',
    'log1p_transverse_distance_left_over_part_scale',
    'log1p_transverse_distance_right_over_part_scale',
    'radius_absolute_difference_over_part_scale',
    'radius_min_over_max',
    'log1p_area_absolute_difference_over_part_scale_squared',
    'area_min_over_max',
    'same_primitive',
    'axis_known_left',
    'axis_known_right',
    'physical_hops_capped_at_8_over_8',
    'both_cylinder',
    'both_plane',
    'radius_left_over_part_scale',
    'radius_right_over_part_scale',
    'area_sum_over_part_scale_squared',
)


def _number(value, name, *, positive=False):
    if isinstance(value, bool):
        raise ValueError(f'{name} must be a geometric number')
    value = float(value)
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f'{name} must be finite and {"positive" if positive else "nonnegative"}')
    return value


def _vector(value, name):
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f'{name} must be a finite three-vector')
    return vector


def _face_arrays(graph):
    norm = graph['part_metadata_v2']['normalization']
    if norm.get('scale_definition') != SCALE_DEFINITION:
        raise ValueError('Relation geometry requires the frozen rotation-invariant part scale')
    scale = _number(norm['part_characteristic_length'], 'part scale', positive=True)
    centers, areas, radii, axes, known, primitive, cylinders, planes = ([] for _ in range(8))
    for node in graph['nodes']:
        attributes = node['attributes']
        physical = attributes['physical']
        geometry = attributes.get('geometry_v2', {})
        specific = attributes.get('specific', {})
        raw = attributes.get('raw_physical_v2', {})
        centers.append(_vector(physical['center_of_mass'], 'face centroid'))
        areas.append(_number(physical['area'], 'face area'))
        radii.append(_number(geometry.get('radius', specific.get('radius', raw.get('radius', 0.))), 'face radius'))
        plane = node['primitive_surface'] == 'plane'
        cylinder = node['primitive_surface'] == 'cylinder'
        direction = geometry.get('normal', specific.get('normal')) if plane else geometry.get(
            'axis_direction', specific.get('axis_direction', specific.get('axis')))
        vector = np.zeros(3, dtype=np.float64) if direction is None else _vector(direction, 'face axis/normal')
        magnitude = float(np.linalg.norm(vector))
        valid = magnitude > 1e-12
        axes.append(vector / magnitude if valid else np.zeros(3, dtype=np.float64))
        known.append(valid)
        primitive.append(node['primitive_surface_id'])
        cylinders.append(cylinder)
        planes.append(plane)
    values = [np.asarray(v) for v in (centers, areas, radii, axes, known, primitive, cylinders, planes)]
    return scale, values


def _hop_distances(adjacency, node_count, left, right):
    """Unweighted undirected physical hops; 8 includes >=8 and disconnected."""
    neighbors = [set() for _ in range(node_count)]
    for a, b in adjacency.T:
        neighbors[a].add(b)
        neighbors[b].add(a)
    values = np.full(len(left), HOP_CAP, dtype=np.float64)
    for source in np.unique(left):
        distance = {int(source): 0}
        frontier = {int(source)}
        for depth in range(1, HOP_CAP):
            frontier = {v for u in frontier for v in neighbors[u] if v not in distance}
            if not frontier:
                break
            distance.update({v: depth for v in frontier})
        positions = np.flatnonzero(left == source)
        values[positions] = [distance.get(int(target), HOP_CAP) for target in right[positions]]
    return values / HOP_CAP


def _pair_array(index, name, node_count):
    if not isinstance(index, torch.Tensor) or index.dtype != torch.int64 or index.ndim != 2 or index.shape[0] != 2:
        raise ValueError(f'{name} must be a 2 x count int64 tensor')
    value = index.detach().cpu().numpy()
    if value.size and (value.min() < 0 or value.max() >= node_count):
        raise ValueError(f'{name} contains an out-of-range face index')
    return value


def pair_attributes(graph, adjacency, existing_edge_features, relation_index):
    """Return [R,20], preserving existing_edge_features at positions [0:E].

    relation_index must have all original adjacency entries first, in their
    original order, including multiplicities. The remaining entries must be
    distinct directed nonadjacent, nonself pairs. They may be any subset/order;
    this function never creates or removes a pair. Indices address geometry and
    topology only; their numerical values never become model feature values.

    Only added pair rows have the registered new meaning. Existing edge rows
    already have their frozen normalized meaning and are not renormalized.
    Output dtype/device are inherited from existing_edge_features.
    """
    graph = geometry_only(graph)
    node_count = len(graph['nodes'])
    original = _pair_array(adjacency, 'adjacency', node_count)
    relations = _pair_array(relation_index, 'relation_index', node_count)
    count = original.shape[1]
    stored = np.asarray(graph['edges'], dtype=np.int64).reshape(-1, 2).T
    if not np.array_equal(stored, original):
        raise ValueError('Adjacency must match the supplied geometry graph, including order and multiplicity')
    if relations.shape[1] < count or not np.array_equal(relations[:, :count], original):
        raise ValueError('The first E relation entries must exactly preserve physical adjacency')
    if not isinstance(existing_edge_features, torch.Tensor) or not existing_edge_features.is_floating_point():
        raise ValueError('Existing physical attributes must be a floating point tensor')
    if existing_edge_features.shape != (count, len(FEATURE_NAMES)) or not torch.isfinite(existing_edge_features).all():
        raise ValueError('Existing physical attributes must be finite E x 20')
    extra = relations[:, count:]
    physical = set(map(tuple, original.T.tolist()))
    additional = list(map(tuple, extra.T.tolist()))
    if len(set(additional)) != len(additional) or any(a == b or (a, b) in physical for a, b in additional):
        raise ValueError('Additional relations must be unique nonself directed nonadjacent pairs')
    if not additional:
        return existing_edge_features.clone()

    scale, (centers, areas, radii, axes, known, primitive, cylinders, planes) = _face_arrays(graph)
    left, right = extra
    delta = (centers[right] - centers[left]) / scale
    distance = np.linalg.norm(delta, axis=1)
    signed_left = np.einsum('ij,ij->i', delta, axes[left])
    signed_right = np.einsum('ij,ij->i', delta, axes[right])
    axial_left = np.abs(signed_left)
    axial_right = np.abs(signed_right)
    transverse_left = np.linalg.norm(delta - signed_left[:, None] * axes[left], axis=1) * known[left]
    transverse_right = np.linalg.norm(delta - signed_right[:, None] * axes[right], axis=1) * known[right]
    radius_left, radius_right = radii[left] / scale, radii[right] / scale
    area_left, area_right = areas[left] / (scale * scale), areas[right] / (scale * scale)
    radius_max, area_max = np.maximum(radius_left, radius_right), np.maximum(area_left, area_right)
    radius_ratio = np.divide(np.minimum(radius_left, radius_right), radius_max, out=np.zeros_like(radius_max), where=radius_max > 0)
    area_ratio = np.divide(np.minimum(area_left, area_right), area_max, out=np.zeros_like(area_max), where=area_max > 0)
    features = np.column_stack((
        -np.ones(len(left)), np.log1p(distance),
        np.clip(np.abs(np.einsum('ij,ij->i', axes[left], axes[right])), 0., 1.),
        np.log1p(axial_left), np.log1p(axial_right),
        np.log1p(transverse_left), np.log1p(transverse_right),
        np.abs(radius_left - radius_right), radius_ratio,
        np.log1p(np.abs(area_left - area_right)), area_ratio,
        primitive[left] == primitive[right], known[left], known[right],
        _hop_distances(original, node_count, left, right),
        cylinders[left] & cylinders[right], planes[left] & planes[right],
        radius_left, radius_right, area_left + area_right,
    ))
    additional_features = torch.as_tensor(features, dtype=existing_edge_features.dtype, device=existing_edge_features.device)
    if not torch.isfinite(additional_features).all():
        raise ValueError('Nonfinite relation geometry; no silent clipping or zero migration')
    return torch.cat((existing_edge_features, additional_features), dim=0)
