"""Versioned, geometry-only STEP contract shared by G07 training and deployment.

Extraction is in an OCC-only process; this module's tensorizer does not read
supervision or generation metadata. Cache identity includes all input versions.
"""
from __future__ import annotations
import copy
import functools
import math
import sys
import time
from g07_common import ROOT, FROZEN, ARTIFACT, REPORT, asset, digest, read, write, sha

SCHEMA = 'topocae_graph_v2_2_1_true_brep_planar_rank_policy_1'
ENTRY_VERSION = 'g07_step_geometry_whitelist_v1'
GUARD_VERSION = 'legacy_projection_degenerate_threshold_1e-10_v1'
ATTR_KEYS = {
 'physical':'area center_of_mass',
 'specific':'normal radius major_radius minor_radius semi_angle axis_direction axis_location axis location apex focal_length degree_u degree_v u_degree v_degree',
 'shape_descriptors':'bbox_dx bbox_dy bbox_dz bbox_longest_extent bbox_middle_extent bbox_shortest_extent bbox_diagonal bbox_aspect_ratio uv_u_span uv_v_span uv_aspect_ratio projected_long_extent projected_short_extent projected_aspect_ratio projected_area_fill_ratio vertex_count edge_count neighbor_count neighbor_area_mean neighbor_area_max neighbor_projected_aspect_mean neighbor_long_extent_mean',
 'raw_physical_v2':'area radius major_radius minor_radius axis_aligned_bbox_extents projected_extents uv_metric_extents pca_extents neighbor_area_mean',
 'topology_v2':'wire_count outer_loop_count inner_loop_count outer_loop_edge_count inner_loop_edge_count',
 'geometry_v2':'surface_type_id surface_family location normal axis_direction axis_location radius major_radius minor_radius semi_angle axis_aligned_bbox_min axis_aligned_bbox_max',
}
RECOGNITION_BASE = '''face_area_over_part_total_area face_area_over_part_bbox_diagonal_sq face_area_over_part_scale_sq radius_over_part_bbox_diagonal radius_over_part_scale major_radius_over_part_scale minor_radius_over_part_scale bbox_long_over_part_bbox_diagonal bbox_middle_over_part_bbox_diagonal bbox_short_over_part_bbox_diagonal projected_long_over_part_scale projected_short_over_part_scale uv_metric_long_over_part_scale uv_metric_short_over_part_scale pca_long_over_part_scale pca_middle_over_part_scale pca_short_over_part_scale face_area_over_neighbor_area_mean neighbor_area_mean_over_part_total_area local_extent_long_middle_ratio local_extent_middle_short_ratio projected_aspect_ratio projected_area_fill_ratio uv_aspect_ratio'''.split()
NODE_RECOGNITION = set(RECOGNITION_BASE + ['log1p_'+k for k in RECOGNITION_BASE] + '''vertex_count edge_count neighbor_count wire_count outer_loop_count inner_loop_count outer_loop_edge_count inner_loop_edge_count oriented_cylindrical_radial_alignment'''.split())
EDGE_KEYS = '''curve_type curve_type_id length is_line is_circle is_ellipse is_spline circle_radius convexity convexity_id dihedral_angle adjacent_face_normal_angle source_face target_face source_primitive_surface_id target_primitive_surface_id source_is_plane target_is_plane source_is_cylinder target_is_cylinder adjacent_plane_loop_role adjacent_plane_loop_role_id adjacent_plane_loop_role_inner adjacent_plane_loop_role_outer adjacent_plane_loop_role_unknown cylindrical_wall_role_hint cylindrical_wall_role_hint_id cylindrical_wall_role_inner_hint cylindrical_wall_role_outer_hint cylindrical_wall_role_ambiguous_hint material_side_evidence'''.split()
EDGE_RECOGNITION = '''edge_length_over_part_bbox_diagonal edge_length_over_part_scale circle_radius_over_part_bbox_diagonal circle_radius_over_part_scale dihedral_angle_over_pi material_side_inner_evidence material_side_outer_evidence material_side_ambiguous_evidence log1p_edge_length_over_part_scale log1p_circle_radius_over_part_scale'''.split()

def subset(value, keys):
    return {k:copy.deepcopy(value[k]) for k in keys if k in value}

def geometry_only(graph):
    if graph.get('schema_version') != SCHEMA:
        raise ValueError(f'Unsupported graph schema {graph.get("schema_version")!r}; requires Graph-v2.2.1 exact planar rank, no zero-fill migration')
    if not graph.get('nodes') or len(graph.get('edges',[])) != len(graph.get('edge_attributes',[])):
        raise ValueError('Empty geometry or edge/attribute length mismatch')
    nodes=[]
    for i,n in enumerate(graph['nodes']):
        if n.get('idx') != i or not isinstance(n.get('primitive_surface_id'),int):
            raise ValueError('Graph node identities must be consecutive final STEP indices')
        a=n['attributes']
        attrs={k:subset(a[k],v.split()) for k,v in ATTR_KEYS.items() if k in a}
        attrs['type_onehot']=copy.deepcopy(a['type_onehot'])
        attrs['recognition_features_v2']=subset(a['recognition_features_v2'],NODE_RECOGNITION)
        nodes.append({'idx':i,'primitive_surface':n['primitive_surface'],
                      'primitive_surface_id':n['primitive_surface_id'],'attributes':attrs})
    edges=[]
    for e in graph['edge_attributes']:
        d=subset(e,EDGE_KEYS)
        d['recognition_features_v2']=subset(e['recognition_features_v2'],EDGE_RECOGNITION)
        d['raw_physical_v2']=subset(e.get('raw_physical_v2',{}),('length','circle_radius'))
        edges.append(d)
    p=graph.get('part_metadata_v2',{})
    result={'schema_version':SCHEMA,'nodes':nodes,'edges':copy.deepcopy(graph['edges']),'edge_attributes':edges,
        'part_metadata_v2':{'normalization':subset(p.get('normalization',{}),'''part_total_area part_bbox_diagonal part_characteristic_length part_characteristic_area scale_definition'''.split()),
            'raw_physical':subset(p.get('raw_physical',{}),'''total_surface_area solid_volume axis_aligned_bbox_min axis_aligned_bbox_max axis_aligned_bbox_extents axis_aligned_bbox_diagonal surface_moment_rms_extents'''.split())}}
    for pair in result['edges']:
        if len(pair)!=2 or any(not isinstance(x,int) or x<0 or x>=len(nodes) for x in pair):
            raise ValueError('Invalid B-Rep adjacency index')
    return result

def numeric_guard(graph, enabled=False):
    graph=geometry_only(graph)
    changed=[]
    if enabled:
        for i,n in enumerate(graph['nodes']):
            d=n['attributes'].get('shape_descriptors',{})
            long=float(d.get('projected_long_extent',0)); short=float(d.get('projected_short_extent',0))
            if long>0 and short<=1e-10*long:
                r=n['attributes']['recognition_features_v2']
                changed.append({'node':i,'previous':r['log1p_projected_area_fill_ratio']})
                d['projected_area_fill_ratio']=0.; r['projected_area_fill_ratio']=0.; r['log1p_projected_area_fill_ratio']=0.
    return graph,changed

def raw_arrays(graph):
    import numpy as np
    if str(FROZEN) not in sys.path: sys.path.insert(0,str(FROZEN))
    from utils.training_export_v2_2_1 import export_graph_v2_2_1_training_record, GRAPH_V2_NODE_FEATURE_SCHEMA, GRAPH_V2_EDGE_FEATURE_SCHEMA
    graph=geometry_only(graph)
    for n in graph['nodes']:
        a=n['attributes']; r=a['recognition_features_v2']
        if len(a['type_onehot'])!=11:
            raise ValueError('Primitive one-hot must have 11 entries')
        missing=set(GRAPH_V2_NODE_FEATURE_SCHEMA[12:])-set(r)
        if missing: raise ValueError(f'Missing required node geometry features: {sorted(missing)}')
    for e in graph['edge_attributes']:
        missing=set(GRAPH_V2_EDGE_FEATURE_SCHEMA)-set(e)-set(e['recognition_features_v2'])
        if missing: raise ValueError(f'Missing required edge geometry features: {sorted(missing)}')
    r=export_graph_v2_2_1_training_record(graph)
    arrays=(np.asarray(r['node_features'],dtype=np.float32).reshape(-1,40),
            np.asarray(r['edges'],dtype=np.int64).reshape(-1,2).T.copy(),
            np.asarray([n['primitive_surface_id'] for n in graph['nodes']],dtype=np.int64),
            np.asarray(r['edge_features'],dtype=np.float32).reshape(-1,20))
    if not all(np.isfinite(a).all() for a in arrays): raise ValueError('Nonfinite model input')
    return arrays

def graph_tensors(graph, metadata, guard=False):
    import numpy as np
    import torch
    graph,_=numeric_guard(graph,guard)
    node,adj,primitive,edge=raw_arrays(graph)
    q,n=metadata['quantile_transform'],metadata['normalization']
    for label,a in (('node',node),('edge',edge)):
        lower=np.asarray(q[label+'_lower'],dtype=np.float32); upper=np.asarray(q[label+'_upper'],dtype=np.float32)
        mean=np.asarray(n[label+'_mean'],dtype=np.float32); std=np.asarray(n[label+'_std'],dtype=np.float32)
        if lower.shape!=(a.shape[1],) or np.any(std<=0): raise ValueError('Invalid frozen normalization')
        transformed=(np.clip(a,lower,upper)-mean)/std
        if not np.isfinite(transformed).all(): raise ValueError('Nonfinite normalized input')
        if label=='node': node=transformed
        else: edge=transformed
    return tuple(torch.from_numpy(a) for a in (node,adj,primitive,edge))

@functools.lru_cache(maxsize=1)
def extractor_versions():
    import OCC
    # Include transitive geometry providers, not just the facade extractor.
    # A compact content-addressed source manifest avoids repeating this list in every cache.
    files=[asset(p) for p in sorted(FROZEN.rglob('*.py'))]
    identity={'scope':'all frozen Python sources, conservative superset of extraction dependencies','files':files}
    path=ARTIFACT/'source_manifests'/(digest(identity)+'.json')
    if not path.exists():write(path,identity)
    return {'OCC':getattr(OCC,'VERSION',getattr(OCC,'__version__','unknown')),
            'source_manifest':asset(path), 'entry':asset(__file__)}

def cache_identity(step, metadata_identity=None, guard=False, transform=None):
    value={'source_step_sha256':sha(step),'schema':SCHEMA,'entry_version':ENTRY_VERSION,
        'extractor':extractor_versions(),'extractor_config':{'build_patches':False},
        'normalization':metadata_identity or 'raw_geometry_no_statistics',
        'numeric_guard':GUARD_VERSION if guard else 'off', 'transform':transform or 'identity'}
    return digest(value),value

def extract_step(step, *, guard=False, metadata_identity=None):
    key,identity=cache_identity(step,metadata_identity,guard)
    path=ARTIFACT/'geometry_cache'/key[:2]/(key+'.json')
    if path.exists():
        prior=read(path)
        if prior['cache_identity']!=identity or digest(prior['graph'])!=prior['geometry_digest']:
            raise ValueError('Geometry cache identity or content mismatch')
        return prior['graph'],{**prior['provenance'],'cache_hit':True,'cache':asset(path)}
    start=time.perf_counter()
    if str(FROZEN) not in sys.path: sys.path.insert(0,str(FROZEN))
    from utils.step_solid_validation import read_step_shape
    from core.graph_v2_2_1 import extract_graph_v2_2_1
    shape=read_step_shape(step)
    graph,changes=numeric_guard(extract_graph_v2_2_1(shape,build_patches=False),guard)
    raw_arrays(graph)
    provenance={'source_step':asset(step),'extraction_seconds':time.perf_counter()-start,
        'guard_changes':changes,'truth_used':False,'cache_hit':False}
    write(path,{'cache_identity':identity,'geometry_digest':digest(graph),'graph':graph,'provenance':provenance})
    return graph,{**provenance,'cache':asset(path)}
