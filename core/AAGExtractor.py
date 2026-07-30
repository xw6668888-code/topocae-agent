import os
import math

DEFAULT_PRIMITIVE_SURFACE_KEY_TO_ID = {
    "unknown_surface": 0,
    "plane": 1,
    "cylinder": 2,
    "cone": 3,
    "sphere": 4,
    "torus": 5,
    "bspline_surface": 6,
    "surface_of_revolution": 7,
    "surface_of_extrusion": 8,
    "offset_surface": 9,
    "other_surface": 10,
}


def _resolve_primitive_surface(surface_type_str):
    mapping = {
        "Plane": "plane",
        "Cylinder": "cylinder",
        "Cone": "cone",
        "Sphere": "sphere",
        "Torus": "torus",
        "BSplineSurface": "bspline_surface",
        "SurfaceOfRevolution": "surface_of_revolution",
        "SurfaceOfExtrusion": "surface_of_extrusion",
        "OffsetSurface": "offset_surface",
        "BezierSurface": "other_surface",
        "OtherSurface": "other_surface",
        "Unknown": "unknown_surface",
    }
    return mapping.get(surface_type_str, "unknown_surface")


def _safe_divide(numerator, denominator):
    return numerator / denominator if abs(denominator) > 1e-12 else 0.0


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _norm(v):
    return math.sqrt(_dot(v, v))


def _normalize(v):
    length = _norm(v)
    if length <= 1e-12:
        return [0.0, 0.0, 0.0]
    return [v[0] / length, v[1] / length, v[2] / length]


def _extent_aspect(long_extent, short_extent):
    return _safe_divide(long_extent, short_extent) if short_extent > 1e-9 else 0.0


def _empty_shape_descriptors():
    return {
        "bbox_dx": 0.0,
        "bbox_dy": 0.0,
        "bbox_dz": 0.0,
        "bbox_longest_extent": 0.0,
        "bbox_middle_extent": 0.0,
        "bbox_shortest_extent": 0.0,
        "bbox_diagonal": 0.0,
        "bbox_aspect_ratio": 0.0,
        "uv_u_span": 0.0,
        "uv_v_span": 0.0,
        "uv_aspect_ratio": 0.0,
        "projected_long_extent": 0.0,
        "projected_short_extent": 0.0,
        "projected_aspect_ratio": 0.0,
        "projected_area_fill_ratio": 0.0,
        "vertex_count": 0.0,
        "edge_count": 0.0,
        "neighbor_count": 0.0,
        "neighbor_area_mean": 0.0,
        "neighbor_area_max": 0.0,
        "neighbor_projected_aspect_mean": 0.0,
        "neighbor_long_extent_mean": 0.0,
    }


def _compute_shape_descriptors(face, surf, area):
    descriptors = _empty_shape_descriptors()
    descriptors.update(_compute_bbox_descriptors(face))
    descriptors.update(_compute_uv_descriptors(face))
    descriptors.update(_compute_topology_counts(face))

    try:
        normal = [surf.Plane().Axis().Direction().X(), surf.Plane().Axis().Direction().Y(), surf.Plane().Axis().Direction().Z()]
        descriptors.update(_compute_projected_planar_descriptors(face, normal, area))
    except Exception:
        pass

    return descriptors


def _compute_bbox_descriptors(face):
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib

    try:
        box = Bnd_Box()
        brepbndlib.Add(face, box)
        xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
        dx = max(0.0, xmax - xmin)
        dy = max(0.0, ymax - ymin)
        dz = max(0.0, zmax - zmin)
        longest, middle, shortest = sorted([dx, dy, dz], reverse=True)
        return {
            "bbox_dx": dx,
            "bbox_dy": dy,
            "bbox_dz": dz,
            "bbox_longest_extent": longest,
            "bbox_middle_extent": middle,
            "bbox_shortest_extent": shortest,
            "bbox_diagonal": math.sqrt(dx * dx + dy * dy + dz * dz),
            "bbox_aspect_ratio": _extent_aspect(longest, middle),
        }
    except Exception:
        return {}


def _compute_uv_descriptors(face):
    from OCC.Core.BRepTools import breptools

    try:
        umin, umax, vmin, vmax = breptools.UVBounds(face)
        u_span = abs(umax - umin)
        v_span = abs(vmax - vmin)
        return {
            "uv_u_span": u_span,
            "uv_v_span": v_span,
            "uv_aspect_ratio": _extent_aspect(max(u_span, v_span), min(u_span, v_span)),
        }
    except Exception:
        return {}


def _compute_topology_counts(face):
    from OCC.Extend.TopologyUtils import TopologyExplorer

    try:
        face_topo = TopologyExplorer(face)
        return {
            "vertex_count": float(len(list(face_topo.vertices()))),
            "edge_count": float(len(list(face_topo.edges()))),
        }
    except Exception:
        return {}


def _compute_projected_planar_descriptors(face, normal, area):
    from OCC.Core.BRep import BRep_Tool
    from OCC.Extend.TopologyUtils import TopologyExplorer

    normal = _normalize(normal)
    if _norm(normal) <= 1e-12:
        return {}

    reference = [1.0, 0.0, 0.0] if abs(normal[0]) < 0.9 else [0.0, 1.0, 0.0]
    basis_u = _normalize(_cross(normal, reference))
    basis_v = _normalize(_cross(normal, basis_u))

    points = []
    seen = set()
    for vertex in TopologyExplorer(face).vertices():
        pnt = BRep_Tool.Pnt(vertex)
        key = (round(pnt.X(), 9), round(pnt.Y(), 9), round(pnt.Z(), 9))
        if key in seen:
            continue
        seen.add(key)
        point = [pnt.X(), pnt.Y(), pnt.Z()]
        points.append((_dot(point, basis_u), _dot(point, basis_v)))

    if len(points) < 2:
        return {}

    mean_x = sum(point[0] for point in points) / len(points)
    mean_y = sum(point[1] for point in points) / len(points)
    centered = [(point[0] - mean_x, point[1] - mean_y) for point in points]
    cxx = sum(point[0] * point[0] for point in centered) / len(centered)
    cyy = sum(point[1] * point[1] for point in centered) / len(centered)
    cxy = sum(point[0] * point[1] for point in centered) / len(centered)
    angle = 0.5 * math.atan2(2.0 * cxy, cxx - cyy) if abs(cxx - cyy) > 1e-12 or abs(cxy) > 1e-12 else 0.0
    axis_1 = (math.cos(angle), math.sin(angle))
    axis_2 = (-math.sin(angle), math.cos(angle))
    extent_1 = _projected_extent(centered, axis_1)
    extent_2 = _projected_extent(centered, axis_2)
    long_extent = max(extent_1, extent_2)
    short_extent = min(extent_1, extent_2)
    projected_bbox_area = long_extent * short_extent

    return {
        "projected_long_extent": long_extent,
        "projected_short_extent": short_extent,
        "projected_aspect_ratio": _extent_aspect(long_extent, short_extent),
        "projected_area_fill_ratio": _safe_divide(area, projected_bbox_area),
    }


def _projected_extent(points, axis):
    values = [point[0] * axis[0] + point[1] * axis[1] for point in points]
    return max(values) - min(values) if values else 0.0


def _annotate_neighborhood_descriptors(nodes, edges_list):
    adjacency = {node["idx"]: set() for node in nodes}
    for src, dst in edges_list:
        adjacency.setdefault(src, set()).add(dst)

    for node in nodes:
        descriptors = node["attributes"].setdefault("shape_descriptors", _empty_shape_descriptors())
        neighbor_indices = sorted(adjacency.get(node["idx"], set()))
        neighbors = [nodes[index] for index in neighbor_indices if 0 <= index < len(nodes)]
        if not neighbors:
            continue

        neighbor_areas = [neighbor["attributes"].get("physical", {}).get("area", 0.0) for neighbor in neighbors]
        neighbor_projected_aspects = [
            neighbor["attributes"].get("shape_descriptors", {}).get("projected_aspect_ratio", 0.0)
            for neighbor in neighbors
        ]
        neighbor_long_extents = [
            neighbor["attributes"].get("shape_descriptors", {}).get("projected_long_extent", 0.0)
            for neighbor in neighbors
        ]

        descriptors["neighbor_count"] = float(len(neighbors))
        descriptors["neighbor_area_mean"] = sum(neighbor_areas) / len(neighbor_areas)
        descriptors["neighbor_area_max"] = max(neighbor_areas)
        descriptors["neighbor_projected_aspect_mean"] = sum(neighbor_projected_aspects) / len(neighbor_projected_aspects)
        descriptors["neighbor_long_extent_mean"] = sum(neighbor_long_extents) / len(neighbor_long_extents)


EDGE_CURVE_TYPE_TO_ID = {
    "unknown": 0,
    "line": 1,
    "circle": 2,
    "ellipse": 3,
    "bspline": 4,
    "bezier": 5,
    "other": 6,
}


EDGE_CONVEXITY_TO_ID = {
    "unknown": 0,
    "convex": 1,
    "concave": 2,
    "smooth": 3,
}


EDGE_PLANE_LOOP_ROLE_TO_ID = {
    "unknown": 0,
    "inner_loop": 1,
    "outer_loop": 2,
}


EDGE_CYLINDRICAL_WALL_ROLE_TO_ID = {
    "unknown": 0,
    "inner_hole_wall": 1,
    "outer_cylinder_wall": 2,
    "ambiguous": 3,
}


def _edge_curve_info(edge):
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GeomAbs import (
        GeomAbs_Line,
        GeomAbs_Circle,
        GeomAbs_Ellipse,
        GeomAbs_BSplineCurve,
        GeomAbs_BezierCurve,
    )
    from OCC.Core.GProp import GProp_GProps

    curve_type = "unknown"
    circle_radius = 0.0
    try:
        curve = BRepAdaptor_Curve(edge)
        raw_type = curve.GetType()
        if raw_type == GeomAbs_Line:
            curve_type = "line"
        elif raw_type == GeomAbs_Circle:
            curve_type = "circle"
            circle_radius = float(curve.Circle().Radius())
        elif raw_type == GeomAbs_Ellipse:
            curve_type = "ellipse"
        elif raw_type == GeomAbs_BSplineCurve:
            curve_type = "bspline"
        elif raw_type == GeomAbs_BezierCurve:
            curve_type = "bezier"
        else:
            curve_type = "other"
    except Exception:
        curve_type = "unknown"

    length = 0.0
    try:
        props = GProp_GProps()
        brepgprop.LinearProperties(edge, props)
        length = float(props.Mass())
    except Exception:
        length = 0.0

    return {
        "curve_type": curve_type,
        "curve_type_id": EDGE_CURVE_TYPE_TO_ID.get(curve_type, 0),
        "length": length,
        "is_line": 1.0 if curve_type == "line" else 0.0,
        "is_circle": 1.0 if curve_type == "circle" else 0.0,
        "is_ellipse": 1.0 if curve_type == "ellipse" else 0.0,
        "is_spline": 1.0 if curve_type in {"bspline", "bezier"} else 0.0,
        "circle_radius": circle_radius,
    }


def _face_normal_from_node(node):
    specific = node.get("attributes", {}).get("specific", {})
    normal = specific.get("normal")
    if normal:
        return _normalize(list(normal))
    axis = specific.get("axis")
    if axis:
        return _normalize(list(axis))
    return [0.0, 0.0, 0.0]


def _angle_between_normals(normal_a, normal_b):
    normal_a = _normalize(normal_a)
    normal_b = _normalize(normal_b)
    if _norm(normal_a) <= 1e-12 or _norm(normal_b) <= 1e-12:
        return 0.0
    cosine = max(-1.0, min(1.0, abs(_dot(normal_a, normal_b))))
    return math.acos(cosine)


def _rough_convexity_from_normals(normal_a, normal_b, angle):
    if angle < math.radians(5.0):
        return "smooth"
    if _norm(normal_a) <= 1e-12 or _norm(normal_b) <= 1e-12:
        return "unknown"
    return "convex"


def _point_to_list(point):
    return [float(point.X()), float(point.Y()), float(point.Z())]


def _node_center(node):
    center = node.get("attributes", {}).get("physical", {}).get("center_of_mass")
    if not center:
        return [0.0, 0.0, 0.0]
    return [float(center[0]), float(center[1]), float(center[2])]


def _edge_midpoint(edge):
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve

    curve = BRepAdaptor_Curve(edge)
    first = float(curve.FirstParameter())
    last = float(curve.LastParameter())
    return curve.Value((first + last) * 0.5)


def _classify_solid_point(shape, point, tolerance=1e-6):
    from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
    from OCC.Core.gp import gp_Pnt
    from OCC.Core.TopAbs import TopAbs_IN, TopAbs_OUT, TopAbs_ON

    classifier = BRepClass3d_SolidClassifier(
        shape,
        gp_Pnt(float(point[0]), float(point[1]), float(point[2])),
        tolerance,
    )
    state = classifier.State()
    if state == TopAbs_IN:
        return "in"
    if state == TopAbs_OUT:
        return "out"
    if state == TopAbs_ON:
        return "on"
    return "unknown"


def _solid_wedge_convexity(shape, edge, source_node, target_node, angle, edge_length):
    if shape is None:
        return None
    if angle < math.radians(5.0):
        return "smooth"

    try:
        midpoint = _point_to_list(_edge_midpoint(edge))
        source_direction = _normalize([a - b for a, b in zip(_node_center(source_node), midpoint)])
        target_direction = _normalize([a - b for a, b in zip(_node_center(target_node), midpoint)])
        probe_direction = _normalize([a + b for a, b in zip(source_direction, target_direction)])
        if _norm(probe_direction) <= 1e-12:
            return None

        offset = min(max(float(edge_length) * 0.02, 1e-4), 0.5)
        for scale in (1.0, 2.0, 5.0):
            probe = [coord + direction * offset * scale for coord, direction in zip(midpoint, probe_direction)]
            state = _classify_solid_point(shape, probe)
            if state == "in":
                return "convex"
            if state == "out":
                return "concave"
    except Exception:
        return None

    return None


def _edge_is_same(edge_a, edge_b):
    try:
        return bool(edge_a.IsSame(edge_b))
    except Exception:
        return edge_a.__hash__() == edge_b.__hash__()


def _wire_contains_edge(wire, edge):
    from OCC.Extend.TopologyUtils import TopologyExplorer

    try:
        for wire_edge in TopologyExplorer(wire).edges():
            if _edge_is_same(wire_edge, edge):
                return True
    except Exception:
        return False
    return False


def _face_wires(face):
    from OCC.Extend.TopologyUtils import TopologyExplorer

    try:
        return list(TopologyExplorer(face).wires())
    except Exception:
        return []


def _plane_loop_role_for_edge(face, edge):
    from OCC.Core.BRepTools import breptools

    if face is None:
        return "unknown"
    try:
        outer_wire = breptools.OuterWire(face)
        if _wire_contains_edge(outer_wire, edge):
            return "outer_loop"
    except Exception:
        outer_wire = None

    found_on_inner_wire = False
    for wire in _face_wires(face):
        if outer_wire is not None and _edge_is_same(wire, outer_wire):
            continue
        if _wire_contains_edge(wire, edge):
            found_on_inner_wire = True
            break
    return "inner_loop" if found_on_inner_wire else "unknown"


def _edge_loop_role(edge, curve_info, source_face_shape, target_face_shape, source_primitive_id, target_primitive_id):
    if curve_info.get("curve_type") != "circle":
        return "unknown"
    if source_primitive_id == 1:
        return _plane_loop_role_for_edge(source_face_shape, edge)
    if target_primitive_id == 1:
        return _plane_loop_role_for_edge(target_face_shape, edge)
    return "unknown"


def _cylindrical_wall_role_hint(loop_role, curve_info, source_primitive_id, target_primitive_id):
    if curve_info.get("curve_type") != "circle":
        return "unknown"
    if {source_primitive_id, target_primitive_id} != {1, 2}:
        return "unknown"
    if loop_role == "inner_loop":
        return "inner_hole_wall"
    if loop_role == "outer_loop":
        return "outer_cylinder_wall"
    return "ambiguous"


def _build_directed_edge_attribute(edge, source_idx, target_idx, nodes, shape=None, source_face_shape=None, target_face_shape=None):
    curve_info = _edge_curve_info(edge)
    source_node = nodes[source_idx]
    target_node = nodes[target_idx]
    source_normal = _face_normal_from_node(source_node)
    target_normal = _face_normal_from_node(target_node)
    normal_angle = _angle_between_normals(source_normal, target_normal)
    convexity = _solid_wedge_convexity(
        shape,
        edge,
        source_node,
        target_node,
        normal_angle,
        curve_info.get("length", 0.0),
    )
    if convexity is None:
        convexity = _rough_convexity_from_normals(source_normal, target_normal, normal_angle)
    source_primitive_id = int(source_node.get("primitive_surface_id", 0))
    target_primitive_id = int(target_node.get("primitive_surface_id", 0))
    loop_role = _edge_loop_role(
        edge,
        curve_info,
        source_face_shape,
        target_face_shape,
        source_primitive_id,
        target_primitive_id,
    )
    wall_role = _cylindrical_wall_role_hint(
        loop_role,
        curve_info,
        source_primitive_id,
        target_primitive_id,
    )

    return {
        **curve_info,
        "convexity": convexity,
        "convexity_id": EDGE_CONVEXITY_TO_ID.get(convexity, 0),
        "dihedral_angle": normal_angle,
        "adjacent_face_normal_angle": normal_angle,
        "source_face": int(source_idx),
        "target_face": int(target_idx),
        "source_primitive_surface_id": source_primitive_id,
        "target_primitive_surface_id": target_primitive_id,
        "source_is_plane": 1.0 if source_primitive_id == 1 else 0.0,
        "target_is_plane": 1.0 if target_primitive_id == 1 else 0.0,
        "source_is_cylinder": 1.0 if source_primitive_id == 2 else 0.0,
        "target_is_cylinder": 1.0 if target_primitive_id == 2 else 0.0,
        "adjacent_plane_loop_role": loop_role,
        "adjacent_plane_loop_role_id": EDGE_PLANE_LOOP_ROLE_TO_ID.get(loop_role, 0),
        "adjacent_plane_loop_role_inner": 1.0 if loop_role == "inner_loop" else 0.0,
        "adjacent_plane_loop_role_outer": 1.0 if loop_role == "outer_loop" else 0.0,
        "adjacent_plane_loop_role_unknown": 1.0 if loop_role == "unknown" else 0.0,
        "cylindrical_wall_role_hint": wall_role,
        "cylindrical_wall_role_hint_id": EDGE_CYLINDRICAL_WALL_ROLE_TO_ID.get(wall_role, 0),
        "cylindrical_wall_role_inner_hint": 1.0 if wall_role == "inner_hole_wall" else 0.0,
        "cylindrical_wall_role_outer_hint": 1.0 if wall_role == "outer_cylinder_wall" else 0.0,
        "cylindrical_wall_role_ambiguous_hint": 1.0 if wall_role == "ambiguous" else 0.0,
    }


def extract_aag_graph(shape, face_infos=None, primitive_surface_index=None, include_label_fields=True):
    """
    提取属性邻接图 (AAG)，包含通用物理特征(面积、质心)、特定几何特征(法向、半径等)。
    """
    from OCC.Extend.TopologyUtils import TopologyExplorer
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.GeomAbs import (
        GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere, 
        GeomAbs_Torus, GeomAbs_BezierSurface, GeomAbs_BSplineSurface, 
        GeomAbs_SurfaceOfRevolution, GeomAbs_SurfaceOfExtrusion, 
        GeomAbs_OffsetSurface, GeomAbs_OtherSurface
    )

    # 11种标准曲面类型序列
    SURFACE_TYPES = [
        GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere, 
        GeomAbs_Torus, GeomAbs_BezierSurface, GeomAbs_BSplineSurface, 
        GeomAbs_SurfaceOfRevolution, GeomAbs_SurfaceOfExtrusion, 
        GeomAbs_OffsetSurface, GeomAbs_OtherSurface
    ]
    SURFACE_NAMES = [
        "Plane", "Cylinder", "Cone", "Sphere", 
        "Torus", "BezierSurface", "BSplineSurface", 
        "SurfaceOfRevolution", "SurfaceOfExtrusion", 
        "OffsetSurface", "OtherSurface"
    ]

    topo = TopologyExplorer(shape)
    faces = list(topo.faces())
    
    nodes = []
    face_to_idx = {face.__hash__(): i for i, face in enumerate(faces)}
    
    instances_dict = {}

    for i, face in enumerate(faces):
        surf = BRepAdaptor_Surface(face)
        surf_type = surf.GetType()
        
        # --- 1. 获取基础与普适通用特征 (Universal Features) ---
        # 1.1 类别独热编码
        type_onehot = [1 if surf_type == t else 0 for t in SURFACE_TYPES]
        surface_type_str = SURFACE_NAMES[type_onehot.index(1)] if 1 in type_onehot else "Unknown"
        primitive_surface_key = _resolve_primitive_surface(surface_type_str)
        primitive_key_to_id = (
            primitive_surface_index["key_to_id"]
            if primitive_surface_index is not None
            else DEFAULT_PRIMITIVE_SURFACE_KEY_TO_ID
        )
        primitive_surface_id = primitive_key_to_id.get(
            primitive_surface_key,
            primitive_key_to_id.get("unknown_surface", 0),
        )

        # 1.2 物理属性：精确面积(Area)与质心(Center of Mass)
        props = GProp_GProps()
        try:
            brepgprop.SurfaceProperties(face, props)
            area = props.Mass()
            com = props.CentreOfMass()
            center_point = [com.X(), com.Y(), com.Z()]
        except Exception:
            area = 0.0
            center_point = [0.0, 0.0, 0.0]

        # --- 2. 获取特定表面专属特征 (Specific Features) ---
        specific_params = {}
        if surf_type == GeomAbs_Plane:
            n = surf.Plane().Axis().Direction()
            specific_params["normal"] = [n.X(), n.Y(), n.Z()]
        
        elif surf_type == GeomAbs_Cylinder:
            cyl = surf.Cylinder()
            n = cyl.Axis().Direction()
            specific_params["axis"] = [n.X(), n.Y(), n.Z()]
            specific_params["radius"] = cyl.Radius()
            
        elif surf_type == GeomAbs_Cone:
            cone = surf.Cone()
            n = cone.Axis().Direction()
            a = cone.Apex()
            specific_params["axis"] = [n.X(), n.Y(), n.Z()]
            specific_params["apex"] = [a.X(), a.Y(), a.Z()]
            specific_params["semi_angle"] = cone.SemiAngle()
            
        elif surf_type == GeomAbs_Sphere:
            sph = surf.Sphere()
            specific_params["radius"] = sph.Radius()
            
        elif surf_type == GeomAbs_Torus:
            tor = surf.Torus()
            specific_params["major_radius"] = tor.MajorRadius()
            specific_params["minor_radius"] = tor.MinorRadius()
            n = tor.Axis().Direction()
            specific_params["axis"] = [n.X(), n.Y(), n.Z()]

        shape_descriptors = _compute_shape_descriptors(face, surf, area)

        # 获取顶层分配的标签
        if face_infos is not None:
            info = face_infos.get(face.__hash__(), {
                "semantic_label": 0,
                "instance_id": 0,
                "macro_feature": "stock",
                "macro_feature_id": 0,
            })
        else:
            info = None

        node_record = {
            "idx": i,
            "primitive_surface": primitive_surface_key,
            "primitive_surface_id": primitive_surface_id,
            "attributes": {
                "surface_type_str": surface_type_str,
                "type_onehot": type_onehot,
                "physical": {
                    "area": area,
                    "center_of_mass": center_point
                },
                "specific": specific_params,
                "shape_descriptors": shape_descriptors,
            }
        }

        if include_label_fields:
            label_info = info or {
                "semantic_label": 0,
                "instance_id": 0,
                "macro_feature": "stock",
                "macro_feature_id": 0,
            }
            node_record["semantic_label"] = label_info["semantic_label"]
            node_record["instance_id"] = label_info["instance_id"]
            node_record["macro_feature"] = label_info["macro_feature"]
            node_record["macro_feature_id"] = label_info.get("macro_feature_id", 0)

        nodes.append(node_record)

        # 收集 Instances
        inst_id = info["instance_id"] if info is not None else 0
        if include_label_fields and inst_id > 0:
            if inst_id not in instances_dict:
                instances_dict[inst_id] = {
                    "instance_id": inst_id,
                    "macro_feature": info["macro_feature"],
                    "macro_feature_id": info.get("macro_feature_id", 0),
                    "node_indices": []
                }
            instances_dict[inst_id]["node_indices"].append(i)

    # 边提取
    edges_list = []
    edge_attributes = []
    for edge in topo.edges():
        connected_faces = list(topo.faces_from_edge(edge))
        if len(connected_faces) == 2:
            idx1 = face_to_idx.get(connected_faces[0].__hash__())
            idx2 = face_to_idx.get(connected_faces[1].__hash__())
            if idx1 is not None and idx2 is not None:
                face1 = connected_faces[0]
                face2 = connected_faces[1]
                edges_list.append([idx1, idx2])
                edge_attributes.append(_build_directed_edge_attribute(edge, idx1, idx2, nodes, shape, face1, face2))
                edges_list.append([idx2, idx1])
                edge_attributes.append(_build_directed_edge_attribute(edge, idx2, idx1, nodes, shape, face2, face1))

    _annotate_neighborhood_descriptors(nodes, edges_list)
    
    feature_sets = list(instances_dict.values())
                
    return {
        "nodes": nodes,
        "edges": edges_list,
        "edge_attributes": edge_attributes,
        "feature_sets": feature_sets,
    }
