import math
import random

from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.Core.GeomAbs import GeomAbs_BSplineSurface, GeomAbs_Cylinder, GeomAbs_Plane, GeomAbs_SurfaceOfExtrusion, GeomAbs_SurfaceOfRevolution, GeomAbs_Torus

from features.base_feature import BaseFeatureGenerator
from features.swept_slot import (
    build_vertical_swept_slot_tool_from_path,
    straight_boundary_path_on_box,
    straight_boundary_path_on_circle,
    swept_profile_metadata,
)
from utils.feature_footprint import footprint_conflicts, polyline_capsule_footprint
from utils.path_geometry import path_tangent_angle, point_to_polyline_distance, polyline_length, sample_circular_arc_points


class SweptSlotGenerator(BaseFeatureGenerator):
    supported_profile_shapes = ("u", "v", "o")
    profile_to_macro_key = {
        "u": "slot_u",
        "v": "slot_v",
        "dovetail": "slot_dovetail",
        "o": "slot_o",
    }
    profile_to_semantic_aliases = {
        "u": ("slot_u_side_plane", "slot_u_bottom_plane"),
        "v": ("slot_v_side_plane", "slot_v_bottom_plane"),
        "dovetail": ("slot_dovetail_side_plane", "slot_dovetail_bottom_plane"),
        "o": ("slot_o_side_plane", "slot_o_bottom_plane"),
    }
    width_to_base_range = (0.045, 0.075)
    depth_to_height_range = (0.12, 0.28)
    sweep_to_profile_ratio_range = (5.0, 10.0)

    def __init__(self, semantic_subtype_index):
        super().__init__(semantic_subtype_index)
        self.feature_name = "swept_slot"
        self.semantic_map = self.build_semantic_map([
            "stock_face",
            "slot_u_side_plane",
            "slot_u_bottom_plane",
            "slot_v_side_plane",
            "slot_v_bottom_plane",
            "slot_dovetail_side_plane",
            "slot_dovetail_bottom_plane",
            "slot_o_side_plane",
            "slot_o_bottom_plane",
        ])

    def apply_feature(self, stock, L, W, H, num_instances, start_inst_id=1, occupied_zones=None):
        if occupied_zones is None:
            occupied_zones = []

        params_list = []
        actual_added = 0
        min_base_size = min(L, W)

        for _ in range(num_instances):
            profile_shape = random.choice(self.supported_profile_shapes)
            macro_key = self.profile_to_macro_key[profile_shape]
            slot_w = random.uniform(
                min_base_size * self.width_to_base_range[0],
                min_base_size * self.width_to_base_range[1],
            )
            slot_depth = random.uniform(H * self.depth_to_height_range[0], H * self.depth_to_height_range[1])
            slot_w, slot_depth = _constrain_profile_dimensions(profile_shape, slot_w, slot_depth)

            valid = False
            path = None
            for _attempt in range(80):
                tx = random.uniform(-L / 2.0 + slot_w + 4.0, L / 2.0 - slot_w - 4.0)
                ty = random.uniform(-W / 2.0 + slot_w + 4.0, W / 2.0 - slot_w - 4.0)
                path_angle = random.uniform(0.0, math.pi)
                try:
                    candidate_path = _sample_slot_path_for_current_stock(
                        tx,
                        ty,
                        path_angle,
                        L,
                        W,
                        self.generation_context.get("current_stock_profile", {}),
                        slot_w,
                    )
                except ValueError:
                    continue

                candidate_l = candidate_path["length"]
                if candidate_l / max(slot_w, 1e-9) < self.sweep_to_profile_ratio_range[0]:
                    continue

                candidate_footprint = polyline_capsule_footprint(
                    candidate_path["path_points_xy"],
                    slot_w,
                    operation="cut",
                    feature_kind=macro_key,
                )
                if footprint_conflicts(candidate_footprint, occupied_zones, clearance=4.0):
                    continue

                path = candidate_path
                valid = True
                break

            if not valid:
                continue

            cx, cy = path["center"]
            slot_l = path["length"]
            angle = path["angle"]
            bounding_radius = slot_l / 2.0 + slot_w / 2.0
            top_margin = max(1.0, min(2.5, H * 0.05))
            inst_id = start_inst_id + actual_added

            tool_path = build_overrun_slot_tool_path(path, slot_width=slot_w)

            try:
                tool = build_vertical_swept_slot_tool_from_path(
                    profile_shape=profile_shape,
                    path_points_xy=tool_path["tool_path_points_xy"],
                    surface_z=H,
                    width=slot_w,
                    depth=slot_depth,
                    top_margin=top_margin,
                    path_curve_kind=tool_path["tool_path_curve_kind"],
                )
            except Exception:
                continue

            params = {
                "inst": inst_id,
                "macro_key": macro_key,
                "cx": cx,
                "cy": cy,
                "w": slot_w,
                "l": slot_l,
                "depth": slot_depth,
                "angle": angle,
                "path_angle": angle,
                "path_start_xy": path["start"],
                "path_end_xy": path["end"],
                "path_seed_xy": path["seed"],
                "path_points_xy": path["path_points_xy"],
                "visible_path_points_xy": path["path_points_xy"],
                "tool_path_start_xy": tool_path["tool_path_start_xy"],
                "tool_path_end_xy": tool_path["tool_path_end_xy"],
                "tool_path_points_xy": tool_path["tool_path_points_xy"],
                "tool_path_length": tool_path["tool_path_length"],
                "tool_path_curve_kind": tool_path["tool_path_curve_kind"],
                "tool_overrun_distance": tool_path["tool_overrun_distance"],
                "tool_path_extends_visible_path": tool_path["tool_path_extends_visible_path"],
                "path_boundary_kind": path["boundary_kind"],
                "path_type": path["path_type"],
                "path_curve_kind": path["path_curve_kind"],
                "slot_modeling_mode": "vertical_closed_profile_surface_sweep",
                "anchor_face": "current_top_planar_face",
                "profile_orientation": "vertical_cross_section",
                "top_margin": top_margin,
                "radius": bounding_radius,
            }
            if "path_sagitta" in path:
                params["path_sagitta"] = path["path_sagitta"]
            params.update(swept_profile_metadata(
                profile_shape,
                slot_l,
                slot_w,
                slot_depth,
                path_type=path["path_type"],
                path_curve_kind=path["path_curve_kind"],
                path_on_face=True,
                path_start_on_boundary=True,
                path_end_on_boundary=True,
                anchor_face_kind="existing_planar_face",
            ))
            params_list.append(params)
            occupied_zones.append(polyline_capsule_footprint(
                path["path_points_xy"],
                slot_w,
                operation="cut",
                feature_kind=macro_key,
            ))

            cut = BRepAlgoAPI_Cut(stock, tool)
            cut.Build()
            stock = cut.Shape()
            actual_added += 1

        return stock, actual_added, occupied_zones, params_list

    def match_face(self, surf, pnt, face_cx, face_cy, H, params_list):
        surface_type = surf.GetType()
        if surface_type not in _SWEPT_SLOT_SURFACE_TYPES:
            return None

        px, py, pz = pnt.X(), pnt.Y(), pnt.Z()
        try:
            normal = surf.Plane().Axis().Direction()
        except Exception:
            normal = None

        for hp in params_list:
            if not _point_is_inside_path_corridor(hp, px, py, tolerance=0.35):
                continue
            if not (H - hp["depth"] - 0.25 <= pz <= H + hp.get("top_margin", 1.0) + 0.25):
                continue
            if surface_type == GeomAbs_Plane and pz > H - 0.05 and normal is not None and abs(normal.Z()) > 0.9:
                continue

            side_alias, bottom_alias = self.profile_to_semantic_aliases[hp["profile_shape"]]
            if surface_type != GeomAbs_Plane:
                if hp["profile_shape"] == "u":
                    if pz <= H - hp["depth"] * 0.35:
                        return self.semantic_map[bottom_alias], hp["inst"], hp["macro_key"]
                    return self.semantic_map[side_alias], hp["inst"], hp["macro_key"]
                if hp["profile_shape"] == "o":
                    return self.semantic_map[side_alias], hp["inst"], hp["macro_key"]
                if pz <= H - hp["depth"] * 0.55:
                    return self.semantic_map[bottom_alias], hp["inst"], hp["macro_key"]
                return self.semantic_map[side_alias], hp["inst"], hp["macro_key"]
            if pz <= H - hp["depth"] * 0.55:
                return self.semantic_map[bottom_alias], hp["inst"], hp["macro_key"]
            return self.semantic_map[side_alias], hp["inst"], hp["macro_key"]

        return None


def _to_path_local(params, px, py):
    dx = px - params["cx"]
    dy = py - params["cy"]
    angle = params["angle"]
    local_s = dx * math.cos(angle) + dy * math.sin(angle)
    local_t = -dx * math.sin(angle) + dy * math.cos(angle)
    return local_s, local_t


def build_overrun_slot_tool_path(path: dict, slot_width: float) -> dict:
    visible_points = [(float(x), float(y)) for x, y in path["path_points_xy"]]
    overrun = _adaptive_slot_tool_overrun(path_length=float(path["length"]), slot_width=float(slot_width))
    tool_points = _extend_polyline_endpoints(visible_points, overrun)
    return {
        **path,
        "path_points_xy": visible_points,
        "tool_path_start_xy": tool_points[0],
        "tool_path_end_xy": tool_points[-1],
        "tool_path_points_xy": tool_points,
        "tool_path_length": polyline_length(tool_points),
        "tool_path_curve_kind": "polyline_overrun",
        "tool_overrun_distance": overrun,
        "tool_path_extends_visible_path": True,
    }


def _adaptive_slot_tool_overrun(path_length: float, slot_width: float) -> float:
    width_term = float(slot_width) * 2.5
    length_term = float(path_length) * 0.08
    return max(width_term, length_term, 2.0)


def _extend_polyline_endpoints(points: list[tuple[float, float]], overrun: float) -> list[tuple[float, float]]:
    if len(points) < 2:
        raise ValueError("Slot tool path requires at least two points.")
    safe_points = [(float(x), float(y)) for x, y in points]

    start_direction = _unit_direction(
        safe_points[0][0] - safe_points[1][0],
        safe_points[0][1] - safe_points[1][1],
    )
    end_direction = _unit_direction(
        safe_points[-1][0] - safe_points[-2][0],
        safe_points[-1][1] - safe_points[-2][1],
    )

    extended = list(safe_points)
    extended[0] = (
        safe_points[0][0] + start_direction[0] * float(overrun),
        safe_points[0][1] + start_direction[1] * float(overrun),
    )
    extended[-1] = (
        safe_points[-1][0] + end_direction[0] * float(overrun),
        safe_points[-1][1] + end_direction[1] * float(overrun),
    )
    return extended


def _unit_direction(dx: float, dy: float) -> tuple[float, float]:
    length = math.hypot(float(dx), float(dy))
    if length <= 1e-9:
        raise ValueError("Cannot extend a zero-length slot path segment.")
    return float(dx) / length, float(dy) / length


_SWEPT_SLOT_SURFACE_TYPES = {
    GeomAbs_Plane,
    GeomAbs_Cylinder,
    GeomAbs_SurfaceOfRevolution,
    GeomAbs_SurfaceOfExtrusion,
    GeomAbs_BSplineSurface,
    GeomAbs_Torus,
}


def _point_is_inside_path_corridor(params, px, py, tolerance=0.35):
    points = params.get("path_points_xy")
    if points:
        return point_to_polyline_distance(float(px), float(py), [(float(x), float(y)) for x, y in points]) <= params["w"] / 2.0 + float(tolerance)
    local_s, local_t = _to_path_local(params, px, py)
    return abs(local_s) <= params["l"] / 2.0 + float(tolerance) and abs(local_t) <= params["w"] / 2.0 + float(tolerance)


def _straight_path_for_current_stock(seed_x, seed_y, path_angle, L, W, stock_profile):
    stock_type = stock_profile.get("stock_type")
    dimensions = stock_profile.get("dimensions", {})
    if stock_type == "cylinder" and "radius" in dimensions:
        path = straight_boundary_path_on_circle(seed_x, seed_y, path_angle, float(dimensions["radius"]))
        path["boundary_kind"] = "circular_stock_boundary"
        return path

    path = straight_boundary_path_on_box(seed_x, seed_y, path_angle, L, W)
    path["boundary_kind"] = "rectangular_workspace_boundary"
    return path


def _sample_slot_path_for_current_stock(seed_x, seed_y, path_angle, L, W, stock_profile, slot_width):
    straight_path = _straight_path_for_current_stock(seed_x, seed_y, path_angle, L, W, stock_profile)
    if random.random() >= 0.35:
        return _path_metadata_from_points(
            _straight_path_points(straight_path),
            path_curve_kind="line",
            seed=straight_path["seed"],
            boundary_kind=straight_path["boundary_kind"],
        )

    chord_length = straight_path["length"]
    sagitta = random.choice((-1.0, 1.0)) * random.uniform(0.12, 0.25) * chord_length
    arc_path = _arc_path_from_straight_path(straight_path, sagitta=sagitta, sample_count=17)
    if not _path_points_inside_current_stock(arc_path["path_points_xy"], L, W, stock_profile, tolerance=slot_width / 2.0):
        raise ValueError("Curved swept slot path leaves current stock workspace.")
    return arc_path


def _straight_path_points(path: dict) -> list[tuple[float, float]]:
    return [
        (float(path["start"][0]), float(path["start"][1])),
        (float(path["end"][0]), float(path["end"][1])),
    ]


def _arc_path_from_straight_path(path: dict, sagitta: float, sample_count: int = 17) -> dict:
    points = sample_circular_arc_points(path["start"], path["end"], sagitta=sagitta, sample_count=sample_count)
    metadata = _path_metadata_from_points(
        points,
        path_curve_kind="circular_arc",
        seed=path["seed"],
        boundary_kind=path["boundary_kind"],
    )
    metadata["path_sagitta"] = float(sagitta)
    return metadata


def _path_metadata_from_points(
    points: list[tuple[float, float]],
    path_curve_kind: str,
    seed: tuple[float, float] | None = None,
    boundary_kind: str | None = None,
) -> dict:
    if len(points) < 2:
        raise ValueError("Slot path requires at least two points.")
    safe_points = [(float(x), float(y)) for x, y in points]
    xs = [point[0] for point in safe_points]
    ys = [point[1] for point in safe_points]
    path_type = "straight" if path_curve_kind == "line" and len(safe_points) == 2 else "curved"
    return {
        "start": safe_points[0],
        "end": safe_points[-1],
        "center": ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0),
        "length": polyline_length(safe_points),
        "angle": path_tangent_angle(safe_points),
        "seed": seed if seed is not None else safe_points[len(safe_points) // 2],
        "boundary_kind": boundary_kind or "unknown_boundary",
        "path_type": path_type,
        "path_curve_kind": path_curve_kind,
        "path_points_xy": safe_points,
    }


def _path_points_inside_current_stock(points, L, W, stock_profile, tolerance=0.0):
    stock_type = stock_profile.get("stock_type")
    dimensions = stock_profile.get("dimensions", {})
    if stock_type == "cylinder" and "radius" in dimensions:
        radius = float(dimensions["radius"]) + float(tolerance)
        return all(math.hypot(x, y) <= radius + 1e-6 for x, y in points)

    half_l = float(L) / 2.0 + float(tolerance)
    half_w = float(W) / 2.0 + float(tolerance)
    return all(-half_l - 1e-6 <= x <= half_l + 1e-6 and -half_w - 1e-6 <= y <= half_w + 1e-6 for x, y in points)


def _constrain_profile_dimensions(profile_shape, slot_w, slot_depth):
    if profile_shape == "u":
        return slot_w, max(slot_depth, slot_w * 0.55)
    if profile_shape == "o":
        return slot_w, slot_w / 2.0
    return slot_w, slot_depth
