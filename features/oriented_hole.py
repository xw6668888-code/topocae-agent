import math
import random

from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeRevol
from OCC.Core.gp import gp_Ax1, gp_Dir, gp_Pnt

from utils.feature_footprint import circle_footprint, footprint_conflicts


def axis_tuple(axis_key):
    if axis_key == "x":
        return (1, 0, 0)
    if axis_key == "y":
        return (0, 1, 0)
    return (0, 0, 1)


def sample_axis_placement(axis_key, L, W, H, radius, occupied_zones, through=True, stock_profile=None):
    margin = radius + 5.0
    if axis_key == "x":
        if W <= 2.0 * margin or H <= 2.0 * margin:
            return None
        cy = random.uniform(-W / 2.0 + margin, W / 2.0 - margin)
        cz = random.uniform(margin, H - margin)
        side = random.choice([-1.0, 1.0])
        sign = 1.0 if through or side < 0 else -1.0
        entry_coord = -L / 2.0 if through or side < 0 else L / 2.0
        placement = {
            "axis": "x",
            "axis_sign": sign,
            "cx": 0.0,
            "cy": cy,
            "cz": cz,
            "entry_coord": entry_coord,
            "span": L,
            "zone_plane": "yz",
            "zone_a": cy,
            "zone_b": cz,
        }
    elif axis_key == "y":
        if L <= 2.0 * margin or H <= 2.0 * margin:
            return None
        cx = random.uniform(-L / 2.0 + margin, L / 2.0 - margin)
        cz = random.uniform(margin, H - margin)
        side = random.choice([-1.0, 1.0])
        sign = 1.0 if through or side < 0 else -1.0
        entry_coord = -W / 2.0 if through or side < 0 else W / 2.0
        placement = {
            "axis": "y",
            "axis_sign": sign,
            "cx": cx,
            "cy": 0.0,
            "cz": cz,
            "entry_coord": entry_coord,
            "span": W,
            "zone_plane": "xz",
            "zone_a": cx,
            "zone_b": cz,
        }
    else:
        if L <= 2.0 * margin or W <= 2.0 * margin:
            return None
        cx = random.uniform(-L / 2.0 + margin, L / 2.0 - margin)
        cy = random.uniform(-W / 2.0 + margin, W / 2.0 - margin)
        placement = {
            "axis": "z",
            "axis_sign": -1.0 if not through else 1.0,
            "cx": cx,
            "cy": cy,
            "cz": H / 2.0,
            "entry_coord": H if not through else -1.0,
            "span": H,
            "zone_plane": "xy",
            "zone_a": cx,
            "zone_b": cy,
        }

    if is_axis_placement_entry_exposed(placement, radius, stock_profile) and is_zone_free(placement, radius, occupied_zones):
        return placement
    return None


def is_zone_free(placement, radius, occupied_zones):
    candidate = circle_footprint(
        placement["zone_a"],
        placement["zone_b"],
        radius,
        operation="cut",
        zone_plane=placement["zone_plane"],
        feature_kind="hole",
    )
    return not footprint_conflicts(candidate, occupied_zones, clearance=4.0)


def occupied_zone(placement, radius):
    return circle_footprint(
        placement["zone_a"],
        placement["zone_b"],
        radius,
        operation="cut",
        zone_plane=placement["zone_plane"],
        feature_kind="hole",
    )


def is_axis_placement_entry_exposed(placement, radius, stock_profile=None):
    stock_profile = stock_profile or {}
    stock_type = stock_profile.get("stock_type")
    axis = placement.get("axis", "z")

    if stock_type == "cylinder":
        return axis == "z"

    if stock_type == "composite":
        if axis != "z":
            return False
        entry_x = float(placement.get("zone_a", placement.get("cx", 0.0)))
        entry_y = float(placement.get("zone_b", placement.get("cy", 0.0)))
        clearance = float(radius)
        for component in stock_profile.get("components", []):
            if component.get("role") != "top_pad":
                continue
            center = component.get("center", [0.0, 0.0, 0.0])
            size = component.get("size", [0.0, 0.0, 0.0])
            if (
                abs(entry_x - float(center[0])) <= float(size[0]) / 2.0 + clearance
                and abs(entry_y - float(center[1])) <= float(size[1]) / 2.0 + clearance
            ):
                return False
        return True

    return True


def coordinate_at(placement, offset):
    return placement["entry_coord"] + placement["axis_sign"] * offset


def profile_point(placement, radial, coord):
    axis = placement["axis"]
    if axis == "x":
        return gp_Pnt(coord, placement["cy"] + radial, placement["cz"])
    if axis == "y":
        return gp_Pnt(placement["cx"] + radial, coord, placement["cz"])
    return gp_Pnt(placement["cx"] + radial, placement["cy"], coord)


def revolve_axis(placement):
    axis = placement["axis"]
    if axis == "x":
        return gp_Ax1(gp_Pnt(0.0, placement["cy"], placement["cz"]), gp_Dir(1, 0, 0))
    if axis == "y":
        return gp_Ax1(gp_Pnt(placement["cx"], 0.0, placement["cz"]), gp_Dir(0, 1, 0))
    return gp_Ax1(gp_Pnt(placement["cx"], placement["cy"], 0.0), gp_Dir(0, 0, 1))


def make_revolved_tool(placement, radial_offset_pairs):
    poly = BRepBuilderAPI_MakePolygon()
    for radial, offset in radial_offset_pairs:
        poly.Add(profile_point(placement, radial, coordinate_at(placement, offset)))
    poly.Close()
    profile = BRepBuilderAPI_MakeFace(poly.Wire()).Face()
    return BRepPrimAPI_MakeRevol(profile, revolve_axis(placement)).Shape()


def axis_cross_distance_from_cylinder(cylinder, params):
    location = cylinder.Axis().Location()
    axis = params.get("axis", "z")
    if axis == "x":
        return math.hypot(location.Y() - params["cy"], location.Z() - params["cz"])
    if axis == "y":
        return math.hypot(location.X() - params["cx"], location.Z() - params["cz"])
    return math.hypot(location.X() - params["cx"], location.Y() - params["cy"])


def point_axis_coord(point, axis_key):
    if axis_key == "x":
        return point.X()
    if axis_key == "y":
        return point.Y()
    return point.Z()


def point_cross_distance(point, params):
    axis = params.get("axis", "z")
    if axis == "x":
        return math.hypot(point.Y() - params["cy"], point.Z() - params["cz"])
    if axis == "y":
        return math.hypot(point.X() - params["cx"], point.Z() - params["cz"])
    return math.hypot(point.X() - params["cx"], point.Y() - params["cy"])


def is_surface_axis_parallel(direction, axis_key, tolerance=1e-4):
    return direction.IsParallel(gp_Dir(*axis_tuple(axis_key)), tolerance)
