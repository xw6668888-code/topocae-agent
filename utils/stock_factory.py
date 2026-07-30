from __future__ import annotations

import math
import random
from copy import deepcopy

from utils.generation_context import DEFAULT_STOCK_PROFILE_CONFIG, deep_merge


def weighted_choice(weights: dict[str, float], rng: random.Random | None = None) -> str:
    rng = rng or random
    positive_items = [(key, float(value)) for key, value in weights.items() if float(value) > 0.0]
    if not positive_items:
        return "block"

    total = sum(value for _key, value in positive_items)
    pick = rng.uniform(0.0, total)
    running = 0.0
    for key, value in positive_items:
        running += value
        if pick <= running:
            return key
    return positive_items[-1][0]


def sample_stock_spec(
    L: float,
    W: float,
    H: float,
    config: dict | None = None,
    rng: random.Random | None = None,
) -> dict:
    rng = rng or random
    cfg = deep_merge(DEFAULT_STOCK_PROFILE_CONFIG, config or {})
    if config and "weights" in config:
        cfg["weights"] = dict(config["weights"])
    if not cfg.get("enabled", True):
        stock_type = "block"
    else:
        stock_type = weighted_choice(cfg.get("weights", {}), rng)

    builders = {
        "block": _block_spec,
        "plate": _plate_spec,
        "beam": _beam_spec,
        "cylinder": _cylinder_spec,
        "l_plate": _l_plate_spec,
        "composite": lambda l, w, h, local_rng: _composite_spec(l, w, h, cfg, local_rng),
    }
    return builders.get(stock_type, _block_spec)(float(L), float(W), float(H), rng)


def _base_spec(stock_type: str, L: float, W: float, H: float, placement_mode: str = "base_top") -> dict:
    return {
        "stock_type": stock_type,
        "dimensions": {
            "L": L,
            "W": W,
            "H": H,
        },
        "feature_workspace": {
            "L": L,
            "W": W,
            "H": H,
            "placement_mode": placement_mode,
        },
        "component_count": 1,
        "components": [
            {
                "type": "box",
                "role": "base",
                "center": [0.0, 0.0, H / 2.0],
                "size": [L, W, H],
            }
        ],
    }


def _block_spec(L: float, W: float, H: float, _rng: random.Random) -> dict:
    edge = (L + W + max(H * 2.0, min(L, W) * 0.7)) / 3.0
    block_L = max(70.0, edge * 1.05)
    block_W = max(70.0, edge * 0.95)
    block_H = max(35.0, min(edge * 0.75, H * 1.15))
    return _base_spec("block", block_L, block_W, block_H)


def _plate_spec(L: float, W: float, H: float, _rng: random.Random) -> dict:
    plate_L = max(L, W) * 1.25
    plate_W = min(L, W) * 0.95
    plate_H = max(8.0, min(H * 0.25, min(plate_L, plate_W) * 0.12))
    return _base_spec("plate", plate_L, plate_W, plate_H)


def _beam_spec(L: float, W: float, H: float, _rng: random.Random) -> dict:
    beam_L = max(L * 2.60, W * 2.90)
    beam_W = max(14.0, min(W * 0.28, L * 0.25))
    beam_H = max(14.0, min(H * 0.55, beam_W * 0.90))
    return _base_spec("beam", beam_L, beam_W, beam_H)


def _cylinder_spec(L: float, W: float, H: float, _rng: random.Random) -> dict:
    radius = min(L, W) / 2.0
    diameter = 2.0 * radius
    safe_square = math.sqrt(2.0) * radius * 0.90
    spec = _base_spec("cylinder", diameter, diameter, H, placement_mode="top_safe_square")
    spec["dimensions"]["radius"] = radius
    spec["dimensions"]["diameter"] = diameter
    spec["feature_workspace"]["L"] = safe_square
    spec["feature_workspace"]["W"] = safe_square
    spec["components"] = [
        {
            "type": "cylinder",
            "role": "base",
            "center": [0.0, 0.0, H / 2.0],
            "radius": radius,
            "height": H,
        }
    ]
    return spec


def _l_plate_spec(L: float, W: float, H: float, rng: random.Random) -> dict:
    plate_L = max(L, W) * 1.20
    plate_W = min(L, W) * 0.90
    base_t = max(8.0, min(H * 0.22, min(plate_L, plate_W) * 0.10))
    wall_t = max(base_t * 0.85, min(plate_L, plate_W) * 0.055)
    wall_h = max(base_t * 4.5, H * 0.72)
    wall_l = plate_W * rng.uniform(0.68, 0.88)
    wall_side = rng.choice(["left", "right", "bottom", "top"])

    base_component = {
        "type": "box",
        "role": "base_plate",
        "center": [0.0, 0.0, base_t / 2.0],
        "size": [plate_L, plate_W, base_t],
    }
    vertical_component, rib_context = _l_plate_wall_component_and_context(
        plate_L=plate_L,
        plate_W=plate_W,
        base_t=base_t,
        wall_t=wall_t,
        wall_h=wall_h,
        wall_l=wall_l,
        wall_side=wall_side,
    )
    spec = _base_spec("l_plate", plate_L, plate_W, base_t + wall_h, placement_mode="l_plate_corner")
    spec["dimensions"].update({
        "base_plate_thickness": base_t,
        "vertical_plate_thickness": wall_t,
        "vertical_plate_height": wall_h,
        "vertical_plate_length": wall_l,
        "wall_side": wall_side,
    })
    spec["feature_workspace"] = {
        "L": plate_L,
        "W": plate_W,
        "H": base_t,
        "placement_mode": "l_plate_corner",
    }
    spec["component_count"] = 2
    spec["components"] = [base_component, vertical_component]
    spec["rib_context"] = rib_context
    return spec


def _l_plate_wall_component_and_context(
    *,
    plate_L: float,
    plate_W: float,
    base_t: float,
    wall_t: float,
    wall_h: float,
    wall_l: float,
    wall_side: str,
) -> tuple[dict, dict]:
    half_l = plate_L / 2.0
    half_w = plate_W / 2.0
    margin = wall_t * 1.5 + 4.0

    if wall_side in {"left", "right"}:
        side_sign = -1.0 if wall_side == "left" else 1.0
        wall_center_x = side_sign * (half_l - wall_t / 2.0)
        wall_inner_x = side_sign * (half_l - wall_t)
        inward_x = -side_sign
        component = {
            "type": "box",
            "role": "vertical_plate",
            "center": [wall_center_x, 0.0, base_t + wall_h / 2.0],
            "size": [wall_t, wall_l, wall_h],
        }
        context = {
            "type": "l_plate_corner",
            "support_label": "stock",
            "base_plane_z": base_t,
            "wall_side": wall_side,
            "wall_inner_x": wall_inner_x,
            "wall_inner_y": 0.0,
            "inward_x": inward_x,
            "inward_y": 0.0,
            "wall_t": wall_t,
            "wall_l": wall_l,
            "wall_h": wall_h,
            "anchor_span_range": [-wall_l / 2.0 + margin, wall_l / 2.0 - margin],
            "anchor_axis": "y",
        }
        return component, context

    side_sign = -1.0 if wall_side == "bottom" else 1.0
    wall_center_y = side_sign * (half_w - wall_t / 2.0)
    wall_inner_y = side_sign * (half_w - wall_t)
    inward_y = -side_sign
    component = {
        "type": "box",
        "role": "vertical_plate",
        "center": [0.0, wall_center_y, base_t + wall_h / 2.0],
        "size": [wall_l, wall_t, wall_h],
    }
    context = {
        "type": "l_plate_corner",
        "support_label": "stock",
        "base_plane_z": base_t,
        "wall_side": wall_side,
        "wall_inner_x": 0.0,
        "wall_inner_y": wall_inner_y,
        "inward_x": 0.0,
        "inward_y": inward_y,
        "wall_t": wall_t,
        "wall_l": wall_l,
        "wall_h": wall_h,
        "anchor_span_range": [-wall_l / 2.0 + margin, wall_l / 2.0 - margin],
        "anchor_axis": "x",
    }
    return component, context


def _composite_spec(L: float, W: float, H: float, config: dict, rng: random.Random) -> dict:
    spec = _base_spec("composite", L, W, H, placement_mode="base_top")
    min_count, max_count = config.get("composite_component_count_range", [2, 3])
    component_count = rng.randint(int(min_count), int(max_count))
    components = deepcopy(spec["components"])

    for component_index in range(1, component_count):
        role = rng.choice(["top_pad", "side_lug", "end_lug"])
        if role == "top_pad":
            size = [
                rng.uniform(L * 0.25, L * 0.55),
                rng.uniform(W * 0.25, W * 0.55),
                rng.uniform(H * 0.18, H * 0.35),
            ]
            center = [
                rng.uniform(-L * 0.20, L * 0.20),
                rng.uniform(-W * 0.20, W * 0.20),
                H + size[2] / 2.0,
            ]
        elif role == "side_lug":
            size = [
                rng.uniform(L * 0.18, L * 0.32),
                rng.uniform(W * 0.25, W * 0.50),
                rng.uniform(H * 0.45, H * 0.90),
            ]
            side = rng.choice([-1.0, 1.0])
            center = [
                side * (L / 2.0 + size[0] / 2.0),
                rng.uniform(-W * 0.18, W * 0.18),
                size[2] / 2.0,
            ]
        else:
            size = [
                rng.uniform(L * 0.25, L * 0.50),
                rng.uniform(W * 0.18, W * 0.32),
                rng.uniform(H * 0.45, H * 0.90),
            ]
            side = rng.choice([-1.0, 1.0])
            center = [
                rng.uniform(-L * 0.18, L * 0.18),
                side * (W / 2.0 + size[1] / 2.0),
                size[2] / 2.0,
            ]

        components.append({
            "type": "box",
            "role": role,
            "center": center,
            "size": size,
        })

    spec["component_count"] = len(components)
    spec["components"] = components
    return spec


def build_stock_shape(stock_spec: dict):
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt

    shape = None
    for component in stock_spec["components"]:
        if component["type"] == "cylinder":
            radius = component["radius"]
            height = component["height"]
            component_shape = BRepPrimAPI_MakeCylinder(
                gp_Ax2(gp_Pnt(0.0, 0.0, 0.0), gp_Dir(0, 0, 1)),
                radius,
                height,
            ).Shape()
        else:
            cx, cy, cz = component["center"]
            sx, sy, sz = component["size"]
            component_shape = BRepPrimAPI_MakeBox(
                gp_Pnt(cx - sx / 2.0, cy - sy / 2.0, cz - sz / 2.0),
                gp_Pnt(cx + sx / 2.0, cy + sy / 2.0, cz + sz / 2.0),
            ).Shape()

        if shape is None:
            shape = component_shape
        else:
            fuse = BRepAlgoAPI_Fuse(shape, component_shape)
            fuse.Build()
            shape = fuse.Shape()

    return shape
