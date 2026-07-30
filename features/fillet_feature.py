import math
import random

from OCC.Core.BRepFilletAPI import BRepFilletAPI_MakeFillet
from OCC.Core.GeomAbs import GeomAbs_Cylinder, GeomAbs_SurfaceOfRevolution, GeomAbs_Torus

from features.base_feature import BaseFeatureGenerator
from features.transition_edge_selection import (
    compute_transition_size_limits,
    extract_transition_edge_candidates,
    merge_transition_policy,
    public_transition_parameters,
)


class FilletGenerator(BaseFeatureGenerator):
    def __init__(self, semantic_subtype_index):
        super().__init__(semantic_subtype_index)
        self.feature_name = "fillet"
        self.semantic_map = self.build_semantic_map([
            "stock_face",
            "fillet_blend_surface",
        ])

    def _edge_candidates(self, stock, L, W, H, existing_params):
        policy = self._transition_policy()
        clearance = policy["midpoint_clearance_factor"] * policy["max_size"]
        return extract_transition_edge_candidates(
            stock,
            existing_params,
            min_edge_length=policy["min_edge_length"],
            midpoint_clearance=clearance,
        )

    def _transition_policy(self):
        return merge_transition_policy(self.generation_context)

    def apply_feature(self, stock, L, W, H, num_instances, start_inst_id=1, occupied_zones=None):
        params_list = []
        actual_added = 0

        for _ in range(num_instances):
            candidates = self._edge_candidates(stock, L, W, H, params_list)
            if not candidates:
                break
            cand = random.choice(candidates)
            min_radius, max_radius = compute_transition_size_limits(
                cand,
                L,
                W,
                H,
                self._transition_policy(),
            )
            if max_radius <= min_radius:
                continue
            radius = random.uniform(min_radius, max_radius)

            builder = BRepFilletAPI_MakeFillet(stock)
            builder.Add(radius, cand["edge"])
            try:
                builder.Build()
            except RuntimeError:
                continue
            if not builder.IsDone():
                continue

            stock = builder.Shape()
            params_list.append(public_transition_parameters(
                inst=start_inst_id + actual_added,
                size_key="radius",
                size_value=radius,
                candidate=cand,
            ))
            actual_added += 1

        return stock, actual_added, occupied_zones if occupied_zones is not None else [], params_list

    def match_face(self, surf, pnt, face_cx, face_cy, H, params_list):
        if surf.GetType() not in {GeomAbs_Cylinder, GeomAbs_Torus, GeomAbs_SurfaceOfRevolution}:
            return None

        px, py, pz = pnt.X(), pnt.Y(), pnt.Z()

        for hp in params_list:
            if surf.GetType() == GeomAbs_Cylinder and hp.get("curve_type", "line") == "line":
                cyl = surf.Cylinder()
                axis = cyl.Axis().Direction()
                axis_vec = [axis.X(), axis.Y(), axis.Z()]
                axis_dot = sum(a * b for a, b in zip(axis_vec, hp["axis"]))
                if abs(abs(axis_dot) - 1.0) > 0.05:
                    continue
                if abs(cyl.Radius() - hp["radius"]) > 0.15:
                    continue
            if math.dist([px, py, pz], hp["midpoint"]) > (hp["radius"] * 3.0 + 1.0):
                continue
            return self.semantic_map["fillet_blend_surface"], hp["inst"], self.feature_name

        return None
