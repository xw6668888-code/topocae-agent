"""Generate the reviewable P2-A1 feature-role-operation policy YAML."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


SURFACE_OPERATIONS = [
    "fixed_support",
    "displacement",
    "pressure",
    "surface_traction",
    "local_mesh_size",
]

FEATURE_META = {
    "through_hole_round": ("hole", ["through_hole"]),
    "blind_hole_round": ("hole", ["blind_hole"]),
    "stepped_hole_round": ("hole", ["stepped_hole"]),
    "counterbore_hole": ("hole", []),
    "countersink_hole": ("hole", []),
    "obround_hole": ("hole", []),
    "slot_rect": ("slot", []),
    "slot_u": ("slot", []),
    "slot_v": ("slot", ["swept_slot_v"]),
    "slot_o": ("slot", ["swept_slot_o"]),
    "pocket_rect": ("pocket", []),
    "pocket_circular": ("pocket", []),
    "cylinder_boss": ("protrusion", []),
    "rectangular_boss": ("protrusion", ["pad", "rectangle_boss"]),
    "rib": ("stiffener", ["web"]),
    "fillet": ("transition", []),
    "chamfer": ("transition", []),
}

# role, selection status, region kind, semantic subtype candidates, engineering note
ROLE_SPECS = {
    "through_hole_round": [
        ("feature_region", "direct", "feature_region", ["through_hole_cylinder"], "Complete through-hole wall region."),
        ("hole_wall", "direct", "face_set", ["through_hole_cylinder"], "Cylindrical through-hole wall."),
    ],
    "blind_hole_round": [
        ("feature_region", "composite", "feature_region", ["blind_hole_cylinder", "blind_hole_bottom"], "Complete blind-hole inner region."),
        ("hole_wall", "direct", "face_set", ["blind_hole_cylinder"], "Blind-hole cylindrical wall."),
        ("hole_bottom", "direct", "face_set", ["blind_hole_bottom"], "Blind-hole bottom face."),
    ],
    "stepped_hole_round": [
        ("feature_region", "composite", "feature_region", ["stepped_hole_large_hole_cylinder", "stepped_hole_step_plane", "stepped_hole_small_hole_cylinder", "stepped_hole_small_hole_bottom"], "Complete stepped-hole inner region."),
        ("large_hole_wall", "direct", "face_set", ["stepped_hole_large_hole_cylinder"], "Large-diameter cylindrical wall."),
        ("step_face", "direct", "face_set", ["stepped_hole_step_plane"], "Annular step face."),
        ("small_hole_wall", "direct", "face_set", ["stepped_hole_small_hole_cylinder"], "Small-diameter cylindrical wall."),
        ("hole_bottom", "direct", "face_set", ["stepped_hole_small_hole_bottom"], "Terminal bottom face."),
    ],
    "counterbore_hole": [
        ("feature_region", "composite", "feature_region", ["counterbore_large_cylinder", "counterbore_step_plane", "counterbore_small_cylinder"], "Complete counterbore inner region."),
        ("large_hole_wall", "direct", "face_set", ["counterbore_large_cylinder"], "Large counterbore wall."),
        ("step_face", "direct", "face_set", ["counterbore_step_plane"], "Counterbore annular step face."),
        ("through_small_hole_wall", "direct", "face_set", ["counterbore_small_cylinder"], "Through small-hole wall."),
    ],
    "countersink_hole": [
        ("feature_region", "composite", "feature_region", ["countersink_cone", "countersink_small_cylinder"], "Complete countersink inner region."),
        ("countersink_conical_face", "direct", "face_set", ["countersink_cone"], "Countersink conical face."),
        ("through_hole_wall", "direct", "face_set", ["countersink_small_cylinder"], "Through cylindrical wall below the cone."),
    ],
    "obround_hole": [
        ("feature_region", "composite", "feature_region", ["obround_hole_side_plane", "obround_hole_end_cylinder"], "Complete obround-hole wall region."),
        ("hole_wall", "composite", "face_set", ["obround_hole_side_plane", "obround_hole_end_cylinder"], "All straight and rounded wall faces."),
        ("straight_side_wall", "direct", "face_set", ["obround_hole_side_plane"], "Straight side-wall faces."),
        ("rounded_end_wall", "direct", "face_set", ["obround_hole_end_cylinder"], "Rounded end-wall faces."),
    ],
    "slot_rect": [
        ("feature_region", "composite", "feature_region", ["slot_rect_side_plane", "slot_rect_bottom_plane"], "Complete slot region using all known inner subtypes."),
        ("complete_slot_inner_wall", "composite", "face_set", ["slot_rect_side_plane", "slot_rect_bottom_plane"], "All ontology-visible slot inner faces."),
        ("side_wall", "direct", "face_set", ["slot_rect_side_plane"], "Side-wall faces; the subtype may also contain closed ends."),
        ("bottom_face", "direct", "face_set", ["slot_rect_bottom_plane"], "Planar slot bottom."),
        ("closed_end_face", "unsupported_no_distinct_subtype", "face_set", ["slot_rect_side_plane"], "The ontology cannot distinguish a closed end from other side planes."),
    ],
    "slot_u": [
        ("feature_region", "composite", "feature_region", ["slot_u_side_plane", "slot_u_end_cylinder", "slot_u_bottom_plane"], "Complete U-slot inner region."),
        ("complete_slot_inner_wall", "composite", "face_set", ["slot_u_side_plane", "slot_u_end_cylinder", "slot_u_bottom_plane"], "All U-slot inner faces."),
        ("side_wall", "direct", "face_set", ["slot_u_side_plane"], "Straight side-wall faces."),
        ("bottom_face", "direct", "face_set", ["slot_u_bottom_plane"], "Planar slot bottom."),
        ("closed_end_face", "direct", "face_set", ["slot_u_end_cylinder"], "Rounded closed end wall."),
    ],
    "slot_v": [
        ("feature_region", "composite", "feature_region", ["slot_v_side_plane", "slot_v_bottom_plane"], "Complete ontology-visible V-slot region."),
        ("complete_slot_inner_wall", "composite", "face_set", ["slot_v_side_plane", "slot_v_bottom_plane"], "All V-slot inner faces."),
        ("side_wall", "direct", "face_set", ["slot_v_side_plane"], "Inclined V-slot side faces."),
        ("bottom_or_root_face", "direct", "face_set", ["slot_v_bottom_plane"], "Bottom/root face when exported as a plane."),
        ("closed_end_face", "unsupported_no_distinct_subtype", "face_set", ["slot_v_side_plane"], "No distinct V-slot closed-end subtype exists."),
    ],
    "slot_o": [
        ("feature_region", "composite", "feature_region", ["slot_o_side_plane", "slot_o_bottom_plane"], "Complete closed O-slot region."),
        ("complete_slot_inner_wall", "composite", "face_set", ["slot_o_side_plane", "slot_o_bottom_plane"], "All ontology-visible O-slot inner faces."),
        ("side_wall", "direct", "face_set", ["slot_o_side_plane"], "O-slot side-wall subtype."),
        ("bottom_face", "direct", "face_set", ["slot_o_bottom_plane"], "O-slot bottom face."),
    ],
    "pocket_rect": [
        ("feature_region", "composite", "feature_region", ["rect_pocket_side_plane", "rect_pocket_bottom_plane"], "Complete rectangular pocket cavity region."),
        ("side_wall", "direct", "face_set", ["rect_pocket_side_plane"], "Rectangular pocket side walls."),
        ("bottom_face", "direct", "face_set", ["rect_pocket_bottom_plane"], "Rectangular pocket bottom."),
    ],
    "pocket_circular": [
        ("feature_region", "composite", "feature_region", ["circular_pocket_side_cylinder", "circular_pocket_bottom_plane"], "Complete circular pocket cavity region."),
        ("side_wall", "direct", "face_set", ["circular_pocket_side_cylinder"], "Circular pocket cylindrical wall."),
        ("bottom_face", "direct", "face_set", ["circular_pocket_bottom_plane"], "Circular pocket bottom."),
    ],
    "cylinder_boss": [
        ("feature_region", "composite", "feature_region", ["cylinder_boss_side", "cylinder_boss_top"], "Complete cylindrical boss region."),
        ("side_wall", "direct", "face_set", ["cylinder_boss_side"], "Cylindrical boss side wall."),
        ("top_face", "direct", "face_set", ["cylinder_boss_top"], "Circular boss top face."),
    ],
    "rectangular_boss": [
        ("feature_region", "composite", "feature_region", ["pad_side_plane", "pad_top_plane"], "Complete rectangular boss region."),
        ("side_wall", "direct", "face_set", ["pad_side_plane"], "Rectangular boss side walls."),
        ("top_face", "direct", "face_set", ["pad_top_plane"], "Rectangular boss top face."),
    ],
    "rib": [
        ("feature_region", "composite", "feature_region", ["rib_side_plane", "rib_top_plane", "web_side_plane", "web_top_plane"], "Complete rib region; legacy web subtypes are accepted."),
        ("side_face", "composite", "face_set", ["rib_side_plane", "web_side_plane"], "Rib side faces including legacy web labeling."),
        ("top_or_end_face", "composite", "face_set", ["rib_top_plane", "web_top_plane"], "Top/end faces are not distinguished by the ontology."),
        ("end_face", "unsupported_no_distinct_subtype", "face_set", ["rib_top_plane", "web_top_plane"], "No subtype isolates rib end faces from top faces."),
    ],
    "fillet": [
        ("feature_region", "direct", "feature_region", ["fillet_blend_surface"], "Complete fillet region."),
        ("blend_face", "direct", "face_set", ["fillet_blend_surface"], "Fillet blend face."),
    ],
    "chamfer": [
        ("feature_region", "direct", "feature_region", ["chamfer_bevel_plane"], "Complete chamfer region."),
        ("bevel_face", "direct", "face_set", ["chamfer_bevel_plane"], "Chamfer bevel face."),
    ],
}


def build_policy() -> dict[str, Any]:
    features: dict[str, Any] = {}
    for canonical, (family, aliases) in FEATURE_META.items():
        roles = []
        all_subtypes: list[str] = []
        unsupported: list[str] = []
        for role, status, region_kind, subtypes, note in ROLE_SPECS[canonical]:
            for subtype in subtypes:
                if subtype not in all_subtypes:
                    all_subtypes.append(subtype)
            allowed = list(SURFACE_OPERATIONS) if status in {"direct", "composite"} else []
            if not allowed:
                unsupported.append(f"{role}: {note}")
            roles.append({
                "role": role,
                "selection_status": status,
                "semantic_subtype_candidates": subtypes,
                "region_kind": region_kind,
                "allowed_operations": allowed,
                "requires_complete_instance": region_kind == "feature_region",
                "selection_notes": note,
            })
        family_notes = [
            "Policy expresses compatibility only; it does not infer the user's load case.",
            "material_assignment and global_mesh_size use whole_part scope, not a feature role.",
        ]
        if canonical in {"slot_rect", "slot_u", "slot_v", "slot_o", "rib"}:
            family_notes.append("Predicted geometry requires abstention or human confirmation under the frozen capability contract.")
        if canonical == "pocket_circular":
            family_notes.append("Class is absent from the frozen capability contract; predicted coverage is not claimed.")
        if canonical in {"fillet", "chamfer"}:
            family_notes.append("Frozen transition-family recognition is limited and must preserve coverage status.")
        features[canonical] = {
            "family": family,
            "canonical_name": canonical,
            "aliases": aliases,
            "selectable_roles": roles,
            "semantic_subtype_candidates": all_subtypes,
            "region_kind": "feature_or_face_region",
            "allowed_operations": list(SURFACE_OPERATIONS),
            "requires_complete_instance": True,
            "may_include_transition": canonical in {"fillet", "chamfer"},
            "selection_notes": family_notes,
            "unsupported_or_ambiguous_cases": unsupported,
        }
    return {
        "schema_version": "topocae_cae_feature_role_policy_1",
        "analysis_type": "static_structural",
        "canonical_feature_count": len(features),
        "operation_types": [
            "fixed_support", "displacement", "pressure", "surface_traction",
            "material_assignment", "global_mesh_size", "local_mesh_size",
        ],
        "whole_part_only_operations": ["material_assignment", "global_mesh_size"],
        "policy_principles": [
            "support and load are analysis-case operations, never inherent geometry labels",
            "allowed means geometrically targetable, not automatically physically appropriate",
            "unsupported precision is explicit and never synthesized beyond the semantic ontology",
            "web is accepted only as an alias and canonicalized to rib",
        ],
        "features": features,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(build_policy(), allow_unicode=True, sort_keys=False, width=120),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
