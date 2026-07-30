from __future__ import annotations

from collections import defaultdict
from typing import Any


RegionRule = dict[str, Any]


SELECTABLE_REGION_RULES: dict[str, list[RegionRule]] = {
    "through_hole_round": [
        {
            "role": "hole_cylindrical_wall",
            "semantic_subtypes": ("through_hole_cylinder",),
            "intended_uses": ("fixed_support", "pressure_load", "coupling", "mesh_refinement"),
            "is_default_for": ("hole_constraint",),
        },
    ],
    "blind_hole_round": [
        {
            "role": "hole_cylindrical_wall",
            "semantic_subtypes": ("blind_hole_cylinder",),
            "intended_uses": ("fixed_support", "pressure_load", "coupling", "mesh_refinement"),
            "is_default_for": ("hole_constraint",),
        },
        {
            "role": "hole_bottom_face",
            "semantic_subtypes": ("blind_hole_bottom",),
            "intended_uses": ("explicit_bottom_selection", "pressure_load", "mesh_refinement"),
            "is_default_for": (),
        },
    ],
    "stepped_hole_round": [
        {
            "role": "large_hole_cylindrical_wall",
            "semantic_subtypes": ("stepped_hole_large_hole_cylinder",),
            "intended_uses": ("fixed_support", "pressure_load", "coupling", "mesh_refinement"),
            "is_default_for": ("hole_constraint",),
        },
        {
            "role": "step_shoulder_face",
            "semantic_subtypes": ("stepped_hole_step_plane",),
            "intended_uses": ("fixed_support", "explicit_shoulder_selection", "pressure_load", "mesh_refinement"),
            "is_default_for": (),
        },
        {
            "role": "small_hole_cylindrical_wall",
            "semantic_subtypes": ("stepped_hole_small_hole_cylinder",),
            "intended_uses": ("fixed_support", "pressure_load", "coupling", "mesh_refinement"),
            "is_default_for": ("hole_constraint",),
        },
        {
            "role": "hole_bottom_face",
            "semantic_subtypes": ("stepped_hole_small_hole_bottom",),
            "intended_uses": ("explicit_bottom_selection", "pressure_load", "mesh_refinement"),
            "is_default_for": (),
        },
    ],
    "counterbore_hole": [
        {
            "role": "counterbore_large_cylindrical_wall",
            "semantic_subtypes": ("counterbore_large_cylinder",),
            "intended_uses": ("fixed_support", "pressure_load", "coupling", "mesh_refinement"),
            "is_default_for": ("hole_constraint",),
        },
        {
            "role": "counterbore_shoulder_face",
            "semantic_subtypes": ("counterbore_step_plane",),
            "intended_uses": ("fixed_support", "explicit_shoulder_selection", "pressure_load", "mesh_refinement"),
            "is_default_for": (),
        },
        {
            "role": "counterbore_small_cylindrical_wall",
            "semantic_subtypes": ("counterbore_small_cylinder",),
            "intended_uses": ("fixed_support", "pressure_load", "coupling", "mesh_refinement"),
            "is_default_for": ("hole_constraint",),
        },
    ],
    "countersink_hole": [
        {
            "role": "countersink_conical_face",
            "semantic_subtypes": ("countersink_cone",),
            "intended_uses": ("pressure_load", "explicit_conical_face_selection", "mesh_refinement"),
            "is_default_for": (),
        },
        {
            "role": "countersink_small_cylindrical_wall",
            "semantic_subtypes": ("countersink_small_cylinder",),
            "intended_uses": ("fixed_support", "pressure_load", "coupling", "mesh_refinement"),
            "is_default_for": ("hole_constraint",),
        },
    ],
    "obround_hole": [
        {
            "role": "obround_hole_wall_faces",
            "semantic_subtypes": ("obround_hole_side_plane", "obround_hole_end_cylinder"),
            "intended_uses": ("fixed_support", "pressure_load", "coupling", "mesh_refinement"),
            "is_default_for": ("hole_constraint",),
        },
        {
            "role": "obround_side_faces",
            "semantic_subtypes": ("obround_hole_side_plane",),
            "intended_uses": ("explicit_side_selection", "mesh_refinement"),
            "is_default_for": (),
        },
        {
            "role": "obround_end_cylinders",
            "semantic_subtypes": ("obround_hole_end_cylinder",),
            "intended_uses": ("explicit_end_selection", "mesh_refinement"),
            "is_default_for": (),
        },
    ],
    "slot_rect": [
        {
            "role": "slot_side_faces",
            "semantic_subtypes": ("slot_rect_side_plane",),
            "intended_uses": ("fixed_support", "pressure_load", "mesh_refinement"),
            "is_default_for": ("slot_side_selection",),
        },
        {
            "role": "slot_bottom_face",
            "semantic_subtypes": ("slot_rect_bottom_plane",),
            "intended_uses": ("pressure_load", "explicit_bottom_selection", "mesh_refinement"),
            "is_default_for": (),
        },
    ],
    "slot_u": [
        {
            "role": "slot_side_faces",
            "semantic_subtypes": ("slot_u_side_plane",),
            "intended_uses": ("fixed_support", "pressure_load", "mesh_refinement"),
            "is_default_for": ("slot_side_selection",),
        },
        {
            "role": "slot_end_cylinders",
            "semantic_subtypes": ("slot_u_end_cylinder",),
            "intended_uses": ("pressure_load", "explicit_end_selection", "mesh_refinement"),
            "is_default_for": (),
        },
        {
            "role": "slot_bottom_face",
            "semantic_subtypes": ("slot_u_bottom_plane",),
            "intended_uses": ("pressure_load", "explicit_bottom_selection", "mesh_refinement"),
            "is_default_for": (),
        },
    ],
    "slot_v": [
        {
            "role": "slot_side_faces",
            "semantic_subtypes": ("slot_v_side_plane",),
            "intended_uses": ("fixed_support", "pressure_load", "mesh_refinement"),
            "is_default_for": ("slot_side_selection",),
        },
        {
            "role": "slot_bottom_face",
            "semantic_subtypes": ("slot_v_bottom_plane",),
            "intended_uses": ("pressure_load", "explicit_bottom_selection", "mesh_refinement"),
            "is_default_for": (),
        },
    ],
    "slot_dovetail": [
        {
            "role": "slot_side_faces",
            "semantic_subtypes": ("slot_dovetail_side_plane",),
            "intended_uses": ("fixed_support", "pressure_load", "mesh_refinement"),
            "is_default_for": ("slot_side_selection",),
        },
        {
            "role": "slot_bottom_face",
            "semantic_subtypes": ("slot_dovetail_bottom_plane",),
            "intended_uses": ("pressure_load", "explicit_bottom_selection", "mesh_refinement"),
            "is_default_for": (),
        },
    ],
    "slot_o": [
        {
            "role": "slot_side_faces",
            "semantic_subtypes": ("slot_o_side_plane",),
            "intended_uses": ("fixed_support", "pressure_load", "mesh_refinement"),
            "is_default_for": ("slot_side_selection",),
        },
        {
            "role": "slot_bottom_face",
            "semantic_subtypes": ("slot_o_bottom_plane",),
            "intended_uses": ("pressure_load", "explicit_bottom_selection", "mesh_refinement"),
            "is_default_for": (),
        },
    ],
    "pocket_rect": [
        {
            "role": "pocket_side_faces",
            "semantic_subtypes": ("rect_pocket_side_plane",),
            "intended_uses": ("fixed_support", "pressure_load", "mesh_refinement"),
            "is_default_for": ("pocket_side_selection",),
        },
        {
            "role": "pocket_bottom_face",
            "semantic_subtypes": ("rect_pocket_bottom_plane",),
            "intended_uses": ("pressure_load", "explicit_bottom_selection", "mesh_refinement"),
            "is_default_for": (),
        },
    ],
    "pocket_circular": [
        {
            "role": "pocket_cylindrical_wall",
            "semantic_subtypes": ("circular_pocket_side_cylinder",),
            "intended_uses": ("fixed_support", "pressure_load", "mesh_refinement"),
            "is_default_for": ("pocket_side_selection",),
        },
        {
            "role": "pocket_bottom_face",
            "semantic_subtypes": ("circular_pocket_bottom_plane",),
            "intended_uses": ("pressure_load", "explicit_bottom_selection", "mesh_refinement"),
            "is_default_for": (),
        },
    ],
    "cylinder_boss": [
        {
            "role": "boss_side_face",
            "semantic_subtypes": ("cylinder_boss_side",),
            "intended_uses": ("fixed_support", "coupling", "mesh_refinement"),
            "is_default_for": (),
        },
        {
            "role": "boss_top_face",
            "semantic_subtypes": ("cylinder_boss_top",),
            "intended_uses": ("fixed_support", "surface_load", "bearing_load", "mesh_refinement"),
            "is_default_for": ("surface_load",),
        },
    ],
    "rectangular_boss": [
        {
            "role": "boss_side_faces",
            "semantic_subtypes": ("pad_side_plane",),
            "intended_uses": ("fixed_support", "coupling", "mesh_refinement"),
            "is_default_for": (),
        },
        {
            "role": "boss_top_face",
            "semantic_subtypes": ("pad_top_plane",),
            "intended_uses": ("fixed_support", "surface_load", "support_candidate", "mesh_refinement"),
            "is_default_for": ("surface_load", "support_candidate"),
        },
    ],
    "rib": [
        {
            "role": "thin_wall_side_faces",
            "semantic_subtypes": ("rib_side_plane",),
            "intended_uses": ("shell_idealization", "mesh_refinement", "retain_for_stiffness"),
            "is_default_for": ("shell_candidate",),
        },
        {
            "role": "rib_top_face",
            "semantic_subtypes": ("rib_top_plane",),
            "intended_uses": ("reference_face", "mesh_refinement"),
            "is_default_for": (),
        },
    ],
    "web": [
        {
            "role": "thin_wall_side_faces",
            "semantic_subtypes": ("web_side_plane",),
            "intended_uses": ("shell_idealization", "mesh_refinement", "retain_for_stiffness"),
            "is_default_for": ("shell_candidate",),
        },
        {
            "role": "web_top_face",
            "semantic_subtypes": ("web_top_plane",),
            "intended_uses": ("reference_face", "mesh_refinement"),
            "is_default_for": (),
        },
    ],
    "fillet": [
        {
            "role": "transition_faces",
            "semantic_subtypes": ("fillet_blend_surface",),
            "intended_uses": ("mesh_refinement", "defeaturing_candidate"),
            "is_default_for": ("transition_feature",),
        },
    ],
    "chamfer": [
        {
            "role": "transition_faces",
            "semantic_subtypes": ("chamfer_bevel_plane",),
            "intended_uses": ("mesh_refinement", "defeaturing_candidate"),
            "is_default_for": ("transition_feature",),
        },
    ],
}


def build_selectable_regions(macro_feature: str, face_descriptors: list[dict]) -> list[dict]:
    subtype_to_faces = _collect_faces_by_subtype(face_descriptors)
    regions = []

    for priority, rule in enumerate(SELECTABLE_REGION_RULES.get(macro_feature, []), start=1):
        source_subtypes = tuple(rule["semantic_subtypes"])
        face_indices = sorted({
            face_idx
            for subtype in source_subtypes
            for face_idx in subtype_to_faces.get(subtype, [])
        })
        if not face_indices:
            continue

        present_subtypes = sorted(
            subtype for subtype in source_subtypes
            if subtype_to_faces.get(subtype)
        )
        regions.append({
            "role": rule["role"],
            "face_indices": face_indices,
            "source_semantic_subtypes": present_subtypes,
            "intended_uses": list(rule.get("intended_uses", ())),
            "is_default_for": list(rule.get("is_default_for", ())),
            "selection_priority": priority,
        })

    return regions


def _collect_faces_by_subtype(face_descriptors: list[dict]) -> dict[str, list[int]]:
    subtype_to_faces: dict[str, list[int]] = defaultdict(list)
    for descriptor in face_descriptors:
        subtype = descriptor.get("semantic_subtype")
        if not subtype:
            continue
        face_idx = descriptor.get("face_idx", descriptor.get("idx"))
        if face_idx is None:
            continue
        subtype_to_faces[str(subtype)].append(int(face_idx))
    return subtype_to_faces
