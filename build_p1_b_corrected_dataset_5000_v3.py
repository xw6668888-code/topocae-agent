"""Official v3 builder: host-height-aware identity labeling for varied stocks."""

from __future__ import annotations

from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.Extend.TopologyUtils import TopologyExplorer

import build_p1_b_corrected_dataset_5000 as base
import build_p1_b_corrected_dataset_5000_v2 as artifact_revision
import features.p1_b_continuous_slot as slot_core
from features.p1_b_continuous_slot import SlotBuildResult


OUTPUT = base.ROOT / "output_generation_p1_b_corrected_balanced_5000_v3"
base.OUTPUT = OUTPUT
artifact_revision.OUTPUT = OUTPUT


def rebuild_with_host_height_roles(raw_result: SlotBuildResult, stock) -> SlotBuildResult:
    cut = BRepAlgoAPI_Cut(stock, raw_result.tool)
    cut.Build()
    if not cut.IsDone():
        raise ValueError("identity-safe slot cut reconstruction failed")
    shape = cut.Shape()
    parameters = dict(raw_result.parameters)
    host_origin = parameters.get("host_local_frame", {}).get("origin", [0.0, 0.0, 25.0])
    host_surface_height = float(host_origin[2])
    role_items = []
    for tool_face in TopologyExplorer(raw_result.tool).faces():
        role = slot_core._tool_face_role(tool_face, parameters, host_surface_height)
        if role is None:
            continue
        for modified in cut.Modified(tool_face):
            if modified.ShapeType() != 4:
                continue
            if not any(modified.IsSame(existing) for existing, _ in role_items):
                role_items.append((modified, role))
    parameters["_face_role_items"] = role_items
    parameters["_face_role_by_hash"] = {hash(face): role for face, role in role_items}
    parameters["_feature_face_hashes"] = sorted({hash(face) for face, _ in role_items})
    parameters["feature_face_count"] = len(role_items)
    parameters["terminal_face_count"] = sum(role == slot_core.TERMINAL_ROLE for _, role in role_items)
    parameters["identity_role_surface_height"] = host_surface_height
    if not role_items:
        raise ValueError("host-height-aware slot labeling produced no feature faces")
    return SlotBuildResult(shape, raw_result.tool, raw_result.guide_wire, raw_result.profile_wire, parameters)


base.rebuild_with_identity_roles = rebuild_with_host_height_roles


if __name__ == "__main__":
    raise SystemExit(artifact_revision.main())
