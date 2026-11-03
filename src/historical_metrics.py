"""Original historical face-equivalence evaluation; applied after response collection."""
from copy import deepcopy
SOURCE_THICKNESS_MM=1e-4

def face_union(groups: list[list[int]]) -> list[int]:
    return sorted({face for group in groups for face in group})


def same_source_parameters(plan: dict, reference: dict, actual_groups: dict,
                           expected_groups: dict) -> tuple[bool, list[dict]]:
    """Auxiliary plan equivalence; never used to turn wrong faces into correct ones."""
    actual = {op["action"]: {k: v for k, v in op.items() if k != "target"}
              for op in plan["operations"]}
    expected = {op["action"]: {k: v for k, v in op.items() if k != "target"}
                for op in reference["operations"]}
    normalized = deepcopy(actual)
    accepted = []
    for action in sorted(set(actual) & set(expected)):
        a, e = actual[action], expected[action]
        same_faces = (action in actual_groups and action in expected_groups
                      and face_union(actual_groups[action]) ==
                      face_union(expected_groups[action]))
        if a.get("role") != e.get("role") and same_faces:
            normalized[action]["role"] = e["role"]
            accepted.append({"action": action, "field": "role",
                             "model": a.get("role"), "reference": e.get("role"),
                             "reason": "identical actual operation face set"})
        if (action == "apply_force" and a.get("basis") != e.get("basis")
                and same_faces and len(actual_groups[action]) == 1
                and len(expected_groups[action]) == 1
                and a["magnitude_N"] == e["magnitude_N"]):
            normalized[action]["basis"] = e["basis"]
            accepted.append({"action": action, "field": "basis",
                             "model": a["basis"], "reference": e["basis"],
                             "reason": "one loading object; equal force magnitude"})
        if (action == "set_thickness"
                and a["old_thickness_mm"] != e["old_thickness_mm"]
                and abs(a["old_thickness_mm"] - e["old_thickness_mm"])
                <= SOURCE_THICKNESS_MM):
            normalized[action]["old_thickness_mm"] = e["old_thickness_mm"]
            accepted.append({"action": action, "field": "old_thickness_mm",
                             "model": a["old_thickness_mm"],
                             "reference": e["old_thickness_mm"],
                             "difference_mm": abs(a["old_thickness_mm"] - e["old_thickness_mm"]),
                             "tolerance_mm": SOURCE_THICKNESS_MM,
                             "reason": "existing source-geometry tolerance; target thickness unchanged"})
    return normalized == expected, accepted

