from __future__ import annotations
import re
_LABEL = re.compile(r"^[A-Za-z][A-Za-z0-9_+-]{0,31}$")
_STAGE_SOURCES = {"initial": "initial_group", "geometric": "raw_geometry", "pre_validation": "bounded_geometry", "pre_boundary": "pre_boundary"}
from collections import Counter
from typing import Any, Mapping, Sequence

STAGES = (
    "initial",
    "geometric",
    "pre_validation",
    "post_validation",
    "global_selection",
    "pre_boundary",
    "post_boundary",
    "final",
)


def _faces(raw: Mapping[str, Any], face_count: int | None) -> tuple[int, ...]:
    supplied = raw.get("face_indices", ())
    if isinstance(supplied, (str, bytes)) or not isinstance(supplied, (list, tuple, set, frozenset)):
        raise ValueError("candidate face_indices must be a collection")
    if any(isinstance(face, bool) or not isinstance(face, int) for face in supplied):
        raise ValueError("candidate face indices must be integers")
    faces = tuple(sorted(set(supplied)))
    if not faces or faces[0] < 0 or (face_count is not None and faces[-1] >= face_count):
        raise ValueError("candidate has empty or out-of-range face set")
    return faces


def _mechanisms(raw: Mapping[str, Any]) -> tuple[str, ...]:
    supplied = raw.get("proposal_sources", ())
    if not isinstance(supplied, (list, tuple)) or any(
        not isinstance(source, str) or not _LABEL.fullmatch(source) for source in supplied
    ):
        raise ValueError("proposal_sources must be short mechanism labels")
    return tuple(sorted(set(supplied)))


def build_candidate_pool(
    branch_stages: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    *,
    face_count: int | None = None,
) -> list[dict[str, Any]]:
    """Deduplicate by predicted class and complete face set across branches/stages.

    No graph, sample identifier, path, truth, candidate ID, or connectivity is
    consulted. Additional fields on input objects are intentionally ignored.
    """
    if face_count is not None and (isinstance(face_count, bool) or not isinstance(face_count, int) or face_count < 0):
        raise ValueError("face_count must be a nonnegative integer")
    entries: dict[tuple[str, tuple[int, ...]], dict[tuple[str, str], set[str]]] = {}
    for branch, stages in branch_stages.items():
        if not isinstance(branch, str) or not _LABEL.fullmatch(branch):
            raise ValueError("branch must be a short model label, not a path or sample name")
        if set(stages) != set(STAGES):
            raise ValueError("all eight candidate stages are required")
        for stage in STAGES:
            for obj in stages[stage]:
                macro = obj.get("macro_feature")
                if not isinstance(macro, str) or not _LABEL.fullmatch(macro):
                    raise ValueError("candidate macro_feature must be a class label")
                key = macro, _faces(obj, face_count)
                entries.setdefault(key, {}).setdefault((branch, stage), set()).update(_mechanisms(obj))

    result = []
    for (macro, faces), origins in sorted(entries.items()):
        provenance = [
            {"branch": branch, "stage": stage, "mechanisms": sorted(origins[(branch, stage)])}
            for branch, stage in sorted(
                origins, key=lambda origin: (STAGES.index(origin[1]), origin[0])
            )
        ]
        sources = sorted({f"{origin['branch']}:{origin['stage']}" for origin in provenance} | {
            f"{origin['branch']}:{origin['stage']}:{mechanism}"
            for origin in provenance for mechanism in origin["mechanisms"]
        } | {_STAGE_SOURCES[origin["stage"]] for origin in provenance if origin["stage"] in _STAGE_SOURCES}
            | {"old_full" for origin in provenance if origin["branch"] == "R" and origin["stage"] == "final"}
            | {"sparse_branch" for origin in provenance if origin["branch"] == "R"}
            | {"dense_branch" for origin in provenance if origin["branch"] == "Q"})
        result.append({"macro_feature": macro, "face_indices": list(faces),
                       "sources": sources, "provenance": provenance})
    return result


def _canonical(objects: Sequence[Mapping[str, Any]]) -> Counter[tuple[str, tuple[int, ...]]]:
    return Counter((str(obj["macro_feature"]), tuple(sorted(set(obj["face_indices"])))) for obj in objects)


def _classify_proposals(proposals, face_prediction, resolve_macro, target_classes):
    objects = []
    for proposal in proposals:
        decision = resolve_macro(proposal, face_prediction)
        if decision.status != "selected" or decision.selected_macro not in target_classes:
            continue
        faces = tuple(sorted(set(int(face) for face in proposal.get("face_indices", ()))))
        if faces:
            objects.append({"macro_feature": str(decision.selected_macro),
                            "face_indices": list(faces),
                            "proposal_sources": list(proposal.get("proposal_sources", ()))})
    return objects


def legacy_stage_objects(
    recovery: Mapping[str, Any], raw_proposals: Sequence[Mapping[str, Any]],
    prediction: Mapping[str, Any], resolve_macro, target_classes: Sequence[str],
    *, final_objects: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Map the actual G03 trace to eight separate pre-selection candidate stages.

    ``global_selection`` is the decoded result, and ``pre_boundary`` is its
    input to boundary splitting; both origins are retained even when identical.
    ``candidate_boundary`` is not a geometric-proposal stage and is excluded.
    """
    objects = recovery["objects"]
    boundary = recovery["trace"]["proposal_boundary"]["proposals"]
    face_prediction = {"faces": prediction["faces"]}
    geometric = _classify_proposals(raw_proposals, face_prediction, resolve_macro, target_classes)
    before_validation = _classify_proposals(boundary, face_prediction, resolve_macro, target_classes)
    if _canonical(before_validation) != _canonical(objects["proposals"]):
        raise ValueError("G03 proposal trace disagrees with the pre-validation stage")

    stages = {
        "initial": objects["candidate"],
        "geometric": geometric,
        "pre_validation": before_validation,
        "post_validation": objects["verified"],
        "global_selection": objects["decoded"],
        "pre_boundary": objects["decoded"],
        "post_boundary": objects["full"],
        "final": objects["full"] if final_objects is None else final_objects,
    }
    # Copy only prediction fields. Legacy sample_id and optional evidence IDs
    # cannot become a model feature, a pool identity, or a selector tiebreaker.
    return {stage: [
        {"macro_feature": str(obj["macro_feature"]), "face_indices": list(obj["face_indices"]),
         **({"proposal_sources": list(obj["proposal_sources"])} if "proposal_sources" in obj else {})}
        for obj in stages[stage]
    ] for stage in STAGES}
