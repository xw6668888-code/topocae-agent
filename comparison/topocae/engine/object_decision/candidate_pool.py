"""Truth-free candidate capture and pooling for G09 object selection.

The pool retains proposals; it does not choose objects. Its canonical ordering is
only a reproducible serialization order and must never break selector score ties.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
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
_STAGE_SOURCES = {
    "initial": "initial_group", "geometric": "raw_geometry",
    "pre_validation": "bounded_geometry", "pre_boundary": "pre_boundary",
}
_LABEL = re.compile(r"^[A-Za-z][A-Za-z0-9_+-]{0,31}$")
_REPO = Path(__file__).resolve().parents[3]
_LEGACY = _REPO / "paper_release/topocae_p1_legacy_baseline_v1"
_G03 = _REPO / "tools/aei_paper_goals_v1/g03"


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


def _legacy_dependencies():
    """Load the frozen G03 pipeline only when capture is requested."""
    for path in (_LEGACY, _G03):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    existing = sys.modules.get("common")
    if existing is not None and Path(existing.__file__).resolve() != (_G03 / "common.py").resolve():
        raise RuntimeError("another common module shadows frozen G03; use a separate process")
    path = _G03 / "stages.py"
    spec = importlib.util.spec_from_file_location("g09_frozen_g03_stages", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("frozen G03 stages could not be loaded")
    stages = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stages)
    from p1_e6r.candidate_decision import resolve_candidate_macro
    from p1_e6r.common import TARGET_CLASSES
    from utils.p1_b_candidate_proposer import generate_candidate_proposals

    return SimpleNamespace(stages=stages, resolve_macro=resolve_candidate_macro,
                           target_classes=TARGET_CLASSES, generate=generate_candidate_proposals)


def capture_legacy_stages(
    graph: Mapping[str, Any], prediction: Mapping[str, Any],
    same: Mapping[str, Any], candidate_thresholds: Mapping[str, Any],
    global_decode: Mapping[str, Any], *, recovery: Mapping[str, Any] | None = None,
    final_objects: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Capture R's raw geometric proposals before their base-boundary split.

    The frozen decoder receives a constant anonymous ID; all identity/label
    fields on the supplied graph are removed by G03's ``without_truth``.
    Reusing an existing G03 recovery avoids a second full decode.
    """
    legacy = _legacy_dependencies()
    clean_graph = legacy.stages.without_truth(graph)
    clean_prediction = {
        "faces": prediction["faces"],
        "edge_probabilities": prediction["edge_probabilities"],
        **({"relation_edges": prediction["relation_edges"]} if "relation_edges" in prediction else {}),
        **({"relation_probabilities": prediction["relation_probabilities"]} if "relation_probabilities" in prediction else {}),
    }
    if recovery is None:
        recovery = legacy.stages.recover_stages(
            clean_graph, clean_prediction, same, candidate_thresholds,
            global_decode, "g09_anonymous",
        )
    face_prediction = {
        "faces": prediction["faces"], "node_embeddings": [],
        "edge_same_instance_probabilities": prediction["edge_probabilities"],
        "contracts": {"truth_used": False, "latent_candidate_classifier_enabled": False},
    }
    raw = legacy.generate(clean_graph, face_prediction, recovery["trace"]["assembly"])["proposals"]
    return legacy_stage_objects(recovery, raw, clean_prediction, legacy.resolve_macro,
                                legacy.target_classes, final_objects=final_objects)


__all__ = ["STAGES", "build_candidate_pool", "capture_legacy_stages", "legacy_stage_objects"]
