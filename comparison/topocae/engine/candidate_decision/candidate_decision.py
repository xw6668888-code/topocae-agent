from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

from .common import TARGET_CLASSES


SCHEMA_VERSION = "topocae_p1_e6r_candidate_macro_decision_1"


@dataclass(frozen=True)
class CandidateMacroDecision:
    schema_version: str
    status: str
    selected_macro: str | None
    allowed_classes: list[str]
    face_indices: list[int]
    face_vote_counts: dict[str, int]
    mean_class_probabilities: dict[str, float | None]
    topology_evidence: Any
    topology_vetoed_classes: list[str]
    final_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_candidate_macro(
    proposal: Mapping[str, Any],
    face_prediction: Mapping[str, Any],
    *,
    known_classes: Sequence[str] = TARGET_CLASSES,
    topology_vetoed_classes: Iterable[str] = (),
) -> CandidateMacroDecision:
    """Resolve a proposal class from GNN evidence within topology-allowed hints."""

    known = set(str(value) for value in known_classes)
    hinted = [str(value) for value in proposal.get("macro_hints", [])]
    allowed = sorted(set(hinted) & known)
    vetoed = sorted(set(str(value) for value in topology_vetoed_classes) & set(allowed))
    eligible = [name for name in allowed if name not in vetoed]
    faces = sorted(set(int(value) for value in proposal.get("face_indices", [])))
    topology = proposal.get("topology_evidence", {})
    if not allowed:
        return _uncertain(allowed, faces, {}, {}, topology, vetoed, "no_known_allowed_class")
    if not eligible:
        return _uncertain(allowed, faces, {}, {}, topology, vetoed, "all_allowed_classes_topology_vetoed")
    if len(eligible) == 1:
        return CandidateMacroDecision(
            SCHEMA_VERSION,
            "selected",
            eligible[0],
            allowed,
            faces,
            {name: 0 for name in allowed},
            {name: None for name in allowed},
            topology,
            vetoed,
            "single_eligible_allowed_class",
        )

    by_face = {
        int(record.get("idx", position)): record
        for position, record in enumerate(face_prediction.get("faces", []))
    }
    votes: Counter[str] = Counter()
    for face in faces:
        predicted = str(by_face.get(face, {}).get("macro_feature", "unknown"))
        if predicted in eligible:
            votes[predicted] += 1
    vote_counts = {name: int(votes[name]) for name in allowed}
    best_vote = max((votes[name] for name in eligible), default=0)
    vote_winners = [name for name in eligible if votes[name] == best_vote]
    means = {name: _mean_probability(by_face, faces, name, known_classes) for name in allowed}
    if len(vote_winners) == 1:
        return CandidateMacroDecision(
            SCHEMA_VERSION,
            "selected",
            vote_winners[0],
            allowed,
            faces,
            vote_counts,
            means,
            topology,
            vetoed,
            "unique_face_vote_majority",
        )

    scored = [(means[name], name) for name in vote_winners if means[name] is not None]
    if scored:
        best_score = max(float(score) for score, _ in scored)
        winners = [name for score, name in scored if math.isclose(float(score), best_score, rel_tol=0.0, abs_tol=1e-12)]
        if len(winners) == 1:
            return CandidateMacroDecision(
                SCHEMA_VERSION,
                "selected",
                winners[0],
                allowed,
                faces,
                vote_counts,
                means,
                topology,
                vetoed,
                "face_vote_tie_broken_by_mean_class_probability",
            )
    return _uncertain(
        allowed,
        faces,
        vote_counts,
        means,
        topology,
        vetoed,
        "unresolved_face_vote_and_probability_tie",
    )


def _mean_probability(
    by_face: Mapping[int, Mapping[str, Any]],
    faces: Sequence[int],
    class_name: str,
    known_classes: Sequence[str],
) -> float | None:
    values: list[float] = []
    for face in faces:
        record = by_face.get(face, {})
        raw = record.get("macro_probabilities")
        value: Any = None
        if isinstance(raw, Mapping):
            value = raw.get(class_name)
        elif isinstance(raw, (list, tuple)):
            try:
                value = raw[list(known_classes).index(class_name)]
            except (ValueError, IndexError):
                value = None
        if value is None:
            continue
        number = float(value)
        if math.isfinite(number):
            values.append(number)
    return sum(values) / len(values) if values else None


def _uncertain(
    allowed: list[str],
    faces: list[int],
    votes: Mapping[str, int],
    means: Mapping[str, float | None],
    topology: Any,
    vetoed: list[str],
    reason: str,
) -> CandidateMacroDecision:
    return CandidateMacroDecision(
        SCHEMA_VERSION,
        "uncertain",
        None,
        allowed,
        faces,
        dict(votes),
        dict(means),
        topology,
        vetoed,
        reason,
    )


__all__ = ["CandidateMacroDecision", "SCHEMA_VERSION", "resolve_candidate_macro"]
