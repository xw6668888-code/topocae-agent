from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Mapping


def enforce_base_candidate_boundaries(
    proposals: Iterable[Mapping[str, Any]],
    assembly: Mapping[str, Any],
) -> dict[str, Any]:
    """Keep topology proposals inside same-instance components without using truth."""
    bases = [
        {
            "candidate_id": str(item.get("candidate_id", index)),
            "faces": {int(value) for value in item.get("face_indices", [])},
        }
        for index, item in enumerate(assembly.get("candidates", []))
    ]
    output: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for proposal_index, source in enumerate(proposals):
        proposal = deepcopy(dict(source))
        faces = {int(value) for value in proposal.get("face_indices", [])}
        touched = [base for base in bases if faces & base["faces"]]
        if len(touched) <= 1:
            proposal["boundary_decision"] = "accepted_within_single_base_candidate"
            proposal["base_candidate_ids"] = [item["candidate_id"] for item in touched]
            output.append(proposal)
            continue
        split_count = 0
        for base in touched:
            split_faces = sorted(faces & base["faces"])
            if not split_faces:
                continue
            split = deepcopy(proposal)
            split["candidate_id"] = f"{proposal.get('candidate_id', proposal_index)}__base_{base['candidate_id']}"
            split["face_indices"] = split_faces
            split["boundary_decision"] = "split_cross_base_candidates_without_positive_same_instance_evidence"
            split["base_candidate_ids"] = [base["candidate_id"]]
            output.append(split)
            split_count += 1
        decisions.append({
            "proposal_id": str(proposal.get("candidate_id", proposal_index)),
            "action": "split" if split_count else "reject",
            "reason": "crosses_multiple_base_candidates_without_positive_same_instance_evidence",
            "touched_base_candidate_ids": [item["candidate_id"] for item in touched],
            "split_count": split_count,
        })
    return {
        "proposals": output,
        "input_count": len(list(proposals)) if isinstance(proposals, list) else None,
        "output_count": len(output),
        "split_or_reject_decisions": decisions,
        "truth_access_count": 0,
        "part_name_access_count": 0,
        "real_part_specific_threshold_count": 0,
        "face_classification_change_count": 0,
    }


__all__ = ["enforce_base_candidate_boundaries"]

