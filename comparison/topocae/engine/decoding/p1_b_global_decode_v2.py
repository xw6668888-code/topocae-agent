"""Global decode v2 with globally selected near-tie subset dominance."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from utils.p1_b_global_decode import decode_feature_objects as _decode_v1


SCHEMA_VERSION = "topocae_p1_b_global_decode_2"


def decode_feature_objects(
    graph_data: Mapping[str, Any],
    verified_candidates: Iterable[Mapping[str, Any]],
    *,
    subset_confidence_tolerance: float,
    selection_sha256: str,
) -> dict[str, Any]:
    tolerance = float(subset_confidence_tolerance)
    if tolerance < 0.0 or tolerance > 0.2:
        raise ValueError("subset_confidence_tolerance must be globally bounded inside [0, 0.2]")
    if not selection_sha256:
        raise ValueError("a Synthetic-Val global-decode selection hash is required")
    candidates = [dict(record) for record in verified_candidates]
    accepted = [record for record in candidates if record.get("status") == "accepted"]
    dominated: list[dict[str, Any]] = []
    for candidate in accepted:
        candidate_faces = set(int(index) for index in candidate.get("refined_face_indices", []))
        candidate_macro = str(candidate.get("predicted_macro", "unknown"))
        candidate_score = float(candidate.get("final_confidence", 0.0) or 0.0)
        dominators = []
        for other in accepted:
            if other is candidate or str(other.get("predicted_macro", "unknown")) != candidate_macro:
                continue
            other_faces = set(int(index) for index in other.get("refined_face_indices", []))
            other_score = float(other.get("final_confidence", 0.0) or 0.0)
            if candidate_faces < other_faces and other_score + tolerance >= candidate_score:
                dominators.append({
                    "candidate_id": other.get("candidate_id"),
                    "face_indices": sorted(other_faces),
                    "confidence": other_score,
                })
        if dominators:
            candidate["status"] = "abstain"
            candidate["abstain"] = True
            candidate["predicted_macro"] = "unknown"
            candidate["abstention_reasons"] = sorted(set(candidate.get("abstention_reasons", [])) | {"near_tie_same_macro_strict_subset_dominated"})
            candidate["subset_dominators"] = dominators
            dominated.append(candidate)
    decoded = _decode_v1(graph_data, candidates)
    decoded["schema_version"] = SCHEMA_VERSION
    decoded["subset_confidence_tolerance"] = tolerance
    decoded["global_decode_selection_sha256"] = selection_sha256
    decoded["subset_dominated_count"] = len(dominated)
    decoded["global_decode_v2_contract"] = {
        "selection_split": "synthetic_val",
        "category_specific_tolerance": False,
        "part_specific_tolerance": False,
        "truth_used_at_inference": False,
        "class_rewrite": False,
        "only_strict_subset_same_macro_duplicates_affected": True,
    }
    return decoded


__all__ = ["SCHEMA_VERSION", "decode_feature_objects"]
