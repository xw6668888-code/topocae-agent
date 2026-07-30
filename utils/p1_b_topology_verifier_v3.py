"""P1-B verifier v3: retain parent-bridged multi-surface memberships."""

from __future__ import annotations

from typing import Any, Mapping

import utils.p1_b_topology_verifier_v2 as revision_2
import utils.p1_b_topology_verifier as base


_BASE_MEMBERSHIP_COMPONENT = base._best_connected_membership_component


def _best_membership_component_v3(
    graph: Mapping[str, Any],
    refined_faces: list[int],
    original_faces: list[int],
    probabilities: list[float],
) -> list[int]:
    if revision_2._components_share_boundary_parent(graph, set(refined_faces)):
        return sorted(set(refined_faces))
    return _BASE_MEMBERSHIP_COMPONENT(graph, refined_faces, original_faces, probabilities)


base._best_connected_membership_component = _best_membership_component_v3

SCHEMA_VERSION = "topocae_p1_b_topology_verifier_3"
select_global_candidate_thresholds = base.select_global_candidate_thresholds
verify_candidate_prediction = base.verify_candidate_prediction


__all__ = ["SCHEMA_VERSION", "select_global_candidate_thresholds", "verify_candidate_prediction"]
