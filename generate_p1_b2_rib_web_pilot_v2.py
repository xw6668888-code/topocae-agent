from __future__ import annotations

from typing import Any

import generate_p1_b2_rib_web_pilot as protocol
from features.p1_b_rib_web_v3 import P1BRibWebResult


_base_audit_sample = protocol.audit_sample


def ontology_canonical_audit_sample(
    result: P1BRibWebResult,
    *,
    step_report: dict[str, Any],
    graph: dict[str, Any],
    labels: list[dict[str, Any]],
) -> dict[str, Any]:
    audit = _base_audit_sample(result, step_report=step_report, graph=graph, labels=labels)
    params = result.parameters
    hole_labels = [label for label in labels if label["macro_feature"] == "through_hole_round"]
    hole_expected = params["variant"] == "hole"
    feature_inst = params["feature_instance_id"]
    audit["hole_face_count"] = len(hole_labels)
    audit["checks"]["hole_instance_independent"] = (not hole_expected) or (
        len(hole_labels) == params["hole_face_count"]
        and bool(hole_labels)
        and {label["instance_id"] for label in hole_labels} == {feature_inst + 1}
    )
    audit["passed"] = all(audit["checks"].values())
    audit["ontology_alias_resolution"] = {"through_hole": "through_hole_round"}
    return audit


def main() -> int:
    protocol.audit_sample = ontology_canonical_audit_sample
    return protocol.main()


if __name__ == "__main__":
    raise SystemExit(main())
