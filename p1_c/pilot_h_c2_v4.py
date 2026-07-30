"""Canonical fail-closed H-C2 protocol after slot-kernel regression."""

from __future__ import annotations

from typing import Any

from p1_c import all_features_h_c2_v3 as generator
from p1_c import pilot_h_c2_v3 as revision3


CANONICAL_SEED_BASE = 2026092000
base_protocol = revision3.revision2.protocol
_ORIGINAL_AUDIT_SAMPLE = getattr(
    base_protocol,
    "_strict_contract_original_audit_sample",
    base_protocol.audit_sample,
)
base_protocol._strict_contract_original_audit_sample = _ORIGINAL_AUDIT_SAMPLE


def audit_sample(
    result: Any,
    labels: list[dict[str, Any]],
    graph: dict[str, Any],
    step_report: dict[str, Any],
) -> dict[str, Any]:
    payload = _ORIGINAL_AUDIT_SAMPLE(result, labels, graph, step_report)
    contract = result.parameters.get("strict_generation_contract_audit", {})
    payload["checks"]["strict_generation_contract"] = contract.get("passed") is True
    payload["checks"]["canonical_generation_entrypoint"] = (
        result.parameters.get("canonical_generation_entrypoint")
        == generator.CANONICAL_ENTRYPOINT
    )
    payload["checks"]["authoritative_slot_kernel_revision"] = (
        result.parameters.get("authoritative_slot_kernel_revision")
        == generator.SLOT_KERNEL_REVISION
    )
    payload["strict_generation_contract"] = contract
    payload["passed"] = all(payload["checks"].values())
    return payload


base_protocol.GENERATOR_REVISION = generator.GENERATOR_REVISION
base_protocol.build_feature_sample = generator.build_feature_sample
base_protocol.CANONICAL_SEED_BASE = CANONICAL_SEED_BASE
base_protocol.audit_sample = audit_sample

generate_h_c2 = base_protocol.generate_h_c2
verify_h_c1 = revision3.verify_h_c1


__all__ = ["generate_h_c2", "verify_h_c1"]
