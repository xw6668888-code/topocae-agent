from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from . import validation as base
from . import formal_validation as v1


def _payload(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _p2_worker(payload: str) -> tuple[str, bool, list[Any]]:
    value = json.loads(payload)
    result = base.p2_api().validate_intent_result(value)
    return payload, bool(result["valid"]), result["errors"]


def install_parallel_p2_proof(
    canonical_records: list[dict[str, Any]], rendering_records: list[dict[str, Any]]
) -> dict[str, Any]:
    payloads = sorted({
        _payload(record["intent_result"])
        for record in canonical_records + rendering_records
    })
    workers = min(12, max(1, os.cpu_count() or 1))
    failures: list[dict[str, Any]] = []
    passed: set[str] = set()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for payload, valid, errors in executor.map(_p2_worker, payloads, chunksize=32):
            if valid:
                passed.add(payload)
            elif len(failures) < 100:
                failures.append({"payload_sha256": v1.canonical_sha256(json.loads(payload)), "errors": errors})
    if len(passed) != len(payloads):
        raise RuntimeError(
            f"Frozen P2 validation failed for {len(payloads) - len(passed)}/{len(payloads)} unique payloads: {failures[:3]}"
        )

    def lookup(value: Any) -> dict[str, Any]:
        payload = _payload(value)
        return {"valid": payload in passed, "errors": [] if payload in passed else ["not in P2 proof set"]}

    v1._p2 = lookup
    return {
        "unique_intent_result_payloads": len(payloads),
        "passed": len(passed),
        "failed": 0,
        "workers": workers,
        "canonical_records_covered": len(canonical_records),
        "rendering_records_covered": len(rendering_records),
        "proof": "Every distinct serialized intent_result from canonical and rendering records passed the frozen P2 validate_intent_result API.",
    }


def audit_formal(
    plans: list[dict[str, Any]],
    canonical_records: list[dict[str, Any]],
    rendering_records: list[dict[str, Any]],
) -> dict[str, Any]:
    p2_proof = install_parallel_p2_proof(canonical_records, rendering_records)
    audit = v1.audit_formal(plans, canonical_records, rendering_records)
    audit["p2_validation"] = p2_proof
    audit["automatic_pass"] = audit["automatic_pass"] and p2_proof["failed"] == 0
    return audit


validate_formal_manifest = v1.validate_formal_manifest
split_leakage_audit = v1.split_leakage_audit

