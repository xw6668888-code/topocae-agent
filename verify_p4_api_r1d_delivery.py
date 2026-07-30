"""Independent verifier for the P4-API-R1D offline delivery."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re

from p4_api_r0_1_audit import verify_full_freeze


ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "reports" / "p4_api_r1d_strict_schema_feasibility_v1"
LEDGER = ROOT / "output_p4_a1_language_grounding_v1" / "test_run_ledger.json"
EXPECTED_FILES = {
    "acceptance.json",
    "CODEX_CONTEXT_P4_API_R1D.json",
    "delivery_manifest.json",
    "final_acceptance.zh-CN.md",
    "migration_plan.zh-CN.md",
    "preflight.json",
    "provider_compatibility.json",
    "schema_inventory.json",
    "security_audit.json",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    actual_files = {path.name for path in REPORT.iterdir() if path.is_file()}
    assert actual_files == EXPECTED_FILES, (actual_files, EXPECTED_FILES)

    manifest = json.loads((REPORT / "delivery_manifest.json").read_text(encoding="utf-8"))
    assets = manifest["assets"]
    assert len(assets) == 8
    assert {asset["path"] for asset in assets} == EXPECTED_FILES - {"delivery_manifest.json"}
    for asset in assets:
        assert digest(REPORT / asset["path"]) == asset["sha256"]

    acceptance = json.loads((REPORT / "acceptance.json").read_text(encoding="utf-8"))
    compatibility = json.loads((REPORT / "provider_compatibility.json").read_text(encoding="utf-8"))
    preflight = json.loads((REPORT / "preflight.json").read_text(encoding="utf-8"))
    security = json.loads((REPORT / "security_audit.json").read_text(encoding="utf-8"))
    inventory = json.loads((REPORT / "schema_inventory.json").read_text(encoding="utf-8"))

    assert acceptance["status"] == "NO_COMPATIBLE_STRICT_PROFILE_OFFLINE_GATE"
    assert acceptance["compatible_candidate_count"] == 0
    assert acceptance["real_api_smoke_performed"] is False
    assert acceptance["backend_requests"] == 0
    assert acceptance["cost_cny"] == 0.0
    assert compatibility["compatible_candidate_count"] == 0
    assert len(compatibility["evaluated_candidates"]) == 3
    assert compatibility["real_api_smoke_authorized_by_gate"] is False
    assert compatibility["backend_requests_performed"] == 0
    assert preflight["formal_test_before"] == {
        "run_count": 0,
        "content_accessed": False,
        "unlock_authorized": False,
    }
    assert verify_full_freeze()["status"] == "PASS"

    expected_hashes = {
        "schemas/cae_intent_result.schema.json": "00d5da9125031158786bd853c172f0aae1fb63c0424fd19b3ff82f6918e37727",
        "schemas/cae_intent.schema.json": "8e94cd9a6f3dad42ccad58d47eb82e801e2cb66870ac03dcd940a56d076aeac4",
    }
    delivered_hashes = {
        row["path"]: row["sha256"] for row in inventory["effective_schema_files"]
    }
    assert delivered_hashes == expected_hashes

    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    assert sum(len(rows) for rows in ledger["runs"].values()) == 0
    assert ledger["test_content_accessed"] is False
    assert ledger["test_unlock_authorized"] is False

    delivery_text = "\n".join(
        path.read_text(encoding="utf-8") for path in REPORT.iterdir() if path.is_file()
    )
    api_key = os.environ.get("TOPOCAE_LLM_API_KEY", "")
    assert not api_key or api_key not in delivery_text
    assert re.search(r"(?i)Bearer\s+[A-Za-z0-9._-]{8,}", delivery_text) is None
    assert security["api_key_present_in_delivery"] is False
    assert security["bearer_value_hits"] == 0
    assert security["query_written"] is False
    assert security["ground_truth_written"] is False
    assert security["raw_model_response_written"] is False

    print(json.dumps({
        "status": "PASS",
        "files": len(actual_files),
        "manifest_assets": len(assets),
        "candidate_profiles": len(compatibility["evaluated_candidates"]),
        "compatible_profiles": 0,
        "backend_requests": 0,
        "cost_cny": 0.0,
    }))


if __name__ == "__main__":
    main()
