"""Read-only verifier for the P4-API-R1A bounded smoke delivery."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from p4_api_r0_1_audit import verify_full_freeze


ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "reports" / "p4_api_r1_smoke_v1"
REQUIRED = {
    "preflight.json",
    "runtime_identity.json",
    "smoke_sample_manifest.json",
    "structured_smoke_metrics.json",
    "direct_smoke_metrics.json",
    "cache_audit.json",
    "cost_latency_audit.json",
    "failure_atlas.zh-CN.md",
    "security_audit.json",
    "acceptance.json",
    "final_acceptance.zh-CN.md",
    "delivery_manifest.json",
    "CODEX_CONTEXT_P4_API_R1A.md",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    files = {path.name for path in REPORT.iterdir() if path.is_file()}
    documents = {
        path.name: json.loads(path.read_text(encoding="utf-8"))
        for path in REPORT.glob("*.json")
    }
    manifest = documents["delivery_manifest.json"]
    manifest_matches = []
    for row in manifest["assets"]:
        path = ROOT / row["path"]
        manifest_matches.append(
            path.is_file()
            and path.stat().st_size == row["bytes"]
            and sha256(path) == row["sha256"]
        )

    delivery_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in REPORT.iterdir()
        if path.is_file()
    )
    secret_hits = {
        "bearer_value": len(
            re.findall(
                r"(?i)Bearer\s+[A-Za-z0-9._-]{8,}", delivery_text
            )
        ),
        "key_literal": len(
            re.findall(r"(?i)sk-[A-Za-z0-9_-]{8,}", delivery_text)
        ),
    }
    ledger = json.loads(
        (
            ROOT
            / "output_p4_a1_language_grounding_v1"
            / "test_run_ledger.json"
        ).read_text(encoding="utf-8")
    )
    preflight = documents["preflight.json"]
    current_contract_hashes = {
        path: sha256(ROOT / path)
        for path in preflight["prompt_schema_hashes"]
    }
    cost = documents["cost_latency_audit.json"]
    result = {
        "required_files_exact": files == REQUIRED,
        "delivery_file_count": len(files),
        "json_valid_count": len(documents),
        "manifest_matches": sum(manifest_matches),
        "manifest_assets": len(manifest_matches),
        "secret_pattern_hits": secret_hits,
        "formal_test_run_count": sum(
            len(rows) for rows in ledger["runs"].values()
        ),
        "formal_test_content_accessed": ledger["test_content_accessed"],
        "prompt_schema_hashes_match": (
            current_contract_hashes == preflight["prompt_schema_hashes"]
        ),
        "frozen_assets_status": verify_full_freeze()["status"],
        "selected_count": documents["smoke_sample_manifest.json"][
            "selected_sample_count"
        ],
        "structured_attempted": documents["structured_smoke_metrics.json"][
            "attempted_count"
        ],
        "direct_attempted": documents["direct_smoke_metrics.json"][
            "attempted_count"
        ],
        "backend_request_range": [
            cost["actual_backend_request_count_minimum"],
            cost["actual_backend_request_count_maximum"],
        ],
        "cost_upper_cny": cost["actual_run_cost_upper_cny"],
        "acceptance_status": documents["acceptance.json"]["status"],
    }
    passed = (
        result["required_files_exact"]
        and result["manifest_matches"] == result["manifest_assets"] == 12
        and not any(secret_hits.values())
        and result["formal_test_run_count"] == 0
        and not result["formal_test_content_accessed"]
        and result["prompt_schema_hashes_match"]
        and result["frozen_assets_status"] == "PASS"
        and result["selected_count"] == 50
        and result["structured_attempted"] == 1
        and result["direct_attempted"] == 0
        and result["backend_request_range"] == [3, 7]
        and result["cost_upper_cny"] < 20.0
        and result["acceptance_status"] == "BLOCKED_WITH_EVIDENCE"
    )
    result["status"] = "PASS" if passed else "FAIL"
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
