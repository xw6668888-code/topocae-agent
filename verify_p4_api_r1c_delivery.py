from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from p4_api_r0_1_audit import verify_full_freeze


ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "reports" / "p4_api_r1c_structured_v1"
OUTPUT = ROOT / "output_p4_a1_language_grounding_v1" / "p4_api_r1c_structured_v1"
LEDGER = ROOT / "output_p4_a1_language_grounding_v1" / "test_run_ledger.json"
REQUIRED = {
    "preflight.json", "sample_manifest.json", "train_metrics.json",
    "val_metrics.json", "combined_metrics.json", "runtime_cost_audit.json",
    "failure_stratification.json", "security_audit.json", "acceptance.json",
    "final_acceptance.zh-CN.md", "CODEX_CONTEXT_P4_API_R1C.json",
    "delivery_manifest.json",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    files = {path.name for path in REPORT.iterdir() if path.is_file()}
    docs = {
        path.name: json.loads(path.read_text(encoding="utf-8"))
        for path in REPORT.glob("*.json")
    }
    manifest = docs["delivery_manifest.json"]
    manifest_ok = all(
        (REPORT / row["path"]).is_file()
        and (REPORT / row["path"]).stat().st_size == row["bytes"]
        and sha256(REPORT / row["path"]) == row["sha256"]
        for row in manifest["assets"]
    )
    delivery_text = "\n".join(path.read_text(encoding="utf-8") for path in REPORT.iterdir() if path.is_file())
    api_key = __import__("os").environ.get("TOPOCAE_LLM_API_KEY", "")
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    outcome_rows = [json.loads(line) for line in (OUTPUT / "outcome_audit.jsonl").read_text(encoding="utf-8").splitlines()]
    evaluation_rows = [json.loads(line) for line in (OUTPUT / "evaluation_audit.jsonl").read_text(encoding="utf-8").splitlines()]
    runtime = docs["runtime_cost_audit.json"]
    samples = docs["sample_manifest.json"]
    result = {
        "required_files_exact": files == REQUIRED,
        "delivery_file_count": len(files),
        "manifest_assets": len(manifest["assets"]),
        "manifest_ok": manifest_ok,
        "json_documents_valid": len(docs),
        "prompt_hash_new": sha256(ROOT / "prompts" / "p4_a1_structured_llm_v1.md"),
        "prompt_hash_matches": sha256(ROOT / "prompts" / "p4_a1_structured_llm_v1.md") == "e289c378c12b225da467310dada3b093bd13b2937736d7434f3039f8892344da",
        "full_freeze_status": verify_full_freeze()["status"],
        "formal_test_run_count": sum(len(rows) for rows in ledger["runs"].values()),
        "formal_test_content_accessed": ledger["test_content_accessed"],
        "selected_train": samples["splits"]["train"]["count"],
        "selected_val": samples["splits"]["val"]["count"],
        "outcome_rows": len(outcome_rows),
        "evaluation_rows": len(evaluation_rows),
        "actual_model_ids": runtime["actual_model_ids"],
        "backend_request_count": runtime["backend_request_count"],
        "backend_request_cap": runtime["backend_request_cap"],
        "accounted_cost_cny": runtime["accounted_cost_cny"],
        "cost_cap_cny": runtime["cost_cap_cny"],
        "api_key_found": bool(api_key and api_key in delivery_text),
        "bearer_value_hits": len(re.findall(r"(?i)Bearer\s+[A-Za-z0-9._-]{8,}", delivery_text)),
        "raw_query_field_hits": len(re.findall(r'"query"\s*:', delivery_text)),
        "raw_ground_truth_field_hits": len(re.findall(r'"ground_truth"\s*:', delivery_text)),
        "acceptance_status": docs["acceptance.json"]["status"],
    }
    passed = (
        result["required_files_exact"]
        and result["delivery_file_count"] == 12
        and result["manifest_assets"] == 11
        and result["manifest_ok"]
        and result["prompt_hash_matches"]
        and result["full_freeze_status"] == "PASS"
        and result["formal_test_run_count"] == 0
        and not result["formal_test_content_accessed"]
        and result["selected_train"] == result["selected_val"] == 50
        and result["outcome_rows"] == result["evaluation_rows"] == 100
        and result["actual_model_ids"] == ["deepseek-v4-pro"]
        and result["backend_request_count"] == 193 <= result["backend_request_cap"]
        and result["accounted_cost_cny"] < result["cost_cap_cny"]
        and not result["api_key_found"]
        and result["bearer_value_hits"] == 0
        and result["raw_query_field_hits"] == 0
        and result["raw_ground_truth_field_hits"] == 0
        and result["acceptance_status"] == "BELOW_P4_GATE_WITH_EVIDENCE"
    )
    result["status"] = "PASS" if passed else "FAIL"
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
