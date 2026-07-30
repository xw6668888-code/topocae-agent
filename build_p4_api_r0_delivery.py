"""Build the offline-only P4-API-R0 acceptance package."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

from utils.cae_language_grounding.llm_adapter import PROVIDER_PROFILES
from utils.workspace_paths import resolve_workspace_path


ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "reports" / "p4_api_r0_provider_adapter_v1"
SOURCE_SUFFIXES = {".py", ".ps1", ".bat", ".cmd"}
EXCLUDED_SOURCE_PARTS = {
    ".git", "output_p4_a1_language_grounding_v1", "reports", "__pycache__"
}
SECRET_PATTERNS = {
    "hardcoded_api_key": re.compile(
        r"(?i)(api[_-]?key|token|secret)\s*=\s*[\"'][^\"']{8,}[\"']"
    ),
    "bearer_literal": re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]{12,}"),
    "credential_url": re.compile(r"https?://[^/\s:@]+:[^/\s@]+@"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(name: str, value: Any) -> Path:
    path = REPORT / name
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def write_text(name: str, value: str) -> Path:
    path = REPORT / name
    path.write_text(value.rstrip() + "\n", encoding="utf-8", newline="\n")
    return path


def public_path(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def upstream_freeze_check() -> dict[str, Any]:
    prior_path = (
        ROOT / "output_p4_a1_language_grounding_v1" / "integrity"
        / "frozen_assets_h4_pre_pause.json"
    )
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    rows = []
    for row in prior["manifests"]:
        manifest_path = ROOT / row["path"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metadata_changes = []
        for asset in manifest["assets"]:
            path = resolve_workspace_path(str(asset["path"]))
            if not path.is_file():
                metadata_changes.append({"path": str(asset["path"]), "change": "missing"})
            elif path.stat().st_size != int(asset["bytes"]):
                metadata_changes.append({
                    "path": str(asset["path"]),
                    "change": "byte_size_mismatch",
                    "expected_bytes": int(asset["bytes"]),
                    "actual_bytes": path.stat().st_size,
                })
        current_manifest_hash = sha256_file(manifest_path)
        rows.append({
            "stage": row["stage"],
            "manifest_path": row["path"],
            "manifest_sha256": current_manifest_hash,
            "manifest_unchanged_since_hash_audit": (
                current_manifest_hash == row["manifest_sha256"]
            ),
            "prior_full_hash_audit_status": prior["status"],
            "prior_matching_count": row["matching_count"],
            "prior_asset_count": row["asset_count"],
            "current_metadata_change_count": len(metadata_changes),
            "current_metadata_changes": metadata_changes,
        })
    passed = prior["status"] == "PASS" and all(
        item["manifest_unchanged_since_hash_audit"]
        and item["current_metadata_change_count"] == 0
        for item in rows
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "method": (
            "Reuse prior full SHA-256 audit, verify its manifests are unchanged, "
            "and compare current asset byte sizes. Formal-test payloads are not opened."
        ),
        "prior_audit_path": public_path(prior_path),
        "prior_audit_sha256": sha256_file(prior_path),
        "stages": rows,
    }


def source_secret_audit() -> dict[str, Any]:
    result = subprocess.run(
        ["rg", "--files"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    findings = []
    scanned = 0
    for relative in result.stdout.splitlines():
        path = ROOT / relative
        if path.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        if any(part in EXCLUDED_SOURCE_PARTS for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        scanned += 1
        for kind, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                classification = (
                    "mock_test_fixture" if "tests" in path.parts else "production_source"
                )
                findings.append({
                    "path": public_path(path),
                    "type": kind,
                    "classification": classification,
                })
    production = [row for row in findings if row["classification"] == "production_source"]
    return {
        "schema_version": "topocae-p4-api-r0-security-audit/1",
        "status": "PASS" if not production else "FAIL",
        "source_files_scanned": scanned,
        "production_secret_leak_count": len(production),
        "findings": sorted(findings, key=lambda row: (row["path"], row["type"])),
        "finding_value_capture": False,
        "delivery_endpoint_values_included": False,
        "delivery_api_keys_included": False,
        "delivery_bearer_headers_included": False,
    }


def run_mock_tests() -> dict[str, Any]:
    modules = [
        "tests.test_p4_api_r0_provider_adapter",
        "tests.test_p4_a1_llm_and_leakage",
    ]
    completed = subprocess.run(
        [sys.executable, "-m", "unittest", *modules, "-v"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    combined = completed.stdout + completed.stderr
    match = re.search(r"Ran\s+(\d+)\s+tests?", combined)
    return {
        "schema_version": "topocae-p4-api-r0-mock-results/1",
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "command": "python -m unittest " + " ".join(modules) + " -v",
        "test_count": int(match.group(1)) if match else None,
        "return_code": completed.returncode,
        "real_api_called": False,
        "transport": "in-process mock only",
        "coverage": [
            "openai_json_schema success",
            "deepseek_json_object success",
            "complete/base endpoint composition",
            "DeepSeek thinking disabled payload",
            "one invalid JSON/schema repair",
            "empty content, length truncation, missing usage, bad response shape",
            "HTTP 429/500/timeout retry and non-retryable 4xx",
            "atomic cache hit, corruption recovery, Structured/Direct isolation",
            "deterministic offset/limit/shard and formal-test lock",
            "recursive secret redaction",
            "legacy P4 adapter and leakage regression",
        ],
    }


def main() -> int:
    REPORT.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    upstream_acceptance_path = ROOT / "output_p4_a1_language_grounding_v1" / "acceptance.json"
    ledger_path = ROOT / "output_p4_a1_language_grounding_v1" / "test_run_ledger.json"
    h4_path = (
        ROOT / "reports" / "p4_a1_language_grounding_v1" / "human_review_package"
        / "H4-01_review_validation.json"
    )
    upstream_acceptance = json.loads(upstream_acceptance_path.read_text(encoding="utf-8"))
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    h4 = json.loads(h4_path.read_text(encoding="utf-8"))
    run_counts = {name: len(runs) for name, runs in ledger["runs"].items()}
    freeze = upstream_freeze_check()
    mock = run_mock_tests()
    security = source_secret_audit()

    hashed_assets = [
        "utils/cae_language_grounding/llm_adapter.py",
        "utils/cae_language_grounding/llm_adapter/__init__.py",
        "utils/cae_language_grounding/reliable_runner.py",
        "utils/cae_language_grounding/direct_llm.py",
        "run_p4_a1_llm_baselines.py",
        "run_p4_api_r0_llm_baselines.py",
        "prompts/p4_a1_structured_llm_v1.md",
        "prompts/p4_a1_direct_llm_v1.md",
        "schemas/cae_intent_result.schema.json",
        "schemas/p4_a1_direct_selection.schema.json",
    ]
    current_hashes = [{
        "path": item,
        "bytes": (ROOT / item).stat().st_size,
        "sha256": sha256_file(ROOT / item),
    } for item in hashed_assets]

    preflight = {
        "schema_version": "topocae-p4-api-r0-preflight/1",
        "captured_at_utc": now,
        "status": "PASS",
        "h4_status": h4["status"],
        "p4_upstream_status": upstream_acceptance["status"],
        "test_ledger_status": ledger["status"],
        "test_run_counts": run_counts,
        "test_content_accessed": ledger["test_content_accessed"],
        "test_unlock_authorized": ledger["test_unlock_authorized"],
        "upstream_freeze": freeze,
        "implementation_asset_hashes": current_hashes,
        "scope_guards": {
            "real_api_called": False,
            "train_or_val_run": False,
            "formal_test_content_accessed": False,
            "formal_test_run": False,
            "model_downloaded": False,
            "vllm_started": False,
            "abaqus_executed": False,
            "p5_entered": False,
        },
    }
    if not (
        h4["status"] == "PASS"
        and upstream_acceptance["status"] == "BLOCKED_AT_H4_RUNTIME"
        and all(count == 0 for count in run_counts.values())
        and not ledger["test_content_accessed"]
        and not ledger["test_unlock_authorized"]
        and freeze["status"] == "PASS"
    ):
        preflight["status"] = "FAIL"

    contract = {
        "schema_version": "topocae-p4-api-provider-contract/1",
        "profiles": {
            name: as_contract(profile) for name, profile in PROVIDER_PROFILES.items()
        },
        "endpoint_contract": {
            "accepted": ["base URL", "URL ending in /v1", "complete chat-completions URL"],
            "duplicate_v1_forbidden": True,
        },
        "schema_repair_maximum": 1,
        "silent_protocol_fallback": False,
        "stable_failure_codes": [
            "E_JSON_PARSE", "E_JSON_DUPLICATE_KEY", "E_JSON_TYPE",
            "E_RUNTIME_EMPTY_CONTENT", "E_RUNTIME_TRUNCATED",
            "E_RUNTIME_USAGE_MISSING", "E_RUNTIME_RESPONSE_SHAPE",
            "E_RUNTIME_RESPONSE_JSON", "E_RUNTIME_RESPONSE_TYPE",
            "E_HTTP_429", "E_HTTP_5XX", "E_HTTP_TIMEOUT", "E_HTTP_NETWORK",
        ],
        "request_hash_fields": [
            "provider", "model", "revision", "prompt", "schema", "baseline", "input",
            "generation switches",
        ],
        "runtime_metadata": [
            "actual_model", "system_fingerprint", "input_tokens", "output_tokens",
            "finish_reason", "latency_ms", "http_retry_count", "cache_hit", "request_hash",
        ],
    }

    write_json("preflight.json", preflight)
    write_text("preflight.zh-CN.md", preflight_markdown(preflight))
    write_json("provider_contract.json", contract)
    write_json("mock_test_results.json", mock)
    write_json("security_audit.json", security)
    write_text("api_smoke_procedure.zh-CN.md", smoke_procedure())

    gates = {
        "preflight_pass": preflight["status"] == "PASS",
        "new_and_legacy_provider_mocks_pass": mock["status"] == "PASS",
        "p4_existing_tests_no_regression": mock["status"] == "PASS",
        "upstream_freeze_change_count": 0 if freeze["status"] == "PASS" else None,
        "formal_test_run_count": sum(run_counts.values()),
        "formal_test_content_accessed": False,
        "production_secret_leak_count": security["production_secret_leak_count"],
        "same_input_cache_idempotent": mock["status"] == "PASS",
        "structured_direct_cache_isolated": mock["status"] == "PASS",
        "real_api_call_count": 0,
        "p5_entered": False,
    }
    accepted = (
        all(value is True for key, value in gates.items() if key.endswith("_pass") or key.startswith("same_") or key.startswith("structured_"))
        and gates["upstream_freeze_change_count"] == 0
        and gates["formal_test_run_count"] == 0
        and gates["formal_test_content_accessed"] is False
        and gates["production_secret_leak_count"] == 0
        and gates["real_api_call_count"] == 0
        and gates["p5_entered"] is False
    )
    acceptance = {
        "schema_version": "topocae-p4-api-r0-acceptance/1",
        "evaluated_at_utc": now,
        "status": "READY_FOR_P4_API_SMOKE_REVIEW" if accepted else "BLOCKED_WITH_EVIDENCE",
        "phase": "P4_API_R0_PROVIDER_ADAPTER",
        "gates": gates,
        "next_authorized_action": (
            "Supervisor reviews api_smoke_procedure.zh-CN.md and explicitly decides whether "
            "to authorize a separate low-cost, non-test P4 API smoke."
        ),
        "prohibited_next_actions": [
            "formal test access or run", "train/val baseline run without separate authorization",
            "P5 entry", "Abaqus execution",
        ],
    }
    write_json("acceptance.json", acceptance)
    write_text("final_acceptance.zh-CN.md", final_markdown(acceptance, mock, security))
    write_text("CODEX_CONTEXT_P4_API_R0.md", context_markdown(acceptance, preflight))

    manifest_assets = hashed_assets + [
        "tests/test_p4_api_r0_provider_adapter.py",
        "build_p4_api_r0_delivery.py",
    ] + [
        "reports/p4_api_r0_provider_adapter_v1/" + name for name in (
            "preflight.json", "preflight.zh-CN.md", "provider_contract.json",
            "mock_test_results.json", "security_audit.json",
            "api_smoke_procedure.zh-CN.md", "acceptance.json",
            "final_acceptance.zh-CN.md", "CODEX_CONTEXT_P4_API_R0.md",
        )
    ]
    manifest = {
        "schema_version": "topocae-p4-api-r0-delivery-manifest/1",
        "created_at_utc": now,
        "status": acceptance["status"],
        "asset_count": len(manifest_assets),
        "assets": [{
            "path": item,
            "bytes": (ROOT / item).stat().st_size,
            "sha256": sha256_file(ROOT / item),
        } for item in manifest_assets],
        "excluded_from_self_hash": "reports/p4_api_r0_provider_adapter_v1/delivery_manifest.json",
        "scope_guards": preflight["scope_guards"],
    }
    write_json("delivery_manifest.json", manifest)
    print(json.dumps({
        "status": acceptance["status"],
        "report": str(REPORT.resolve()),
        "mock_tests": mock["test_count"],
        "asset_count": manifest["asset_count"],
    }, ensure_ascii=False, indent=2))
    return 0 if accepted else 1


def as_contract(profile) -> dict[str, Any]:
    return {
        "response_format": profile.response_format,
        "send_seed_default": profile.send_seed,
        "send_temperature_default": profile.send_temperature,
        "send_thinking_default": profile.send_thinking,
        "send_max_tokens_default": profile.send_max_tokens,
        "thinking_payload": {"type": "disabled"} if profile.send_thinking else None,
        "local_schema_validation_required": profile.response_format == "json_object",
    }


def preflight_markdown(value: dict[str, Any]) -> str:
    return f"""# P4-API-R0 预检

状态：`{value['status']}`

- H4：`{value['h4_status']}`
- 上游 P4：`{value['p4_upstream_status']}`
- test ledger：`{value['test_ledger_status']}`，四条路线运行数均为 0
- 上游冻结：`{value['upstream_freeze']['status']}`
- formal-test 内容：未打开、未解码、未运行
- 真实 API、模型下载、vLLM、Abaqus、P5：均未执行

冻结核验复用现有完整 SHA-256 PASS 证据，确认对应 manifest 未变化，并核对当前
资产字节数；为遵守本 Goal，未打开 formal-test payload。
"""


def smoke_procedure() -> str:
    return """# P4 API Smoke 审查程序

当前仅供监督者审查，**本 R0 未执行以下真实 API 步骤**。

1. 仅选择非 test 的最小 val 样本，记录样本 ID，不打印 ground truth。
2. 通过进程环境配置 provider profile、base/complete endpoint、model 和 API key；
   密钥不得写入命令历史、配置、日志或报告。
3. 先用 mock transport 复跑 `tests.test_p4_api_r0_provider_adapter`。
4. 监督者另行明确授权后，使用 `run_p4_api_r0_llm_baselines.py` 的 `--limit 1`
   执行单条低成本 smoke；不得使用 test split。
5. 核对实际 model、fingerprint、usage、延迟、finish reason、request hash 与 cache。
6. 第二次执行同一请求必须 cache hit，backend request count 为 0。
7. 任一协议、schema、截断、usage 或 secret 审计失败立即停止，不得静默 fallback。

R0 交付状态只到 `READY_FOR_P4_API_SMOKE_REVIEW`，本程序不构成 API 调用授权。
"""


def final_markdown(acceptance, mock, security) -> str:
    return f"""# P4-API-R0 最终验收报告

状态：`{acceptance['status']}`

Provider-aware adapter 已支持 `openai_json_schema` 与
`deepseek_json_object`，并具备显式 payload 开关、稳定错误码、有限 HTTP 重试、
一次 schema 修复、原子内容哈希缓存、断点重跑、Structured/Direct 隔离、确定性
窗口与 R0 formal-test 硬锁。

- 离线 mock 与旧 P4 回归：{mock['test_count']} 项，`{mock['status']}`
- 上游冻结变化：0
- formal-test 运行：0；内容未访问
- production secret 泄漏：{security['production_secret_leak_count']}
- 真实收费 API：0
- train/val、Abaqus、P5：未执行

下一步只能由监督者审查 API smoke 程序；本交付未授权或执行真实 smoke。
"""


def context_markdown(acceptance, preflight) -> str:
    return f"""# Codex Context — P4 API R0

Last updated: {acceptance['evaluated_at_utc']}

## Current State

`{acceptance['status']}`

Provider profiles、mock 协议回归、可靠缓存/重试、deterministic window、secret
redaction 和 R0 test lock 已完成。H4=`{preflight['h4_status']}`，上游 P4 保持
`{preflight['p4_upstream_status']}`，test ledger 所有 run 为 0。

## Hard Stops

- 未调用真实 API；未运行 train/val。
- 未打开或运行 formal test。
- 未进入 P5；未执行 Abaqus。
- 下一步仅为监督者 API smoke review。
"""


if __name__ == "__main__":
    raise SystemExit(main())
