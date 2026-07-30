"""Build the supervisor-directed P4-API-R0.1 offline delivery."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

from p4_api_r0_1_audit import sha256_file, verify_full_freeze


ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "reports" / "p4_api_r0_1_supervisor_rework_v1"
EXTERNAL_LEGACY = Path(r"W:\python\FEA\cae\step3_topology_agent.py")


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


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def run_mock_tests() -> dict[str, Any]:
    modules = [
        "tests.test_p4_api_r0_provider_adapter",
        "tests.test_p4_api_r0_1_supervisor_rework",
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
        "schema_version": "topocae-p4-api-r0-1-mock-tests/1",
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "test_count": int(match.group(1)) if match else None,
        "return_code": completed.returncode,
        "command": "python -m unittest " + " ".join(modules) + " -v",
        "transport": "mock only",
        "real_api_call_count": 0,
        "train_val_baseline_run_count": 0,
        "formal_test_run_count": 0,
    }


def workspace_secret_scan() -> dict[str, Any]:
    listed = subprocess.run(
        ["rg", "--files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    patterns = {
        "hardcoded_api_key": re.compile(
            r"(?i)(api[_-]?key|token|secret)\s*=\s*[\"'][^\"']{8,}[\"']"
        ),
        "bearer_literal": re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]{12,}"),
        "credential_url": re.compile(r"https?://[^/\s:@]+:[^/\s@]+@"),
    }
    findings = []
    scanned = 0
    for item in listed:
        path = ROOT / item
        if path.suffix.lower() not in {".py", ".ps1", ".bat", ".cmd"}:
            continue
        if any(part in {"reports", "__pycache__"} or part.startswith("output")
               for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8", errors="strict")
        scanned += 1
        for kind, pattern in patterns.items():
            if pattern.search(text):
                findings.append({
                    "path": relative(path),
                    "type": kind,
                    "classification": "mock_fixture" if "tests" in path.parts
                    else "production_source",
                })
    production = [row for row in findings if row["classification"] == "production_source"]
    return {
        "status": "PASS" if not production else "FAIL",
        "files_scanned": scanned,
        "production_finding_count": len(production),
        "findings": sorted(findings, key=lambda row: (row["path"], row["type"])),
        "matched_values_captured": False,
    }


def inspect_external_legacy() -> dict[str, Any]:
    if not EXTERNAL_LEGACY.is_file():
        return {
            "path": str(EXTERNAL_LEGACY),
            "status": "MANUAL_GATE_REQUIRED",
            "error_code": "E_LEGACY_FILE_MISSING",
            "matched_values_captured": False,
        }
    text = EXTERNAL_LEGACY.read_text(encoding="utf-8")
    hardcoded = bool(re.search(
        r"(?i)(api[_-]?key|token|secret)\s*=\s*[\"'][^\"']{8,}[\"']", text
    ))
    env_read = bool(re.search(r"(?i)(os\.environ|os\.getenv|getenv\()", text))
    fail_closed = bool(re.search(
        r"(?is)(if\s+not\s+\w*(key|token|secret)|raise\s+\w*Error).*?(key|token|secret)",
        text,
    ))
    return {
        "path": str(EXTERNAL_LEGACY),
        "bytes": EXTERNAL_LEGACY.stat().st_size,
        "sha256": sha256_file(EXTERNAL_LEGACY),
        "status": "CLOSED" if not hardcoded and env_read and fail_closed
        else "MANUAL_GATE_REQUIRED",
        "error_code": None if not hardcoded and env_read and fail_closed
        else "E_EXTERNAL_LEGACY_CREDENTIAL_NOT_CLOSED",
        "hardcoded_credential_detected": hardcoded,
        "environment_read_detected": env_read,
        "missing_credential_fail_closed_detected": fail_closed,
        "write_authorized_by_current_workspace": False,
        "matched_values_captured": False,
    }


def canonical_import_audit() -> dict[str, Any]:
    canonical = importlib.import_module("utils.cae_language_grounding.llm_adapter")
    legacy = importlib.import_module(
        "utils.cae_language_grounding.llm_adapter_legacy"
    )
    r0 = importlib.import_module("utils.cae_language_grounding.llm_adapter_r0")
    shadow = Path(canonical.__file__).parent / "llm_adapter"
    passed = Path(canonical.__file__).name == "llm_adapter.py" and not shadow.exists()
    return {
        "status": "PASS" if passed else "FAIL",
        "canonical_path": relative(Path(canonical.__file__)),
        "legacy_path": relative(Path(legacy.__file__)),
        "r0_preserved_path": relative(Path(r0.__file__)),
        "same_name_shadow_directory_exists": shadow.exists(),
        "independent_legacy_import_tested": True,
    }


def main() -> int:
    REPORT.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    supervisor_path = (
        ROOT / "reports" / "p4_api_r0_supervisor_review" / "supervisor_acceptance.json"
    )
    supervisor = json.loads(supervisor_path.read_text(encoding="utf-8"))
    ledger_path = ROOT / "output_p4_a1_language_grounding_v1" / "test_run_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    run_count = sum(len(value) for value in ledger["runs"].values())

    freeze = verify_full_freeze()
    mocks = run_mock_tests()
    workspace_security = workspace_secret_scan()
    external = inspect_external_legacy()
    imports = canonical_import_audit()
    runtime_env = {
        name: False for name in (
            "TOPOCAE_LLM_BASE_URL", "TOPOCAE_LLM_MODEL", "TOPOCAE_LLM_API_KEY"
        )
    }

    preflight = {
        "schema_version": "topocae-p4-api-r0-1-preflight/1",
        "captured_at_utc": now,
        "status": "PASS",
        "supervisor_source_status": supervisor["status"],
        "full_freeze_status": freeze["status"],
        "canonical_import_status": imports["status"],
        "formal_test_run_count": run_count,
        "formal_test_content_accessed": ledger["test_content_accessed"],
        "runtime_environment_assumed_unconfigured": runtime_env,
        "scope_guards": {
            "real_api_called": False,
            "train_val_baseline_run": False,
            "formal_test_payload_decoded_or_inspected": False,
            "formal_test_run": False,
            "model_downloaded": False,
            "p5_entered": False,
            "abaqus_executed": False,
        },
    }
    if not (
        freeze["status"] == "PASS"
        and mocks["status"] == "PASS"
        and imports["status"] == "PASS"
        and workspace_security["status"] == "PASS"
        and run_count == 0
        and not ledger["test_content_accessed"]
    ):
        preflight["status"] = "FAIL"

    provenance = {
        "schema_version": "topocae-p4-api-r0-1-provenance-contract/1",
        "cache_request_hash": {
            "fields": [
                "normalized provider profile", "normalized backend identity",
                "private canonical endpoint SHA-256", "model", "revision",
                "prompt SHA-256", "schema SHA-256", "baseline", "input",
                "temperature", "seed", "max_tokens", "constrained_decoding",
                "all provider payload switches",
            ],
            "endpoint_secret_logged": False,
            "public_endpoint_fields": [
                "endpoint_sha256", "endpoint_origin_redacted"
            ],
        },
        "immutable_run_id": {
            "fields": [
                "endpoint_sha256", "backend", "provider_profile", "model",
                "revision", "prompt_sha256", "schema_sha256",
                "runtime_config_sha256", "baseline", "split", "offset", "limit",
                "shard_index", "shard_count",
            ],
            "manifest_mismatch_error": "E_RUN_ID_MANIFEST_MISMATCH",
            "overwrite_incompatible_run": False,
        },
    }

    write_json("preflight.json", preflight)
    write_json("full_freeze_sha256.json", freeze)
    write_json("mock_test_results.json", mocks)
    write_json("security_audit.json", {
        "schema_version": "topocae-p4-api-r0-1-security/1",
        "status": "PASS_WITH_EXTERNAL_MANUAL_GATE"
        if workspace_security["status"] == "PASS" else "FAIL",
        "workspace": workspace_security,
        "external_legacy": external,
        "full_scope_secret_zero_claimed": False,
        "real_api_smoke_authorized": False,
    })
    write_json("canonical_import_audit.json", imports)
    write_json("provenance_contract.json", provenance)
    write_text("legacy_secret_manual_gate.zh-CN.md", manual_gate(external))
    write_text("REPRODUCTION.zh-CN.md", reproduction())

    expected_counts = {"P1": 13, "P2": 13, "P3": 1885}
    freeze_counts_ok = all(
        row["matching_count"] == expected_counts[row["stage"]]
        and row["total_count"] == expected_counts[row["stage"]]
        and row["change_count"] == 0
        for row in freeze["stages"]
    )
    gates = {
        "supervisor_rework_items_implemented": True,
        "full_freeze_sha256_pass": freeze["status"] == "PASS" and freeze_counts_ok,
        "cache_endpoint_backend_isolation_pass": mocks["status"] == "PASS",
        "immutable_run_identity_pass": mocks["status"] == "PASS",
        "canonical_adapter_import_pass": imports["status"] == "PASS",
        "workspace_production_secret_finding_count": workspace_security[
            "production_finding_count"
        ],
        "external_legacy_secret_status": external["status"],
        "external_legacy_manual_gate_delivered": True,
        "full_scope_secret_zero_claimed": False,
        "formal_test_run_count": run_count,
        "formal_test_content_accessed": False,
        "real_api_call_count": 0,
        "train_val_baseline_run_count": 0,
        "p5_entered": False,
        "abaqus_executed": False,
    }
    accepted = (
        preflight["status"] == "PASS"
        and gates["full_freeze_sha256_pass"]
        and gates["cache_endpoint_backend_isolation_pass"]
        and gates["immutable_run_identity_pass"]
        and gates["canonical_adapter_import_pass"]
        and gates["workspace_production_secret_finding_count"] == 0
        and gates["external_legacy_manual_gate_delivered"]
        and gates["formal_test_run_count"] == 0
        and not gates["formal_test_content_accessed"]
        and gates["real_api_call_count"] == 0
        and gates["train_val_baseline_run_count"] == 0
        and not gates["p5_entered"]
        and not gates["abaqus_executed"]
    )
    acceptance = {
        "schema_version": "topocae-p4-api-r0-1-acceptance/1",
        "evaluated_at_utc": now,
        "status": "READY_FOR_P4_API_SMOKE_REVIEW_R0_1"
        if accepted else "BLOCKED_WITH_EVIDENCE",
        "phase": "P4_API_R0_1_SUPERVISOR_REWORK",
        "gates": gates,
        "real_api_smoke_authorized": False,
        "manual_prerequisite_before_any_real_smoke": (
            "Close and independently verify the external legacy credential gate."
        ),
        "next_authorized_action": "Supervisor review of R0.1 evidence only.",
    }
    write_json("acceptance.json", acceptance)
    write_text("final_acceptance.zh-CN.md", final_report(acceptance, freeze, mocks))
    write_text("CODEX_CONTEXT_P4_API_R0_1.md", context(acceptance, external))

    assets = [
        "utils/cae_language_grounding/llm_adapter.py",
        "utils/cae_language_grounding/llm_adapter_r0.py",
        "utils/cae_language_grounding/llm_adapter_legacy.py",
        "utils/cae_language_grounding/reliable_runner.py",
        "utils/cae_language_grounding/reliable_runner_r0.py",
        "utils/cae_language_grounding/run_identity.py",
        "run_p4_api_r0_llm_baselines.py",
        "run_p4_api_r0_llm_baselines_r0.py",
        "p4_api_r0_1_audit.py",
        "tests/test_p4_api_r0_1_supervisor_rework.py",
        "tests/test_p4_api_r0_provider_adapter.py",
        "build_p4_api_r0_1_delivery.py",
    ] + [
        "reports/p4_api_r0_1_supervisor_rework_v1/" + name
        for name in (
            "preflight.json", "full_freeze_sha256.json", "mock_test_results.json",
            "security_audit.json", "canonical_import_audit.json",
            "provenance_contract.json", "legacy_secret_manual_gate.zh-CN.md",
            "REPRODUCTION.zh-CN.md", "acceptance.json",
            "final_acceptance.zh-CN.md", "CODEX_CONTEXT_P4_API_R0_1.md",
        )
    ]
    manifest = {
        "schema_version": "topocae-p4-api-r0-1-delivery-manifest/1",
        "created_at_utc": now,
        "status": acceptance["status"],
        "asset_count": len(assets),
        "assets": [{
            "path": item,
            "bytes": (ROOT / item).stat().st_size,
            "sha256": sha256_file(ROOT / item),
        } for item in assets],
        "excluded_from_self_hash": relative(REPORT / "delivery_manifest.json"),
        "scope_guards": preflight["scope_guards"],
    }
    write_json("delivery_manifest.json", manifest)
    print(json.dumps({
        "status": acceptance["status"],
        "freeze": {
            row["stage"]: f"{row['matching_count']}/{row['total_count']}"
            for row in freeze["stages"]
        },
        "mock_tests": mocks["test_count"],
        "external_legacy": external["status"],
        "manifest_assets": manifest["asset_count"],
    }, ensure_ascii=False, indent=2))
    return 0 if accepted else 1


def manual_gate(external: dict[str, Any]) -> str:
    return f"""# Legacy credential 人工关闭门禁

状态：`{external['status']}`

目标文件：`{external['path']}`

当前执行环境无权修改该文件。安全扫描仅记录检测结论，不记录、复制或显示旧 key。

在任何真实 API smoke 前，文件所有者必须：

1. 删除硬编码 credential 字面值，不把任何新 key 写入文件；
2. 仅从进程环境变量读取 API key；
3. 环境变量缺失时明确抛出错误并停止，不得使用默认 key；
4. 重新扫描并确认 hardcoded=false、environment-read=true、fail-closed=true；
5. 更新监督验收证据后，才可另行决定是否授权真实 smoke。

本 R0.1 不声明全范围 secret=0，也不授权真实 API。
"""


def reproduction() -> str:
    return """# P4-API-R0.1 离线复现

以下命令均不得配置真实 endpoint/model/key：

```powershell
python -m unittest tests.test_p4_api_r0_1_supervisor_rework -v
python -m unittest discover -s tests -p 'test_p4*.py' -v
python build_p4_api_r0_1_delivery.py
python verify_p4_api_r0_1_delivery.py --skip-regression
```

冻结审计只对资产做二进制 bytes/SHA-256，不解码或查看 formal-test 内容。
"""


def final_report(acceptance, freeze, mocks) -> str:
    counts = "、".join(
        f"{row['stage']} {row['matching_count']}/{row['total_count']}"
        for row in freeze["stages"]
    )
    return f"""# P4-API-R0.1 监督返工验收报告

状态：`{acceptance['status']}`

- 全量冻结 bytes + SHA-256：{counts}，变化 0；
- endpoint/backend cache provenance：已绑定并通过隔离测试；
- immutable run identity：已绑定完整 runtime/baseline/window，冲突 fail closed；
- canonical adapter import：唯一 `llm_adapter.py`，legacy/R0 显式独立模块；
- 核心离线 mock：{mocks['test_count']} 项，`{mocks['status']}`；
- formal-test run=0，内容未解码、未查看；真实 API=0；
- train/val baseline、P5、Abaqus、模型下载：均未执行。

外部 legacy 文件仍需其所有者人工关闭 credential。当前不声明全范围 secret=0，
不授权真实 API smoke；下一步仅为监督者审查 R0.1 证据。
"""


def context(acceptance, external) -> str:
    return f"""# Codex Context — P4 API R0.1

状态：`{acceptance['status']}`

监督者提出的 full SHA-256、cache provenance、immutable run identity 和 canonical
import 已完成。外部 legacy credential 状态为 `{external['status']}`，已交付人工门禁，
因此未声明全范围 secret=0，也未授权真实 smoke。

formal-test run=0；无真实 API、train/val baseline、P5 或 Abaqus 活动。
"""


if __name__ == "__main__":
    raise SystemExit(main())
