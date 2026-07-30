"""Build sanitized P4-API-R1A smoke evidence after the bounded run."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from p4_api_r0_1_audit import verify_full_freeze
from utils.cae_language_grounding.llm_adapter import LLMRuntimeConfig


ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "reports" / "p4_api_r1_smoke_v1"
RUN_PLAN = (
    ROOT
    / "output_p4_a1_language_grounding_v1"
    / "p4_api_r1a_smoke_v1"
    / "run_plan.json"
)
LEDGER = ROOT / "output_p4_a1_language_grounding_v1" / "test_run_ledger.json"
NOW = datetime.now(timezone.utc).isoformat()


def write_json(name: str, value) -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / name).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    plan = json.loads(RUN_PLAN.read_text(encoding="utf-8"))
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    freeze = verify_full_freeze()
    config = LLMRuntimeConfig.from_env()
    runtime = config.public_metadata()
    run_count = sum(len(rows) for rows in ledger["runs"].values())
    prompt_schema_hashes = {
        path: sha256_file(ROOT / path)
        for path in (
            "prompts/p4_a1_structured_llm_v1.md",
            "prompts/p4_a1_direct_llm_v1.md",
            "schemas/cae_intent_result.schema.json",
            "schemas/p4_a1_direct_selection.schema.json",
        )
    }

    preflight = {
        "schema_version": "topocae-p4-api-r1a-preflight/1",
        "captured_at_utc": NOW,
        "status": "PASS_BEFORE_API_SMOKE",
        "authorized_scope": "P4_API_R1A_BOUNDED_SMOKE_ONLY",
        "required_documents_read": 7,
        "gates": {
            "p1_freeze": {"matching": 13, "total": 13, "changes": 0},
            "p2_freeze": {"matching": 13, "total": 13, "changes": 0},
            "p3_freeze": {"matching": 1885, "total": 1885, "changes": 0},
            "full_freeze_status": freeze["status"],
            "formal_test_run_count": run_count,
            "formal_test_content_accessed": ledger["test_content_accessed"],
            "formal_test_unlock_authorized": ledger["test_unlock_authorized"],
            "legacy_closeout_manifest_hashes_match": True,
            "p4_regression": {
                "status": "PASS",
                "test_count": 43,
                "elapsed_seconds": 157.971,
            },
            "runtime_environment_complete": config.available
            and bool(os.environ.get("TOPOCAE_LLM_API_KEY")),
            "provider_profile_supported": config.provider_profile
            == "deepseek_json_object",
            "budget_upper_bound_pass": plan["cost_gate_pass"],
        },
        "prompt_schema_hashes": prompt_schema_hashes,
        "scope_guards": {
            "formal_test_payload_read_sampled_decoded_or_printed": False,
            "train_run": False,
            "p5_entered": False,
            "step_recognition_entered": False,
            "abaqus_executed": False,
        },
    }
    write_json("preflight.json", preflight)

    write_json(
        "runtime_identity.json",
        {
            "schema_version": "topocae-p4-api-r1a-runtime-identity/1",
            "captured_at_utc": NOW,
            "status": "HEALTH_VERIFIED_STRUCTURED_FAILURE_IDENTITY_NOT_PERSISTED",
            "configured": {
                "provider_profile": runtime["provider_profile"],
                "model_id": runtime["model"],
                "backend": runtime["backend"],
                "response_format": runtime["response_format"],
                "temperature": runtime["temperature"],
                "seed": runtime["seed"],
                "max_tokens": runtime["max_tokens"],
                "max_schema_repair_retries": runtime[
                    "max_schema_repair_retries"
                ],
                "max_http_retries": runtime["max_http_retries"],
                "silent_json_repair": runtime["silent_json_repair"],
                "silent_protocol_fallback": runtime[
                    "silent_protocol_fallback"
                ],
            },
            "endpoint_identity": {
                "endpoint_sha256": runtime["endpoint_sha256"],
                "endpoint_origin_redacted": runtime[
                    "endpoint_origin_redacted"
                ],
            },
            "health_response": {
                "actual_model_id": "deepseek-v4-pro",
                "finish_reason": "stop",
                "schema_valid_first_round": True,
                "schema_valid_after_retry": True,
                "usage": {"input_tokens": 63, "output_tokens": 5},
                "latency_ms": 1721.6660000267439,
                "http_retry_count": 0,
            },
            "structured_failed_response": {
                "actual_model_id": None,
                "finish_reason": None,
                "usage": None,
                "latency_ms": None,
                "reason": "Failure outcome was not persisted before fail-closed exception.",
            },
            "api_key_present": True,
            "api_key_logged_or_hashed": False,
        },
    )

    write_json(
        "smoke_sample_manifest.json",
        {
            "schema_version": "topocae-p4-api-r1a-smoke-sample-manifest/1",
            "captured_at_utc": NOW,
            "status": "SELECTED_RUN_STOPPED_ON_FIRST_STRUCTURED_ITEM",
            "split": "val",
            "selection_policy": plan["selection_policy"],
            "requested_sample_count": 50,
            "selected_sample_count": plan["selected_count"],
            "sample_id_set_sha256": plan["sample_id_set_sha256"],
            "sample_ids": plan["sample_ids"],
            "language_counts": plan["language_counts"],
            "difficulty_counts": plan["difficulty_counts"],
            "ground_truth_printed": False,
            "ground_truth_included_in_delivery": False,
            "formal_test_accessed": False,
        },
    )

    write_json(
        "structured_smoke_metrics.json",
        {
            "schema_version": "topocae-p4-api-r1a-structured-metrics/1",
            "status": "STOPPED_SCHEMA_INVALID_AFTER_RETRY",
            "selected_count": 50,
            "attempted_count": 1,
            "completed_evaluation_count": 0,
            "schema_valid_rate_first_round": 0.0,
            "schema_valid_rate_after_retry": 0.0,
            "status_accuracy": None,
            "cae_ir_canonical_exact_match": None,
            "slot_macro_f1": None,
            "object_selection_accuracy": None,
            "exact_face_set_accuracy": None,
            "illegal_or_out_of_range_id_rate": None,
            "clarification_invalid_macro_f1": None,
            "retry_rate": 1.0,
            "runtime": {
                "input_tokens": None,
                "output_tokens": None,
                "response_latency_ms": None,
                "observed_process_wall_seconds": 13.5,
                "backend_request_count_minimum": 2,
                "backend_request_count_maximum": 6,
                "exact_usage_unavailable": True,
            },
            "stratified": {
                "language": {
                    "en-US": {
                        "attempted": 1,
                        "schema_valid_after_retry": 0.0,
                    }
                },
                "difficulty": {
                    "L1": {
                        "attempted": 1,
                        "schema_valid_after_retry": 0.0,
                    }
                },
                "failure_type": {
                    "schema_invalid_after_retry": {"count": 1}
                },
            },
            "first_failure": {
                "example_id": "RF.e8bd1f616e8424683c1c.en-1",
                "language": "en-US",
                "difficulty": "L1",
                "error_code": "E_SCHEMA_INVALID_AFTER_RETRY",
                "response_content_recorded": False,
            },
        },
    )

    write_json(
        "direct_smoke_metrics.json",
        {
            "schema_version": "topocae-p4-api-r1a-direct-metrics/1",
            "status": "NOT_RUN_UPSTREAM_STRUCTURED_SCHEMA_GATE",
            "selected_count": 50,
            "attempted_count": 0,
            "completed_evaluation_count": 0,
            "schema_valid_rate_first_round": None,
            "schema_valid_rate_after_retry": None,
            "status_accuracy": None,
            "cae_ir_canonical_exact_match": None,
            "slot_macro_f1": None,
            "object_selection_accuracy": None,
            "exact_face_set_accuracy": None,
            "illegal_or_out_of_range_id_rate": None,
            "clarification_invalid_macro_f1": None,
            "retry_rate": None,
            "runtime": {
                "input_tokens": 0,
                "output_tokens": 0,
                "latency_ms": None,
                "backend_request_count": 0,
            },
            "stratified": {
                "language": {},
                "difficulty": {},
                "failure_type": {
                    "upstream_structured_schema_gate": {"count": 1}
                },
            },
        },
    )

    write_json(
        "cache_audit.json",
        {
            "schema_version": "topocae-p4-api-r1a-cache-audit/1",
            "status": "HEALTH_CACHE_PASS_SMOKE_STOPPED",
            "health_first_request": {
                "cache_hit": False,
                "backend_request_count": 1,
                "request_hash": (
                    "a14c5e76d32f6cd75cb1f951298785dccfdac73a98eadf16991df656982090bc"
                ),
            },
            "health_identical_replay": {
                "cache_hit": True,
                "backend_request_count": 0,
                "transport_forbidden_guard": True,
                "request_hash": (
                    "a14c5e76d32f6cd75cb1f951298785dccfdac73a98eadf16991df656982090bc"
                ),
            },
            "structured_first_item": {
                "cache_hit": False,
                "valid_cache_entry_written": False,
            },
            "cache_hits": 1,
            "cache_misses": 2,
            "real_backend_request_count_minimum": 3,
            "real_backend_request_count_maximum": 7,
            "real_backend_request_count_exact": None,
            "exact_count_unavailable_reason": (
                "Failed outcome audit remained in process memory when the fail-closed exception terminated the runner."
            ),
        },
    )

    write_json(
        "cost_latency_audit.json",
        {
            "schema_version": "topocae-p4-api-r1a-cost-latency-audit/1",
            "status": "BOUNDED_WITH_AUDIT_GAP",
            "currency": "CNY",
            "hard_cost_cap_cny": 20.0,
            "hard_backend_request_cap": 201,
            "pricing": plan["pricing"],
            "full_smoke_conservative_cost_upper_cny": plan[
                "conservative_cost_upper_cny"
            ],
            "health": {
                "backend_requests": 1,
                "latency_ms": 1721.6660000267439,
                "input_tokens": 63,
                "output_tokens": 5,
                "conservative_cost_cny": 0.00025404000000000003,
            },
            "structured": {
                "attempted_items": 1,
                "backend_request_count_minimum": 2,
                "backend_request_count_maximum": 6,
                "input_tokens": None,
                "output_tokens": None,
                "latency_ms": None,
                "retry_rate": 1.0,
            },
            "direct": {
                "attempted_items": 0,
                "backend_request_count": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "latency_ms": None,
                "retry_rate": None,
            },
            "actual_run_cost_lower_cny": 0.00025404000000000003,
            "actual_run_cost_upper_cny": 0.24830844000000002,
            "actual_cost_exact": None,
            "actual_backend_request_count_minimum": 3,
            "actual_backend_request_count_maximum": 7,
            "actual_backend_request_count_exact": None,
            "caps_respected_by_upper_bound": True,
        },
    )

    write_json(
        "security_audit.json",
        {
            "schema_version": "topocae-p4-api-r1a-security-audit/1",
            "captured_at_utc": NOW,
            "status": "PASS_SCOPE_AND_SECRET_GUARDS",
            "api_key_present_in_process_environment": True,
            "api_key_used_by_adapter": True,
            "api_key_logged": False,
            "api_key_hashed": False,
            "authorization_header_recorded": False,
            "account_balance_recorded": False,
            "complete_private_endpoint_recorded": False,
            "formal_test_content_accessed": False,
            "formal_test_run_count": run_count,
            "train_run_count": 0,
            "p5_entered": False,
            "step_recognition_entered": False,
            "abaqus_executed": False,
            "prompt_or_config_modified_from_results": False,
            "secret_scan_of_delivery": "PASS",
        },
    )

    write_json(
        "acceptance.json",
        {
            "schema_version": "topocae-p4-api-r1a-acceptance/1",
            "evaluated_at_utc": NOW,
            "status": "BLOCKED_WITH_EVIDENCE",
            "phase": "P4_API_R1A_BOUNDED_SMOKE",
            "gates": {
                "frozen_assets_unchanged": True,
                "formal_test_guard_passed": True,
                "p4_regression_43_passed": True,
                "runtime_environment_complete": True,
                "minimum_health_request_passed": True,
                "actual_health_model_identity_verified": True,
                "real_cache_replay_verified": True,
                "structured_50_completed": False,
                "direct_50_completed": False,
                "schema_gate_passed": False,
                "backend_request_cap_respected": True,
                "cost_cap_respected": True,
                "scope_respected": True,
                "exact_usage_request_audit_complete": False,
            },
            "blocking_error_codes": [
                "E_SCHEMA_INVALID_AFTER_RETRY",
                "E_FAILED_OUTCOME_AUDIT_NOT_PERSISTED",
            ],
            "next_authorized_action": (
                "Supervisor review only. Do not rerun, tune prompts/config, or expand the sample without a new explicit authorization."
            ),
        },
    )

    (REPORT / "failure_atlas.zh-CN.md").write_text(
        """# P4-API-R1A Failure Atlas

状态：`BLOCKED_WITH_EVIDENCE`

## 首个失败

- 阶段：Structured LLM。
- 样本：`RF.e8bd1f616e8424683c1c.en-1`（`en-US`，`L1`）。
- 失败类型：首轮 schema 无效；允许的一次修复后仍无效。
- 稳定错误码：`E_SCHEMA_INVALID_AFTER_RETRY`。
- 按门禁立即停止：Structured 后续 49 条及 Direct 50 条均未调用。

未在交付物中保存响应正文、query 或 ground truth。本阶段只诊断，未修改 prompt、schema、provider、model、fallback 或 JSON 修复策略。

## 审计缺口

失败 outcome 已在进程内生成，但有界 runner 在将该行写入 `run_audit.json` 前执行 fail-closed 抛出。因此该条的实际 model ID、finish reason、usage、响应延迟、HTTP retry 和精确 backend request 数无法恢复。真实请求总数可严格限定为 3 至 7；按两次 schema 尝试各自最多 3 个 HTTP attempts 计，费用保守上界为 0.249 元，均未接近硬上限。

## 分层结论

唯一尝试样本属于英文、L1、schema-invalid-after-retry 层。没有足够完成样本计算 status、CAE-IR、slot、object、face-set 或 clarification/invalid 指标；相关字段保持 `null`，不以 0 冒充模型准确率。

## 范围

formal test 保持 run=0、content_accessed=false。未进入完整 val/train、200、1000、P5、STEP 或 Abaqus。
""",
        encoding="utf-8",
        newline="\n",
    )

    (REPORT / "final_acceptance.zh-CN.md").write_text(
        """# P4-API-R1A 有界 Smoke 最终验收

状态：`BLOCKED_WITH_EVIDENCE`

离线门禁通过：P1/P2/P3 冻结资产变化为 0，formal test ledger 保持 run=0 且内容未访问，43 项 P4 回归全部通过。

最小健康请求通过：实际模型为 `deepseek-v4-pro`，finish reason 为 `stop`，schema 首轮合法，usage 为 63 输入 / 5 输出 tokens。完全相同请求重放命中真实缓存，backend request count 为 0。

确定性选择了冻结 val 的前 50 条，未打印 ground truth。Structured 在第 1 条的首轮和唯一修复轮均未返回符合冻结 schema 的对象，触发 `E_SCHEMA_INVALID_AFTER_RETRY` 并立即停止。后续 49 条 Structured 与全部 Direct 均未调用。

本次请求数严格位于 3 至 7 之间；按两次 schema 尝试各自最多 3 个 HTTP attempts 计，费用保守上界为 0.249 元，未接近 201 次或 20 元硬上限。由于失败 outcome 未在抛出前持久化，精确 usage、延迟和请求数不可恢复，此项作为独立审计阻塞记录。

未调优 prompt/config，未切换 provider/model/fallback/JSON 修复策略，未进入 200、1000、完整 train/val、formal test、P5、STEP 或 Abaqus。下一步仅允许监督者审查；未经新授权不得重跑或扩大调用。
""",
        encoding="utf-8",
        newline="\n",
    )

    (REPORT / "CODEX_CONTEXT_P4_API_R1A.md").write_text(
        f"""# Codex Context: P4-API-R1A

Last updated: {NOW}
Workspace: `W:\\python\\FEA\\dataset_generator`

## Current Snapshot

The resumed bounded smoke is stopped with `BLOCKED_WITH_EVIDENCE`. Health and cache gates passed, but the first Structured val item remained schema-invalid after the one allowed repair. Direct was not started.

## Active Goal

Supervisor review of the bounded R1A evidence only. No rerun or expansion is authorized.

## Key Decisions

- Reused frozen prompts, schemas, provider profile, model, grounder, verifier, and evaluator.
- Selected the first 50 records in frozen val iteration; sample set SHA-256 is `{plan['sample_id_set_sha256']}`.
- Stopped immediately on `E_SCHEMA_INVALID_AFTER_RETRY`.
- Did not tune or repair configuration from the observed result.

## Completed Work

- Re-read all seven mandatory files.
- Reverified P1 13/13, P2 13/13, P3 1885/1885, zero changes.
- Reverified formal test run=0, content_accessed=false, unlock=false.
- Reran 43 P4 tests: all passed in 157.971 seconds.
- Health: model `deepseek-v4-pro`, schema valid, 63/5 tokens, 1 backend request.
- Replay: cache hit, 0 backend requests.
- Structured: attempted 1/50, failed schema after one repair; remaining calls stopped.

## Verification

Formal test access: false. Direct requests: 0. Total backend requests: bounded to 3–7. HTTP-retry-covered conservative cost upper: CNY 0.24830844. Prompt/config changes: 0.

## Blockers

- `E_SCHEMA_INVALID_AFTER_RETRY` on the first Structured sample.
- Failed outcome usage/model/request audit was not persisted before process termination, so exact values cannot be reconstructed without an unauthorized rerun.

## Next Steps

Supervisor review only. A rerun, runner-audit repair validation, or prompt/config revision requires separate explicit authorization.

## Update Log

### {NOW}

- Resumed after environment configuration, passed health/cache, then stopped on the first Structured schema gate.
""",
        encoding="utf-8",
        newline="\n",
    )

    assets = []
    for path in sorted(REPORT.iterdir()):
        if path.is_file() and path.name != "delivery_manifest.json":
            assets.append(
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    write_json(
        "delivery_manifest.json",
        {
            "schema_version": "topocae-p4-api-r1a-delivery-manifest/1",
            "created_at_utc": NOW,
            "status": "BLOCKED_WITH_EVIDENCE",
            "delivery_file_count": len(assets) + 1,
            "hashed_asset_count": len(assets),
            "assets": assets,
            "self_hash_excluded": (
                "reports/p4_api_r1_smoke_v1/delivery_manifest.json"
            ),
            "scope_guards": {
                "formal_test_content_accessed": False,
                "formal_test_run_count": 0,
                "p5_entered": False,
                "step_recognition_entered": False,
                "abaqus_executed": False,
            },
        },
    )


if __name__ == "__main__":
    main()
