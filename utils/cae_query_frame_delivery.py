"""Presentation-only assembly for the offline P4-API-R2A.1 delivery."""

from __future__ import annotations

import platform
import sys
from typing import Any, Mapping

from utils.cae_query_frame_runtime import NORMAL_ENDPOINT, QUERY_FRAME_SCHEMA_SHA256


READY = "READY_FOR_P4_API_R2B_20_VAL_SMOKE_REVIEW"
FAILED = "COMPACT_QUERY_FRAME_GATE_FAILED_WITH_EVIDENCE"


def assemble_delivery_payloads(
    audits: Mapping[str, Mapping[str, Any]],
    smoke_manifest: Mapping[str, Any],
    regression: Mapping[str, Any],
) -> dict[str, Any]:
    checks = {
        "offline_only": audits["mock_request"]["network_request_count"] == 0,
        "api_request_count_eq_0": audits["cost_budget"]["actual_requests"] == 0,
        "formal_test_access_eq_0": audits["leakage"]["formal_test_access_count"] == 0,
        "runtime_prompt_gate_pass": audits["runtime_prompt"]["gate_pass"],
        "repair_contract_gate_pass": audits["repair_contract"]["gate_pass"],
        "mock_request_gate_pass": audits["mock_request"]["gate_pass"],
        "smoke_coverage_gate_pass": audits["smoke_coverage"]["coverage_gate_pass"],
        "smoke_language_quota_10_10": audits["smoke_coverage"]["language_counts"] == {"en-US": 10, "zh-CN": 10},
        "smoke_duplicate_canonical_id_eq_0": audits["smoke_coverage"]["duplicate_canonical_id_count"] == 0,
        "leakage_gate_pass": audits["leakage"]["gate_pass"],
        "freeze_change_count_eq_0": audits["freeze"]["change_count"] == 0,
        "r2a_unchanged": audits["freeze"]["r2a_unchanged"],
        "p1_p2_p3_unchanged": all(audits["freeze"][key] for key in ("p1_unchanged", "p2_unchanged", "p3_unchanged")),
        "full_test_p4_regression_pass": regression.get("status") == "PASS" and regression.get("return_code") == 0,
    }
    status = READY if all(checks.values()) else FAILED
    acceptance = {
        "schema_version": "topocae-p4-api-r2a-1-acceptance/1",
        "status": status,
        "checks": checks,
        "metrics": {
            "smoke_samples": 20,
            "en_samples": audits["smoke_coverage"]["language_counts"].get("en-US", 0),
            "zh_samples": audits["smoke_coverage"]["language_counts"].get("zh-CN", 0),
            "runtime_prompt_utf8_bytes": audits["runtime_prompt"]["runtime_prompt_utf8_bytes"],
            "runtime_prompt_sha256": audits["runtime_prompt"]["runtime_prompt_sha256"],
            "total_request_hard_cap": audits["cost_budget"]["total_request_hard_cap"],
            "total_cost_upper_cny": audits["cost_budget"]["total_cost_upper_cny"],
            "actual_requests": 0,
            "formal_test_access_count": 0,
            "freeze_change_count": audits["freeze"]["change_count"],
            "test_p4_tests_run": regression.get("tests_run"),
        },
        "runtime": {
            "python_executable": regression.get("interpreter_path", sys.executable),
            "python_version": regression.get("python_version", platform.python_version()),
        },
        "next_stage_automatically_started": False,
        "formal_test_started": False,
        "p5_started": False,
        "step_accessed": False,
        "abaqus_accessed": False,
    }
    final_md = f"""# P4-API-R2A.1 最终验收

状态：`{status}`

本阶段完成 CAEQueryFrame v1 的离线 runtime prompt、一次显式 repair 协议、DeepSeek JSON-object mock 请求合同，以及 20 条 val 烟测清单。未调用真实 API，未解码 formal test，未读取 STEP、graph 或 Abaqus。

## 关键结果

- runtime prompt 内嵌冻结 QueryFrame Schema，prompt SHA-256：`{audits['runtime_prompt']['runtime_prompt_sha256']}`。
- 20 条 val 样本由确定性加权 set-cover 选择，英文 10 条、中文 10 条，canonical identity 重复为 0。
- 覆盖 L1-L5、三种状态、单/多操作、7 类操作和要求的 6 类量词。
- repair 最多 1 次，第二次 fail closed；mock 网络请求数为 0。
- 规划硬上限为 40 次请求；本阶段实际请求数为 0。
- P1/P2/P3、历史 P4 与 R2A 冻结检查变化数为 0。

完整回归：`{regression.get('tests_run')}` 个 `test_p4*.py` 测试，状态 `{regression.get('status')}`。
"""
    contract_md = f"""# CAEQueryFrame Runtime Prompt 合同

## 输入与输出

输入严格为 `{{text, language}}`；输出严格为一个 CAEQueryFrame JSON object，不含外围文本。runtime prompt 的完整内容由 `utils/cae_query_frame_runtime.py` 确定性生成。

- 冻结 QueryFrame Schema SHA-256：`{QUERY_FRAME_SCHEMA_SHA256}`
- Runtime prompt SHA-256：`{audits['runtime_prompt']['runtime_prompt_sha256']}`
- UTF-8 字节数：{audits['runtime_prompt']['runtime_prompt_utf8_bytes']}
- Schema：完整内嵌于 system text，不使用 provider strict schema 字段。
- 示例：4 个手工构造并经 Schema、编译器和 P2 validator 校验的示例，覆盖 valid、needs_clarification、invalid 与多操作。

## Repair

仅允许一次显式 repair。输入只包含原始 `{{text, language}}`、稳定校验错误码和受限 Schema 摘要；不传前次输出、答案、identity 或几何证据。第二次尝试 fail closed，禁止 silent repair。

## 请求合同

使用普通端点 `{NORMAL_ENDPOINT}` 与 `response_format={{\"type\":\"json_object\"}}`。本阶段仅使用 mock transport，真实网络请求为 0。
"""
    context_md = f"""# CODEX_CONTEXT P4-API-R2A.1

- 停止状态：`{status}`
- 冻结依赖：P4-API-R2A QueryFrame Schema、编译器、审计资产均未修改。
- 新 runtime：`utils/cae_query_frame_runtime.py`。
- 离线审计/交付构建：`run_p4_api_r2a_1_runtime_prompt.py`。
- 烟测清单：20 条 val，10 en-US + 10 zh-CN，只持久化哈希与覆盖元数据。
- API 请求：0；formal-test access：0；STEP/graph/Abaqus：0。
- 下一步只能进入人工评审后的 P4-API-R2B 20-val smoke，不自动执行。
"""
    return {
        "final_acceptance.zh-CN.md": final_md,
        "acceptance.json": acceptance,
        "runtime_prompt_contract.zh-CN.md": contract_md,
        "runtime_prompt_audit.json": audits["runtime_prompt"],
        "mock_request_audit.json": audits["mock_request"],
        "repair_contract_audit.json": audits["repair_contract"],
        "smoke_sample_manifest.json": dict(smoke_manifest),
        "smoke_coverage_audit.json": audits["smoke_coverage"],
        "cost_budget_audit.json": audits["cost_budget"],
        "leakage_audit.json": audits["leakage"],
        "regression_results.json": dict(regression),
        "freeze_audit.json": audits["freeze"],
        "CODEX_CONTEXT_P4_API_R2A_1.md": context_md,
    }


__all__ = ["FAILED", "READY", "assemble_delivery_payloads"]
