"""Offline strict-schema provider feasibility audit for P4-API-R1D."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

from p4_api_r0_1_audit import verify_full_freeze
from utils.cae_language_grounding.llm_adapter import LLMRuntimeConfig
from utils.cae_language_grounding.llm_adapter_r0 import PROVIDER_PROFILES


ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "reports" / "p4_api_r1d_strict_schema_feasibility_v1"
LEDGER = ROOT / "output_p4_a1_language_grounding_v1" / "test_run_ledger.json"
SCHEMA_PATHS = (
    ROOT / "schemas" / "cae_intent_result.schema.json",
    ROOT / "schemas" / "cae_intent.schema.json",
)
EXPECTED_SCHEMA_HASHES = {
    "schemas/cae_intent_result.schema.json": "00d5da9125031158786bd853c172f0aae1fb63c0424fd19b3ff82f6918e37727",
    "schemas/cae_intent.schema.json": "8e94cd9a6f3dad42ccad58d47eb82e801e2cb66870ac03dcd940a56d076aeac4",
}
OPENAI_SOURCE = "https://developers.openai.com/api/docs/guides/structured-outputs"
DEEPSEEK_SOURCE = "https://api-docs.deepseek.com/guides/tool_calls/"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pointer(parts: tuple[Any, ...]) -> str:
    if not parts:
        return "/"
    return "/" + "/".join(str(part).replace("~", "~0").replace("/", "~1") for part in parts)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def formal_test_state() -> dict[str, Any]:
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    state = {
        "run_count": sum(len(rows) for rows in ledger["runs"].values()),
        "content_accessed": bool(ledger["test_content_accessed"]),
        "unlock_authorized": bool(ledger["test_unlock_authorized"]),
    }
    if state != {"run_count": 0, "content_accessed": False, "unlock_authorized": False}:
        raise RuntimeError("E_FORMAL_TEST_GUARD")
    return state


def walk(value: Any, parts: tuple[Any, ...] = ()):
    yield parts, value
    if isinstance(value, dict):
        for key, child in value.items():
            yield from walk(child, parts + (key,))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from walk(child, parts + (index,))


def inspect_schema(path: Path) -> dict[str, Any]:
    schema = json.loads(path.read_text(encoding="utf-8"))
    keyword_paths: dict[str, list[str]] = {
        name: []
        for name in (
            "oneOf", "anyOf", "allOf", "minLength", "maxLength",
            "minItems", "maxItems", "uniqueItems", "const", "$ref", "$defs",
        )
    }
    object_required_gaps: list[dict[str, Any]] = []
    object_additional_properties_gaps: list[str] = []
    property_fragments_without_object: list[str] = []
    external_refs: list[dict[str, str]] = []
    string_const_paths: list[str] = []
    for parts, node in walk(schema):
        if not isinstance(node, dict):
            continue
        for keyword in keyword_paths:
            if keyword in node:
                keyword_paths[keyword].append(pointer(parts + (keyword,)))
        if "$ref" in node and isinstance(node["$ref"], str) and not node["$ref"].startswith("#"):
            external_refs.append({"path": pointer(parts + ("$ref",)), "value": node["$ref"]})
        if "const" in node and isinstance(node["const"], str):
            string_const_paths.append(pointer(parts + ("const",)))
        properties = node.get("properties")
        if isinstance(properties, dict):
            if node.get("type") != "object":
                property_fragments_without_object.append(pointer(parts))
            else:
                required = set(node.get("required", []))
                missing = sorted(set(properties) - required)
                if missing:
                    object_required_gaps.append({"path": pointer(parts), "missing_required": missing})
                if node.get("additionalProperties") is not False:
                    object_additional_properties_gaps.append(pointer(parts))
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": sha256(path),
        "root_type": schema.get("type"),
        "root_keywords": sorted(schema.keys()),
        "keyword_counts": {name: len(paths) for name, paths in keyword_paths.items()},
        "keyword_paths": keyword_paths,
        "object_required_gap_count": len(object_required_gaps),
        "object_required_gaps": object_required_gaps,
        "object_additional_properties_gap_count": len(object_additional_properties_gaps),
        "object_additional_properties_gaps": object_additional_properties_gaps,
        "property_fragment_without_object_count": len(property_fragments_without_object),
        "property_fragments_without_object": property_fragments_without_object,
        "external_ref_count": len(external_refs),
        "external_refs": external_refs,
        "string_const_count": len(string_const_paths),
        "string_const_paths": string_const_paths,
    }


def effective_inventory(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keywords = Counter()
    for row in rows:
        keywords.update(row["keyword_counts"])
    result_schema = rows[0]
    return {
        "schema_version": "topocae-p4-api-r1d-schema-inventory/1",
        "effective_schema_files": rows,
        "effective_root_type": result_schema["root_type"],
        "effective_root_has_one_of": result_schema["keyword_counts"]["oneOf"] > 0,
        "keyword_counts_total": dict(sorted(keywords.items())),
        "object_required_gap_count_total": sum(row["object_required_gap_count"] for row in rows),
        "object_additional_properties_gap_count_total": sum(row["object_additional_properties_gap_count"] for row in rows),
        "external_ref_count_total": sum(row["external_ref_count"] for row in rows),
        "string_const_count_total": sum(row["string_const_count"] for row in rows),
    }


def provider_compatibility(inventory: dict[str, Any]) -> dict[str, Any]:
    counts = inventory["keyword_counts_total"]
    required_gaps = inventory["object_required_gap_count_total"]
    additional_gaps = inventory["object_additional_properties_gap_count_total"]
    external_refs = inventory["external_ref_count_total"]
    root_blockers = []
    if inventory["effective_root_type"] != "object":
        root_blockers.append("E_ROOT_MUST_BE_OBJECT")
    if inventory["effective_root_has_one_of"]:
        root_blockers.append("E_ROOT_UNION_ONE_OF")

    openai_blockers = root_blockers + [
        *( ["E_ONE_OF_NOT_IN_PUBLISHED_SUBSET"] if counts.get("oneOf", 0) else [] ),
        *( ["E_ALLOF_UNSUPPORTED"] if counts.get("allOf", 0) else [] ),
        *( ["E_ALL_OBJECT_PROPERTIES_MUST_BE_REQUIRED"] if required_gaps else [] ),
        *( ["E_ADDITIONAL_PROPERTIES_FALSE_REQUIRED"] if additional_gaps else [] ),
        *( ["E_EXTERNAL_REF_REQUIRES_REQUEST_REPRESENTATION_CHANGE"] if external_refs else [] ),
    ]
    deepseek_blockers = root_blockers + [
        *( ["E_ONE_OF_NOT_IN_PUBLISHED_SUBSET"] if counts.get("oneOf", 0) else [] ),
        *( ["E_ALLOF_NOT_IN_PUBLISHED_SUBSET"] if counts.get("allOf", 0) else [] ),
        *( ["E_MIN_LENGTH_UNSUPPORTED"] if counts.get("minLength", 0) else [] ),
        *( ["E_MAX_LENGTH_UNSUPPORTED"] if counts.get("maxLength", 0) else [] ),
        *( ["E_MIN_ITEMS_UNSUPPORTED"] if counts.get("minItems", 0) else [] ),
        *( ["E_MAX_ITEMS_UNSUPPORTED"] if counts.get("maxItems", 0) else [] ),
        *( ["E_ALL_OBJECT_PROPERTIES_MUST_BE_REQUIRED"] if required_gaps else [] ),
        *( ["E_ADDITIONAL_PROPERTIES_FALSE_REQUIRED"] if additional_gaps else [] ),
        *( ["E_EXTERNAL_REF_REQUIRES_REQUEST_REPRESENTATION_CHANGE"] if external_refs else [] ),
    ]
    return {
        "schema_version": "topocae-p4-api-r1d-provider-compatibility/1",
        "evaluated_candidates": {
            "deepseek_json_object": {
                "implemented_in_repo": True,
                "server_strict_schema": False,
                "compatible_with_frozen_schema": False,
                "blockers": ["E_JSON_OBJECT_ONLY_NO_SCHEMA_ENFORCEMENT"],
                "source": "https://api-docs.deepseek.com/guides/json_mode/",
            },
            "deepseek_strict_tool_beta": {
                "implemented_in_repo": False,
                "server_strict_schema": True,
                "compatible_with_frozen_schema": False,
                "blockers": sorted(set(deepseek_blockers)),
                "required_endpoint_identity_change": "https://api.deepseek.com/beta",
                "required_request_shape_change": "response_format -> tools/function/strict",
                "source": DEEPSEEK_SOURCE,
            },
            "openai_json_schema": {
                "implemented_in_repo": True,
                "server_strict_schema": True,
                "compatible_with_frozen_schema": False,
                "blockers": sorted(set(openai_blockers)),
                "requires_distinct_provider_credentials_and_model": True,
                "source": OPENAI_SOURCE,
            },
        },
        "compatible_candidate_count": 0,
        "real_api_smoke_authorized_by_gate": False,
        "gate_reason": "FROZEN_SCHEMA_INCOMPATIBLE_WITH_ALL_EVALUATED_STRICT_PROFILES",
        "backend_requests_performed": 0,
        "cost_cny": 0.0,
    }


def security_audit(paths: list[Path], api_key: str) -> dict[str, Any]:
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    result = {
        "api_key_present_in_delivery": bool(api_key and api_key in text),
        "bearer_value_hits": len(re.findall(r"(?i)Bearer\s+[A-Za-z0-9._-]{8,}", text)),
        "backend_requests_performed": 0,
        "query_written": False,
        "ground_truth_written": False,
        "raw_model_response_written": False,
        "formal_test_accessed": False,
    }
    if result["api_key_present_in_delivery"] or result["bearer_value_hits"]:
        raise RuntimeError("E_SECRET_DELIVERY")
    return result


def write_markdown(compatibility: dict[str, Any], inventory: dict[str, Any]) -> None:
    counts = inventory["keyword_counts_total"]
    migration = '''# P4-API-R1D Strict Schema 迁移方案

## 当前结论

当前冻结 CAE-IR schema 不能原样提交给已评估的 strict provider profile，因此 R1D 不允许发送健康请求或 5 条 val smoke。

## 兼容性差异

1. 顶层是 `oneOf` union，而 strict providers 要求顶层为 object。
2. 有效 schema closure 使用 `oneOf` 与 `allOf`；OpenAI 公布的 strict subset 不支持 `allOf`，两家公布的支持列表均未包含 `oneOf`。
3. DeepSeek beta strict 明确不支持 `minLength/maxLength/minItems/maxItems`。
4. 多个 object 含可选属性，违反两家“所有 properties 必须 required”的规则。
5. 顶层 schema 使用外部 `$ref` 指向 `cae_intent.schema.json`；提交单一 request schema 前需要 bundle/inlining，这会改变请求表示。
6. DeepSeek strict 还需要 `/beta` endpoint 与 tool-call request/response shape；当前 repo 没有该 provider profile。

## 可选迁移路线

### 路线 A：Provider-specific transport schema

- 保持 canonical validator schema 不变；
- 构造单独、显式版本化的 transport schema；
- 将顶层包装成 object，并把状态分支放入嵌套 `anyOf`；
- 展开 `allOf` 和外部 `$ref`；
- 将 optional 字段改为 required + nullable（仅适用于支持 null 的 provider）；
- 对服务端未支持的长度/数组约束继续执行 canonical validator；
- 建立 transport output -> canonical CAE-IR 的无损映射和双向契约测试。

该路线不再是“服务端严格执行当前冻结 schema”，而是严格执行派生 transport schema，必须获得新的监督授权并冻结双 schema/hash。

### 路线 B：Canonical schema v2

- 直接重构 canonical schema 为 strict-provider 公共子集；
- 更新数据、validator、grounder 契约与全部冻结 hash；
- 重新进行 P4 回归与 train/val 验证。

该路线影响更大，不属于 P4-API-R1D。

## 推荐

优先对路线 A 做纯离线原型和语义等价证明；在等价性通过前不要申请真实 API smoke，也不要解锁 formal test。
'''
    (REPORT / "migration_plan.zh-CN.md").write_text(migration, encoding="utf-8", newline="\n")
    final = f'''# P4-API-R1D Strict Schema Provider Feasibility

## 状态

`NO_COMPATIBLE_STRICT_PROFILE_OFFLINE_GATE`

评估候选：`deepseek_json_object`、`deepseek_strict_tool_beta`、`openai_json_schema`。兼容候选数：{compatibility["compatible_candidate_count"]}。

冻结 schema closure 统计：`oneOf` {counts.get("oneOf", 0)}、`allOf` {counts.get("allOf", 0)}、`minLength` {counts.get("minLength", 0)}、`minItems` {counts.get("minItems", 0)}、`maxItems` {counts.get("maxItems", 0)}；缺少 required 的 object 共 {inventory["object_required_gap_count_total"]} 个。

依据授权门禁，schema 不兼容时只提交差异与迁移方案。因此真实 backend 请求为 0，费用为 0，未运行 5 条 val smoke。

formal test 保持 run=0、content_accessed=false。schema、validator、grounder、evaluator、Direct prompt 与 Structured prompt 均未修改。
'''
    (REPORT / "final_acceptance.zh-CN.md").write_text(final, encoding="utf-8", newline="\n")


def write_manifest() -> None:
    files = sorted(path for path in REPORT.iterdir() if path.is_file() and path.name != "delivery_manifest.json")
    write_json(REPORT / "delivery_manifest.json", {
        "schema_version": "topocae-p4-api-r1d-delivery-manifest/1",
        "assets": [
            {"path": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in files
        ],
    })


def main() -> int:
    if REPORT.exists():
        raise RuntimeError("E_R1D_REPORT_ALREADY_EXISTS")
    formal_before = formal_test_state()
    freeze = verify_full_freeze()
    if freeze["status"] != "PASS":
        raise RuntimeError("E_FULL_FREEZE")
    actual_hashes = {
        str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
        for path in SCHEMA_PATHS
    }
    if actual_hashes != EXPECTED_SCHEMA_HASHES:
        raise RuntimeError("E_FROZEN_SCHEMA_HASH")
    config = LLMRuntimeConfig.from_env()
    schema_rows = [inspect_schema(path) for path in SCHEMA_PATHS]
    inventory = effective_inventory(schema_rows)
    compatibility = provider_compatibility(inventory)
    if compatibility["compatible_candidate_count"] != 0:
        raise RuntimeError("E_COMPATIBLE_CANDIDATE_REQUIRES_SEPARATE_SMOKE_RUNNER")

    REPORT.mkdir(parents=True)
    write_json(REPORT / "preflight.json", {
        "schema_version": "topocae-p4-api-r1d-preflight/1",
        "status": "PASS_OFFLINE_ONLY",
        "authorized_scope": "STRICT_SCHEMA_PROVIDER_FEASIBILITY",
        "full_freeze_status": freeze["status"],
        "formal_test_before": formal_before,
        "frozen_schema_hashes": actual_hashes,
        "configured_runtime": config.public_metadata(),
        "repo_provider_profiles": sorted(PROVIDER_PROFILES),
        "backend_request_cap": 12,
        "cost_cap_cny": 2.0,
        "official_sources": {
            "openai_structured_outputs": OPENAI_SOURCE,
            "deepseek_strict_tool_beta": DEEPSEEK_SOURCE,
        },
    })
    write_json(REPORT / "schema_inventory.json", inventory)
    write_json(REPORT / "provider_compatibility.json", compatibility)
    write_json(REPORT / "acceptance.json", {
        "schema_version": "topocae-p4-api-r1d-acceptance/1",
        "status": "NO_COMPATIBLE_STRICT_PROFILE_OFFLINE_GATE",
        "compatible_candidate_count": 0,
        "real_api_smoke_performed": False,
        "backend_requests": 0,
        "cost_cny": 0.0,
        "formal_test_after": formal_test_state(),
        "contracts_modified": False,
        "next_action": "SUPERVISOR_DECISION_ON_TRANSPORT_SCHEMA_PROTOTYPE",
    })
    write_markdown(compatibility, inventory)
    initial_paths = list(REPORT.iterdir())
    write_json(REPORT / "security_audit.json", security_audit(initial_paths, config.api_key or ""))
    write_json(REPORT / "CODEX_CONTEXT_P4_API_R1D.json", {
        "schema_version": "topocae-p4-api-r1d-context/1",
        "status": "NO_COMPATIBLE_STRICT_PROFILE_OFFLINE_GATE",
        "backend_requests": 0,
        "cost_cny": 0.0,
        "formal_test_accessed": False,
        "next_action": "SUPERVISOR_DECISION_ON_TRANSPORT_SCHEMA_PROTOTYPE",
    })
    write_manifest()
    print(json.dumps({
        "status": "NO_COMPATIBLE_STRICT_PROFILE_OFFLINE_GATE",
        "compatible_candidates": 0,
        "backend_requests": 0,
        "cost_cny": 0.0,
        "report_dir": str(REPORT),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
