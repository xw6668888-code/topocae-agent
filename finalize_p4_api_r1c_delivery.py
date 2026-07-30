from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "reports" / "p4_api_r1c_structured_v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name: str):
    return json.loads((REPORT / name).read_text(encoding="utf-8"))


def main() -> None:
    acceptance = load("acceptance.json")
    train = load("train_metrics.json")
    val = load("val_metrics.json")
    runtime = load("runtime_cost_audit.json")
    failures = load("failure_stratification.json")
    train_l = train["language_cae_ir"]
    val_l = val["language_cae_ir"]
    train_g = train["grounding"]
    val_g = val["grounding"]
    failure_categories = failures["counts"]["failure_category"]
    content = f'''# P4-API-R1C Structured Prompt 通用修订验收

## 状态

`{acceptance["status"]}`

已完成且只完成一次 Structured prompt 通用修订。结果低于 P4 门槛，未自动进行第二轮修订。

## Prompt 冻结

- 旧 SHA-256：`2f6285afc985afe94295df80e44107dc72692d118ca5410a9a74401f964dcc2b`
- 新 SHA-256：`e289c378c12b225da467310dada3b093bd13b2937736d7434f3039f8892344da`
- provider、model、schema、validator、grounder、Direct prompt 和修复策略：未变

## 有界执行

- train：确定性 50 条
- val：确定性 50 条
- backend 请求：{runtime["backend_request_count"]} / {runtime["backend_request_cap"]}
- HTTP 重试：{runtime["http_retry_count"]}
- schema 修复：{runtime["schema_retry_count"]}，重试率 {runtime["schema_retry_rate"]:.2%}
- input/output tokens：{runtime["input_tokens"]} / {runtime["output_tokens"]}
- 费用审计：{runtime["accounted_cost_cny"]:.8f} / {runtime["cost_cap_cny"]:.2f} CNY
- 实际响应模型：`{runtime["actual_model_ids"][0]}`；100 条 finish reason 均为 `stop`

## 指标

| 指标 | train | val | val 门槛 |
|---|---:|---:|---:|
| 首轮 schema-valid | {train_l["schema_valid_rate_first_round"]:.2%} | {val_l["schema_valid_rate_first_round"]:.2%} | - |
| 修复后 schema-valid | {train_l["schema_valid_rate_after_retry"]:.2%} | {val_l["schema_valid_rate_after_retry"]:.2%} | >=99% |
| status accuracy | {train_l["status_accuracy"]:.2%} | {val_l["status_accuracy"]:.2%} | >=95% |
| CAE-IR canonical exact | {train_l["cae_ir_canonical_exact_match"]:.2%} | {val_l["cae_ir_canonical_exact_match"]:.2%} | - |
| slot Macro-F1 | {train_l["slot_macro_f1"]:.2%} | {val_l["slot_macro_f1"]:.2%} | >=95% |
| clarification/invalid Macro-F1 | {train_l["clarification_invalid_detection"]["macro_f1"]:.2%} | {val_l["clarification_invalid_detection"]["macro_f1"]:.2%} | >=85% |
| object selection accuracy | {train_g["object_selection_accuracy"]:.2%} | {val_g["object_selection_accuracy"]:.2%} | >=95% |
| exact face-set accuracy | {train_g["exact_face_set_accuracy"]:.2%} | {val_g["exact_face_set_accuracy"]:.2%} | >=90% |
| illegal/out-of-range ID rate | {train_g["illegal_out_of_range_id_rate"]:.2%} | {val_g["illegal_out_of_range_id_rate"]:.2%} | 0% |

## 失败分层

- parser error：{failure_categories.get("parser_error", 0)} / 100
- face-role expansion error：{failure_categories.get("face_role_expansion_error", 0)} / 100
- 主要 schema 路径：`/operations/0/parameters`、`/intent/operations/0/parameters`、`/operations/0/operation_type`、`/intent/operations/0/operation_type`、根路径 `/`
- val 语言层：en-US status accuracy {val["stratified"]["language"]["en-US"]["status_accuracy"]:.2%}；zh-CN {val["stratified"]["language"]["zh-CN"]["status_accuracy"]:.2%}
- val 难度层：L1/L2/L3 status accuracy 均为 0%；L4/L5 均为 33.33%

## 结论

三分支示例提高了“修复后 JSON schema 可接受率”，但普通 DeepSeek `json_object` 仍无法稳定执行完整 CAE-IR 结构和语义契约。当前结果不足以进入完整 train/val、Direct 或 formal test。下一步需要监督者决定是否另立 provider-profile / 服务端 strict schema 方案；本阶段不作该变更。

## 安全与范围

formal test 仍为 run=0、content_accessed=false。未保存 API Key、Bearer 值、账户余额、query、ground truth 或原始模型响应；未运行 Direct、P5、STEP 或 Abaqus。
'''
    (REPORT / "final_acceptance.zh-CN.md").write_text(content, encoding="utf-8", newline="\n")
    files = sorted(path for path in REPORT.iterdir() if path.is_file() and path.name != "delivery_manifest.json")
    manifest = {
        "schema_version": "topocae-p4-api-r1c-delivery-manifest/1",
        "assets": [
            {"path": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in files
        ],
    }
    (REPORT / "delivery_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
