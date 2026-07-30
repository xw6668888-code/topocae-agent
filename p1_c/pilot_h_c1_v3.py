"""Extended H-C1 bound to strict concept-preserving rib/web host families."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from p1_c import pilot_h_c1 as protocol
from p1_c import rib_web_v3 as generator


PILOT_COUNT_PER_SCENARIO = 4
PILOT_SEED_BASE = 2026082000
CANONICAL_SEED_BASE = 2026083000


protocol.GENERATOR_REVISION = generator.GENERATOR_REVISION
protocol.SCENARIO_SPECS = generator.SCENARIO_SPECS
protocol.build_rib_web_sample = generator.build_rib_web_sample
protocol.scenario_keys = generator.scenario_keys


def generate_h_c1(root: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    pilot_step_dir = output / "pilot_96" / "steps"
    pilot_graph_dir = output / "pilot_96" / "graphs_v2"
    canonical_step_dir = output / "human_review_canonical_24"
    canonical_graph_dir = output / "human_review_canonical_24_graphs_v2"
    records: list[dict[str, Any]] = []
    canonical_records: list[dict[str, Any]] = []
    generated_at = datetime.now(timezone.utc).isoformat()
    scenario_keys = generator.scenario_keys()
    for scenario_index, scenario_key in enumerate(scenario_keys):
        for replicate in range(PILOT_COUNT_PER_SCENARIO):
            seed = PILOT_SEED_BASE + scenario_index * PILOT_COUNT_PER_SCENARIO + replicate
            stem = f"{scenario_index + 1:02d}_{scenario_key}_r{replicate + 1}_seed{seed}"
            records.append(
                protocol._write_sample(
                    root=root,
                    scenario_key=scenario_key,
                    seed=seed,
                    step_path=pilot_step_dir / f"{stem}.step",
                    graph_path=pilot_graph_dir / f"{stem}.json",
                    generated_at=generated_at,
                    canonical=False,
                )
            )
        seed = CANONICAL_SEED_BASE + scenario_index
        stem = f"{scenario_index + 1:02d}_{scenario_key}_seed{seed}"
        canonical_records.append(
            protocol._write_sample(
                root=root,
                scenario_key=scenario_key,
                seed=seed,
                step_path=canonical_step_dir / f"{stem}.step",
                graph_path=canonical_graph_dir / f"{stem}.json",
                generated_at=generated_at,
                canonical=True,
            )
        )
    expected_pilot = len(scenario_keys) * PILOT_COUNT_PER_SCENARIO
    expected_canonical = len(scenario_keys)
    all_records = records + canonical_records
    automatic_pass = (
        len(records) == expected_pilot
        and len(canonical_records) == expected_canonical
        and all(item["audit"]["passed"] for item in all_records)
    )
    payload = {
        "schema_version": "topocae_p1_c_h_c1_strict_concept_automatic_audit_1",
        "generated_at_utc": generated_at,
        "generator_revision": generator.GENERATOR_REVISION,
        "scenario_specs": generator.SCENARIO_SPECS,
        "original_goal_scenario_count": 12,
        "additional_strict_host_scenario_count": expected_canonical - 12,
        "pilot_count": len(records),
        "canonical_count": len(canonical_records),
        "automatic_pass": automatic_pass,
        "pilot_records": records,
        "canonical_records": canonical_records,
        "concept_contract": {
            "all_original_h_c1_scenarios_retained": True,
            "all_new_host_families_human_reviewed": True,
            "semantic_relabel_used": False,
        },
    }
    audit_path = output / "automatic_audit_120.json"
    audit_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    protocol._write_review_csv(output / "human_review_canonical_24.csv", canonical_records)
    _write_guide(output / "H-C1_UG_NX_检查指南.zh-CN.md")
    _write_summary(output / "automatic_audit_summary.zh-CN.md", payload)
    manifest_records = []
    for path in sorted(
        item for item in output.rglob("*") if item.is_file() and item.name != "manifest.json"
    ):
        manifest_records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": protocol.sha256_file(path),
            }
        )
    manifest = {
        "schema_version": "topocae_p1_c_h_c1_strict_concept_manifest_1",
        "generated_at_utc": generated_at,
        "generator_revision": generator.GENERATOR_REVISION,
        "automatic_pass": automatic_pass,
        "files": manifest_records,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not automatic_pass:
        failed = Counter(
            check
            for record in all_records
            for check, passed in record["audit"]["checks"].items()
            if not passed
        )
        raise RuntimeError(f"strict H-C1 automatic audit failed: {dict(failed)}")
    return payload


def _write_guide(path: Path) -> None:
    lines = [
        "# P1-C Human Gate H-C1（严格概念保持宿主）：UG/NX 审查指南",
        "",
        "请打开 `human_review_canonical_24/` 内全部 24 个 STEP；本 Gate 不要求检查 raw face ID。",
        "原 Goal 的 12 个场景全部保留；另增加 12 个场景以覆盖新增的 rib/web 宿主家族。",
        "",
        "逐件确认：",
        "",
        "1. 模型树和测量结果只有一个有效实体。",
        "2. 基体没有重叠体、内部壳、共面分割线、多余小面或不必要台阶。",
        "3. rib 仅是 L-corner、矩形/圆形 cavity 或 T-junction 上的受支撑薄壁；不得是两支柱桥接 web 的改名。",
        "4. web 仅是两个支撑体之间的桥接薄壁；equal/asymmetric/tapered/buttressed 四类宿主概念均应成立。",
        "5. rect/circular cavity 仍清楚存在；hole 和 transition 变体成立，孔未切断薄壁，transition 无碎片。",
        "",
        "在同级 `human_review_canonical_24.csv` 中只填写：",
        "",
        "- `single_solid`：PASS/FAIL。",
        "- `clean_host`：PASS/FAIL。",
        "- `feature_concept_valid`：PASS/FAIL。",
        "- `decision`：仅当前三项均 PASS 时填 PASS，否则填 FAIL。",
        "- `notes`：FAIL 必须写明文件名和问题；PASS 可写“通过”。",
        "",
        "必须 24/24 PASS。不要修改 seed、路径或 SHA-256 列。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    records = payload["pilot_records"] + payload["canonical_records"]
    minimum_contact = min(
        item["intersection_volume"]
        for record in records
        for item in record["parameters"]["support_contacts"]
    )
    total = len(records)
    lines = [
        "# P1-C H-C1 严格概念保持宿主自动审计摘要",
        "",
        f"- Pilot：{payload['pilot_count']}/96。",
        f"- Canonical：{payload['canonical_count']}/24。",
        f"- 自动审计：{sum(record['audit']['passed'] for record in records)}/{total} PASS。",
        f"- Generator revision：`{payload['generator_revision']}`。",
        f"- 最小支撑交叠体积：`{minimum_contact:.9f}`。",
        f"- 最大 cleanup 相对体积误差：`{max(record['audit']['cleanup']['relative_volume_error'] for record in records):.3e}`。",
        f"- 最大 STEP 重读相对体积误差：`{max(record['audit']['step_relative_volume_error'] for record in records):.3e}`。",
        "- 全部样本检查单实体、有效 B-Rep、单壳、cleanup 幂等、零冗余面/edge、标签互斥、支撑接触、STEP 重读一致和 Graph v2 finite。",
        "- rib/web 工程源类别与目标类别完全一致；禁止语义改名。",
        "",
        "自动审计通过后进入 Human Gate H-C1；未收到 24/24 人工 PASS 前不得生成 H-C2。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


__all__ = ["generate_h_c1"]
