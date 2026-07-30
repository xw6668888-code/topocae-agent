"""Regenerate the formal 7200 set through one canonical fail-closed entrypoint."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
from typing import Any

import generate_p1_c_clean_context_7200 as protocol
from p1_c import all_features_h_c2_v3 as h_c2_generator
from p1_c import formal_generator_v7 as generator
from p1_c import formal_plan_v6 as plan_revision
from p1_c import rib_web_v5
from p1_c import supported_wall_families_v3


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output_generation_p1_c_clean_context_7200_strict_concept_r2"
PLAN_PATH = OUTPUT / "manifests" / "generation_plan.jsonl"
H_C1_CSV = (
    ROOT
    / "output_generation_p1_c1_strict_concept_hosts_pilot_96_r2"
    / "human_review_canonical_24.csv"
)
H_C2_CSV = (
    ROOT
    / "output_generation_p1_c2_all_class_canonical_36_strict_concept_r3"
    / "human_review_canonical_36.csv"
)
H_C1_AUTOMATIC = H_C1_CSV.parent / "automatic_audit_120.json"
H_C2_AUTOMATIC = H_C2_CSV.parent / "automatic_audit_36.json"
INVALIDATED_DATASET = ROOT / "output_generation_p1_c_clean_context_7200"

_ORIGINAL_PREPARE = protocol.prepare
_ORIGINAL_AUDIT_GENERATED_SAMPLE = protocol._audit_generated_sample
_ORIGINAL_GENERATE_ONE = protocol._generate_one
_ORIGINAL_FINALIZE = protocol.finalize
_ORIGINAL_VALIDATE = protocol.validate
_ORIGINAL_VERIFY_PREPARED = protocol._verify_prepared


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _verify_gate_artifacts(
    csv_path: Path,
    automatic_path: Path,
    *,
    expected_rows: int,
    expected_generator_revision: str,
    always_pass: tuple[str, ...],
    conditional_pass: tuple[str, ...],
    record_key: str,
    require_strict_contract: bool,
) -> dict[str, Any]:
    gate = protocol._verify_gate(
        csv_path,
        expected_rows=expected_rows,
        always_pass=always_pass,
        conditional_pass=conditional_pass,
    )
    rows = _read_csv(csv_path)
    row_by_step = {row["step_path"].replace("\\", "/"): row for row in rows}
    for relative, row in row_by_step.items():
        step_path = ROOT / Path(relative)
        if not step_path.is_file():
            raise RuntimeError(f"Human Gate STEP missing: {step_path}")
        if protocol.sha256_file(step_path).lower() != row["step_sha256"].strip().lower():
            raise RuntimeError(f"Human Gate STEP hash mismatch: {step_path}")

    automatic = json.loads(automatic_path.read_text(encoding="utf-8"))
    records = automatic[record_key]
    if automatic.get("automatic_pass") is not True or len(records) != expected_rows:
        raise RuntimeError(f"Human Gate automatic audit incomplete: {automatic_path}")
    if automatic.get("generator_revision") != expected_generator_revision:
        raise RuntimeError(f"Human Gate generator revision mismatch: {automatic_path}")
    for record in records:
        relative_step = str(record["step_path"]).replace("\\", "/")
        if relative_step not in row_by_step:
            raise RuntimeError(f"automatic record was not reviewed: {relative_step}")
        row = row_by_step[relative_step]
        if record["step_sha256"].lower() != row["step_sha256"].strip().lower():
            raise RuntimeError(f"automatic/review STEP hash mismatch: {relative_step}")
        graph_path = ROOT / Path(str(record["graph_path"]))
        if not graph_path.is_file():
            raise RuntimeError(f"Human Gate Graph v2 missing: {graph_path}")
        if protocol.sha256_file(graph_path).lower() != record["graph_sha256"].lower():
            raise RuntimeError(f"Human Gate Graph v2 hash mismatch: {graph_path}")
        if record.get("audit", {}).get("passed") is not True:
            raise RuntimeError(f"Human Gate automatic record failed: {relative_step}")
        if require_strict_contract:
            contract = record.get("parameters", {}).get(
                "strict_generation_contract_audit", {}
            )
            if contract.get("passed") is not True:
                raise RuntimeError(f"Human Gate strict contract failed: {relative_step}")
    return {
        **gate,
        "automatic_audit_path": automatic_path.relative_to(ROOT).as_posix(),
        "automatic_audit_sha256": protocol.sha256_file(automatic_path),
        "step_hashes_verified": expected_rows,
        "graph_v2_hashes_verified": expected_rows,
        "strict_contract_verified": require_strict_contract,
        "generator_revision": expected_generator_revision,
    }


def verify_human_gates() -> dict[str, Any]:
    h_c1 = _verify_gate_artifacts(
        H_C1_CSV,
        H_C1_AUTOMATIC,
        expected_rows=24,
        expected_generator_revision=rib_web_v5.GENERATOR_REVISION,
        always_pass=("single_solid", "clean_host", "feature_concept_valid", "decision"),
        conditional_pass=(),
        record_key="canonical_records",
        require_strict_contract=False,
    )
    h_c2 = _verify_gate_artifacts(
        H_C2_CSV,
        H_C2_AUTOMATIC,
        expected_rows=36,
        expected_generator_revision=h_c2_generator.GENERATOR_REVISION,
        always_pass=("target_valid", "clean_host", "decision"),
        conditional_pass=("secondary_valid", "composition_valid"),
        record_key="records",
        require_strict_contract=True,
    )
    return {"H-C1": h_c1, "H-C2": h_c2, "all_pass": True}


def prepare(output: Path) -> dict[str, Any]:
    payload = _ORIGINAL_PREPARE(output)
    config_path = output / "manifests" / "generation_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(
        {
            "schema_version": "topocae_p1_c_clean_context_strict_generation_config_2",
            "canonical_generation_entrypoint": generator.CANONICAL_ENTRYPOINT,
            "authoritative_slot_kernel_revision": generator.SLOT_KERNEL_REVISION,
            "supported_wall_adapter_revision": generator.SUPPORTED_WALL_ADAPTER_REVISION,
            "rib_web_generator_revision": generator.RIB_WEB_GENERATOR_REVISION,
            "strict_generation_contract_schema": (
                "topocae_p1_c_strict_generation_contract_audit_1"
            ),
            "invalidated_predecessor_dataset": INVALIDATED_DATASET.relative_to(ROOT).as_posix(),
            "invalidated_predecessor_records_reused": 0,
        }
    )
    config["contracts"].update(
        {
            "canonical_entrypoint_checked_per_sample": True,
            "strict_generation_contract_checked_per_sample": True,
            "human_gate_step_and_graph_hashes_verified": True,
            "legacy_semantic_relabel_permitted": False,
            "legacy_planform_slot_tool_permitted": False,
            "invalidated_formal_dataset_reused": False,
        }
    )
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    payload["generation_config_sha256"] = protocol.sha256_file(config_path)
    payload["strict_fail_closed"] = True
    return payload


def _audit_generated_sample(
    planned: Any,
    result: Any,
    shape: Any,
    labels: list[dict[str, Any]],
    graph: dict[str, Any],
    step_report: dict[str, Any],
    transform_audit: dict[str, Any],
) -> dict[str, Any]:
    payload = _ORIGINAL_AUDIT_GENERATED_SAMPLE(
        planned, result, shape, labels, graph, step_report, transform_audit
    )
    parameters = result.parameters
    contract = parameters.get("strict_generation_contract_audit", {})
    strict_checks = {
        "strict_generation_contract": contract.get("passed") is True,
        "canonical_generation_entrypoint": (
            parameters.get("canonical_generation_entrypoint")
            == generator.CANONICAL_ENTRYPOINT
        ),
        "formal_generator_revision": (
            parameters.get("generator_revision") == generator.GENERATOR_REVISION
        ),
        "formal_plan_revision": (
            parameters.get("plan_revision") == plan_revision.PLAN_REVISION
        ),
        "authoritative_slot_kernel_revision": (
            parameters.get("authoritative_slot_kernel_revision")
            == generator.SLOT_KERNEL_REVISION
        ),
        "supported_wall_adapter_revision": (
            parameters.get("supported_wall_adapter_revision")
            == generator.SUPPORTED_WALL_ADAPTER_REVISION
        ),
        "rib_web_generator_revision": (
            parameters.get("rib_web_generator_revision")
            == generator.RIB_WEB_GENERATOR_REVISION
        ),
    }
    payload["checks"].update(strict_checks)
    payload["strict_generation_contract"] = contract
    payload["formal_entrypoint_audit"] = {
        "passed": all(strict_checks.values()),
        "checks": strict_checks,
    }
    payload["passed"] = all(payload["checks"].values())
    payload["failed_checks"] = [
        name for name, passed in payload["checks"].items() if not passed
    ]
    return payload


def _generate_one(planned: Any, output: Path) -> dict[str, Any]:
    record = _ORIGINAL_GENERATE_ONE(planned, output)
    graph_path = output / record["graph_relative_path"]
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    audit = graph["p1_c_automatic_audit"]
    record["strict_generation_contract_audit"] = audit[
        "strict_generation_contract"
    ]
    record["formal_entrypoint_audit"] = audit["formal_entrypoint_audit"]
    return record


def _strict_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    contract_pass = sum(
        record.get("strict_generation_contract_audit", {}).get("passed") is True
        for record in records
    )
    entrypoint_pass = sum(
        record.get("formal_entrypoint_audit", {}).get("passed") is True
        for record in records
    )
    primary_counts = Counter(record["primary_target"] for record in records)
    payload = {
        "schema_version": "topocae_p1_c_formal_strict_generation_audit_1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(records),
        "strict_contract_pass_count": contract_pass,
        "formal_entrypoint_pass_count": entrypoint_pass,
        "primary_counts": dict(sorted(primary_counts.items())),
        "canonical_generation_entrypoint": generator.CANONICAL_ENTRYPOINT,
        "generator_revision": generator.GENERATOR_REVISION,
        "plan_revision": plan_revision.PLAN_REVISION,
        "slot_kernel_revision": generator.SLOT_KERNEL_REVISION,
        "supported_wall_adapter_revision": generator.SUPPORTED_WALL_ADAPTER_REVISION,
        "rib_web_generator_revision": generator.RIB_WEB_GENERATOR_REVISION,
        "semantic_relabel_permitted": False,
        "invalidated_predecessor_records_reused": 0,
    }
    payload["passed"] = (
        len(records) == 7200
        and contract_pass == len(records)
        and entrypoint_pass == len(records)
        and len(primary_counts) == 18
        and all(count == 400 for count in primary_counts.values())
    )
    return payload


def finalize(output: Path) -> dict[str, Any]:
    _ORIGINAL_FINALIZE(output)
    records = protocol._load_records(output)
    strict = _strict_report(records)
    if not strict["passed"]:
        raise RuntimeError(f"formal strict audit failed: {strict}")
    strict_path = output / "reports" / "strict_generation_contract_audit.json"
    strict_path.write_text(
        json.dumps(strict, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    meta_path = output / "dataset_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["strict_generation_contract_audit"] = strict
    meta["supersedes_invalidated_dataset"] = INVALIDATED_DATASET.relative_to(ROOT).as_posix()
    meta["automatic_pass"] = bool(meta["automatic_pass"] and strict["passed"])
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    card_path = output / "DATASET_CARD.zh-CN.md"
    strict_section = (
        "\n## 严格构造与防回归合同\n\n"
        f"- 正式唯一入口：`{generator.CANONICAL_ENTRYPOINT}`。\n"
        f"- 每样本严格合同：{strict['strict_contract_pass_count']}/7200 PASS。\n"
        f"- 每样本入口依赖合同：{strict['formal_entrypoint_pass_count']}/7200 PASS。\n"
        "- 旧错误 7200 数据仅保留为失败证据，复用记录数：0。\n"
    )
    card_path.write_text(
        protocol._dataset_card(meta) + strict_section, encoding="utf-8"
    )
    artifact_path = output / "manifests" / "artifact_manifest.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["dataset_meta_sha256"] = protocol.sha256_file(meta_path)
    artifact["dataset_card_sha256"] = protocol.sha256_file(card_path)
    artifact["strict_generation_contract_audit_sha256"] = protocol.sha256_file(
        strict_path
    )
    artifact_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return meta


def validate(output: Path) -> dict[str, Any]:
    payload = _ORIGINAL_VALIDATE(output)
    records = protocol._load_records(output)
    strict = _strict_report(records)
    stored_path = output / "reports" / "strict_generation_contract_audit.json"
    stored = json.loads(stored_path.read_text(encoding="utf-8"))
    extra_checks = {
        "strict_contract_7200_of_7200": strict["strict_contract_pass_count"] == 7200,
        "canonical_entrypoint_7200_of_7200": strict["formal_entrypoint_pass_count"] == 7200,
        "stored_strict_report_passed": stored.get("passed") is True,
        "strict_report_revision_matches": (
            stored.get("generator_revision") == generator.GENERATOR_REVISION
            and stored.get("plan_revision") == plan_revision.PLAN_REVISION
        ),
    }
    payload["checks"].update(extra_checks)
    payload["strict_generation_contract"] = strict
    payload["passed"] = all(payload["checks"].values())
    return payload


def _verify_prepared(output: Path) -> None:
    _ORIGINAL_VERIFY_PREPARED(output)
    config = json.loads(
        (output / "manifests" / "generation_config.json").read_text(
            encoding="utf-8"
        )
    )
    expected = {
        "plan_revision": plan_revision.PLAN_REVISION,
        "generator_revision": generator.GENERATOR_REVISION,
        "canonical_generation_entrypoint": generator.CANONICAL_ENTRYPOINT,
        "authoritative_slot_kernel_revision": generator.SLOT_KERNEL_REVISION,
        "supported_wall_adapter_revision": generator.SUPPORTED_WALL_ADAPTER_REVISION,
        "rib_web_generator_revision": generator.RIB_WEB_GENERATOR_REVISION,
        "invalidated_predecessor_records_reused": 0,
    }
    differences = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if differences:
        raise RuntimeError(f"strict prepared config mismatch: {differences}")
    gates = verify_human_gates()
    frozen = json.loads(
        (output / "manifests" / "human_gate_freeze.json").read_text(
            encoding="utf-8"
        )
    )
    for gate in ("H-C1", "H-C2"):
        if frozen[gate]["sha256"] != gates[gate]["sha256"]:
            raise RuntimeError(f"frozen {gate} review hash changed")


protocol.OUTPUT = OUTPUT
protocol.PLAN_PATH = PLAN_PATH
protocol.H_C1_CSV = H_C1_CSV
protocol.H_C2_CSV = H_C2_CSV
protocol.GENERATOR_REVISION = generator.GENERATOR_REVISION
protocol.build_planned_result = generator.build_planned_result
protocol.transformed_shape_and_audit = generator.transformed_shape_and_audit
protocol.PLAN_REVISION = plan_revision.PLAN_REVISION
protocol.CONTEXT_QUOTAS = plan_revision.CONTEXT_QUOTAS
protocol.PlannedSample = plan_revision.PlannedSample
protocol.audit_plan = plan_revision.audit_plan
protocol.build_plan = plan_revision.build_plan
protocol.load_plan = plan_revision.load_plan
protocol.sha256_file = plan_revision.sha256_file
protocol.write_plan = plan_revision.write_plan
protocol.verify_human_gates = verify_human_gates
protocol.prepare = prepare
protocol._audit_generated_sample = _audit_generated_sample
protocol._generate_one = _generate_one
protocol.finalize = finalize
protocol.validate = validate
protocol._verify_prepared = _verify_prepared


def main() -> int:
    return protocol.main()


if __name__ == "__main__":
    raise SystemExit(main())
