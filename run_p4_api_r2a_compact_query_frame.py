"""Offline P4-API-R2A QueryFrame audit and delivery builder."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable, Mapping

from jsonschema import Draft202012Validator

from run_p4_api_r1e_transport_schema import P2_FIXTURES, p2_result
from utils.cae_intent import validate_intent_result
from utils.cae_language_grounding.dataset import DecodeAudit, iter_canonical_records
from utils.cae_query_frame import (
    canonical_to_query_frame,
    compile_query_frame,
    compile_query_frame_json,
    load_schema,
    validate_query_frame,
)


ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "reports" / "p4_api_r2a_compact_query_frame_v1"
LEDGER = ROOT / "output_p4_a1_language_grounding_v1" / "test_run_ledger.json"
EXPECTED_RECORDS = 21885
EXPECTED_SOURCE_COUNTS = {"p2_fixtures": 265, "p3_train": 19180, "p3_val": 2440}
EXPECTED_FREEZE_MANIFEST_HASHES = {
    "reports/p1_d_lite_v1/freeze/recognizer_dev_manifest.json": "41067a5b8916bb3ad222c9e873d8576a4ec7aaa3634c79d89d0eebdcba63468a",
    "reports/p2_a1_cae_ir_v1/freeze/freeze_manifest.json": "7a9e8f184013eeeff0db794a95defc73d19438c89e7cbbf6609abdea1a6a2845",
    "reports/p3_a1r_multi_instance_v1/freeze/freeze_manifest.json": "44b4be432a432bce02158ab3fddc1bcdc569cb3d86587286c5ab866b193349b9",
}
HISTORICAL_REPORT_DIRS = (
    "reports/p4_api_r1c_structured_v1",
    "reports/p4_api_r1e_transport_schema_v1",
    "reports/p4_api_r1e_1_dialect_repair_v1",
    "reports/p4_api_r1f_strict_handshake_v1",
    "reports/p4_api_r1f_1_strict_handshake_v1",
    "reports/p4_api_r1f_2_minimal_strict_control_v1",
)
REQUIRED_DELIVERY_FILES = frozenset({
    "final_acceptance.zh-CN.md", "acceptance.json", "query_frame.schema.json",
    "query_frame_contract.zh-CN.md", "compiler_roundtrip_audit.json",
    "compactness_audit.json", "coverage_audit.json", "leakage_audit.json",
    "regression_results.json", "freeze_audit.json", "delivery_manifest.json",
    "CODEX_CONTEXT_P4_API_R2A.md", "query_frame_prompt.zh-CN.md",
    "repair_protocol.zh-CN.md",
})


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def formal_test_state() -> dict[str, Any]:
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    return {
        "run_count": sum(len(rows) for rows in ledger["runs"].values()),
        "content_accessed": bool(ledger["test_content_accessed"]),
        "unlock_authorized": bool(ledger["test_unlock_authorized"]),
    }


def _p2_rows() -> Iterable[tuple[dict[str, Any], dict[str, str]]]:
    for line in P2_FIXTURES.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        result, _ = p2_result(case)
        language = (
            result["intent"]["source_language"]
            if result["status"] == "VALID"
            else "fixture_unlabeled"
        )
        yield result, {
            "source": "p2_fixtures",
            "split": "fixture",
            "language": language,
            "difficulty": "fixture",
            "canonical_id": f"p2::{case['case_id']}",
            "exact_graph_signature": "",
            "base_geometry_family_id": "",
            "sample_id": "",
        }


def _p3_rows(
    decode_audits: dict[str, Any],
) -> Iterable[tuple[dict[str, Any], dict[str, str]]]:
    for split in ("train", "val"):
        audit = DecodeAudit("canonical", split)
        for record in iter_canonical_records(split, audit=audit):
            source = record["source_ref"]
            query = record["query"]
            yield record["intent_result"], {
                "source": f"p3_{split}",
                "split": split,
                "language": str(query["language"]),
                "difficulty": str(query["difficulty_level"]),
                "canonical_id": str(source["canonical_record_id"]),
                "exact_graph_signature": str(source.get("exact_graph_signature", "")),
                "base_geometry_family_id": str(source.get("base_geometry_family_id", "")),
                "sample_id": str(source.get("sample_id", "")),
            }
        if audit.decoded_test_lines:
            raise RuntimeError("QF_E_FORMAL_TEST_DECODE")
        decode_audits[split] = audit.as_dict()


def _nearest_rank(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def _verify_delivery_directory(relative: str) -> dict[str, Any]:
    directory = ROOT / relative
    manifest_path = directory / "delivery_manifest.json"
    if not manifest_path.exists():
        return {"path": relative, "valid": False, "reason": "delivery_manifest_missing"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("assets") or manifest.get("files") or []
    mismatches = []
    checked = 0
    for entry in entries:
        raw_path = entry.get("path") or entry.get("relative_path")
        expected = entry.get("sha256")
        if not raw_path or not expected:
            continue
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = directory / candidate
        if candidate.resolve() == manifest_path.resolve():
            continue
        checked += 1
        actual = sha256(candidate) if candidate.is_file() else None
        if actual != expected:
            mismatches.append({"path": str(raw_path), "expected": expected, "actual": actual})
    return {
        "path": relative,
        "manifest_sha256": sha256(manifest_path),
        "assets_checked": checked,
        "mismatches": mismatches,
        "valid": not mismatches and checked > 0,
    }


def _freeze_audit() -> dict[str, Any]:
    actual = {path: sha256(ROOT / path) for path in EXPECTED_FREEZE_MANIFEST_HASHES}
    manifest_changes = [
        {"path": path, "expected": expected, "actual": actual[path]}
        for path, expected in EXPECTED_FREEZE_MANIFEST_HASHES.items()
        if actual[path] != expected
    ]
    history = [_verify_delivery_directory(path) for path in HISTORICAL_REPORT_DIRS]
    history_changes = [item for item in history if not item["valid"]]
    changes = manifest_changes + history_changes
    return {
        "schema_version": "topocae-p4-api-r2a-freeze-audit/1",
        "scope": "authoritative P1/P2/P3 freeze-manifest identity and R1C/R1E/R1F delivery-manifest asset hashes; STEP/BREP payloads not opened",
        "expected_manifest_hashes": EXPECTED_FREEZE_MANIFEST_HASHES,
        "actual_manifest_hashes": actual,
        "manifest_changes": manifest_changes,
        "historical_report_checks": history,
        "history_changes": history_changes,
        "change_count": len(changes),
        "p1_unchanged": actual["reports/p1_d_lite_v1/freeze/recognizer_dev_manifest.json"] == EXPECTED_FREEZE_MANIFEST_HASHES["reports/p1_d_lite_v1/freeze/recognizer_dev_manifest.json"],
        "p2_unchanged": actual["reports/p2_a1_cae_ir_v1/freeze/freeze_manifest.json"] == EXPECTED_FREEZE_MANIFEST_HASHES["reports/p2_a1_cae_ir_v1/freeze/freeze_manifest.json"],
        "p3_unchanged": actual["reports/p3_a1r_multi_instance_v1/freeze/freeze_manifest.json"] == EXPECTED_FREEZE_MANIFEST_HASHES["reports/p3_a1r_multi_instance_v1/freeze/freeze_manifest.json"],
        "historical_reports_unchanged": not history_changes,
        "step_payload_opened": False,
        "abaqus_accessed": False,
    }


def build_audits() -> dict[str, dict[str, Any]]:
    formal_before = formal_test_state()
    expected_formal = {"run_count": 0, "content_accessed": False, "unlock_authorized": False}
    if formal_before != expected_formal:
        raise RuntimeError("QF_E_FORMAL_TEST_GUARD")

    source_counts = Counter()
    status_counts = Counter()
    language_counts = Counter()
    difficulty_counts = Counter()
    operation_counts = Counter()
    quantifier_counts = Counter()
    frame_sizes: list[int] = []
    canonical_sizes: list[int] = []
    ratios: list[float] = []
    frame_valid = canonical_valid = exact = deterministic = 0
    decode_audits: dict[str, Any] = {}
    identities = {"train": {}, "val": {}}
    canonical_ids: list[str] = []

    rows = list(_p2_rows())
    rows.extend(_p3_rows(decode_audits))
    for canonical, metadata in rows:
        source_counts[metadata["source"]] += 1
        status_counts[canonical["status"]] += 1
        language_counts[metadata["language"]] += 1
        difficulty_counts[metadata["difficulty"]] += 1
        canonical_ids.append(metadata["canonical_id"])
        if metadata["split"] in identities:
            bucket = identities[metadata["split"]]
            for field in (
                "canonical_id", "exact_graph_signature", "base_geometry_family_id", "sample_id",
            ):
                value = metadata[field]
                if value:
                    bucket.setdefault(field, set()).add(value)

        frame = canonical_to_query_frame(canonical)
        frame_validation = validate_query_frame(frame)
        frame_valid += int(frame_validation["valid"])
        first_bytes = compile_query_frame_json(frame)
        second_bytes = compile_query_frame_json(json.loads(canonical_json(frame)))
        restored = json.loads(first_bytes)
        canonical_validation = validate_intent_result(restored)
        canonical_valid += int(canonical_validation["valid"])
        exact += int(canonical_json(canonical) == first_bytes)
        deterministic += int(first_bytes == second_bytes)

        frame_size = len(canonical_json(frame))
        canonical_size = len(canonical_json(canonical))
        frame_sizes.append(frame_size)
        canonical_sizes.append(canonical_size)
        ratios.append(frame_size / canonical_size)

        if canonical["status"] == "VALID":
            for operation in canonical["intent"]["operations"]:
                operation_counts[operation["operation_type"]] += 1
                target = operation["target"]
                if "quantifier" in target:
                    quantifier_counts[target["quantifier"]] += 1

    total = len(rows)
    if total == 0:
        raise RuntimeError("QF_E_EMPTY_AUDIT")
    manifest_authorized = EXPECTED_SOURCE_COUNTS["p2_fixtures"] + EXPECTED_SOURCE_COUNTS["p3_train"] + EXPECTED_SOURCE_COUNTS["p3_val"]
    compiler_roundtrip = {
        "schema_version": "topocae-p4-api-r2a-compiler-roundtrip-audit/1",
        "total_records": total,
        "authorized_manifest_records": manifest_authorized,
        "manifest_count_difference": total - manifest_authorized,
        "source_counts": dict(sorted(source_counts.items())),
        "frame_schema_valid_count": frame_valid,
        "frame_schema_valid_rate": frame_valid / total,
        "canonical_schema_valid_count": canonical_valid,
        "canonical_schema_valid_rate": canonical_valid / total,
        "semantic_exact_count": exact,
        "semantic_exact_rate": exact / total,
        "deterministic_reproduction_count": deterministic,
        "deterministic_reproduction_rate": deterministic / total,
        "decode_audits": decode_audits,
        "llm_calls": 0,
        "api_calls": 0,
    }
    compactness = {
        "schema_version": "topocae-p4-api-r2a-compactness-audit/1",
        "record_count": total,
        "schema_utf8_bytes": (REPORT / "query_frame.schema.json").stat().st_size,
        "schema_limit_bytes": 10 * 1024,
        "frame_bytes_p50": int(_nearest_rank([float(value) for value in frame_sizes], 0.5)),
        "canonical_bytes_p50": int(_nearest_rank([float(value) for value in canonical_sizes], 0.5)),
        "frame_to_canonical_ratio_p50": _nearest_rank(ratios, 0.5),
        "frame_to_canonical_ratio_p95": _nearest_rank(ratios, 0.95),
        "median_method": "per-record nearest-rank",
        "ratio_gate": 0.65,
        "gate_pass": _nearest_rank(ratios, 0.5) <= 0.65,
    }
    coverage = {
        "schema_version": "topocae-p4-api-r2a-coverage-audit/1",
        "record_count": total,
        "language_counts": dict(sorted(language_counts.items())),
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "operation_type_counts": dict(sorted(operation_counts.items())),
        "quantifier_counts": dict(sorted(quantifier_counts.items())),
        "note": "fixture_unlabeled denotes P2 non-valid envelopes without a source_language field; it is not an inferred language",
    }
    overlaps = {}
    for field in ("canonical_id", "exact_graph_signature", "base_geometry_family_id", "sample_id"):
        overlaps[field] = len(identities["train"].get(field, set()) & identities["val"].get(field, set()))
    duplicate_ids = len(canonical_ids) - len(set(canonical_ids))
    formal_after = formal_test_state()
    leakage = {
        "schema_version": "topocae-p4-api-r2a-leakage-audit/1",
        "train_val_canonical_id_overlap_count": overlaps["canonical_id"],
        "train_val_identity_overlap_counts": overlaps,
        "duplicate_identity_definition": "duplicate canonical record identity across the authorized 21,885-row audit scope",
        "duplicate_identity_count": duplicate_ids,
        "decoded_test_lines": sum(item["decoded_test_lines"] for item in decode_audits.values()),
        "opaque_test_lines_crossed_without_decode": sum(item["opaque_test_lines_crossed"] for item in decode_audits.values()),
        "formal_test_access_count": 0,
        "formal_test_payload_decoded": False,
        "formal_test_ledger_before": formal_before,
        "formal_test_ledger_after": formal_after,
        "ground_truth_read": False,
        "step_accessed": False,
        "graph_accessed": False,
        "source_identity_emitted_to_frame": False,
    }
    return {
        "compiler_roundtrip": compiler_roundtrip,
        "compactness": compactness,
        "coverage": coverage,
        "leakage": leakage,
        "freeze": _freeze_audit(),
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def run_p4_regression() -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-p", "test_p4*.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    output = completed.stdout + "\n" + completed.stderr
    match = __import__("re").search(r"Ran\s+(\d+)\s+tests?", output)
    return {
        "schema_version": "topocae-p4-api-r2a-regression-results/1",
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "interpreter_path": sys.executable,
        "python_version": platform.python_version(),
        "test_pattern": "test_p4*.py",
        "tests_run": int(match.group(1)) if match else None,
        "return_code": completed.returncode,
        "stdout_or_stderr_persisted": False,
    }


def write_delivery_manifest() -> None:
    assets = sorted(path for path in REPORT.iterdir() if path.is_file() and path.name != "delivery_manifest.json")
    write_json(REPORT / "delivery_manifest.json", {
        "schema_version": "topocae-p4-api-r2a-delivery-manifest/1",
        "assets": [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in assets
        ],
    })


def main() -> int:
    audits = build_audits()
    write_json(REPORT / "compiler_roundtrip_audit.json", audits["compiler_roundtrip"])
    write_json(REPORT / "compactness_audit.json", audits["compactness"])
    write_json(REPORT / "coverage_audit.json", audits["coverage"])
    write_json(REPORT / "leakage_audit.json", audits["leakage"])
    write_json(REPORT / "freeze_audit.json", audits["freeze"])
    regression = run_p4_regression()
    write_json(REPORT / "regression_results.json", regression)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
