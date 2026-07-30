from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import OUTPUT_ROOT, ROOT, read_json, sha256_file, write_json


FREEZE_ROOT = OUTPUT_ROOT / "freeze"
PIPELINE_PATH = ROOT / "p1_b_general_recognition_pipeline_v4.py"
ENTRY_PATH = ROOT / "p1_d_lite/recognizer_dev.py"


def _asset(path: Path, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "read_only_reference": True,
    }


def _selected_metrics() -> dict[str, float]:
    layered = read_json(OUTPUT_ROOT / "layered_comparison/metrics.json")
    s2 = next(item for item in layered["views"]["S2-Overlap"] if item["system"] == "P1-B frozen")
    full_b = next(item for item in layered["views"]["Real-Dev-Full"] if item["system"] == "P1-B frozen")
    full_c = next(item for item in layered["views"]["Real-Dev-Full"] if item["system"] == "P1-C frozen")
    return {
        "s2_overlap_object_f1": float(s2["layers"]["L3_decoder"]["f1"]),
        "real_dev_full_object_f1": float(full_b["layers"]["L3_decoder"]["f1"]),
        "p1_c_real_dev_full_object_f1": float(full_c["layers"]["L3_decoder"]["f1"]),
        "relative_to_p1_c": float(full_b["layers"]["L3_decoder"]["f1"] - full_c["layers"]["L3_decoder"]["f1"]),
    }


def capability_contract() -> dict[str, Any]:
    metrics = read_json(ROOT / "reports/p1_b_real_dev_v1/metrics.json")
    per_class = metrics["area_macro"]["per_class"]
    target_classes = sorted(per_class)
    return {
        "schema_version": "topocae_recognizer_dev_capability_contract_1",
        "recognizer": "Recognizer-Dev / P1-B frozen",
        "capability_level": "LIMITED_CONTROLLED_DOMAIN",
        "input": {
            "graph_schema": "topocae_graph_v2_relative_1_1",
            "required_fields": ["schema_version", "nodes", "edges", "edge_attributes", "graph_v2_protocol"],
            "optional_step_path": "used only when graph construction/geometry augmentation is required",
            "truth_fields": "removed before inference and never consumed",
        },
        "output": {
            "feature_object_schema": "topocae_recognizer_dev_feature_objects_1",
            "fields": ["feature_objects", "coverage_status", "failure_reason", "abstentions", "conflicts", "pipeline_manifest"],
            "face_identity": "zero-based indices from the supplied graph; original identities preserved",
        },
        "target_classes": target_classes,
        "confidence": {
            "object_source": "frozen candidate classifier plus topology confidence correction",
            "calibration_warning": "not calibrated for unknown CAD domains; do not interpret as universal probability",
        },
        "abstention": {
            "coverage_status_values": ["ACCEPTED", "PARTIAL", "ABSTAIN", "ERROR"],
            "required_for": [
                "slot or rib/web decisions",
                "no accepted object",
                "low candidate confidence or validity",
                "topology veto or empty membership",
                "global face conflict",
                "unsupported graph schema or unknown CAD context",
            ],
            "downstream_action": "request human confirmation or use Oracle feature objects; never silently coerce a class",
        },
        "known_supported_scope": {
            "geometry": "single-part, single-solid or single-part-like STEP/B-Rep in the controlled development domain",
            "analysis_role": "feature-object memory for downstream region selection, not arbitrary industrial feature recognition",
            "families": {
                "hole": {"real_dev_recall": 0.8208955223880597, "status": "supported_with_abstention"},
                "transition": {"real_dev_recall": 0.38461538461538464, "status": "limited"},
                "slot": {"real_dev_recall": 0.0, "status": "unsupported_predicted_path"},
                "rib_web": {"real_dev_recall": 0.0, "status": "unsupported_predicted_path"},
            },
        },
        "error_codes": {
            "E_INPUT": "input file missing or invalid JSON",
            "E_SCHEMA": "graph schema or topology indices invalid",
            "E_MODEL_ASSET": "frozen model/config asset missing",
            "E_INFERENCE": "model or pipeline inference failed",
            "E_OUTPUT_EXISTS": "CLI refuses to overwrite an existing output",
        },
        "prohibited_uses": [
            "claiming unknown-real generalization from Real-Dev",
            "using truth/area tables during formal inference",
            "silently accepting slot or rib/web predictions without confirmation",
            "part-, sample-, face- or absolute-area-specific rules",
        ],
    }


def known_limitations_text() -> str:
    return """# Recognizer-Dev 已知限制

冻结能力等级：`LIMITED_CONTROLLED_DOMAIN`。

- Real-Dev-Full object F1 为 0.600000；相对 P1-C 提升 0.153785，但该集合是开发域，不是未知真实件测试。
- hole family recall 为 0.820896；transition family recall 为 0.384615。
- slot family 和 rib/web family recall 均为 0，predicted geometry 路径不得静默用于这两类；必须 abstain、人工确认或使用 Oracle feature objects。
- Oracle 表明首要瓶颈是 face/GNN（F1 上界增益 +0.137566），可靠改进需要 Mixed-Context/跨 CAD 数据与训练，不适合一次规则修复。
- verifier/decode 是次要瓶颈（Oracle 增益 +0.122078），但现有证据未隔离到单一通用 B-Rep 谓词。
- 置信度只在冻结 Synthetic-Val 上选择，未做未知 CAD 域校准；不能当作通用概率。
- 不支持装配体、严重相交特征、任意复杂工业件、未知类别扩展或直接从语言生成 raw face ID。
- 正式推理必须 truth-free；面积 truth、人工 face-set 和 Oracle 只用于离线误差分解。

后续 Agent 论文必须同时报告 Oracle geometry 与 predicted geometry，并把拒识率、人工确认和失败原因纳入结果。
"""


def main() -> int:
    selected = read_json(ROOT / "reports/p1_b_freeze_v1/preholdout_selection.json")
    oracle = read_json(OUTPUT_ROOT / "oracle/metrics.json")
    metrics = _selected_metrics()
    assets = [
        _asset(Path(selected["selected_gnn_checkpoint"]), "gnn_checkpoint"),
        _asset(Path(selected["frozen_files"][1]["path"]), "gnn_config"),
        _asset(Path(selected["selected_candidate_checkpoint"]), "candidate_checkpoint"),
        _asset(Path(selected["frozen_files"][6]["path"]), "candidate_config"),
        _asset(Path(selected["synthetic_val_thresholds"]["same_instance"]), "same_instance_thresholds"),
        _asset(Path(selected["synthetic_val_thresholds"]["candidate"]), "candidate_thresholds"),
        _asset(Path(selected["synthetic_val_thresholds"]["global_decode"]), "global_decode_thresholds"),
        _asset(ROOT / "CAE_feature_ontology.yaml", "macro_label_map"),
        _asset(ROOT / "face_semantic_subtype_ontology.yaml", "semantic_label_map"),
        _asset(PIPELINE_PATH, "frozen_pipeline_entry"),
        _asset(ENTRY_PATH, "recognizer_dev_read_only_entry"),
    ]
    contract = capability_contract()
    write_json(FREEZE_ROOT / "capability_contract.json", contract)
    limitations_path = FREEZE_ROOT / "known_limitations.zh-CN.md"
    limitations_path.parent.mkdir(parents=True, exist_ok=True)
    limitations_path.write_text(known_limitations_text(), encoding="utf-8")
    assets.extend([
        _asset(FREEZE_ROOT / "capability_contract.json", "capability_contract"),
        _asset(limitations_path, "known_limitations"),
    ])
    manifest = {
        "schema_version": "topocae_recognizer_dev_manifest_1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "recognizer_name": "Recognizer-Dev",
        "selected_pipeline": "P1-B frozen / p1_b_general_recognition_pipeline_v4",
        "capability_level": "LIMITED_CONTROLLED_DOMAIN",
        "selection_metrics": metrics,
        "stopping_condition": "relative Real-Dev-Full object F1 improvement over P1-C >= 0.08",
        "stopping_condition_value": metrics["relative_to_p1_c"],
        "minimal_concept_fix_implemented": False,
        "minimal_fix_reason": "primary Oracle bottleneck is face/GNN and requires data/training; no single safe rule isolated",
        "oracle_primary_bottleneck": oracle["primary_bottleneck"],
        "oracle_secondary_bottleneck": oracle["secondary_bottleneck"],
        "human_gate_triggered": False,
        "read_only_python_api": "p1_d_lite.recognizer_dev.recognize_graph",
        "cli_entry": "W:\\python\\envs\\feagpt\\python.exe -m p1_d_lite.recognizer_dev --graph <graph.json> [--step <part.step>] [--output <new.json>]",
        "assets": assets,
        "contracts": {
            "historical_checkpoints_copied_or_modified": False,
            "truth_used_during_inference": False,
            "formal_holdout_rerun": False,
            "downstream_requires_oracle_predicted_dual_track": True,
        },
    }
    manifest_path = FREEZE_ROOT / "recognizer_dev_manifest.json"
    write_json(manifest_path, manifest)
    write_json(FREEZE_ROOT / "recognizer_dev_manifest.sha256.json", {
        "schema_version": "topocae_recognizer_dev_manifest_sha256_1",
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "referenced_asset_count": len(assets),
        "referenced_assets": [{"path": item["path"], "sha256": item["sha256"]} for item in assets],
    })
    print(FREEZE_ROOT.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

