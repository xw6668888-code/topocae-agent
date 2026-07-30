from __future__ import annotations

"""Run the preholdout-frozen P1-C model once on the allowed Real-Dev scope."""

import csv
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import torch

import run_p1_b_real_dev_evaluation as protocol
from models.p1_b_candidate_topology_classifier import CandidateModelDimensions, CandidateTopologyClassifier
from p1_c.oriented_cylinder_polarity_v1 import augment_graph_with_oriented_cylinder_polarity
from p1_c_general_recognition_pipeline_v1 import run_p1_c_general_recognition


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "reports" / "p1_c_real_dev_v1"
FREEZE = ROOT / "reports" / "p1_c_freeze_v1" / "preholdout_selection.json"
FREEZE_SIDECAR = ROOT / "reports" / "p1_c_freeze_v1" / "preholdout_selection.sha256.json"
GNN_RUN = ROOT / "reports" / "p1_c_strict_training_runs_v2" / "clean_7200_seed_20260718"
CANDIDATE_ROOT = ROOT / "reports" / "p1_c_stage6_iteration1_runs_v1" / "clean_7200_seed_20260718" / "candidate_model_v1"
ITERATION2 = ROOT / "reports" / "p1_c_stage6_iteration2_runs_v1" / "clean_7200_seed_20260718"
TRAINING_PACKAGE = ROOT / "output_generation_p1_c_clean_context_7200_strict_concept_r2" / "training_packages" / "clean_7200"
P1_B_REAL = ROOT / "reports" / "p1_b_real_dev_v1" / "metrics.json"
AREA_TRUTH = ROOT / "reports" / "p1_b_real_dev_v1" / "area_truth.json"


def _sha(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest().lower()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _assert_freeze() -> dict[str, Any]:
    freeze = _read(FREEZE)
    sidecar = _read(FREEZE_SIDECAR)
    if _sha(FREEZE) != sidecar["sha256"]:
        raise RuntimeError("P1-C preholdout freeze hash mismatch")
    if freeze["selected_variant"] != "clean_7200" or int(freeze["selected_seed"]) != 20260718:
        raise RuntimeError("unexpected P1-C frozen selection")
    if freeze["contracts"]["real_dev_access_before_freeze"]:
        raise RuntimeError("invalid Real-Dev chronology contract")
    return freeze


_extract_graph = protocol.extract_graph_v2


def _extract_graph_with_polarity(shape, *args, **kwargs):
    graph = _extract_graph(shape, *args, **kwargs)
    return augment_graph_with_oriented_cylinder_polarity(shape, graph)


def _load_candidate_model(device: torch.device) -> CandidateTopologyClassifier:
    checkpoint = torch.load(CANDIDATE_ROOT / "best.pt", map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = CandidateTopologyClassifier(
        CandidateModelDimensions(**config["dimensions"]),
        macro_classes=config["macro_classes"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


_base_read_json = protocol._read_json


def _mapped_read_json(path: Path):
    source = Path(path)
    if source == GNN_RUN / "candidate_model_v1" / "candidate_threshold_selection.json":
        source = ITERATION2 / "candidate_thresholds_iteration2.json"
    elif source == GNN_RUN / "global_decode_v2_selection.json":
        source = ITERATION2 / "global_candidate_decode_selection.json"
    return _base_read_json(source)


def _run_pipeline(*args, **kwargs):
    global_selection = kwargs.pop("global_decode_selection")
    result = run_p1_c_general_recognition(
        *args,
        global_candidate_decode_selection=global_selection,
        **kwargs,
    )
    proposals = result["candidate_proposals"]
    proposals.setdefault("proposal_count", int(proposals.get("candidate_count", len(proposals.get("proposals", [])))))
    return result


def _allowed_real_steps() -> list[Path]:
    allowed = [ROOT / "test_part_s", ROOT / "test_part_s1", ROOT / "test_part_s2"]
    result = []
    for directory in allowed:
        for path in sorted(directory.glob("*.stp")):
            if directory.name == "test_part_s2" and path.stem == "part10":
                continue
            resolved = path.resolve()
            if resolved.parent not in [item.resolve() for item in allowed]:
                raise RuntimeError("Real-Dev path escaped allowed roots")
            if "test_part_s3" in str(resolved).lower():
                raise RuntimeError("forbidden Real collection")
            result.append(resolved)
    return result


def _family_summary(per_class: dict[str, Any]) -> dict[str, Any]:
    groups = {
        "rib": ["rib"],
        "slot_family": ["slot_u", "slot_rect", "slot_v", "slot_o"],
        "fillet": ["fillet"],
        "blind_hole": ["blind_hole_round"],
        "chamfer": ["chamfer"],
        "cylinder_boss": ["cylinder_boss"],
        "hole_family": [
            "through_hole_round", "blind_hole_round", "stepped_hole_round", "counterbore_hole",
            "countersink_hole", "obround_hole",
        ],
    }
    result = {}
    for name, members in groups.items():
        records = [per_class[item] for item in members if item in per_class]
        truth = sum(int(row["truth"]) for row in records)
        predicted = sum(int(row["prediction"]) for row in records)
        matched = sum(int(row["matched"]) for row in records)
        precision = matched / predicted if predicted else 0.0
        recall = matched / truth if truth else 0.0
        result[name] = {
            "classes_present": [item for item in members if item in per_class],
            "truth": truth, "prediction": predicted, "matched": matched,
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        }
    return result


def _rewrite_outputs(freeze: dict[str, Any]) -> None:
    metrics_path = OUTPUT / "metrics.json"
    result = _read(metrics_path)
    p1_b = _read(P1_B_REAL)
    area = result["area_macro"]
    result["schema_version"] = "topocae_p1_c_real_dev_report_1"
    result["comparison"] = [
        {"pipeline": "P1-B_A0_frozen", "precision": None, "recall": None, "f1": 0.534, "scope_note": "frozen historical baseline"},
        {"pipeline": "P1-B_final_frozen", "precision": p1_b["area_macro"]["precision"], "recall": p1_b["area_macro"]["recall"], "f1": p1_b["area_macro"]["f1"], "scope_note": "referenced without rerun"},
        {"pipeline": "P1-C_clean_7200_seed_20260718", "precision": area["precision"], "recall": area["recall"], "f1": area["f1"], "scope_note": "preholdout-frozen model and Synthetic-Val thresholds"},
    ]
    result["required_class_and_family_metrics"] = _family_summary(area["per_class"])
    result["real_dev_targets_non_tuning"] = {
        "f1_gt_0_600": area["f1"] > 0.6,
        "rib_recall_gt_0": result["required_class_and_family_metrics"]["rib"]["recall"] > 0.0,
        "slot_family_recall_gt_0": result["required_class_and_family_metrics"]["slot_family"]["recall"] > 0.0,
    }
    result["preholdout_freeze"] = {"path": str(FREEZE), "sha256": _sha(FREEZE)}
    result["contracts"].update({
        "test_part_s3_used": False,
        "real_dev_caused_model_threshold_data_or_rule_change": False,
        "p1_c_only_one_selected_model_evaluated": True,
        "p1_b_result_referenced_without_rerun": True,
    })
    _write(metrics_path, result)

    rows = result["comparison"]
    with (OUTPUT / "pipeline_comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# P1-C Real-Dev 冻结比较",
        "",
        "本次评估发生在 preholdout 冻结之后，只运行一个由 Synthetic-Val 预先选定的 P1-C 模型；结果未用于任何开发或阈值修改。",
        "",
        f"- P1-B A0：F1=0.534000",
        f"- P1-B final：F1={p1_b['area_macro']['f1']:.6f}",
        f"- P1-C selected：P={area['precision']:.6f}，R={area['recall']:.6f}，F1={area['f1']:.6f}",
        "",
        "逐类、族级指标见 `metrics.json`；逐样本失败图谱见 `failure_atlas.json/csv`。",
        "",
        "协议：仅 test_part_s/test_part_s1/test_part_s2，排除 test_part_s2/part10；未访问 test_part_s3；固定 5 mm² 或 1% area+macro 一对一匹配。",
    ]
    (OUTPUT / "report.zh-CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest_files = [
        OUTPUT / "metrics.json", OUTPUT / "failure_atlas.json", OUTPUT / "pipeline_comparison.csv",
        OUTPUT / "human_review_comparison.csv", OUTPUT / "failure_atlas.csv", OUTPUT / "timing.csv",
        OUTPUT / "report.zh-CN.md",
    ]
    _write(OUTPUT / "manifest.json", {
        "schema_version": "topocae_p1_c_real_dev_manifest_1",
        "preholdout_freeze_sha256": _sha(FREEZE),
        "files": [{"path": str(path), "sha256": _sha(path), "bytes": path.stat().st_size} for path in manifest_files],
        "source_area_truth": {"path": str(AREA_TRUTH), "sha256": _sha(AREA_TRUTH)},
        "p1_b_metrics": {"path": str(P1_B_REAL), "sha256": _sha(P1_B_REAL)},
        "contracts": {"test_part_s3_used": False, "real_data_used_for_selection": False},
    })


def main() -> int:
    freeze = _assert_freeze()
    if OUTPUT.exists():
        raise RuntimeError("P1-C Real-Dev formal output already exists; refusing a second run")
    protocol.OUTPUT = OUTPUT
    protocol.RUN = GNN_RUN
    protocol.TRAINING_PACKAGE = TRAINING_PACKAGE
    protocol.AREA_TRUTH = AREA_TRUTH
    protocol.ALLOWED_DIRS = [ROOT / "test_part_s", ROOT / "test_part_s1", ROOT / "test_part_s2"]
    protocol.extract_graph_v2 = _extract_graph_with_polarity
    protocol._load_candidate_model = _load_candidate_model
    protocol._read_json = _mapped_read_json
    protocol.run_p1_b_general_recognition = _run_pipeline
    protocol._allowed_real_steps = _allowed_real_steps
    code = int(protocol.main())
    _rewrite_outputs(freeze)
    result = _read(OUTPUT / "metrics.json")
    print(json.dumps({
        "real_dev_complete": True,
        "evaluated_steps": result["evaluated_step_count"],
        "area_macro": {key: result["area_macro"][key] for key in ("precision", "recall", "f1")},
        "targets_non_tuning": result["real_dev_targets_non_tuning"],
        "output": str(OUTPUT),
    }, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
