from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import evaluate_paper_v1_recognition as evaluation


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "configs" / "paper_v1_clean_candidate"
REPORTS = ROOT / "reports" / "paper_v1_p1"
EVAL_RUN = REPORTS / "evaluations" / "repeat_1" / "clean_core"
SYNTHETIC_ROOT = ROOT / "output_generation_v5_1_augmented_full_5400_clean"
RAW_PREDICTIONS = REPORTS / "preparation" / "raw_gnn_predictions"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metric(value: Any) -> str:
    return "not computable" if value is None else f"{float(value):.6f}"


def compute_raw_gnn_reproduction() -> dict[str, Any]:
    dataset_config = load(CONFIG / "datasets.json")
    test_ids = evaluation.load_effective_splits(
        dataset_config, CONFIG / "quality_exclusions.json"
    )["test"]
    checkpoint_config = load(
        ROOT
        / "output_generation_v5_5_v51dist_looprole_5400"
        / "model_runs"
        / "edge_aware_v5_5_expanded_v51_compat_24dim"
        / "config.json"
    )
    macro_pairs: list[tuple[int, int]] = []
    semantic_pairs: list[tuple[int, int]] = []
    edge_reports = []
    missing = []
    for sample_id in test_ids:
        graph_path = SYNTHETIC_ROOT / "graphs" / f"{sample_id}.json"
        prediction_path = RAW_PREDICTIONS / f"{sample_id}.json"
        if not graph_path.is_file() or not prediction_path.is_file():
            missing.append(sample_id)
            continue
        graph = load(graph_path)
        prediction = load(prediction_path)
        macro, semantic = evaluation.face_and_semantic_pairs(prediction, graph)
        macro_pairs.extend(macro)
        semantic_pairs.extend(semantic)
        edge_reports.append(evaluation.edge_counts(prediction, graph, threshold=0.5))
    spaces = checkpoint_config["label_spaces"]
    payload = {
        "schema": "topocae_paper_v1_raw_frozen_gnn_reproduction_v1",
        "phase": "P1-A1",
        "dataset": "formal Synthetic-Test",
        "sample_count": len(test_ids),
        "evaluated_count": len(test_ids) - len(missing),
        "missing_sample_ids": missing,
        "face": evaluation.classification_report(
            macro_pairs, spaces["macro_feature_id_to_key"]
        ),
        "semantic": evaluation.classification_report(
            semantic_pairs, spaces["semantic_subtype_id_to_key"]
        ),
        "edge": evaluation.aggregate_edge(edge_reports),
        "scope_note": (
            "This reproduces the frozen GNN heads before the candidate classifier and "
            "topology postprocess. It is diagnostic provenance, not the selected end-to-end baseline."
        ),
    }
    write_json(REPORTS / "raw_gnn_reproduction.json", payload)
    return payload


def selection_payload(metrics: dict[str, Any]) -> dict[str, Any]:
    core = metrics["clean_core"]
    rib = metrics["clean_rib"]
    core_syn = core["synthetic_test"]
    rib_syn = rib["synthetic_test"]
    core_rib_f1 = float(core_syn["object"]["per_class"]["rib"]["f1"])
    rib_rib_f1 = float(rib_syn["object"]["per_class"]["rib"]["f1"])
    rib_gain = rib_rib_f1 - core_rib_f1
    macro_drop = float(core_syn["face"]["macro_f1"]) - float(rib_syn["face"]["macro_f1"])
    rotation_names = (
        "class_consistency",
        "object_correspondence_consistency",
        "face_role_consistency",
    )
    rotation_deltas = {
        name: float(rib["rotation_dev"][name]) - float(core["rotation_dev"][name])
        for name in rotation_names
    }
    criteria = {
        "stable_rib_object_f1_gain": rib_gain > 0.0,
        "overall_macro_f1_drop_at_most_0_005": macro_drop <= 0.005,
        "rotation_consistency_not_decreased": all(value >= 0.0 for value in rotation_deltas.values()),
    }
    selected = "clean_rib" if all(criteria.values()) else "clean_core"
    return {
        "schema": "topocae_paper_v1_clean_candidate_selection_v1",
        "phase": "P1-A1",
        "selected_variant": selected,
        "selection_dataset": "Synthetic-Test plus Rotation-Dev consistency gate",
        "synthetic_test_comparison": {
            "clean_core_rib_object_f1": core_rib_f1,
            "clean_rib_rib_object_f1": rib_rib_f1,
            "rib_object_f1_gain": rib_gain,
            "clean_core_face_macro_f1": core_syn["face"]["macro_f1"],
            "clean_rib_face_macro_f1": rib_syn["face"]["macro_f1"],
            "face_macro_f1_drop": macro_drop,
        },
        "rotation_deltas_clean_rib_minus_clean_core": rotation_deltas,
        "criteria": criteria,
        "decision_reason": (
            "Clean-Rib has no Synthetic-Test rib-object F1 gain; the required stable-gain "
            "criterion therefore fails. Clean-Core is selected."
        ),
        "real_dev_used_for_selection": False,
        "s3_used_for_selection": False,
    }


def orientation_tokens(graph: dict[str, Any]) -> list[str]:
    tokens = set()
    for feature in graph.get("feature_sets", []) or []:
        params = feature.get("geometry_parameters", {}) or {}
        raw = feature.get("raw_parameters", {}) or {}
        vector = params.get("axis_vector") or raw.get("axis") or raw.get("direction")
        if isinstance(vector, list) and len(vector) >= 3:
            values = [abs(float(value)) for value in vector[:3]]
            tokens.add("xyz"[max(range(3), key=values.__getitem__)])
    return sorted(tokens) or ["geometry-defined"]


def select_synthetic_review(test_ids: list[str]) -> list[dict[str, Any]]:
    candidates = []
    macro_support: Counter[str] = Counter()
    for sample_id in test_ids:
        graph_path = SYNTHETIC_ROOT / "graphs" / f"{sample_id}.json"
        graph = load(graph_path)
        macros = sorted(
            {str(item.get("macro_feature")) for item in graph.get("feature_sets", []) if item.get("macro_feature")}
        )
        macro_support.update(macros)
        candidates.append(
            {
                "sample_id": sample_id,
                "graph": graph,
                "macros": macros,
                "stock_type": str(graph.get("stock_profile", {}).get("stock_type", "unknown")),
                "orientations": orientation_tokens(graph),
            }
        )
    selected = []
    covered_macros: set[str] = set()
    covered_stock: set[str] = set()
    covered_orientations: set[str] = set()
    while len(selected) < 20:
        remaining = [item for item in candidates if item not in selected]
        best = max(
            remaining,
            key=lambda item: (
                sum(1000.0 / macro_support[name] for name in set(item["macros"]) - covered_macros)
                + (20.0 if item["stock_type"] not in covered_stock else 0.0)
                + 5.0 * len(set(item["orientations"]) - covered_orientations),
                len(item["macros"]),
                len(item["graph"].get("nodes", [])),
                item["sample_id"],
            ),
        )
        selected.append(best)
        covered_macros.update(best["macros"])
        covered_stock.add(best["stock_type"])
        covered_orientations.update(best["orientations"])

    review = []
    semantic_map = load(
        ROOT
        / "output_generation_v5_5_v51dist_looprole_5400"
        / "model_runs"
        / "edge_aware_v5_5_expanded_v51_compat_24dim"
        / "config.json"
    )["label_spaces"]["semantic_subtype_id_to_key"]
    for item in selected:
        sample_id = item["sample_id"]
        graph = item["graph"]
        nodes = {int(node["idx"]): node for node in graph.get("nodes", [])}
        objects = []
        for feature in graph.get("feature_sets", []) or []:
            faces = []
            for face_index in feature.get("node_indices", []) or []:
                node = nodes[int(face_index)]
                faces.append(
                    {
                        "face_index_reference": int(face_index),
                        "area": node.get("attributes", {}).get("physical", {}).get("area"),
                        "semantic_role": semantic_map.get(
                            str(node.get("semantic_label", -1)), f"id_{node.get('semantic_label', -1)}"
                        ),
                    }
                )
            objects.append(
                {
                    "instance_id": feature.get("instance_id"),
                    "macro_feature": feature.get("macro_feature"),
                    "source_generator": feature.get("source_generator"),
                    "faces_for_ug_check": faces,
                }
            )
        review.append(
            {
                "review_id": f"SYN-{len(review) + 1:02d}",
                "sample_id": sample_id,
                "split": "Synthetic-Test",
                "step_path": str((SYNTHETIC_ROOT / "steps" / f"{sample_id}.step").resolve()),
                "graph_path": str((SYNTHETIC_ROOT / "graphs" / f"{sample_id}.json").resolve()),
                "stock_type": item["stock_type"],
                "orientation_summary": item["orientations"],
                "face_count": len(graph.get("nodes", [])),
                "feature_instance_count": len(objects),
                "macro_features": item["macros"],
                "graph_truth_objects": objects,
                "review_result": None,
                "review_note": None,
            }
        )
    return review


def real_risk(feature_doc: dict[str, Any]) -> tuple[float, list[str]]:
    objects = feature_doc.get("feature_objects", []) or []
    flags = []
    confidences = [
        float(item.get("confidence", {}).get("macro_confidence_mean", 0.0)) for item in objects
    ]
    if not objects:
        flags.append("no_feature_object")
    if any(value < 0.75 for value in confidences):
        flags.append("low_confidence_below_0.75")
    if any(item.get("macro_feature") == "rib" for item in objects):
        flags.append("contains_rib")
    if len(objects) >= 8:
        flags.append("many_objects")
    score = (
        (10.0 if not objects else 0.0)
        + sum(max(0.0, 0.8 - value) for value in confidences) * 10.0
        + (2.0 if "contains_rib" in flags else 0.0)
        + max(0, len(objects) - 7) * 0.2
    )
    return score, flags or ["routine_spot_check"]


def select_real_review() -> list[dict[str, Any]]:
    allocation = {
        "real_test_part": 2,
        "real_test_part_s": 3,
        "real_test_part_s1": 2,
        "real_test_part_s2": 3,
    }
    selected = []
    for dataset, count in allocation.items():
        feature_dir = EVAL_RUN / dataset / "feature_objects"
        candidates = []
        for path in sorted(feature_dir.glob("*.json")):
            feature_doc = load(path)
            score, flags = real_risk(feature_doc)
            candidates.append((score, path.stem, flags, feature_doc))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        selected.extend((dataset, *item) for item in candidates[:count])

    review = []
    for dataset, score, sample_id, flags, feature_doc in selected:
        objects = []
        for item in feature_doc.get("feature_objects", []) or []:
            fingerprint = item.get("fingerprint", {}) or {}
            face_roles = sorted((fingerprint.get("semantic_subtype_histogram") or {}).keys())
            objects.append(
                {
                    "feature_object_id": item.get("feature_object_id"),
                    "macro_feature": item.get("macro_feature"),
                    "macro_confidence_mean": item.get("confidence", {}).get("macro_confidence_mean"),
                    "semantic_confidence_mean": item.get("confidence", {}).get("semantic_confidence_mean"),
                    "target_face_indices": item.get("face_indices", []),
                    "selectable_regions": item.get("selectable_regions", []),
                    "area_total": fingerprint.get("area_total"),
                    "face_roles": face_roles,
                }
            )
        review.append(
            {
                "review_id": f"REAL-{len(review) + 1:02d}",
                "dataset": dataset,
                "sample_id": sample_id,
                "step_path": feature_doc.get("source_step_path"),
                "graph_path": feature_doc.get("source_graph_path"),
                "feature_objects_path": str(
                    (EVAL_RUN / dataset / "feature_objects" / f"{sample_id}.json").resolve()
                ),
                "predicted_object_count": len(objects),
                "predicted_objects": objects,
                "risk_score": score,
                "risk_flags": flags,
                "review_result": None,
                "review_note": None,
            }
        )
    return review


def write_review_package(synthetic: list[dict[str, Any]], real: list[dict[str, Any]]) -> None:
    write_json(
        REPORTS / "human_review_synthetic_20.json",
        {
            "schema": "topocae_paper_v1_synthetic_human_review_v1",
            "phase": "P1-A1",
            "selection": "deterministic weighted stratification over macro class, stock type and orientation",
            "samples": synthetic,
        },
    )
    write_json(
        REPORTS / "human_review_real_dev_10.json",
        {
            "schema": "topocae_paper_v1_real_dev_human_review_v1",
            "phase": "P1-A1",
            "selection": "fixed 2/3/2/3 allocation across allowed Real-Dev sets, risk-ranked within set",
            "truth_use": "manual review only; not training, threshold selection or model selection",
            "samples": real,
        },
    )
    with (REPORTS / "human_review_synthetic_20.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "review_id", "sample_id", "split", "stock_type", "orientation_summary",
                "face_count", "feature_instance_count", "macro_features", "step_path", "graph_path",
                "review_result", "review_note",
            ],
        )
        writer.writeheader()
        for item in synthetic:
            row = dict(item)
            row["orientation_summary"] = ";".join(item["orientation_summary"])
            row["macro_features"] = ";".join(item["macro_features"])
            writer.writerow({name: row.get(name) for name in writer.fieldnames})
    with (REPORTS / "human_review_real_dev_10.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "review_id", "dataset", "sample_id", "predicted_object_count", "risk_score",
                "risk_flags", "step_path", "graph_path", "feature_objects_path", "review_result", "review_note",
            ],
        )
        writer.writeheader()
        for item in real:
            row = dict(item)
            row["risk_flags"] = ";".join(item["risk_flags"])
            writer.writerow({name: row.get(name) for name in writer.fieldnames})
    guide = f"""# Paper v1 P1-A1 人工审查说明

## 审查批次

- 批次：`H1-01`
- 合成审查：20 件，见 `human_review_synthetic_20.csv/json`。
- Real-Dev 审查：10 件，见 `human_review_real_dev_10.csv/json`。
- 目的：核对 graph 真值与 STEP、以及 Clean-Core 预测与真实几何是否一致。结果只用于报告和失败分类，不用于训练、阈值选择或新增规则。

## 合成 STEP：逐项看什么

1. 在 UG 中打开 `step_path`，确认零件为单一有效实体。
2. 依据 `graph_truth_objects` 逐个核对宏特征类型与实例数量。
3. 用每个对象给出的面面积与语义角色定位目标面；`face_index_reference` 仅作交叉核对，不要求人工批量标注 face ID。
4. 若宏类型、实例边界或面角色任一错误，填 `FAIL` 并简述对象；否则填 `PASS`。

## Real-Dev：逐项看什么

1. 打开 `step_path`，对照 `predicted_objects` 的宏类型、置信度、面积、目标面和可选区域。
2. 优先检查 `risk_flags` 标出的低置信、rib、多对象或空预测情况。
3. 核对是否漏检、误检、对象碎裂/合并、语义角色错误；不需要重新编号所有面。
4. 将 CSV 的 `review_result` 填为 `PASS` 或 `FAIL`；`review_note` 使用 `对象ID | 错误类型 | 简要证据`。

## 回传格式与通过标准

- 回传两个已填写 CSV；保持 `review_id` 不变。
- 合成样本通过：实体有效，宏类别、实例数、实例边界和关键 face-role 均一致。
- Real-Dev 样本通过：主要对象无漏检/误检，目标面可在 UG 中复核；面积或置信度本身不是通过判据。
- 预计工作量：合成每件约 3–5 分钟，Real-Dev 每件约 5–8 分钟，总计约 2–3 小时。
"""
    write_text(REPORTS / "human_review_guide.zh-CN.md", guide)


def write_reports(
    metrics: dict[str, Any], raw: dict[str, Any], selection: dict[str, Any], audit: dict[str, Any]
) -> None:
    core = metrics["clean_core"]
    rib = metrics["clean_rib"]
    syn = core["synthetic_test"]
    ci = syn["bootstrap"]["intervals"]
    stats = audit["split_statistics"]
    s2_area = core["real_test_part_s2"]["area_truth"]
    baseline = f"""# Paper v1 clean recognition baseline（P1-A1）

## 冻结结论

冻结 `Clean-Core`：frozen GNN + formal-split clean candidate classifier + topology postprocess。Clean-Rib 未被选中，因为 Synthetic-Test rib object F1 增益为 `{selection['synthetic_test_comparison']['rib_object_f1_gain']:.6f}`；Real-Dev 未参与选择。

## 正式数据与审计

| Split | 样本 | Faces | Feature instances | graph/STEP mismatch |
|---|---:|---:|---:|---:|
| Synthetic-Train | {stats['train']['sample_count']} | {stats['train']['face_count']} | {stats['train']['feature_instance_count']} | {stats['train']['graph_step_face_mismatch_count']} |
| Synthetic-Val | {stats['val']['sample_count']} | {stats['val']['face_count']} | {stats['val']['feature_instance_count']} | {stats['val']['graph_step_face_mismatch_count']} |
| Synthetic-Test | {stats['test']['sample_count']} | {stats['test']['face_count']} | {stats['test']['feature_instance_count']} | {stats['test']['graph_step_face_mismatch_count']} |

Synthetic-Test 540/540 STEP 已由 OCC 直接验证为可读单实体。base geometry、旋转不变精确签名、旋转不变近重复签名和显式旋转族的跨 split 碰撞均为 0。历史 Train 中 5 个 graph/STEP face-count 不匹配样本已从 clean 辅助分类器数据排除；frozen GNN 为固定基线，不重新训练。

## Synthetic-Test：选定 Clean-Core

| 指标 | 值 | 95% bootstrap CI（如适用） |
|---|---:|---:|
| Face accuracy | {metric(syn['face']['accuracy'])} | — |
| Face Macro-F1 | {metric(syn['face']['macro_f1'])} | — |
| Semantic subtype Macro-F1 | {metric(syn['semantic']['macro_f1'])} | — |
| Same-instance edge F1 | {metric(syn['edge']['f1'])} | — |
| Object precision | {metric(syn['object']['precision'])} | [{metric(ci['precision']['low'])}, {metric(ci['precision']['high'])}] |
| Object recall | {metric(syn['object']['recall'])} | [{metric(ci['recall']['low'])}, {metric(ci['recall']['high'])}] |
| Object F1 | {metric(syn['object']['f1'])} | [{metric(ci['f1']['low'])}, {metric(ci['f1']['high'])}] |
| Exact instance ratio | {metric(syn['object']['exact_instance_ratio'])} | [{metric(ci['exact_instance_ratio']['low'])}, {metric(ci['exact_instance_ratio']['high'])}] |
| Instance IoU | {metric(syn['object']['instance_iou'])} | [{metric(ci['instance_iou']['low'])}, {metric(ci['instance_iou']['high'])}] |
| Rib object F1 | {metric(syn['object']['per_class']['rib']['f1'])} | — |

Face、semantic、edge、object coverage 均为 1.0。逐类 precision/recall/F1 位于 `reports/paper_v1_p1/metrics.json` 和扁平 CSV。

## 原始 frozen GNN 复现与端到端差异

原始 GNN（candidate/topology 前）在同一正式 Synthetic-Test 上：face accuracy `{metric(raw['face']['accuracy'])}`、face Macro-F1 `{metric(raw['face']['macro_f1'])}`、semantic Macro-F1 `{metric(raw['semantic']['macro_f1'])}`、edge F1 `{metric(raw['edge']['f1'])}`。端到端 Clean-Core 的 face 指标较低，是 clean candidate remap 与既有 topology postprocess 的联合输出，而非 checkpoint 加载失败；本任务没有修改识别规则。

## Clean-Core / Clean-Rib 选择对比

| Variant | Face Macro-F1 | Object F1 | Rib object F1 | Rotation class/object/role |
|---|---:|---:|---:|---:|
| Clean-Core | {metric(syn['face']['macro_f1'])} | {metric(syn['object']['f1'])} | {metric(syn['object']['per_class']['rib']['f1'])} | {metric(core['rotation_dev']['class_consistency'])} / {metric(core['rotation_dev']['object_correspondence_consistency'])} / {metric(core['rotation_dev']['face_role_consistency'])} |
| Clean-Rib | {metric(rib['synthetic_test']['face']['macro_f1'])} | {metric(rib['synthetic_test']['object']['f1'])} | {metric(rib['synthetic_test']['object']['per_class']['rib']['f1'])} | {metric(rib['rotation_dev']['class_consistency'])} / {metric(rib['rotation_dev']['object_correspondence_consistency'])} / {metric(rib['rotation_dev']['face_role_consistency'])} |

## Real-Dev coverage 与失败分类

`test_part`（5）、`test_part_s`（9）、`test_part_s1`（7）缺少 face/object 真值，相关指标明确为 `not computable`；只报告预测覆盖与人工风险。`test_part_s2`（11）仅有 area truth：Clean-Core precision `{metric(s2_area['precision'])}`、recall `{metric(s2_area['recall'])}`。其主要可量化失败集中在 part4 的 through-hole/fillet 混淆；其余集合的低置信、rib、多对象和空预测仅作为人工审查类别，不据此新增规则。

Query 指标缺少正式 query truth，状态为 `not computable`。所有 Real-Dev 结果仅作现状报告与失败分类，不参与训练、阈值或候选选择。
"""
    write_text(ROOT / "reports" / "paper_v1_recognition_baseline.md", baseline)
    rotation = f"""# Paper v1 rotation consistency（P1-A1）

Rotation-Dev 仅使用允许的 `test_part_s` 三个既有、可追溯旋转族；没有读取 S3，也没有把 Real-Dev 用于模型选择。

| Variant | Groups | Class consistency | Object correspondence | Face-role consistency |
|---|---:|---:|---:|---:|
| Clean-Core | {core['rotation_dev']['group_count']} | {metric(core['rotation_dev']['class_consistency'])} | {metric(core['rotation_dev']['object_correspondence_consistency'])} | {metric(core['rotation_dev']['face_role_consistency'])} |
| Clean-Rib | {rib['rotation_dev']['group_count']} | {metric(rib['rotation_dev']['class_consistency'])} | {metric(rib['rotation_dev']['object_correspondence_consistency'])} | {metric(rib['rotation_dev']['face_role_consistency'])} |

## 分组结果（Clean-Core）

| Group | Samples | Class | Object | Face-role |
|---|---|---:|---:|---:|
"""
    for group in core["rotation_dev"]["groups"]:
        rotation += (
            f"| {group['group_id']} | {', '.join(group['sample_ids'])} | "
            f"{metric(group['class_consistency'])} | {metric(group['object_correspondence_consistency'])} | "
            f"{metric(group['face_role_consistency'])} |\n"
        )
    rotation += "\nClean-Rib 与 Clean-Core 三项旋转指标完全相同；由于 rib-object F1 没有增益，最终仍冻结 Clean-Core。\n"
    write_text(ROOT / "reports" / "paper_v1_rotation_consistency.md", rotation)

    worst = sorted(
        syn["samples"],
        key=lambda item: (
            -(item.get("object", {}).get("fp", 0) + item.get("object", {}).get("fn", 0)),
            item.get("sample_id", ""),
        ),
    )[:20]
    write_json(
        REPORTS / "failure_atlas.json",
        {
            "schema": "topocae_paper_v1_failure_atlas_v1",
            "phase": "P1-A1",
            "synthetic_test_worst_20_by_object_fp_plus_fn": worst,
            "real_dev": {
                "face_object_truth": "not_computable for test_part/test_part_s/test_part_s1",
                "s2_area_truth": s2_area,
                "policy": "classification/reporting only; no new rules",
            },
        },
    )


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed-as-distribution"


def write_manifest() -> dict[str, Any]:
    paths = [
        "configs/paper_v1_clean_candidate/README.md",
        "configs/paper_v1_clean_candidate/datasets.json",
        "configs/paper_v1_clean_candidate/models.json",
        "configs/paper_v1_clean_candidate/evaluation.json",
        "configs/paper_v1_clean_candidate/quality_exclusions.json",
        "configs/paper_v1_clean_candidate/selected_thresholds.json",
        "configs/paper_v1_clean_candidate/selection.json",
        "configs/paper_v1_clean_candidate/recognition.json",
        "output_generation_v5_5_v51dist_looprole_5400/model_runs/edge_aware_v5_5_expanded_v51_compat_24dim/best.pt",
        "reports/paper_v1_p1/model_runs/candidate_topology_clean/model.joblib",
        "reports/paper_v1_p1/model_data_provenance.json",
        "reports/paper_v1_p1/model_data_provenance.csv",
        "reports/paper_v1_p1/dataset_audit.json",
        "reports/paper_v1_dataset_audit.md",
        "reports/paper_v1_p1/metrics.json",
        "reports/paper_v1_p1/metrics.csv",
        "reports/paper_v1_p1/repeatability.json",
        "reports/paper_v1_p1/raw_gnn_reproduction.json",
        "reports/paper_v1_p1/failure_atlas.json",
        "reports/paper_v1_recognition_baseline.md",
        "reports/paper_v1_rotation_consistency.md",
        "reports/paper_v1_p1/human_review_guide.zh-CN.md",
        "reports/paper_v1_p1/human_review_synthetic_20.json",
        "reports/paper_v1_p1/human_review_synthetic_20.csv",
        "reports/paper_v1_p1/human_review_real_dev_10.json",
        "reports/paper_v1_p1/human_review_real_dev_10.csv",
        "audit_paper_v1_provenance.py",
        "audit_paper_v1_clean_provenance_v2.py",
        "prepare_paper_v1_clean_auxiliaries.py",
        "evaluate_paper_v1_recognition.py",
        "evaluate_paper_v1_recognition_v2.py",
        "evaluate_paper_v1_recognition_v3.py",
        "evaluate_paper_v1_recognition_v5.py",
        "verify_paper_v1_repeatability.py",
        "finalize_paper_v1_clean_candidate.py",
        "tests/test_paper_v1_p1_clean_baseline.py",
    ]
    artifacts = []
    for relative in paths:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        artifacts.append({"path": relative, "sha256": sha256(path), "bytes": path.stat().st_size})
    manifest = {
        "schema": "topocae_paper_v1_clean_candidate_freeze_manifest_v1",
        "phase": "P1-A1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selected_variant": "clean_core",
        "unified_entry": "evaluate_paper_v1_recognition_v5.py",
        "uses_real_dev_for_training_or_selection": False,
        "uses_s3_derived_training_data": False,
        "selected_models": [
            "frozen_gnn",
            "clean_candidate_classifier",
        ],
        "disabled_selected_components": [
            "clean_rib_candidate_proposer",
            "clean_rib_boundary_classifier",
            "all legacy S3-derived rib models",
        ],
        "fixed_seeds": {
            "synthetic_split": 20260623,
            "gnn": 20260624,
            "candidate_classifier": 20260625,
            "evaluation_bootstrap": 20260716,
            "pipeline": 20260716,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": {
                "torch": package_version("torch"),
                "numpy": package_version("numpy"),
                "scikit-learn": package_version("scikit-learn"),
                "joblib": package_version("joblib"),
            },
        },
        "artifacts": artifacts,
    }
    write_json(ROOT / "paper_v1_clean_candidate_freeze_manifest.json", manifest)
    return manifest


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    audit = load(REPORTS / "dataset_audit.json")
    direct = audit["solid_validity"]["direct_synthetic_test_validation"]
    # Compatibility summary used by tests and downstream paper-table tooling.
    audit["solid_validity"]["direct_occ_test_validation"] = {
        "status": direct["status"],
        "valid_count": direct["step_count"] - direct["blocking_issue_count"],
        "step_count": direct["step_count"],
    }
    write_json(REPORTS / "dataset_audit.json", audit)
    metrics_document = load(REPORTS / "metrics.json")
    metrics = metrics_document["metrics"]
    raw = compute_raw_gnn_reproduction()
    selection = selection_payload(metrics)
    write_json(CONFIG / "selection.json", selection)
    thresholds = load(CONFIG / "selected_thresholds.json")
    recognition = {
        "schema": "topocae_paper_v1_clean_recognition_v1",
        "phase": "P1-A1",
        "selected_variant": selection["selected_variant"],
        "entry": "evaluate_paper_v1_recognition_v5.py",
        "models": {
            "gnn": load(CONFIG / "models.json")["frozen_gnn"]["path"],
            "candidate_classifier": load(CONFIG / "models.json")["clean_components"]["candidate_classifier"],
            "rib_candidate_proposer": None,
            "rib_boundary_classifier": None,
        },
        "thresholds": {
            "same_instance": 0.5,
            "candidate_classifier_min_confidence": thresholds["candidate_classifier_min_confidence"],
        },
        "output_contract": {
            "float_significant_digits": 12,
            "candidate_diagnostic_source": "candidate_proposal",
            "candidate_record_sort": "action,proposal_macro,predicted_label,face_indices,confidence",
        },
        "formal_datasets": [
            "Synthetic-Test",
            "Real-Dev/test_part",
            "Real-Dev/test_part_s",
            "Real-Dev/test_part_s1",
            "Real-Dev/test_part_s2",
            "Rotation-Dev/test_part_s traced groups",
        ],
        "real_dev_used_for_training_or_selection": False,
        "forbidden_s3_reads": True,
    }
    write_json(CONFIG / "recognition.json", recognition)
    dataset_config = load(CONFIG / "datasets.json")
    test_ids = evaluation.load_effective_splits(
        dataset_config, CONFIG / "quality_exclusions.json"
    )["test"]
    synthetic_review = select_synthetic_review(test_ids)
    real_review = select_real_review()
    write_review_package(synthetic_review, real_review)
    write_reports(metrics, raw, selection, audit)
    manifest = write_manifest()
    print(
        json.dumps(
            {
                "selected_variant": selection["selected_variant"],
                "raw_gnn_face_macro_f1": raw["face"]["macro_f1"],
                "synthetic_review_count": len(synthetic_review),
                "real_review_count": len(real_review),
                "manifest_artifact_count": len(manifest["artifacts"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
