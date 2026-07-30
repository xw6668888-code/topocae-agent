from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parent
REPORT_ROOT = REPO_ROOT / "reports" / "paper_v1_p1_a2"
PACKAGE_ROOT = REPORT_ROOT / "human_review_package_v2"
FREEZE_MANIFEST = REPO_ROOT / "paper_v1_clean_candidate_v2_freeze_manifest.json"
EXPECTED_FREEZE_MANIFEST_SHA256 = (
    "7f3d9ee54111a33d4f38e333ef9d3afddd3c3d187f2eab94213db52bb4670847"
)
HOLDOUT_ROOT = REPO_ROOT / "output_generation_paper_v1_synthetic_holdout_v1"
HOLDOUT_FORMAL_ROOT = REPORT_ROOT / "formal" / "holdout_once"
REAL_FORMAL_ROOT = REPORT_ROOT / "formal" / "repeat_1"
AREA_TRUTH_PATH = REPO_ROOT / "configs" / "test_part_s2_area_truth_v5_4_g_looprole.json"

SYNTHETIC_CLASSES = (
    "stepped_hole_round",
    "through_hole_round",
    "blind_hole_round",
    "counterbore_hole",
    "countersink_hole",
    "obround_hole",
    "slot_u",
    "slot_v",
    "slot_o",
    "slot_rect",
    "pocket_rect",
    "pocket_circular",
    "cylinder_boss",
    "rectangular_boss",
    "rib",
    "web",
    "fillet",
    "chamfer",
)
SYNTHETIC_SAMPLE_IDS = tuple(
    [f"holdout_{class_name}_000" for class_name in SYNTHETIC_CLASSES]
    + ["holdout_stepped_hole_round_001", "holdout_through_hole_round_001"]
)

REAL_GRAPH_ROOTS = {
    "test_part_s": REPO_ROOT / "output_real_step_s_v5_6_b_rib_boundary" / "graphs",
    "test_part_s1": REPO_ROOT / "output_real_step_s1_v5_6_b_rib_boundary" / "graphs",
    "test_part_s2": REPO_ROOT / "output_real_step_s2_v5_4_g_looprole" / "graphs",
}
REAL_SELECTION = (
    ("test_part_s", "part11", "矩形 boss、孔和 pocket 的简单组合"),
    ("test_part_s", "part21", "盲孔、阶梯孔、沉孔和通孔组合"),
    ("test_part_s", "part31", "槽、fillet/chamfer 与 web 组合"),
    ("test_part_s1", "part11", "boss、孔和 transition 的低对象数样本"),
    ("test_part_s1", "part31", "圆柱 boss、孔和 web 组合"),
    ("test_part_s2", "part1", "已有面积真值的简单通孔与 transition 样本"),
    ("test_part_s2", "part2", "已有面积真值的孔、slot 和 transition 样本"),
    ("test_part_s2", "part4", "已有面积真值的 rib、孔和 slot 样本"),
    ("test_part_s2", "part6", "已有面积真值的 rib、slot 和 transition 样本"),
    ("test_part_s2", "part11", "已有面积真值的 U/V/O 槽和 counterbore 样本"),
)

REVIEW_RESULT_OPTIONS = ("PASS", "FAIL", "INVALID_TEST_GEOMETRY")
ERROR_TYPE_OPTIONS = (
    "NONE",
    "MISS",
    "FALSE_POSITIVE",
    "FRAGMENT",
    "MERGE",
    "ROLE_ERROR",
    "GEOMETRY_LABEL_ERROR",
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _rounded(value: Any) -> float:
    return round(float(value), 6)


def _require_files(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing review source files:\n" + "\n".join(missing))


def verify_frozen_sources() -> dict[str, Any]:
    actual_manifest_sha = sha256_file(FREEZE_MANIFEST)
    if actual_manifest_sha != EXPECTED_FREEZE_MANIFEST_SHA256:
        raise RuntimeError(
            "P1-A2 freeze manifest changed; P1-A2R must not rebuild against a different freeze"
        )
    manifest = load_json(FREEZE_MANIFEST)
    if manifest.get("selected_variant") != "A0":
        raise RuntimeError("P1-A2R requires the frozen A0 review outputs")
    if manifest.get("holdout_inference_pass_count") != 1:
        raise RuntimeError("Holdout inference pass count is not the frozen value 1")
    mismatches = []
    for artifact in manifest["artifacts"]:
        path = Path(artifact["path"])
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.is_file() or sha256_file(path) != artifact["sha256"]:
            mismatches.append(artifact["path"])
    if mismatches:
        raise RuntimeError("Frozen artifact hash mismatch: " + ", ".join(mismatches))
    return {
        "path": str(FREEZE_MANIFEST.resolve()),
        "sha256": actual_manifest_sha,
        "selected_variant": "A0",
        "artifact_count": len(manifest["artifacts"]),
        "artifact_hashes_verified": True,
        "holdout_inference_pass_count": 1,
    }


def _semantic_role(graph: dict, node: dict) -> str:
    mapping = graph.get("label_maps", {}).get("face_semantic_subtype", {}).get(
        "id_to_key", {}
    )
    semantic_id = node.get("semantic_label", node.get("semantic_subtype_id", 0))
    return str(mapping.get(str(semantic_id), mapping.get(semantic_id, "unknown_semantic_role")))


def build_truth_objects(graph: dict) -> list[dict[str, Any]]:
    node_by_id = {int(node["idx"]): node for node in graph.get("nodes", [])}
    objects = []
    for feature_set in graph.get("feature_sets", []):
        face_ids = sorted(int(value) for value in feature_set.get("node_indices", []))
        faces = []
        for face_id in face_ids:
            node = node_by_id[face_id]
            faces.append(
                {
                    "face_id": face_id,
                    "area": _rounded(node["attributes"]["physical"]["area"]),
                    "semantic_role": _semantic_role(graph, node),
                    "primitive_surface": str(node.get("primitive_surface", "unknown")),
                }
            )
        objects.append(
            {
                "object_id": f"truth_instance_{feature_set['instance_id']}",
                "macro": str(feature_set["macro_feature"]),
                "instance_id": int(feature_set["instance_id"]),
                "face_ids": face_ids,
                "face_count": len(face_ids),
                "total_area": _rounded(sum(face["area"] for face in faces)),
                "semantic_roles": sorted({face["semantic_role"] for face in faces}),
                "faces": faces,
                "source_generator": feature_set.get("source_generator"),
                "generation_context": {
                    key: feature_set.get("raw_parameters", {}).get(key)
                    for key in ("placement_mode", "connects_main_faces", "side")
                    if key in feature_set.get("raw_parameters", {})
                },
            }
        )
    return objects


def build_prediction_objects(
    graph: dict, prediction: dict, feature_document: dict
) -> list[dict[str, Any]]:
    node_by_id = {int(node["idx"]): node for node in graph.get("nodes", [])}
    prediction_by_id = {int(face["idx"]): face for face in prediction.get("faces", [])}
    objects = []
    for feature in feature_document.get("feature_objects", []):
        face_ids = sorted(int(value) for value in feature.get("face_indices", []))
        faces = []
        for face_id in face_ids:
            node = node_by_id[face_id]
            predicted_face = prediction_by_id[face_id]
            faces.append(
                {
                    "face_id": face_id,
                    "area": _rounded(node["attributes"]["physical"]["area"]),
                    "semantic_role": str(predicted_face["semantic_subtype"]),
                    "primitive_surface": str(node.get("primitive_surface", "unknown")),
                    "macro_confidence": _rounded(predicted_face["macro_confidence"]),
                    "semantic_confidence": _rounded(
                        predicted_face["semantic_confidence"]
                    ),
                }
            )
        confidence = feature.get("confidence", {})
        objects.append(
            {
                "object_id": str(feature["feature_object_id"]),
                "macro": str(feature["macro_feature"]),
                "face_ids": face_ids,
                "face_count": len(face_ids),
                "total_area": _rounded(sum(face["area"] for face in faces)),
                "semantic_roles": sorted({face["semantic_role"] for face in faces}),
                "confidence": {
                    "macro_mean": _rounded(confidence.get("macro_confidence_mean", 0.0)),
                    "semantic_mean": _rounded(
                        confidence.get("semantic_confidence_mean", 0.0)
                    ),
                },
                "faces": faces,
            }
        )
    return objects


def build_synthetic_hints(
    truth_objects: list[dict[str, Any]], prediction_objects: list[dict[str, Any]]
) -> dict[str, Any]:
    hints: list[dict[str, Any]] = []
    for truth in truth_objects:
        truth_faces = set(truth["face_ids"])
        matches = [
            item
            for item in prediction_objects
            if item["macro"] == truth["macro"]
            and truth_faces.intersection(item["face_ids"])
        ]
        if not matches:
            hints.append(
                {
                    "type": "POSSIBLE_MISS",
                    "truth_object_id": truth["object_id"],
                    "detail": "没有同 macro 且 face 有重叠的预测对象",
                }
            )
        elif len(matches) > 1:
            hints.append(
                {
                    "type": "POSSIBLE_FRAGMENT",
                    "truth_object_id": truth["object_id"],
                    "prediction_object_ids": [item["object_id"] for item in matches],
                    "detail": "一个真值对象与多个同 macro 预测对象重叠",
                }
            )
    for predicted in prediction_objects:
        predicted_faces = set(predicted["face_ids"])
        matches = [
            item
            for item in truth_objects
            if item["macro"] == predicted["macro"]
            and predicted_faces.intersection(item["face_ids"])
        ]
        if not matches:
            hints.append(
                {
                    "type": "POSSIBLE_FALSE_POSITIVE",
                    "prediction_object_id": predicted["object_id"],
                    "detail": "没有同 macro 且 face 有重叠的真值对象",
                }
            )
        elif len(matches) > 1:
            hints.append(
                {
                    "type": "POSSIBLE_MERGE",
                    "prediction_object_id": predicted["object_id"],
                    "truth_object_ids": [item["object_id"] for item in matches],
                    "detail": "一个预测对象与多个同 macro 真值对象重叠",
                }
            )
    counts = Counter(item["type"] for item in hints)
    return {
        "method": "same-macro face-overlap heuristic",
        "not_a_human_conclusion": True,
        "counts": dict(sorted(counts.items())),
        "items": hints,
    }


def _sample_paths(sample_id: str, *, synthetic: bool, dataset: str = "") -> dict[str, Path]:
    if synthetic:
        return {
            "step_path": HOLDOUT_ROOT / "steps" / f"{sample_id}.step",
            "graph_path": HOLDOUT_ROOT / "graphs" / f"{sample_id}.json",
            "prediction_path": HOLDOUT_FORMAL_ROOT / "predictions" / f"{sample_id}.json",
            "feature_objects_path": HOLDOUT_FORMAL_ROOT
            / "feature_objects"
            / f"{sample_id}.json",
        }
    return {
        "step_path": REPO_ROOT / dataset / f"{sample_id}.stp",
        "graph_path": REAL_GRAPH_ROOTS[dataset] / f"{sample_id}.json",
        "prediction_path": REAL_FORMAL_ROOT
        / dataset
        / "predictions"
        / f"{sample_id}.json",
        "feature_objects_path": REAL_FORMAL_ROOT
        / dataset
        / "feature_objects"
        / f"{sample_id}.json",
    }


def build_synthetic_samples() -> list[dict[str, Any]]:
    samples = []
    for sample_id in SYNTHETIC_SAMPLE_IDS:
        target_class = sample_id.removeprefix("holdout_").rsplit("_", 1)[0]
        paths = _sample_paths(sample_id, synthetic=True)
        _require_files(paths.values())
        graph = load_json(paths["graph_path"])
        prediction = load_json(paths["prediction_path"])
        feature_document = load_json(paths["feature_objects_path"])
        truth_objects = build_truth_objects(graph)
        prediction_objects = build_prediction_objects(graph, prediction, feature_document)
        special_question = ""
        if sample_id == "holdout_rib_000":
            special_question = (
                "该 rib 是否连接 L 形两板或位于腔内？若只是普通块/板边界上的孤立薄板，"
                "请填写 review_result=INVALID_TEST_GEOMETRY。"
            )
        samples.append(
            {
                "review_id": f"SYN-{len(samples) + 1:02d}",
                "dataset": "Synthetic-Holdout-v1",
                "sample_id": sample_id,
                "target_class": target_class,
                **{key: str(value.resolve()) for key, value in paths.items()},
                "source_sha256": {
                    key: sha256_file(value) for key, value in paths.items()
                },
                "truth_object_count": len(truth_objects),
                "prediction_object_count": len(prediction_objects),
                "truth_objects": truth_objects,
                "prediction_objects": prediction_objects,
                "automated_hints": build_synthetic_hints(
                    truth_objects, prediction_objects
                ),
                "special_question": special_question,
                "review_result": "",
                "error_type": "",
                "review_note": "",
            }
        )
    return samples


def _real_risk_hints(
    prediction_objects: list[dict[str, Any]], reference_truth: dict[str, Any] | None
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for item in prediction_objects:
        if min(item["confidence"].values()) < 0.6:
            items.append(
                {
                    "type": "LOW_CONFIDENCE_OBJECT",
                    "object_id": item["object_id"],
                    "macro": item["macro"],
                    "confidence": item["confidence"],
                }
            )
        if item["macro"] in {"rib", "web"}:
            items.append(
                {
                    "type": "THIN_WALL_GEOMETRY_REVIEW",
                    "object_id": item["object_id"],
                    "macro": item["macro"],
                }
            )
    if len(prediction_objects) >= 15:
        items.append(
            {
                "type": "HIGH_PREDICTED_OBJECT_COUNT",
                "prediction_object_count": len(prediction_objects),
            }
        )
    if reference_truth is not None:
        items.append(
            {
                "type": "AREA_TRUTH_REFERENCE_AVAILABLE",
                "detail": "仅提供 macro/面积/face_count 参考，不含 face ID，不代替人工判断",
            }
        )
    minimum_confidence = min(
        (
            min(item["confidence"]["macro_mean"], item["confidence"]["semantic_mean"])
            for item in prediction_objects
        ),
        default=1.0,
    )
    return {
        "not_a_human_conclusion": True,
        "risk_score": _rounded(max(0.0, 1.0 - minimum_confidence)),
        "items": items,
    }


def _area_truth_reference(area_truth: dict, dataset: str, sample_id: str) -> dict[str, Any] | None:
    if dataset != "test_part_s2":
        return None
    records = area_truth.get("samples", {}).get(sample_id)
    if records is None:
        return None
    macro_counts = Counter(str(record["macro"]) for record in records)
    return {
        "status": "area_truth_only_no_face_ids",
        "source_path": str(AREA_TRUTH_PATH.resolve()),
        "object_count": len(records),
        "macro_counts": dict(sorted(macro_counts.items())),
        "objects": [
            {
                "macro": str(record["macro"]),
                "area": _rounded(record["area"]),
                "face_count": int(record["face_count"]),
                **({"note": record["note"]} if record.get("note") else {}),
            }
            for record in records
        ],
    }


def build_real_samples() -> list[dict[str, Any]]:
    area_truth = load_json(AREA_TRUTH_PATH)
    samples = []
    for dataset, sample_id, selection_reason in REAL_SELECTION:
        paths = _sample_paths(sample_id, synthetic=False, dataset=dataset)
        _require_files(paths.values())
        if paths["step_path"].suffix.lower() != ".stp":
            raise RuntimeError(f"Real-Dev review source is not .stp: {paths['step_path']}")
        graph = load_json(paths["graph_path"])
        prediction = load_json(paths["prediction_path"])
        feature_document = load_json(paths["feature_objects_path"])
        prediction_objects = build_prediction_objects(graph, prediction, feature_document)
        reference_truth = _area_truth_reference(area_truth, dataset, sample_id)
        samples.append(
            {
                "review_id": f"REAL-{len(samples) + 1:02d}",
                "dataset": dataset,
                "sample_id": sample_id,
                "selection_reason": selection_reason,
                **{key: str(value.resolve()) for key, value in paths.items()},
                "source_sha256": {
                    key: sha256_file(value) for key, value in paths.items()
                },
                "prediction_object_count": len(prediction_objects),
                "prediction_objects": prediction_objects,
                "area_truth_reference": reference_truth,
                "risk_hints": _real_risk_hints(
                    prediction_objects, reference_truth
                ),
                "review_result": "",
                "error_type": "",
                "review_note": "",
            }
        )
    return samples


def _compact_objects(objects: list[dict[str, Any]], *, predicted: bool) -> str:
    chunks = []
    for item in objects:
        text = (
            f"{item['object_id']} macro={item['macro']} faces={item['face_ids']} "
            f"area={item['total_area']:.6f} roles={item['semantic_roles']}"
        )
        if predicted:
            text += (
                f" conf={item['confidence']['macro_mean']:.6f}/"
                f"{item['confidence']['semantic_mean']:.6f}"
            )
        chunks.append(text)
    return " | ".join(chunks)


def _compact_hints(payload: dict[str, Any]) -> str:
    items = payload.get("items", [])
    if not items:
        return "NONE_DETECTED_BY_AUTOMATED_HEURISTIC"
    return " | ".join(
        f"{item['type']}:{item.get('object_id') or item.get('truth_object_id') or item.get('prediction_object_id') or ''}"
        for item in items
    )


def write_review_csvs(
    synthetic_samples: list[dict[str, Any]], real_samples: list[dict[str, Any]]
) -> tuple[Path, Path]:
    synthetic_path = PACKAGE_ROOT / "human_review_synthetic_20.csv"
    real_path = PACKAGE_ROOT / "human_review_real_dev_10.csv"
    synthetic_fields = [
        "review_id",
        "dataset",
        "sample_id",
        "target_class",
        "step_path",
        "graph_path",
        "prediction_path",
        "feature_objects_path",
        "truth_object_count",
        "prediction_object_count",
        "truth_summary",
        "prediction_summary",
        "automated_hints",
        "special_question",
        "review_result",
        "error_type",
        "review_note",
    ]
    real_fields = [
        "review_id",
        "dataset",
        "sample_id",
        "selection_reason",
        "step_path",
        "graph_path",
        "prediction_path",
        "feature_objects_path",
        "prediction_object_count",
        "reference_truth_available",
        "prediction_summary",
        "risk_hints",
        "review_result",
        "error_type",
        "review_note",
    ]
    with synthetic_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=synthetic_fields)
        writer.writeheader()
        for sample in synthetic_samples:
            writer.writerow(
                {
                    **{key: sample.get(key, "") for key in synthetic_fields},
                    "truth_summary": _compact_objects(
                        sample["truth_objects"], predicted=False
                    ),
                    "prediction_summary": _compact_objects(
                        sample["prediction_objects"], predicted=True
                    ),
                    "automated_hints": _compact_hints(sample["automated_hints"]),
                }
            )
    with real_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=real_fields)
        writer.writeheader()
        for sample in real_samples:
            writer.writerow(
                {
                    **{key: sample.get(key, "") for key in real_fields},
                    "reference_truth_available": (
                        "YES" if sample["area_truth_reference"] is not None else "NO"
                    ),
                    "prediction_summary": _compact_objects(
                        sample["prediction_objects"], predicted=True
                    ),
                    "risk_hints": _compact_hints(sample["risk_hints"]),
                }
            )
    return synthetic_path, real_path


def write_guide() -> Path:
    path = PACKAGE_ROOT / "human_review_guide.zh-CN.md"
    text = """# P1-A2R 人工审查操作说明

本包只用于人工确认冻结的 `A0 Raw-GNN` 输出。不要用审查结果改模型、规则或阈值。JSON 和 HTML 只读；CSV 是唯一回填文件。

## 1. 在 UG/NX 中打开模型

1. 打开 `human_review_synthetic_20.csv` 或 `human_review_real_dev_10.csv`。
2. 逐行复制 `step_path` 的完整绝对路径。
3. 在 UG/NX 中选择“文件 → 打开”，粘贴该路径并打开 STEP/STP；30 个路径均已由生成器验证存在。
4. 不修改或另存源模型。若模型无法打开，在 `review_note` 记录现象。

## 2. 核对 graph 与冻结预测

- `human_review_summary.html` 提供逐对象摘要；需要原始细节时查看 CSV 中的 `graph_path`、`prediction_path` 和 `feature_objects_path`。
- 在 UG/NX 的“分析/测量几何（或测量面）”中查看面面积；用摘要中的 area、macro 和 semantic role 定位并判断对象。
- face ID 是 graph/exporter 的交叉引用，不要求在 UG 中重新批量标注。用户只需借助面积、形状、连接关系和 semantic role 判断整体对象是否合理。
- 预测摘要中的置信度仅用于安排检查优先级，不能代替人工结论。

## 3. Synthetic 与 Real 分别判断什么

- Synthetic 20：对照 truth 与 prediction 摘要，检查 macro、对象数量、face 组成、面积和 semantic role；自动列出的 MISS/FALSE_POSITIVE/FRAGMENT/MERGE 只是 face-overlap 提示。
- Real-Dev 10：没有统一 face-ID 真值时，直接依据 CAD 几何判断预测对象是否存在、是否完整以及 role 是否正确。`test_part_s2` 的 area truth 仅作面积和 macro 参考。

## 4. rib 专门判断

对 `holdout_rib_000` 必须回答：该 rib 是否连接 L 形两板或位于腔内？若只是普通块/板边界上的孤立薄板，填写：

- `review_result = INVALID_TEST_GEOMETRY`
- `error_type = GEOMETRY_LABEL_ERROR`
- 在 `review_note` 简述看到的连接关系

其他 rib/web 仍需检查薄壁是否与主体有合理连接、是否被碎裂或合并。

## 5. 用户只填写三列

- `review_result`：`PASS` / `FAIL` / `INVALID_TEST_GEOMETRY`
- `error_type`：`NONE` / `MISS` / `FALSE_POSITIVE` / `FRAGMENT` / `MERGE` / `ROLE_ERROR` / `GEOMETRY_LABEL_ERROR`
- `review_note`：简短说明判断依据

`PASS` 通常搭配 `NONE`。不要编辑其余列，不要重新标注或改写 face ID。

## 6. 完成后返回

只需返回以下两个已填写文件：

1. `human_review_synthetic_20.csv`
2. `human_review_real_dev_10.csv`
"""
    path.write_text(text, encoding="utf-8")
    return path


def _path_html(path_text: str) -> str:
    path = Path(path_text)
    return (
        f'<a href="{html.escape(path.as_uri(), quote=True)}"><code>'
        f"{html.escape(path_text)}</code></a>"
    )


def _objects_html(objects: list[dict[str, Any]], *, predicted: bool) -> str:
    rows = []
    for item in objects:
        face_details = "<br>".join(
            html.escape(
                f"face {face['face_id']}: area={face['area']:.6f}, role={face['semantic_role']}"
                + (
                    f", conf={face['macro_confidence']:.6f}/{face['semantic_confidence']:.6f}"
                    if predicted
                    else ""
                )
            )
            for face in item["faces"]
        )
        confidence = ""
        if predicted:
            confidence = (
                f"{item['confidence']['macro_mean']:.6f} / "
                f"{item['confidence']['semantic_mean']:.6f}"
            )
        rows.append(
            "<tr>"
            f"<td>{html.escape(item['object_id'])}</td>"
            f"<td>{html.escape(item['macro'])}</td>"
            f"<td>{html.escape(str(item['face_ids']))}</td>"
            f"<td>{item['total_area']:.6f}</td>"
            f"<td>{html.escape(', '.join(item['semantic_roles']))}</td>"
            f"<td>{html.escape(confidence)}</td>"
            f"<td>{face_details}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Object</th><th>Macro</th><th>Face IDs</th>"
        "<th>Total area</th><th>Semantic roles</th><th>Macro/Semantic conf.</th>"
        "<th>Face details</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def write_html(
    synthetic_samples: list[dict[str, Any]], real_samples: list[dict[str, Any]]
) -> Path:
    path = PACKAGE_ROOT / "human_review_summary.html"
    sections = []
    for sample in synthetic_samples:
        special = (
            f'<p class="warning"><strong>专门问题：</strong>{html.escape(sample["special_question"])}</p>'
            if sample["special_question"]
            else ""
        )
        sections.append(
            f'<section id="{sample["review_id"]}"><h3>{sample["review_id"]} — '
            f'{html.escape(sample["sample_id"])} ({html.escape(sample["target_class"])})</h3>'
            f'<p>STEP: {_path_html(sample["step_path"])}</p>{special}'
            f'<p>Truth objects: {sample["truth_object_count"]}; prediction objects: '
            f'{sample["prediction_object_count"]}; automated hints are not conclusions.</p>'
            "<h4>Truth</h4>"
            + _objects_html(sample["truth_objects"], predicted=False)
            + "<h4>Frozen prediction</h4>"
            + _objects_html(sample["prediction_objects"], predicted=True)
            + f'<pre>{html.escape(json.dumps(sample["automated_hints"], ensure_ascii=False, indent=2))}</pre>'
            "</section>"
        )
    for sample in real_samples:
        reference = sample["area_truth_reference"]
        reference_html = (
            f'<pre>{html.escape(json.dumps(reference, ensure_ascii=False, indent=2))}</pre>'
            if reference is not None
            else "<p>无统一对象真值；按 CAD 几何人工判断。</p>"
        )
        sections.append(
            f'<section id="{sample["review_id"]}"><h3>{sample["review_id"]} — '
            f'{html.escape(sample["dataset"] + "/" + sample["sample_id"])}</h3>'
            f'<p>STEP: {_path_html(sample["step_path"])}</p>'
            f'<p>选择理由：{html.escape(sample["selection_reason"])}</p>'
            f'<p>Prediction objects: {sample["prediction_object_count"]}; risk hints are not conclusions.</p>'
            + _objects_html(sample["prediction_objects"], predicted=True)
            + "<h4>Existing reference</h4>"
            + reference_html
            + f'<pre>{html.escape(json.dumps(sample["risk_hints"], ensure_ascii=False, indent=2))}</pre>'
            + "</section>"
        )
    document = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>P1-A2R 人工审查摘要</title>
<style>body{font-family:Segoe UI,Microsoft YaHei,sans-serif;margin:2rem;line-height:1.5}table{border-collapse:collapse;width:100%;font-size:.88rem}th,td{border:1px solid #bbb;padding:.35rem;vertical-align:top}th{background:#eee}section{margin:2rem 0;padding-top:1rem;border-top:3px solid #555}.notice{background:#eef6ff;padding:1rem}.warning{background:#fff2cc;padding:1rem;border-left:5px solid #d88a00}code,pre{white-space:pre-wrap;word-break:break-all}</style>
</head><body><h1>P1-A2R 人工审查只读摘要</h1>
<p class="notice">本 HTML 只用于浏览。唯一回填文件是 human_review_synthetic_20.csv 和 human_review_real_dev_10.csv；只填写 review_result、error_type、review_note。</p>
""" + "\n".join(sections) + "\n</body></html>\n"
    path.write_text(document, encoding="utf-8")
    return path


def _guard_unsubmitted_csv(path: Path) -> None:
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if any(row.get(key, "").strip() for key in ("review_result", "error_type", "review_note")):
                raise RuntimeError(f"Refusing to overwrite user-filled review CSV: {path}")


def write_manifest(
    frozen_guard: dict[str, Any],
    package_files: list[Path],
    synthetic_samples: list[dict[str, Any]],
    real_samples: list[dict[str, Any]],
) -> Path:
    path = PACKAGE_ROOT / "review_package_manifest.json"
    files = [
        {
            "path": item.relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256_file(item),
            "bytes": item.stat().st_size,
        }
        for item in sorted(package_files)
    ]
    implementation_paths = [
        Path(__file__).resolve(),
        REPO_ROOT / "tests" / "test_paper_v1_p1_a2_human_review_package.py",
    ]
    implementation_files = [
        {
            "path": item.relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256_file(item),
            "bytes": item.stat().st_size,
        }
        for item in implementation_paths
    ]
    payload = {
        "schema": "topocae_p1_a2r_human_review_package_manifest_v1",
        "phase": "P1-A2R",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "human-review-material-only; no inference or metric recomputation",
        "frozen_source_guard": frozen_guard,
        "freeze_manifest_modified": False,
        "synthetic_sample_count": len(synthetic_samples),
        "synthetic_class_coverage": sorted(
            {sample["target_class"] for sample in synthetic_samples}
        ),
        "real_sample_count": len(real_samples),
        "real_directory_coverage": sorted(
            {sample["dataset"] for sample in real_samples}
        ),
        "user_return_files": [
            "human_review_synthetic_20.csv",
            "human_review_real_dev_10.csv",
        ],
        "review_result_options": list(REVIEW_RESULT_OPTIONS),
        "error_type_options": list(ERROR_TYPE_OPTIONS),
        "files": files,
        "implementation_files": implementation_files,
        "manifest_excludes_self_hash": True,
    }
    write_json(path, payload)
    return path


def build_package() -> dict[str, Any]:
    frozen_guard = verify_frozen_sources()
    PACKAGE_ROOT.mkdir(parents=True, exist_ok=True)
    for name in ("human_review_synthetic_20.csv", "human_review_real_dev_10.csv"):
        _guard_unsubmitted_csv(PACKAGE_ROOT / name)
    synthetic_samples = build_synthetic_samples()
    real_samples = build_real_samples()
    synthetic_json = PACKAGE_ROOT / "human_review_synthetic_20.json"
    real_json = PACKAGE_ROOT / "human_review_real_dev_10.json"
    write_json(
        synthetic_json,
        {
            "schema": "topocae_p1_a2r_synthetic_human_review_v2",
            "selected_variant": "A0 Raw-GNN",
            "source_inference_reused": True,
            "samples": synthetic_samples,
        },
    )
    write_json(
        real_json,
        {
            "schema": "topocae_p1_a2r_real_dev_human_review_v2",
            "selected_variant": "A0 Raw-GNN",
            "source_inference_reused": True,
            "samples": real_samples,
        },
    )
    synthetic_csv, real_csv = write_review_csvs(synthetic_samples, real_samples)
    guide = write_guide()
    summary_html = write_html(synthetic_samples, real_samples)
    package_files = [
        synthetic_json,
        real_json,
        synthetic_csv,
        real_csv,
        guide,
        summary_html,
    ]
    manifest_path = write_manifest(
        frozen_guard, package_files, synthetic_samples, real_samples
    )
    if sha256_file(FREEZE_MANIFEST) != EXPECTED_FREEZE_MANIFEST_SHA256:
        raise RuntimeError("Freeze manifest changed while building the review package")
    return {
        "package_root": str(PACKAGE_ROOT.resolve()),
        "synthetic_sample_count": len(synthetic_samples),
        "synthetic_class_count": len(
            {sample["target_class"] for sample in synthetic_samples}
        ),
        "real_sample_count": len(real_samples),
        "real_directory_count": len({sample["dataset"] for sample in real_samples}),
        "contains_holdout_rib_000": any(
            sample["sample_id"] == "holdout_rib_000" for sample in synthetic_samples
        ),
        "freeze_manifest_sha256": sha256_file(FREEZE_MANIFEST),
        "review_package_manifest": str(manifest_path.resolve()),
    }


def main() -> None:
    print(json.dumps(build_package(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
