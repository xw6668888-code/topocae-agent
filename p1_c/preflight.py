from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROTECTED_RELATIVE_PATHS = (
    "output_generation_p1_b_corrected_balanced_5000_v3",
    "reports/p1_b_training_runs_v1",
    "reports/p1_b_training_ablation_summary_v1",
    "reports/p1_b_real_dev_v1",
    "output_generation_p1_b_synthetic_holdout_v2",
    "reports/p1_b_holdout_v2_once",
    "reports/p1_b_final_acceptance_v1",
)

NEW_RELATIVE_PATHS = (
    "configs/p1_c/generation.json",
    "p1_c_pipeline.py",
    "reports/p1_c_preflight",
    "output_generation_p1_c1_clean_rib_web_pilot_48",
    "output_generation_p1_c2_all_class_canonical_36",
    "output_generation_p1_c_clean_context_7200",
    "reports/p1_c_training_runs_v1",
    "reports/p1_c_training_ablation_summary_v1",
    "reports/p1_c_real_dev_v1",
    "output_generation_p1_c_synthetic_holdout_v3",
    "reports/p1_c_holdout_v3_once",
    "reports/p1_c_final_acceptance_v1",
)

FROZEN_METRICS = {
    "p1_b_dataset_sample_count": 5000,
    "p1_b_dataset_instance_count": 8050,
    "synthetic_val_three_seed_mean": {
        "face_macro_f1": 0.9774265760696488,
        "object_macro_f1": 0.9647108535004254,
        "exact_instance_ratio": 0.9524432809773123,
        "same_instance_edge_f1": 0.9996268830671788,
    },
    "synthetic_holdout_v2": {
        "face_macro_f1": 0.934394,
        "object_macro_f1": 0.919950,
        "exact_instance_ratio": 0.937877,
        "boundary_f1": 0.999212,
    },
    "real_dev_area_macro": {
        "precision": 0.672897,
        "recall": 0.541353,
        "f1": 0.600000,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _files_under(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    if not path.exists():
        return
    for candidate in sorted(path.rglob("*"), key=lambda item: item.as_posix().lower()):
        if candidate.is_file():
            yield candidate


def protected_inventory(root: Path) -> dict[str, Any]:
    directories: list[dict[str, Any]] = []
    all_files: list[dict[str, Any]] = []
    for relative in PROTECTED_RELATIVE_PATHS:
        directory = root / relative
        files: list[dict[str, Any]] = []
        byte_count = 0
        for path in _files_under(directory):
            size = path.stat().st_size
            byte_count += size
            record = {
                "path": path.relative_to(root).as_posix(),
                "bytes": size,
                "sha256": sha256_file(path),
            }
            files.append(record)
            all_files.append(record)
        directories.append(
            {
                "path": relative,
                "exists": directory.exists(),
                "file_count": len(files),
                "byte_count": byte_count,
                "files": files,
            }
        )
    return {
        "schema_version": "topocae_p1_c_protected_inventory_1",
        "root": str(root),
        "directories": directories,
        "total_file_count": len(all_files),
        "total_byte_count": sum(item["bytes"] for item in all_files),
        "aggregate_sha256": _aggregate_hash(all_files),
    }


def _aggregate_hash(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item["path"]):
        digest.update(record["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(record["sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def environment_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "packages": {},
    }
    for import_name, display_name in (
        ("OCC", "OpenCASCADE/pythonocc-core"),
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("sklearn", "scikit-learn"),
        ("networkx", "NetworkX"),
        ("yaml", "PyYAML"),
    ):
        try:
            module = __import__(import_name)
            version = getattr(module, "__version__", getattr(module, "VERSION", "unknown"))
            snapshot["packages"][display_name] = str(version)
        except Exception as exc:  # pragma: no cover - diagnostic path
            snapshot["packages"][display_name] = f"ERROR: {type(exc).__name__}: {exc}"
    try:
        import torch

        cuda: dict[str, Any] = {
            "available": bool(torch.cuda.is_available()),
            "runtime_version": torch.version.cuda,
            "device_count": int(torch.cuda.device_count()),
            "devices": [],
        }
        for index in range(torch.cuda.device_count()):
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            cuda["devices"].append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "free_bytes": int(free_bytes),
                    "total_bytes": int(total_bytes),
                }
            )
        snapshot["cuda"] = cuda
    except Exception as exc:  # pragma: no cover - diagnostic path
        snapshot["cuda"] = {"error": f"{type(exc).__name__}: {exc}"}
    return snapshot


def _key_artifacts(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    tokens = (
        "checkpoint",
        "best_model",
        "threshold",
        "selection",
        "metrics.json",
        "summary.json",
        "acceptance.json",
        "freeze_manifest",
        "dataset_meta.json",
    )
    results: list[dict[str, Any]] = []
    for directory in inventory["directories"]:
        for record in directory["files"]:
            lower = record["path"].lower()
            if any(token in lower for token in tokens) or lower.endswith((".pt", ".pth")):
                results.append(record)
    return results


def build_preflight(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = root.resolve()
    inventory = protected_inventory(root)
    report = {
        "schema_version": "topocae_p1_c_preflight_1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "environment": environment_snapshot(),
        "frozen_metrics": FROZEN_METRICS,
        "protected_summary": {
            "total_file_count": inventory["total_file_count"],
            "total_byte_count": inventory["total_byte_count"],
            "aggregate_sha256": inventory["aggregate_sha256"],
            "directories": [
                {key: item[key] for key in ("path", "exists", "file_count", "byte_count")}
                for item in inventory["directories"]
            ],
        },
        "key_artifact_sha256": _key_artifacts(inventory),
        "new_paths": list(NEW_RELATIVE_PATHS),
        "single_entrypoint": "p1_c_pipeline.py",
        "contracts": {
            "prohibited_directory_accessed": False,
            "real_dev_used_for_development": False,
            "holdout_v2_used_for_development": False,
            "superpower_skills_used": False,
            "p1_b_geometry_reused": False,
            "p2_entered": False,
        },
    }
    return report, inventory


def write_preflight(root: Path) -> dict[str, Any]:
    report, inventory = build_preflight(root)
    output = root / "reports" / "p1_c_preflight"
    output.mkdir(parents=True, exist_ok=True)
    (output / "preflight.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "protected_inventory.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "preflight.zh-CN.md").write_text(_markdown(report), encoding="utf-8")
    return report


def _markdown(report: dict[str, Any]) -> str:
    env = report["environment"]
    lines = [
        "# P1-C 阶段0 Preflight 冻结报告",
        "",
        f"生成时间（UTC）：`{report['generated_at_utc']}`",
        "",
        "## 运行环境",
        "",
        f"- Python：`{env['python_version'].splitlines()[0]}`",
        f"- 解释器：`{env['python_executable']}`",
        f"- 平台：`{env['platform']}`；CPU 逻辑核：`{env['cpu_count']}`",
    ]
    for name, version in env["packages"].items():
        lines.append(f"- {name}：`{version}`")
    cuda = env.get("cuda", {})
    lines.extend(
        [
            f"- CUDA：available=`{cuda.get('available')}`，runtime=`{cuda.get('runtime_version')}`",
        ]
    )
    for device in cuda.get("devices", []):
        lines.append(
            "- GPU {index}：`{name}`，free={free:.2f} GiB / total={total:.2f} GiB".format(
                index=device["index"],
                name=device["name"],
                free=device["free_bytes"] / 2**30,
                total=device["total_bytes"] / 2**30,
            )
        )
    lines.extend(["", "## P1-B 冻结指标", ""])
    metrics = report["frozen_metrics"]
    lines.extend(
        [
            f"- 修正数据集：{metrics['p1_b_dataset_sample_count']} 样本，{metrics['p1_b_dataset_instance_count']} 实例。",
            "- Synthetic-Val 三种子均值：face Macro-F1 "
            f"`{metrics['synthetic_val_three_seed_mean']['face_macro_f1']:.6f}`，object Macro-F1 "
            f"`{metrics['synthetic_val_three_seed_mean']['object_macro_f1']:.6f}`，exact instance "
            f"`{metrics['synthetic_val_three_seed_mean']['exact_instance_ratio']:.6f}`，same-instance edge F1 "
            f"`{metrics['synthetic_val_three_seed_mean']['same_instance_edge_f1']:.6f}`。",
            "- Synthetic-Holdout-v2：face `0.934394`，object `0.919950`，exact `0.937877`，boundary `0.999212`。",
            "- Real-Dev area+macro：P=`0.672897` / R=`0.541353` / F1=`0.600000`。",
            "- 以上数值仅作只读历史对比，禁止重算后覆盖。",
            "",
            "## 受保护目录冻结清单",
            "",
            "| 目录 | 存在 | 文件数 | 字节数 |",
            "|---|---:|---:|---:|",
        ]
    )
    for item in report["protected_summary"]["directories"]:
        lines.append(
            f"| `{item['path']}` | {item['exists']} | {item['file_count']} | {item['byte_count']} |"
        )
    lines.extend(
        [
            "",
            f"聚合 SHA-256：`{report['protected_summary']['aggregate_sha256']}`。",
            "完整逐文件 SHA-256 位于 `protected_inventory.json`；checkpoint、阈值、Real-Dev、Holdout-v2 与最终验收关键文件哈希同时写入 `preflight.json`。",
            "",
            "## 新建范围与入口",
            "",
            f"单一入口：`{report['single_entrypoint']}`。本 Goal 的新建路径：",
            "",
        ]
    )
    lines.extend(f"- `{path}`" for path in report["new_paths"])
    lines.extend(
        [
            "",
            "## 强制声明",
            "",
            "- 未访问被禁止的目录，且后续入口包含拒绝访问契约。",
            "- Real-Dev 不参与开发、选模、阈值或比例选择。",
            "- Synthetic-Holdout-v2 不参与开发、选模、阈值或比例选择。",
            "- 未复用 P1-B 旧 STEP/BREP 或旧修正训练几何。",
            "- 未使用任何 superpower skill，未让 LLM 直接识别 STEP，未进入 P2。",
            "",
        ]
    )
    if len(lines) > 250:
        raise RuntimeError(f"preflight markdown exceeds 250 lines: {len(lines)}")
    return "\n".join(lines)


__all__ = [
    "FROZEN_METRICS",
    "NEW_RELATIVE_PATHS",
    "PROTECTED_RELATIVE_PATHS",
    "build_preflight",
    "protected_inventory",
    "sha256_file",
    "write_preflight",
]
