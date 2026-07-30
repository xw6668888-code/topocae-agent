from __future__ import annotations

"""Export three exact train-split geometries per P1-B class as STEP review copies."""

import csv
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from OCC.Core.BRepTools import breptools

from utils.step_solid_validation import inspect_step_file, write_step_with_positive_volume


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "output_generation_p1_b_corrected_balanced_5000_v3"
TRAIN = DATASET / "splits" / "train.jsonl"
OUTPUT = ROOT / "reports" / "p1_b_training_step_spotcheck_54"
STEP_DIR = OUTPUT / "steps"


def main() -> int:
    classes = list(_read_json(DATASET / "dataset_meta.json")["target_classes"])
    records = [json.loads(line) for line in TRAIN.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected: dict[str, list[dict[str, Any]]] = {}
    for macro in classes:
        candidates = [record for record in records if int(record.get("macro_instance_counts", {}).get(macro, 0)) > 0]
        candidates.sort(key=lambda record: (
            len(record.get("macro_instance_counts", {})) != 1,
            record.get("derivation_kind") != "identity",
            record.get("geometry_format") != "STEP",
            record["sample_id"],
        ))
        chosen = []
        used_base = set()
        for record in candidates:
            base_id = str(record["base_geometry_family_id"])
            if base_id in used_base:
                continue
            chosen.append(record); used_base.add(base_id)
            if len(chosen) == 3:
                break
        if len(chosen) != 3:
            raise RuntimeError(f"cannot find three distinct train base families for {macro}")
        selected[macro] = chosen
    STEP_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for macro in classes:
        for rank, record in enumerate(selected[macro], start=1):
            source = Path(record["geometry_path"])
            destination = STEP_DIR / f"{macro}__{rank:02d}__{record['sample_id']}.step"
            if record["geometry_format"] == "STEP":
                shutil.copy2(source, destination)
                export_method = "exact_STEP_copy"
            elif record["geometry_format"] == "OpenCASCADE_BREP_text":
                shape = breptools.ReadFromString(source.read_text(encoding="utf-8"))
                write_step_with_positive_volume(shape, destination)
                export_method = "exact_BREP_to_STEP_review_export"
            else:
                raise ValueError(f"unsupported geometry format: {record['geometry_format']}")
            validation = inspect_step_file(destination)
            valid = bool(validation.get("readable") and validation.get("solid_count") == 1 and validation.get("valid") and not validation.get("issues"))
            if not valid:
                raise RuntimeError(f"review STEP failed validation: {destination}")
            rows.append({
                "macro_feature": macro,
                "review_rank": rank,
                "file_name": destination.name,
                "sample_id": record["sample_id"],
                "split": record["split"],
                "source_geometry_format": record["geometry_format"],
                "export_method": export_method,
                "source_geometry_path": str(source.resolve()),
                "source_geometry_sha256": record["geometry_sha256"],
                "review_step_path": str(destination.resolve()),
                "review_step_sha256": _sha256(destination),
                "base_geometry_family_id": record["base_geometry_family_id"],
                "all_macro_instance_counts": json.dumps(record["macro_instance_counts"], ensure_ascii=False, sort_keys=True),
                "valid_single_solid": valid,
            })
    _write_csv(OUTPUT / "spotcheck_manifest.csv", rows)
    manifest = {
        "schema_version": "topocae_p1_b_training_step_spotcheck_54_1",
        "source_dataset": str(DATASET.resolve()),
        "source_split": "train",
        "class_count": len(classes),
        "samples_per_class": 3,
        "step_count": len(rows),
        "all_valid_single_solid": all(row["valid_single_solid"] for row in rows),
        "selection_contract": "three distinct base families per class; prefer single-class identity STEP when available",
        "conversion_contract": "BREP sources are exact training geometry exported to STEP only for human inspection; source dataset is unchanged",
        "classes": {macro: [row["file_name"] for row in rows if row["macro_feature"] == macro] for macro in classes},
        "records": rows,
    }
    _write_json(OUTPUT / "spotcheck_manifest.json", manifest)
    (OUTPUT / "README.zh-CN.md").write_text(
        "# P1-B 训练 STEP 抽查包\n\n"
        "本目录从正式 train split 为18类各选3个不同 base family，共54件。原始 STEP 为逐字节复制；原始 BREP 为同一训练几何的 STEP 审查导出。"
        "这些文件仅供人工打开，不参与训练、调参或 Holdout。每件来源和 SHA-256 见 spotcheck_manifest.csv/json。\n",
        encoding="utf-8",
    )
    print(json.dumps({"automatic_pass": True, "class_count": len(classes), "step_count": len(rows), "output": str(STEP_DIR)}, ensure_ascii=False))
    return 0


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


if __name__ == "__main__":
    raise SystemExit(main())
