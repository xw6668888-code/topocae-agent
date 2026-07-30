from __future__ import annotations

"""Read the user-provided real-part area truth workbook into immutable JSON."""

import hashlib
import json
from pathlib import Path
from typing import Any

from openpyxl import load_workbook


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "reports" / "paper_v1_p1_a2" / "human_review_package_v2" / "Feature_Comparison_Table_New.xlsx"
OUTPUT = ROOT / "reports" / "p1_b_real_dev_v1" / "area_truth.json"
ALIASES = {
    "through_hole": "through_hole_round", "blind_hole": "blind_hole_round",
    "stepped_hole": "stepped_hole_round", "poctet_rect": "pocket_rect",
    "rectangle_boss": "rectangular_boss", "cylindrical_boss": "cylinder_boss",
}


def main() -> int:
    workbook = load_workbook(SOURCE, read_only=True, data_only=True)
    records = []
    current_path: Path | None = None
    truth_index = 0
    for sheet in workbook.worksheets:
        for area, macro, face_count, *_ in sheet.iter_rows(values_only=True):
            if isinstance(area, str) and area.strip().lower().endswith((".stp", ".step")):
                current_path = Path(area.strip())
                truth_index = 0
                continue
            if current_path is None or not isinstance(area, (int, float)) or not isinstance(macro, str):
                continue
            dataset = current_path.parent.name
            sample_id = current_path.stem
            if dataset == "test_part_s2" and sample_id == "part10":
                continue
            normalized = ALIASES.get(macro.strip(), macro.strip())
            records.append({
                "dataset": dataset, "sample_id": sample_id,
                "sample_key": f"{dataset}/{sample_id}", "source_step_path": str(current_path),
                "truth_index": truth_index, "macro_feature": normalized,
                "area": float(area), "face_count": int(face_count or 0),
            })
            truth_index += 1
    by_sample: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_sample.setdefault(record["sample_key"], []).append(record)
    payload = {
        "schema_version": "topocae_p1_b_real_area_truth_1",
        "source_xlsx": str(SOURCE.resolve()), "source_xlsx_sha256": _sha256(SOURCE),
        "sheet_names": workbook.sheetnames, "truth_count": len(records),
        "sample_count": len(by_sample), "excluded": ["test_part_s2/part10:user_confirmed_modeling_error"],
        "samples": by_sample,
        "contracts": {"evaluation_only": True, "used_for_training_or_thresholds": False},
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "automatic_pass": len(records) > 0 and len(by_sample) > 0,
        "truth_count": len(records), "sample_count": len(by_sample),
        "source_sha256": payload["source_xlsx_sha256"], "output": str(OUTPUT),
    }, ensure_ascii=False))
    return 0 if records else 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


if __name__ == "__main__":
    raise SystemExit(main())
