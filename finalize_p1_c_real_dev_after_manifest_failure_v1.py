from __future__ import annotations

"""Resume only Real-Dev reporting after all 26 formal inference samples completed."""

from datetime import datetime, timezone
import json
from pathlib import Path

import run_p1_c_real_dev_evaluation_v1 as evaluation


OUTPUT = evaluation.OUTPUT
RECEIPT = OUTPUT / "execution_receipt.json"


def main() -> int:
    expected = 26
    inventories = {
        "graphs_v2": sorted((OUTPUT / "graphs_v2").glob("*/*.json")),
        "predictions": sorted((OUTPUT / "predictions").glob("*/*.json")),
        "feature_objects": sorted((OUTPUT / "feature_objects").glob("*/*.json")),
    }
    counts = {name: len(paths) for name, paths in inventories.items()}
    if any(count != expected for count in counts.values()):
        raise RuntimeError(f"Real-Dev inference inventory incomplete: {counts}")
    required_reports = [
        OUTPUT / "metrics.json", OUTPUT / "failure_atlas.json", OUTPUT / "pipeline_comparison.csv",
        OUTPUT / "human_review_comparison.csv", OUTPUT / "failure_atlas.csv", OUTPUT / "timing.csv",
    ]
    missing = [str(path) for path in required_reports if not path.is_file()]
    if missing:
        raise RuntimeError(f"pre-manifest reports missing: {missing}")
    if RECEIPT.exists():
        raise RuntimeError("Real-Dev recovery receipt already exists")

    freeze = evaluation._assert_freeze()
    evaluation._rewrite_outputs(freeze)
    result = evaluation._read(OUTPUT / "metrics.json")
    receipt = {
        "schema_version": "topocae_p1_c_real_dev_execution_receipt_1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_inference_pass_count": 1,
        "formal_inference_completed_sample_count": expected,
        "postprocess_resume_count": 1,
        "inference_recomputed_during_resume": False,
        "inventory_counts": counts,
        "original_failure": "legacy reporter expected area_truth.json inside new output directory after all inference samples had completed",
        "recovery_scope": "manifest and final comparison only",
        "preholdout_freeze_sha256": evaluation._sha(evaluation.FREEZE),
        "contracts": {"test_part_s3_used": False, "real_data_used_for_selection": False},
    }
    evaluation._write(RECEIPT, receipt)
    print(json.dumps({
        "real_dev_complete": True,
        "formal_inference_pass_count": 1,
        "evaluated_steps": result["evaluated_step_count"],
        "area_macro": {key: result["area_macro"][key] for key in ("precision", "recall", "f1")},
        "targets_non_tuning": result["real_dev_targets_non_tuning"],
        "receipt": str(RECEIPT),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
