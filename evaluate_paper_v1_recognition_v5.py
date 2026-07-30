from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import evaluate_paper_v1_recognition as base
from evaluate_paper_v1_recognition_v2 import fast_bootstrap_object_metrics
from evaluate_paper_v1_recognition_v3 import resumable_cached_predict_dataset
from evaluate_paper_v1_recognition_v4 import normalize_floats


def normalize_output_contract(payload: Any) -> Any:
    """Freeze diagnostics whose candidate de-duplication order has no semantic effect."""
    value = normalize_floats(payload)
    if not isinstance(value, dict):
        return value
    postprocess = value.get("candidate_classifier_postprocess")
    if isinstance(postprocess, dict) and isinstance(postprocess.get("records"), list):
        records = []
        for original in postprocess["records"]:
            record = dict(original)
            if record.get("source") in {"prediction_component", "topology_rule"}:
                record["source"] = "candidate_proposal"
            if isinstance(record.get("face_indices"), list):
                record["face_indices"] = sorted(record["face_indices"])
            records.append(record)
        postprocess["records"] = sorted(
            records,
            key=lambda record: (
                str(record.get("action")),
                str(record.get("proposal_macro")),
                str(record.get("predicted_label")),
                tuple(record.get("face_indices", [])),
                float(record.get("confidence", 0.0)),
            ),
        )
    return value


def deterministic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(normalize_output_contract(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def normalize_existing_run(run_root: Path) -> dict[str, Any]:
    files = sorted(
        path
        for path in run_root.rglob("*.json")
        if path.parent.name in {"predictions", "feature_objects"}
    )
    for path in files:
        deterministic_write_json(path, json.loads(path.read_text(encoding="utf-8")))
    return {
        "run_root": str(run_root.resolve()),
        "normalized_file_count": len(files),
        "serialized_float_significant_digits": 12,
        "candidate_diagnostic_source": "candidate_proposal",
        "candidate_records_order": "action,proposal_macro,predicted_label,face_indices,confidence",
    }


def main() -> None:
    if "--normalize-run" in sys.argv:
        parser = argparse.ArgumentParser(
            description="Normalize an existing run to the frozen paper-v1 output contract."
        )
        parser.add_argument("--normalize-run", type=Path, required=True)
        args = parser.parse_args()
        print(json.dumps(normalize_existing_run(args.normalize_run.resolve()), indent=2))
        return
    base.bootstrap_object_metrics = fast_bootstrap_object_metrics
    base.predict_dataset = resumable_cached_predict_dataset
    base.write_json = deterministic_write_json
    base.main()


if __name__ == "__main__":
    main()
