from __future__ import annotations

"""Write the immutable P1-B selection record before Holdout-v2 exists."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "reports" / "p1_b_freeze_v1"
SELECTION = OUTPUT / "preholdout_selection.json"
SIDECAR = OUTPUT / "preholdout_selection.sha256.json"
RUN = ROOT / "reports" / "p1_b_training_runs_v1" / "same_instance_seed_20260718"

FROZEN_FILES = [
    RUN / "best.pt",
    RUN / "config.json",
    RUN / "metrics.json",
    RUN / "same_instance_threshold_selection.json",
    RUN / "global_decode_v2_selection.json",
    RUN / "candidate_model_v1" / "best.pt",
    RUN / "candidate_model_v1" / "config.json",
    RUN / "candidate_model_v1" / "metrics.json",
    RUN / "candidate_model_v1" / "candidate_threshold_selection.json",
    ROOT / "output_generation_p1_b_corrected_balanced_5000_v3" / "dataset_meta.json",
    ROOT / "output_generation_p1_b_corrected_balanced_5000_v3" / "training_p1_b_v2" / "metadata.json",
    ROOT / "reports" / "p1_b_training_ablation_summary_v1" / "summary.json",
    ROOT / "reports" / "p1_b_real_dev_v1" / "metrics.json",
    ROOT / "reports" / "p1_b_real_dev_v1" / "failure_atlas.json",
    ROOT / "reports" / "p1_b_real_dev_v1" / "manifest.json",
    ROOT / "output_generation_p1_b1_slot_pilot_80" / "human_review_canonical_20.csv",
    ROOT / "output_generation_p1_b2_rib_web_pilot_48" / "human_review_canonical_12.csv",
    ROOT / "p1_b_general_recognition_pipeline_v4.py",
    ROOT / "core" / "graph_v2_1.py",
    ROOT / "utils" / "p1_b_instance_assembly.py",
    ROOT / "utils" / "p1_b_candidate_proposer.py",
    ROOT / "utils" / "p1_b_candidate_features.py",
    ROOT / "models" / "p1_b_candidate_topology_classifier.py",
    ROOT / "utils" / "p1_b_topology_verifier_v4.py",
    ROOT / "utils" / "p1_b_global_decode_v2.py",
    ROOT / "features" / "p1_b_continuous_slot.py",
    ROOT / "features" / "p1_b_rib_web_v3.py",
    ROOT / "CAE_feature_ontology.yaml",
    ROOT / "face_semantic_subtype_ontology.yaml",
]


def main() -> int:
    if SELECTION.exists() or SIDECAR.exists():
        raise FileExistsError("P1-B preholdout selection already exists and is immutable")
    missing = [str(path) for path in FROZEN_FILES if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen inputs: {missing}")
    dataset = _read_json(ROOT / "output_generation_p1_b_corrected_balanced_5000_v3" / "dataset_meta.json")
    training = _read_json(ROOT / "reports" / "p1_b_training_ablation_summary_v1" / "summary.json")
    real = _read_json(ROOT / "reports" / "p1_b_real_dev_v1" / "metrics.json")
    h_b1 = _sha256(ROOT / "output_generation_p1_b1_slot_pilot_80" / "human_review_canonical_20.csv")
    h_b2 = _sha256(ROOT / "output_generation_p1_b2_rib_web_pilot_48" / "human_review_canonical_12.csv")
    if not dataset.get("automatic_pass"):
        raise RuntimeError("corrected dataset automatic gate is not PASS")
    if h_b1 != dataset["human_gate_hashes"]["H_B1"] or h_b2 != dataset["human_gate_hashes"]["H_B2"]:
        raise RuntimeError("Human Gate hashes no longer match corrected dataset evidence")
    records = [_record(path) for path in FROZEN_FILES]
    payload: dict[str, Any] = {
        "schema_version": "topocae_p1_b_preholdout_selection_1",
        "phase": "P1-B",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_seed": 20260718,
        "selection_basis": "predeclared primary seed; all model and threshold choices use Synthetic-Val only",
        "selected_pipeline": "p1_b_general_recognition_pipeline_v4",
        "selected_gnn_checkpoint": str((RUN / "best.pt").resolve()),
        "selected_candidate_checkpoint": str((RUN / "candidate_model_v1" / "best.pt").resolve()),
        "synthetic_val_thresholds": {
            "same_instance": str((RUN / "same_instance_threshold_selection.json").resolve()),
            "candidate": str((RUN / "candidate_model_v1" / "candidate_threshold_selection.json").resolve()),
            "global_decode": str((RUN / "global_decode_v2_selection.json").resolve()),
        },
        "human_gate_hashes": {"H_B1": h_b1, "H_B2": h_b2},
        "synthetic_val_evidence": training.get("automatic_gate", training.get("automatic_pass", True)),
        "real_dev_evidence": {
            "metrics_sha256": _sha256(ROOT / "reports" / "p1_b_real_dev_v1" / "metrics.json"),
            "area_macro_f1": real["area_macro"]["f1"],
            "used_for_selection": False,
        },
        "frozen_files": records,
        "contracts": {
            "real_data_used_for_training_or_selection": False,
            "holdout_v2_existed_before_selection": False,
            "part_specific_thresholds": False,
            "llm_step_recognition": False,
            "test_part_s3_used": False,
            "holdout_results_must_not_modify_selection": True,
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _write_json(SELECTION, payload)
    digest = _sha256(SELECTION)
    _write_json(SIDECAR, {
        "schema_version": "topocae_p1_b_preholdout_selection_sidecar_1",
        "selection_path": str(SELECTION.resolve()),
        "sha256": digest,
        "verified_after_write": _sha256(SELECTION) == digest,
    })
    print(json.dumps({"selection": str(SELECTION), "sha256": digest, "frozen_file_count": len(records)}, ensure_ascii=False))
    return 0


def _record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


if __name__ == "__main__":
    raise SystemExit(main())
