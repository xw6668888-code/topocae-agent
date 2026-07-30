from __future__ import annotations

"""Fail-closed provenance for Stage 6 iteration 1."""

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from p1_c.strict_candidate_provenance_v2 import preflight_gnn_source
from utils.p1_c_iteration1_run_paths import iteration_run_dir
from utils.p1_c_strict_run_paths import DATASET_ROOT, ROOT


PLAN = ROOT / "configs" / "p1_c" / "stage6_iteration1_oriented_cylinder_polarity_v1.json"
PLAN_SHA256 = "b60a57c55294ecf012ced1bc24edca6cffa73d728594f6e5b77ca3916eeae036"
GRAPH_MANIFEST = DATASET_ROOT / "graph_polarity_v1_manifest.json"
GRAPH_MANIFEST_SHA256 = "e9fb9a09407623af76bf6c4395edf4832be27fcecaa3f3982da0e6f978d609b4"


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().lower()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def preflight_iteration_source(variant: str, seed: int) -> dict[str, Any]:
    if variant not in {"clean_5000", "clean_7200"}:
        raise RuntimeError(f"iteration-1 variant not approved: {variant}")
    if file_sha256(PLAN) != PLAN_SHA256:
        raise RuntimeError("iteration-1 plan drift")
    if file_sha256(GRAPH_MANIFEST) != GRAPH_MANIFEST_SHA256:
        raise RuntimeError("iteration-1 Graph polarity manifest drift")
    manifest = read_json(GRAPH_MANIFEST)
    if int(manifest.get("graph_count", 0)) != 7200:
        raise RuntimeError("iteration-1 Graph polarity inventory is incomplete")
    if not bool(manifest.get("contracts", {}).get("feature_uses_labels") is False):
        raise RuntimeError("iteration-1 polarity feature leakage contract failed")
    source = preflight_gnn_source(variant, seed)
    return {
        **source,
        "iteration_plan_sha256": PLAN_SHA256,
        "graph_polarity_manifest_sha256": GRAPH_MANIFEST_SHA256,
    }


def preflight_iteration_package(variant: str, seed: int) -> dict[str, Any]:
    source = preflight_iteration_source(variant, seed)
    run = iteration_run_dir(variant, seed)
    receipt = read_json(run / "iteration1_candidate_package_receipt.json")
    manifest = run / "candidate_training_v1" / "manifest.json"
    observed = file_sha256(manifest)
    if receipt.get("iteration_plan_sha256") != PLAN_SHA256:
        raise RuntimeError("iteration-1 package plan mismatch")
    if receipt.get("candidate_package_manifest_sha256") != observed:
        raise RuntimeError("iteration-1 candidate package drift")
    metadata = read_json(run / "candidate_training_v1" / "metadata.json")
    if int(metadata["dimensions"]["engineered_feature_dim"]) != 71:
        raise RuntimeError("iteration-1 candidate feature dimension is not 71")
    return {**source, "candidate_package_manifest_sha256": observed}


def preflight_iteration_model(variant: str, seed: int) -> dict[str, Any]:
    source = preflight_iteration_package(variant, seed)
    run = iteration_run_dir(variant, seed)
    receipt = read_json(run / "iteration1_candidate_model_receipt.json")
    model = run / "candidate_model_v1"
    expected = {
        "candidate_model_config_sha256": file_sha256(model / "config.json"),
        "candidate_model_checkpoint_sha256": file_sha256(model / "best.pt"),
        "candidate_model_metrics_sha256": file_sha256(model / "metrics.json"),
        "candidate_model_predictions_sha256": file_sha256(model / "predictions.pt"),
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise RuntimeError(f"iteration-1 candidate model drift: {key}")
    return {**source, **expected}

