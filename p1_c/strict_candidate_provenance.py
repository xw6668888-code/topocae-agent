from __future__ import annotations

"""Shared fail-closed provenance checks for strict P1-C candidate stages."""

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from utils.p1_c_strict_run_paths import ROOT, VARIANT_SPECS, strict_run_dir, training_metadata_path, validate_variant_seed


PLAN_PATH = ROOT / "configs" / "p1_c" / "candidate_training_plan_strict_v2.json"


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


def candidate_plan() -> tuple[dict[str, Any], str]:
    plan = read_json(PLAN_PATH)
    if not bool(plan.get("frozen_before_candidate_models")):
        raise RuntimeError("strict candidate plan is not frozen")
    return plan, file_sha256(PLAN_PATH)


def preflight_gnn_source(variant: str, seed: int) -> dict[str, Any]:
    validate_variant_seed(variant, seed)
    plan, plan_hash = candidate_plan()
    run_name = f"{variant}_seed_{seed}"
    source = plan.get("source_gnn_runs", {}).get(run_name)
    if not source:
        raise RuntimeError(f"strict candidate plan omits GNN source {run_name}")
    run_dir = strict_run_dir(variant, seed)
    predictions = run_dir / "predictions.pt"
    config = run_dir / "config.json"
    observed = {
        "predictions_sha256": file_sha256(predictions),
        "config_sha256": file_sha256(config),
    }
    for key, value in observed.items():
        if value != str(source[key]).lower():
            raise RuntimeError(f"strict GNN source drift for {run_name}: {key}")
    package_name = str(VARIANT_SPECS[variant]["package"])
    expected_package_hash = str(plan["training_packages"][package_name]["manifest_sha256"]).lower()
    package_manifest = training_metadata_path(variant).parent / "manifest.json"
    if file_sha256(package_manifest) != expected_package_hash:
        raise RuntimeError(f"strict training package drift for {package_name}")
    return {
        "plan_sha256": plan_hash,
        "run_name": run_name,
        **observed,
        "training_package": package_name,
        "training_package_manifest_sha256": expected_package_hash,
    }


def preflight_candidate_package(variant: str, seed: int) -> dict[str, Any]:
    source = preflight_gnn_source(variant, seed)
    run_dir = strict_run_dir(variant, seed)
    receipt_path = run_dir / "strict_candidate_package_receipt.json"
    receipt = read_json(receipt_path)
    if receipt.get("candidate_plan_sha256") != source["plan_sha256"]:
        raise RuntimeError("candidate-package receipt references a different frozen plan")
    manifest = run_dir / "candidate_training_v1" / "manifest.json"
    observed = file_sha256(manifest)
    if receipt.get("candidate_package_manifest_sha256") != observed:
        raise RuntimeError("candidate-package manifest drift")
    metadata = read_json(run_dir / "candidate_training_v1" / "metadata.json")
    if float(metadata["splits"]["val"]["proposal_oracle_recall_iou_0_5"]) < 0.95:
        raise RuntimeError("candidate proposal oracle recall gate failed")
    if not bool(metadata["splits"]["val"]["all_seven_sources_present"]):
        raise RuntimeError("candidate seven-source gate failed")
    return {**source, "candidate_package_manifest_sha256": observed}

