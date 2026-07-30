from __future__ import annotations

"""Revision 2: allow the single-only ablation to have no multi-context stratum."""

from hashlib import sha256
from pathlib import Path
from typing import Any

import finalize_p1_c_stage6_summary_v1 as implementation


def _file_sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().lower()


def _pipeline_row(root: Path, variant: str, seed: int, iteration: str) -> dict[str, Any]:
    path = root / implementation.run_name(variant, seed) / "synthetic_pipeline_v1" / "metrics.json"
    payload = implementation.read_json(path)
    split = payload["splits"]["val"]
    final = split["topology_verifier_global_decode"]
    class_name, class_record = min(final["per_class"].items(), key=lambda item: float(item[1]["f1"]))
    context_group = final["stratified"]["context_group"]
    single = float(context_group["single_context"]["object_macro_f1"])
    multi_record = context_group.get("multi_context")
    multi = float(multi_record["object_macro_f1"]) if multi_record else None
    return {
        "iteration": iteration,
        "variant": variant,
        "seed": seed,
        "object_macro_f1": float(final["object_macro_f1"]),
        "object_micro_f1": float(final["object_micro_f1"]),
        "exact_instance_ratio": float(final["exact_instance_ratio"]),
        "instance_iou": float(final["mean_best_instance_iou"]),
        "boundary_f1": float(final["boundary"]["f1"]),
        "proposal_oracle_recall_iou_0_5": float(split["candidate_proposals"]["oracle_recall_iou_0_5"]),
        "single_context_object_macro_f1": single,
        "multi_context_object_macro_f1": multi,
        "single_multi_gap_abs": abs(single - multi) if multi is not None else None,
        "worst_class": class_name,
        "worst_class_object_f1": float(class_record["f1"]),
        "automatic_pass": bool(payload.get("contracts", {}).get("automatic_pass", False)),
        "metrics_path": str(path),
        "metrics_sha256": _file_sha(path),
    }


implementation.pipeline_row = _pipeline_row


if __name__ == "__main__":
    raise SystemExit(implementation.main())
