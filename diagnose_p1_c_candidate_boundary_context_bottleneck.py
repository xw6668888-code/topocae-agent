from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from utils.p1_c_candidate_iteration_paths import MODEL_NAME
from utils.p1_c_run_paths import p1_c_run_dir
from utils.p1_c_topology_verifier_v5 import verify_candidate_prediction


ROOT = Path(__file__).resolve().parent
SEED = 20260718


def main() -> int:
    run = p1_c_run_dir(SEED)
    metadata = _read_json(
        ROOT / "output_generation_p1_c_clean_context_7200" / "training_packages" / "clean_7200" / "metadata.json"
    )
    compact_macros = list(metadata["macro_classes"])
    macro_classes = compact_macros[1:]
    gnn = torch.load(run / "predictions.pt", map_location="cpu", weights_only=False)["val"]
    candidates = torch.load(run / MODEL_NAME / "predictions.pt", map_location="cpu", weights_only=False)["val"]
    thresholds = _read_json(run / MODEL_NAME / "candidate_threshold_selection.json")
    gnn_by_sample = {str(record["sample_id"]): record for record in gnn}
    graph_cache: dict[str, dict[str, Any]] = {}
    reasons: Counter[str] = Counter()
    by_truth: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "accepted": 0, "top1": Counter(), "reasons": Counter(), "confidence": []}
    )
    for candidate in candidates:
        sample_id = str(candidate["sample_id"])
        graph_path = str(candidate["source_graph_path"])
        if graph_path not in graph_cache:
            graph_cache[graph_path] = _read_json(Path(graph_path))
        raw_macros = [
            compact_macros[int(value)] for value in gnn_by_sample[sample_id]["macro_prediction"]
        ]
        verified = verify_candidate_prediction(
            graph_cache[graph_path],
            candidate["proposal"],
            {
                "macro_classes": macro_classes,
                "macro_logits": candidate["macro_logits"],
                "macro_probabilities": candidate["macro_probabilities"],
                "validity_score": candidate["validity_score"],
                "membership_probabilities": candidate["membership_probabilities"],
                "boundary_probabilities": candidate["boundary_probabilities"],
                "face_macro_predictions": raw_macros,
            },
            thresholds=thresholds,
        )
        reasons.update(verified.get("abstention_reasons", []))
        if int(candidate["validity_target"]) != 1:
            continue
        macro = str(candidate.get("matched_truth_macro") or "unknown")
        audit = by_truth[macro]
        audit["count"] += 1
        audit["accepted"] += int(verified["status"] == "accepted")
        audit["top1"].update([str(verified.get("model_unmasked_top1", "unknown"))])
        audit["reasons"].update(verified.get("abstention_reasons", []))
        audit["confidence"].append(float(verified.get("final_confidence", 0.0) or 0.0))
    serialized = {
        macro: {
            "valid_candidate_count": int(audit["count"]),
            "accepted_count": int(audit["accepted"]),
            "accepted_rate": int(audit["accepted"]) / max(int(audit["count"]), 1),
            "model_top1": dict(audit["top1"].most_common()),
            "abstention_reasons": dict(audit["reasons"].most_common()),
            "mean_final_confidence": mean(audit["confidence"]) if audit["confidence"] else 0.0,
        }
        for macro, audit in sorted(by_truth.items())
    }
    result = {
        "schema_version": "topocae_p1_c_candidate_boundary_context_bottleneck_1",
        "scope": "Synthetic-Val only",
        "seed": SEED,
        "candidate_count": len(candidates),
        "all_abstention_reasons": dict(reasons.most_common()),
        "valid_candidates_by_truth_macro": serialized,
        "contracts": {
            "real_dev_used": False,
            "holdout_v2_used": False,
            "holdout_v3_used": False,
            "test_part_s3_accessed": False,
        },
    }
    destination = ROOT / "reports" / "p1_c_iterations_v1" / "iteration_3_candidate_bottleneck_diagnostics.json"
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "output": str(destination),
        "top_reasons": dict(reasons.most_common(8)),
        "rib": serialized.get("rib"),
        "web": serialized.get("web"),
        "pocket_circular": serialized.get("pocket_circular"),
        "cylinder_boss": serialized.get("cylinder_boss"),
    }, ensure_ascii=False))
    return 0


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
