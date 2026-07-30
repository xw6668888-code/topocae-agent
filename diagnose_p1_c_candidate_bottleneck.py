from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

import torch

from utils.p1_b_topology_verifier_v4 import verify_candidate_prediction
from utils.p1_c_run_paths import p1_c_run_dir


ROOT = Path(__file__).resolve().parent
SEED = 20260718


def main() -> int:
    run_dir = p1_c_run_dir(SEED)
    metadata = _read_json(
        ROOT / "output_generation_p1_c_clean_context_7200" / "training_packages" / "clean_7200" / "metadata.json"
    )
    compact_macros = list(metadata["macro_classes"])
    macro_classes = compact_macros[1:]
    gnn_records = torch.load(run_dir / "predictions.pt", map_location="cpu", weights_only=False)["val"]
    candidate_records = torch.load(
        run_dir / "candidate_model_v1" / "predictions.pt", map_location="cpu", weights_only=False
    )["val"]
    thresholds = _read_json(run_dir / "candidate_model_v1" / "candidate_threshold_selection.json")
    gnn_by_sample = {str(record["sample_id"]): record for record in gnn_records}
    graph_cache: dict[str, dict[str, Any]] = {}
    all_reasons: Counter[str] = Counter()
    by_truth: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "valid_candidate_count": 0,
            "accepted_count": 0,
            "model_top1": Counter(),
            "masked_top1": Counter(),
            "abstention_reasons": Counter(),
            "refined_face_count": Counter(),
        }
    )
    for candidate in candidate_records:
        sample_id = str(candidate["sample_id"])
        graph_path = str(candidate["source_graph_path"])
        graph = graph_cache.setdefault(graph_path, _read_json(Path(graph_path)))
        raw_macros = [
            compact_macros[int(value)] for value in gnn_by_sample[sample_id]["macro_prediction"]
        ]
        verified = verify_candidate_prediction(
            graph,
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
        all_reasons.update(verified.get("abstention_reasons", []))
        if int(candidate["validity_target"]) != 1:
            continue
        macro = str(candidate.get("matched_truth_macro") or "unknown")
        audit = by_truth[macro]
        audit["valid_candidate_count"] += 1
        audit["accepted_count"] += int(verified["status"] == "accepted")
        audit["model_top1"].update([str(verified.get("model_unmasked_top1", "unknown"))])
        audit["masked_top1"].update([str(verified.get("masked_top1", "unknown"))])
        audit["abstention_reasons"].update(verified.get("abstention_reasons", []))
        audit["refined_face_count"].update([len(verified.get("refined_face_indices", []))])
    serialized = {}
    for macro, audit in sorted(by_truth.items()):
        count = int(audit["valid_candidate_count"])
        serialized[macro] = {
            "valid_candidate_count": count,
            "accepted_count": int(audit["accepted_count"]),
            "accepted_rate": int(audit["accepted_count"]) / max(count, 1),
            "model_top1": dict(audit["model_top1"].most_common()),
            "masked_top1": dict(audit["masked_top1"].most_common()),
            "abstention_reasons": dict(audit["abstention_reasons"].most_common()),
            "refined_face_count": {str(k): v for k, v in sorted(audit["refined_face_count"].items())},
        }
    result = {
        "schema_version": "topocae_p1_c_candidate_bottleneck_diagnostics_1",
        "scope": "Synthetic-Val only",
        "seed": SEED,
        "candidate_count": len(candidate_records),
        "all_abstention_reasons": dict(all_reasons.most_common()),
        "valid_candidates_by_truth_macro": serialized,
        "contracts": {
            "real_dev_used": False,
            "holdout_v2_used": False,
            "holdout_v3_used": False,
            "test_part_s3_accessed": False,
        },
    }
    destination = ROOT / "reports" / "p1_c_iterations_v1" / "iteration_2_bottleneck_diagnostics.json"
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "output": str(destination),
        "candidate_count": len(candidate_records),
        "top_reasons": dict(all_reasons.most_common(8)),
        "rib": serialized.get("rib"),
        "web": serialized.get("web"),
    }, ensure_ascii=False))
    return 0


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
