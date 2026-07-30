from __future__ import annotations

"""Corrected full rejection record for Stage 6 iteration 1."""

import json

from utils.p1_c_iteration1_run_paths import iteration_run_dir
from utils.p1_c_strict_run_paths import ROOT


OUTPUT = ROOT / "reports" / "p1_c_iterations_v1" / "stage6_iteration1_result_v2.json"


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    rows = []
    for variant in ("clean_5000", "clean_7200"):
        for seed in (20260718, 20260719, 20260720):
            run = iteration_run_dir(variant, seed)
            model = _read(run / "candidate_model_v1" / "metrics.json")
            thresholds = _read(run / "candidate_model_v1" / "candidate_threshold_selection.json")
            metrics = _read(run / "synthetic_pipeline_v1" / "metrics.json")["splits"]["val"]
            pipeline = metrics["topology_verifier_global_decode"]
            worst_f1, worst_class = min(
                (float(record["f1"]), name)
                for name, record in pipeline["per_class"].items()
                if int(record["truth"]) > 0
            )
            absolute_pass = bool(pipeline["object_macro_f1"] >= 0.90 and pipeline["exact_instance_ratio"] >= 0.88 and worst_f1 >= 0.80)
            rows.append({
                "variant": variant,
                "seed": seed,
                "candidate_macro_f1": model["evaluation"]["val"]["macro"]["macro_f1"],
                "min_candidate_confidence": thresholds["min_candidate_confidence"],
                "object_macro_f1": pipeline["object_macro_f1"],
                "exact_instance_ratio": pipeline["exact_instance_ratio"],
                "worst_class": worst_class,
                "worst_class_f1": worst_f1,
                "abstention_rate": metrics["topology_diagnostics"]["abstention_rate"],
                "absolute_object_gates_pass": absolute_pass
            })
    failed = [f"{row['variant']}_seed_{row['seed']}" for row in rows if not row["absolute_object_gates_pass"]]
    result = {
        "schema_version": "topocae_p1_c_stage6_iteration1_result_2",
        "iteration_index": 1,
        "hypothesis_candidate_representation_supported": all(row["candidate_macro_f1"] >= 0.998 for row in rows),
        "formal_iteration_kept": False,
        "failed_run_count": len(failed),
        "failed_runs": failed,
        "rejection_reason": "five of six seeds fail at least one object/exact/worst-class gate; two suffer global abstention collapse and three additional runs retain cylinder_boss worst-class F1 below 0.80",
        "rows": rows,
        "contracts": {"evidence_scope": "Synthetic-Val only", "real_data_used": False, "holdout_used": False}
    }
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "failed_runs": failed}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
