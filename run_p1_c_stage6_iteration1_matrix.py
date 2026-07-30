from __future__ import annotations

"""Resumable six-run execution for the frozen Stage 6 iteration-1 plan."""

import argparse
from datetime import datetime, timezone
import json
import subprocess
import sys

from utils.p1_c_strict_run_paths import ROOT


RUNS = [(variant, seed) for variant in ("clean_5000", "clean_7200") for seed in (20260718, 20260719, 20260720)]
STAGES = {
    "packages": "build_p1_c_candidate_training_package_iteration1.py",
    "models": "train_p1_c_candidate_classifier_iteration1.py",
    "select": "select_p1_c_global_decode_iteration1.py",
    "evaluate": "evaluate_p1_c_synthetic_pipeline_iteration1.py"
}
REPORT = ROOT / "reports" / "p1_c_iterations_v1" / "stage6_iteration1_matrix_execution.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=(*STAGES, "all"), required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    stages = list(STAGES) if args.stage == "all" else [args.stage]
    results = []
    for stage in stages:
        for variant, seed in RUNS:
            command = [sys.executable, str(ROOT / STAGES[stage]), "--variant", variant, "--seed", str(seed)]
            if args.force:
                command.append("--force")
            started = datetime.now(timezone.utc).isoformat()
            print(json.dumps({"stage": stage, "variant": variant, "seed": seed, "status": "started"}), flush=True)
            code = subprocess.run(command, cwd=ROOT, check=False).returncode
            results.append({"stage": stage, "variant": variant, "seed": seed, "started_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat(), "returncode": code})
            REPORT.write_text(json.dumps({
                "schema_version": "topocae_p1_c_stage6_iteration1_matrix_execution_1",
                "results": results,
                "contracts": {"real_data_used": False, "holdout_used": False}
            }, indent=2, ensure_ascii=False), encoding="utf-8")
            if code:
                return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
