from __future__ import annotations

"""Resumable strict execution of the nine-run P1-C candidate matrix."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from utils.p1_c_strict_run_paths import ROOT, VARIANT_SPECS


STAGES = {
    "packages": "build_p1_c_candidate_training_package_strict.py",
    "models": "train_p1_c_candidate_classifier_strict.py",
    "select": "select_p1_c_global_decode_strict.py",
    "evaluate": "evaluate_p1_c_synthetic_pipeline_strict.py",
}
REPORT = ROOT / "reports" / "p1_c_iterations_v1" / "strict_candidate_matrix_execution_v1.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=(*STAGES, "all"), required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    requested = list(STAGES) if args.stage == "all" else [args.stage]
    results: list[dict[str, Any]] = []
    for stage in requested:
        script = ROOT / STAGES[stage]
        for variant, spec in VARIANT_SPECS.items():
            for seed in spec["seeds"]:
                command = [sys.executable, str(script), "--variant", variant, "--seed", str(seed)]
                if args.force:
                    command.append("--force")
                started = datetime.now(timezone.utc).isoformat()
                print(json.dumps({"stage": stage, "variant": variant, "seed": seed, "status": "started"}), flush=True)
                completed = subprocess.run(command, cwd=ROOT, check=False)
                record = {
                    "stage": stage,
                    "variant": variant,
                    "seed": seed,
                    "started_at_utc": started,
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                    "returncode": completed.returncode,
                }
                results.append(record)
                REPORT.parent.mkdir(parents=True, exist_ok=True)
                REPORT.write_text(json.dumps({
                    "schema_version": "topocae_p1_c_strict_candidate_matrix_execution_1",
                    "results": results,
                    "contracts": {"real_data_used": False, "holdout_used": False},
                }, indent=2, ensure_ascii=False), encoding="utf-8")
                if completed.returncode != 0:
                    print(json.dumps({**record, "status": "failed"}), flush=True)
                    return completed.returncode
                print(json.dumps({**record, "status": "passed"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
