from __future__ import annotations

import argparse
import subprocess
import sys

from utils.p1_c_strict_run_paths import ROOT


RUNS = [(variant, seed) for variant in ("clean_5000", "clean_7200") for seed in (20260718, 20260719, 20260720)]
SCRIPTS = {"select": "select_p1_c_candidate_decode_iteration2.py", "evaluate": "evaluate_p1_c_synthetic_pipeline_iteration2_v2.py"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=SCRIPTS, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for variant, seed in RUNS:
        command = [sys.executable, str(ROOT / SCRIPTS[args.stage]), "--variant", variant, "--seed", str(seed)]
        if args.force:
            command.append("--force")
        print(f"stage={args.stage} variant={variant} seed={seed}", flush=True)
        code = subprocess.run(command, cwd=ROOT, check=False).returncode
        if code:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
