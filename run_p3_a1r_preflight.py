from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.cae_query_ground_p3_a1r.frozen_assets import verify_freeze_manifest
from utils.cae_query_ground_p3_a1r.semantic_coverage import audit_dataset


ROOT = Path(__file__).resolve().parent
OLD_DATASET = ROOT / "output_cae_query_ground_v1"
OLD_FREEZE = ROOT / "reports" / "p3_a1_cae_queryground_v1" / "freeze" / "freeze_manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only P3-A1R preflight audits")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {
        "schema_version": "p3-a1r-preflight/1",
        "old_p3_freeze": verify_freeze_manifest(ROOT, OLD_FREEZE),
        "old_p3_semantic_coverage": audit_dataset(OLD_DATASET),
        "scope_guards": {
            "old_p3_read_only": True,
            "llm_called": False,
            "model_trained": False,
            "solver_executed": False,
            "p4_entered": False,
            "forbidden_data_accessed": False,
        },
    }
    payload = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8", newline="\n")
    print(payload, end="")
    return 0 if result["old_p3_freeze"]["automatic_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
