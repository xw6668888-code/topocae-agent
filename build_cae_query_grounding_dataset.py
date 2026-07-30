"""Build or finalize the deterministic, no-LLM CAE-QueryGround v1 dataset."""

from __future__ import annotations

import argparse
import json

from utils.cae_query_ground.formal_finalize_release import finalize
from utils.cae_query_ground.formal_pipeline_release_v3 import run_formal


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help="Freeze existing fully audited formal outputs without regenerating data.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = finalize(force=True) if args.finalize_existing else run_formal(force=args.force)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

