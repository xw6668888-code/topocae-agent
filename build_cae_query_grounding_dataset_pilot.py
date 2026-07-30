"""Deterministic no-LLM P3-A1 audit and H3-01 pilot entrypoint.

No formal-scale, freeze, model-training, LLM, solver, Holdout, real-part, or P4
mode exists in this executable.
"""

from __future__ import annotations

import argparse
import json

from utils.cae_query_ground.orchestrator import execute


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("audit", "pilot"), default="pilot")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(execute(stage=args.stage, force=args.force), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

