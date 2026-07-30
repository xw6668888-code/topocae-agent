"""Build and freeze the deterministic P3-A1 formal dataset without LLM or solver use."""

from __future__ import annotations

import argparse
import json

from utils.cae_query_ground.formal_pipeline import run_formal


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_formal(force=args.force), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

