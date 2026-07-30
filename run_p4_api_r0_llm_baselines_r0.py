"""Reliable P4 Structured/Direct runner for train/val only.

R0 deliberately cannot unlock or inspect the formal test split.  The existing
baseline functions are reused, while this entry point adds deterministic
windowing, isolated output directories, provider profiles, and response-cache
resume behavior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import run_p4_a1_llm_baselines as baseline_runner
from utils.cae_language_grounding.llm_adapter import (
    LLMRuntimeConfig,
    OpenAICompatibleJSONClient,
)
from utils.cae_language_grounding.records import iter_language_examples
from utils.cae_language_grounding.reliable_runner import assert_split_authorized


ROOT = Path(__file__).resolve().parent


def iter_window(split: str, *, offset: int, limit: int | None,
                shard_index: int, shard_count: int) -> Iterable[object]:
    """Select a stable shard first, then apply offset/limit within that shard."""

    assert_split_authorized(split)
    emitted = 0
    skipped = 0
    for index, example in enumerate(iter_language_examples(split)):
        if index % shard_count != shard_index:
            continue
        if skipped < offset:
            skipped += 1
            continue
        if limit is not None and emitted >= limit:
            break
        emitted += 1
        yield example


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run cached P4 API baselines on an authorized train/val window."
    )
    parser.add_argument("--baseline", choices=("structured", "direct", "both"),
                        default="both")
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if args.offset < 0 or (args.limit is not None and args.limit < 0):
        parser.error("offset and limit must be non-negative")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("shard-index must be in [0, shard-count)")
    assert_split_authorized(args.split)

    config = LLMRuntimeConfig.from_env()
    if not config.available:
        print(json.dumps({
            "status": "BLOCKED_AT_H4_RUNTIME",
            "missing_required": config.missing_required,
            "runtime": config.public_metadata(),
        }, ensure_ascii=False, indent=2))
        return 2

    window_name = (
        f"{config.provider_profile}/shard-{args.shard_index}-of-{args.shard_count}"
        f"/offset-{args.offset}-limit-{args.limit if args.limit is not None else 'all'}"
    )
    baseline_runner.OUTPUT_ROOT = (
        ROOT / "output_p4_a1_language_grounding_v1" / "baselines_api_r0" / window_name
    )
    original_iterator = baseline_runner.iter_language_examples
    baseline_runner.iter_language_examples = lambda split: iter_window(
        split,
        offset=args.offset,
        limit=args.limit,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    client = OpenAICompatibleJSONClient(config)
    results = []
    try:
        if args.baseline in {"structured", "both"}:
            results.append(baseline_runner.run_structured(args.split, client))
        if args.baseline in {"direct", "both"}:
            results.append(baseline_runner.run_direct(args.split, client))
    finally:
        baseline_runner.iter_language_examples = original_iterator
    print(json.dumps({
        "status": "COMPLETE",
        "split": args.split,
        "provider_profile": config.provider_profile,
        "output_root": str(baseline_runner.OUTPUT_ROOT.resolve()),
        "results": results,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
