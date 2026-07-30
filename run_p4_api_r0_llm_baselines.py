"""R0.1 immutable-run P4 Structured/Direct runner for train/val only."""

from __future__ import annotations

import argparse
from dataclasses import asdict
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
from utils.cae_language_grounding.run_identity import (
    build_run_manifest,
    prepare_immutable_run_directory,
)


ROOT = Path(__file__).resolve().parent
RUN_ROOT = ROOT / "output_p4_a1_language_grounding_v1" / "baselines_api_r0" / "runs"
BASELINE_CONTRACTS = {
    "structured": (
        ROOT / "prompts" / "p4_a1_structured_llm_v1.md",
        ROOT / "schemas" / "cae_intent_result.schema.json",
    ),
    "direct": (
        ROOT / "prompts" / "p4_a1_direct_llm_v1.md",
        ROOT / "schemas" / "p4_a1_direct_selection.schema.json",
    ),
}


def iter_window(split: str, *, offset: int, limit: int | None,
                shard_index: int, shard_count: int) -> Iterable[object]:
    assert_split_authorized(split)
    emitted = skipped = 0
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


def _runtime_identity(config: LLMRuntimeConfig) -> dict:
    profile = config.profile
    return {
        "backend": config.backend,
        "provider_profile": config.provider_profile,
        "model": config.model,
        "revision": config.revision,
        "temperature": config.temperature,
        "seed": config.seed,
        "max_tokens": config.max_tokens,
        "constrained_decoding": config.constrained_decoding,
        "timeout_seconds": config.timeout_seconds,
        "max_http_retries": config.max_http_retries,
        "retry_backoff_seconds": config.retry_backoff_seconds,
        "payload_profile": asdict(profile),
    }


def _prepare_run(config: LLMRuntimeConfig, *, baseline: str, split: str,
                 offset: int, limit: int | None, shard_index: int,
                 shard_count: int):
    prompt_path, schema_path = BASELINE_CONTRACTS[baseline]
    prompt = prompt_path.read_text(encoding="utf-8")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    manifest = build_run_manifest(
        endpoint=config.base_url,
        backend=config.backend,
        provider_profile=config.provider_profile,
        model=config.model,
        revision=config.revision,
        prompt=prompt,
        schema=schema,
        runtime_config=_runtime_identity(config),
        baseline=baseline,
        split=split,
        offset=offset,
        limit=limit,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    return prepare_immutable_run_directory(RUN_ROOT, manifest), manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run cached P4 API baselines in immutable train/val runs."
    )
    parser.add_argument(
        "--baseline", choices=("structured", "direct", "both"), default="both"
    )
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

    requested = (
        ("structured", "direct") if args.baseline == "both"
        else (args.baseline,)
    )
    client = OpenAICompatibleJSONClient(config)
    original_iterator = baseline_runner.iter_language_examples
    completed = []
    try:
        baseline_runner.iter_language_examples = lambda split: iter_window(
            split,
            offset=args.offset,
            limit=args.limit,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        )
        for baseline in requested:
            run_dir, manifest = _prepare_run(
                config,
                baseline=baseline,
                split=args.split,
                offset=args.offset,
                limit=args.limit,
                shard_index=args.shard_index,
                shard_count=args.shard_count,
            )
            baseline_runner.OUTPUT_ROOT = run_dir
            if baseline == "structured":
                result = baseline_runner.run_structured(args.split, client)
            else:
                result = baseline_runner.run_direct(args.split, client)
            completed.append({
                "baseline": baseline,
                "run_id": manifest["run_id"],
                "run_directory": str(run_dir.resolve()),
                "result": result,
            })
    finally:
        baseline_runner.iter_language_examples = original_iterator
    print(json.dumps({
        "status": "COMPLETE",
        "split": args.split,
        "runs": completed,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
