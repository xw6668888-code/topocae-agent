"""Run and evaluate the bilingual P4-A1 keyword baseline on train/val."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from utils.cae_language_grounding.evaluation import (
    BaselineAccumulator,
    MemoryCache,
)
from utils.cae_language_grounding.grounder import ground_intent_result
from utils.cae_language_grounding.keyword_parser import parse_keyword_query
from utils.cae_language_grounding.records import iter_language_examples


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = (
    ROOT
    / "output_p4_a1_language_grounding_v1"
    / "baselines"
    / "keyword"
)


def run_split(split: str) -> dict:
    output_dir = OUTPUT_ROOT / split
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    metrics_path = output_dir / "metrics.json"
    failure_path = output_dir / "failure_atlas.json"
    cache = MemoryCache()
    accumulator = BaselineAccumulator(baseline="keyword", split=split)
    with predictions_path.open("w", encoding="utf-8", newline="\n") as handle:
        for example in iter_language_examples(split):
            inference_input = example.inference_input()
            started = time.perf_counter()
            prediction = parse_keyword_query(**inference_input)
            memory, instance_map = cache.get(str(example.source_ref["graph_path"]))
            grounding = ground_intent_result(
                prediction,
                memory,
                policy=cache.policy,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            outcome = accumulator.add(
                example=example,
                prediction=prediction,
                grounding=grounding,
                instance_map=instance_map,
                latency_ms=elapsed_ms,
            )
            handle.write(
                json.dumps(
                    {
                        "schema_version": "topocae-p4-a1-keyword-prediction/1",
                        "baseline": "keyword",
                        "split": split,
                        "example_id": example.example_id,
                        "canonical_record_id": example.canonical_record_id,
                        "language": example.language,
                        "difficulty_level": example.difficulty_level,
                        "prediction": prediction,
                        "grounding": grounding,
                        "latency_ms": elapsed_ms,
                        "evaluation_flags": outcome,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
    metrics = accumulator.finalize(unique_source_graphs=len(cache))
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    failure_path.write_text(
        json.dumps(
            {
                "schema_version": "topocae-p4-a1-failure-atlas/1",
                "baseline": "keyword",
                "split": split,
                "categories": dict(sorted(accumulator.failure_atlas.items())),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {
        "split": split,
        "predictions": str(predictions_path.resolve()),
        "metrics": str(metrics_path.resolve()),
        "failure_atlas": str(failure_path.resolve()),
        "summary": {
            "record_count": metrics["record_count"],
            "schema_valid_rate": metrics["language_cae_ir"][
                "schema_valid_rate_after_retry"
            ],
            "status_accuracy": metrics["language_cae_ir"]["status_accuracy"],
            "cae_ir_exact": metrics["language_cae_ir"][
                "cae_ir_canonical_exact_match"
            ],
            "slot_macro_f1": metrics["language_cae_ir"]["slot_macro_f1"],
            "exact_face_set_accuracy": metrics["grounding"][
                "exact_face_set_accuracy"
            ],
            "executable_success": metrics["grounding"]["executable_success"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run P4-A1 keyword baseline on train/val"
    )
    parser.add_argument(
        "--split",
        action="append",
        choices=("train", "val"),
        dest="splits",
    )
    args = parser.parse_args()
    splits = args.splits or ["train", "val"]
    results = [run_split(split) for split in splits]
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
