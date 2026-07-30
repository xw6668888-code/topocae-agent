"""Run P4-A1 Structured-LLM and Direct-LLM baselines on train/val only."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time
from typing import Any, Mapping

from utils.cae_language_grounding.direct_llm import (
    expand_and_verify_direct_selection,
    infer_direct_selection,
)
from utils.cae_language_grounding.evaluation import (
    BaselineAccumulator,
    MemoryCache,
)
from utils.cae_language_grounding.grounder import ground_intent_result
from utils.cae_language_grounding.llm_adapter import (
    LLMRuntimeConfig,
    OpenAICompatibleJSONClient,
)
from utils.cae_language_grounding.records import iter_language_examples


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "output_p4_a1_language_grounding_v1" / "baselines"


def _invalid_intent_result() -> dict[str, Any]:
    return {
        "schema_version": "cae-intent-result/1.0.0",
        "status": "INVALID",
        "error_code": "E_SCHEMA_VALIDATION",
        "field_path": "/",
        "missing_information": [],
        "ambiguities": [],
        "clarification_questions": [],
    }


def _invalid_direct_selection() -> dict[str, Any]:
    return {
        "schema_version": "topocae-direct-selection/1.0.0",
        "status": "INVALID",
        "error_code": "E_DIRECT_RUNTIME_OUTPUT",
        "message": "Backend output failed strict parsing/schema validation.",
        "selections": [],
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def run_structured(
    split: str,
    client: OpenAICompatibleJSONClient,
) -> dict[str, Any]:
    output_dir = OUTPUT_ROOT / "structured_llm" / split
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = MemoryCache()
    accumulator = BaselineAccumulator(
        baseline="structured_llm", split=split
    )
    predictions_path = output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8", newline="\n") as handle:
        for example in iter_language_examples(split):
            started = time.perf_counter()
            outcome = client.infer_structured(example.inference_input())
            prediction = dict(outcome.value or _invalid_intent_result())
            memory, instance_map = cache.get(
                str(example.source_ref["graph_path"])
            )
            grounding = ground_intent_result(
                prediction, memory, policy=cache.policy
            )
            latency_ms = (time.perf_counter() - started) * 1000.0
            flags = accumulator.add(
                example=example,
                prediction=prediction,
                grounding=grounding,
                instance_map=instance_map,
                latency_ms=latency_ms,
                schema_valid_first_round=outcome.schema_valid_first_round,
                retry_count=outcome.retry_count,
                input_tokens=outcome.input_tokens,
                output_tokens=outcome.output_tokens,
            )
            handle.write(
                json.dumps(
                    {
                        "schema_version":
                            "topocae-p4-a1-structured-llm-prediction/1",
                        "baseline": "structured_llm",
                        "split": split,
                        "example_id": example.example_id,
                        "canonical_record_id": example.canonical_record_id,
                        "prediction": prediction,
                        "grounding": grounding,
                        "runtime_outcome": {
                            "schema_valid_first_round":
                                outcome.schema_valid_first_round,
                            "schema_valid_after_retry":
                                outcome.schema_valid_after_retry,
                            "retry_count": outcome.retry_count,
                            "failure_code": outcome.failure_code,
                            "input_tokens": outcome.input_tokens,
                            "output_tokens": outcome.output_tokens,
                        },
                        "evaluation_flags": flags,
                        "latency_ms": latency_ms,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
    metrics = accumulator.finalize(unique_source_graphs=len(cache))
    metrics["runtime_config"] = client.config.public_metadata()
    _write_json(output_dir / "metrics.json", metrics)
    _write_json(
        output_dir / "failure_atlas.json",
        {
            "schema_version": "topocae-p4-a1-failure-atlas/1",
            "baseline": "structured_llm",
            "split": split,
            "categories": dict(sorted(accumulator.failure_atlas.items())),
        },
    )
    return {
        "baseline": "structured_llm",
        "split": split,
        "record_count": metrics["record_count"],
        "metrics": str((output_dir / "metrics.json").resolve()),
    }


def _direct_evaluation_flags(
    *,
    example,
    grounding: Mapping[str, Any],
    instance_map: Mapping[str, int],
) -> dict[str, bool]:
    expected_status = str(example.oracle_intent_result["status"])
    status_correct = grounding.get("status") == expected_status
    if expected_status != "VALID":
        return {
            "status_correct": status_correct,
            "object_exact": status_correct,
            "face_exact": status_correct,
            "executable_success": status_correct,
        }
    expected = {
        str(item["operation_id"]): item
        for item in example.ground_truth["targets"]
    }
    actual = {
        str(item["selection_id"]): item
        for item in grounding.get("operations", [])
    }
    object_exact = set(expected) == set(actual)
    face_exact = object_exact
    executable = status_correct and object_exact
    for operation_id, target in expected.items():
        item = actual.get(operation_id)
        if item is None:
            continue
        if target["target_contract"] == "whole_part":
            ok = (
                item.get("target_contract") == "whole_part"
                and not item.get("selected_object_ids")
                and not item.get("selected_face_indices")
            )
            object_exact = object_exact and ok
            face_exact = face_exact and ok
            executable = executable and ok
            continue
        actual_instances = {
            instance_map[value]
            for value in item.get("selected_object_ids", [])
            if value in instance_map
        }
        expected_instances = {
            int(value) for value in target["target_instance_ids"]
        }
        target_object_exact = actual_instances == expected_instances
        target_face_exact = set(item.get("selected_face_indices", [])) == {
            int(value) for value in target["target_face_indices"]
        }
        object_exact = object_exact and target_object_exact
        face_exact = face_exact and target_face_exact
        executable = executable and target_object_exact and target_face_exact
    return {
        "status_correct": status_correct,
        "object_exact": object_exact,
        "face_exact": face_exact,
        "executable_success": executable,
    }


def run_direct(
    split: str,
    client: OpenAICompatibleJSONClient,
) -> dict[str, Any]:
    output_dir = OUTPUT_ROOT / "direct_llm" / split
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = MemoryCache()
    counts = Counter()
    latencies = []
    atlas: dict[str, list[dict[str, Any]]] = defaultdict(list)
    predictions_path = output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8", newline="\n") as handle:
        for example in iter_language_examples(split):
            memory, instance_map = cache.get(
                str(example.source_ref["graph_path"])
            )
            started = time.perf_counter()
            outcome = infer_direct_selection(
                client,
                text=example.text,
                language=example.language,
                memory=memory,
            )
            selection = dict(outcome.value or _invalid_direct_selection())
            grounding = expand_and_verify_direct_selection(selection, memory)
            latency_ms = (time.perf_counter() - started) * 1000.0
            flags = _direct_evaluation_flags(
                example=example,
                grounding=grounding,
                instance_map=instance_map,
            )
            counts["records"] += 1
            counts["schema_valid_first_round"] += int(
                outcome.schema_valid_first_round
            )
            counts["schema_valid_after_retry"] += int(
                outcome.schema_valid_after_retry
            )
            counts["retry_count"] += outcome.retry_count
            counts["input_tokens"] += outcome.input_tokens
            counts["output_tokens"] += outcome.output_tokens
            for key, value in flags.items():
                counts[key] += int(value)
            verifier = grounding.get("verifier", {})
            counts["illegal_object_ids"] += int(
                verifier.get("illegal_object_id_count", 0)
            )
            counts["illegal_face_ids"] += int(
                verifier.get("illegal_face_id_count", 0)
            )
            counts["silent_fallbacks"] += int(
                verifier.get("silent_fallback_count", 0)
            )
            latencies.append(latency_ms)
            if not flags["executable_success"] and split == "val":
                category = (
                    "direct_llm_invalid_id_error"
                    if verifier.get("illegal_object_id_count", 0)
                    else "direct_llm_selection_error"
                )
                if len(atlas[category]) < 10:
                    atlas[category].append(
                        {
                            "example_id": example.example_id,
                            "query": example.text,
                            "oracle_status":
                                example.oracle_intent_result["status"],
                            "grounding_status": grounding.get("status"),
                            "error_code": grounding.get("error_code"),
                        }
                    )
            handle.write(
                json.dumps(
                    {
                        "schema_version":
                            "topocae-p4-a1-direct-llm-prediction/1",
                        "baseline": "direct_llm",
                        "split": split,
                        "example_id": example.example_id,
                        "canonical_record_id": example.canonical_record_id,
                        "selection": selection,
                        "grounding": grounding,
                        "runtime_outcome": {
                            "schema_valid_first_round":
                                outcome.schema_valid_first_round,
                            "schema_valid_after_retry":
                                outcome.schema_valid_after_retry,
                            "retry_count": outcome.retry_count,
                            "failure_code": outcome.failure_code,
                            "input_tokens": outcome.input_tokens,
                            "output_tokens": outcome.output_tokens,
                        },
                        "evaluation_flags": flags,
                        "latency_ms": latency_ms,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
    total = max(counts["records"], 1)
    metrics = {
        "schema_version": "topocae-p4-a1-direct-metrics/1",
        "baseline": "direct_llm",
        "split": split,
        "record_count": counts["records"],
        "unique_source_graphs": len(cache),
        "schema_valid_rate_first_round":
            counts["schema_valid_first_round"] / total,
        "schema_valid_rate_after_retry":
            counts["schema_valid_after_retry"] / total,
        "status_accuracy": counts["status_correct"] / total,
        "object_selection_accuracy": counts["object_exact"] / total,
        "exact_face_set_accuracy": counts["face_exact"] / total,
        "executable_success": counts["executable_success"] / total,
        "illegal_object_id_count": counts["illegal_object_ids"],
        "illegal_face_id_count": counts["illegal_face_ids"],
        "silent_fallback_count": counts["silent_fallbacks"],
        "runtime": {
            "latency_ms_mean": sum(latencies) / len(latencies)
            if latencies
            else 0.0,
            "input_tokens": counts["input_tokens"],
            "output_tokens": counts["output_tokens"],
            "retry_count": counts["retry_count"],
            "retry_rate": counts["retry_count"] / total,
        },
        "runtime_config": client.config.public_metadata(),
    }
    _write_json(output_dir / "metrics.json", metrics)
    _write_json(
        output_dir / "failure_atlas.json",
        {
            "schema_version": "topocae-p4-a1-failure-atlas/1",
            "baseline": "direct_llm",
            "split": split,
            "categories": dict(sorted(atlas.items())),
        },
    )
    return {
        "baseline": "direct_llm",
        "split": split,
        "record_count": counts["records"],
        "metrics": str((output_dir / "metrics.json").resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run P4-A1 LLM baselines on train/val only."
    )
    parser.add_argument(
        "--baseline",
        choices=("structured", "direct", "both"),
        default="both",
    )
    parser.add_argument(
        "--split",
        action="append",
        choices=("train", "val"),
        dest="splits",
    )
    args = parser.parse_args()
    config = LLMRuntimeConfig.from_env()
    if not config.available:
        print(
            json.dumps(
                {
                    "status": "BLOCKED_AT_H4_RUNTIME",
                    "missing_required": config.missing_required,
                    "runtime": config.public_metadata(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2
    client = OpenAICompatibleJSONClient(config)
    results = []
    for split in args.splits or ["train", "val"]:
        if args.baseline in {"structured", "both"}:
            results.append(run_structured(split, client))
        if args.baseline in {"direct", "both"}:
            results.append(run_direct(split, client))
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
