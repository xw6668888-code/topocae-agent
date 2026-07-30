from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

from .common import canonical_object_signature, read_json, strip_graph_truth, validate_graph
from .runtime import (
    device_from_name,
    get_spec,
    load_candidate_model,
    load_gnn,
    predict_faces,
    prepare_graph,
    run_frozen_pipeline,
)


OUTPUT_SCHEMA = "topocae_recognizer_dev_feature_objects_1"
PIPELINE_KEY = "p1_b"


class RecognizerDevError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def recognize_graph(
    graph: Mapping[str, Any],
    *,
    step_path: Path | None = None,
    device_name: str = "cpu",
) -> dict[str, Any]:
    """Run the frozen P1-B recognizer without mutating model or historical assets."""

    try:
        validate_graph(graph)
    except (TypeError, ValueError) as exc:
        raise RecognizerDevError("E_SCHEMA", str(exc)) from exc
    spec = get_spec(PIPELINE_KEY)
    device = device_from_name(device_name)
    try:
        prepared = prepare_graph(spec, graph, step_path)
        gnn, gnn_config = load_gnn(spec, device)
        candidate_model = load_candidate_model(spec, device)
        prediction = predict_faces(spec, prepared, gnn, gnn_config, device)
        pipeline = run_frozen_pipeline(spec, prepared, prediction, candidate_model, device)
    except FileNotFoundError as exc:
        raise RecognizerDevError("E_MODEL_ASSET", str(exc)) from exc
    except RecognizerDevError:
        raise
    except Exception as exc:
        raise RecognizerDevError("E_INFERENCE", str(exc)) from exc

    decoded = pipeline["decoded_feature_objects"]
    objects = list(decoded.get("feature_objects", []))
    abstained = list(decoded.get("abstained_candidates", []))
    conflicts = list(decoded.get("global_conflict_rejections", []))
    if not objects:
        coverage_status = "ABSTAIN"
        failure_reason = "no accepted feature object"
    elif abstained or conflicts:
        coverage_status = "PARTIAL"
        failure_reason = "one or more candidates abstained or conflicted"
    else:
        coverage_status = "ACCEPTED"
        failure_reason = None
    return {
        "schema_version": OUTPUT_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "recognizer": "Recognizer-Dev / P1-B frozen",
        "capability_level": "LIMITED_CONTROLLED_DOMAIN",
        "graph_schema": prepared.get("schema_version"),
        "sample_id": graph.get("sample_id"),
        "coverage_status": coverage_status,
        "failure_reason": failure_reason,
        "feature_objects": objects,
        "feature_object_signature": canonical_object_signature(objects),
        "abstentions": [
            {
                "candidate_id": record.get("candidate_id"),
                "reasons": record.get("abstention_reasons", []),
                "final_confidence": record.get("final_confidence"),
            }
            for record in abstained
        ],
        "conflicts": conflicts,
        "pipeline_manifest": pipeline["pipeline_manifest"],
        "contracts": {
            "truth_used_during_inference": False,
            "input_truth_fields_removed": strip_graph_truth(graph) == prepared,
            "historical_assets_mutated": False,
            "sample_specific_thresholds": False,
        },
    }


def error_payload(error: RecognizerDevError) -> dict[str, Any]:
    return {
        "schema_version": OUTPUT_SCHEMA,
        "coverage_status": "ERROR",
        "error": {"code": error.code, "message": str(error)},
        "feature_objects": [],
        "abstentions": [],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only Recognizer-Dev inference entry")
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--step", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        graph = read_json(args.graph)
        result = recognize_graph(graph, step_path=args.step, device_name=args.device)
        code = 0
    except (OSError, json.JSONDecodeError) as exc:
        result = error_payload(RecognizerDevError("E_INPUT", str(exc)))
        code = 2
    except RecognizerDevError as exc:
        result = error_payload(exc)
        code = 2
    text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        if args.output.exists():
            raise SystemExit("refusing to overwrite an existing output")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return code


if __name__ == "__main__":
    raise SystemExit(main())

