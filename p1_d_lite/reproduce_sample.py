from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import time

from .common import (
    canonical_object_signature,
    read_json,
    sha256_file,
    strip_graph_truth,
    write_json,
)
from .runtime import (
    device_from_name,
    get_spec,
    load_candidate_model,
    load_gnn,
    predict_faces,
    prepare_graph,
    run_frozen_pipeline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", choices=("p1_b", "p1_c"), required=True)
    parser.add_argument("--mode", choices=("synthetic_dev", "real_dev"), required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--step", type=Path)
    parser.add_argument("--reference-objects", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec = get_spec(args.pipeline)
    graph_source = read_json(args.graph)
    offline_truth_count = len(graph_source.get("feature_sets", []))
    device = device_from_name(args.device)
    started = time.perf_counter()
    graph = prepare_graph(spec, graph_source, args.step)
    gnn, gnn_config = load_gnn(spec, device)
    candidate_model = load_candidate_model(spec, device)
    face_prediction = predict_faces(spec, graph, gnn, gnn_config, device)
    pipeline = run_frozen_pipeline(spec, graph, face_prediction, candidate_model, device)
    feature_objects = pipeline["decoded_feature_objects"]["feature_objects"]
    signature = canonical_object_signature(feature_objects)
    reference_signature = None
    if args.reference_objects:
        reference = read_json(args.reference_objects)
        reference_signature = canonical_object_signature(reference.get("feature_objects", []))
    result = {
        "schema_version": "topocae_p1_d_lite_reproduction_1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": spec.name,
        "pipeline_entry": spec.pipeline,
        "mode": args.mode,
        "device": str(device),
        "graph_path": str(args.graph.resolve()),
        "graph_sha256": sha256_file(args.graph),
        "step_path": str(args.step.resolve()) if args.step else None,
        "step_sha256": sha256_file(args.step) if args.step else None,
        "truth_fields_present_in_source": bool(graph_source.get("feature_sets")),
        "truth_fields_present_at_inference": bool(strip_graph_truth(graph).get("feature_sets")),
        "offline_truth_object_count": offline_truth_count,
        "face_count": len(graph.get("nodes", [])),
        "proposal_count": len(pipeline["candidate_proposals"].get("proposals", [])),
        "verified_candidate_count": len(pipeline["verified_candidates"]),
        "accepted_candidate_count": int(pipeline["decoded_feature_objects"]["accepted_candidate_count"]),
        "abstention_count": int(pipeline["decoded_feature_objects"]["abstention_count"]),
        "feature_object_count": len(feature_objects),
        "feature_object_signature": signature,
        "reference_object_path": str(args.reference_objects.resolve()) if args.reference_objects else None,
        "reference_signature_match": signature == reference_signature if reference_signature is not None else None,
        "pipeline_manifest": pipeline["pipeline_manifest"],
        "elapsed_seconds": time.perf_counter() - started,
        "contracts": {
            "truth_used_during_inference": False,
            "historical_output_modified": False,
            "formal_holdout_rerun": False,
        },
    }
    write_json(args.output, result)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

