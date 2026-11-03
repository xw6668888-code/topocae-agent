"""Truth-free frozen G07 R/Q predictions on the sealed G09 V09 geometry.

The pilot has a different output namespace and cannot be passed off as a
complete 930-part validation run.  This process never opens target artifacts.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time

from candidate_pool import STAGES, build_candidate_pool, capture_legacy_stages
from g09_common import ARTIFACT, FROZEN, G03_ARTIFACT, G03_TOOLS, G07_TOOLS, REPORT, SEEDS, checked, descriptor, now, read, write_json


V09 = ARTIFACT / "v2/v09"
PILOT = ARTIFACT / "v2/d2/pilot"
PREDICTIONS = V09 / "predictions"
BRANCHES = {"R": ("old", None), "Q": ("new", "dense_instance")}
PILOT_NAMESPACES = {"R": "pilot_2_e_aligned_v3", "Q": "pilot_2_e_aligned_v3"}
COUNT = 930


def source_rows(*, pilot: bool = False) -> tuple[dict, list[dict]]:
    """Return only input identities, never a target descriptor or label."""
    folder = PILOT if pilot else V09
    manifest_path = folder / "manifest.json"
    manifest = read(manifest_path)
    rows = manifest["rows"]
    if pilot:
        if manifest.get("schema") != "g09_pilot_smoke_manifest_v1" or len(rows) < 2:
            raise ValueError("Two accepted v2 pilot inputs are required")
        rows = rows[:2]
        if any(row.get("use") != "pilot" for row in rows):
            raise ValueError("Only pilot geometry is permitted in pilot mode")
    else:
        seal = read(folder / "COMPLETE.json")
        if (seal.get("manifest") != descriptor(manifest_path)
                or seal.get("model_scores_used") is not False
                or seal.get("final_test_generated") is not False):
            raise ValueError("V09 geometry is not independently sealed")
        if (manifest.get("schema") != "g09_v09_validation_geometry_manifest_v1"
                or manifest.get("complete") is not True
                or manifest.get("geometry_only") is not True
                or manifest.get("model_scores_used") is not False
                or manifest.get("final_test_generated") is not False
                or len(rows) != COUNT
                or Counter(row.get("use") for row in rows) !=
                    {"val_parameter": 465, "val_structure": 465}):
            raise ValueError("V09 sealed geometry denominator or strata changed")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate input sample identity")
    inputs = []
    for row in rows:
        step, graph = checked(row["step"]), checked(row["input_graph"])
        if not step.resolve().is_relative_to((folder / "inputs/steps").resolve()):
            raise ValueError("STEP source is outside the registered unlabeled inputs")
        if not graph.resolve().is_relative_to((folder / "inputs/graphs").resolve()):
            raise ValueError("Graph source is outside the registered unlabeled inputs")
        if not isinstance(row.get("face_count"), int) or row["face_count"] < 1:
            raise ValueError("Invalid input face count")
        inputs.append({"sample_id": row["sample_id"], "step": row["step"],
                       "input_graph": row["input_graph"], "face_count": row["face_count"]})
    return descriptor(manifest_path), inputs


def _frozen_imports():
    if str(G07_TOOLS) not in sys.path:
        sys.path.insert(0, str(G07_TOOLS))
    from step_entry import geometry_only
    from topo_predictor import Predictor, public_objects
    return geometry_only, Predictor, public_objects


def _identity(predictor, branch: str, seed: int) -> dict:
    checkpoint = descriptor(predictor.checkpoint)
    metadata = descriptor(predictor.metadata_path)
    if branch == "R":
        threshold = descriptor(Path(predictor.checkpoint).parent / "same_instance_threshold_selection.json")
        registry = descriptor(REPORT.parent / "g03/CONFIGURATION_FREEZE.json")
        training = {"configuration_freeze": registry}
    else:
        metrics_path = G03_ARTIFACT / f"training_runs/dense_instance/seed_{seed}/metrics.json"
        metrics = read(metrics_path)
        if checked(metrics["checkpoint"]) != Path(predictor.checkpoint).resolve():
            raise ValueError("Q checkpoint differs from its frozen training run")
        threshold = descriptor(Path(predictor.checkpoint).parent / "same_instance_threshold_selection.json")
        training = {"metrics": descriptor(metrics_path),
                    "configuration_freeze": descriptor(REPORT.parent / "g03/CONFIGURATION_FREEZE.json")}
    decoder_base = FROZEN / "reports/p1_e3c_seed_matched_candidate_v1/pipelines/old_same_instance_seed_20260718"
    decoder = [descriptor(decoder_base / path) for path in
               ("candidate_model/candidate_threshold_selection.json", "global_decode_selection.json")]
    sources = [descriptor(path) for path in
               (G07_TOOLS / "topo_predictor.py", G07_TOOLS / "step_entry.py",
                G07_TOOLS / "decoder.py", G03_TOOLS / "inference.py",
                G03_TOOLS / "stages.py", G03_TOOLS / "ablations.py",
                Path(__file__), Path(__file__).with_name("candidate_pool.py"))]
    return {"dataset": "D0", "seed": seed, "branch": branch,
            "configuration": predictor.config, "checkpoint": checkpoint,
            "normalization": metadata, "threshold": threshold,
            "legacy_decoder_assets": decoder, "training": training,
            "inference_sources": sources}


def _prediction_row(row: dict, graph: dict, prediction: dict, predictor,
                    public_objects) -> dict:
    count = row["face_count"]
    faces = prediction["faces"]
    if len(graph["nodes"]) != count or len(faces) != count or [face["idx"] for face in faces] != list(range(count)):
        raise ValueError("Predicted face mapping differs from final STEP face indices")
    if len(prediction["edge_probabilities"]) != len(graph["edges"]):
        raise ValueError("Predicted physical-edge count changed")
    if predictor.method == "new":
        if len(prediction["relation_edges"]) != len(prediction["relation_probabilities"]):
            raise ValueError("Predicted relation-pair count changed")
        recovered, _, _ = predictor.recover(graph, prediction)
        stages = capture_legacy_stages(graph, prediction, predictor.same,
                                       predictor.candidate, predictor.global_decode)
        alternative = {"dense_minimal": public_objects(recovered["full"])}
    else:
        stages = capture_legacy_stages(graph, prediction, predictor.same,
                                       predictor.candidate, predictor.global_decode)
        alternative = {}
    if set(stages) != set(STAGES):
        raise ValueError("Eight registered candidate stages are required")
    pool = build_candidate_pool({"R" if predictor.method == "old" else "Q": stages},
                                face_count=count)
    objects = public_objects(stages["final"])
    if any(len(face["macro_probabilities"]) != 19 for face in faces):
        raise ValueError("Frozen face probabilities must cover 19 classes")
    return {"sample_id": row["sample_id"], "source_step_sha256": row["step"]["sha256"],
            "face_count": count, "face_macro_classes": [face["macro_feature"] for face in faces],
            "objects": objects, "raw_prediction": prediction, "stages": stages,
            "candidate_pool": pool, "alternative_origins": alternative,
            "truth_used": False}


def predict(branch: str, seed: int, *, pilot: bool = False, batch_size: int = 4) -> dict:
    if branch not in BRANCHES or seed not in SEEDS:
        raise ValueError("Use a registered R/Q branch and one of the three frozen seeds")
    if not 1 <= batch_size <= 16:
        raise ValueError("CPU batch size must be between 1 and 16")
    manifest, rows = source_rows(pilot=pilot)
    geometry_only, Predictor, public_objects = _frozen_imports()
    import torch
    torch.set_num_threads(min(4, torch.get_num_threads()))
    method, configuration = BRANCHES[branch]
    predictor = Predictor("D0", method, seed, device="cpu", configuration=configuration)
    identity = _identity(predictor, branch, seed)
    label = f"{branch}_seed_{seed}"
    output = PREDICTIONS / (PILOT_NAMESPACES[branch] if pilot else "full_930") / label
    seal_path = output / "COMPLETE.json"
    if seal_path.exists():
        previous = read(seal_path)
        if (previous.get("manifest") != manifest or previous.get("model_identity") != identity
                or previous.get("sample_count") != len(rows) or previous.get("method") != label
                or previous.get("truth_used") is not False):
            raise ValueError("Existing R/Q prediction seal has another source or model identity")
        checked(previous["predictions"])
        return previous
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Incomplete prediction run retained; choose a new versioned attempt")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    count = 0
    temporary = output / "predictions.jsonl.partial"
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for start in range(0, len(rows), batch_size):
                block = rows[start:start + batch_size]
                graphs = []
                for row in block:
                    # Only the input graph is opened; geometry_only drops all nongeometry keys.
                    graph = geometry_only(read(checked(row["input_graph"])))
                    if len(graph["nodes"]) != row["face_count"]:
                        raise ValueError("Input graph final STEP face count changed")
                    graphs.append(graph)
                predictions = predictor.forward(predictor.inputs(graphs))
                if len(predictions) != len(block):
                    raise ValueError("Frozen network returned a different batch denominator")
                for row, graph, prediction in zip(block, graphs, predictions, strict=True):
                    result = _prediction_row(row, graph, prediction, predictor, public_objects)
                    stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                    count += 1
                if count % 50 < batch_size or count == len(rows):
                    print(f"G09 {label} {count}/{len(rows)}", flush=True)
        final = output / "predictions.jsonl"
        os.replace(temporary, final)
        seal = {"schema": "g09_v09_predictions_v1", "at_utc": now(),
                "method": label, "manifest": manifest, "predictions": descriptor(final),
                "sample_count": len(rows), "model_identity": identity,
                "elapsed_seconds": time.perf_counter() - started,
                "pilot_only": pilot, "truth_used": False, "final_test_exposed": False}
        write_json(seal_path, seal)
        return seal
    except BaseException as error:
        write_json(output / "FAILED.json", {"at_utc": now(), "method": label,
                                             "manifest": manifest, "completed_samples": count,
                                             "error_type": type(error).__name__, "error": str(error),
                                             "partial_output_retained": True})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", choices=tuple(BRANCHES), required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--pilot-two", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    result = predict(args.branch, args.seed, pilot=args.pilot_two, batch_size=args.batch_size)
    print(json.dumps({key: result[key] for key in ("method", "sample_count", "predictions", "truth_used")},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
