from __future__ import annotations

import json
from pathlib import Path

import evaluate_paper_v1_recognition as base
from evaluate_paper_v1_recognition_v2 import fast_bootstrap_object_metrics
from prediction_to_feature_objects import build_feature_objects
from utils.rib_boundary_classifier_postprocess import apply_rib_boundary_classifier_postprocess
from utils.rib_candidate_proposer_postprocess import apply_rib_candidate_proposer_postprocess


def resumable_cached_predict_dataset(
    *,
    variant: str,
    graph_paths: list[Path],
    output_dir: Path,
    models: dict,
    dataset_config: dict,
) -> dict:
    prediction_dir = output_dir / "predictions"
    feature_dir = output_dir / "feature_objects"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    reused = 0
    generated = 0
    if variant == "clean_rib":
        core_dir = output_dir.parent.parent / "clean_core" / output_dir.name
        if not (core_dir / "manifest.json").is_file():
            raise RuntimeError(
                f"Clean-Core cache must be complete before Clean-Rib: {core_dir}"
            )
    for index, graph_path in enumerate(graph_paths, start=1):
        graph_path = base.assert_allowed_path(graph_path, dataset_config)
        graph = base.load_json(graph_path)
        prediction_path = prediction_dir / f"{graph_path.stem}.json"
        feature_path = feature_dir / f"{graph_path.stem}.json"
        can_reuse = prediction_path.is_file() and feature_path.is_file()
        if can_reuse:
            candidate_prediction = base.load_json(prediction_path)
            candidate_features = base.load_json(feature_path)
            can_reuse = (
                candidate_prediction.get("sample_id") == graph_path.stem
                and candidate_features.get("sample_id") == graph_path.stem
            )
        if can_reuse:
            reused += 1
        elif variant == "clean_core":
            prediction = base.predict_one_graph(
                model=models["model"],
                graph_data=graph,
                graph_path=graph_path,
                checkpoint_config=models["checkpoint"]["config"],
                ndigits=6,
                device=models["device"],
                topology_postprocess=True,
                same_instance_threshold=0.5,
                candidate_classifier_bundle=models["candidate"],
                candidate_classifier_min_confidence=0.5,
            )
            feature_doc = build_feature_objects(prediction, graph, ndigits=6)
            base.write_json(prediction_path, prediction)
            base.write_json(feature_path, feature_doc)
            generated += 1
        else:
            core_dir = output_dir.parent.parent / "clean_core" / output_dir.name
            core_prediction_path = core_dir / "predictions" / f"{graph_path.stem}.json"
            prediction = base.load_json(core_prediction_path)
            prediction = apply_rib_candidate_proposer_postprocess(
                prediction,
                graph,
                models["proposer"],
                min_confidence=models["proposer_threshold"],
            )
            prediction = apply_rib_boundary_classifier_postprocess(
                prediction,
                graph,
                models["boundary"],
                min_confidence=models["boundary_threshold"],
            )
            feature_doc = build_feature_objects(prediction, graph, ndigits=6)
            base.write_json(prediction_path, prediction)
            base.write_json(feature_path, feature_doc)
            generated += 1
        manifest.append(
            {
                "sample_id": graph_path.stem,
                "graph_path": str(graph_path.resolve()),
                "prediction_path": str(prediction_path.resolve()),
                "feature_objects_path": str(feature_path.resolve()),
            }
        )
        if index % 250 == 0 or index == len(graph_paths):
            print(
                f"evaluation_progress={variant}:{output_dir.name}:{index}/{len(graph_paths)} "
                f"generated={generated} reused={reused}"
            )
    payload = {
        "variant": variant,
        "checkpoint": str(models["checkpoint_path"].resolve()),
        "candidate_classifier": str(models["candidate_path"].resolve()),
        "rib_candidate_proposer": str(models["proposer_path"].resolve()) if variant == "clean_rib" else None,
        "rib_boundary_classifier": str(models["boundary_path"].resolve()) if variant == "clean_rib" else None,
        "rib_candidate_proposer_threshold": models["proposer_threshold"] if variant == "clean_rib" else None,
        "rib_boundary_classifier_threshold": models["boundary_threshold"] if variant == "clean_rib" else None,
        "sample_count": len(manifest),
        "generated_count": generated,
        "reused_count": reused,
        "clean_rib_optimization": (
            "apply rib postprocessors to the same-run Clean-Core prediction after candidate+topology"
            if variant == "clean_rib"
            else None
        ),
        "samples": manifest,
    }
    base.write_json(output_dir / "manifest.json", payload)
    return payload


if __name__ == "__main__":
    base.bootstrap_object_metrics = fast_bootstrap_object_metrics
    base.predict_dataset = resumable_cached_predict_dataset
    base.main()
