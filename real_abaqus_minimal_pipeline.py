from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    current_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run the minimal real-Abaqus integration pipeline from STEP to Abaqus action plan and script."
    )
    parser.add_argument("--step-path", type=Path, required=True)
    parser.add_argument("--topology-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--feature-round-ndigits", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--simulation-name", default="AI_Job_RealPipeline")
    parser.add_argument("--model-name", default="Model-1")
    parser.add_argument("--part-name", default="Convex_Concave")
    parser.add_argument("--instance-name", default="Convex_Concave-1")
    parser.add_argument("--material-name", default="Steel")
    parser.add_argument("--elastic-modulus", type=float, default=210000.0)
    parser.add_argument("--poisson-ratio", type=float, default=0.3)
    parser.add_argument("--global-size", type=float, default=2.0)
    parser.add_argument("--local-size", type=float, default=0.5)
    parser.add_argument(
        "--min-region-confidence",
        choices=["low_confidence", "medium_confidence", "high_confidence"],
        default="medium_confidence",
    )
    parser.add_argument(
        "--bc-mode",
        choices=["none", "largest_stock_planar", "region_name"],
        default="largest_stock_planar",
    )
    parser.add_argument("--bc-region-name", default=None)
    parser.add_argument("--bc-u1", type=int, choices=[0, 1], default=0)
    parser.add_argument("--bc-u2", type=int, choices=[0, 1], default=0)
    parser.add_argument("--bc-u3", type=int, choices=[0, 1], default=0)
    parser.add_argument(
        "--load-mode",
        choices=["none", "first_feature_region", "region_name", "macro_feature"],
        default="first_feature_region",
    )
    parser.add_argument("--load-region-name", default=None)
    parser.add_argument("--load-macro-feature", default=None)
    parser.add_argument("--load-type", choices=["Pressure"], default="Pressure")
    parser.add_argument("--load-value", type=float, default=10.0)
    parser.add_argument(
        "--cae-step5-script",
        type=Path,
        default=current_dir.parent / "cae" / "step5_script_gen.py",
        help="Path to the Abaqus script generator.",
    )
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    current_dir = Path(__file__).resolve().parent
    python_exe = sys.executable

    step_path = args.step_path.resolve()
    topology_path = args.topology_path.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output_root = args.output_root.resolve()

    if not step_path.exists():
        raise FileNotFoundError(f"STEP file not found: {step_path}")
    if not topology_path.exists():
        raise FileNotFoundError(f"Topology snapshot not found: {topology_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    inference_dir = output_root / "inference_graphs"
    prediction_dir = output_root / "predictions"
    feature_object_dir = output_root / "feature_objects"
    mapping_dir = output_root / "abaqus_mapping"
    action_plan_dir = output_root / "action_plans"
    abaqus_script_dir = output_root / "abaqus_scripts"

    sample_id = step_path.stem
    inference_graph_path = inference_dir / f"{sample_id}.json"
    prediction_path = prediction_dir / f"{sample_id}.json"
    feature_object_path = feature_object_dir / f"{sample_id}.json"
    mapping_path = mapping_dir / f"{sample_id}.json"
    action_plan_path = action_plan_dir / f"{sample_id}.json"
    abaqus_script_path = abaqus_script_dir / f"{sample_id}.py"

    step_to_graph_cmd = [
        python_exe,
        str((current_dir / "step_to_inference_graph.py").resolve()),
        "--input-path", str(step_path),
        "--output-dir", str(inference_dir),
    ]
    if args.overwrite:
        step_to_graph_cmd.append("--overwrite")
    run_command(step_to_graph_cmd)

    run_command([
        python_exe,
        str((current_dir / "run_gnn_inference.py").resolve()),
        "--checkpoint", str(checkpoint_path),
        "--input-path", str(inference_graph_path),
        "--output-dir", str(prediction_dir),
        "--device", args.device,
        "--feature-round-ndigits", str(args.feature_round_ndigits),
    ])

    run_command([
        python_exe,
        str((current_dir / "prediction_to_feature_objects.py").resolve()),
        "--prediction-path", str(prediction_path),
        "--graph-dir", str(inference_dir),
        "--output-dir", str(feature_object_dir),
    ])

    run_command([
        python_exe,
        str((current_dir / "prediction_to_abaqus_mapping.py").resolve()),
        "--feature-object-path", str(feature_object_path),
        "--topology-path", str(topology_path),
        "--output-dir", str(mapping_dir),
    ])

    action_plan_cmd = [
        python_exe,
        str((current_dir / "abaqus_mapping_to_action_plan.py").resolve()),
        "--mapping-path", str(mapping_path),
        "--output-path", str(action_plan_path),
        "--simulation-name", args.simulation_name,
        "--model-name", args.model_name,
        "--part-name", args.part_name,
        "--instance-name", args.instance_name,
        "--material-name", args.material_name,
        "--elastic-modulus", str(args.elastic_modulus),
        "--poisson-ratio", str(args.poisson_ratio),
        "--global-size", str(args.global_size),
        "--local-size", str(args.local_size),
        "--min-region-confidence", args.min_region_confidence,
        "--bc-mode", args.bc_mode,
        "--bc-u1", str(args.bc_u1),
        "--bc-u2", str(args.bc_u2),
        "--bc-u3", str(args.bc_u3),
        "--load-mode", args.load_mode,
        "--load-type", args.load_type,
        "--load-value", str(args.load_value),
    ]
    if args.bc_region_name:
        action_plan_cmd.extend(["--bc-region-name", args.bc_region_name])
    if args.load_region_name:
        action_plan_cmd.extend(["--load-region-name", args.load_region_name])
    if args.load_macro_feature:
        action_plan_cmd.extend(["--load-macro-feature", args.load_macro_feature])
    run_command(action_plan_cmd)

    run_command([
        python_exe,
        str(args.cae_step5_script.resolve()),
        "--input-action-file", str(action_plan_path),
        "--output-script-file", str(abaqus_script_path),
        "--model-name", args.model_name,
        "--part-name", args.part_name,
        "--instance-name", args.instance_name,
    ])

    required_outputs = [
        inference_graph_path,
        prediction_path,
        feature_object_path,
        mapping_path,
        action_plan_path,
        abaqus_script_path,
    ]
    missing_outputs = [str(path) for path in required_outputs if not path.exists()]
    if missing_outputs:
        raise FileNotFoundError(f"Pipeline completed with missing outputs: {missing_outputs}")

    manifest = {
        "pipeline_mode": "real_abaqus_minimal_pipeline",
        "sample_id": sample_id,
        "source_step_path": str(step_path),
        "source_topology_path": str(topology_path),
        "checkpoint_path": str(checkpoint_path),
        "outputs": {
            "inference_graph_path": str(inference_graph_path.resolve()),
            "prediction_path": str(prediction_path.resolve()),
            "feature_object_path": str(feature_object_path.resolve()),
            "mapping_path": str(mapping_path.resolve()),
            "action_plan_path": str(action_plan_path.resolve()),
            "abaqus_script_path": str(abaqus_script_path.resolve()),
        },
        "configuration": {
            "simulation_name": args.simulation_name,
            "model_name": args.model_name,
            "part_name": args.part_name,
            "instance_name": args.instance_name,
            "material_name": args.material_name,
            "global_size": args.global_size,
            "local_size": args.local_size,
            "min_region_confidence": args.min_region_confidence,
            "bc_mode": args.bc_mode,
            "bc_region_name": args.bc_region_name,
            "load_mode": args.load_mode,
            "load_region_name": args.load_region_name,
            "load_macro_feature": args.load_macro_feature,
            "load_type": args.load_type,
            "load_value": args.load_value,
        },
    }
    write_json(output_root / "real_pipeline_manifest.json", manifest)
    print(f"real pipeline manifest written: {output_root / 'real_pipeline_manifest.json'}")


if __name__ == "__main__":
    main()
