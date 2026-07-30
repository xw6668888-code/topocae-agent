from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import random
from pathlib import Path
from typing import Any

import main as dataset_main
from utils.ablation_dataset_policy import TARGET_MACRO_FEATURES, audit_ablation_sample, balanced_target_sequence
from utils.generation_context import build_generation_context
from utils.ontology import export_layer_label_map, load_layer_index, load_ontology_maps
from utils.step_solid_validation import inspect_step_file, write_step_with_positive_volume
from utils.topology_rule_audit import audit_graph_directory, audit_graph_topology
from utils.training_export import NODE_FEATURE_MODE_ROTATION_INVARIANT, export_training_package


DEFAULT_SIZE_RANGE = {
    "L": [110.0, 160.0],
    "W": [90.0, 150.0],
    "H": [35.0, 65.0],
}

NO_COMPOSITE_STOCK_PROFILES = {
    "enabled": True,
    "weights": {
        "block": 0.34,
        "plate": 0.26,
        "beam": 0.24,
        "cylinder": 0.16,
        "l_plate": 0.0,
        "composite": 0.0,
    },
    "composite_component_count_range": [2, 3],
}

RIB_L_PLATE_STOCK_PROFILES = {
    "enabled": True,
    "weights": {
        "block": 0.0,
        "plate": 0.0,
        "beam": 0.0,
        "cylinder": 0.0,
        "l_plate": 1.0,
        "composite": 0.0,
    },
    "composite_component_count_range": [2, 3],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate controlled ablation datasets for CAE STEP feature recognition."
    )
    parser.add_argument("--mode", choices=["single_feature", "nonintersecting_multi"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=3200)
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument("--max-generation-attempts", type=int, default=14)
    parser.add_argument("--max-sample-retries", type=int, default=180)
    parser.add_argument("--max-features-per-sample", type=int, default=4)
    parser.add_argument("--min-features-per-sample", type=int, default=2)
    parser.add_argument(
        "--target-macros",
        default="",
        help=(
            "Comma-separated macro features to cover. Defaults to the full "
            "ablation target set."
        ),
    )
    parser.add_argument("--skip-training-export", action="store_true")
    parser.add_argument("--node-feature-mode", default=NODE_FEATURE_MODE_ROTATION_INVARIANT)
    return parser.parse_args()


def resolve_target_macros(target_macros: str | list[str] | None) -> list[str]:
    if target_macros is None or target_macros == "":
        return list(TARGET_MACRO_FEATURES)
    if isinstance(target_macros, str):
        targets = [item.strip() for item in target_macros.split(",") if item.strip()]
    else:
        targets = [str(item).strip() for item in target_macros if str(item).strip()]
    unknown = [target for target in targets if target not in TARGET_MACRO_FEATURES]
    if unknown:
        raise KeyError(f"Unknown ablation target macro feature(s): {unknown}")
    if not targets:
        raise ValueError("target macro list cannot be empty")
    return targets


def load_label_context(root: Path) -> dict[str, Any]:
    ontology_path = root / "CAE_feature_ontology.yaml"
    semantic_subtype_ontology_path = root / "face_semantic_subtype_ontology.yaml"
    ontology_maps = load_ontology_maps(ontology_path)
    primitive_surface_index = load_layer_index(ontology_path, "primitive_surface")
    idealization_tag_index = load_layer_index(ontology_path, "idealization_tag")
    semantic_subtype_index = load_layer_index(semantic_subtype_ontology_path, "face_semantic_subtype")
    primitive_label_map = export_layer_label_map(primitive_surface_index)
    idealization_label_map = export_layer_label_map(idealization_tag_index)
    semantic_label_map = export_layer_label_map(semantic_subtype_index)
    return {
        "ontology_path": str(ontology_path),
        "semantic_subtype_ontology_path": str(semantic_subtype_ontology_path),
        "ontology_maps": ontology_maps,
        "primitive_surface_index": primitive_surface_index,
        "idealization_tag_index": idealization_tag_index,
        "semantic_subtype_index": semantic_subtype_index,
        "primitive_label_map": primitive_label_map,
        "idealization_label_map": idealization_label_map,
        "semantic_label_map": semantic_label_map,
    }


def build_generation_cfg(mode: str, max_features_per_sample: int) -> dict[str, Any]:
    if mode == "single_feature":
        density = {
            "enabled": True,
            "max_total_instances_per_sample": 1,
            "max_instances_per_generator": 1,
        }
    else:
        density = {
            "enabled": True,
            "max_total_instances_per_sample": max_features_per_sample,
            "max_instances_per_generator": 1,
        }

    return {
        "stock_profiles": NO_COMPOSITE_STOCK_PROFILES,
        "feature_ordering": {"enabled": True},
        "feature_orientation": {"hole_axes": ["z", "x", "y"]},
        "feature_density": density,
        "transition_features": {
            "min_edge_length": 8.0,
            "min_size": 0.8,
            "max_size": 2.4,
            "midpoint_clearance_factor": 6.0,
        },
        "rib_features": {
            "placement_modes": ["corner"],
            "corner_probability": 1.0,
        },
    }


def generation_context_for_candidate(
    generation_context: dict[str, Any],
    *,
    target_macro: str | None,
    generator_names: list[str],
) -> dict[str, Any]:
    candidate_context = deepcopy(generation_context)
    has_rib = "RibGenerator" in generator_names
    has_pocket_parent = any(name in {"RectPocketGenerator", "CircularPocketGenerator"} for name in generator_names)
    if has_rib and has_pocket_parent:
        candidate_context["rib_features"] = {
            "placement_modes": ["cavity"],
            "cavity_probability": 1.0,
        }
    elif target_macro == "rib" or has_rib:
        candidate_context["stock_profiles"] = deepcopy(RIB_L_PLATE_STOCK_PROFILES)
        candidate_context["rib_features"] = {
            "placement_modes": ["corner"],
            "corner_probability": 1.0,
        }
    return candidate_context


def generate_dataset(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    random.seed(args.seed)
    label_context = load_label_context(root)
    output_dir = args.output_dir.resolve()
    target_macros = resolve_target_macros(args.target_macros)
    dataset_main.clear_previous_dataset_outputs(str(output_dir))
    label_paths = dataset_main.ensure_label_maps(
        str(output_dir),
        primitive_label_map=label_context["primitive_label_map"],
        macro_label_map=label_context["ontology_maps"].export_label_map(),
        semantic_label_map=label_context["semantic_label_map"],
        idealization_label_map=label_context["idealization_label_map"],
    )
    generation_cfg = build_generation_cfg(args.mode, args.max_features_per_sample)
    generation_context = build_generation_context(generation_cfg)

    if args.mode == "single_feature":
        macro_sequence = balanced_target_sequence(target_macros, args.sample_count, seed=args.seed)
    else:
        macro_sequence = []

    accepted = []
    rejects = Counter()
    macro_counter = Counter()
    for sample_index in range(args.sample_count):
        graph_data = None
        attempts_used = 0
        for attempt in range(1, args.max_sample_retries + 1):
            attempts_used = attempt
            candidate = try_generate_candidate(
                mode=args.mode,
                target_macro=macro_sequence[sample_index] if macro_sequence else None,
                generation_context=generation_context,
                label_context=label_context,
                target_macros=target_macros,
                max_generation_attempts=args.max_generation_attempts,
                max_features_per_sample=args.max_features_per_sample,
            )
            if candidate is None:
                rejects["generation_failed"] += 1
                continue
            candidate_graph = candidate["graph_data"]
            policy_report = audit_ablation_sample(
                candidate_graph,
                policy=args.mode,
                max_feature_sets=args.max_features_per_sample,
                min_feature_sets=args.min_features_per_sample,
            )
            if not policy_report["valid"]:
                for reason in policy_report["reasons"]:
                    rejects[reason] += 1
                continue
            topology_report = audit_graph_topology(candidate_graph, graph_path=f"candidate_{sample_index:03d}.json")
            if topology_report.get("invalid_feature_set_count", 0) > 0:
                rejects["invalid_topology_rule"] += 1
                continue
            if args.mode == "single_feature" and not dataset_main.is_valid_catalog_sample(candidate_graph, macro_sequence[sample_index]):
                rejects["target_macro_mismatch"] += 1
                continue
            if not write_validated_outputs(
                candidate["stock"],
                candidate_graph,
                output_dir,
                sample_index,
            ):
                rejects["invalid_step_write"] += 1
                continue

            candidate_graph["ablation_policy"] = policy_report
            write_graph(output_dir / "graphs" / f"composite_feature_{sample_index:03d}.json", candidate_graph)
            graph_data = candidate_graph
            break

        if graph_data is None:
            raise RuntimeError(
                f"Failed to generate accepted {args.mode} sample index={sample_index} "
                f"after {args.max_sample_retries} retries; recent rejects={dict(rejects)}"
            )

        macro_counter.update(dataset_main.count_feature_instances(graph_data))
        accepted.append({
            "sample_id": f"composite_feature_{sample_index:03d}",
            "attempts_used": attempts_used,
            "feature_set_count": len(graph_data.get("feature_sets", [])),
            "macro_counts": dict(dataset_main.count_feature_instances(graph_data)),
            "stock_type": graph_data.get("stock_profile", {}).get("stock_type"),
        })
        if (sample_index + 1) % 100 == 0:
            print(f"{args.mode}: accepted {sample_index + 1}/{args.sample_count}")

    coverage_summary = {
        "enabled": True,
        "ablation_mode": args.mode,
        "actual_samples_written": len(accepted),
        "target_macro_features": list(target_macros),
        "achieved_instance_counts": dict(sorted(macro_counter.items())),
        "reject_counts": dict(sorted(rejects.items())),
    }
    write_json(output_dir / "reports" / "ablation_generation_report.json", {
        "mode": args.mode,
        "sample_count": len(accepted),
        "seed": args.seed,
        "target_macro_features": list(target_macros),
        "max_features_per_sample": args.max_features_per_sample,
        "min_features_per_sample": args.min_features_per_sample,
        "macro_counts": dict(sorted(macro_counter.items())),
        "reject_counts": dict(sorted(rejects.items())),
        "samples": accepted,
    })

    dataset_main.write_dataset_meta(
        output_dir=str(output_dir),
        config_path="generate_ablation_dataset.py",
        ontology_path=label_context["ontology_path"],
        semantic_subtype_ontology_path=label_context["semantic_subtype_ontology_path"],
        primitive_label_map_path=label_paths[0],
        macro_label_map_path=label_paths[1],
        semantic_label_map_path=label_paths[2],
        idealization_tag_label_map_path=label_paths[3],
        runtime_config=runtime_config_for_meta(args, generation_cfg),
        actual_samples_written=len(accepted),
        coverage_summary=coverage_summary,
    )
    audit_graph_directory(output_dir / "graphs", output_dir / "reports" / "topology_rule_audit.json")
    if not args.skip_training_export:
        export_training_package(
            output_dir=str(output_dir),
            split_ratios={"train": 0.8, "val": 0.1, "test": 0.1},
            seed=args.seed,
            node_feature_mode=args.node_feature_mode,
        )
    return coverage_summary


def try_generate_candidate(
    *,
    mode: str,
    target_macro: str | None,
    generation_context: dict[str, Any],
    label_context: dict[str, Any],
    target_macros: list[str],
    max_generation_attempts: int,
    max_features_per_sample: int,
) -> dict[str, Any] | None:
    size = DEFAULT_SIZE_RANGE
    if mode == "single_feature":
        generator_name = dataset_main.CATALOG_GENERATOR_BY_MACRO.get(target_macro or "")
        if generator_name is None:
            raise KeyError(f"No generator for target macro: {target_macro}")
        generator_names = [generator_name]
        instance_count_range = (1, 1)
    else:
        max_k = min(max_features_per_sample, len(target_macros))
        min_k = min(2, max_k)
        selected_targets = random.sample(target_macros, k=random.randint(min_k, max_k))
        generator_names = sorted({
            dataset_main.CATALOG_GENERATOR_BY_MACRO[target]
            for target in selected_targets
            if target in dataset_main.CATALOG_GENERATOR_BY_MACRO
        })
        instance_count_range = (1, 1)

    try:
        candidate_generation_context = generation_context_for_candidate(
            generation_context,
            target_macro=target_macro,
            generator_names=generator_names,
        )
        stock, graph_data, _feature_count = dataset_main.generate_annotated_sample(
            L=random.uniform(*size["L"]),
            W=random.uniform(*size["W"]),
            H=random.uniform(*size["H"]),
            ontology_maps=label_context["ontology_maps"],
            primitive_surface_index=label_context["primitive_surface_index"],
            semantic_subtype_index=label_context["semantic_subtype_index"],
            idealization_tag_index=label_context["idealization_tag_index"],
            primitive_label_map=label_context["primitive_label_map"],
            semantic_label_map=label_context["semantic_label_map"],
            idealization_label_map=label_context["idealization_label_map"],
            generator_names=generator_names,
            instance_count_range=instance_count_range,
            require_nonempty_sample=True,
            max_generation_attempts=max_generation_attempts,
            generation_context=candidate_generation_context,
        )
    except Exception:
        return None
    return {"stock": stock, "graph_data": graph_data}


def write_validated_outputs(stock, graph_data: dict[str, Any], output_dir: Path, sample_index: int) -> bool:
    output_step = output_dir / "steps" / f"composite_feature_{sample_index:03d}.step"
    output_graph = output_dir / "graphs" / f"composite_feature_{sample_index:03d}.json"
    output_step.parent.mkdir(parents=True, exist_ok=True)
    output_graph.parent.mkdir(parents=True, exist_ok=True)
    write_step_with_positive_volume(stock, output_step)
    step_report = inspect_step_file(output_step)
    if not dataset_main.step_write_report_is_acceptable(step_report):
        try:
            output_step.unlink()
        except FileNotFoundError:
            pass
        return False
    graph_data.setdefault("generation_quality", {})
    graph_data["generation_quality"]["step_write_validation"] = step_report
    write_graph(output_graph, graph_data)
    return True


def runtime_config_for_meta(args: argparse.Namespace, generation_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": {
            "num_samples": args.sample_count,
            "random_seed": args.seed,
            "stock_size_range": DEFAULT_SIZE_RANGE,
        },
        "generation": {
            "enabled_generators": list(dataset_main.STATIC_STRUCTURAL_V1_GENERATORS),
            "instance_count_range": [1, 1],
            "require_nonempty_sample": True,
                "coverage_control": {
                    "enabled": True,
                    "target_macro_features": resolve_target_macros(args.target_macros),
                    "min_instances_per_macro": 0,
                },
            **generation_cfg,
        },
        "exports": {
            "write_step": True,
            "write_graph": True,
            "build_training_package": not args.skip_training_export,
            "split_ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
            "llm_round_ndigits": 4,
            "gnn_round_ndigits": 6,
            "node_feature_mode": args.node_feature_mode,
        },
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")


def write_graph(path: Path, graph_data: dict[str, Any]) -> None:
    path.write_text(json.dumps(graph_data, indent=4, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    summary = generate_dataset(args)
    print(f"ablation_dataset_complete: {args.output_dir}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
