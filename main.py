import argparse
import json
import os
import random
import shutil
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from OCC.Core.gp import gp_Pnt
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs

from core.AAGExtractor import extract_aag_graph
from features.bearing_seat import BearingSeatGenerator
from features.bolt_hole_pattern import BoltHolePatternGenerator
from features.blind_hole import BlindHoleGenerator
from features.boss_feature import BossFeatureGenerator
from features.chamfer_feature import ChamferGenerator
from features.circular_pocket import CircularPocketGenerator
from features.counterbore_hole import CounterboreHoleGenerator
from features.countersink_hole import CountersinkHoleGenerator
from features.fillet_feature import FilletGenerator
from features.flange import FlangeGenerator
from features.gusset import GussetGenerator
from features.notch_relief import NotchReliefGenerator
from features.obround_hole import ObroundHoleGenerator
from features.pad import PadGenerator
from features.polygon_hole import PolygonHoleGenerator
from features.polyhedron_boss import PolyhedronBossGenerator
from features.rect_pocket import RectPocketGenerator
from features.rib import RibGenerator
from features.slot_rect import SlotRectGenerator
from features.slot_u import USlotGenerator
from features.stepped_hole import SteppedHoleGenerator
from features.support_pad import SupportPadGenerator
from features.swept_channel_slot import SweptSlotGenerator
from features.through_hole import ThroughHoleGenerator
from features.web import WebGenerator
from utils.derived_labels import annotate_derived_cae_labels
from utils.feature_geometry import attach_feature_parameters, build_feature_parameter_lookup
from utils.generation_context import build_generation_context, cap_requested_instances, order_generator_names
from utils.ontology import export_layer_label_map, load_layer_index, load_ontology_maps
from utils.shape_quality import inspect_generation_candidate
from utils.step_solid_validation import inspect_step_file, write_step_with_positive_volume
from utils.stock_factory import build_stock_shape, sample_stock_spec
from utils.topology_rule_audit import audit_graph_directory
from utils.training_export import export_training_package


GENERATOR_CLASSES = {
    "SteppedHoleGenerator": SteppedHoleGenerator,
    "ThroughHoleGenerator": ThroughHoleGenerator,
    "BlindHoleGenerator": BlindHoleGenerator,
    "BoltHolePatternGenerator": BoltHolePatternGenerator,
    "CounterboreHoleGenerator": CounterboreHoleGenerator,
    "CountersinkHoleGenerator": CountersinkHoleGenerator,
    "ObroundHoleGenerator": ObroundHoleGenerator,
    "PolygonHoleGenerator": PolygonHoleGenerator,
    "USlotGenerator": USlotGenerator,
    "BossFeatureGenerator": BossFeatureGenerator,
    "PolyhedronBossGenerator": PolyhedronBossGenerator,
    "RectPocketGenerator": RectPocketGenerator,
    "CircularPocketGenerator": CircularPocketGenerator,
    "SlotRectGenerator": SlotRectGenerator,
    "SweptSlotGenerator": SweptSlotGenerator,
    "NotchReliefGenerator": NotchReliefGenerator,
    "RibGenerator": RibGenerator,
    "WebGenerator": WebGenerator,
    "FlangeGenerator": FlangeGenerator,
    "GussetGenerator": GussetGenerator,
    "PadGenerator": PadGenerator,
    "BearingSeatGenerator": BearingSeatGenerator,
    "SupportPadGenerator": SupportPadGenerator,
    "FilletGenerator": FilletGenerator,
    "ChamferGenerator": ChamferGenerator,
}

STATIC_STRUCTURAL_V1_GENERATORS = [
    "SteppedHoleGenerator",
    "ThroughHoleGenerator",
    "BlindHoleGenerator",
    "CounterboreHoleGenerator",
    "CountersinkHoleGenerator",
    "ObroundHoleGenerator",
    "SweptSlotGenerator",
    "SlotRectGenerator",
    "RectPocketGenerator",
    "CircularPocketGenerator",
    "BossFeatureGenerator",
    "PadGenerator",
    "RibGenerator",
    "WebGenerator",
    "FilletGenerator",
    "ChamferGenerator",
]

FEATURE_CATALOG_TARGETS = [
    {"macro_key": "stepped_hole_round", "generator_name": "SteppedHoleGenerator"},
    {"macro_key": "through_hole_round", "generator_name": "ThroughHoleGenerator"},
    {"macro_key": "blind_hole_round", "generator_name": "BlindHoleGenerator"},
    {"macro_key": "counterbore_hole", "generator_name": "CounterboreHoleGenerator"},
    {"macro_key": "countersink_hole", "generator_name": "CountersinkHoleGenerator"},
    {"macro_key": "obround_hole", "generator_name": "ObroundHoleGenerator"},
    {"macro_key": "slot_u", "generator_name": "SweptSlotGenerator"},
    {"macro_key": "slot_v", "generator_name": "SweptSlotGenerator"},
    {"macro_key": "slot_dovetail", "generator_name": "SweptSlotGenerator"},
    {"macro_key": "slot_o", "generator_name": "SweptSlotGenerator"},
    {"macro_key": "slot_rect", "generator_name": "SlotRectGenerator"},
    {"macro_key": "pocket_rect", "generator_name": "RectPocketGenerator"},
    {"macro_key": "pocket_circular", "generator_name": "CircularPocketGenerator"},
    {"macro_key": "cylinder_boss", "generator_name": "BossFeatureGenerator"},
    {"macro_key": "rectangular_boss", "generator_name": "PadGenerator"},
    {"macro_key": "rib", "generator_name": "RibGenerator"},
    {"macro_key": "web", "generator_name": "WebGenerator"},
    {"macro_key": "fillet", "generator_name": "FilletGenerator"},
    {"macro_key": "chamfer", "generator_name": "ChamferGenerator"},
]
CATALOG_GENERATOR_BY_MACRO = {spec["macro_key"]: spec["generator_name"] for spec in FEATURE_CATALOG_TARGETS}

DEFAULT_CONFIG = {
    "dataset": {
        "num_samples": 200,
        "random_seed": None,
        "stock_size_range": {
            "L": [100.0, 150.0],
            "W": [100.0, 150.0],
            "H": [40.0, 60.0],
        },
        "output_dir": None,
    },
    "generation": {
        "enabled_generators": list(STATIC_STRUCTURAL_V1_GENERATORS),
        "instance_count_range": [0, 2],
        "require_nonempty_sample": True,
        "max_generation_attempts": 10,
        "stock_profiles": {
            "enabled": True,
            "weights": {
                "block": 0.22,
                "plate": 0.20,
                "beam": 0.20,
                "cylinder": 0.18,
                "composite": 0.20,
            },
            "composite_component_count_range": [2, 3],
        },
        "feature_ordering": {
            "enabled": True,
        },
        "feature_orientation": {
            "hole_axes": ["z", "x", "y"],
        },
        "feature_density": {
            "enabled": True,
            "max_total_instances_per_sample": 8,
            "max_instances_per_generator": 1,
        },
        "coverage_control": {
            "enabled": True,
            "min_instances_per_macro": 40,
            "target_macro_features": [],
            "max_extra_samples": 400,
            "max_attempts_per_target": 80,
        },
        "topology_rule_audit": {
            "enabled": True,
        },
    },
    "exports": {
        "write_step": True,
        "write_graph": True,
        "build_training_package": True,
        "split_ratios": {
            "train": 0.8,
            "val": 0.1,
            "test": 0.1,
        },
        "llm_round_ndigits": 4,
        "gnn_round_ndigits": 6,
    },
    "feature_catalog": {
        "enabled": False,
        "only": False,
        "output_subdir": "feature_catalog",
        "variants_per_feature": 1,
        "max_attempts_per_feature": 80,
        "write_step": True,
        "write_graph": True,
    },
}


def parse_args(default_config_path):
    parser = argparse.ArgumentParser(description="Build a CAE feature recognition dataset and export training-ready corpora.")
    parser.add_argument("--config", default=default_config_path)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skip-training-export", action="store_true")
    parser.add_argument("--build-feature-catalog", action="store_true")
    parser.add_argument("--feature-catalog-only", action="store_true")
    parser.add_argument("--feature-catalog-variants", type=int, default=None)
    return parser.parse_args()


def read_seed_from_env():
    seed_value = os.environ.get("DATASET_SEED")
    if seed_value is None or seed_value == "":
        return None
    return int(seed_value)


def deep_merge(base, override):
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config_file(config_path):
    if not config_path or not os.path.exists(config_path):
        return {}

    if config_path.endswith(".json"):
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("Loading YAML config requires PyYAML in the active environment.") from exc

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def normalize_split_ratios(split_ratios):
    total = sum(float(v) for v in split_ratios.values())
    if total <= 0:
        raise ValueError("Split ratios must sum to a positive value.")
    return {key: float(value) / total for key, value in split_ratios.items()}


def build_runtime_config(args, current_dir):
    config = deepcopy(DEFAULT_CONFIG)
    config = deep_merge(config, load_config_file(args.config))

    env_seed = read_seed_from_env()
    if args.num_samples is not None:
        config["dataset"]["num_samples"] = args.num_samples
    if args.output_dir is not None:
        config["dataset"]["output_dir"] = args.output_dir
    if args.seed is not None:
        config["dataset"]["random_seed"] = args.seed
    elif env_seed is not None:
        config["dataset"]["random_seed"] = env_seed

    if config["dataset"]["output_dir"] is None:
        config["dataset"]["output_dir"] = os.path.join(current_dir, "output")

    if args.skip_training_export:
        config["exports"]["build_training_package"] = False
    if args.build_feature_catalog:
        config["feature_catalog"]["enabled"] = True
    if args.feature_catalog_only:
        config["feature_catalog"]["enabled"] = True
        config["feature_catalog"]["only"] = True
        config["exports"]["build_training_package"] = False
    if args.feature_catalog_variants is not None:
        config["feature_catalog"]["variants_per_feature"] = args.feature_catalog_variants

    config["exports"]["split_ratios"] = normalize_split_ratios(config["exports"]["split_ratios"])
    return config


def instantiate_generators(generator_names, semantic_subtype_index, generation_context=None):
    generation_context = generation_context or build_generation_context({})
    ordered_names = order_generator_names(
        generator_names,
        enabled=bool(generation_context.get("feature_ordering", {}).get("enabled", True)),
    )
    generators = []
    for name in ordered_names:
        generator_cls = GENERATOR_CLASSES.get(name)
        if generator_cls is None:
            raise KeyError(f"Unknown generator class '{name}'.")
        generator = generator_cls(semantic_subtype_index)
        generator.set_generation_context(generation_context)
        generators.append(generator)
    return generators


def generate_annotated_sample(
    L,
    W,
    H,
    ontology_maps,
    primitive_surface_index,
    semantic_subtype_index,
    idealization_tag_index,
    primitive_label_map,
    semantic_label_map,
    idealization_label_map,
    generator_names,
    instance_count_range,
    require_nonempty_sample,
    max_generation_attempts,
    generation_context=None,
):
    generation_context = generation_context or build_generation_context({})
    min_instances, max_instances = instance_count_range
    if min_instances > max_instances:
        raise ValueError("instance_count_range must satisfy min <= max.")

    rejection_summaries = []
    last_result = None
    for attempt_index in range(max_generation_attempts):
        stock_spec = sample_stock_spec(L, W, H, generation_context["stock_profiles"])
        stock = build_stock_shape(stock_spec)
        workspace = stock_spec["feature_workspace"]
        feature_L = workspace["L"]
        feature_W = workspace["W"]
        feature_H = workspace["H"]
        current_inst_id = 1
        occupied_zones = []
        feature_registry = []
        sample_generation_context = deepcopy(generation_context)
        sample_generation_context["current_stock_profile"] = stock_spec
        generators = instantiate_generators(generator_names, semantic_subtype_index, sample_generation_context)
        feature_density = generation_context.get("feature_density", {})
        density_enabled = bool(feature_density.get("enabled", True))
        total_feature_budget = (
            int(feature_density.get("max_total_instances_per_sample", max_instances * max(len(generators), 1)))
            if density_enabled
            else max_instances * max(len(generators), 1)
        )
        max_instances_per_generator = (
            int(feature_density.get("max_instances_per_generator", max_instances))
            if density_enabled
            else None
        )
        remaining_feature_budget = total_feature_budget

        for gen in generators:
            if remaining_feature_budget <= 0:
                break
            requested_instances = random.randint(min_instances, max_instances)
            requested_instances = cap_requested_instances(
                requested_instances,
                remaining_budget=remaining_feature_budget,
                max_instances_per_generator=max_instances_per_generator,
            )
            if requested_instances <= 0:
                continue
            stock, added, occupied_zones, params = gen.apply_feature(
                stock,
                feature_L,
                feature_W,
                feature_H,
                requested_instances,
                current_inst_id,
                occupied_zones,
            )
            if added > 0:
                feature_registry.append((gen, params))
                current_inst_id += added
                remaining_feature_budget -= added

        feature_count = current_inst_id - 1
        quality_report = inspect_generation_candidate(
            stock,
            feature_count=feature_count,
            require_nonempty_sample=require_nonempty_sample,
        )
        if not quality_report["acceptable"]:
            rejection_summaries.append({
                "attempt": attempt_index + 1,
                "issues": quality_report["issues"],
                "feature_count": quality_report["feature_count"],
                "solid_count": quality_report["solid_count"],
                "signed_volume": quality_report["signed_volume"],
            })
            continue

        last_result = (
            stock,
            feature_registry,
            current_inst_id,
            stock_spec,
            feature_H,
            quality_report,
            attempt_index + 1,
            rejection_summaries,
        )
        break

    if last_result is None:
        raise RuntimeError(
            "Failed to generate an acceptable sample. "
            f"attempts={max_generation_attempts}, rejections={rejection_summaries[-5:]}"
        )

    (
        stock,
        feature_registry,
        current_inst_id,
        stock_spec,
        feature_H,
        quality_report,
        accepted_attempt,
        rejection_summaries,
    ) = last_result

    from OCC.Extend.TopologyUtils import TopologyExplorer
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.GeomAbs import GeomAbs_Cylinder

    final_face_infos = {}
    for face in TopologyExplorer(stock).faces():
        surf = BRepAdaptor_Surface(face)
        u = (surf.FirstUParameter() + surf.LastUParameter()) / 2.0
        v = (surf.FirstVParameter() + surf.LastVParameter()) / 2.0
        try:
            pnt = surf.Value(u, v)
        except Exception:
            continue

        face_cx, face_cy = pnt.X(), pnt.Y()
        if surf.GetType() == GeomAbs_Cylinder:
            try:
                face_cx, face_cy = surf.Cylinder().Location().X(), surf.Cylinder().Location().Y()
            except Exception:
                pass

        matched_res = None
        for gen, params in feature_registry:
            matched_res = gen.match_face(surf, pnt, face_cx, face_cy, feature_H, params)
            if matched_res is not None:
                break

        if matched_res:
            macro_resolution = ontology_maps.resolve_macro_feature(matched_res[2])
            final_face_infos[face.__hash__()] = {
                "semantic_label": matched_res[0],
                "instance_id": matched_res[1],
                "macro_feature": macro_resolution.key,
                "macro_feature_id": macro_resolution.id,
                "raw_macro_feature": macro_resolution.raw_key,
            }
        else:
            final_face_infos[face.__hash__()] = {
                "semantic_label": 0,
                "instance_id": 0,
                "macro_feature": "stock",
                "macro_feature_id": ontology_maps.stock_id,
                "raw_macro_feature": "stock",
            }

    graph_data = extract_aag_graph(stock, final_face_infos, primitive_surface_index=primitive_surface_index)
    graph_data["stock_profile"] = stock_spec
    graph_data["generation_quality"] = {
        "accepted_attempt": accepted_attempt,
        "rejected_attempt_count": len(rejection_summaries),
        "recent_rejections": rejection_summaries[-5:],
        "accepted_shape": quality_report,
    }
    graph_data["generation_context"] = {
        "feature_ordering": generation_context["feature_ordering"],
        "feature_orientation": generation_context["feature_orientation"],
        "feature_density": generation_context["feature_density"],
    }
    feature_parameter_lookup = build_feature_parameter_lookup(feature_registry, ontology_maps)
    attach_feature_parameters(graph_data, feature_parameter_lookup)
    graph_data["label_maps"] = ontology_maps.export_label_map()
    graph_data["label_maps"]["primitive_surface"] = primitive_label_map["primitive_surface"]
    graph_data["label_maps"]["face_semantic_subtype"] = semantic_label_map["face_semantic_subtype"]
    graph_data["label_maps"]["idealization_tag"] = idealization_label_map["idealization_tag"]
    annotate_derived_cae_labels(
        graph_data=graph_data,
        ontology_maps=ontology_maps,
        idealization_tag_index=idealization_tag_index,
    )

    return stock, graph_data, current_inst_id - 1


def write_sample_outputs(stock, graph_data, output_step, output_json, write_step=True, write_graph=True):
    if write_step:
        write_step_with_positive_volume(stock, output_step)

    if write_graph:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, indent=4, ensure_ascii=False)


def step_write_report_is_acceptable(report):
    return report.get("readable") and len(report.get("issues", [])) == 0


def build_single_dataset_sample(
    L,
    W,
    H,
    output_step,
    output_json,
    ontology_maps,
    primitive_surface_index,
    semantic_subtype_index,
    idealization_tag_index,
    primitive_label_map,
    semantic_label_map,
    idealization_label_map,
    generator_names,
    instance_count_range,
    require_nonempty_sample,
    max_generation_attempts,
    write_step=True,
    write_graph=True,
    generation_context=None,
):
    step_write_rejections = []
    for write_attempt_index in range(max_generation_attempts):
        stock, graph_data, feature_count = generate_annotated_sample(
            L=L,
            W=W,
            H=H,
            ontology_maps=ontology_maps,
            primitive_surface_index=primitive_surface_index,
            semantic_subtype_index=semantic_subtype_index,
            idealization_tag_index=idealization_tag_index,
            primitive_label_map=primitive_label_map,
            semantic_label_map=semantic_label_map,
            idealization_label_map=idealization_label_map,
            generator_names=generator_names,
            instance_count_range=instance_count_range,
            require_nonempty_sample=require_nonempty_sample,
            max_generation_attempts=max_generation_attempts,
            generation_context=generation_context,
        )

        step_report = None
        if write_step:
            write_step_with_positive_volume(stock, output_step)
            step_report = inspect_step_file(output_step)
            if not step_write_report_is_acceptable(step_report):
                step_write_rejections.append({
                    "attempt": write_attempt_index + 1,
                    "issues": step_report.get("issues", []),
                    "solid_count": step_report.get("solid_count"),
                    "signed_volume": step_report.get("signed_volume"),
                    "valid": step_report.get("valid"),
                })
                continue

        graph_data.setdefault("generation_quality", {})
        graph_data["generation_quality"]["step_write_validation"] = step_report
        graph_data["generation_quality"]["step_write_rejected_attempt_count"] = len(step_write_rejections)
        graph_data["generation_quality"]["recent_step_write_rejections"] = step_write_rejections[-5:]

        if write_graph:
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(graph_data, f, indent=4, ensure_ascii=False)

        return graph_data

    raise RuntimeError(
        "Failed to write an acceptable STEP sample. "
        f"attempts={max_generation_attempts}, rejections={step_write_rejections[-5:]}"
    )

    print(f"生成样本完成: features={feature_count} -> {output_json}")
def write_dataset_meta(
    output_dir,
    config_path,
    ontology_path,
    semantic_subtype_ontology_path,
    primitive_label_map_path,
    macro_label_map_path,
    semantic_label_map_path,
    idealization_tag_label_map_path,
    runtime_config,
    actual_samples_written,
    coverage_summary,
):
    meta = {
        "dataset_name": "cae_feature_dataset",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": output_dir,
        "config_path": config_path,
        "num_samples_requested": runtime_config["dataset"]["num_samples"],
        "actual_samples_written": actual_samples_written,
        "random_seed": runtime_config["dataset"]["random_seed"],
        "stock_size_range": runtime_config["dataset"]["stock_size_range"],
        "enabled_generators": runtime_config["generation"]["enabled_generators"],
        "generator_classes": runtime_config["generation"]["enabled_generators"],
        "instance_count_range": runtime_config["generation"]["instance_count_range"],
        "require_nonempty_sample": runtime_config["generation"]["require_nonempty_sample"],
        "coverage_control": runtime_config["generation"]["coverage_control"],
        "stock_profiles": runtime_config["generation"].get("stock_profiles", {}),
        "feature_ordering": runtime_config["generation"].get("feature_ordering", {}),
        "feature_orientation": runtime_config["generation"].get("feature_orientation", {}),
        "coverage_summary": coverage_summary,
        "macro_feature_ontology_path": ontology_path,
        "semantic_subtype_ontology_path": semantic_subtype_ontology_path,
        "primitive_surface_label_map_path": primitive_label_map_path,
        "macro_feature_label_map_path": macro_label_map_path,
        "semantic_subtype_label_map_path": semantic_label_map_path,
        "idealization_tag_label_map_path": idealization_tag_label_map_path,
        "semantic_subtype_connected": True,
        "derived_cae_labels_connected": True,
        "graph_node_fields": [
            "idx",
            "primitive_surface",
            "primitive_surface_id",
            "semantic_label",
            "instance_id",
            "macro_feature",
            "macro_feature_id",
            "derived_macro_feature_ids",
            "derived_macro_features",
            "idealization_tag_ids",
            "idealization_tags",
            "attributes",
        ],
        "feature_set_fields": [
            "instance_id",
            "macro_feature",
            "macro_feature_id",
            "node_indices",
            "geometry_parameters",
        ],
        "exports": runtime_config["exports"],
    }
    meta_path = os.path.join(output_dir, "dataset_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)


def count_feature_instances(graph_data):
    counter = Counter()
    for feature_set in graph_data.get("feature_sets", []):
        macro_key = feature_set.get("macro_feature")
        if macro_key and macro_key != "stock":
            counter[macro_key] += 1
    return counter


def resolve_coverage_targets(ontology_maps, coverage_cfg):
    target_macro_features = coverage_cfg.get("target_macro_features") or []
    if target_macro_features:
        valid_macro_keys = {item.key for item in ontology_maps.macro_features}
        unknown = [key for key in target_macro_features if key not in valid_macro_keys]
        if unknown:
            raise KeyError(f"Unknown coverage target macro features: {unknown}")
        return list(target_macro_features)

    return [item.key for item in ontology_maps.macro_features if item.enabled and item.key != "stock"]


def coverage_deficits(instance_counter, target_macro_features, min_instances_per_macro):
    deficits = {}
    for macro_key in target_macro_features:
        current = instance_counter.get(macro_key, 0)
        missing = max(0, min_instances_per_macro - current)
        if missing > 0:
            deficits[macro_key] = missing
    return deficits


def is_coverage_satisfied(instance_counter, target_macro_features, min_instances_per_macro):
    return not coverage_deficits(instance_counter, target_macro_features, min_instances_per_macro)


def select_next_coverage_target(instance_counter, target_macro_features, min_instances_per_macro):
    deficits = coverage_deficits(instance_counter, target_macro_features, min_instances_per_macro)
    if not deficits:
        return None
    return max(deficits.items(), key=lambda item: (item[1], -instance_counter.get(item[0], 0), item[0]))[0]


def write_coverage_report(output_dir, coverage_summary):
    reports_dir = os.path.join(output_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, "coverage_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(coverage_summary, f, indent=4, ensure_ascii=False)
    return report_path


def clear_previous_dataset_outputs(output_dir):
    steps_dir = os.path.join(output_dir, "steps")
    graphs_dir = os.path.join(output_dir, "graphs")
    os.makedirs(steps_dir, exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)

    for filename in os.listdir(steps_dir):
        if filename.startswith("composite_feature_") and filename.endswith(".step"):
            os.remove(os.path.join(steps_dir, filename))
    for filename in os.listdir(graphs_dir):
        if filename.startswith("composite_feature_") and filename.endswith(".json"):
            os.remove(os.path.join(graphs_dir, filename))

    for dirname in ("manifests", "splits", "reports", "training"):
        path = os.path.join(output_dir, dirname)
        if os.path.isdir(path):
            shutil.rmtree(path)
    for filename in ("dataset_meta.json",):
        path = os.path.join(output_dir, filename)
        if os.path.exists(path):
            os.remove(path)


def build_targeted_coverage_sample(
    sample_index,
    target_macro_key,
    output_dir,
    size_range,
    ontology_maps,
    primitive_surface_index,
    semantic_subtype_index,
    idealization_tag_index,
    primitive_label_map,
    semantic_label_map,
    idealization_label_map,
    max_generation_attempts,
    max_attempts_per_target,
    write_step=True,
    write_graph=True,
    generation_context=None,
):
    generator_name = CATALOG_GENERATOR_BY_MACRO.get(target_macro_key)
    if generator_name is None:
        raise KeyError(f"No generator mapping found for coverage target '{target_macro_key}'.")

    for attempt_index in range(max_attempts_per_target):
        stock, graph_data, feature_count = generate_annotated_sample(
            L=random.uniform(*size_range["L"]),
            W=random.uniform(*size_range["W"]),
            H=random.uniform(*size_range["H"]),
            ontology_maps=ontology_maps,
            primitive_surface_index=primitive_surface_index,
            semantic_subtype_index=semantic_subtype_index,
            idealization_tag_index=idealization_tag_index,
            primitive_label_map=primitive_label_map,
            semantic_label_map=semantic_label_map,
            idealization_label_map=idealization_label_map,
            generator_names=[generator_name],
            instance_count_range=(1, 1),
            require_nonempty_sample=True,
            max_generation_attempts=max_generation_attempts,
            generation_context=generation_context,
        )
        if feature_count != 1 or not is_valid_catalog_sample(graph_data, target_macro_key):
            continue

        output_step = os.path.join(output_dir, "steps", f"composite_feature_{sample_index:03d}.step")
        output_json = os.path.join(output_dir, "graphs", f"composite_feature_{sample_index:03d}.json")
        write_sample_outputs(
            stock=stock,
            graph_data=graph_data,
            output_step=output_step,
            output_json=output_json,
            write_step=write_step,
            write_graph=write_graph,
        )
        return graph_data, attempt_index + 1

    return None, max_attempts_per_target


def build_dataset_with_coverage_control(
    output_dir,
    runtime_config,
    ontology_maps,
    primitive_surface_index,
    semantic_subtype_index,
    idealization_tag_index,
    primitive_label_map,
    semantic_label_map,
    idealization_label_map,
):
    clear_previous_dataset_outputs(output_dir)

    size_range = runtime_config["dataset"]["stock_size_range"]
    generation_cfg = runtime_config["generation"]
    generation_context = build_generation_context(generation_cfg)
    coverage_cfg = generation_cfg["coverage_control"]
    coverage_enabled = bool(coverage_cfg.get("enabled", False))
    target_macro_features = resolve_coverage_targets(ontology_maps, coverage_cfg) if coverage_enabled else []
    min_instances_per_macro = int(coverage_cfg.get("min_instances_per_macro", 0))
    max_extra_samples = int(coverage_cfg.get("max_extra_samples", 0))
    max_attempts_per_target = int(coverage_cfg.get("max_attempts_per_target", 0))

    actual_samples_written = 0
    extra_samples_added = 0
    instance_counter = Counter()
    booster_attempts = Counter()
    booster_failures = Counter()

    for i in range(runtime_config["dataset"]["num_samples"]):
        graph_data = build_single_dataset_sample(
            random.uniform(*size_range["L"]),
            random.uniform(*size_range["W"]),
            random.uniform(*size_range["H"]),
            os.path.join(output_dir, "steps", f"composite_feature_{i:03d}.step"),
            os.path.join(output_dir, "graphs", f"composite_feature_{i:03d}.json"),
            ontology_maps=ontology_maps,
            primitive_surface_index=primitive_surface_index,
            semantic_subtype_index=semantic_subtype_index,
            idealization_tag_index=idealization_tag_index,
            primitive_label_map=primitive_label_map,
            semantic_label_map=semantic_label_map,
            idealization_label_map=idealization_label_map,
            generator_names=generation_cfg["enabled_generators"],
            instance_count_range=generation_cfg["instance_count_range"],
            require_nonempty_sample=generation_cfg["require_nonempty_sample"],
            max_generation_attempts=generation_cfg["max_generation_attempts"],
            write_step=runtime_config["exports"]["write_step"],
            write_graph=runtime_config["exports"]["write_graph"],
            generation_context=generation_context,
        )
        instance_counter.update(count_feature_instances(graph_data))
        actual_samples_written += 1

    if coverage_enabled:
        while extra_samples_added < max_extra_samples and not is_coverage_satisfied(instance_counter, target_macro_features, min_instances_per_macro):
            target_macro_key = select_next_coverage_target(instance_counter, target_macro_features, min_instances_per_macro)
            if target_macro_key is None:
                break

            graph_data, attempts_used = build_targeted_coverage_sample(
                sample_index=actual_samples_written,
                target_macro_key=target_macro_key,
                output_dir=output_dir,
                size_range=size_range,
                ontology_maps=ontology_maps,
                primitive_surface_index=primitive_surface_index,
                semantic_subtype_index=semantic_subtype_index,
                idealization_tag_index=idealization_tag_index,
                primitive_label_map=primitive_label_map,
                semantic_label_map=semantic_label_map,
                idealization_label_map=idealization_label_map,
                max_generation_attempts=generation_cfg["max_generation_attempts"],
                max_attempts_per_target=max_attempts_per_target,
                write_step=runtime_config["exports"]["write_step"],
                write_graph=runtime_config["exports"]["write_graph"],
                generation_context=generation_context,
            )
            booster_attempts[target_macro_key] += attempts_used
            if graph_data is None:
                booster_failures[target_macro_key] += 1
                break

            instance_counter.update(count_feature_instances(graph_data))
            extra_samples_added += 1
            actual_samples_written += 1

    deficits = coverage_deficits(instance_counter, target_macro_features, min_instances_per_macro) if coverage_enabled else {}
    coverage_summary = {
        "enabled": coverage_enabled,
        "count_unit": "feature_instances",
        "base_samples_requested": runtime_config["dataset"]["num_samples"],
        "base_samples_written": runtime_config["dataset"]["num_samples"],
        "extra_samples_added": extra_samples_added,
        "actual_samples_written": actual_samples_written,
        "min_instances_per_macro": min_instances_per_macro if coverage_enabled else 0,
        "target_macro_features": target_macro_features,
        "achieved_instance_counts": {macro_key: int(instance_counter.get(macro_key, 0)) for macro_key in target_macro_features},
        "deficits_remaining": deficits,
        "status": "satisfied" if coverage_enabled and not deficits else ("disabled" if not coverage_enabled else "partial"),
        "booster_attempts_by_macro": dict(sorted(booster_attempts.items())),
        "booster_failures_by_macro": dict(sorted(booster_failures.items())),
    }
    coverage_report_path = write_coverage_report(output_dir, coverage_summary)
    coverage_summary["coverage_report_path"] = coverage_report_path

    return actual_samples_written, coverage_summary


def ensure_label_maps(output_dir, primitive_label_map, macro_label_map, semantic_label_map, idealization_label_map):
    label_map_dir = os.path.join(output_dir, "label_maps")
    os.makedirs(label_map_dir, exist_ok=True)

    primitive_label_map_path = os.path.join(label_map_dir, "primitive_surface_label_map.json")
    with open(primitive_label_map_path, "w", encoding="utf-8") as f:
        json.dump(primitive_label_map, f, indent=4, ensure_ascii=False)

    macro_label_map_path = os.path.join(label_map_dir, "macro_feature_label_map.json")
    with open(macro_label_map_path, "w", encoding="utf-8") as f:
        json.dump(macro_label_map, f, indent=4, ensure_ascii=False)

    semantic_label_map_path = os.path.join(label_map_dir, "face_semantic_subtype_label_map.json")
    with open(semantic_label_map_path, "w", encoding="utf-8") as f:
        json.dump(semantic_label_map, f, indent=4, ensure_ascii=False)

    idealization_label_map_path = os.path.join(label_map_dir, "idealization_tag_label_map.json")
    with open(idealization_label_map_path, "w", encoding="utf-8") as f:
        json.dump(idealization_label_map, f, indent=4, ensure_ascii=False)

    return primitive_label_map_path, macro_label_map_path, semantic_label_map_path, idealization_label_map_path


def is_valid_catalog_sample(graph_data, target_macro_key):
    non_stock_macros = sorted({
        node["macro_feature"]
        for node in graph_data.get("nodes", [])
        if node.get("macro_feature") != "stock"
    })
    feature_sets = graph_data.get("feature_sets", [])
    if non_stock_macros != [target_macro_key]:
        return False
    if len(feature_sets) != 1:
        return False
    return feature_sets[0].get("macro_feature") == target_macro_key


def write_feature_catalog_meta(
    catalog_dir,
    runtime_config,
    ontology_path,
    semantic_subtype_ontology_path,
    primitive_label_map_path,
    macro_label_map_path,
    semantic_label_map_path,
    idealization_tag_label_map_path,
    manifest_entries,
    failures,
):
    meta = {
        "catalog_name": "cae_feature_catalog",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "catalog_root": catalog_dir,
        "random_seed": runtime_config["dataset"]["random_seed"],
        "variants_per_feature": runtime_config["feature_catalog"]["variants_per_feature"],
        "max_attempts_per_feature": runtime_config["feature_catalog"]["max_attempts_per_feature"],
        "macro_feature_ontology_path": ontology_path,
        "semantic_subtype_ontology_path": semantic_subtype_ontology_path,
        "primitive_surface_label_map_path": primitive_label_map_path,
        "macro_feature_label_map_path": macro_label_map_path,
        "semantic_subtype_label_map_path": semantic_label_map_path,
        "idealization_tag_label_map_path": idealization_tag_label_map_path,
        "feature_count_requested": len(FEATURE_CATALOG_TARGETS),
        "feature_count_generated": len(manifest_entries),
        "features": manifest_entries,
        "failures": failures,
    }
    meta_path = os.path.join(catalog_dir, "feature_catalog_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)


def build_feature_catalog(
    output_dir,
    runtime_config,
    ontology_path,
    semantic_subtype_ontology_path,
    ontology_maps,
    primitive_surface_index,
    semantic_subtype_index,
    idealization_tag_index,
    primitive_label_map,
    semantic_label_map,
    idealization_label_map,
):
    catalog_cfg = runtime_config["feature_catalog"]
    catalog_dir = os.path.join(output_dir, catalog_cfg["output_subdir"])
    os.makedirs(catalog_dir, exist_ok=True)

    primitive_label_map_path, macro_label_map_path, semantic_label_map_path, idealization_tag_label_map_path = ensure_label_maps(
        catalog_dir,
        primitive_label_map=primitive_label_map,
        macro_label_map=ontology_maps.export_label_map(),
        semantic_label_map=semantic_label_map,
        idealization_label_map=idealization_label_map,
    )

    size_range = runtime_config["dataset"]["stock_size_range"]
    variant_count = catalog_cfg["variants_per_feature"]
    max_attempts = catalog_cfg["max_attempts_per_feature"]
    manifest_entries = []
    failures = []
    generation_context = build_generation_context(runtime_config["generation"])

    for spec in FEATURE_CATALOG_TARGETS:
        target_macro_key = spec["macro_key"]
        generator_name = spec["generator_name"]
        feature_dir = os.path.join(catalog_dir, target_macro_key)
        steps_dir = os.path.join(feature_dir, "steps")
        graphs_dir = os.path.join(feature_dir, "graphs")
        os.makedirs(steps_dir, exist_ok=True)
        os.makedirs(graphs_dir, exist_ok=True)

        generated = 0
        attempts = 0
        while generated < variant_count and attempts < max_attempts:
            attempts += 1
            stock, graph_data, feature_count = generate_annotated_sample(
                L=random.uniform(*size_range["L"]),
                W=random.uniform(*size_range["W"]),
                H=random.uniform(*size_range["H"]),
                ontology_maps=ontology_maps,
                primitive_surface_index=primitive_surface_index,
                semantic_subtype_index=semantic_subtype_index,
                idealization_tag_index=idealization_tag_index,
                primitive_label_map=primitive_label_map,
                semantic_label_map=semantic_label_map,
                idealization_label_map=idealization_label_map,
                generator_names=[generator_name],
                instance_count_range=(1, 1),
                require_nonempty_sample=True,
                max_generation_attempts=runtime_config["generation"]["max_generation_attempts"],
                generation_context=generation_context,
            )

            if feature_count != 1 or not is_valid_catalog_sample(graph_data, target_macro_key):
                continue

            output_step = os.path.join(steps_dir, f"{target_macro_key}_{generated:03d}.step")
            output_json = os.path.join(graphs_dir, f"{target_macro_key}_{generated:03d}.json")
            write_sample_outputs(
                stock=stock,
                graph_data=graph_data,
                output_step=output_step,
                output_json=output_json,
                write_step=catalog_cfg["write_step"],
                write_graph=catalog_cfg["write_graph"],
            )
            manifest_entries.append({
                "macro_feature": target_macro_key,
                "generator_name": generator_name,
                "variant_index": generated,
                "attempts_used": attempts,
                "step_path": output_step,
                "graph_path": output_json,
                "node_count": len(graph_data.get("nodes", [])),
                "feature_set_count": len(graph_data.get("feature_sets", [])),
            })
            print(f"单特征样本完成: macro_feature={target_macro_key} -> {output_step}")
            generated += 1
            attempts = 0

        if generated < variant_count:
            failures.append({
                "macro_feature": target_macro_key,
                "generator_name": generator_name,
                "generated_variants": generated,
                "requested_variants": variant_count,
                "max_attempts_per_feature": max_attempts,
            })

    write_feature_catalog_meta(
        catalog_dir=catalog_dir,
        runtime_config=runtime_config,
        ontology_path=ontology_path,
        semantic_subtype_ontology_path=semantic_subtype_ontology_path,
        primitive_label_map_path=primitive_label_map_path,
        macro_label_map_path=macro_label_map_path,
        semantic_label_map_path=semantic_label_map_path,
        idealization_tag_label_map_path=idealization_tag_label_map_path,
        manifest_entries=manifest_entries,
        failures=failures,
    )


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    default_config_path = os.path.join(current_dir, "configs", "default_dataset_config.yaml")
    args = parse_args(default_config_path)
    runtime_config = build_runtime_config(args, current_dir)

    ontology_path = os.path.join(current_dir, "CAE_feature_ontology.yaml")
    semantic_subtype_ontology_path = os.path.join(current_dir, "face_semantic_subtype_ontology.yaml")

    ontology_maps = load_ontology_maps(ontology_path)
    primitive_surface_index = load_layer_index(ontology_path, "primitive_surface")
    idealization_tag_index = load_layer_index(ontology_path, "idealization_tag")
    semantic_subtype_index = load_layer_index(semantic_subtype_ontology_path, "face_semantic_subtype")
    primitive_label_map = export_layer_label_map(primitive_surface_index)
    idealization_label_map = export_layer_label_map(idealization_tag_index)
    semantic_label_map = export_layer_label_map(semantic_subtype_index)

    random_seed = runtime_config["dataset"]["random_seed"]
    if random_seed is not None:
        random.seed(random_seed)

    output_dir = runtime_config["dataset"]["output_dir"]
    if not runtime_config["feature_catalog"]["only"]:
        (
            primitive_label_map_path,
            macro_label_map_path,
            semantic_label_map_path,
            idealization_tag_label_map_path,
        ) = ensure_label_maps(
            output_dir,
            primitive_label_map=primitive_label_map,
            macro_label_map=ontology_maps.export_label_map(),
            semantic_label_map=semantic_label_map,
            idealization_label_map=idealization_label_map,
        )

        actual_samples_written, coverage_summary = build_dataset_with_coverage_control(
            output_dir=output_dir,
            runtime_config=runtime_config,
            ontology_maps=ontology_maps,
            primitive_surface_index=primitive_surface_index,
            semantic_subtype_index=semantic_subtype_index,
            idealization_tag_index=idealization_tag_index,
            primitive_label_map=primitive_label_map,
            semantic_label_map=semantic_label_map,
            idealization_label_map=idealization_label_map,
        )

        write_dataset_meta(
            output_dir=output_dir,
            config_path=args.config,
            ontology_path=ontology_path,
            semantic_subtype_ontology_path=semantic_subtype_ontology_path,
            primitive_label_map_path=primitive_label_map_path,
            macro_label_map_path=macro_label_map_path,
            semantic_label_map_path=semantic_label_map_path,
            idealization_tag_label_map_path=idealization_tag_label_map_path,
            runtime_config=runtime_config,
            actual_samples_written=actual_samples_written,
            coverage_summary=coverage_summary,
        )

        topology_audit_cfg = runtime_config["generation"].get("topology_rule_audit", {})
        if topology_audit_cfg.get("enabled", True):
            audit_path = os.path.join(output_dir, "reports", "topology_rule_audit.json")
            audit_graph_directory(Path(output_dir) / "graphs", Path(audit_path))

        if runtime_config["exports"]["build_training_package"]:
            export_training_package(
                output_dir=output_dir,
                split_ratios=runtime_config["exports"]["split_ratios"],
                seed=random_seed,
                llm_round_ndigits=runtime_config["exports"]["llm_round_ndigits"],
                gnn_round_ndigits=runtime_config["exports"]["gnn_round_ndigits"],
            )

    if runtime_config["feature_catalog"]["enabled"]:
        build_feature_catalog(
            output_dir=output_dir,
            runtime_config=runtime_config,
            ontology_path=ontology_path,
            semantic_subtype_ontology_path=semantic_subtype_ontology_path,
            ontology_maps=ontology_maps,
            primitive_surface_index=primitive_surface_index,
            semantic_subtype_index=semantic_subtype_index,
            idealization_tag_index=idealization_tag_index,
            primitive_label_map=primitive_label_map,
            semantic_label_map=semantic_label_map,
            idealization_label_map=idealization_label_map,
        )


if __name__ == "__main__":
    main()
