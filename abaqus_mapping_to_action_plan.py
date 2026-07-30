from __future__ import annotations

import argparse
import json
from pathlib import Path


STATUS_RANK = {
    "low_confidence": 0,
    "medium_confidence": 1,
    "high_confidence": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Abaqus face mapping output into a step5_script_gen.py-compatible action plan."
    )
    parser.add_argument("--mapping-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--simulation-name", default="AI_Job_FromMapping")
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
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def confidence_allows(status: str, minimum: str) -> bool:
    return STATUS_RANK.get(status, -1) >= STATUS_RANK[minimum]


def region_catalog(mapping_data: dict) -> tuple[list[dict], list[dict], dict[str, dict]]:
    suggested = mapping_data.get("suggested_abaqus_regions", {})
    feature_regions = suggested.get("feature_object_regions", [])
    stock_regions = suggested.get("stock_planar_regions", [])
    by_name = {}
    for region in feature_regions + stock_regions:
        by_name[region["region_name"]] = region
    return feature_regions, stock_regions, by_name


def build_mesh_refinement_faces(feature_regions: list[dict], min_confidence: str) -> list[int]:
    face_ids = set()
    for region in feature_regions:
        if not confidence_allows(region.get("mapping_status", "low_confidence"), min_confidence):
            continue
        for face_id in region.get("abaqus_face_ids", []):
            face_ids.add(int(face_id))
    return sorted(face_ids)


def select_bc_region(
    stock_regions: list[dict],
    region_by_name: dict[str, dict],
    args: argparse.Namespace,
) -> dict | None:
    if args.bc_mode == "none":
        return None
    if args.bc_mode == "largest_stock_planar":
        return stock_regions[0] if stock_regions else None
    if args.bc_mode == "region_name" and args.bc_region_name:
        return region_by_name.get(args.bc_region_name)
    return None


def select_load_region(
    feature_regions: list[dict],
    region_by_name: dict[str, dict],
    args: argparse.Namespace,
) -> dict | None:
    if args.load_mode == "none":
        return None
    if args.load_mode == "first_feature_region":
        for region in feature_regions:
            if confidence_allows(region.get("mapping_status", "low_confidence"), args.min_region_confidence):
                return region
        return None
    if args.load_mode == "region_name" and args.load_region_name:
        return region_by_name.get(args.load_region_name)
    if args.load_mode == "macro_feature" and args.load_macro_feature:
        for region in feature_regions:
            if region.get("macro_feature") != args.load_macro_feature:
                continue
            if confidence_allows(region.get("mapping_status", "low_confidence"), args.min_region_confidence):
                return region
    return None


def build_reasoning(
    refinement_faces: list[int],
    bc_region: dict | None,
    load_region: dict | None,
    args: argparse.Namespace,
) -> str:
    parts = [
        f"Assign {args.material_name} to the whole part.",
        f"Use a global mesh size of {args.global_size}.",
    ]
    if refinement_faces:
        parts.append(
            f"Refine all mapped feature faces meeting {args.min_region_confidence} confidence with local size {args.local_size}."
        )
    if bc_region:
        parts.append(f"Apply a displacement boundary condition to region {bc_region['region_name']}.")
    if load_region:
        parts.append(
            f"Apply a {args.load_type} load of {args.load_value} to region {load_region['region_name']}."
        )
    return " ".join(parts)


def build_action_plan(mapping_data: dict, args: argparse.Namespace) -> dict:
    feature_regions, stock_regions, region_by_name = region_catalog(mapping_data)

    refinement_faces = build_mesh_refinement_faces(feature_regions, args.min_region_confidence)
    bc_region = select_bc_region(stock_regions, region_by_name, args)
    load_region = select_load_region(feature_regions, region_by_name, args)

    operations = [
        {
            "type": "Material",
            "material_name": args.material_name,
            "props": {
                "E": args.elastic_modulus,
                "nu": args.poisson_ratio,
            },
        },
        {
            "type": "Mesh",
            "global_size": args.global_size,
        },
    ]

    if refinement_faces:
        operations.append(
            {
                "type": "MeshRefinement",
                "faces": refinement_faces,
                "local_size": args.local_size,
            }
        )

    if bc_region:
        operations.append(
            {
                "type": "BoundaryCondition",
                "faces": list(bc_region.get("abaqus_face_ids", [])),
                "constraints": {
                    "U1": args.bc_u1,
                    "U2": args.bc_u2,
                    "U3": args.bc_u3,
                },
            }
        )

    if load_region:
        operations.append(
            {
                "type": "Load",
                "faces": list(load_region.get("abaqus_face_ids", [])),
                "load_type": args.load_type,
                "value": args.load_value,
            }
        )

    return {
        "reasoning": build_reasoning(refinement_faces, bc_region, load_region, args),
        "simulation_name": args.simulation_name,
        "model_name": args.model_name,
        "part_name": args.part_name,
        "instance_name": args.instance_name,
        "source_mapping_path": str(args.mapping_path.resolve()),
        "available_regions": {
            "feature_object_regions": feature_regions,
            "stock_planar_regions": stock_regions,
        },
        "selection_trace": {
            "mesh_refinement_face_count": len(refinement_faces),
            "bc_region_name": None if bc_region is None else bc_region["region_name"],
            "load_region_name": None if load_region is None else load_region["region_name"],
        },
        "operations": operations,
    }


def main() -> None:
    args = parse_args()
    mapping_data = load_json(args.mapping_path)
    action_plan = build_action_plan(mapping_data, args)
    write_json(args.output_path, action_plan)
    print(f"action plan written: {args.output_path}")


if __name__ == "__main__":
    main()
