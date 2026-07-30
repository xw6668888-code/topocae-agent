from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "output_generation_p1_b_corrected_balanced_5000_v3"
SOURCE_MANIFEST = SOURCE_ROOT / "manifests" / "sample_manifest.jsonl"
SOURCE_ARTIFACT_MANIFEST = SOURCE_ROOT / "manifests" / "artifact_manifest.json"
SOURCE_META = SOURCE_ROOT / "dataset_meta.json"
SOURCE_SPLITS = {
    "train": SOURCE_ROOT / "splits" / "train.jsonl",
    "val": SOURCE_ROOT / "splits" / "val.jsonl",
    "development_test": SOURCE_ROOT / "splits" / "development_test.jsonl",
}
DATASET_SPLIT_MAP = {"train": "train", "val": "val", "development_test": "test"}

P2_FREEZE = ROOT / "reports" / "p2_a1_cae_ir_v1" / "freeze"
P2_FREEZE_MANIFEST = P2_FREEZE / "freeze_manifest.json"
P2_POLICY = P2_FREEZE / "configs" / "cae_feature_role_policy.yaml"
P2_ALIASES = P2_FREEZE / "configs" / "cae_feature_aliases.yaml"
P2_RESULT_SCHEMA = P2_FREEZE / "schemas" / "cae_intent_result.schema.json"
P2_INTENT_SCHEMA = P2_FREEZE / "schemas" / "cae_intent.schema.json"
P2_GROUNDING_SCHEMA = P2_FREEZE / "schemas" / "cae_grounding_context.schema.json"
P2_RUNNER = ROOT / "utils" / "run_p2_a1_cae_ir_tests.py"
P2_RUNNER_EXPECTED_SHA256 = "577beccbfb2d1503401b82aaaf0af6f7bd01e85b9fe535e826887e3a62ef99ba"

CROSSWALK_PATH = ROOT / "configs" / "paper_v1" / "cae_query_role_crosswalk.yaml"
REPORT_ROOT = ROOT / "reports" / "p3_a1_cae_queryground_v1"
SOURCE_AUDIT_JSON = REPORT_ROOT / "source_and_role_audit.json"
SOURCE_AUDIT_MD = REPORT_ROOT / "source_and_role_audit.zh-CN.md"
PILOT_ROOT = ROOT / "output_cae_query_ground_v1" / "pilot"

CAE_IR_VERSION = "1.0.0-h2.01"
RECORD_VERSION = "cae-query-ground-record/1.0.0-h3.01-pilot"
DATASET_VERSION = "cae-query-ground/1.0.0-h3.01-pilot"
GRAPH_SCHEMA = "topocae_graph_v2_relative_1_1"
SEED = 20260721

CANONICAL_FEATURES = (
    "through_hole_round",
    "blind_hole_round",
    "stepped_hole_round",
    "counterbore_hole",
    "countersink_hole",
    "obround_hole",
    "slot_rect",
    "slot_u",
    "slot_v",
    "slot_o",
    "pocket_rect",
    "pocket_circular",
    "cylinder_boss",
    "rectangular_boss",
    "rib",
    "fillet",
    "chamfer",
)
OPERATIONS = (
    "fixed_support",
    "displacement",
    "pressure",
    "surface_traction",
    "material_assignment",
    "global_mesh_size",
    "local_mesh_size",
)
FEATURE_OPERATIONS = frozenset(
    {"fixed_support", "displacement", "pressure", "surface_traction", "local_mesh_size"}
)
WHOLE_PART_OPERATIONS = frozenset({"material_assignment", "global_mesh_size"})
LEVELS = ("L1", "L2", "L3", "L4", "L5")
STATUSES = ("VALID", "NEEDS_CLARIFICATION", "INVALID")

# Synthetic P1 geometry uses millimetre-valued OCCT model coordinates. STEP inputs
# additionally carry their native unit declaration; BREP text is unitless and therefore
# relies on this explicit dataset convention. The planner records this provenance.
MODEL_LENGTH_UNIT = "mm"
MODEL_AREA_UNIT = "mm2"
UNIT_CONTRACT = "p1_b_synthetic_occt_model_coordinates_mm"

SOURCE_MACRO_TO_CANONICAL = {
    **{feature: feature for feature in CANONICAL_FEATURES},
    "web": "rib",
}


def assert_within(path: Path, parent: Path) -> Path:
    resolved = path.resolve()
    parent_resolved = parent.resolve()
    if resolved != parent_resolved and parent_resolved not in resolved.parents:
        raise ValueError(f"Path escapes allowed root: {resolved}")
    return resolved


def assert_source_path(path: str | Path) -> Path:
    return assert_within(Path(path), SOURCE_ROOT)
