# v5.17 Role-Aware Candidate Feature Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add rotation-invariant, role-aware B-Rep candidate features so the candidate classifier can distinguish valid feature objects from same-source false positives, especially `through_hole_round` inner walls vs outer cylinders, transition faces, rectangular bosses, and other hybrid proposals.

**Architecture:** Keep STEP generation, graph extraction, GNN inference, hybrid proposal generation, and topology postprocess unchanged. Extend only candidate-level feature extraction, rebuild candidate datasets with and without v5.16 rebalancing, retrain candidate classifiers, and compare v5.13/v5.14/v5.16/v5.17 on the same S3 validation protocol.

**Tech Stack:** Python 3 from `W:/python/envs/feagpt/python.exe`, `unittest`, JSON/JSONL, existing scikit-learn RandomForest training in `train_candidate_topology_classifier.py`, existing `object_proposal_recall_report.py`, existing S3 area truth.

---

## 0. Review of v5.16 Output

v5.16 was implemented according to plan and passed the required tests:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_build_instance_candidate_dataset tests.test_train_candidate_topology_classifier tests.test_candidate_classifier_error_atlas -v
```

Observed:

```text
Ran 15 tests
OK
```

v5.16 dataset rebalance was mechanically successful:

```text
record_count: 35185
source_macro_positive_clone_count: 2549
rebalance_action_counts:
  default: 15524
  downweighted_reject: 5276
  positive_nontruth: 11816
  positive_source_macro_clone: 2549
  positive_source_macro_pair: 20
sample_weight_stats:
  min: 0.35
  max: 5.0
```

But v5.16 failed the real S3 promotion criteria:

```text
| Version | Proposal recall | Classified recall | Accepted recall | Final-survival recall | No proposal | Classifier wrong/rejected |
| v5.13   | 0.6814          | 0.5343            | 0.4804          | 0.5196                | 65          | 32                        |
| v5.14   | 0.6814          | 0.4069            | 0.3971          | 0.4020                | 65          | 56                        |
| v5.16   | 0.6814          | 0.3971            | 0.3873          | 0.3922                | 65          | 58                        |
```

Recommendation from v5.16:

```text
do_not_promote_v5_16_move_to_candidate_feature_revision
```

This recommendation is reasonable. v5.16 fixed the obvious training label bias, but real candidate classification did not improve. That means source/macro weighting is not enough; the classifier lacks discriminative object-level features.

Key observation from real graph edge attributes:

```text
adjacent_plane_loop_role: inner_loop / outer_loop / unknown
adjacent_plane_loop_role_inner: 1.0 or 0.0
adjacent_plane_loop_role_outer: 1.0 or 0.0
cylindrical_wall_role_hint: inner_hole_wall / outer_* / ambiguous / unknown
cylindrical_wall_role_inner_hint: 1.0 or 0.0
cylindrical_wall_role_outer_hint: 1.0 or 0.0
```

These role hints are already present in graph `edge_attributes`, but the candidate classifier currently mostly sees coarse counts such as `boundary_circle_count` and `primitive_cylinder_ratio`. v5.17 exposes these role hints and topology rule reasons as candidate features.

## 1. Scope Control

### Modify

- `utils/instance_candidate_features.py`
- `tests/test_instance_candidate_features.py`
- `compare_candidate_classifier_recall_reports.py`

### Use Existing Scripts Without Changing

- `build_instance_candidate_dataset.py`
- `train_candidate_topology_classifier.py`
- `object_proposal_recall_report.py`
- `candidate_classifier_error_atlas.py`

### Generate

- `output_generation_v5_17_candidate_features_plain/instance_candidate_dataset/`
- `output_generation_v5_17_candidate_features_plain/model_runs/random_forest_instance_candidate_v5_17_plain/`
- `output_generation_v5_17_candidate_features_rebalanced/instance_candidate_dataset/`
- `output_generation_v5_17_candidate_features_rebalanced/model_runs/random_forest_instance_candidate_v5_17_rebalanced/`
- `output_real_step_s3_v5_17_candidate_features_plain/reports/proposal_recall_s3_v5_17_plain.json`
- `output_real_step_s3_v5_17_candidate_features_plain/reports/proposal_recall_s3_v5_17_plain.md`
- `output_real_step_s3_v5_17_candidate_features_rebalanced/reports/proposal_recall_s3_v5_17_rebalanced.json`
- `output_real_step_s3_v5_17_candidate_features_rebalanced/reports/proposal_recall_s3_v5_17_rebalanced.md`
- `output_real_step_s3_v5_17_candidate_features_comparison/reports/v5_13_v5_14_v5_16_v5_17_comparison.md`
- `output_real_step_s3_v5_17_candidate_features_comparison/reports/v5_17_decision.md`

### Do Not Modify

- `models/gnn_baseline.py`
- `train_gnn_baseline.py`
- `utils/object_candidate_proposer.py`
- `utils/topology_postprocess.py`
- `utils/instance_candidate_postprocess.py`
- `run_gnn_inference.py`
- STEP generation scripts
- existing v5.13/v5.14/v5.16 report directories

## 2. Fixed Inputs

Synthetic candidate dataset inputs:

```text
Synthetic graph dir:
output_generation_v5_5_v51dist_looprole_5400/graphs

Synthetic raw prediction dir:
output_generation_v5_10_instance_candidate_dataset_v55/predictions_raw
```

Real S3 evaluation inputs:

```text
S3 graph dir:
output_real_step_s3_v5_6_b_rib_boundary/graphs

S3 raw prediction dir:
output_real_step_s3_v5_11_proposal_diagnostics/raw_predictions

S3 area truth:
configs/test_part_s3_area_truth_part20_mm2.json

Reference final prediction dir:
output_real_step_s3_v5_10_instance_plus_v59/predictions

v5.13 old classifier:
output_generation_v5_10_instance_candidate_dataset_v55/model_runs/random_forest_instance_candidate_v5_10_v55/model.joblib

v5.14 regressed classifier:
output_generation_v5_14_hybrid_instance_candidate_dataset/model_runs/random_forest_instance_candidate_v5_14_hybrid/model.joblib

v5.16 rebalanced classifier:
output_generation_v5_16_rebalanced_instance_candidate_dataset/model_runs/random_forest_instance_candidate_v5_16_rebalanced/model.joblib
```

## 3. Success Criteria

v5.17 is acceptable as an experiment if:

```text
tests/test_instance_candidate_features.py passes
tests/test_build_instance_candidate_dataset.py passes
tests/test_train_candidate_topology_classifier.py passes
```

The new feature schema must include:

```text
boundary_unique_circle_count
boundary_loop_role_inner_count
boundary_loop_role_outer_count
boundary_cylinder_role_inner_count
boundary_cylinder_role_outer_count
candidate_boundary_inner_wall_face_fraction
candidate_boundary_outer_wall_face_fraction
topology_reason_count
topology_evidence_count
```

Promotion rule:

```text
Promote v5.17 only if one v5.17 classifier beats v5.13 accepted recall 0.4804
and does not exceed v5.13 classifier wrong/rejected count 32.
```

If v5.17 beats v5.14 but not v5.13, keep v5.13 as current best and save v5.17 as a partial feature ablation.

If both v5.17 variants fail to beat v5.14, stop candidate classifier work and move next to proposal generation for `no_area_compatible_proposal` failures, especially rib/slot/transition proposal recall.

## 4. Task 1: Add Role-Aware Feature Tests

**Files:**

- Modify: `tests/test_instance_candidate_features.py`
- Modify: `utils/instance_candidate_features.py`

- [ ] **Step 1: Add test helpers**

Append these helper functions near the top of `tests/test_instance_candidate_features.py`, after existing imports and before the test class:

```python
def role_node(idx, primitive="plane", area=10.0, normal=None, radius=None):
    specific = {}
    if normal is not None:
        specific["normal"] = normal
    if radius is not None:
        specific["radius"] = radius
        specific["axis"] = [0.0, 0.0, 1.0]
    return {
        "idx": idx,
        "primitive_surface": primitive,
        "attributes": {
            "physical": {"area": area, "center_of_mass": [float(idx), 0.0, 0.0]},
            "specific": specific,
            "shape_descriptors": {
                "projected_long_extent": 10.0,
                "projected_short_extent": 2.0,
                "projected_aspect_ratio": 5.0,
            },
        },
    }


def role_edge(src, dst, *, curve_type="circle", length=10.0, loop_role="inner_loop", wall_role="inner_hole_wall"):
    loop_inner = 1.0 if loop_role == "inner_loop" else 0.0
    loop_outer = 1.0 if loop_role == "outer_loop" else 0.0
    loop_unknown = 1.0 if loop_role == "unknown" else 0.0
    wall_inner = 1.0 if wall_role == "inner_hole_wall" else 0.0
    wall_outer = 1.0 if wall_role.startswith("outer") else 0.0
    wall_ambiguous = 1.0 if wall_role == "ambiguous" else 0.0
    return [src, dst], {
        "source_face": src,
        "target_face": dst,
        "curve_type": curve_type,
        "curve_type_id": 2 if curve_type == "circle" else 1,
        "is_circle": 1.0 if curve_type == "circle" else 0.0,
        "is_line": 1.0 if curve_type == "line" else 0.0,
        "length": length,
        "circle_radius": length / 6.283185307179586 if curve_type == "circle" else 0.0,
        "convexity": "smooth" if curve_type == "circle" else "convex",
        "convexity_id": 3 if curve_type == "circle" else 1,
        "adjacent_plane_loop_role": loop_role,
        "adjacent_plane_loop_role_inner": loop_inner,
        "adjacent_plane_loop_role_outer": loop_outer,
        "adjacent_plane_loop_role_unknown": loop_unknown,
        "cylindrical_wall_role_hint": wall_role,
        "cylindrical_wall_role_inner_hint": wall_inner,
        "cylindrical_wall_role_outer_hint": wall_outer,
        "cylindrical_wall_role_ambiguous_hint": wall_ambiguous,
    }
```

- [ ] **Step 2: Add test for inner through-hole role features**

Append this test method inside the existing `InstanceCandidateFeaturesTest` class:

```python
    def test_role_aware_features_identify_inner_cylindrical_wall(self):
        from utils.instance_candidate_features import extract_instance_candidate_features

        nodes = [
            role_node(0, "cylinder", area=20.0, radius=2.0),
            role_node(1, "plane", area=100.0, normal=[0.0, 0.0, 1.0]),
            role_node(2, "plane", area=100.0, normal=[0.0, 0.0, -1.0]),
        ]
        edges = []
        edge_attrs = []
        for pair, attr in [
            role_edge(0, 1, loop_role="inner_loop", wall_role="inner_hole_wall"),
            role_edge(1, 0, loop_role="inner_loop", wall_role="inner_hole_wall"),
            role_edge(0, 2, loop_role="inner_loop", wall_role="inner_hole_wall"),
            role_edge(2, 0, loop_role="inner_loop", wall_role="inner_hole_wall"),
        ]:
            edges.append(pair)
            edge_attrs.append(attr)

        graph = {"nodes": nodes, "edges": edges, "edge_attributes": edge_attrs}
        candidate = {"source": "fallback_rule", "proposal_macro": "through_hole_round", "face_indices": [0]}

        features = extract_instance_candidate_features(graph, candidate, candidate_macro="through_hole_round")

        self.assertEqual(features["boundary_unique_circle_count"], 2.0)
        self.assertEqual(features["boundary_loop_role_inner_count"], 2.0)
        self.assertEqual(features["boundary_loop_role_outer_count"], 0.0)
        self.assertEqual(features["boundary_cylinder_role_inner_count"], 2.0)
        self.assertEqual(features["boundary_cylinder_role_outer_count"], 0.0)
        self.assertAlmostEqual(features["candidate_boundary_inner_wall_face_fraction"], 1.0)
        self.assertAlmostEqual(features["candidate_boundary_outer_wall_face_fraction"], 0.0)
        self.assertEqual(features["topology_valid"], 1.0)
```

- [ ] **Step 3: Add test for outer-cylinder false-positive features**

Append this test method inside `InstanceCandidateFeaturesTest`:

```python
    def test_role_aware_features_identify_outer_cylindrical_wall(self):
        from utils.instance_candidate_features import extract_instance_candidate_features

        nodes = [
            role_node(0, "cylinder", area=30.0, radius=3.0),
            role_node(1, "plane", area=100.0, normal=[0.0, 0.0, 1.0]),
            role_node(2, "plane", area=100.0, normal=[0.0, 0.0, -1.0]),
        ]
        edges = []
        edge_attrs = []
        for pair, attr in [
            role_edge(0, 1, loop_role="outer_loop", wall_role="outer_cylindrical_wall"),
            role_edge(1, 0, loop_role="outer_loop", wall_role="outer_cylindrical_wall"),
            role_edge(0, 2, loop_role="outer_loop", wall_role="outer_cylindrical_wall"),
            role_edge(2, 0, loop_role="outer_loop", wall_role="outer_cylindrical_wall"),
        ]:
            edges.append(pair)
            edge_attrs.append(attr)

        graph = {"nodes": nodes, "edges": edges, "edge_attributes": edge_attrs}
        candidate = {"source": "fallback_rule", "proposal_macro": "through_hole_round", "face_indices": [0]}

        features = extract_instance_candidate_features(graph, candidate, candidate_macro="through_hole_round")

        self.assertEqual(features["boundary_unique_circle_count"], 2.0)
        self.assertEqual(features["boundary_loop_role_inner_count"], 0.0)
        self.assertEqual(features["boundary_loop_role_outer_count"], 2.0)
        self.assertEqual(features["boundary_cylinder_role_inner_count"], 0.0)
        self.assertEqual(features["boundary_cylinder_role_outer_count"], 2.0)
        self.assertAlmostEqual(features["candidate_boundary_inner_wall_face_fraction"], 0.0)
        self.assertAlmostEqual(features["candidate_boundary_outer_wall_face_fraction"], 1.0)
```

- [ ] **Step 4: Add test for topology reason expansion**

Append this test method inside `InstanceCandidateFeaturesTest`:

```python
    def test_topology_reason_features_explain_invalid_through_hole_candidate(self):
        from utils.instance_candidate_features import extract_instance_candidate_features

        nodes = [
            role_node(0, "cylinder", area=20.0, radius=2.0),
            role_node(1, "plane", area=100.0, normal=[0.0, 0.0, 1.0]),
        ]
        pair, attr = role_edge(0, 1, loop_role="inner_loop", wall_role="inner_hole_wall")
        graph = {"nodes": nodes, "edges": [pair], "edge_attributes": [attr]}
        candidate = {"source": "fallback_rule", "proposal_macro": "through_hole_round", "face_indices": [0]}

        features = extract_instance_candidate_features(graph, candidate, candidate_macro="through_hole_round")

        self.assertEqual(features["topology_valid"], 0.0)
        self.assertGreaterEqual(features["topology_reason_count"], 1.0)
        self.assertEqual(features["topology_reason_requires_two_circular_opening_edges"], 1.0)
```

- [ ] **Step 5: Run tests and confirm failure**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_instance_candidate_features -v
```

Expected before implementation:

```text
FAILED
```

The failure should mention missing feature keys such as `boundary_unique_circle_count`.

## 5. Task 2: Implement Role-Aware Edge and Topology Features

**Files:**

- Modify: `utils/instance_candidate_features.py`

- [ ] **Step 1: Add constants**

Add these constants after `CONVEXITY_BUCKETS`:

```python
LOOP_ROLE_BUCKETS = ("inner", "outer", "unknown")
CYLINDER_ROLE_BUCKETS = ("inner", "outer", "ambiguous", "unknown")

TOPOLOGY_REASON_BUCKETS = (
    "single_cylindrical_wall_required",
    "through_hole_should_not_include_bottom_or_countersink_faces",
    "requires_two_circular_opening_edges",
    "missing_cylindrical_wall",
    "missing_bottom_face",
    "missing_wall_bottom_circular_edge",
    "missing_conical_countersink_face",
    "missing_small_cylindrical_wall",
    "missing_cone_cylinder_circular_edge",
    "min_2_coaxial_cylinders_required",
    "missing_shoulder_plane",
    "counterbore_should_not_include_bottom_face",
    "shoulder_plane_must_connect_large_and_small_cylinders",
    "multi_radius_difference_missing",
    "missing_circular_step_edges",
    "min_5_faces_required",
    "requires_planar_bottom_and_side_faces",
    "requires_planar_top_and_side_faces",
    "insufficient_concave_internal_edges",
    "insufficient_convex_internal_edges",
    "thin_parallel_face_pair_required",
)

TOPOLOGY_EVIDENCE_BUCKETS = (
    "single_cylindrical_wall_with_two_openings",
    "cylinder_wall_plus_bottom_face",
    "large_cylinder_shoulder_small_through_cylinder",
    "cone_plus_small_cylindrical_wall",
    "bottom_plus_four_side_faces",
    "top_plus_four_side_faces",
    "cylindrical_side_plus_top_face",
    "thin_parallel_face_pair",
    "transition_face_candidate",
)
```

- [ ] **Step 2: Replace the initial topology feature calculation**

In `extract_instance_candidate_features()`, replace:

```python
        "topology_valid": _topology_valid(graph_data, macro, face_indices),
```

with:

```python
        "topology_valid": 0.0,
```

After:

```python
    internal_edges, boundary_edges = _split_candidate_edges(graph_data, face_set)
```

add:

```python
    topology_result = _topology_result(graph_data, macro, face_indices)
    _add_topology_result_features(features, topology_result)
```

- [ ] **Step 3: Call role-aware feature helpers**

After the existing calls:

```python
    _add_edge_features(features, internal_edges, "internal")
    _add_edge_features(features, boundary_edges, "boundary")
```

add:

```python
    _add_unique_edge_role_features(features, internal_edges, "internal")
    _add_unique_edge_role_features(features, boundary_edges, "boundary")
    _add_candidate_wall_role_features(features, boundary_edges, face_set)
```

- [ ] **Step 4: Implement unique edge and role helpers**

Add these helper functions after `_add_edge_features()`:

```python
def _add_unique_edge_role_features(features: dict[str, float], edges: list[dict[str, Any]], prefix: str) -> None:
    unique_edges = _unique_edge_records(edges)
    edge_count = len(unique_edges)
    features[f"{prefix}_unique_edge_count"] = float(edge_count)

    circle_edges = [edge for edge in unique_edges if _curve_type(edge) == "circle"]
    line_edges = [edge for edge in unique_edges if _curve_type(edge) == "line"]
    features[f"{prefix}_unique_circle_count"] = float(len(circle_edges))
    features[f"{prefix}_unique_line_count"] = float(len(line_edges))
    features[f"{prefix}_unique_circle_ratio"] = _safe_div(len(circle_edges), edge_count)
    features[f"{prefix}_unique_line_ratio"] = _safe_div(len(line_edges), edge_count)
    features[f"{prefix}_unique_circle_length_sum"] = float(sum(_edge_length(edge) for edge in circle_edges))
    features[f"{prefix}_unique_line_length_sum"] = float(sum(_edge_length(edge) for edge in line_edges))
    features[f"{prefix}_unique_circle_length_mean"] = _safe_div(
        sum(_edge_length(edge) for edge in circle_edges),
        len(circle_edges),
    )

    loop_counts = Counter(_loop_role(edge) for edge in unique_edges)
    cylinder_counts = Counter(_cylinder_wall_role(edge) for edge in unique_edges)
    for bucket in LOOP_ROLE_BUCKETS:
        count = float(loop_counts.get(bucket, 0))
        features[f"{prefix}_loop_role_{bucket}_count"] = count
        features[f"{prefix}_loop_role_{bucket}_ratio"] = _safe_div(count, edge_count)
        circle_count = float(sum(1 for edge in circle_edges if _loop_role(edge) == bucket))
        features[f"{prefix}_loop_role_{bucket}_circle_count"] = circle_count
        features[f"{prefix}_loop_role_{bucket}_circle_ratio"] = _safe_div(circle_count, len(circle_edges))
    for bucket in CYLINDER_ROLE_BUCKETS:
        count = float(cylinder_counts.get(bucket, 0))
        features[f"{prefix}_cylinder_role_{bucket}_count"] = count
        features[f"{prefix}_cylinder_role_{bucket}_ratio"] = _safe_div(count, edge_count)
        circle_count = float(sum(1 for edge in circle_edges if _cylinder_wall_role(edge) == bucket))
        features[f"{prefix}_cylinder_role_{bucket}_circle_count"] = circle_count
        features[f"{prefix}_cylinder_role_{bucket}_circle_ratio"] = _safe_div(circle_count, len(circle_edges))


def _unique_edge_records(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    unique: list[dict[str, Any]] = []
    for edge in edges:
        src = int(edge.get("source_face", -1)) if _is_int_like(edge.get("source_face", -1)) else -1
        dst = int(edge.get("target_face", -1)) if _is_int_like(edge.get("target_face", -1)) else -1
        key = (
            min(src, dst),
            max(src, dst),
            _curve_type(edge),
            _loop_role(edge),
            _cylinder_wall_role(edge),
            round(_edge_length(edge), 6),
            round(float(edge.get("circle_radius", 0.0) or 0.0), 6),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(edge)
    return unique


def _edge_length(edge: Mapping[str, Any]) -> float:
    return _finite_float(edge.get("length", 0.0))


def _loop_role(edge: Mapping[str, Any]) -> str:
    role = str(edge.get("adjacent_plane_loop_role", "") or "").strip().lower()
    if role in {"inner_loop", "inner"}:
        return "inner"
    if role in {"outer_loop", "outer"}:
        return "outer"
    if _finite_float(edge.get("adjacent_plane_loop_role_inner", 0.0)) > 0.5:
        return "inner"
    if _finite_float(edge.get("adjacent_plane_loop_role_outer", 0.0)) > 0.5:
        return "outer"
    role_id = edge.get("adjacent_plane_loop_role_id")
    if _is_int_like(role_id):
        return {1: "inner", 2: "outer"}.get(int(role_id), "unknown")
    return "unknown"


def _cylinder_wall_role(edge: Mapping[str, Any]) -> str:
    role = str(edge.get("cylindrical_wall_role_hint", "") or "").strip().lower()
    if "inner" in role:
        return "inner"
    if "outer" in role:
        return "outer"
    if "ambiguous" in role:
        return "ambiguous"
    if _finite_float(edge.get("cylindrical_wall_role_inner_hint", 0.0)) > 0.5:
        return "inner"
    if _finite_float(edge.get("cylindrical_wall_role_outer_hint", 0.0)) > 0.5:
        return "outer"
    if _finite_float(edge.get("cylindrical_wall_role_ambiguous_hint", 0.0)) > 0.5:
        return "ambiguous"
    role_id = edge.get("cylindrical_wall_role_hint_id")
    if _is_int_like(role_id):
        return {1: "inner", 2: "outer", 3: "ambiguous"}.get(int(role_id), "unknown")
    return "unknown"
```

- [ ] **Step 5: Implement candidate wall role fractions**

Add this helper after `_cylinder_wall_role()`:

```python
def _add_candidate_wall_role_features(
    features: dict[str, float],
    boundary_edges: list[dict[str, Any]],
    face_set: set[int],
) -> None:
    role_faces: dict[str, set[int]] = {bucket: set() for bucket in CYLINDER_ROLE_BUCKETS}
    loop_faces: dict[str, set[int]] = {bucket: set() for bucket in LOOP_ROLE_BUCKETS}
    for edge in _unique_edge_records(boundary_edges):
        src = int(edge.get("source_face", -1)) if _is_int_like(edge.get("source_face", -1)) else -1
        dst = int(edge.get("target_face", -1)) if _is_int_like(edge.get("target_face", -1)) else -1
        candidate_face = src if src in face_set else dst if dst in face_set else None
        if candidate_face is None:
            continue
        role_faces[_cylinder_wall_role(edge)].add(candidate_face)
        loop_faces[_loop_role(edge)].add(candidate_face)

    candidate_count = len(face_set)
    for bucket in CYLINDER_ROLE_BUCKETS:
        features[f"candidate_boundary_{bucket}_wall_face_count"] = float(len(role_faces[bucket]))
        features[f"candidate_boundary_{bucket}_wall_face_fraction"] = _safe_div(len(role_faces[bucket]), candidate_count)
    for bucket in LOOP_ROLE_BUCKETS:
        features[f"candidate_boundary_{bucket}_loop_face_count"] = float(len(loop_faces[bucket]))
        features[f"candidate_boundary_{bucket}_loop_face_fraction"] = _safe_div(len(loop_faces[bucket]), candidate_count)
```

- [ ] **Step 6: Implement topology result feature expansion**

Replace `_topology_valid()` with:

```python
def _topology_result(graph_data: dict[str, Any], candidate_macro: str | None, face_indices: list[int]) -> dict[str, Any]:
    if candidate_macro in UNKNOWN_MACROS:
        return {"valid": False, "reasons": ["unknown_macro"], "evidence": [], "may_attach_to_parent_feature": False}
    try:
        return validate_feature_topology(str(candidate_macro), face_indices, graph_data)
    except Exception:
        return {"valid": False, "reasons": ["topology_validation_exception"], "evidence": [], "may_attach_to_parent_feature": False}


def _add_topology_result_features(features: dict[str, float], result: Mapping[str, Any]) -> None:
    reasons = [str(item) for item in result.get("reasons", []) if item is not None]
    evidence = [str(item) for item in result.get("evidence", []) if item is not None]
    features["topology_valid"] = 1.0 if result.get("valid") else 0.0
    features["topology_reason_count"] = float(len(reasons))
    features["topology_evidence_count"] = float(len(evidence))
    features["topology_may_attach_to_parent_feature"] = 1.0 if result.get("may_attach_to_parent_feature") else 0.0
    reason_set = {_sanitize_feature_token(item) for item in reasons}
    evidence_set = {_sanitize_feature_token(item) for item in evidence}
    for reason in TOPOLOGY_REASON_BUCKETS:
        features[f"topology_reason_{reason}"] = 1.0 if reason in reason_set else 0.0
    for item in evidence_set:
        if item in TOPOLOGY_EVIDENCE_BUCKETS:
            features[f"topology_evidence_{item}"] = 1.0
    for item in TOPOLOGY_EVIDENCE_BUCKETS:
        features.setdefault(f"topology_evidence_{item}", 0.0)


def _sanitize_feature_token(value: str) -> str:
    token = "".join(ch if ch.isalnum() else "_" for ch in value.strip().lower())
    while "__" in token:
        token = token.replace("__", "_")
    return token.strip("_")
```

- [ ] **Step 7: Run feature tests**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_instance_candidate_features -v
```

Expected:

```text
OK
```

- [ ] **Step 8: Run dataset-builder tests**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_build_instance_candidate_dataset -v
```

Expected:

```text
OK
```

## 6. Task 3: Generalize the Comparison Script for v5.17

**Files:**

- Modify: `compare_candidate_classifier_recall_reports.py`
- Create: `tests/test_compare_candidate_classifier_recall_reports.py`

- [ ] **Step 1: Add comparison tests**

Create `tests/test_compare_candidate_classifier_recall_reports.py`:

```python
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from compare_candidate_classifier_recall_reports import metric_row, parse_named_report_arg, write_decision


def report(final_recall, accepted_recall, classifier_wrong):
    return {
        "best_by_final_survival_recall": {
            "proposal_stage": {"recall": 0.6814},
            "classified_stage": {"recall": 0.5},
            "accepted_stage": {"recall": accepted_recall},
            "final_survival_stage": {"recall": final_recall},
            "failure_stage_counts": {
                "no_area_compatible_proposal": 65,
                "proposal_exists_classifier_wrong_or_rejected": classifier_wrong,
                "classifier_accepts_but_final_changes_or_removes": 1,
                "survives_final_prediction": 100,
            },
        }
    }


class CompareCandidateClassifierRecallReportsTest(unittest.TestCase):
    def test_parse_named_report_arg(self):
        name, path = parse_named_report_arg("v5.17=some/path/report.json")
        self.assertEqual(name, "v5.17")
        self.assertEqual(str(path), "some\\path\\report.json" if "\\" in str(path) else "some/path/report.json")

    def test_metric_row_reads_best_final_metrics(self):
        row = metric_row("v5.17", report(final_recall=0.55, accepted_recall=0.51, classifier_wrong=28))

        self.assertEqual(row["name"], "v5.17")
        self.assertAlmostEqual(row["accepted_recall"], 0.51)
        self.assertEqual(row["classifier_wrong_or_rejected"], 28)

    def test_decision_promotes_only_if_candidate_beats_v513(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decision.md"
            rows = [
                metric_row("v5.13", report(final_recall=0.5196, accepted_recall=0.4804, classifier_wrong=32)),
                metric_row("v5.17_plain", report(final_recall=0.50, accepted_recall=0.49, classifier_wrong=30)),
                metric_row("v5.17_rebalanced", report(final_recall=0.45, accepted_recall=0.44, classifier_wrong=40)),
            ]

            write_decision(path, rows)

            text = path.read_text(encoding="utf-8")
            self.assertIn("promote_v5_17_plain", text)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Replace fixed CLI with repeated `--report` CLI**

Modify `compare_candidate_classifier_recall_reports.py` so it supports:

```powershell
--report v5.13=path\to\report.json
--report v5.14=path\to\report.json
--report v5.17_plain=path\to\report.json
```

Add this function:

```python
def parse_named_report_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"report argument must use name=path format, got {value!r}")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    raw_path = raw_path.strip()
    if not name or not raw_path:
        raise ValueError(f"report argument must use name=path format, got {value!r}")
    return name, Path(raw_path)
```

Change `parse_args()` to:

```python
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare candidate classifier recall reports.")
    parser.add_argument("--report", action="append", required=True, help="Named report in name=path format.")
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--decision-md", type=Path, required=True)
    return parser.parse_args()
```

Change `main()` to:

```python
def main() -> int:
    args = parse_args()
    named_reports = [parse_named_report_arg(item) for item in args.report]
    rows = [metric_row(name, load_json(path)) for name, path in named_reports]
    write_comparison(args.output_md, rows)
    write_decision(args.decision_md, rows)
    print(json.dumps({"rows": rows}, indent=2, ensure_ascii=False))
    return 0
```

Replace `write_decision()` with:

```python
def write_decision(path: Path, rows: list[dict[str, Any]]) -> None:
    by_name = {row["name"]: row for row in rows}
    v513 = by_name.get("v5.13")
    if v513 is None:
        raise ValueError("v5.13 baseline row is required for decision")

    candidates = [
        row for row in rows
        if row["name"].startswith("v5.17")
    ]
    promotable = [
        row for row in candidates
        if row["accepted_recall"] > v513["accepted_recall"]
        and row["classifier_wrong_or_rejected"] <= v513["classifier_wrong_or_rejected"]
    ]
    if promotable:
        best = max(promotable, key=lambda row: (row["accepted_recall"], row["final_survival_recall"]))
        recommendation = f"promote_{best['name']}"
        reason = (
            f"{best['name']} beats v5.13 accepted recall "
            f"({best['accepted_recall']:.4f} > {v513['accepted_recall']:.4f}) "
            f"without increasing classifier wrong/rejected "
            f"({best['classifier_wrong_or_rejected']} <= {v513['classifier_wrong_or_rejected']})."
        )
    elif any(row["accepted_recall"] > by_name.get("v5.14", v513)["accepted_recall"] for row in candidates):
        recommendation = "keep_v5_13_but_record_v5_17_partial_gain"
        reason = "At least one v5.17 variant improves over v5.14 but does not beat the v5.13 current-best classifier."
    else:
        recommendation = "keep_v5_13_and_move_to_proposal_recall_repairs"
        reason = "v5.17 role-aware features do not improve real S3 accepted recall enough; remaining work should target no-compatible-proposal failures."

    lines = [
        "# v5.17 Decision",
        "",
        "## Recommendation",
        "",
        f"**{recommendation}**",
        "",
        "## Reasoning",
        "",
        reason,
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
```

- [ ] **Step 3: Run comparison tests**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_compare_candidate_classifier_recall_reports -v
```

Expected:

```text
OK
```

## 7. Task 4: Rebuild v5.17 Plain Candidate Dataset

**Files:**

- Generate: `output_generation_v5_17_candidate_features_plain/instance_candidate_dataset/`

- [ ] **Step 1: Build the plain dataset without v5.16 source/macro rebalancing**

Run:

```powershell
W:\python\envs\feagpt\python.exe build_instance_candidate_dataset.py `
  --graph-dir output_generation_v5_5_v51dist_looprole_5400\graphs `
  --prediction-dir output_generation_v5_10_instance_candidate_dataset_v55\predictions_raw `
  --output-dir output_generation_v5_17_candidate_features_plain\instance_candidate_dataset `
  --recursive `
  --proposal-mode hybrid `
  --seed 20260702 `
  --train-ratio 0.80 `
  --val-ratio 0.10
```

Expected console output:

```text
"record_count"
"split_counts"
"label_counts"
```

- [ ] **Step 2: Verify feature schema contains role-aware features**

Run:

```powershell
Select-String -LiteralPath output_generation_v5_17_candidate_features_plain\instance_candidate_dataset\feature_schema.json -Pattern "boundary_unique_circle_count|boundary_loop_role_inner_count|candidate_boundary_inner_wall_face_fraction|topology_reason_count"
```

Expected:

```text
At least four matching lines
```

## 8. Task 5: Rebuild v5.17 Rebalanced Candidate Dataset

**Files:**

- Generate: `output_generation_v5_17_candidate_features_rebalanced/instance_candidate_dataset/`

- [ ] **Step 1: Build the rebalanced dataset with the same v5.16 policy**

Run:

```powershell
W:\python\envs\feagpt\python.exe build_instance_candidate_dataset.py `
  --graph-dir output_generation_v5_5_v51dist_looprole_5400\graphs `
  --prediction-dir output_generation_v5_10_instance_candidate_dataset_v55\predictions_raw `
  --output-dir output_generation_v5_17_candidate_features_rebalanced\instance_candidate_dataset `
  --recursive `
  --proposal-mode hybrid `
  --source-macro-positive-clones "fallback_rule:through_hole_round,topology_rule:through_hole_round,topology_rule:rectangular_boss" `
  --reject-downweight-source-macros "fallback_rule:through_hole_round,topology_rule:through_hole_round,topology_rule:rectangular_boss" `
  --default-sample-weight 1.0 `
  --positive-nontruth-weight 3.0 `
  --positive-clone-weight 5.0 `
  --downweighted-reject-weight 0.35 `
  --seed 20260702 `
  --train-ratio 0.80 `
  --val-ratio 0.10
```

Expected console output:

```text
"record_count"
"split_counts"
"label_counts"
```

- [ ] **Step 2: Verify rebalanced summary**

Run:

```powershell
Get-Content -LiteralPath output_generation_v5_17_candidate_features_rebalanced\instance_candidate_dataset\dataset_summary.json
```

Expected conditions:

```text
"proposal_mode": "hybrid"
"source_macro_positive_clone_count" is greater than 0
"downweighted_reject" appears under "rebalance_action_counts"
```

## 9. Task 6: Train v5.17 Plain and Rebalanced Candidate Classifiers

**Files:**

- Generate: `output_generation_v5_17_candidate_features_plain/model_runs/random_forest_instance_candidate_v5_17_plain/`
- Generate: `output_generation_v5_17_candidate_features_rebalanced/model_runs/random_forest_instance_candidate_v5_17_rebalanced/`

- [ ] **Step 1: Train v5.17 plain**

Run:

```powershell
W:\python\envs\feagpt\python.exe train_candidate_topology_classifier.py `
  --dataset-dir output_generation_v5_17_candidate_features_plain\instance_candidate_dataset `
  --output-dir output_generation_v5_17_candidate_features_plain\model_runs `
  --run-name random_forest_instance_candidate_v5_17_plain `
  --model-kind random_forest `
  --seed 20260702 `
  --n-estimators 300 `
  --max-depth 18 `
  --hard-source-weight 4.0
```

Expected:

```text
"test_macro_f1"
```

- [ ] **Step 2: Train v5.17 rebalanced**

Run:

```powershell
W:\python\envs\feagpt\python.exe train_candidate_topology_classifier.py `
  --dataset-dir output_generation_v5_17_candidate_features_rebalanced\instance_candidate_dataset `
  --output-dir output_generation_v5_17_candidate_features_rebalanced\model_runs `
  --run-name random_forest_instance_candidate_v5_17_rebalanced `
  --model-kind random_forest `
  --seed 20260702 `
  --n-estimators 300 `
  --max-depth 18 `
  --hard-source-weight 1.0
```

Expected:

```text
"test_macro_f1"
```

- [ ] **Step 3: Confirm both models exist**

Run:

```powershell
Test-Path output_generation_v5_17_candidate_features_plain\model_runs\random_forest_instance_candidate_v5_17_plain\model.joblib
Test-Path output_generation_v5_17_candidate_features_rebalanced\model_runs\random_forest_instance_candidate_v5_17_rebalanced\model.joblib
```

Expected:

```text
True
True
```

## 10. Task 7: Run Real S3 Evaluation for Both v5.17 Variants

**Files:**

- Generate: `output_real_step_s3_v5_17_candidate_features_plain/reports/proposal_recall_s3_v5_17_plain.json`
- Generate: `output_real_step_s3_v5_17_candidate_features_plain/reports/proposal_recall_s3_v5_17_plain.md`
- Generate: `output_real_step_s3_v5_17_candidate_features_rebalanced/reports/proposal_recall_s3_v5_17_rebalanced.json`
- Generate: `output_real_step_s3_v5_17_candidate_features_rebalanced/reports/proposal_recall_s3_v5_17_rebalanced.md`

- [ ] **Step 1: Evaluate v5.17 plain**

Run:

```powershell
W:\python\envs\feagpt\python.exe object_proposal_recall_report.py `
  --graph-dir output_real_step_s3_v5_6_b_rib_boundary\graphs `
  --raw-prediction-dir output_real_step_s3_v5_11_proposal_diagnostics\raw_predictions `
  --truth-json configs\test_part_s3_area_truth_part20_mm2.json `
  --classifier output_generation_v5_17_candidate_features_plain\model_runs\random_forest_instance_candidate_v5_17_plain\model.joblib `
  --final-prediction-dir output_real_step_s3_v5_10_instance_plus_v59\predictions `
  --output-json output_real_step_s3_v5_17_candidate_features_plain\reports\proposal_recall_s3_v5_17_plain.json `
  --output-md output_real_step_s3_v5_17_candidate_features_plain\reports\proposal_recall_s3_v5_17_plain.md `
  --proposal-mode hybrid `
  --strong-thresholds "0.3,0.4,0.5,0.55,0.6,0.7" `
  --weak-thresholds "0.2,0.3,0.35,0.4,0.5" `
  --min-confidence 0.50 `
  --absolute-area-tolerance 10.0 `
  --relative-area-tolerance 0.02
```

Expected:

```text
"best_by_final_survival_recall"
```

- [ ] **Step 2: Evaluate v5.17 rebalanced**

Run:

```powershell
W:\python\envs\feagpt\python.exe object_proposal_recall_report.py `
  --graph-dir output_real_step_s3_v5_6_b_rib_boundary\graphs `
  --raw-prediction-dir output_real_step_s3_v5_11_proposal_diagnostics\raw_predictions `
  --truth-json configs\test_part_s3_area_truth_part20_mm2.json `
  --classifier output_generation_v5_17_candidate_features_rebalanced\model_runs\random_forest_instance_candidate_v5_17_rebalanced\model.joblib `
  --final-prediction-dir output_real_step_s3_v5_10_instance_plus_v59\predictions `
  --output-json output_real_step_s3_v5_17_candidate_features_rebalanced\reports\proposal_recall_s3_v5_17_rebalanced.json `
  --output-md output_real_step_s3_v5_17_candidate_features_rebalanced\reports\proposal_recall_s3_v5_17_rebalanced.md `
  --proposal-mode hybrid `
  --strong-thresholds "0.3,0.4,0.5,0.55,0.6,0.7" `
  --weak-thresholds "0.2,0.3,0.35,0.4,0.5" `
  --min-confidence 0.50 `
  --absolute-area-tolerance 10.0 `
  --relative-area-tolerance 0.02
```

Expected:

```text
"best_by_final_survival_recall"
```

## 11. Task 8: Compare v5.13/v5.14/v5.16/v5.17

**Files:**

- Generate: `output_real_step_s3_v5_17_candidate_features_comparison/reports/v5_13_v5_14_v5_16_v5_17_comparison.md`
- Generate: `output_real_step_s3_v5_17_candidate_features_comparison/reports/v5_17_decision.md`

- [ ] **Step 1: Run generalized comparison**

Run:

```powershell
W:\python\envs\feagpt\python.exe compare_candidate_classifier_recall_reports.py `
  --report v5.13=output_real_step_s3_v5_13_transition_opening_proposals\reports\proposal_recall_s3_v5_13.json `
  --report v5.14=output_real_step_s3_v5_14_hybrid_classifier\reports\proposal_recall_s3_v5_14.json `
  --report v5.16=output_real_step_s3_v5_16_rebalanced_classifier\reports\proposal_recall_s3_v5_16.json `
  --report v5.17_plain=output_real_step_s3_v5_17_candidate_features_plain\reports\proposal_recall_s3_v5_17_plain.json `
  --report v5.17_rebalanced=output_real_step_s3_v5_17_candidate_features_rebalanced\reports\proposal_recall_s3_v5_17_rebalanced.json `
  --output-md output_real_step_s3_v5_17_candidate_features_comparison\reports\v5_13_v5_14_v5_16_v5_17_comparison.md `
  --decision-md output_real_step_s3_v5_17_candidate_features_comparison\reports\v5_17_decision.md
```

Expected:

```text
"name": "v5.17_plain"
"name": "v5.17_rebalanced"
```

- [ ] **Step 2: Inspect decision**

Run:

```powershell
Get-Content -LiteralPath output_real_step_s3_v5_17_candidate_features_comparison\reports\v5_17_decision.md
```

Expected:

```text
One of:
promote_v5_17_plain
promote_v5_17_rebalanced
keep_v5_13_but_record_v5_17_partial_gain
keep_v5_13_and_move_to_proposal_recall_repairs
```

## 12. Final Verification Commands

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_instance_candidate_features tests.test_build_instance_candidate_dataset tests.test_train_candidate_topology_classifier tests.test_compare_candidate_classifier_recall_reports -v
```

Expected:

```text
OK
```

Inspect:

```powershell
Get-Content -LiteralPath output_generation_v5_17_candidate_features_plain\model_runs\random_forest_instance_candidate_v5_17_plain\report.md
Get-Content -LiteralPath output_generation_v5_17_candidate_features_rebalanced\model_runs\random_forest_instance_candidate_v5_17_rebalanced\report.md
Get-Content -LiteralPath output_real_step_s3_v5_17_candidate_features_comparison\reports\v5_13_v5_14_v5_16_v5_17_comparison.md
Get-Content -LiteralPath output_real_step_s3_v5_17_candidate_features_comparison\reports\v5_17_decision.md
```

## 13. What the Implementing Agent Must Report

The final response must include:

```text
1. Test command and result.
2. Whether the new feature schema contains role-aware features.
3. v5.17 plain synthetic test macro-F1.
4. v5.17 rebalanced synthetic test macro-F1.
5. v5.17 plain S3 accepted recall and classifier wrong/rejected count.
6. v5.17 rebalanced S3 accepted recall and classifier wrong/rejected count.
7. Whether either v5.17 variant beats v5.13.
8. The exact recommendation from v5_17_decision.md.
```

## 14. How I Will Review This Work

I will first read:

```text
output_real_step_s3_v5_17_candidate_features_comparison/reports/v5_13_v5_14_v5_16_v5_17_comparison.md
output_real_step_s3_v5_17_candidate_features_comparison/reports/v5_17_decision.md
output_generation_v5_17_candidate_features_plain/instance_candidate_dataset/feature_schema.json
output_generation_v5_17_candidate_features_rebalanced/instance_candidate_dataset/feature_schema.json
```

Then I will run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_instance_candidate_features tests.test_build_instance_candidate_dataset tests.test_train_candidate_topology_classifier tests.test_compare_candidate_classifier_recall_reports -v
```

I will promote v5.17 only if it beats v5.13 on real S3 accepted recall without increasing classifier wrong/rejected failures. If it does not, the next plan should stop candidate classifier iterations and focus on proposal recall repairs for `no_area_compatible_proposal`, especially rib, slot, fillet/chamfer, and cylinder boss.

## 15. English Handoff Summary

v5.16 proved that simple source/macro rebalancing is not enough. Implement v5.17 by exposing role-aware edge attributes and topology-rule reasons in `utils/instance_candidate_features.py`, then train both a plain hybrid candidate classifier and a rebalanced hybrid candidate classifier. Compare both against v5.13, v5.14, and v5.16 on the same S3 protocol. Do not change the GNN, graph extraction, STEP generation, or hybrid proposal generation in this iteration.
