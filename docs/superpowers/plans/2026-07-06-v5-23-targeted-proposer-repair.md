# v5.23 Targeted Proposer Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a small, review-aware, topology-rule proposal layer for the largest confirmed residual failures from v5.22, then rerun s3 strict/semantic/review-aware evaluation to decide whether proposer repair is actually helping.

**Architecture:** Do not retrain GNN or candidate classifiers in v5.23. Add narrowly-scoped topology proposal finders for residual rib and transition cases, wire them into `utils/object_candidate_proposer.py` with explicit `assembly_reasons`, and compare v5.23 against v5.21/v5.22 using the existing recall, semantic-policy, proposal-miss, and review-aware atlas tools. The new rules must be geometry/topology-driven, not area-truth driven.

**Tech Stack:** Python 3 from `W:/python/envs/feagpt/python.exe`, `unittest`, existing `utils/topology_postprocess.py`, existing `utils/object_candidate_proposer.py`, existing v5.21/v5.22 evaluation scripts.

---

## 0. Review of v5.22 Output

v5.22 satisfies the previous plan.

Verification command:

```powershell
W:\python\envs\feagpt\python.exe -m unittest `
  tests.test_review_aware_residual_failure_atlas `
  tests.test_export_area_truth_from_xlsx `
  tests.test_apply_semantic_review_policy_to_recall_report `
  tests.test_write_v5_21_decision `
  tests.test_semantic_review_decision_ingest `
  tests.test_semantic_cross_macro_audit -v
```

Observed:

```text
Ran 32 tests
OK
```

v5.22 review-aware residual status counts:

```text
non_cross_macro_engineering_failure: 41
confirmed_strict_cross_macro: 18
truth_corrected_strict_failure: 4
semantic_policy_accepted_auxiliary: 2
unreviewed_cross_macro: 0
```

Top residual macros:

```text
rib: 23
fillet: 18
through_hole_round: 6
chamfer: 4
counterbore_hole: 3
slot_rect: 2
countersink_hole: 2
cylinder_boss: 2
```

Interpretation:

- The agent's v5.22 recommendation `write_targeted_proposer_repair_plan` is reasonable.
- No human semantic review is needed now.
- The next version should not re-review cross-macro rows.
- The next version should not start with training a larger neural model.
- The next version should first test whether missing/incorrect object proposals are the bottleneck for `rib`, `fillet`, and `chamfer`.

## 1. Human Task

No human intervention is required to implement v5.23.

The user should inspect only the final v5.23 comparison report:

```text
output_real_step_s3_v5_23_targeted_proposer_repair\reports\v5_23_vs_v5_22_review_aware_comparison.md
```

If v5.23 reduces residual `rib` and `fillet/chamfer` misses without creating many new false-positive proposal matches, continue to candidate-classifier or postprocess repair. If v5.23 does not improve proposal-stage recall, stop adding rules and move to model/candidate-classifier redesign.

## 2. Scope Control

### Modify

- `utils/topology_postprocess.py`
- `utils/object_candidate_proposer.py`

### Create

- `tests/test_topology_postprocess_v5_23.py`
- `tests/test_object_candidate_proposer_v5_23.py`
- `compare_review_aware_residual_atlases.py`
- `tests/test_compare_review_aware_residual_atlases.py`
- `output_real_step_s3_v5_23_targeted_proposer_repair/reports/proposal_recall_s3_v5_23_targeted_proposer.json`
- `output_real_step_s3_v5_23_targeted_proposer_repair/reports/proposal_recall_s3_v5_23_targeted_proposer.md`
- `output_real_step_s3_v5_23_targeted_proposer_repair/reports/semantic_policy_recall_s3_v5_23.json`
- `output_real_step_s3_v5_23_targeted_proposer_repair/reports/semantic_policy_recall_s3_v5_23.md`
- `output_real_step_s3_v5_23_targeted_proposer_repair/reports/v5_23_proposal_miss_atlas.json`
- `output_real_step_s3_v5_23_targeted_proposer_repair/reports/v5_23_proposal_miss_atlas.md`
- `output_real_step_s3_v5_23_targeted_proposer_repair/reports/review_aware_residual_atlas.json`
- `output_real_step_s3_v5_23_targeted_proposer_repair/reports/review_aware_residual_atlas.md`
- `output_real_step_s3_v5_23_targeted_proposer_repair/reports/v5_23_vs_v5_22_review_aware_comparison.md`
- `output_real_step_s3_v5_23_targeted_proposer_repair/reports/v5_23_decision.md`

### Do Not Modify

- GNN architecture
- GNN training data
- candidate classifier training data
- model checkpoint files
- graph extraction
- STEP generation
- area truth tables
- v5.20-v5.22 report directories

v5.23 is a proposer-layer experiment only.

## 3. Repair Strategy

Add three review-targeted topology proposal functions:

```text
find_review_targeted_rib_candidates
find_review_targeted_fillet_candidates
find_review_targeted_chamfer_candidates
```

Design constraints:

- These functions must not use area-truth values or part names.
- They may use B-Rep descriptors already present in graph nodes and edges: primitive surface type, area, projected aspect ratio, projected long/short extent, normals, edge curve type, edge convexity, edge length, and neighbor count.
- They should be permissive enough to propose residual objects, but still guarded by shape checks and existing validation helpers when possible.
- Each proposal must carry a distinct assembly reason:

```text
topology_review_targeted_rib_residual
topology_review_targeted_fillet_residual
topology_review_targeted_chamfer_residual
```

Expected v5.23 improvement target:

- Proposal-stage residual misses for `rib` should decrease.
- Proposal-stage residual misses for `fillet` and `chamfer` should decrease.
- Strict final recall may improve less than proposal recall if the candidate classifier rejects the new proposals. That outcome is still useful because it tells us v5.24 should repair classifier/postprocess rather than proposer.

## 4. Fixed Evaluation Inputs

Reuse the reviewed truth from v5.21:

```text
configs\test_part_s3_area_truth_v5_21_reviewed.json
```

Reuse the same s3 graph/prediction/model inputs:

```text
graph_dir: output_real_step_s3_v5_6_b_rib_boundary\graphs
raw_prediction_dir: output_real_step_s3_v5_11_proposal_diagnostics\raw_predictions
classifier: output_generation_v5_10_instance_candidate_dataset_v55\model_runs\random_forest_instance_candidate_v5_10_v55\model.joblib
final_prediction_dir: output_real_step_s3_v5_10_instance_plus_v59\predictions
proposal_mode: hybrid
absolute_area_tolerance: 10.0
relative_area_tolerance: 0.02
strong_thresholds: 0.3,0.4,0.5,0.55,0.6,0.7
weak_thresholds: 0.2,0.3,0.35,0.4,0.5
```

## 5. Task 1: Add Topology Finder Tests

**Files:**

- Create: `tests/test_topology_postprocess_v5_23.py`
- Modify: `utils/topology_postprocess.py`

- [ ] **Step 1: Create failing tests**

Create `tests/test_topology_postprocess_v5_23.py`:

```python
from __future__ import annotations

import unittest

from utils.topology_postprocess import (
    find_review_targeted_chamfer_candidates,
    find_review_targeted_fillet_candidates,
    find_review_targeted_rib_candidates,
)


def plane_node(idx: int, area: float, aspect: float, long_extent: float, short_extent: float, normal=(0.0, 0.0, 1.0), center=(0.0, 0.0, 0.0)):
    return {
        "idx": idx,
        "primitive_surface": "plane",
        "area": area,
        "attributes": {
            "physical": {"area": area},
            "shape_descriptors": {
                "projected_aspect_ratio": aspect,
                "projected_long_extent": long_extent,
                "projected_short_extent": short_extent,
                "center": list(center),
                "normal": list(normal),
            },
        },
    }


def cylinder_node(idx: int, area: float, aspect: float, long_extent: float, short_extent: float):
    return {
        "idx": idx,
        "primitive_surface": "cylinder",
        "area": area,
        "attributes": {
            "physical": {"area": area},
            "shape_descriptors": {
                "projected_aspect_ratio": aspect,
                "projected_long_extent": long_extent,
                "projected_short_extent": short_extent,
            },
        },
    }


def graph(nodes, edges, edge_attributes):
    return {"nodes": nodes, "edges": edges, "edge_attributes": edge_attributes}


class TopologyPostprocessV523Test(unittest.TestCase):
    def test_review_targeted_rib_finds_parallel_thin_sides_with_cap(self):
        nodes = [
            plane_node(1, area=500.0, aspect=8.0, long_extent=100.0, short_extent=12.0, normal=(0.0, 1.0, 0.0), center=(0.0, -1.0, 0.0)),
            plane_node(2, area=500.0, aspect=8.0, long_extent=100.0, short_extent=12.0, normal=(0.0, 1.0, 0.0), center=(0.0, 1.0, 0.0)),
            plane_node(3, area=120.0, aspect=6.0, long_extent=100.0, short_extent=2.0, normal=(0.0, 0.0, 1.0), center=(0.0, 0.0, 6.0)),
            plane_node(10, area=5000.0, aspect=1.0, long_extent=100.0, short_extent=50.0, normal=(1.0, 0.0, 0.0), center=(0.0, 0.0, -1.0)),
        ]
        faces = [
            {"idx": 1, "macro_feature": "stock", "semantic_subtype": "stock_face"},
            {"idx": 2, "macro_feature": "stock", "semantic_subtype": "stock_face"},
            {"idx": 3, "macro_feature": "stock", "semantic_subtype": "stock_face"},
            {"idx": 10, "macro_feature": "stock", "semantic_subtype": "stock_face"},
        ]
        data = graph(
            nodes,
            edges=[{"source": 1, "target": 3}, {"source": 2, "target": 3}, {"source": 1, "target": 10}, {"source": 2, "target": 10}],
            edge_attributes=[
                {"curve_type": "line", "convexity": "convex", "length": 100.0},
                {"curve_type": "line", "convexity": "convex", "length": 100.0},
                {"curve_type": "line", "convexity": "concave", "length": 100.0},
                {"curve_type": "line", "convexity": "concave", "length": 100.0},
            ],
        )

        candidates = find_review_targeted_rib_candidates(faces, data, set())

        self.assertIn([1, 2, 3], candidates)

    def test_review_targeted_fillet_finds_single_curved_transition_face(self):
        nodes = [
            cylinder_node(4, area=235.0, aspect=12.0, long_extent=50.0, short_extent=4.0),
            plane_node(1, area=2000.0, aspect=1.5, long_extent=80.0, short_extent=50.0),
            plane_node(2, area=1800.0, aspect=1.4, long_extent=80.0, short_extent=45.0),
        ]
        faces = [
            {"idx": 4, "macro_feature": "stock", "semantic_subtype": "stock_face", "macro_confidence": 0.35},
            {"idx": 1, "macro_feature": "stock", "semantic_subtype": "stock_face"},
            {"idx": 2, "macro_feature": "stock", "semantic_subtype": "stock_face"},
        ]
        data = graph(
            nodes,
            edges=[{"source": 4, "target": 1}, {"source": 4, "target": 2}],
            edge_attributes=[
                {"curve_type": "line", "convexity": "convex", "length": 50.0},
                {"curve_type": "line", "convexity": "convex", "length": 50.0},
            ],
        )

        candidates = find_review_targeted_fillet_candidates(faces, data, set())

        self.assertIn([4], candidates)

    def test_review_targeted_chamfer_finds_narrow_bevel_plane(self):
        nodes = [
            plane_node(5, area=16.0, aspect=9.0, long_extent=40.0, short_extent=4.0),
            plane_node(1, area=2000.0, aspect=1.5, long_extent=80.0, short_extent=50.0),
            plane_node(2, area=1800.0, aspect=1.4, long_extent=80.0, short_extent=45.0),
        ]
        faces = [
            {"idx": 5, "macro_feature": "stock", "semantic_subtype": "stock_face", "macro_confidence": 0.30},
            {"idx": 1, "macro_feature": "stock", "semantic_subtype": "stock_face"},
            {"idx": 2, "macro_feature": "stock", "semantic_subtype": "stock_face"},
        ]
        data = graph(
            nodes,
            edges=[{"source": 5, "target": 1}, {"source": 5, "target": 2}],
            edge_attributes=[
                {"curve_type": "line", "convexity": "convex", "length": 40.0, "dihedral_angle": 0.785},
                {"curve_type": "line", "convexity": "convex", "length": 40.0, "dihedral_angle": 0.785},
            ],
        )

        candidates = find_review_targeted_chamfer_candidates(faces, data, set())

        self.assertIn([5], candidates)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_topology_postprocess_v5_23 -v
```

Expected before implementation:

```text
ImportError
```

The missing names should be:

```text
find_review_targeted_rib_candidates
find_review_targeted_fillet_candidates
find_review_targeted_chamfer_candidates
```

## 6. Task 2: Implement Targeted Topology Finders

**Files:**

- Modify: `utils/topology_postprocess.py`

- [ ] **Step 1: Add helper functions near existing rib/transition finder helpers**

Add these functions near `find_rib_stock_candidates`, `find_fillet_stock_candidates`, and `find_chamfer_stock_candidates`:

```python
def _node_center(node: dict) -> tuple[float, float, float]:
    descriptors = node.get("attributes", {}).get("shape_descriptors", {})
    values = descriptors.get("center") or node.get("center") or [0.0, 0.0, 0.0]
    if len(values) < 3:
        return (0.0, 0.0, 0.0)
    return (float(values[0]), float(values[1]), float(values[2]))


def _review_targeted_face_allowed(face: dict, allowed_macros: set[str], allowed_semantics: set[str]) -> bool:
    return (
        face.get("macro_feature") in allowed_macros
        or face.get("raw_macro_feature") in allowed_macros
        or face.get("semantic_subtype") in allowed_semantics
        or face.get("raw_semantic_subtype") in allowed_semantics
        or float(face.get("macro_confidence", 1.0) or 1.0) < 0.50
    )


def _review_targeted_neighbor_large_support_count(face_idx: int, graph_data: dict, *, min_area_ratio: float = 4.0) -> int:
    nodes = _nodes_by_idx(graph_data)
    node = nodes.get(face_idx, {})
    area = _node_area(node)
    if area <= 1e-9:
        return 0
    count = 0
    for src, dst, attr in _edge_records(graph_data):
        if face_idx not in {src, dst}:
            continue
        other = dst if src == face_idx else src
        if _node_area(nodes.get(other, {})) >= area * min_area_ratio:
            count += 1
    return count


def _review_targeted_has_two_long_edges(face_idx: int, graph_data: dict) -> bool:
    lengths = []
    for src, dst, attr in _edge_records(graph_data):
        if face_idx not in {src, dst}:
            continue
        if _is_line_edge(attr):
            lengths.append(float(attr.get("length", 0.0) or 0.0))
    positive = [length for length in lengths if length > 1e-9]
    if len(positive) < 2:
        return False
    positive.sort(reverse=True)
    return positive[1] >= positive[0] * 0.35
```

- [ ] **Step 2: Add rib finder**

Add:

```python
def find_review_targeted_rib_candidates(faces: list[dict], graph_data: dict, repaired_faces: set[int]) -> list[list[int]]:
    face_by_idx = {int(face["idx"]): face for face in faces}
    nodes = _nodes_by_idx(graph_data)
    candidate_faces = [
        idx
        for idx, node in nodes.items()
        if idx not in repaired_faces
        and node.get("primitive_surface") == "plane"
        and _review_targeted_face_allowed(
            face_by_idx.get(idx, {}),
            {"stock", "rib", "web", "fillet", "chamfer"},
            {"stock_face", "rib_side_plane", "rib_top_plane", "fillet_blend_surface", "chamfer_bevel_plane"},
        )
    ]
    candidates: list[list[int]] = []
    for i, left_idx in enumerate(candidate_faces):
        for right_idx in candidate_faces[i + 1:]:
            left = nodes[left_idx]
            right = nodes[right_idx]
            if _angle_between_abs(_node_normal(left), _node_normal(right)) > 0.22:
                continue
            left_aspect = _node_descriptor(left, "projected_aspect_ratio")
            right_aspect = _node_descriptor(right, "projected_aspect_ratio")
            if max(left_aspect, right_aspect) < 3.0:
                continue
            thickness = _distance_between_nodes(left, right)
            if thickness <= 1e-9:
                continue
            long_extent = max(_node_descriptor(left, "projected_long_extent"), _node_descriptor(right, "projected_long_extent"))
            short_extent = max(_node_descriptor(left, "projected_short_extent"), _node_descriptor(right, "projected_short_extent"))
            if long_extent <= 1e-9 or short_extent <= 1e-9:
                continue
            if long_extent / thickness < 6.0:
                continue
            if thickness / max(short_extent, 1e-9) > 0.45:
                continue
            cap_idx = _rib_cap_between_side_faces(left_idx, right_idx, candidate_faces, graph_data)
            if cap_idx is None or cap_idx in repaired_faces:
                continue
            cap = nodes.get(cap_idx, {})
            if cap.get("primitive_surface") != "plane":
                continue
            if _node_area(cap) > max(_node_area(left), _node_area(right)) * 0.6:
                continue
            candidate = sorted([left_idx, right_idx, cap_idx])
            if _rib_candidate_is_too_large_for_model_context(candidate, graph_data):
                continue
            if candidate not in candidates:
                candidates.append(candidate)
    return candidates
```

- [ ] **Step 3: Add fillet finder**

Add:

```python
def find_review_targeted_fillet_candidates(faces: list[dict], graph_data: dict, repaired_faces: set[int]) -> list[list[int]]:
    face_by_idx = {int(face["idx"]): face for face in faces}
    nodes = _nodes_by_idx(graph_data)
    candidates: list[list[int]] = []
    for idx, node in nodes.items():
        if idx in repaired_faces:
            continue
        primitive = node.get("primitive_surface")
        if primitive not in {"cylinder", "toroid", "surface_of_revolution", "surface_of_extrusion", "bspline_surface"}:
            continue
        face = face_by_idx.get(idx, {})
        if not _review_targeted_face_allowed(face, {"stock", "fillet", "blind_hole_round", "pocket_circular"}, {"stock_face", "fillet_blend_surface"}):
            continue
        if _node_area(node) <= 1e-6:
            continue
        if primitive == "cylinder" and _looks_like_exterior_cylinder_not_slot_bottom(idx, node, graph_data):
            continue
        if primitive == "cylinder" and _looks_like_convex_partial_cylinder_opening(idx, graph_data):
            continue
        if _boundary_neighbor_count(idx, graph_data) < 2:
            continue
        if _review_targeted_neighbor_large_support_count(idx, graph_data, min_area_ratio=3.0) < 1:
            continue
        if not _review_targeted_has_two_long_edges(idx, graph_data):
            continue
        candidate = [idx]
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates
```

- [ ] **Step 4: Add chamfer finder**

Add:

```python
def find_review_targeted_chamfer_candidates(faces: list[dict], graph_data: dict, repaired_faces: set[int]) -> list[list[int]]:
    face_by_idx = {int(face["idx"]): face for face in faces}
    nodes = _nodes_by_idx(graph_data)
    candidates: list[list[int]] = []
    for idx, node in nodes.items():
        if idx in repaired_faces or node.get("primitive_surface") != "plane":
            continue
        face = face_by_idx.get(idx, {})
        if not _review_targeted_face_allowed(face, {"stock", "chamfer"}, {"stock_face", "chamfer_bevel_plane"}):
            continue
        area = _node_area(node)
        if area <= 1e-6:
            continue
        aspect = _node_descriptor(node, "projected_aspect_ratio")
        if aspect < 7.0:
            continue
        if _boundary_neighbor_count(idx, graph_data) > 8:
            continue
        if _review_targeted_neighbor_large_support_count(idx, graph_data, min_area_ratio=5.0) < 1:
            continue
        if not _review_targeted_has_two_long_edges(idx, graph_data):
            continue
        candidate = [idx]
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates
```

- [ ] **Step 5: Run topology tests**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_topology_postprocess_v5_23 -v
```

Expected:

```text
OK
```

If a test fails because an existing helper requires a slightly different descriptor key, adapt the synthetic test fixture and implementation together. Keep the public function names unchanged.

## 6. Task 3: Wire Targeted Finders into Object Proposer

**Files:**

- Modify: `utils/object_candidate_proposer.py`
- Create: `tests/test_object_candidate_proposer_v5_23.py`

- [ ] **Step 1: Create object proposer test**

Create `tests/test_object_candidate_proposer_v5_23.py`:

```python
from __future__ import annotations

import unittest
from unittest.mock import patch


class ObjectCandidateProposerV523Test(unittest.TestCase):
    def test_topology_rule_proposals_include_review_targeted_residual_finders(self):
        from utils.object_candidate_proposer import topology_rule_proposals

        graph = {"nodes": [], "edges": [], "edge_attributes": []}
        faces = []
        patchers = [
            patch("utils.object_candidate_proposer.find_through_hole_opening_candidates", return_value=[]),
            patch("utils.object_candidate_proposer.find_blind_hole_candidates", return_value=[]),
            patch("utils.object_candidate_proposer.find_counterbore_candidates", return_value=[]),
            patch("utils.object_candidate_proposer.find_countersink_candidates", return_value=[]),
            patch("utils.object_candidate_proposer.find_cylinder_boss_candidates", return_value=[]),
            patch("utils.object_candidate_proposer.find_obround_hole_candidates", return_value=[]),
            patch("utils.object_candidate_proposer.find_rectangular_prism_candidates", return_value=[]),
            patch("utils.object_candidate_proposer.find_supported_rectangular_boss_candidates", return_value=[]),
            patch("utils.object_candidate_proposer.find_u_slot_candidates", return_value=[]),
            patch("utils.object_candidate_proposer.find_rectangular_slot_candidates", return_value=[]),
            patch("utils.object_candidate_proposer.find_o_slot_candidates", return_value=[]),
            patch("utils.object_candidate_proposer.find_v_slot_stock_candidates", return_value=[]),
            patch("utils.object_candidate_proposer.find_mixed_rib_candidates", return_value=[]),
            patch("utils.object_candidate_proposer.find_rib_stock_candidates", return_value=[]),
            patch("utils.object_candidate_proposer.find_existing_rib_web_cap_completion_candidates", return_value=[]),
            patch("utils.object_candidate_proposer.find_transition_cap_rib_candidates", return_value=[]),
            patch("utils.object_candidate_proposer.find_pocket_embedded_rib_candidates", return_value=[]),
            patch("utils.object_candidate_proposer.find_existing_rib_transition_absorption_candidates", return_value=[]),
            patch("utils.object_candidate_proposer.find_fillet_stock_candidates", return_value=[]),
            patch("utils.object_candidate_proposer.find_chamfer_stock_candidates", return_value=[]),
            patch("utils.object_candidate_proposer.find_review_targeted_rib_candidates", return_value=[[1, 2, 3]]),
            patch("utils.object_candidate_proposer.find_review_targeted_fillet_candidates", return_value=[[4]]),
            patch("utils.object_candidate_proposer.find_review_targeted_chamfer_candidates", return_value=[[5]]),
        ]
        with patchers[0]:
            with patchers[1]:
                with patchers[2]:
                    with patchers[3]:
                        with patchers[4]:
                            with patchers[5]:
                                with patchers[6]:
                                    with patchers[7]:
                                        with patchers[8]:
                                            with patchers[9]:
                                                with patchers[10]:
                                                    with patchers[11]:
                                                        with patchers[12]:
                                                            with patchers[13]:
                                                                with patchers[14]:
                                                                    with patchers[15]:
                                                                        with patchers[16]:
                                                                            with patchers[17]:
                                                                                with patchers[18]:
                                                                                    with patchers[19]:
                                                                                        with patchers[20]:
                                                                                            with patchers[21]:
                                                                                                with patchers[22]:
                                                                                                    proposals = topology_rule_proposals(graph, faces)

        simplified = sorted(
            (proposal["proposal_macro"], proposal["face_indices"], proposal["assembly_reasons"])
            for proposal in proposals
        )
        self.assertIn(("chamfer", [5], ["topology_review_targeted_chamfer_residual"]), simplified)
        self.assertIn(("fillet", [4], ["topology_review_targeted_fillet_residual"]), simplified)
        self.assertIn(("rib", [1, 2, 3], ["topology_review_targeted_rib_residual"]), simplified)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test and confirm failure**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_object_candidate_proposer_v5_23 -v
```

Expected before implementation:

```text
AttributeError
```

The missing patched attributes should be the three new finder imports.

- [ ] **Step 3: Import new finders**

Modify the import block in `utils/object_candidate_proposer.py` to import:

```python
    find_review_targeted_chamfer_candidates,
    find_review_targeted_fillet_candidates,
    find_review_targeted_rib_candidates,
```

- [ ] **Step 4: Add proposal specs**

In `topology_rule_proposals`, after existing fillet/chamfer specs, add:

```python
        ("rib", "topology_review_targeted_rib_residual", find_review_targeted_rib_candidates(faces, graph, repaired)),
        ("fillet", "topology_review_targeted_fillet_residual", find_review_targeted_fillet_candidates(faces, graph, repaired)),
        ("chamfer", "topology_review_targeted_chamfer_residual", find_review_targeted_chamfer_candidates(faces, graph, repaired)),
```

- [ ] **Step 5: Run object proposer tests**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_object_candidate_proposer tests.test_object_candidate_proposer_v5_23 -v
```

Expected:

```text
OK
```

## 7. Task 4: Add Review-Aware Comparison Tool

**Files:**

- Create: `compare_review_aware_residual_atlases.py`
- Create: `tests/test_compare_review_aware_residual_atlases.py`

- [ ] **Step 1: Create tests**

Create `tests/test_compare_review_aware_residual_atlases.py`:

```python
from __future__ import annotations

import unittest

from compare_review_aware_residual_atlases import compare_summaries, delta_counts


class CompareReviewAwareResidualAtlasesTest(unittest.TestCase):
    def test_delta_counts_subtracts_new_from_old_by_key(self):
        old = {"rib": 23, "fillet": 18, "slot_u": 1}
        new = {"rib": 20, "fillet": 18, "chamfer": 3}

        delta = delta_counts(old, new)

        self.assertEqual(delta["rib"], -3)
        self.assertEqual(delta["fillet"], 0)
        self.assertEqual(delta["slot_u"], -1)
        self.assertEqual(delta["chamfer"], 3)

    def test_compare_summaries_reports_macro_and_status_deltas(self):
        old = {"summary": {"by_macro": {"rib": 23}, "by_review_status": {"non_cross_macro_engineering_failure": 41}}}
        new = {"summary": {"by_macro": {"rib": 20}, "by_review_status": {"non_cross_macro_engineering_failure": 38}}}

        comparison = compare_summaries(old, new)

        self.assertEqual(comparison["macro_delta"]["rib"], -3)
        self.assertEqual(comparison["review_status_delta"]["non_cross_macro_engineering_failure"], -3)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Implement comparison tool**

Create `compare_review_aware_residual_atlases.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def delta_counts(old: dict[str, int], new: dict[str, int]) -> dict[str, int]:
    keys = sorted(set(old) | set(new))
    return {key: int(new.get(key, 0)) - int(old.get(key, 0)) for key in keys}


def compare_summaries(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    old_summary = old.get("summary", {})
    new_summary = new.get("summary", {})
    return {
        "old_recommendation": old.get("recommendation", ""),
        "new_recommendation": new.get("recommendation", ""),
        "macro_delta": delta_counts(old_summary.get("by_macro", {}), new_summary.get("by_macro", {})),
        "review_status_delta": delta_counts(old_summary.get("by_review_status", {}), new_summary.get("by_review_status", {})),
        "miss_count_delta": int(new_summary.get("miss_count", 0)) - int(old_summary.get("miss_count", 0)),
    }


def write_markdown(path: Path, comparison: dict[str, Any]) -> None:
    lines = [
        "# v5.23 vs v5.22 Review-Aware Residual Comparison",
        "",
        f"- old_recommendation: `{comparison['old_recommendation']}`",
        f"- new_recommendation: `{comparison['new_recommendation']}`",
        f"- miss_count_delta: `{comparison['miss_count_delta']}`",
        "",
        "## Macro Delta",
        "",
        "| Macro | Delta |",
        "|-------|-------|",
    ]
    for macro, delta in comparison["macro_delta"].items():
        lines.append(f"| {macro} | {delta} |")
    lines.extend(["", "## Review Status Delta", "", "| Status | Delta |", "|--------|-------|"])
    for status, delta in comparison["review_status_delta"].items():
        lines.append(f"| {status} | {delta} |")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two review-aware residual atlases.")
    parser.add_argument("--old-json", type=Path, required=True)
    parser.add_argument("--new-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    comparison = compare_summaries(load_json(args.old_json), load_json(args.new_json))
    write_markdown(args.output_md, comparison)
    print(json.dumps(comparison, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Run tests**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_compare_review_aware_residual_atlases -v
```

Expected:

```text
OK
```

## 8. Task 5: Run v5.23 Evaluation Loop

**Files:**

- Generate all v5.23 reports under `output_real_step_s3_v5_23_targeted_proposer_repair/reports`.

- [ ] **Step 1: Rerun strict recall with targeted proposer code**

Run:

```powershell
W:\python\envs\feagpt\python.exe object_proposal_recall_report.py `
  --graph-dir output_real_step_s3_v5_6_b_rib_boundary\graphs `
  --raw-prediction-dir output_real_step_s3_v5_11_proposal_diagnostics\raw_predictions `
  --truth-json configs\test_part_s3_area_truth_v5_21_reviewed.json `
  --classifier output_generation_v5_10_instance_candidate_dataset_v55\model_runs\random_forest_instance_candidate_v5_10_v55\model.joblib `
  --final-prediction-dir output_real_step_s3_v5_10_instance_plus_v59\predictions `
  --proposal-mode hybrid `
  --strong-thresholds 0.3,0.4,0.5,0.55,0.6,0.7 `
  --weak-thresholds 0.2,0.3,0.35,0.4,0.5 `
  --absolute-area-tolerance 10.0 `
  --relative-area-tolerance 0.02 `
  --output-json output_real_step_s3_v5_23_targeted_proposer_repair\reports\proposal_recall_s3_v5_23_targeted_proposer.json `
  --output-md output_real_step_s3_v5_23_targeted_proposer_repair\reports\proposal_recall_s3_v5_23_targeted_proposer.md
```

Expected:

```text
Report written to output_real_step_s3_v5_23_targeted_proposer_repair\reports\proposal_recall_s3_v5_23_targeted_proposer.json
```

- [ ] **Step 2: Apply semantic policy**

Run:

```powershell
W:\python\envs\feagpt\python.exe apply_semantic_review_policy_to_recall_report.py `
  --report-json output_real_step_s3_v5_23_targeted_proposer_repair\reports\proposal_recall_s3_v5_23_targeted_proposer.json `
  --review-csv output_real_step_s3_v5_20_user_semantic_review\reports\semantic_review_template.csv `
  --output-json output_real_step_s3_v5_23_targeted_proposer_repair\reports\semantic_policy_recall_s3_v5_23.json `
  --output-md output_real_step_s3_v5_23_targeted_proposer_repair\reports\semantic_policy_recall_s3_v5_23.md `
  --area-tolerance 0.02
```

- [ ] **Step 3: Build proposal miss atlas**

Run:

```powershell
W:\python\envs\feagpt\python.exe proposal_miss_atlas.py `
  --report v5_23=output_real_step_s3_v5_23_targeted_proposer_repair\reports\proposal_recall_s3_v5_23_targeted_proposer.json `
  --output-dir output_real_step_s3_v5_23_targeted_proposer_repair\reports
```

Expected generated files:

```text
v5_23_proposal_miss_atlas.json
v5_23_proposal_miss_atlas.md
```

- [ ] **Step 4: Build review-aware residual atlas**

Run:

```powershell
W:\python\envs\feagpt\python.exe review_aware_residual_failure_atlas.py `
  --atlas-json output_real_step_s3_v5_23_targeted_proposer_repair\reports\v5_23_proposal_miss_atlas.json `
  --review-csv output_real_step_s3_v5_20_user_semantic_review\reports\semantic_review_template.csv `
  --semantic-policy-json output_real_step_s3_v5_23_targeted_proposer_repair\reports\semantic_policy_recall_s3_v5_23.json `
  --output-dir output_real_step_s3_v5_23_targeted_proposer_repair\reports
```

- [ ] **Step 5: Compare v5.22 and v5.23**

Run:

```powershell
W:\python\envs\feagpt\python.exe compare_review_aware_residual_atlases.py `
  --old-json output_real_step_s3_v5_22_review_aware_residual_atlas\reports\review_aware_residual_atlas.json `
  --new-json output_real_step_s3_v5_23_targeted_proposer_repair\reports\review_aware_residual_atlas.json `
  --output-md output_real_step_s3_v5_23_targeted_proposer_repair\reports\v5_23_vs_v5_22_review_aware_comparison.md
```

## 9. Task 6: Final Verification

**Files:**

- Verify code and all v5.23 reports.

- [ ] **Step 1: Run relevant unit tests**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest `
  tests.test_topology_postprocess_v5_23 `
  tests.test_object_candidate_proposer_v5_23 `
  tests.test_compare_review_aware_residual_atlases `
  tests.test_object_candidate_proposer `
  tests.test_topology_postprocess_v5_8 `
  tests.test_review_aware_residual_failure_atlas -v
```

Expected:

```text
OK
```

- [ ] **Step 2: Verify report outputs**

Run:

```powershell
Get-ChildItem -LiteralPath output_real_step_s3_v5_23_targeted_proposer_repair\reports | Select-Object Name,Length,LastWriteTime
```

Expected files:

```text
proposal_recall_s3_v5_23_targeted_proposer.json
proposal_recall_s3_v5_23_targeted_proposer.md
semantic_policy_recall_s3_v5_23.json
semantic_policy_recall_s3_v5_23.md
v5_23_proposal_miss_atlas.json
v5_23_proposal_miss_atlas.md
review_aware_residual_atlas.json
review_aware_residual_atlas.md
v5_23_vs_v5_22_review_aware_comparison.md
```

- [ ] **Step 3: Write v5.23 decision note**

Create `output_real_step_s3_v5_23_targeted_proposer_repair/reports/v5_23_decision.md` manually from the report numbers, with this structure:

```markdown
# v5.23 Decision

## Recommendation

**<one of: continue_proposer_repair | switch_to_candidate_classifier_repair | rollback_v5_23_rules | stop_engineering_and_write_results>**

## Evidence

- strict_recall_v5_21:
- strict_recall_v5_23:
- semantic_policy_recall_v5_21:
- semantic_policy_recall_v5_23:
- rib_delta:
- fillet_delta:
- chamfer_delta:
- non_cross_macro_engineering_failure_delta:

## Reasoning

<short explanation grounded in v5_23_vs_v5_22_review_aware_comparison.md>
```

Choose:

- `continue_proposer_repair` if residual rib/fillet/chamfer counts drop and no important macro gets worse.
- `switch_to_candidate_classifier_repair` if proposal-stage recall improves but accepted/final-survival recall does not.
- `rollback_v5_23_rules` if residual counts or obvious false-positive-compatible rows get worse.
- `stop_engineering_and_write_results` if the remaining errors are small enough and mostly ambiguous.

## 10. Final Response Requirements

The implementing agent must report:

```text
1. Test commands and results.
2. Which files were modified.
3. v5.21/v5.23 strict recall comparison.
4. v5.21/v5.23 semantic-policy recall comparison.
5. v5.22/v5.23 review-aware residual macro deltas for rib, fillet, chamfer, through_hole_round, slot_rect, cylinder_boss.
6. Whether proposal-stage recall improved but final recall did not.
7. v5.23 recommendation.
8. Confirmation that no training, graph extraction, STEP generation, or model checkpoint files were modified.
```

## 11. How I Will Review This Work

I will inspect:

```text
utils/topology_postprocess.py
utils/object_candidate_proposer.py
tests/test_topology_postprocess_v5_23.py
tests/test_object_candidate_proposer_v5_23.py
compare_review_aware_residual_atlases.py
tests/test_compare_review_aware_residual_atlases.py
output_real_step_s3_v5_23_targeted_proposer_repair\reports\v5_23_vs_v5_22_review_aware_comparison.md
output_real_step_s3_v5_23_targeted_proposer_repair\reports\v5_23_decision.md
```

I will run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest `
  tests.test_topology_postprocess_v5_23 `
  tests.test_object_candidate_proposer_v5_23 `
  tests.test_compare_review_aware_residual_atlases `
  tests.test_object_candidate_proposer `
  tests.test_topology_postprocess_v5_8 `
  tests.test_review_aware_residual_failure_atlas -v
```

I will reject the implementation if:

- It uses area-truth values inside proposal generation.
- It marks semantic-policy auxiliary rows as strict successes.
- It changes model files or retrains anything in v5.23.
- It improves one macro by creating broad, uncontrolled false positives in other macros.
- It does not produce a v5.22-vs-v5.23 comparison report.

## 12. Next Plan After v5.23

Use `v5_23_decision.md`:

- If `continue_proposer_repair`, write a second targeted proposer plan for holes/slots/cylinder-boss residuals.
- If `switch_to_candidate_classifier_repair`, train or recalibrate a candidate-level classifier using the new v5.23 proposals and the review-aware residual atlas.
- If `rollback_v5_23_rules`, remove the new broad proposal rules and move to model/data improvements.
- If `stop_engineering_and_write_results`, freeze the pipeline and start paper experiment/report writing.

## 13. English Handoff Summary

v5.22 confirmed there are no new unreviewed semantic conflicts. The largest strict residual macros are rib and fillet, followed by through_hole_round and chamfer. v5.23 should add narrowly-scoped topology-rule proposal finders for rib, fillet, and chamfer only, then rerun the full s3 evaluation loop and compare review-aware residual counts against v5.22. Do not train models or touch generated data in this version.
