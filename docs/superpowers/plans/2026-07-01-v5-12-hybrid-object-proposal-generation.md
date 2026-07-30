# v5.12 Hybrid Object Proposal Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a diagnostic hybrid object proposer that combines same-instance assembly, raw macro connected components, and existing topology-rule candidates, then verify whether proposal recall on `test_part_s3` improves over v5.11.1.

**Architecture:** Keep v5.9-C as the current usable baseline and keep v5.10/v5.12 as a research branch. Add a new proposal utility module, integrate it into `object_proposal_recall_report.py` behind `--proposal-mode hybrid`, and generate a source-attributed recall report. Do not change GNN architecture, training data, model checkpoints, data generation, topology postprocess behavior, or final inference defaults.

**Tech Stack:** Python 3 from `W:/python/envs/feagpt/python.exe`, `unittest`, existing JSON graph/prediction artifacts, existing `utils.topology_postprocess` candidate finders, existing `utils.area_truth_normalization`.

---

## 0. Current Review Result

v5.11.1 fixed the v5.11 evaluator macro alias problem.

Verified results:

```text
v5.11 old proposal recall        0.1716
v5.11.1 macrofix proposal recall 0.3775
v5.11 old final-survival recall        0.1667
v5.11.1 macrofix final-survival recall 0.3480
```

Corrected v5.11.1 failure-stage counts:

```text
survives_final_prediction                         71
classifier_accepts_but_final_changes_or_removes   1
proposal_exists_classifier_wrong_or_rejected       7
no_area_compatible_proposal                      125
```

Corrected v5.11.1 decision:

```text
proposal_generation_fix
```

This recommendation is reasonable because `no_area_compatible_proposal` remains dominant after macro normalization. The main issue is not the candidate classifier. The current candidate source is too narrow: same-instance assembly depends on edge same-instance probabilities, and the threshold grid has no effect on recall. v5.12 must test broader proposal sources before changing the neural network or dataset again.

## 1. Scope Control

### Files to Create

- `utils/object_candidate_proposer.py`
- `tests/test_object_candidate_proposer.py`
- `output_real_step_s3_v5_12_hybrid_proposals/reports/proposal_recall_s3_hybrid.json`
- `output_real_step_s3_v5_12_hybrid_proposals/reports/proposal_recall_s3_hybrid.md`
- `output_real_step_s3_v5_12_hybrid_proposals/reports/v5_12_hybrid_decision.md`
- `output_real_step_s3_v5_12_hybrid_proposals/reports/v5_11_1_vs_v5_12_hybrid_comparison.md`

### Files to Modify

- `object_proposal_recall_report.py`
- `tests/test_object_proposal_recall_report.py`

### Files Not to Modify

Do not modify:

- `models/gnn_baseline.py`
- `train_gnn_baseline.py`
- `build_instance_candidate_dataset.py`
- `utils/instance_candidate_assembly.py`
- `utils/instance_candidate_postprocess.py`
- `run_gnn_inference.py`
- `prediction_to_feature_objects.py`
- any file under `features/`
- any model checkpoint

This task only tests whether broader candidate proposals can recover the missing objects. It must not change production/default inference behavior.

## 2. Design Requirements

### Proposal Sources

The hybrid proposer must emit candidates from three sources:

1. `same_instance`
   - Existing `utils.instance_candidate_assembly.assemble_instance_candidates`.

2. `raw_macro_component`
   - Connected components of adjacent faces with the same normalized non-stock predicted macro.
   - This source approximates the older connected-component grouping that currently performs better than assembled instance candidates.

3. `topology_rule`
   - Existing topology-rule candidate finders from `utils.topology_postprocess`.
   - These propose objects from B-Rep topology even when raw macro labels are fragmented.

### Candidate Record Format

Every proposal record must use this schema:

```python
{
    "source": "raw_macro_component",
    "proposal_sources": ["raw_macro_component"],
    "proposal_macro": "through_hole_round",
    "face_indices": [7],
    "seed_face_indices": [7],
    "assembly_reasons": ["raw_macro_connected_component"],
    "mean_same_instance_probability": None,
    "min_same_instance_probability": None,
}
```

If the same macro and face set is found by multiple sources, deduplicate it into one record:

```python
{
    "source": "hybrid_merged",
    "proposal_sources": ["same_instance", "raw_macro_component", "topology_rule"],
    "proposal_macro": "rib",
    "face_indices": [1, 2, 3],
    "seed_face_indices": [1, 2, 3],
    "assembly_reasons": ["strong_same_instance_edge", "raw_macro_connected_component", "topology_rib_stock"],
    "mean_same_instance_probability": 0.82,
    "min_same_instance_probability": 0.74,
}
```

Deduplication key:

```python
(proposal_macro, tuple(sorted(face_indices)))
```

### Target Macros

Use this exact target macro set:

```python
TARGET_PROPOSAL_MACROS = {
    "through_hole_round",
    "blind_hole_round",
    "stepped_hole_round",
    "counterbore_hole",
    "countersink_hole",
    "obround_hole",
    "slot_rect",
    "slot_u",
    "slot_o",
    "slot_v",
    "pocket_rect",
    "rib",
    "web",
    "fillet",
    "chamfer",
    "cylinder_boss",
    "rectangular_boss",
}
```

Exclude these macros from proposal generation:

```python
STOCK_OR_BACKGROUND_MACROS = {
    "",
    "stock",
    "stock_face",
    "background",
    "unknown",
    "unknown_feature",
    "pocket_circular",
}
```

`pocket_circular` remains excluded from this proposal experiment because it is not in the current s3 truth target set and would add noise to the v5.12 decision.

## 3. Task 1: Add Tests for Raw Macro Connected-Component Proposals

**Files:**

- Create: `tests/test_object_candidate_proposer.py`
- Create: `utils/object_candidate_proposer.py`

- [ ] **Step 1: Create failing tests**

Create `tests/test_object_candidate_proposer.py` with:

```python
from __future__ import annotations

import unittest
from unittest.mock import patch


class ObjectCandidateProposerTest(unittest.TestCase):
    def test_merge_face_records_uses_prediction_macro_and_graph_area(self):
        from utils.object_candidate_proposer import merge_face_records

        graph = {
            "nodes": [
                {"idx": 0, "primitive_surface": "cylinder", "attributes": {"physical": {"area": 20.0}}},
                {"idx": 1, "primitive_surface": "plane", "attributes": {"physical": {"area": 5.0}}},
            ],
            "edges": [[0, 1]],
        }
        prediction = {
            "faces": [
                {"idx": 0, "macro_feature": "through_hole", "semantic_subtype": "through_hole_cylinder"},
                {"idx": 1, "macro_feature": "stock", "semantic_subtype": "stock_face"},
            ]
        }

        faces = merge_face_records(graph, prediction)
        by_idx = {face["idx"]: face for face in faces}

        self.assertEqual(by_idx[0]["macro_feature"], "through_hole_round")
        self.assertEqual(by_idx[0]["raw_macro_feature"], "through_hole")
        self.assertEqual(by_idx[0]["area"], 20.0)
        self.assertEqual(by_idx[0]["primitive_surface"], "cylinder")
        self.assertEqual(by_idx[1]["macro_feature"], "stock")

    def test_raw_macro_component_proposals_group_adjacent_same_macro_faces(self):
        from utils.object_candidate_proposer import raw_macro_component_proposals

        graph = {
            "nodes": [
                {"idx": 0, "primitive_surface": "plane", "attributes": {"physical": {"area": 10.0}}},
                {"idx": 1, "primitive_surface": "plane", "attributes": {"physical": {"area": 11.0}}},
                {"idx": 2, "primitive_surface": "plane", "attributes": {"physical": {"area": 100.0}}},
                {"idx": 3, "primitive_surface": "plane", "attributes": {"physical": {"area": 12.0}}},
            ],
            "edges": [[0, 1], [1, 2], [2, 3]],
        }
        faces = [
            {"idx": 0, "macro_feature": "rib", "area": 10.0},
            {"idx": 1, "macro_feature": "rib", "area": 11.0},
            {"idx": 2, "macro_feature": "stock", "area": 100.0},
            {"idx": 3, "macro_feature": "rib", "area": 12.0},
        ]

        proposals = raw_macro_component_proposals(graph, faces)

        self.assertEqual(
            [(p["proposal_macro"], p["face_indices"], p["proposal_sources"]) for p in proposals],
            [
                ("rib", [0, 1], ["raw_macro_component"]),
                ("rib", [3], ["raw_macro_component"]),
            ],
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_object_candidate_proposer -v
```

Expected:

```text
ERROR
```

because `utils.object_candidate_proposer` does not exist yet.

- [ ] **Step 3: Create `utils/object_candidate_proposer.py`**

Create the file with:

```python
from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable
from typing import Any

from utils.area_truth_normalization import normalize_truth_macro


TARGET_PROPOSAL_MACROS = {
    "through_hole_round",
    "blind_hole_round",
    "stepped_hole_round",
    "counterbore_hole",
    "countersink_hole",
    "obround_hole",
    "slot_rect",
    "slot_u",
    "slot_o",
    "slot_v",
    "pocket_rect",
    "rib",
    "web",
    "fillet",
    "chamfer",
    "cylinder_boss",
    "rectangular_boss",
}

STOCK_OR_BACKGROUND_MACROS = {
    "",
    "stock",
    "stock_face",
    "background",
    "unknown",
    "unknown_feature",
    "pocket_circular",
}


def normalize_macro(value: object) -> str:
    text = str(value).strip()
    if not text:
        return "unknown"
    return normalize_truth_macro(text)


def node_idx(record: dict[str, Any]) -> int:
    for key in ("idx", "face_index", "index", "id"):
        value = record.get(key)
        if value is not None:
            return int(value)
    raise ValueError(f"face record has no index field: {record}")


def node_area(record: dict[str, Any]) -> float:
    attributes = record.get("attributes")
    if isinstance(attributes, dict):
        physical = attributes.get("physical")
        if isinstance(physical, dict) and physical.get("area") is not None:
            return float(physical["area"])
    if record.get("area") is not None:
        return float(record["area"])
    return 0.0


def extract_graph_edges(graph: dict[str, Any]) -> list[tuple[int, int]]:
    edges = []
    for edge in graph.get("edges", []):
        if isinstance(edge, dict):
            if "src" in edge and "dst" in edge:
                edges.append((int(edge["src"]), int(edge["dst"])))
            elif "source" in edge and "target" in edge:
                edges.append((int(edge["source"]), int(edge["target"])))
        elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
            edges.append((int(edge[0]), int(edge[1])))
    return edges


def graph_nodes_by_idx(graph: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {node_idx(node): node for node in graph.get("nodes", graph.get("faces", []))}


def prediction_faces_by_idx(prediction: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {node_idx(face): face for face in prediction.get("faces", prediction.get("nodes", []))}


def merge_face_records(graph: dict[str, Any], prediction: dict[str, Any]) -> list[dict[str, Any]]:
    graph_nodes = graph_nodes_by_idx(graph)
    predicted_faces = prediction_faces_by_idx(prediction)
    merged = []
    for idx in sorted(graph_nodes):
        node = graph_nodes[idx]
        pred = predicted_faces.get(idx, {})
        raw_macro = pred.get("macro_feature") or pred.get("predicted_macro_feature") or pred.get("macro") or ""
        raw_semantic = pred.get("semantic_subtype") or pred.get("predicted_semantic_subtype") or ""
        primitive = pred.get("primitive_surface") or node.get("primitive_surface") or node.get("primitive") or "unknown"
        record = {
            "idx": idx,
            "primitive_surface": primitive,
            "macro_feature": normalize_macro(raw_macro),
            "raw_macro_feature": str(raw_macro),
            "semantic_subtype": str(raw_semantic),
            "raw_semantic_subtype": str(raw_semantic),
            "area": node_area(node),
            "attributes": node.get("attributes", {}),
        }
        for key in ("macro_confidence", "semantic_confidence", "derived_macro_features", "derived_macro_feature_scores"):
            if key in pred:
                record[key] = pred[key]
        merged.append(record)
    return merged


def candidate_record(source: str, macro: str, face_indices: Iterable[int], reason: str) -> dict[str, Any]:
    faces = sorted(int(idx) for idx in face_indices)
    return {
        "source": source,
        "proposal_sources": [source],
        "proposal_macro": normalize_macro(macro),
        "face_indices": faces,
        "seed_face_indices": faces,
        "assembly_reasons": [reason],
        "mean_same_instance_probability": None,
        "min_same_instance_probability": None,
    }


def raw_macro_component_proposals(graph: dict[str, Any], faces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    face_by_idx = {int(face["idx"]): face for face in faces}
    target_faces = {
        idx
        for idx, face in face_by_idx.items()
        if face.get("macro_feature") in TARGET_PROPOSAL_MACROS
        and face.get("macro_feature") not in STOCK_OR_BACKGROUND_MACROS
    }
    adjacency: dict[int, set[int]] = {idx: set() for idx in target_faces}
    for src, dst in extract_graph_edges(graph):
        if src not in target_faces or dst not in target_faces:
            continue
        if face_by_idx[src].get("macro_feature") != face_by_idx[dst].get("macro_feature"):
            continue
        adjacency[src].add(dst)
        adjacency[dst].add(src)

    visited: set[int] = set()
    proposals: list[dict[str, Any]] = []
    for seed in sorted(target_faces):
        if seed in visited:
            continue
        macro = face_by_idx[seed]["macro_feature"]
        queue = deque([seed])
        visited.add(seed)
        component = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in sorted(adjacency.get(current, set())):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append(neighbor)
        proposals.append(candidate_record("raw_macro_component", macro, component, "raw_macro_connected_component"))
    return proposals
```

- [ ] **Step 4: Run tests and confirm pass**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_object_candidate_proposer -v
```

Expected:

```text
OK
```

## 4. Task 2: Add Topology-Rule Proposal Source

**Files:**

- Modify: `utils/object_candidate_proposer.py`
- Modify: `tests/test_object_candidate_proposer.py`

- [ ] **Step 1: Add topology-rule test with patched finders**

Append this test method inside `ObjectCandidateProposerTest`:

```python
    def test_topology_rule_proposals_wrap_existing_finders(self):
        from utils.object_candidate_proposer import topology_rule_proposals

        graph = {
            "nodes": [
                {"idx": 0, "primitive_surface": "cylinder", "attributes": {"physical": {"area": 30.0}}},
                {"idx": 1, "primitive_surface": "plane", "attributes": {"physical": {"area": 8.0}}},
            ],
            "edges": [[0, 1]],
        }
        faces = [
            {"idx": 0, "macro_feature": "stock", "area": 30.0, "primitive_surface": "cylinder"},
            {"idx": 1, "macro_feature": "stock", "area": 8.0, "primitive_surface": "plane"},
        ]

        with patch("utils.object_candidate_proposer.find_through_hole_opening_candidates", return_value=[[0]]), \
             patch("utils.object_candidate_proposer.find_blind_hole_candidates", return_value=[[0, 1]]), \
             patch("utils.object_candidate_proposer.find_counterbore_candidates", return_value=[]), \
             patch("utils.object_candidate_proposer.find_countersink_candidates", return_value=[]), \
             patch("utils.object_candidate_proposer.find_cylinder_boss_candidates", return_value=[]), \
             patch("utils.object_candidate_proposer.find_obround_hole_candidates", return_value=[]), \
             patch("utils.object_candidate_proposer.find_rectangular_prism_candidates", return_value=[]), \
             patch("utils.object_candidate_proposer.find_supported_rectangular_boss_candidates", return_value=[]), \
             patch("utils.object_candidate_proposer.find_u_slot_candidates", return_value=[]), \
             patch("utils.object_candidate_proposer.find_rectangular_slot_candidates", return_value=[]), \
             patch("utils.object_candidate_proposer.find_o_slot_candidates", return_value=[]), \
             patch("utils.object_candidate_proposer.find_v_slot_stock_candidates", return_value=[]), \
             patch("utils.object_candidate_proposer.find_mixed_rib_candidates", return_value=[]), \
             patch("utils.object_candidate_proposer.find_rib_stock_candidates", return_value=[]), \
             patch("utils.object_candidate_proposer.find_existing_rib_web_cap_completion_candidates", return_value=[]), \
             patch("utils.object_candidate_proposer.find_transition_cap_rib_candidates", return_value=[]), \
             patch("utils.object_candidate_proposer.find_pocket_embedded_rib_candidates", return_value=[]), \
             patch("utils.object_candidate_proposer.find_existing_rib_transition_absorption_candidates", return_value=[]):
            proposals = topology_rule_proposals(graph, faces)

        self.assertEqual(
            [(p["proposal_macro"], p["face_indices"], p["proposal_sources"]) for p in proposals],
            [
                ("through_hole_round", [0], ["topology_rule"]),
                ("blind_hole_round", [0, 1], ["topology_rule"]),
            ],
        )
```

- [ ] **Step 2: Run focused test and confirm failure**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_object_candidate_proposer.ObjectCandidateProposerTest.test_topology_rule_proposals_wrap_existing_finders -v
```

Expected:

```text
ERROR
```

because `topology_rule_proposals` does not exist yet.

- [ ] **Step 3: Import topology candidate finders**

Add these imports to `utils/object_candidate_proposer.py`:

```python
from utils.topology_postprocess import (
    find_blind_hole_candidates,
    find_counterbore_candidates,
    find_countersink_candidates,
    find_cylinder_boss_candidates,
    find_existing_rib_transition_absorption_candidates,
    find_existing_rib_web_cap_completion_candidates,
    find_mixed_rib_candidates,
    find_o_slot_candidates,
    find_obround_hole_candidates,
    find_pocket_embedded_rib_candidates,
    find_rectangular_prism_candidates,
    find_rectangular_slot_candidates,
    find_rib_stock_candidates,
    find_supported_rectangular_boss_candidates,
    find_through_hole_opening_candidates,
    find_transition_cap_rib_candidates,
    find_u_slot_candidates,
    find_v_slot_stock_candidates,
)
```

- [ ] **Step 4: Implement topology-rule proposals**

Add:

```python
def topology_rule_proposals(
    graph: dict[str, Any],
    faces: list[dict[str, Any]],
    repaired_faces: set[int] | None = None,
) -> list[dict[str, Any]]:
    repaired = set(repaired_faces or set())
    proposal_specs: list[tuple[str, str, list[list[int]]]] = [
        ("through_hole_round", "topology_through_hole_opening", find_through_hole_opening_candidates(faces, graph, repaired)),
        ("blind_hole_round", "topology_blind_hole_wall_bottom", find_blind_hole_candidates(faces, graph, repaired)),
        ("counterbore_hole", "topology_counterbore", find_counterbore_candidates(faces, graph, repaired)),
        ("countersink_hole", "topology_countersink", find_countersink_candidates(faces, graph, repaired)),
        ("cylinder_boss", "topology_cylinder_boss", find_cylinder_boss_candidates(faces, graph, repaired)),
        ("obround_hole", "topology_obround_hole", find_obround_hole_candidates(faces, graph, repaired)),
        ("pocket_rect", "topology_rectangular_pocket", find_rectangular_prism_candidates(faces, graph, repaired, "pocket_rect", "concave")),
        ("rectangular_boss", "topology_rectangular_boss", find_rectangular_prism_candidates(faces, graph, repaired, "rectangular_boss", "convex")),
        ("rectangular_boss", "topology_supported_rectangular_boss", find_supported_rectangular_boss_candidates(faces, graph, repaired)),
        ("slot_u", "topology_u_slot", find_u_slot_candidates(faces, graph, repaired)),
        ("slot_rect", "topology_rectangular_slot", find_rectangular_slot_candidates(faces, graph, repaired)),
        ("slot_o", "topology_o_slot", find_o_slot_candidates(faces, graph, repaired)),
        ("slot_v", "topology_v_slot", find_v_slot_stock_candidates(faces, graph, repaired)),
        ("rib", "topology_mixed_rib", find_mixed_rib_candidates(faces, graph, repaired)),
        ("rib", "topology_rib_stock", find_rib_stock_candidates(faces, graph, repaired)),
        ("rib", "topology_existing_rib_cap_completion", find_existing_rib_web_cap_completion_candidates(faces, graph, repaired)),
        ("rib", "topology_transition_cap_rib", find_transition_cap_rib_candidates(faces, graph, repaired)),
        ("rib", "topology_pocket_embedded_rib", find_pocket_embedded_rib_candidates(faces, graph, repaired)),
        ("rib", "topology_existing_rib_transition_absorption", find_existing_rib_transition_absorption_candidates(faces, graph, repaired)),
    ]
    proposals: list[dict[str, Any]] = []
    for macro, reason, candidates in proposal_specs:
        for face_indices in candidates:
            proposals.append(candidate_record("topology_rule", macro, face_indices, reason))
    return proposals
```

- [ ] **Step 5: Run focused test**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_object_candidate_proposer.ObjectCandidateProposerTest.test_topology_rule_proposals_wrap_existing_finders -v
```

Expected:

```text
OK
```

## 5. Task 3: Add Hybrid Proposal Deduplication

**Files:**

- Modify: `utils/object_candidate_proposer.py`
- Modify: `tests/test_object_candidate_proposer.py`

- [ ] **Step 1: Add deduplication test**

Append this test method inside `ObjectCandidateProposerTest`:

```python
    def test_deduplicate_proposals_merges_sources_for_same_macro_and_faces(self):
        from utils.object_candidate_proposer import deduplicate_proposals

        proposals = [
            {
                "source": "same_instance",
                "proposal_sources": ["same_instance"],
                "proposal_macro": "rib",
                "face_indices": [3, 1, 2],
                "seed_face_indices": [3, 1, 2],
                "assembly_reasons": ["strong_same_instance_edge"],
                "mean_same_instance_probability": 0.8,
                "min_same_instance_probability": 0.7,
            },
            {
                "source": "raw_macro_component",
                "proposal_sources": ["raw_macro_component"],
                "proposal_macro": "rib",
                "face_indices": [1, 2, 3],
                "seed_face_indices": [1, 2, 3],
                "assembly_reasons": ["raw_macro_connected_component"],
                "mean_same_instance_probability": None,
                "min_same_instance_probability": None,
            },
        ]

        merged = deduplicate_proposals(proposals)

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["source"], "hybrid_merged")
        self.assertEqual(merged[0]["proposal_sources"], ["same_instance", "raw_macro_component"])
        self.assertEqual(merged[0]["face_indices"], [1, 2, 3])
        self.assertEqual(
            merged[0]["assembly_reasons"],
            ["strong_same_instance_edge", "raw_macro_connected_component"],
        )
        self.assertEqual(merged[0]["mean_same_instance_probability"], 0.8)
        self.assertEqual(merged[0]["min_same_instance_probability"], 0.7)
```

- [ ] **Step 2: Run focused test and confirm failure**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_object_candidate_proposer.ObjectCandidateProposerTest.test_deduplicate_proposals_merges_sources_for_same_macro_and_faces -v
```

Expected:

```text
ERROR
```

- [ ] **Step 3: Implement deduplication and hybrid generation**

Add:

```python
def proposal_key(proposal: dict[str, Any]) -> tuple[str, tuple[int, ...]]:
    return (
        normalize_macro(proposal.get("proposal_macro", "")),
        tuple(sorted(int(idx) for idx in proposal.get("face_indices", []))),
    )


def deduplicate_proposals(proposals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged_by_key: dict[tuple[str, tuple[int, ...]], dict[str, Any]] = {}
    for proposal in proposals:
        key = proposal_key(proposal)
        if not key[1]:
            continue
        current = merged_by_key.get(key)
        normalized = dict(proposal)
        normalized["proposal_macro"] = key[0]
        normalized["face_indices"] = list(key[1])
        normalized["seed_face_indices"] = sorted(int(idx) for idx in normalized.get("seed_face_indices", key[1]))
        normalized["proposal_sources"] = list(normalized.get("proposal_sources", [normalized.get("source", "unknown")]))
        normalized["assembly_reasons"] = list(normalized.get("assembly_reasons", []))
        if current is None:
            merged_by_key[key] = normalized
            continue
        source_set = set(current.get("proposal_sources", [])) | set(normalized.get("proposal_sources", []))
        reason_list = list(current.get("assembly_reasons", []))
        for reason in normalized.get("assembly_reasons", []):
            if reason not in reason_list:
                reason_list.append(reason)
        current["source"] = "hybrid_merged"
        current["proposal_sources"] = sorted(source_set, key=lambda item: ["same_instance", "raw_macro_component", "topology_rule"].index(item) if item in {"same_instance", "raw_macro_component", "topology_rule"} else 99)
        current["assembly_reasons"] = reason_list
        for key_name in ("mean_same_instance_probability", "min_same_instance_probability"):
            if current.get(key_name) is None and normalized.get(key_name) is not None:
                current[key_name] = normalized[key_name]
    return [
        merged_by_key[key]
        for key in sorted(merged_by_key, key=lambda item: (item[0], item[1][0], len(item[1]), item[1]))
    ]


def generate_hybrid_object_proposals(
    graph: dict[str, Any],
    prediction: dict[str, Any],
    same_instance_proposals: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    faces = merge_face_records(graph, prediction)
    proposals: list[dict[str, Any]] = []
    for proposal in same_instance_proposals or []:
        item = dict(proposal)
        item["source"] = "same_instance"
        item["proposal_sources"] = ["same_instance"]
        item["proposal_macro"] = normalize_macro(item.get("proposal_macro", ""))
        proposals.append(item)
    proposals.extend(raw_macro_component_proposals(graph, faces))
    proposals.extend(topology_rule_proposals(graph, faces))
    return deduplicate_proposals(proposals)
```

- [ ] **Step 4: Run all proposer tests**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_object_candidate_proposer -v
```

Expected:

```text
OK
```

## 6. Task 4: Integrate Hybrid Proposal Mode into Diagnostic Report

**Files:**

- Modify: `object_proposal_recall_report.py`
- Modify: `tests/test_object_proposal_recall_report.py`

- [ ] **Step 1: Add diagnostic mode test**

Append this test class to `tests/test_object_proposal_recall_report.py` before the final `if __name__ == "__main__":` block:

```python
class TestHybridProposalMode(unittest.TestCase):
    @patch("object_proposal_recall_report.generate_hybrid_object_proposals")
    @patch("object_proposal_recall_report.assemble_instance_candidates")
    def test_replay_candidates_uses_hybrid_proposer_when_requested(self, mock_same_instance, mock_hybrid):
        from object_proposal_recall_report import replay_candidates_for_sample

        mock_same_instance.return_value = [
            {
                "source": "same_instance_seed_component",
                "proposal_macro": "rib",
                "face_indices": [1],
                "seed_face_indices": [1],
                "assembly_reasons": ["single_face_seed"],
            }
        ]
        mock_hybrid.return_value = [
            {
                "source": "raw_macro_component",
                "proposal_sources": ["raw_macro_component"],
                "proposal_macro": "rib",
                "face_indices": [1, 2, 3],
                "seed_face_indices": [1, 2, 3],
                "assembly_reasons": ["raw_macro_connected_component"],
            }
        ]

        candidates = replay_candidates_for_sample(
            graph={"nodes": [], "edges": []},
            raw_prediction={"faces": []},
            classifier_bundle=None,
            strong_threshold=0.3,
            weak_threshold=0.2,
            min_confidence=0.5,
            proposal_mode="hybrid",
        )

        mock_same_instance.assert_called_once()
        mock_hybrid.assert_called_once()
        self.assertEqual(candidates[0]["face_indices"], [1, 2, 3])
        self.assertEqual(candidates[0]["proposal_sources"], ["raw_macro_component"])
```

- [ ] **Step 2: Run focused test and confirm failure**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_object_proposal_recall_report.TestHybridProposalMode -v
```

Expected:

```text
ERROR
```

because `replay_candidates_for_sample` does not accept `proposal_mode` and the module does not import `generate_hybrid_object_proposals`.

- [ ] **Step 3: Move imports to module level**

In `object_proposal_recall_report.py`, add module-level imports:

```python
from utils.instance_candidate_assembly import assemble_instance_candidates
from utils.instance_candidate_postprocess import classify_instance_candidate_proposals
from utils.object_candidate_proposer import generate_hybrid_object_proposals
```

Remove the local imports inside `replay_candidates_for_sample`.

- [ ] **Step 4: Preserve proposal sources in candidate summaries**

Inside `build_candidate_summaries`, add `proposal_sources`:

```python
proposal_sources = list(proposal.get("proposal_sources", [proposal.get("source", "unknown")]))
```

Add this field to the `candidate` dict:

```python
"proposal_sources": proposal_sources,
```

- [ ] **Step 5: Add `proposal_mode` to replay and threshold builder**

Change `replay_candidates_for_sample` signature to:

```python
def replay_candidates_for_sample(
    graph: dict,
    raw_prediction: dict,
    classifier_bundle: object | None,
    strong_threshold: float,
    weak_threshold: float,
    min_confidence: float,
    proposal_mode: str = "same_instance",
) -> list[dict]:
```

Replace proposal construction with:

```python
same_instance_proposals = assemble_instance_candidates(
    graph,
    raw_prediction,
    strong_threshold=strong_threshold,
    weak_threshold=weak_threshold,
)

if proposal_mode == "hybrid":
    proposals = generate_hybrid_object_proposals(
        graph,
        raw_prediction,
        same_instance_proposals=same_instance_proposals,
    )
else:
    proposals = same_instance_proposals
```

Change `build_threshold_result` signature to include:

```python
proposal_mode: str = "same_instance",
```

Pass `proposal_mode=proposal_mode` into `replay_candidates_for_sample`.

In `build_report`, pass:

```python
proposal_mode=args.proposal_mode,
```

into `build_threshold_result`.

- [ ] **Step 6: Add CLI argument**

In `parse_args`, add:

```python
parser.add_argument("--proposal-mode", choices=("same_instance", "hybrid"), default="same_instance")
```

Add `"proposal_mode": args.proposal_mode` to the top-level report dict.

- [ ] **Step 7: Run focused test**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_object_proposal_recall_report.TestHybridProposalMode -v
```

Expected:

```text
OK
```

## 7. Task 5: Add Proposal Source Coverage to the Report

**Files:**

- Modify: `object_proposal_recall_report.py`
- Modify: `tests/test_object_proposal_recall_report.py`

- [ ] **Step 1: Add source coverage test**

Append this method to `TestDatasetLevelResultBuilder`:

```python
    @patch("object_proposal_recall_report.replay_candidates_for_sample")
    def test_build_threshold_result_reports_proposal_source_coverage(self, mock_replay):
        from object_proposal_recall_report import build_threshold_result

        mock_replay.return_value = [
            {
                "source": "hybrid_merged",
                "proposal_sources": ["same_instance", "raw_macro_component"],
                "proposal_macro": "rib",
                "classified_macro": "rib",
                "action": "accepted_remap",
                "face_indices": [1, 2, 3],
                "face_count": 3,
                "area": 100.0,
            }
        ]

        result = build_threshold_result(
            graph_by_sample={"part1": {"nodes": [], "edges": []}},
            raw_prediction_by_sample={"part1": {"faces": []}},
            truth_by_sample={"part1": [{"sample": "part1", "macro": "rib", "area": 100.0, "face_count": 3, "truth_index": 0}]},
            final_prediction_by_sample={"part1": {"faces": [{"idx": 1, "macro_feature": "rib"}, {"idx": 2, "macro_feature": "rib"}, {"idx": 3, "macro_feature": "rib"}]}},
            classifier_bundle=None,
            strong_threshold=0.3,
            weak_threshold=0.2,
            min_confidence=0.5,
            absolute_area_tolerance=10.0,
            relative_area_tolerance=0.02,
            proposal_mode="hybrid",
        )

        self.assertEqual(result["proposal_candidate_count_by_source"]["same_instance"], 1)
        self.assertEqual(result["proposal_candidate_count_by_source"]["raw_macro_component"], 1)
        self.assertEqual(result["proposal_matched_truth_count_by_source"]["same_instance"], 1)
        self.assertEqual(result["proposal_matched_truth_count_by_source"]["raw_macro_component"], 1)
```

- [ ] **Step 2: Run focused test and confirm failure**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_object_proposal_recall_report.TestDatasetLevelResultBuilder.test_build_threshold_result_reports_proposal_source_coverage -v
```

Expected:

```text
FAIL
```

because source coverage fields do not exist.

- [ ] **Step 3: Implement source coverage helper**

Add:

```python
def proposal_source_counts(candidates: list[dict]) -> dict[str, int]:
    counts: Counter = Counter()
    for candidate in candidates:
        for source in candidate.get("proposal_sources", [candidate.get("source", "unknown")]):
            counts[str(source)] += 1
    return dict(counts)


def matched_truth_count_by_source(
    truth_entries: list[dict],
    candidates: list[dict],
    absolute_area_tolerance: float,
    relative_area_tolerance: float,
) -> dict[str, int]:
    counts: Counter = Counter()
    for truth in truth_entries:
        matched_sources: set[str] = set()
        for candidate in candidates:
            if candidate_matches_truth(
                candidate,
                truth,
                "proposal",
                absolute_area_tolerance=absolute_area_tolerance,
                relative_area_tolerance=relative_area_tolerance,
            ):
                matched_sources.update(str(source) for source in candidate.get("proposal_sources", [candidate.get("source", "unknown")]))
        for source in matched_sources:
            counts[source] += 1
    return dict(counts)
```

Inside `build_threshold_result`, accumulate:

```python
candidate_count_by_source: Counter = Counter()
matched_count_by_source: Counter = Counter()
```

After candidates are created for each sample:

```python
candidate_count_by_source.update(proposal_source_counts(candidates))
matched_count_by_source.update(
    matched_truth_count_by_source(
        truth_entries,
        candidates,
        absolute_area_tolerance,
        relative_area_tolerance,
    )
)
```

Add fields to the returned threshold result:

```python
"proposal_candidate_count_by_source": dict(candidate_count_by_source),
"proposal_matched_truth_count_by_source": dict(matched_count_by_source),
```

Inside sample result, add:

```python
"proposal_candidate_count_by_source": proposal_source_counts(candidates),
"proposal_matched_truth_count_by_source": matched_truth_count_by_source(
    truth_entries,
    candidates,
    absolute_area_tolerance,
    relative_area_tolerance,
),
```

- [ ] **Step 4: Add source coverage to markdown**

In `write_markdown`, after failure-stage counts, add:

```python
lines.append("## Proposal Source Coverage")
lines.append("")
lines.append("| Source | Candidate Count | Matched Truth Count |")
lines.append("|--------|-----------------|---------------------|")
source_candidates = bp.get("proposal_candidate_count_by_source", {})
source_matches = bp.get("proposal_matched_truth_count_by_source", {})
for source in sorted(set(source_candidates) | set(source_matches)):
    lines.append(f"| {source} | {source_candidates.get(source, 0)} | {source_matches.get(source, 0)} |")
lines.append("")
lines.append("Source matched counts are non-exclusive because one deduplicated candidate can have multiple proposal sources.")
lines.append("")
```

- [ ] **Step 5: Run focused test**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_object_proposal_recall_report.TestDatasetLevelResultBuilder.test_build_threshold_result_reports_proposal_source_coverage -v
```

Expected:

```text
OK
```

## 8. Task 6: Run v5.12 Hybrid Proposal Diagnostic

**Files:**

- Use existing raw predictions: `output_real_step_s3_v5_11_proposal_diagnostics/raw_predictions`
- Create reports under: `output_real_step_s3_v5_12_hybrid_proposals/reports`

- [ ] **Step 1: Run full tests before report generation**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_object_candidate_proposer tests.test_object_proposal_recall_report tests.test_instance_candidate_postprocess tests.test_prediction_to_feature_objects tests.test_run_gnn_inference_edge_aware tests.test_selectable_regions -v
```

Expected:

```text
OK
```

- [ ] **Step 2: Generate v5.12 hybrid diagnostic report**

Run:

```powershell
W:\python\envs\feagpt\python.exe object_proposal_recall_report.py `
  --proposal-mode hybrid `
  --graph-dir output_real_step_s3_v5_6_b_rib_boundary\graphs `
  --raw-prediction-dir output_real_step_s3_v5_11_proposal_diagnostics\raw_predictions `
  --truth-json configs\test_part_s3_area_truth_part20_mm2.json `
  --classifier output_generation_v5_10_instance_candidate_dataset_v55\model_runs\random_forest_instance_candidate_v5_10_v55\model.joblib `
  --final-prediction-dir output_real_step_s3_v5_10_instance_plus_v59\predictions `
  --output-json output_real_step_s3_v5_12_hybrid_proposals\reports\proposal_recall_s3_hybrid.json `
  --output-md output_real_step_s3_v5_12_hybrid_proposals\reports\proposal_recall_s3_hybrid.md `
  --strong-thresholds 0.30,0.40,0.50,0.55,0.60,0.70 `
  --weak-thresholds 0.20,0.30,0.35,0.40,0.50 `
  --min-confidence 0.50 `
  --absolute-area-tolerance 10.0 `
  --relative-area-tolerance 0.02
```

- [ ] **Step 3: Verify report fields**

Run:

```powershell
@'
import json
from pathlib import Path
p = Path("output_real_step_s3_v5_12_hybrid_proposals/reports/proposal_recall_s3_hybrid.json")
data = json.loads(p.read_text(encoding="utf-8"))
bp = data["best_by_proposal_recall"]
print("proposal_mode", data.get("proposal_mode"))
print("requested", data.get("requested_threshold_pair_count"))
print("evaluated", data.get("evaluated_threshold_pair_count"))
print("skipped", len(data.get("skipped_threshold_pairs", [])))
print("proposal_recall", bp["proposal_stage"]["recall"])
print("final_survival_recall", bp["final_survival_stage"]["recall"])
print("failure_stage_counts", bp["failure_stage_counts"])
print("candidate_count_by_source", bp.get("proposal_candidate_count_by_source"))
print("matched_truth_by_source", bp.get("proposal_matched_truth_count_by_source"))
'@ | W:\python\envs\feagpt\python.exe -
```

Expected:

```text
proposal_mode hybrid
requested 30
evaluated 23
skipped 7
```

No fixed recall value is required in this step because it depends on the new proposer. The report must contain non-empty source coverage counts for at least `same_instance` and `raw_macro_component`. If topology rules produce no candidates, still keep the `topology_rule` source implementation and report that its count is zero.

## 9. Task 7: Write v5.11.1 vs v5.12 Comparison and Decision

**Files:**

- Create: `output_real_step_s3_v5_12_hybrid_proposals/reports/v5_11_1_vs_v5_12_hybrid_comparison.md`
- Create: `output_real_step_s3_v5_12_hybrid_proposals/reports/v5_12_hybrid_decision.md`

- [ ] **Step 1: Generate comparison and decision reports**

Run:

```powershell
@'
import json
from pathlib import Path

old_path = Path("output_real_step_s3_v5_11_1_macrofix_diagnostics/reports/proposal_recall_s3_macrofix.json")
new_path = Path("output_real_step_s3_v5_12_hybrid_proposals/reports/proposal_recall_s3_hybrid.json")
comparison_path = Path("output_real_step_s3_v5_12_hybrid_proposals/reports/v5_11_1_vs_v5_12_hybrid_comparison.md")
decision_path = Path("output_real_step_s3_v5_12_hybrid_proposals/reports/v5_12_hybrid_decision.md")

old = json.loads(old_path.read_text(encoding="utf-8"))
new = json.loads(new_path.read_text(encoding="utf-8"))

old_bp = old["best_by_proposal_recall"]
new_bp = new["best_by_proposal_recall"]

old_prop = old_bp["proposal_stage"]["recall"]
new_prop = new_bp["proposal_stage"]["recall"]
old_final = old_bp["final_survival_stage"]["recall"]
new_final = new_bp["final_survival_stage"]["recall"]
old_failure = old_bp["failure_stage_counts"]
new_failure = new_bp["failure_stage_counts"]

source_candidates = new_bp.get("proposal_candidate_count_by_source", {})
source_matches = new_bp.get("proposal_matched_truth_count_by_source", {})

comparison_lines = [
    "# v5.11.1 vs v5.12 Hybrid Proposal Comparison",
    "",
    "## Summary",
    "",
    "| Metric | v5.11.1 macrofix | v5.12 hybrid |",
    "|--------|------------------|--------------|",
    f"| Proposal recall | {old_prop:.4f} | {new_prop:.4f} |",
    f"| Final-survival recall | {old_final:.4f} | {new_final:.4f} |",
    f"| No compatible proposal | {old_failure.get('no_area_compatible_proposal', 0)} | {new_failure.get('no_area_compatible_proposal', 0)} |",
    f"| Classifier wrong/rejected | {old_failure.get('proposal_exists_classifier_wrong_or_rejected', 0)} | {new_failure.get('proposal_exists_classifier_wrong_or_rejected', 0)} |",
    f"| Accepted but final changes/removes | {old_failure.get('classifier_accepts_but_final_changes_or_removes', 0)} | {new_failure.get('classifier_accepts_but_final_changes_or_removes', 0)} |",
    f"| Survives final | {old_failure.get('survives_final_prediction', 0)} | {new_failure.get('survives_final_prediction', 0)} |",
    "",
    "## Proposal Source Coverage",
    "",
    "| Source | Candidate Count | Matched Truth Count |",
    "|--------|-----------------|---------------------|",
]
for source in sorted(set(source_candidates) | set(source_matches)):
    comparison_lines.append(f"| {source} | {source_candidates.get(source, 0)} | {source_matches.get(source, 0)} |")

comparison_lines.extend([
    "",
    "Source matched counts are non-exclusive.",
    "",
])

comparison_path.parent.mkdir(parents=True, exist_ok=True)
comparison_path.write_text("\\n".join(comparison_lines), encoding="utf-8")

if new_prop < 0.50:
    recommendation = "raw_gnn_or_data_distribution_repair"
    reason = "Hybrid proposals still fail to recover area-truth objects. The missing objects are probably not visible in raw macro predictions or current topology finders."
elif new_prop < 0.65:
    recommendation = "proposal_source_expansion"
    reason = "Hybrid proposals improve recall but remain below the current v5.9-C real-part recall level. Add targeted topology proposal rules for the remaining missed macros."
elif new_bp["accepted_stage"]["recall"] < new_prop * 0.80:
    recommendation = "hybrid_candidate_classifier_retraining"
    reason = "Hybrid proposal recall is adequate, but accepted-stage recall is much lower. The classifier is rejecting or misclassifying new-source candidates."
else:
    recommendation = "hybrid_pipeline_integration_trial"
    reason = "Hybrid proposal recall and accepted-stage recall are both strong enough to justify an inference-pipeline integration trial."

decision_lines = [
    "# v5.12 Hybrid Proposal Decision",
    "",
    "## Recommendation",
    "",
    f"**{recommendation}**",
    "",
    "## Reasoning",
    "",
    reason,
    "",
    "## Supporting Data",
    "",
    f"- v5.11.1 proposal recall: {old_prop:.4f}",
    f"- v5.12 proposal recall: {new_prop:.4f}",
    f"- v5.12 accepted-stage recall: {new_bp['accepted_stage']['recall']:.4f}",
    f"- v5.12 final-survival recall: {new_final:.4f}",
    f"- v5.12 no compatible proposal: {new_failure.get('no_area_compatible_proposal', 0)}",
    f"- v5.12 classifier wrong/rejected: {new_failure.get('proposal_exists_classifier_wrong_or_rejected', 0)}",
    f"- v5.12 accepted but final changes/removes: {new_failure.get('classifier_accepts_but_final_changes_or_removes', 0)}",
    "",
]
decision_path.write_text("\\n".join(decision_lines), encoding="utf-8")

print(comparison_path)
print(decision_path)
'@ | W:\python\envs\feagpt\python.exe -
```

- [ ] **Step 2: Inspect generated decision**

Run:

```powershell
Get-Content output_real_step_s3_v5_12_hybrid_proposals\reports\v5_12_hybrid_decision.md
```

Expected:

- Exactly one recommendation appears in bold.
- It is one of:
  - `raw_gnn_or_data_distribution_repair`
  - `proposal_source_expansion`
  - `hybrid_candidate_classifier_retraining`
  - `hybrid_pipeline_integration_trial`

## 10. Task 8: Final Verification

**Files:**

- Verify all changed files and generated reports.

- [ ] **Step 1: Run full test suite**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_object_candidate_proposer tests.test_object_proposal_recall_report tests.test_instance_candidate_postprocess tests.test_prediction_to_feature_objects tests.test_run_gnn_inference_edge_aware tests.test_selectable_regions -v
```

Expected:

```text
OK
```

- [ ] **Step 2: Run compile check**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m py_compile utils\object_candidate_proposer.py object_proposal_recall_report.py tests\test_object_candidate_proposer.py tests\test_object_proposal_recall_report.py
```

Expected:

```text
no output
```

- [ ] **Step 3: Verify report files**

Run:

```powershell
Test-Path output_real_step_s3_v5_12_hybrid_proposals\reports\proposal_recall_s3_hybrid.json
Test-Path output_real_step_s3_v5_12_hybrid_proposals\reports\proposal_recall_s3_hybrid.md
Test-Path output_real_step_s3_v5_12_hybrid_proposals\reports\v5_12_hybrid_decision.md
Test-Path output_real_step_s3_v5_12_hybrid_proposals\reports\v5_11_1_vs_v5_12_hybrid_comparison.md
```

Expected:

```text
True
True
True
True
```

- [ ] **Step 4: Verify v5.11.1 default compatibility**

Run:

```powershell
W:\python\envs\feagpt\python.exe object_proposal_recall_report.py `
  --proposal-mode same_instance `
  --graph-dir output_real_step_s3_v5_6_b_rib_boundary\graphs `
  --raw-prediction-dir output_real_step_s3_v5_11_proposal_diagnostics\raw_predictions `
  --truth-json configs\test_part_s3_area_truth_part20_mm2.json `
  --classifier output_generation_v5_10_instance_candidate_dataset_v55\model_runs\random_forest_instance_candidate_v5_10_v55\model.joblib `
  --final-prediction-dir output_real_step_s3_v5_10_instance_plus_v59\predictions `
  --output-json output_real_step_s3_v5_12_hybrid_proposals\reports\proposal_recall_s3_same_instance_compat.json `
  --output-md output_real_step_s3_v5_12_hybrid_proposals\reports\proposal_recall_s3_same_instance_compat.md `
  --strong-thresholds 0.30 `
  --weak-thresholds 0.20 `
  --min-confidence 0.50 `
  --absolute-area-tolerance 10.0 `
  --relative-area-tolerance 0.02
```

Then run:

```powershell
@'
import json
from pathlib import Path
p = Path("output_real_step_s3_v5_12_hybrid_proposals/reports/proposal_recall_s3_same_instance_compat.json")
data = json.loads(p.read_text(encoding="utf-8"))
bp = data["best_by_proposal_recall"]
print("proposal_mode", data.get("proposal_mode"))
print("proposal_recall", round(bp["proposal_stage"]["recall"], 4))
'@ | W:\python\envs\feagpt\python.exe -
```

Expected:

```text
proposal_mode same_instance
proposal_recall 0.3775
```

This confirms the new CLI mode did not break the previous evaluator behavior.

## 11. Required Final Response From Implementing Agent

The implementing agent must report:

1. Tests run and whether they passed.
2. Whether `--proposal-mode same_instance` still reproduces proposal recall `0.3775`.
3. v5.12 hybrid proposal recall.
4. v5.12 hybrid accepted-stage recall.
5. v5.12 hybrid final-survival recall.
6. v5.12 failure-stage counts.
7. Proposal source coverage:
   - candidate count by source;
   - matched truth count by source.
8. The single recommendation from `v5_12_hybrid_decision.md`.
9. Paths to:
   - `proposal_recall_s3_hybrid.json`
   - `proposal_recall_s3_hybrid.md`
   - `v5_12_hybrid_decision.md`
   - `v5_11_1_vs_v5_12_hybrid_comparison.md`
10. Any deviation from this plan.

## 12. Review Checklist for Codex

After implementation, review these items:

1. `utils/object_candidate_proposer.py` exists and has no dependency on model training.
2. `object_proposal_recall_report.py` supports `--proposal-mode same_instance` and `--proposal-mode hybrid`.
3. Same-instance compatibility report reproduces v5.11.1 proposal recall `0.3775`.
4. Hybrid report includes `proposal_candidate_count_by_source`.
5. Hybrid report includes `proposal_matched_truth_count_by_source`.
6. Source coverage is non-exclusive and documented as non-exclusive.
7. No GNN/model/generator files were changed.
8. If v5.12 proposal recall is below `0.50`, do not continue with candidate classifier retraining; inspect raw GNN/data distribution.
9. If v5.12 proposal recall is at least `0.65` but accepted-stage recall is much lower, next work is candidate classifier retraining for hybrid proposal sources.
10. If v5.12 proposal recall improves but remains between `0.50` and `0.65`, next work is targeted topology proposal expansion for the largest missed macros.

The next technical decision must be based on `v5_12_hybrid_decision.md`, not on subjective inspection.
