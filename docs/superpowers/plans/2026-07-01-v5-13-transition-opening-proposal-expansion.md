# v5.13 Transition and Opening Proposal Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve v5.12 hybrid proposal recall by adding the omitted transition proposal finders and a bounded through-hole opening fallback, then regenerate s3 diagnostics with source-by-macro evidence.

**Architecture:** Keep v5.12 hybrid proposer as a diagnostic branch. Extend `utils/object_candidate_proposer.py` with two low-risk proposal sources: existing fillet/chamfer topology finders and a conservative single-cylinder opening fallback. Extend `object_proposal_recall_report.py` only for clearer report labeling and source-by-macro reporting. Do not modify GNN architecture, model training, data generation, model checkpoints, or default inference behavior.

**Tech Stack:** Python 3 from `W:/python/envs/feagpt/python.exe`, `unittest`, existing graph/prediction JSON, existing `utils.topology_postprocess` candidate finders.

---

## 0. Current Review Result

v5.12 implementation satisfies the previous plan:

```text
tests: 35 OK
same_instance compatibility proposal recall: 0.3775
v5.12 hybrid proposal recall: 0.6225
v5.12 hybrid accepted-stage recall: 0.4755
v5.12 hybrid final-survival recall: 0.5147
```

v5.12 source coverage:

```text
same_instance        candidates 281, matched truth 77
raw_macro_component  candidates 219, matched truth 73
topology_rule        candidates 183, matched truth 107
```

v5.12 remaining failures:

```text
no_area_compatible_proposal                       77
proposal_exists_classifier_wrong_or_rejected      21
classifier_accepts_but_final_changes_or_removes    1
survives_final_prediction                        105
```

Dominant no-proposal misses by macro:

```text
rib                 23
fillet              22
through_hole_round  14
chamfer              4
counterbore_hole     3
```

Review finding:

- `utils.topology_postprocess.py` already has `find_fillet_stock_candidates` and `find_chamfer_stock_candidates`.
- v5.12 `utils/object_candidate_proposer.py` did not import or call those two finders.
- Therefore many `fillet` and `chamfer` no-proposal failures may be caused by incomplete hybrid proposal wiring, not by model failure.

Decision:

- v5.13 should first fix omitted transition proposals and add a bounded through-hole fallback.
- Do not start candidate-classifier retraining yet.
- Do not start GNN architecture changes yet.
- Do not write aggressive rib rules yet; rib remains a likely v5.14 focused task after this lower-risk correction.

## 1. Scope Control

### Files to Modify

- `utils/object_candidate_proposer.py`
- `tests/test_object_candidate_proposer.py`
- `object_proposal_recall_report.py`
- `tests/test_object_proposal_recall_report.py`

### Files to Create

- `output_real_step_s3_v5_13_transition_opening_proposals/reports/proposal_recall_s3_v5_13.json`
- `output_real_step_s3_v5_13_transition_opening_proposals/reports/proposal_recall_s3_v5_13.md`
- `output_real_step_s3_v5_13_transition_opening_proposals/reports/v5_13_decision.md`
- `output_real_step_s3_v5_13_transition_opening_proposals/reports/v5_12_vs_v5_13_comparison.md`

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

## 2. Task 1: Fix Report Title for Hybrid Mode

**Files:**

- Modify: `tests/test_object_proposal_recall_report.py`
- Modify: `object_proposal_recall_report.py`

- [ ] **Step 1: Add title regression test**

Append this test class before the final `if __name__ == "__main__":` block in `tests/test_object_proposal_recall_report.py`:

```python
class TestMarkdownReportModeTitle(unittest.TestCase):
    def test_write_markdown_title_mentions_hybrid_mode(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from object_proposal_recall_report import write_markdown

        report = {
            "proposal_mode": "hybrid",
            "graph_dir": "graphs",
            "raw_prediction_dir": "raw",
            "truth_json": "truth.json",
            "classifier": "model.joblib",
            "final_prediction_dir": "predictions",
            "requested_threshold_pair_count": 30,
            "evaluated_threshold_pair_count": 23,
            "skipped_threshold_pairs": [{"strong_threshold": 0.3, "weak_threshold": 0.3}],
            "absolute_area_tolerance": 10.0,
            "relative_area_tolerance": 0.02,
            "threshold_results": [],
            "best_by_proposal_recall": {
                "strong_threshold": 0.3,
                "weak_threshold": 0.2,
                "proposal_stage": {"recall": 0.5, "truth_by_macro": {}, "matched_by_macro": {}},
                "final_survival_stage": {"recall": 0.4, "truth_by_macro": {}, "matched_by_macro": {}},
                "failure_stage_counts": {},
                "samples": [],
            },
            "best_by_final_survival_recall": {
                "strong_threshold": 0.3,
                "weak_threshold": 0.2,
                "proposal_stage": {"recall": 0.5, "truth_by_macro": {}, "matched_by_macro": {}},
                "final_survival_stage": {"recall": 0.4, "truth_by_macro": {}, "matched_by_macro": {}},
                "failure_stage_counts": {},
                "samples": [],
            },
        }

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.md"
            write_markdown(path, report)
            text = path.read_text(encoding="utf-8")

        self.assertTrue(text.startswith("# v5.13 Object Proposal Recall Diagnostic Report"))
        self.assertIn("- Proposal mode: `hybrid`", text)
```

- [ ] **Step 2: Run focused test and confirm failure**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_object_proposal_recall_report.TestMarkdownReportModeTitle -v
```

Expected before implementation:

```text
FAIL
```

because the current markdown title still says `v5.11.1 Object Proposal Recall Diagnostic Report (Macrofix)`.

- [ ] **Step 3: Update markdown title and inputs**

In `object_proposal_recall_report.py`, inside `write_markdown`, replace the current first title line with:

```python
proposal_mode = str(report.get("proposal_mode", "same_instance"))
lines.append("# v5.13 Object Proposal Recall Diagnostic Report")
```

In the `## Inputs` section, after graph/raw/truth/classifier/final prediction lines, add:

```python
lines.append(f"- Proposal mode: `{proposal_mode}`")
```

- [ ] **Step 4: Run focused test**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_object_proposal_recall_report.TestMarkdownReportModeTitle -v
```

Expected:

```text
OK
```

## 3. Task 2: Wire Omitted Fillet and Chamfer Topology Finders

**Files:**

- Modify: `utils/object_candidate_proposer.py`
- Modify: `tests/test_object_candidate_proposer.py`

- [ ] **Step 1: Add transition finder test**

Append this method inside `ObjectCandidateProposerTest` in `tests/test_object_candidate_proposer.py`:

```python
    def test_topology_rule_proposals_include_fillet_and_chamfer_finders(self):
        from utils.object_candidate_proposer import topology_rule_proposals

        graph = {
            "nodes": [
                {"idx": 0, "primitive_surface": "torus", "attributes": {"physical": {"area": 12.0}}},
                {"idx": 1, "primitive_surface": "plane", "attributes": {"physical": {"area": 4.0}}},
            ],
            "edges": [],
        }
        faces = [
            {"idx": 0, "macro_feature": "stock", "area": 12.0, "primitive_surface": "torus"},
            {"idx": 1, "macro_feature": "stock", "area": 4.0, "primitive_surface": "plane"},
        ]

        with patch("utils.object_candidate_proposer.find_through_hole_opening_candidates", return_value=[]), \
             patch("utils.object_candidate_proposer.find_blind_hole_candidates", return_value=[]), \
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
             patch("utils.object_candidate_proposer.find_existing_rib_transition_absorption_candidates", return_value=[]), \
             patch("utils.object_candidate_proposer.find_fillet_stock_candidates", return_value=[[0]]), \
             patch("utils.object_candidate_proposer.find_chamfer_stock_candidates", return_value=[[1]]):
            proposals = topology_rule_proposals(graph, faces)

        self.assertEqual(
            [(p["proposal_macro"], p["face_indices"], p["assembly_reasons"]) for p in proposals],
            [
                ("fillet", [0], ["topology_fillet_stock_transition"]),
                ("chamfer", [1], ["topology_chamfer_stock_transition"]),
            ],
        )
```

- [ ] **Step 2: Run focused test and confirm failure**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_object_candidate_proposer.ObjectCandidateProposerTest.test_topology_rule_proposals_include_fillet_and_chamfer_finders -v
```

Expected before implementation:

```text
ERROR
```

because `find_fillet_stock_candidates` and `find_chamfer_stock_candidates` are not imported by `utils.object_candidate_proposer`.

- [ ] **Step 3: Import omitted finders**

In `utils/object_candidate_proposer.py`, add these names to the existing `from utils.topology_postprocess import (...)` block:

```python
    find_chamfer_stock_candidates,
    find_fillet_stock_candidates,
```

- [ ] **Step 4: Add omitted finders to `topology_rule_proposals`**

Inside `topology_rule_proposals`, append these two proposal specs after the rib specs:

```python
        ("fillet", "topology_fillet_stock_transition", find_fillet_stock_candidates(faces, graph, repaired)),
        ("chamfer", "topology_chamfer_stock_transition", find_chamfer_stock_candidates(faces, graph, repaired)),
```

- [ ] **Step 5: Run focused test**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_object_candidate_proposer.ObjectCandidateProposerTest.test_topology_rule_proposals_include_fillet_and_chamfer_finders -v
```

Expected:

```text
OK
```

## 4. Task 3: Add Conservative Through-Hole Opening Fallback Proposals

**Files:**

- Modify: `utils/object_candidate_proposer.py`
- Modify: `tests/test_object_candidate_proposer.py`

- [ ] **Step 1: Add fallback opening-cylinder test**

Append this method inside `ObjectCandidateProposerTest`:

```python
    def test_opening_cylinder_fallback_proposes_stock_cylinder_with_two_circular_plane_boundaries(self):
        from utils.object_candidate_proposer import opening_cylinder_fallback_proposals

        graph = {
            "nodes": [
                {"idx": 0, "primitive_surface": "cylinder", "attributes": {"physical": {"area": 188.5}}},
                {"idx": 1, "primitive_surface": "plane", "attributes": {"physical": {"area": 1000.0}}},
                {"idx": 2, "primitive_surface": "plane", "attributes": {"physical": {"area": 1000.0}}},
                {"idx": 3, "primitive_surface": "cylinder", "attributes": {"physical": {"area": 900.0}}},
            ],
            "edges": [[0, 1], [0, 2], [3, 1]],
            "edge_attributes": [
                {"curve_type": "circle", "convexity": "convex"},
                {"curve_type": "circle", "convexity": "convex"},
                {"curve_type": "line", "convexity": "convex"},
            ],
        }
        faces = [
            {"idx": 0, "macro_feature": "stock", "area": 188.5, "primitive_surface": "cylinder"},
            {"idx": 1, "macro_feature": "stock", "area": 1000.0, "primitive_surface": "plane"},
            {"idx": 2, "macro_feature": "stock", "area": 1000.0, "primitive_surface": "plane"},
            {"idx": 3, "macro_feature": "stock", "area": 900.0, "primitive_surface": "cylinder"},
        ]

        proposals = opening_cylinder_fallback_proposals(graph, faces)

        self.assertEqual(
            [(p["proposal_macro"], p["face_indices"], p["assembly_reasons"]) for p in proposals],
            [("through_hole_round", [0], ["fallback_two_circular_plane_boundaries"])],
        )
```

- [ ] **Step 2: Run focused test and confirm failure**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_object_candidate_proposer.ObjectCandidateProposerTest.test_opening_cylinder_fallback_proposes_stock_cylinder_with_two_circular_plane_boundaries -v
```

Expected before implementation:

```text
ERROR
```

because `opening_cylinder_fallback_proposals` does not exist yet.

- [ ] **Step 3: Add edge-attribute helpers**

In `utils/object_candidate_proposer.py`, after `extract_graph_edges`, add:

```python
def edge_key(src: int, dst: int) -> tuple[int, int]:
    left, right = sorted((int(src), int(dst)))
    return (left, right)


def edge_attributes_by_key(graph: dict[str, Any]) -> dict[tuple[int, int], list[dict[str, Any]]]:
    edges = extract_graph_edges(graph)
    attributes = graph.get("edge_attributes", [])
    result: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for position, edge in enumerate(edges):
        attr = attributes[position] if position < len(attributes) and isinstance(attributes[position], dict) else {}
        result[edge_key(*edge)].append(attr)
    return result


def is_circle_edge(attr: dict[str, Any]) -> bool:
    curve_type = str(attr.get("curve_type", "")).lower()
    return curve_type == "circle" or attr.get("is_circle") == 1.0 or attr.get("curve_type_id") == 2


def is_line_edge(attr: dict[str, Any]) -> bool:
    curve_type = str(attr.get("curve_type", "")).lower()
    return curve_type == "line" or attr.get("is_line") == 1.0 or attr.get("curve_type_id") == 1
```

- [ ] **Step 4: Add fallback through-hole proposal function**

In `utils/object_candidate_proposer.py`, after `topology_rule_proposals`, add:

```python
def opening_cylinder_fallback_proposals(
    graph: dict[str, Any],
    faces: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    nodes = graph_nodes_by_idx(graph)
    face_by_idx = {int(face["idx"]): face for face in faces}
    attrs_by_key = edge_attributes_by_key(graph)
    circular_plane_neighbors: dict[int, set[int]] = defaultdict(set)
    line_edge_count: dict[int, int] = defaultdict(int)

    for src, dst in extract_graph_edges(graph):
        attrs = attrs_by_key.get(edge_key(src, dst), [{}])
        src_node = nodes.get(src, {})
        dst_node = nodes.get(dst, {})
        for attr in attrs:
            if is_circle_edge(attr):
                if src_node.get("primitive_surface") == "cylinder" and dst_node.get("primitive_surface") == "plane":
                    circular_plane_neighbors[src].add(dst)
                if dst_node.get("primitive_surface") == "cylinder" and src_node.get("primitive_surface") == "plane":
                    circular_plane_neighbors[dst].add(src)
            if is_line_edge(attr):
                line_edge_count[src] += 1
                line_edge_count[dst] += 1

    proposals: list[dict[str, Any]] = []
    for idx, node in sorted(nodes.items()):
        if node.get("primitive_surface") != "cylinder":
            continue
        face = face_by_idx.get(idx, {})
        macro = normalize_macro(face.get("macro_feature", ""))
        if macro in {"fillet", "slot_o", "slot_u", "cylinder_boss"}:
            continue
        if node_area(node) <= 1e-9:
            continue
        if len(circular_plane_neighbors.get(idx, set())) < 2:
            continue
        if line_edge_count.get(idx, 0) > 2:
            continue
        proposals.append(candidate_record("fallback_rule", "through_hole_round", [idx], "fallback_two_circular_plane_boundaries"))
    return proposals
```

- [ ] **Step 5: Add fallback proposals to hybrid generator**

Inside `generate_hybrid_object_proposals`, after:

```python
proposals.extend(topology_rule_proposals(graph, faces))
```

add:

```python
proposals.extend(opening_cylinder_fallback_proposals(graph, faces))
```

- [ ] **Step 6: Run focused test**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_object_candidate_proposer.ObjectCandidateProposerTest.test_opening_cylinder_fallback_proposes_stock_cylinder_with_two_circular_plane_boundaries -v
```

Expected:

```text
OK
```

## 5. Task 4: Add Source-by-Macro Coverage to Diagnostic JSON and Markdown

**Files:**

- Modify: `object_proposal_recall_report.py`
- Modify: `tests/test_object_proposal_recall_report.py`

- [ ] **Step 1: Add source-by-macro regression test**

Append this method to `TestDatasetLevelResultBuilder`:

```python
    @patch("object_proposal_recall_report.replay_candidates_for_sample")
    def test_build_threshold_result_reports_proposal_source_coverage_by_macro(self, mock_replay):
        from object_proposal_recall_report import build_threshold_result

        mock_replay.return_value = [
            {
                "source": "fallback_rule",
                "proposal_sources": ["fallback_rule"],
                "proposal_macro": "through_hole_round",
                "classified_macro": "through_hole_round",
                "action": "accepted_remap",
                "face_indices": [7],
                "face_count": 1,
                "area": 200.0,
            }
        ]

        result = build_threshold_result(
            graph_by_sample={"part1": {"nodes": [], "edges": []}},
            raw_prediction_by_sample={"part1": {"faces": []}},
            truth_by_sample={"part1": [{"sample": "part1", "macro": "through_hole_round", "area": 200.0, "face_count": 1, "truth_index": 0}]},
            final_prediction_by_sample={"part1": {"faces": [{"idx": 7, "macro_feature": "through_hole_round"}]}},
            classifier_bundle=None,
            strong_threshold=0.3,
            weak_threshold=0.2,
            min_confidence=0.5,
            absolute_area_tolerance=10.0,
            relative_area_tolerance=0.02,
            proposal_mode="hybrid",
        )

        self.assertEqual(result["proposal_candidate_count_by_source_macro"]["fallback_rule"]["through_hole_round"], 1)
        self.assertEqual(result["proposal_matched_truth_count_by_source_macro"]["fallback_rule"]["through_hole_round"], 1)
```

- [ ] **Step 2: Run focused test and confirm failure**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_object_proposal_recall_report.TestDatasetLevelResultBuilder.test_build_threshold_result_reports_proposal_source_coverage_by_macro -v
```

Expected before implementation:

```text
FAIL
```

because source-by-macro fields do not exist.

- [ ] **Step 3: Add source-by-macro helper functions**

In `object_proposal_recall_report.py`, after `proposal_source_counts`, add:

```python
def proposal_source_macro_counts(candidates: list[dict]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for candidate in candidates:
        macro = candidate_macro_for_stage(candidate, "proposal") or "unknown"
        for source in candidate.get("proposal_sources", [candidate.get("source", "unknown")]):
            counts[str(source)][macro] += 1
    return {source: dict(counter) for source, counter in counts.items()}


def matched_truth_count_by_source_macro(
    truth_entries: list[dict],
    candidates: list[dict],
    absolute_area_tolerance: float,
    relative_area_tolerance: float,
) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter] = defaultdict(Counter)
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
            counts[source][str(truth.get("macro", "unknown"))] += 1
    return {source: dict(counter) for source, counter in counts.items()}
```

- [ ] **Step 4: Accumulate source-by-macro in `build_threshold_result`**

Inside `build_threshold_result`, initialize:

```python
candidate_count_by_source_macro: dict[str, Counter] = defaultdict(Counter)
matched_count_by_source_macro: dict[str, Counter] = defaultdict(Counter)
```

After candidates are created for each sample, add:

```python
for source, macro_counts in proposal_source_macro_counts(candidates).items():
    candidate_count_by_source_macro[source].update(macro_counts)
for source, macro_counts in matched_truth_count_by_source_macro(
    truth_entries,
    candidates,
    absolute_area_tolerance,
    relative_area_tolerance,
).items():
    matched_count_by_source_macro[source].update(macro_counts)
```

Add these fields to each `sample_result`:

```python
"proposal_candidate_count_by_source_macro": proposal_source_macro_counts(candidates),
"proposal_matched_truth_count_by_source_macro": matched_truth_count_by_source_macro(
    truth_entries,
    candidates,
    absolute_area_tolerance,
    relative_area_tolerance,
),
```

Add these fields to the returned threshold result:

```python
"proposal_candidate_count_by_source_macro": {source: dict(counter) for source, counter in candidate_count_by_source_macro.items()},
"proposal_matched_truth_count_by_source_macro": {source: dict(counter) for source, counter in matched_count_by_source_macro.items()},
```

- [ ] **Step 5: Add source-by-macro markdown table**

In `write_markdown`, after the `Proposal Source Coverage` table, add:

```python
    source_macro_candidates = best_proposal.get("proposal_candidate_count_by_source_macro", {})
    source_macro_matches = best_proposal.get("proposal_matched_truth_count_by_source_macro", {})
    if source_macro_candidates or source_macro_matches:
        lines.append("### Proposal Source Coverage by Macro")
        lines.append("")
        lines.append("| Source | Macro | Candidate Count | Matched Truth Count |")
        lines.append("|--------|-------|-----------------|---------------------|")
        all_sources = sorted(set(source_macro_candidates) | set(source_macro_matches))
        for source in all_sources:
            macros = sorted(set(source_macro_candidates.get(source, {})) | set(source_macro_matches.get(source, {})))
            for macro in macros:
                lines.append(
                    f"| {source} | {macro} | {source_macro_candidates.get(source, {}).get(macro, 0)} | "
                    f"{source_macro_matches.get(source, {}).get(macro, 0)} |"
                )
        lines.append("")
```

- [ ] **Step 6: Run focused test**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_object_proposal_recall_report.TestDatasetLevelResultBuilder.test_build_threshold_result_reports_proposal_source_coverage_by_macro -v
```

Expected:

```text
OK
```

## 6. Task 5: Run v5.13 Diagnostic

**Files:**

- Use existing raw predictions: `output_real_step_s3_v5_11_proposal_diagnostics/raw_predictions`
- Create reports under: `output_real_step_s3_v5_13_transition_opening_proposals/reports`

- [ ] **Step 1: Run full tests**

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

- [ ] **Step 3: Generate v5.13 hybrid report**

Run:

```powershell
W:\python\envs\feagpt\python.exe object_proposal_recall_report.py `
  --proposal-mode hybrid `
  --graph-dir output_real_step_s3_v5_6_b_rib_boundary\graphs `
  --raw-prediction-dir output_real_step_s3_v5_11_proposal_diagnostics\raw_predictions `
  --truth-json configs\test_part_s3_area_truth_part20_mm2.json `
  --classifier output_generation_v5_10_instance_candidate_dataset_v55\model_runs\random_forest_instance_candidate_v5_10_v55\model.joblib `
  --final-prediction-dir output_real_step_s3_v5_10_instance_plus_v59\predictions `
  --output-json output_real_step_s3_v5_13_transition_opening_proposals\reports\proposal_recall_s3_v5_13.json `
  --output-md output_real_step_s3_v5_13_transition_opening_proposals\reports\proposal_recall_s3_v5_13.md `
  --strong-thresholds 0.30,0.40,0.50,0.55,0.60,0.70 `
  --weak-thresholds 0.20,0.30,0.35,0.40,0.50 `
  --min-confidence 0.50 `
  --absolute-area-tolerance 10.0 `
  --relative-area-tolerance 0.02
```

- [ ] **Step 4: Verify report fields**

Run:

```powershell
@'
import json
from pathlib import Path
p = Path("output_real_step_s3_v5_13_transition_opening_proposals/reports/proposal_recall_s3_v5_13.json")
data = json.loads(p.read_text(encoding="utf-8"))
bp = data["best_by_proposal_recall"]
print("proposal_mode", data.get("proposal_mode"))
print("proposal_recall", bp["proposal_stage"]["recall"])
print("accepted_recall", bp["accepted_stage"]["recall"])
print("final_survival_recall", bp["final_survival_stage"]["recall"])
print("failure_stage_counts", bp["failure_stage_counts"])
print("source_candidates", bp.get("proposal_candidate_count_by_source"))
print("source_matches", bp.get("proposal_matched_truth_count_by_source"))
print("source_macro_has_fallback", "fallback_rule" in bp.get("proposal_candidate_count_by_source_macro", {}))
'@ | W:\python\envs\feagpt\python.exe -
```

Expected:

```text
proposal_mode hybrid
source_macro_has_fallback True
```

The v5.13 proposal recall should be compared against v5.12 proposal recall `0.6225`. Do not assume it must improve; if the fallback creates no area-compatible candidates, report that fact.

## 7. Task 6: Write v5.12 vs v5.13 Comparison and Decision

**Files:**

- Create: `output_real_step_s3_v5_13_transition_opening_proposals/reports/v5_12_vs_v5_13_comparison.md`
- Create: `output_real_step_s3_v5_13_transition_opening_proposals/reports/v5_13_decision.md`

- [ ] **Step 1: Generate comparison and decision**

Run:

```powershell
@'
import json
from pathlib import Path

old_path = Path("output_real_step_s3_v5_12_hybrid_proposals/reports/proposal_recall_s3_hybrid.json")
new_path = Path("output_real_step_s3_v5_13_transition_opening_proposals/reports/proposal_recall_s3_v5_13.json")
comparison_path = Path("output_real_step_s3_v5_13_transition_opening_proposals/reports/v5_12_vs_v5_13_comparison.md")
decision_path = Path("output_real_step_s3_v5_13_transition_opening_proposals/reports/v5_13_decision.md")

old = json.loads(old_path.read_text(encoding="utf-8"))
new = json.loads(new_path.read_text(encoding="utf-8"))
old_bp = old["best_by_proposal_recall"]
new_bp = new["best_by_proposal_recall"]

old_prop = old_bp["proposal_stage"]["recall"]
new_prop = new_bp["proposal_stage"]["recall"]
old_final = old_bp["final_survival_stage"]["recall"]
new_final = new_bp["final_survival_stage"]["recall"]
old_fail = old_bp["failure_stage_counts"]
new_fail = new_bp["failure_stage_counts"]
source_candidates = new_bp.get("proposal_candidate_count_by_source", {})
source_matches = new_bp.get("proposal_matched_truth_count_by_source", {})
source_macro_matches = new_bp.get("proposal_matched_truth_count_by_source_macro", {})

comparison_lines = [
    "# v5.12 vs v5.13 Proposal Expansion Comparison",
    "",
    "## Summary",
    "",
    "| Metric | v5.12 hybrid | v5.13 transition/opening |",
    "|--------|--------------|--------------------------|",
    f"| Proposal recall | {old_prop:.4f} | {new_prop:.4f} |",
    f"| Final-survival recall | {old_final:.4f} | {new_final:.4f} |",
    f"| No compatible proposal | {old_fail.get('no_area_compatible_proposal', 0)} | {new_fail.get('no_area_compatible_proposal', 0)} |",
    f"| Classifier wrong/rejected | {old_fail.get('proposal_exists_classifier_wrong_or_rejected', 0)} | {new_fail.get('proposal_exists_classifier_wrong_or_rejected', 0)} |",
    f"| Accepted but final changes/removes | {old_fail.get('classifier_accepts_but_final_changes_or_removes', 0)} | {new_fail.get('classifier_accepts_but_final_changes_or_removes', 0)} |",
    f"| Survives final | {old_fail.get('survives_final_prediction', 0)} | {new_fail.get('survives_final_prediction', 0)} |",
    "",
    "## v5.13 Source Coverage",
    "",
    "| Source | Candidate Count | Matched Truth Count |",
    "|--------|-----------------|---------------------|",
]
for source in sorted(set(source_candidates) | set(source_matches)):
    comparison_lines.append(f"| {source} | {source_candidates.get(source, 0)} | {source_matches.get(source, 0)} |")
comparison_lines.extend([
    "",
    "## v5.13 Source Matches by Macro",
    "",
    "| Source | Macro | Matched Truth Count |",
    "|--------|-------|---------------------|",
])
for source in sorted(source_macro_matches):
    for macro, count in sorted(source_macro_matches[source].items()):
        comparison_lines.append(f"| {source} | {macro} | {count} |")

comparison_path.parent.mkdir(parents=True, exist_ok=True)
comparison_path.write_text("\\n".join(comparison_lines) + "\\n", encoding="utf-8")

if new_prop >= 0.65 and new_bp["accepted_stage"]["recall"] < new_prop * 0.80:
    recommendation = "hybrid_candidate_classifier_retraining"
    reason = "Proposal recall is now high enough, but accepted-stage recall is substantially lower. Train or recalibrate the candidate classifier for hybrid proposal sources."
elif new_prop >= 0.65:
    recommendation = "hybrid_pipeline_integration_trial"
    reason = "Proposal and accepted-stage recalls are strong enough for a controlled inference-pipeline integration trial."
elif new_prop > old_prop + 0.03:
    recommendation = "continue_targeted_proposal_expansion"
    reason = "The new proposal sources improved recall, but proposal recall remains below 0.65. Continue with targeted rib proposal expansion."
else:
    recommendation = "rib_focused_proposal_diagnosis"
    reason = "Transition/opening expansion did not materially improve proposal recall. The remaining proposal problem should be diagnosed around rib and other still-missed macros."

decision_lines = [
    "# v5.13 Proposal Expansion Decision",
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
    f"- v5.12 proposal recall: {old_prop:.4f}",
    f"- v5.13 proposal recall: {new_prop:.4f}",
    f"- v5.13 accepted-stage recall: {new_bp['accepted_stage']['recall']:.4f}",
    f"- v5.13 final-survival recall: {new_final:.4f}",
    f"- v5.13 no compatible proposal: {new_fail.get('no_area_compatible_proposal', 0)}",
    f"- v5.13 classifier wrong/rejected: {new_fail.get('proposal_exists_classifier_wrong_or_rejected', 0)}",
    f"- v5.13 accepted but final changes/removes: {new_fail.get('classifier_accepts_but_final_changes_or_removes', 0)}",
    "",
]
decision_path.write_text("\\n".join(decision_lines) + "\\n", encoding="utf-8")

print(comparison_path)
print(decision_path)
'@ | W:\python\envs\feagpt\python.exe -
```

- [ ] **Step 2: Inspect decision**

Run:

```powershell
Get-Content output_real_step_s3_v5_13_transition_opening_proposals\reports\v5_13_decision.md
```

Expected:

- Exactly one recommendation appears in bold.
- It is one of:
  - `hybrid_candidate_classifier_retraining`
  - `hybrid_pipeline_integration_trial`
  - `continue_targeted_proposal_expansion`
  - `rib_focused_proposal_diagnosis`

## 8. Final Verification

- [ ] **Step 1: Run full tests**

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

- [ ] **Step 3: Verify reports exist**

Run:

```powershell
Test-Path output_real_step_s3_v5_13_transition_opening_proposals\reports\proposal_recall_s3_v5_13.json
Test-Path output_real_step_s3_v5_13_transition_opening_proposals\reports\proposal_recall_s3_v5_13.md
Test-Path output_real_step_s3_v5_13_transition_opening_proposals\reports\v5_13_decision.md
Test-Path output_real_step_s3_v5_13_transition_opening_proposals\reports\v5_12_vs_v5_13_comparison.md
```

Expected:

```text
True
True
True
True
```

## 9. Required Final Response From Implementing Agent

The implementing agent must report:

1. Tests run and whether they passed.
2. v5.12 proposal recall and v5.13 proposal recall.
3. v5.13 accepted-stage recall.
4. v5.13 final-survival recall.
5. v5.13 failure-stage counts.
6. v5.13 source coverage, including `fallback_rule`.
7. v5.13 source-by-macro matched truth counts.
8. Whether fillet/chamfer no-proposal failures decreased.
9. Whether through-hole no-proposal failures decreased.
10. The single recommendation from `v5_13_decision.md`.
11. Paths to:
    - `proposal_recall_s3_v5_13.json`
    - `proposal_recall_s3_v5_13.md`
    - `v5_13_decision.md`
    - `v5_12_vs_v5_13_comparison.md`
12. Any deviation from this plan.

## 10. Review Checklist for Codex

When reviewing implementation:

1. `utils/object_candidate_proposer.py` imports and calls `find_fillet_stock_candidates`.
2. `utils/object_candidate_proposer.py` imports and calls `find_chamfer_stock_candidates`.
3. `opening_cylinder_fallback_proposals` exists and is included in `generate_hybrid_object_proposals`.
4. `fallback_rule` appears in v5.13 source coverage.
5. v5.13 markdown title says `v5.13 Object Proposal Recall Diagnostic Report`.
6. v5.13 source-by-macro fields exist in JSON.
7. v5.13 does not modify GNN/model/training/generator files.
8. If v5.13 proposal recall reaches at least `0.65`, the next likely task is candidate classifier retraining or pipeline integration, depending on accepted-stage recall.
9. If v5.13 proposal recall stays below `0.65`, the next likely task is rib-focused proposal diagnosis.
