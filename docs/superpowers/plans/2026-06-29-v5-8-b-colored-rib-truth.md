# v5.8-B Colored Rib Truth Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert user-colored STEP faces into face-level rib truth for the existing `rib_face_truth_template_v5_8_b.json`.

**Architecture:** Read face colors from STEPCAF/XCAF when available, treat `Medium Steel` as non-rib and every other face-level color as a rib instance color, then map colored STEP faces back to graph/template `face_idx` by geometry fingerprints. The output is a reviewed template JSON plus a Markdown diagnostics report for unresolved or ambiguous matches.

**Tech Stack:** Python, pythonOCC STEPCAF/XCAF APIs, existing AAG graph JSON, unittest.

---

### Task 1: Core Color Semantics

**Files:**
- Create: `utils/colored_step_rib_truth.py`
- Test: `tests/test_colored_step_rib_truth.py`

- [ ] **Step 1: Write failing tests**

Test that `Medium Steel` is treated as background and distinct non-background colors are mapped to stable rib instance ids.

- [ ] **Step 2: Run test to verify it fails**

Run: `W:\python\envs\feagpt\python.exe -m unittest tests.test_colored_step_rib_truth -v`
Expected: FAIL because `utils.colored_step_rib_truth` does not exist.

- [ ] **Step 3: Implement minimal color semantics**

Add pure helpers:
- `is_background_color_name(name)`
- `is_background_rgb(rgb)`
- `rib_instance_id_for_color(color_name, rgb, seen_colors)`

- [ ] **Step 4: Run test to verify it passes**

Run the same unittest command and expect PASS.

### Task 2: Template Review Application

**Files:**
- Modify: `utils/colored_step_rib_truth.py`
- Test: `tests/test_colored_step_rib_truth.py`

- [ ] **Step 1: Write failing tests**

Given a small template and colored face labels, verify that:
- colored rib faces become `review.is_rib_face = true`
- background-colored candidate faces become `false`
- same color gets same `rib_instance_id`
- faces not present in the template are reported but do not crash.

- [ ] **Step 2: Implement review application**

Add `apply_colored_rib_truth_to_template(template, colored_truth_by_sample)` returning `(reviewed_template, diagnostics)`.

- [ ] **Step 3: Run tests**

Run: `W:\python\envs\feagpt\python.exe -m unittest tests.test_colored_step_rib_truth -v`

### Task 3: STEP Face Color Extraction and Geometry Matching

**Files:**
- Modify: `utils/colored_step_rib_truth.py`
- Create: `apply_colored_rib_truth_template.py`

- [ ] **Step 1: Implement STEPCAF reader**

Read color labels with `STEPCAFControl_Reader`, `XCAFDoc_DocumentTool.ShapeTool`, and `XCAFDoc_DocumentTool.ColorTool`.

- [ ] **Step 2: Match colored STEP faces to graph nodes**

Use geometry fingerprints: primitive surface, area, center, radius/axis/normal. Prefer direct OCC face order when possible; fall back to nearest fingerprint match.

- [ ] **Step 3: Implement CLI**

Inputs:
- `--template-json`
- `--colored-step-dir`
- `--graph-dir`
- `--output-json`
- `--output-md`

Outputs:
- reviewed template JSON
- Markdown diagnostics with matched, unmatched, and ambiguous faces.

### Task 4: Real Colored Data Run

**Files:**
- Output under `output_real_step_s3_v5_8_a_role_repair/reports/`

- [ ] **Step 1: Run CLI**

Use `test_part_s3_rib_colored` as the colored STEP directory and the existing v5.8-B template.

- [ ] **Step 2: Run regression tests**

Run:
`W:\python\envs\feagpt\python.exe -m unittest tests.test_colored_step_rib_truth tests.test_rib_face_truth_template tests.test_rib_boundary_classifier_dataset tests.test_rib_boundary_hardcase_dataset tests.test_rib_boundary_classifier_postprocess -v`

- [ ] **Step 3: Summarize diagnostics**

Report sample count, matched colored rib faces, candidate faces filled true/false/null, and any ambiguous matches requiring human inspection.
