# CAE STEP Dataset Generator v2.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve synthetic STEP generation after user UG inspection by fixing obround-hole labeling, making stock profiles visually distinct, limiting destructive feature clutter, and introducing surface-anchored swept slot metadata/generation v1.

**Architecture:** Keep the existing generator pipeline and add small, testable policies. v2.1 does not build a full CAD feature kernel; it adds axis-aware matching, stock profile constraints, per-sample feature caps, and a first surface-anchored swept-slot abstraction for straight boundary-to-boundary paths.

**Tech Stack:** Python, pythonocc/OpenCASCADE, unittest, existing `feagpt` environment at `W:\python\envs\feagpt`.

---

### Task 1: Fix Obround Hole Side-Wall Labeling

**Files:**
- Modify: `features/obround_hole.py`
- Test: `tests/test_obround_hole_axis_matching.py`

- [ ] **Step 1: Write the failing test**

Create a test that builds fake plane surfaces with normals perpendicular to an `x` or `y` hole axis and verifies `ObroundHoleGenerator.match_face()` returns `obround_hole_side_plane` instead of `None`.

- [ ] **Step 2: Run test to verify it fails**

Run: `W:\python\envs\feagpt\python.exe -m unittest tests.test_obround_hole_axis_matching -v`

- [ ] **Step 3: Implement axis-aware side-plane matching**

Use the obround hole's `axis` parameter and a helper that treats side planes as planes whose normal is perpendicular to the hole axis, rather than checking only `abs(normal.Z()) < 1e-4`.

- [ ] **Step 4: Verify**

Run the targeted test and then the full test suite.

### Task 2: Make Stock Profiles Visually Distinct

**Files:**
- Modify: `utils/stock_factory.py`
- Test: `tests/test_stock_factory.py`

- [ ] **Step 1: Write failing tests**

Add assertions that plate thickness is much smaller than its planform dimensions and beam length is much larger than both cross-section dimensions.

- [ ] **Step 2: Run test to verify it fails**

Run: `W:\python\envs\feagpt\python.exe -m unittest tests.test_stock_factory -v`

- [ ] **Step 3: Adjust `_plate_spec()` and `_beam_spec()`**

Make plate thickness approximately 8-14% of the smaller planform dimension. Make beam length at least 4x the larger cross-section dimension.

- [ ] **Step 4: Verify**

Run stock tests and full tests.

### Task 3: Limit Per-Sample Feature Clutter

**Files:**
- Modify: `utils/generation_context.py`
- Modify: `main.py`
- Modify: `configs/static_structural_v2_pilot_80.yaml`
- Test: `tests/test_generation_context.py`

- [ ] **Step 1: Write failing tests**

Add a default `feature_density` policy with `max_total_instances_per_sample`. Add a small deterministic test that no more than the cap can be requested across generator stages.

- [ ] **Step 2: Run failing tests**

Run: `W:\python\envs\feagpt\python.exe -m unittest tests.test_generation_context -v`

- [ ] **Step 3: Implement feature cap**

In `generate_annotated_sample()`, track remaining feature budget and reduce requested instances per generator to prevent excessive feature count. Keep `require_nonempty_sample`.

- [ ] **Step 4: Verify**

Run generation context tests and full tests.

### Task 4: Surface-Anchored Swept Slot v1

**Files:**
- Modify: `features/swept_slot.py`
- Modify: `features/slot_rect.py`
- Modify: `features/slot_u.py`
- Test: `tests/test_swept_slot_profiles.py`

- [ ] **Step 1: Write failing tests**

Add metadata tests for `path_on_face`, `path_start_on_boundary`, `path_end_on_boundary`, `path_type`, and supported profile shapes `u`, `v`, `dovetail`, `o`.

- [ ] **Step 2: Run failing tests**

Run: `W:\python\envs\feagpt\python.exe -m unittest tests.test_swept_slot_profiles -v`

- [ ] **Step 3: Implement metadata and profile builders**

Add v1 profile metadata and simple wire builders for straight surface slots. Existing `slot_rect` and `slot_u` should emit the new metadata. V/dovetail/O can be introduced as metadata-ready wire builders before broader generator integration.

- [ ] **Step 4: Verify**

Run swept slot tests and full tests.

### Task 5: Generate Pilot And Validate

**Files:**
- Output: `output_generation_v2_1_pilot_80`

- [ ] **Step 1: Compile key modules**

Run: `W:\python\envs\feagpt\python.exe -m py_compile main.py utils\generation_context.py utils\stock_factory.py features\obround_hole.py features\swept_slot.py features\slot_rect.py features\slot_u.py`

- [ ] **Step 2: Run full tests**

Run: `W:\python\envs\feagpt\python.exe -m unittest discover -s tests -v`

- [ ] **Step 3: Generate 80 samples**

Run: `W:\python\envs\feagpt\python.exe main.py --config configs\static_structural_v2_pilot_80.yaml --output-dir output_generation_v2_1_pilot_80 --seed 20260604 --skip-training-export`

- [ ] **Step 4: Validate STEP topology**

Run: `W:\python\envs\feagpt\python.exe utils\step_solid_validation.py --steps-dir output_generation_v2_1_pilot_80\steps --output-path output_generation_v2_1_pilot_80\reports\step_solid_validation.json --fail-on-negative-volume`

- [ ] **Step 5: Report distributions**

Count stock types, feature instances, hole axes, feature cap rejections, and swept-slot metadata coverage.
