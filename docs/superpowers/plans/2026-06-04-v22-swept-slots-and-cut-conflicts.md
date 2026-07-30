# CAE Dataset Generator v2.2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the old U-slot generation with true closed-profile swept slots and add a conservative cut-feature conflict mechanism.

**Architecture:** Add a focused swept-slot generator that builds a vertical closed profile, sweeps it along a top-face path, and cuts the stock. Add footprint conflict utilities so cut features reserve a 2D projected region before applying Boolean subtraction.

**Tech Stack:** Python, pythonocc-core/OCC B-Rep builders, existing YAML ontologies, unittest.

---

### Task 1: Swept Slot Geometry Helpers

**Files:**
- Modify: `features/swept_slot.py`
- Test: `tests/test_swept_slot_v2.py`

- [ ] Add failing tests for vertical closed profiles `u`, `v`, `dovetail`, and `o`.
- [ ] Implement `vertical_swept_slot_profile_points()` with `surface_z`, `width`, `depth`, and `top_margin`.
- [ ] Implement `build_vertical_swept_slot_tool()` by making a vertical profile face and sweeping it along a straight path vector.
- [ ] Verify tests pass.

### Task 2: Cut Feature Footprint Conflicts

**Files:**
- Create: `utils/feature_footprint.py`
- Test: `tests/test_feature_footprint.py`

- [ ] Add failing tests for circle-vs-circle and capsule-vs-circle conflicts.
- [ ] Implement `circle_footprint()`, `capsule_footprint()`, and `footprint_conflicts()`.
- [ ] Verify tests pass.

### Task 3: New Swept Slot Generator

**Files:**
- Create: `features/swept_channel_slot.py`
- Modify: `main.py`
- Modify: `configs/static_structural_v2_pilot_80.yaml`
- Test: `tests/test_swept_slot_v2.py`

- [ ] Add failing tests showing the generator supports all four profile shapes.
- [ ] Implement `SweptSlotGenerator` with profile cycling/sampling.
- [ ] Return macro keys `slot_u`, `slot_v`, `slot_dovetail`, and `slot_o`.
- [ ] Match side and bottom faces using corridor-local coordinates.
- [ ] Register `SweptSlotGenerator` in `main.py`.

### Task 4: Ontology and Selectable Regions

**Files:**
- Modify: `CAE_feature_ontology.yaml`
- Modify: `face_semantic_subtype_ontology.yaml`
- Modify: `utils/selectable_regions.py`
- Modify: `utils/requirement_matching.py`
- Test: existing ontology and requirement tests

- [ ] Add enabled macro labels for `slot_v`, `slot_dovetail`, and `slot_o`.
- [ ] Add semantic subtypes for side and bottom faces of the new slot classes.
- [ ] Add selectable regions for new slot classes.
- [ ] Verify ontology loading and existing tests pass.

### Task 5: Pilot Generation and Validation

**Files:**
- Output: `output_generation_v2_2_pilot_80`
- Output: `output_generation_v2_2_pilot_80/reports/step_solid_validation.json`

- [ ] Run full unittest suite.
- [ ] Generate 80 v2.2 samples.
- [ ] Run STEP solid validation.
- [ ] Report stock distribution, feature distribution, swept slot profile distribution, and cut conflict skip counts if available.
