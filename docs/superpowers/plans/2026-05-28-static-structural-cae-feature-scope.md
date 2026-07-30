# Static Structural CAE Feature Scope Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Narrow the dataset generator to the V1 single-part static structural CAE feature scope while preserving old generator modules for later reuse.

**Architecture:** The core change is configuration and ontology-driven: keep existing generator classes, but make the default V1 path use only local, queryable features. Canonicalize rectangular pads as `rectangular_boss`, keep obround holes under holes, and move hole groups/flanges/support pads out of default training scope.

**Tech Stack:** Python, YAML ontology files, existing OCC-based feature generators, lightweight Python regression checks.

**Current Status:** Tasks 1-8 are implemented and verified in this workspace. The `feagpt` environment at `W:\python\envs\feagpt` can import `OCC`, `PyYAML`, `scikit-learn`, and `torch`; V1 smoke dataset generation and dataset verification passed.

---

### Task 1: Freeze The V1 Scope In Documentation

**Files:**
- Create: `docs/superpowers/specs/2026-05-28-static-structural-cae-feature-scope-design.md`
- Create: `docs/superpowers/plans/2026-05-28-static-structural-cae-feature-scope.md`

- [ ] **Step 1: Write the V1 design spec**

Record the agreed scope:

```text
single-part static structural preprocessing
local queryable features only
obround_hole remains a hole feature
hole groups are derived after single-hole recognition
flange/support_pad/load_pad are not V1 macro features
```

- [ ] **Step 2: Write this implementation plan**

Use this plan file to keep the first implementation pass auditable.

### Task 2: Canonicalize The Macro Ontology

**Files:**
- Modify: `CAE_feature_ontology.yaml`

- [ ] **Step 1: Keep V1 core features enabled**

Enabled V1 macro features should be:

```text
stock
through_hole_round
blind_hole_round
stepped_hole_round
counterbore_hole
countersink_hole
obround_hole
slot_u
slot_rect
pocket_rect
pocket_circular
cylinder_boss
rectangular_boss
rib
web
fillet
chamfer
```

- [ ] **Step 2: Disable non-core labels for default coverage**

Set `enabled: false` for:

```text
triangular_cutout
rectangular_cutout
notch_relief
cone_boss
triangle_prism_boss
rectangle_prism_boss
triangle_frustum_boss
rectangle_frustum_boss
flange
gusset
bolt_hole_pattern
bearing_seat
support_pad
```

- [ ] **Step 3: Canonicalize pad**

Rename the macro label with id `24` from `pad` to `rectangular_boss`, keep aliases `pad`, `rectangle_boss`, and `rectangular_boss`.

### Task 3: Align Semantic Subtypes With V1

**Files:**
- Modify: `face_semantic_subtype_ontology.yaml`

- [ ] **Step 1: Set pad subtypes to rectangular boss**

Change:

```yaml
parent_macro_feature: pad
```

to:

```yaml
parent_macro_feature: rectangular_boss
```

for `pad_side_plane` and `pad_top_plane`.

- [ ] **Step 2: Narrow the recommended subtype subset**

Keep hole, slot, pocket, cylinder boss, rectangular boss, rib/web, fillet, and chamfer subtype keys. Remove flange, support pad, bolt pattern, bearing seat, gusset, notch, cone boss, polygon cutout, and polygon/frustum boss subtypes from `recommended_v1_label_subset`.

### Task 4: Narrow Default Generation

**Files:**
- Modify: `main.py`
- Modify: `configs/default_dataset_config.yaml`
- Create: `configs/static_structural_v1_dataset.yaml`

- [ ] **Step 1: Add a V1 generator list**

Define a constant in `main.py`:

```python
STATIC_STRUCTURAL_V1_GENERATORS = [
    "SteppedHoleGenerator",
    "ThroughHoleGenerator",
    "BlindHoleGenerator",
    "CounterboreHoleGenerator",
    "CountersinkHoleGenerator",
    "ObroundHoleGenerator",
    "USlotGenerator",
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
```

- [ ] **Step 2: Use the V1 list as the default enabled generators**

Change `DEFAULT_CONFIG["generation"]["enabled_generators"]` to use `STATIC_STRUCTURAL_V1_GENERATORS`.

- [ ] **Step 3: Update feature catalog targets**

Remove non-core catalog entries and map `rectangular_boss` to `PadGenerator`.

- [ ] **Step 4: Add a named V1 config**

Create `configs/static_structural_v1_dataset.yaml` with the same V1 generator list and explicit coverage targets.

### Task 5: Stabilize Boss Generation For V1

**Files:**
- Modify: `features/boss_feature.py`

- [ ] **Step 1: Restrict random boss type**

Set:

```python
self.allowed_boss_types = ["cylinder"]
```

and select:

```python
boss_type = random.choice(self.allowed_boss_types)
```

This keeps `cone_boss` as ontology-compatible legacy material but prevents it from entering V1 default data.

### Task 6: Update Derived CAE Labels

**Files:**
- Modify: `utils/derived_labels.py`

- [ ] **Step 1: Remove non-core thin-wall semantic keys**

Keep:

```python
THIN_WALL_SEMANTIC_KEYS = {"rib_side_plane", "web_side_plane"}
```

- [ ] **Step 2: Use rectangular boss transition labels**

Keep `pad_side_plane` and `pad_top_plane` because they now represent `rectangular_boss`, and set:

```python
THICKNESS_TRANSITION_MACROS = {"rectangular_boss"}
```

### Task 7: Add Regression Checks

**Files:**
- Create: `tests/test_static_structural_v1_scope.py`

- [ ] **Step 1: Test ontology and config scope**

Add checks that:

```python
assert "obround_hole" in enabled macro keys
assert "rectangular_boss" in enabled macro keys
assert "bolt_hole_pattern" not in enabled macro keys
assert "flange" not in default enabled generators
assert "PadGenerator" in default enabled generators
```

- [ ] **Step 2: Run the regression check**

Run:

```powershell
python -m unittest tests.test_static_structural_v1_scope -v
```

Expected: all tests pass.

### Task 8: Smoke Verify Dataset Infrastructure

**Files:**
- No code changes.

- [ ] **Step 1: Verify ontology loading**

Run:

```powershell
python - <<'PY'
from pathlib import Path
from utils.ontology import load_ontology_maps
maps = load_ontology_maps(Path("CAE_feature_ontology.yaml"))
print([x.key for x in maps.macro_features if x.enabled])
PY
```

Expected: enabled macro keys are the V1 core features plus derived CAE labels that are intentionally enabled.

- [ ] **Step 2: Generate a tiny smoke dataset**

Run:

```powershell
python main.py --config configs/static_structural_v1_dataset.yaml --output-dir output_smoke_static_v1 --num-samples 2 --seed 20260528 --skip-training-export
```

Expected: the command writes STEP and graph files without producing non-core macro features.
