# v2.3 Curved Swept Slots And Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 v2.2.1 的解析 U/O/V 槽从“直线路径扫掠”推进到“直线或圆弧路径扫掠”，并让 footprint 冲突检测、face matching、数据验证报告同步支持曲线路径。

**Architecture:** v2.3 只扩展槽生成闭环，不重写整个特征生成器。路径几何、footprint、OCC sweep tool、face matching 分层实现：先用可测试的二维路径描述，再把路径描述转成 OCC path wire，最后由生成器统一记录 raw parameters 和图标签。

**Tech Stack:** Python, pythonocc-core/OCC, unittest, existing dataset generator, `W:\python\envs\feagpt\python.exe`.

---

## Scope

v2.3 做：

- `slot_u / slot_v / slot_o` 支持 `line` 和 `circular_arc` 两类扫掠路径。
- U/O 截面继续使用 v2.2.1 的真实 OCC 圆弧/圆，不回退到折线。
- 曲线路径的二维 footprint 冲突检测，防止槽和孔/口袋等 cut feature 明显互相破坏。
- 曲线路径的 face matching，避免曲线槽面被误标为 `stock_face`。
- 新增 v2.3 pilot 配置和 80-100 个检查样本。

v2.3 不做：

- 不恢复 `slot_dovetail` 随机生成。
- 不做任意三维非平面曲线轨迹，只做 top planar face 上的二维圆弧轨迹。
- 不把所有特征都改成完全依赖当前实体面的顺序建模。这个拆到后续 v2.4/v2.5。

---

## Files

- Modify: `features/swept_slot.py`
  - Add OCC path wire builders for line and circular arc.
  - Add curved sweep tool using `BRepOffsetAPI_MakePipe`.
  - Keep existing straight prism path for line case if it remains more stable.

- Modify: `features/swept_channel_slot.py`
  - Sample `line` or `circular_arc` path specs.
  - Use path spec for footprint, tool creation, metadata, and face matching.
  - Record `path_curve_kind`, `path_points_xy`, `path_radius`, `path_sagitta`, and `path_length`.

- Modify: `utils/feature_footprint.py`
  - Add `polyline_capsule_footprint()` and distance checks against circle/capsule/polyline capsule.

- Create: `utils/path_geometry.py`
  - Pure Python helpers for path length, sampled circular arc, point-to-polyline distance, and path tangent.
  - This keeps OCC-specific code out of footprint/matching tests.

- Create: `tests/test_path_geometry.py`
  - Unit tests for circular arc sampling and point distance.

- Modify: `tests/test_feature_footprint.py`
  - Add conflict tests for polyline capsule vs circle/capsule.

- Modify: `tests/test_swept_slot_v2.py`
  - Add curved path OCC tool test and curved path metadata test.

- Create: `configs/static_structural_v2_3_pilot_100.yaml`
  - 100 samples, same broad v2.2 stock/features, but use v2.3 slot generator settings.

- Optional create: `utils/slot_generation_audit.py`
  - Read graph JSON files and report slot counts, path kinds, node counts, and empty-label instances.

---

## Task 1: Path Geometry Helpers

**Files:**
- Create: `utils/path_geometry.py`
- Create: `tests/test_path_geometry.py`

- [ ] **Step 1: Write failing tests**

Create tests for:

```python
def test_sample_circular_arc_points_keeps_endpoints_and_length_above_chord():
    points = sample_circular_arc_points((-10.0, 0.0), (10.0, 0.0), sagitta=5.0, sample_count=9)
    assert points[0] == (-10.0, 0.0)
    assert points[-1] == (10.0, 0.0)
    assert polyline_length(points) > 20.0


def test_point_to_polyline_distance_is_zero_on_path():
    points = [(-10.0, 0.0), (0.0, 4.0), (10.0, 0.0)]
    assert point_to_polyline_distance(0.0, 4.0, points) == 0.0


def test_path_tangent_angle_uses_first_segment():
    points = [(-10.0, 0.0), (-6.0, 3.0), (10.0, 0.0)]
    assert abs(path_tangent_angle(points) - math.atan2(3.0, 4.0)) < 1e-9
```

- [ ] **Step 2: Run test to verify red**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_path_geometry
```

Expected: import/function missing failures.

- [ ] **Step 3: Implement pure Python helpers**

Implement:

```python
def polyline_length(points: list[tuple[float, float]]) -> float: ...
def point_to_polyline_distance(px: float, py: float, points: list[tuple[float, float]]) -> float: ...
def sample_circular_arc_points(start, end, sagitta: float, sample_count: int = 17) -> list[tuple[float, float]]: ...
def path_tangent_angle(points: list[tuple[float, float]], at_end: bool = False) -> float: ...
```

Rules:

- `sample_circular_arc_points()` keeps exact start/end points.
- `sagitta > 0` bends to one side of the chord; `sagitta < 0` bends to the other side.
- `sample_count` minimum is 3.

- [ ] **Step 4: Verify green**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_path_geometry
```

Expected: tests pass.

---

## Task 2: Polyline Capsule Footprints

**Files:**
- Modify: `utils/feature_footprint.py`
- Modify: `tests/test_feature_footprint.py`

- [ ] **Step 1: Write failing tests**

Add tests:

```python
def test_polyline_capsule_conflicts_with_circle_near_any_segment():
    slot = polyline_capsule_footprint([(-20.0, 0.0), (0.0, 8.0), (20.0, 0.0)], width=6.0, feature_kind="slot_u")
    hole = circle_footprint(0.0, 8.0, 3.0, feature_kind="blind_hole_round")
    assert footprint_conflicts(slot, [hole], clearance=1.0)


def test_polyline_capsule_does_not_conflict_with_far_circle():
    slot = polyline_capsule_footprint([(-20.0, 0.0), (0.0, 8.0), (20.0, 0.0)], width=6.0, feature_kind="slot_u")
    hole = circle_footprint(0.0, 30.0, 3.0, feature_kind="blind_hole_round")
    assert not footprint_conflicts(slot, [hole], clearance=1.0)
```

- [ ] **Step 2: Run test to verify red**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_feature_footprint
```

- [ ] **Step 3: Implement footprint type**

Add:

```python
def polyline_capsule_footprint(points, width, *, operation="cut", zone_plane="xy", feature_kind="slot") -> dict:
    return {
        "footprint_type": "polyline_capsule",
        "points": [(float(x), float(y)) for x, y in points],
        "width": float(width),
        "radius": float(width) / 2.0,
        "operation": operation,
        "zone_plane": zone_plane,
        "feature_kind": feature_kind,
    }
```

Extend `_footprint_distance()` for:

- polyline capsule vs circle
- circle vs polyline capsule
- polyline capsule vs capsule
- capsule vs polyline capsule
- polyline capsule vs polyline capsule

Use existing segment distance helpers.

- [ ] **Step 4: Verify green**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_feature_footprint
```

---

## Task 3: Curved OCC Sweep Tool

**Files:**
- Modify: `features/swept_slot.py`
- Modify: `tests/test_swept_slot_v2.py`

- [ ] **Step 1: Write failing test**

Add a test that builds a curved U/O/V tool:

```python
def test_vertical_swept_slot_tool_supports_circular_arc_path_points():
    path_points = [(-25.0, 0.0), (-10.0, 8.0), (10.0, 8.0), (25.0, 0.0)]
    tool = build_vertical_swept_slot_tool_from_path(
        profile_shape="u",
        path_points_xy=path_points,
        surface_z=20.0,
        width=8.0,
        depth=6.0,
        top_margin=1.0,
    )
    assert tool is not None
```

- [ ] **Step 2: Run test to verify red**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_swept_slot_v2
```

- [ ] **Step 3: Implement path wire and pipe sweep**

Add:

```python
def build_path_wire_from_points(path_points_xy: list[tuple[float, float]], surface_z: float):
    ...


def build_vertical_swept_slot_tool_from_path(
    profile_shape: str,
    path_points_xy: list[tuple[float, float]],
    surface_z: float,
    width: float,
    depth: float,
    top_margin: float = 1.0,
):
    ...
```

Implementation guidance:

- Use first path segment angle for `build_vertical_swept_slot_profile_wire()`.
- Build profile at `path_points_xy[0]`.
- For `len(path_points_xy) == 2`, delegate to existing `build_vertical_swept_slot_tool()` or build a two-edge path wire.
- For curved/polyline paths, use `BRepOffsetAPI_MakePipe(path_wire, profile_face).Shape()`.
- If `MakePipe` fails for a candidate during generation, generator should skip that candidate and retry.

- [ ] **Step 4: Verify green**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_swept_slot_v2
```

---

## Task 4: Curved Path Sampling In SweptSlotGenerator

**Files:**
- Modify: `features/swept_channel_slot.py`
- Modify: `tests/test_swept_slot_v2.py`

- [ ] **Step 1: Write tests for path metadata**

Add a unit-level test around a helper:

```python
def test_curved_slot_path_metadata_records_arc_kind():
    path = _path_metadata_from_points([(-20.0, 0.0), (0.0, 6.0), (20.0, 0.0)], path_curve_kind="circular_arc")
    assert path["path_curve_kind"] == "circular_arc"
    assert path["path_type"] == "curved"
    assert path["length"] > 40.0
```

- [ ] **Step 2: Add generator helpers**

Add helpers in `features/swept_channel_slot.py`:

```python
def _sample_slot_path_for_current_stock(...): ...
def _straight_path_points(path: dict) -> list[tuple[float, float]]: ...
def _arc_path_from_straight_path(path: dict, sagitta: float, sample_count: int = 17) -> dict: ...
def _path_metadata_from_points(points, path_curve_kind): ...
```

Rules:

- Randomly choose `line` about 65 percent and `circular_arc` about 35 percent.
- Arc endpoints still come from the stock boundary line intersections.
- Arc sagitta range: `0.12-0.25 * straight_chord_length`, capped so sampled points stay inside the top-face workspace.
- If sampled arc points leave the stock boundary envelope, retry.

- [ ] **Step 3: Use polyline footprint and curved tool**

Replace straight-only footprint with:

```python
candidate_footprint = polyline_capsule_footprint(
    candidate_path["path_points_xy"],
    slot_w,
    operation="cut",
    feature_kind=macro_key,
)
```

Use `build_vertical_swept_slot_tool_from_path()` instead of only `build_vertical_swept_slot_tool()`.

- [ ] **Step 4: Verify generator tests**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_swept_slot_v2 tests.test_feature_footprint tests.test_path_geometry
```

---

## Task 5: Curved Path Face Matching

**Files:**
- Modify: `features/swept_channel_slot.py`
- Modify: `utils/path_geometry.py`
- Modify: `tests/test_swept_slot_v2.py`

- [ ] **Step 1: Add tests for matching math**

Test pure helper:

```python
def test_point_to_slot_path_local_accepts_near_curved_path():
    params = {"path_points_xy": [(-20.0, 0.0), (0.0, 8.0), (20.0, 0.0)], "w": 6.0, "depth": 5.0}
    assert _point_is_inside_path_corridor(params, 0.0, 8.2, tolerance=0.5)
    assert not _point_is_inside_path_corridor(params, 0.0, 20.0, tolerance=0.5)
```

- [ ] **Step 2: Replace line-only local coordinate matching**

Keep `_to_path_local()` for legacy straight params, but add:

```python
def _point_is_inside_path_corridor(params, px, py, tolerance=0.35):
    points = params.get("path_points_xy")
    if points:
        return point_to_polyline_distance(px, py, points) <= params["w"] / 2.0 + tolerance
    local_s, local_t = _to_path_local(params, px, py)
    return abs(local_s) <= params["l"] / 2.0 + tolerance and abs(local_t) <= params["w"] / 2.0 + tolerance
```

Use this in `match_face()`.

- [ ] **Step 3: Verify matching tests**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_swept_slot_v2
```

---

## Task 6: v2.3 Config And Audit Report

**Files:**
- Create: `configs/static_structural_v2_3_pilot_100.yaml`
- Optional create: `utils/slot_generation_audit.py`

- [ ] **Step 1: Add config**

Base it on `configs/static_structural_v2_2_pilot_80.yaml`, with:

```yaml
dataset:
  num_samples: 100
  random_seed: 20260604

generation:
  feature_density:
    enabled: true
    max_total_instances_per_sample: 7
    max_instances_per_generator: 1
```

Keep `SweptSlotGenerator` enabled.

- [ ] **Step 2: Add slot audit script**

The script should print:

- total sample count
- slot_u/slot_v/slot_o counts
- line vs circular_arc counts
- empty slot feature_sets count
- min/mean/max slot node count

- [ ] **Step 3: Run pilot generation**

Run:

```powershell
W:\python\envs\feagpt\python.exe main.py --config configs\static_structural_v2_3_pilot_100.yaml --output-dir output_generation_v2_3_pilot_100 --seed 20260604 --skip-training-export
W:\python\envs\feagpt\python.exe utils\step_solid_validation.py --steps-dir output_generation_v2_3_pilot_100\steps --output-path output_generation_v2_3_pilot_100\reports\step_solid_validation.json
```

Expected:

- 100 STEP files written.
- `step_solid_validation.json` has `status=ok` and `blocking_issue_count=0`.

- [ ] **Step 4: Run slot audit**

Run:

```powershell
W:\python\envs\feagpt\python.exe utils\slot_generation_audit.py --graphs-dir output_generation_v2_3_pilot_100\graphs --output-path output_generation_v2_3_pilot_100\reports\slot_generation_audit.json
```

Expected:

- At least 10 curved slot instances in 100 samples.
- No empty slot feature_sets.

---

## Task 7: Manual Review Package For User

**Files:**
- Create or update: `output_generation_v2_3_pilot_100\reports\manual_review_targets.md`

- [ ] **Step 1: Select examples**

Choose:

- 3 `slot_u` with `path_curve_kind=line`
- 3 `slot_u` with `path_curve_kind=circular_arc`
- 3 `slot_o` with `path_curve_kind=circular_arc`
- 3 `slot_v` with `path_curve_kind=circular_arc`

- [ ] **Step 2: Write review file**

For each example record:

```markdown
- STEP: `output_generation_v2_3_pilot_100\steps\composite_feature_XXX.step`
  Graph: `output_generation_v2_3_pilot_100\graphs\composite_feature_XXX.json`
  Macro: `slot_u`
  Path: `circular_arc`
  User check: 截面是否正确、路径是否在表面、是否破坏其它 cut 特征、UG 中是否为实体
```

- [ ] **Step 3: Ask user to inspect in UG**

User should decide:

- 是否接受 O 槽“圆截面中心在表面附近，布尔切出圆弧形通道”的定义。
- 是否接受曲线槽仅在 top planar face 上出现。
- 是否需要把 `slot_o_side_plane` / `slot_u_bottom_plane` subtype 名称改成更严谨的 `surface` 命名。

---

## Verification Checklist

Before reporting v2.3 complete, run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest discover -s tests -p "test_*.py"
W:\python\envs\feagpt\python.exe -m py_compile features\swept_slot.py features\swept_channel_slot.py utils\feature_footprint.py utils\path_geometry.py utils\slot_generation_audit.py
W:\python\envs\feagpt\python.exe main.py --config configs\static_structural_v2_3_pilot_100.yaml --output-dir output_generation_v2_3_pilot_100 --seed 20260604 --skip-training-export
W:\python\envs\feagpt\python.exe utils\step_solid_validation.py --steps-dir output_generation_v2_3_pilot_100\steps --output-path output_generation_v2_3_pilot_100\reports\step_solid_validation.json
W:\python\envs\feagpt\python.exe utils\slot_generation_audit.py --graphs-dir output_generation_v2_3_pilot_100\graphs --output-path output_generation_v2_3_pilot_100\reports\slot_generation_audit.json
```

---

## After v2.3

### v2.4: Remaining Multi-Axis Hole Features

Goal: 将阶梯孔、沉孔、锥沉孔、长圆孔迁移到与通孔/盲孔一致的多轴框架。

Files likely touched:

- `features/stepped_hole.py`
- `features/counterbore_hole.py`
- `features/countersink_hole.py`
- `features/obround_hole.py`
- `features/oriented_hole.py`
- tests for axis matching and geometry parameters

User involvement:

- 抽查侧向阶梯孔、侧向沉孔、侧向锥沉孔、侧向长圆孔在 UG 中是否符合工程直觉。

### v2.5: Transition Feature And Interference Quality

Goal: 让圆角/倒角更可控地出现在毛坯边、孔口边、槽边、口袋边，并记录 convex/concave/context metadata。

Files likely touched:

- `features/fillet_feature.py`
- `features/chamfer_feature.py`
- new transition edge audit utilities

User involvement:

- 判断哪些倒角/圆角应该作为 CAE selectable region，哪些只是几何噪声。

### v2.6: External STEP Generalization Evaluation

Goal: 用用户手工工程零件和标注模板评估模型泛化，而不是只看合成验证集。

Files likely touched:

- `generate_real_step_annotation_template.py`
- `run_gnn_inference.py`
- `evaluate_gnn_predictions.py`
- reporting scripts

User involvement:

- 给 5-10 个真实 STEP 修正模板，标出孔、槽、口袋、肋/薄壁、凸台、圆角/倒角等关键面集。

### Paper Experiment Layer

最终论文建议保留这些实验：

- Synthetic validation: macro-F1, semantic-subtype-F1, instance grouping IoU/coverage.
- Ablation: no B-Rep descriptors vs descriptors; simple stock vs diverse stock; straight slot only vs swept slot; no interference filter vs interference filter.
- External STEP test: feature-set precision/recall/F1, selectable-region accuracy, user requirement matching success rate.
- Failure analysis: 曲面槽、侧向复杂孔、交叉特征、倒角/圆角边界。

---

## Execution Result - 2026-06-04

Implemented in the current workspace.

Key outputs:

- `configs/static_structural_v2_3_pilot_100.yaml`
- `utils/path_geometry.py`
- `utils/slot_generation_audit.py`
- `output_generation_v2_3_pilot_100`
- `output_generation_v2_3_pilot_100/reports/step_solid_validation.json`
- `output_generation_v2_3_pilot_100/reports/slot_generation_audit.json`
- `output_generation_v2_3_pilot_100/reports/manual_review_targets.md`

Verification:

- `W:\python\envs\feagpt\python.exe -m unittest discover -s tests -p "test_*.py"`: 93 tests passed.
- `W:\python\envs\feagpt\python.exe -m py_compile features\swept_slot.py features\swept_channel_slot.py utils\feature_footprint.py utils\path_geometry.py utils\slot_generation_audit.py`: passed.
- STEP solid validation: `status=ok`, `step_count=100`, `blocking_issue_count=0`.
- Slot audit: `slot_total=45`, `path_kind_counts={"circular_arc":20,"line":25}`, `empty_slot_feature_set_count=0`.

Implementation note:

- `polyline_capsule_footprint()` keeps precise `tube_radius` for curve-slot conflict distance and also exposes legacy `cx/cy/radius` as a bounding circle because several existing feature generators still read `occupied_zones` through that older implicit contract.
