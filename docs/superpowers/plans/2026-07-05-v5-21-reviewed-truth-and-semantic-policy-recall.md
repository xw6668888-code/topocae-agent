# v5.21 Reviewed Truth and Semantic Policy Recall Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consume the user's v5.20 semantic review decisions, refresh the s3 area-truth JSON from the updated Excel truth table, rerun strict s3 recall, add a secondary evaluation-only hole-family policy metric, and regenerate the proposal-miss atlas for the remaining strict failures.

**Architecture:** Do not train a new model and do not change inference output labels in this version. First make the evaluation inputs correct, then compute strict metrics and a clearly separated semantic-policy metric, then diagnose the remaining strict failures. Use lightweight `.xlsx` XML parsing instead of adding `openpyxl`, because the current `feagpt` environment does not contain spreadsheet libraries.

**Tech Stack:** Python 3 from `W:/python/envs/feagpt/python.exe`, `unittest`, JSON, CSV, Markdown, ZIP/XML parsing for `.xlsx`, existing `object_proposal_recall_report.py`, existing `proposal_miss_atlas.py`.

---

## 0. Review of v5.20 Output

v5.20 satisfies the previous implementation plan.

Verification command:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_semantic_review_decision_ingest tests.test_semantic_cross_macro_audit -v
```

Observed:

```text
Ran 15 tests
OK
```

The user-filled review file is:

```text
W:\python\FEA\dataset_generator\output_real_step_s3_v5_20_user_semantic_review\reports\semantic_review_template.csv
```

The completed decisions are:

```text
accept_family_match: 2
keep_strict: 18
truth_label_should_change: 4
```

Category/decision breakdown:

```text
hole_subfeature_vs_composite_hole + accept_family_match: 2
hole_subfeature_vs_composite_hole + truth_label_should_change: 2
circular_recess_or_hole_conflict + truth_label_should_change: 2
transition_vs_nontransition_conflict + keep_strict: 11
thinwall_rib_vs_boss_or_web + keep_strict: 5
rectangular_recess_slot_vs_pocket + keep_strict: 1
unknown_cross_macro_conflict + keep_strict: 1
```

Interpretation:

- The agent's v5.20 recommendation `return_to_proposal_generation_repair_for_remaining_no_same_macro_misses` is directionally correct.
- However, before touching proposal-generation code, v5.21 must first apply the 4 truth corrections and report the 2 accepted hole-family rows as an evaluation-only secondary metric.
- Because 18/24 reviewed rows remain strict failures, a broad semantic relaxation is not scientifically defensible.
- The primary paper metric should remain strict macro-level recall; the hole-family policy may be reported as an auxiliary CAE-region-selection metric.

## 1. Human Task

No new STEP modelling is required for v5.21.

The user only needs to review the generated v5.21 reports after the next agent runs this plan:

```text
output_real_step_s3_v5_21_reviewed_truth_semantic_policy\reports\truth_export_diff.md
output_real_step_s3_v5_21_reviewed_truth_semantic_policy\reports\semantic_policy_recall_s3_v5_21.md
output_real_step_s3_v5_21_reviewed_truth_semantic_policy\reports\v5_21_decision.md
```

If `truth_export_diff.md` shows unexpected changes outside `part12`, the user should inspect the Excel truth table. Otherwise, the agent can continue from the v5.21 decision report.

## 2. Scope Control

### Create

- `export_area_truth_from_xlsx.py`
- `tests/test_export_area_truth_from_xlsx.py`
- `apply_semantic_review_policy_to_recall_report.py`
- `tests/test_apply_semantic_review_policy_to_recall_report.py`
- `configs/test_part_s3_area_truth_v5_21_reviewed.json`
- `output_real_step_s3_v5_21_reviewed_truth_semantic_policy/reports/truth_export_diff.md`
- `output_real_step_s3_v5_21_reviewed_truth_semantic_policy/reports/proposal_recall_s3_v5_21_reviewed_truth.json`
- `output_real_step_s3_v5_21_reviewed_truth_semantic_policy/reports/proposal_recall_s3_v5_21_reviewed_truth.md`
- `output_real_step_s3_v5_21_reviewed_truth_semantic_policy/reports/semantic_policy_recall_s3_v5_21.json`
- `output_real_step_s3_v5_21_reviewed_truth_semantic_policy/reports/semantic_policy_recall_s3_v5_21.md`
- `output_real_step_s3_v5_21_reviewed_truth_semantic_policy/reports/v5_21_proposal_miss_atlas.json`
- `output_real_step_s3_v5_21_reviewed_truth_semantic_policy/reports/v5_21_proposal_miss_atlas.md`
- `output_real_step_s3_v5_21_reviewed_truth_semantic_policy/reports/v5_21_decision.md`

### Do Not Modify

- `train_gnn_baseline.py`
- `train_candidate_topology_classifier.py`
- any model checkpoint
- any graph extractor
- any STEP generator
- `utils/object_candidate_proposer.py`
- `utils/topology_postprocess.py`
- `utils/instance_candidate_features.py`
- `utils/instance_candidate_postprocess.py`
- existing report directories from v5.13 through v5.20

This version is evaluation-input correction and diagnostic reporting only.

## 3. Fixed Inputs

Use these inputs exactly:

```text
W:\python\FEA\dataset_generator\test_part_s3\Feature_Comparison_Table_s3_mm2.xlsx
W:\python\FEA\dataset_generator\configs\test_part_s3_area_truth_part20_mm2.json
W:\python\FEA\dataset_generator\output_real_step_s3_v5_20_user_semantic_review\reports\semantic_review_template.csv
W:\python\FEA\dataset_generator\output_real_step_s3_v5_13_transition_opening_proposals\reports\proposal_recall_s3_v5_13.json
```

Reuse the v5.13 recall-report runtime inputs:

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

## 4. Expected v5.21 Outcome

Expected logic after user review:

- 4 rows marked `truth_label_should_change` should disappear as old-truth failures after regenerating truth JSON from the updated Excel table.
- 2 rows marked `accept_family_match` should be counted only in a secondary semantic-policy metric, not in the strict macro metric.
- The remaining failures should be diagnosed by `proposal_miss_atlas.py`.
- If the new atlas is still dominated by `no_same_macro_candidates`, the next implementation plan should target proposal generation, not classifier weighting.

## 5. Task 1: Add Tests for Excel Area Truth Export

**Files:**

- Create: `tests/test_export_area_truth_from_xlsx.py`
- Create: `export_area_truth_from_xlsx.py`

- [ ] **Step 1: Create failing tests**

Create `tests/test_export_area_truth_from_xlsx.py`:

```python
from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from export_area_truth_from_xlsx import (
    build_area_truth_payload,
    parse_xlsx_rows,
    write_area_truth_json,
)


def write_minimal_xlsx(path: Path) -> None:
    shared_strings = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="12" uniqueCount="9">
  <si><t>part1</t></si>
  <si><t>Area</t></si>
  <si><t>Macro</t></si>
  <si><t>number of faces</t></si>
  <si><t>through_hole</t></si>
  <si><t>fillet</t></si>
  <si><t>part2</t></si>
  <si><t>counterbore_hole</t></si>
  <si><t>rib</t></si>
</sst>"""
    sheet = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>
    <row r="1"><c r="A1" t="s"><v>0</v></c></row>
    <row r="2"><c r="A2" t="s"><v>1</v></c><c r="B2" t="s"><v>2</v></c><c r="C2" t="s"><v>3</v></c></row>
    <row r="3"><c r="A3"><v>12.5</v></c><c r="B3" t="s"><v>4</v></c><c r="C3"><v>1</v></c></row>
    <row r="4"><c r="A4"><v>6.25</v></c><c r="B4" t="s"><v>5</v></c><c r="C4"><v>1</v></c></row>
    <row r="6"><c r="A6" t="s"><v>6</v></c></row>
    <row r="7"><c r="A7" t="s"><v>1</v></c><c r="B7" t="s"><v>2</v></c><c r="C7" t="s"><v>3</v></c></row>
    <row r="8"><c r="A8"><v>99.0</v></c><c r="B8" t="s"><v>7</v></c><c r="C8"><v>3</v></c></row>
    <row r="9"><c r="A9"><v>44.0</v></c><c r="B9" t="s"><v>8</v></c><c r="C9"><v>2</v></c></row>
  </sheetData>
</worksheet>"""
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("xl/sharedStrings.xml", shared_strings)
        zf.writestr("xl/worksheets/sheet1.xml", sheet)


class ExportAreaTruthFromXlsxTest(unittest.TestCase):
    def test_parse_xlsx_rows_groups_part_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            xlsx_path = Path(tmp) / "truth.xlsx"
            write_minimal_xlsx(xlsx_path)

            rows = parse_xlsx_rows(xlsx_path)

            self.assertEqual(rows[0], {"sample": "part1", "area": 12.5, "macro": "through_hole", "face_count": 1})
            self.assertEqual(rows[1], {"sample": "part1", "area": 6.25, "macro": "fillet", "face_count": 1})
            self.assertEqual(rows[2], {"sample": "part2", "area": 99.0, "macro": "counterbore_hole", "face_count": 3})
            self.assertEqual(rows[3], {"sample": "part2", "area": 44.0, "macro": "rib", "face_count": 2})

    def test_build_area_truth_payload_matches_existing_schema(self):
        rows = [
            {"sample": "part1", "area": 12.5, "macro": "through_hole", "face_count": 1},
            {"sample": "part1", "area": 6.25, "macro": "fillet", "face_count": 1},
        ]

        payload = build_area_truth_payload(rows, source_xlsx="truth.xlsx")

        self.assertEqual(payload["schema"], "cae_area_truth_table_v1")
        self.assertEqual(payload["source_xlsx"], "truth.xlsx")
        self.assertEqual(payload["samples"]["part1"][0]["macro"], "through_hole")
        self.assertEqual(payload["samples"]["part1"][1]["face_count"], 1)

    def test_write_area_truth_json_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "truth.json"
            payload = build_area_truth_payload(
                [{"sample": "part12", "area": 125.6637, "macro": "fillet", "face_count": 1}],
                source_xlsx="Feature_Comparison_Table_s3_mm2.xlsx",
            )

            write_area_truth_json(output_path, payload)

            loaded = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["samples"]["part12"][0]["macro"], "fillet")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_export_area_truth_from_xlsx -v
```

Expected before implementation:

```text
ERROR: ModuleNotFoundError: No module named 'export_area_truth_from_xlsx'
```

## 6. Task 2: Implement Excel Area Truth Exporter

**Files:**

- Modify: `export_area_truth_from_xlsx.py`

- [ ] **Step 1: Create implementation**

Create `export_area_truth_from_xlsx.py`:

```python
from __future__ import annotations

import argparse
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
PART_RE = re.compile(r"^part\d+$", re.IGNORECASE)


def cell_column(cell_ref: str) -> str:
    return re.sub(r"\d+", "", cell_ref or "")


def read_zip_text(xlsx_path: Path, member: str) -> str:
    with zipfile.ZipFile(xlsx_path, "r") as zf:
        with zf.open(member, "r") as handle:
            return handle.read().decode("utf-8")


def load_shared_strings(xlsx_path: Path) -> list[str]:
    xml_text = read_zip_text(xlsx_path, "xl/sharedStrings.xml")
    root = ET.fromstring(xml_text)
    values: list[str] = []
    for item in root.findall("x:si", NS):
        parts = [node.text or "" for node in item.findall(".//x:t", NS)]
        values.append("".join(parts))
    return values


def parse_cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    value_node = cell.find("x:v", NS)
    if value_node is None:
        return ""
    value = value_node.text or ""
    if cell.attrib.get("t") == "s":
        return shared_strings[int(value)]
    return value


def parse_xlsx_rows(xlsx_path: Path) -> list[dict[str, Any]]:
    shared_strings = load_shared_strings(xlsx_path)
    sheet_xml = read_zip_text(xlsx_path, "xl/worksheets/sheet1.xml")
    root = ET.fromstring(sheet_xml)
    current_sample = ""
    rows: list[dict[str, Any]] = []
    for row in root.findall(".//x:row", NS):
        values: dict[str, str] = {}
        for cell in row.findall("x:c", NS):
            column = cell_column(cell.attrib.get("r", ""))
            values[column] = parse_cell_value(cell, shared_strings).strip()
        first = values.get("A", "")
        second = values.get("B", "")
        if PART_RE.match(first) and second == "":
            current_sample = first.lower()
            continue
        if first.lower() == "area" or second.lower() == "macro":
            continue
        if not current_sample or not first or not second:
            continue
        rows.append(
            {
                "sample": current_sample,
                "area": float(first),
                "macro": second,
                "face_count": int(float(values.get("C", "1"))),
            }
        )
    return rows


def build_area_truth_payload(rows: list[dict[str, Any]], source_xlsx: str) -> dict[str, Any]:
    samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        samples[str(row["sample"])].append(
            {
                "area": float(row["area"]),
                "macro": str(row["macro"]),
                "face_count": int(row["face_count"]),
            }
        )
    return {
        "schema": "cae_area_truth_table_v1",
        "source_xlsx": source_xlsx,
        "unit_note": "Areas are read from Feature_Comparison_Table_s3_mm2.xlsx. Values are expected to be mm^2; part20 was already corrected by the user.",
        "samples": dict(sorted(samples.items())),
    }


def write_area_truth_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    samples = payload.get("samples", {})
    macro_counts: dict[str, int] = defaultdict(int)
    total = 0
    for entries in samples.values():
        total += len(entries)
        for entry in entries:
            macro_counts[str(entry.get("macro", ""))] += 1
    return {
        "sample_count": len(samples),
        "truth_count": total,
        "macro_counts": dict(sorted(macro_counts.items())),
    }


def diff_payloads(old_payload: dict[str, Any], new_payload: dict[str, Any]) -> dict[str, Any]:
    old_samples = old_payload.get("samples", {})
    new_samples = new_payload.get("samples", {})
    sample_names = sorted(set(old_samples) | set(new_samples))
    changed: list[dict[str, Any]] = []
    for sample in sample_names:
        old_entries = old_samples.get(sample, [])
        new_entries = new_samples.get(sample, [])
        if old_entries != new_entries:
            changed.append({"sample": sample, "old": old_entries, "new": new_entries})
    return {
        "changed_sample_count": len(changed),
        "changed_samples": changed,
    }


def write_diff_markdown(path: Path, old_payload: dict[str, Any], new_payload: dict[str, Any], diff: dict[str, Any]) -> None:
    old_summary = summarize_payload(old_payload)
    new_summary = summarize_payload(new_payload)
    lines = [
        "# v5.21 Truth Export Diff",
        "",
        "## Summary",
        "",
        f"- old_truth_count: `{old_summary['truth_count']}`",
        f"- new_truth_count: `{new_summary['truth_count']}`",
        f"- changed_sample_count: `{diff['changed_sample_count']}`",
        "",
        "## Changed Samples",
        "",
    ]
    for change in diff["changed_samples"]:
        lines.append(f"### {change['sample']}")
        lines.append("")
        lines.append("Old:")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(change["old"], indent=2, ensure_ascii=False))
        lines.append("```")
        lines.append("")
        lines.append("New:")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(change["new"], indent=2, ensure_ascii=False))
        lines.append("```")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export area truth JSON from the simple s3 Excel truth table.")
    parser.add_argument("--xlsx", type=Path, required=True)
    parser.add_argument("--old-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--diff-md", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = parse_xlsx_rows(args.xlsx)
    payload = build_area_truth_payload(rows, source_xlsx=str(args.xlsx))
    write_area_truth_json(args.output_json, payload)
    old_payload = load_json(args.old_json)
    diff = diff_payloads(old_payload, payload)
    write_diff_markdown(args.diff_md, old_payload, payload, diff)
    print(json.dumps({"summary": summarize_payload(payload), "diff": {"changed_sample_count": diff["changed_sample_count"]}}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run tests**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_export_area_truth_from_xlsx -v
```

Expected:

```text
OK
```

## 7. Task 3: Generate Reviewed s3 Truth JSON

**Files:**

- Generate: `configs/test_part_s3_area_truth_v5_21_reviewed.json`
- Generate: `output_real_step_s3_v5_21_reviewed_truth_semantic_policy/reports/truth_export_diff.md`

- [ ] **Step 1: Export reviewed truth JSON**

Run:

```powershell
W:\python\envs\feagpt\python.exe export_area_truth_from_xlsx.py `
  --xlsx test_part_s3\Feature_Comparison_Table_s3_mm2.xlsx `
  --old-json configs\test_part_s3_area_truth_part20_mm2.json `
  --output-json configs\test_part_s3_area_truth_v5_21_reviewed.json `
  --diff-md output_real_step_s3_v5_21_reviewed_truth_semantic_policy\reports\truth_export_diff.md
```

Expected:

```text
"sample_count": 21
```

- [ ] **Step 2: Verify part12 truth corrections**

Run:

```powershell
Select-String -LiteralPath configs\test_part_s3_area_truth_v5_21_reviewed.json -Pattern '"part12"' -Context 0,60
```

Expected within the part12 block:

```text
"area": 125.6637
"macro": "fillet"
"area": 157.0796
"macro": "fillet"
```

- [ ] **Step 3: Inspect truth diff**

Run:

```powershell
Get-Content -LiteralPath output_real_step_s3_v5_21_reviewed_truth_semantic_policy\reports\truth_export_diff.md -TotalCount 120
```

Expected:

```text
The diff must include part12.
```

If the diff includes many samples other than part12, stop and report the changed sample list to the user before continuing.

## 8. Task 4: Add Tests for Semantic Review Policy Metric

**Files:**

- Create: `tests/test_apply_semantic_review_policy_to_recall_report.py`
- Create: `apply_semantic_review_policy_to_recall_report.py`

- [ ] **Step 1: Create failing tests**

Create `tests/test_apply_semantic_review_policy_to_recall_report.py`:

```python
from __future__ import annotations

import unittest

from apply_semantic_review_policy_to_recall_report import (
    accepted_policy_keys,
    apply_semantic_policy,
    normalize_key_macro,
)


class ApplySemanticReviewPolicyToRecallReportTest(unittest.TestCase):
    def test_normalize_key_macro_unifies_hole_aliases(self):
        self.assertEqual(normalize_key_macro("through_hole"), "through_hole_round")
        self.assertEqual(normalize_key_macro("blind_hole"), "blind_hole_round")
        self.assertEqual(normalize_key_macro("cylindrical_boss"), "cylinder_boss")

    def test_accepted_policy_keys_uses_only_accept_family_match(self):
        rows = [
            {"sample": "part10", "truth_macro": "through_hole_round", "truth_area": "1912.3562", "user_decision": "accept_family_match"},
            {"sample": "part12", "truth_macro": "through_hole_round", "truth_area": "125.6637", "user_decision": "truth_label_should_change"},
            {"sample": "part18", "truth_macro": "slot_o", "truth_area": "3553.06", "user_decision": "keep_strict"},
        ]

        keys = accepted_policy_keys(rows)

        self.assertEqual(keys, {("part10", "through_hole_round", 1912.3562)})

    def test_apply_semantic_policy_recovers_only_matching_strict_failures(self):
        report = {
            "best_by_final_survival_recall": {
                "truth_count": 4,
                "final_survival_stage": {"matched_truth_count": 1, "recall": 0.25},
                "samples": [
                    {
                        "sample": "part10",
                        "truth_failures": [
                            {"macro": "through_hole_round", "area": 1912.3562, "face_count": 1},
                            {"macro": "fillet", "area": 88.0, "face_count": 1},
                        ],
                    },
                    {
                        "sample": "part11",
                        "truth_failures": [
                            {"macro": "through_hole_round", "area": 558.5587, "face_count": 1},
                        ],
                    },
                ],
            }
        }
        review_rows = [
            {"sample": "part10", "truth_macro": "through_hole_round", "truth_area": "1912.3562", "user_decision": "accept_family_match"},
            {"sample": "part11", "truth_macro": "through_hole_round", "truth_area": "558.5587", "user_decision": "accept_family_match"},
            {"sample": "part12", "truth_macro": "through_hole_round", "truth_area": "125.6637", "user_decision": "truth_label_should_change"},
        ]

        adjusted = apply_semantic_policy(report, review_rows, area_tolerance=0.02)

        self.assertEqual(adjusted["strict_matched_truth_count"], 1)
        self.assertEqual(adjusted["policy_recovered_count"], 2)
        self.assertEqual(adjusted["semantic_policy_matched_truth_count"], 3)
        self.assertEqual(adjusted["truth_count"], 4)
        self.assertAlmostEqual(adjusted["semantic_policy_recall"], 0.75)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_apply_semantic_review_policy_to_recall_report -v
```

Expected before implementation:

```text
ERROR: ModuleNotFoundError: No module named 'apply_semantic_review_policy_to_recall_report'
```

## 9. Task 5: Implement Evaluation-Only Semantic Policy Script

**Files:**

- Modify: `apply_semantic_review_policy_to_recall_report.py`

- [ ] **Step 1: Create implementation**

Create `apply_semantic_review_policy_to_recall_report.py`:

```python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


MACRO_ALIASES = {
    "through_hole": "through_hole_round",
    "blind_hole": "blind_hole_round",
    "stepped_hole": "stepped_hole_round",
    "cylindrical_boss": "cylinder_boss",
    "rectangle_boss": "rectangular_boss",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def read_review_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def normalize_key_macro(macro: str) -> str:
    token = str(macro or "").strip()
    return MACRO_ALIASES.get(token, token)


def rounded_area(value: Any) -> float:
    return round(float(value), 4)


def accepted_policy_keys(rows: list[dict[str, str]]) -> set[tuple[str, str, float]]:
    keys: set[tuple[str, str, float]] = set()
    for row in rows:
        if str(row.get("user_decision", "")).strip() != "accept_family_match":
            continue
        keys.add(
            (
                str(row.get("sample", "")).strip(),
                normalize_key_macro(str(row.get("truth_macro", "")).strip()),
                rounded_area(row.get("truth_area", "0")),
            )
        )
    return keys


def areas_match(a: float, b: float, area_tolerance: float) -> bool:
    return abs(a - b) <= area_tolerance


def apply_semantic_policy(report: dict[str, Any], review_rows: list[dict[str, str]], area_tolerance: float) -> dict[str, Any]:
    best = report.get("best_by_final_survival_recall") or {}
    strict_stage = best.get("final_survival_stage") or {}
    truth_count = int(best.get("truth_count", report.get("truth_count", 0)))
    strict_matched = int(strict_stage.get("matched_truth_count", 0))
    accepted_keys = accepted_policy_keys(review_rows)
    recovered: list[dict[str, Any]] = []
    for sample in best.get("samples", []):
        sample_name = str(sample.get("sample", ""))
        for failure in sample.get("truth_failures", []):
            failure_macro = normalize_key_macro(str(failure.get("macro", "")))
            failure_area = float(failure.get("area", 0.0))
            for key_sample, key_macro, key_area in accepted_keys:
                if key_sample == sample_name and key_macro == failure_macro and areas_match(key_area, failure_area, area_tolerance):
                    recovered.append(
                        {
                            "sample": sample_name,
                            "macro": failure_macro,
                            "area": failure_area,
                            "matched_policy_key": {"sample": key_sample, "macro": key_macro, "area": key_area},
                        }
                    )
                    break
    policy_recovered = len(recovered)
    semantic_matched = strict_matched + policy_recovered
    return {
        "schema": "semantic_review_policy_recall_v1",
        "truth_count": truth_count,
        "strict_matched_truth_count": strict_matched,
        "strict_recall": strict_matched / truth_count if truth_count else 0.0,
        "policy_recovered_count": policy_recovered,
        "semantic_policy_matched_truth_count": semantic_matched,
        "semantic_policy_recall": semantic_matched / truth_count if truth_count else 0.0,
        "accepted_policy_key_count": len(accepted_keys),
        "recovered_rows": recovered,
        "note": "Evaluation-only metric. This does not change model predictions or strict macro labels.",
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# v5.21 Semantic Policy Recall",
        "",
        "> This is an evaluation-only CAE hole-family metric. It does not change prediction labels.",
        "",
        "## Summary",
        "",
        f"- truth_count: `{payload['truth_count']}`",
        f"- strict_matched_truth_count: `{payload['strict_matched_truth_count']}`",
        f"- strict_recall: `{payload['strict_recall']:.4f}`",
        f"- policy_recovered_count: `{payload['policy_recovered_count']}`",
        f"- semantic_policy_matched_truth_count: `{payload['semantic_policy_matched_truth_count']}`",
        f"- semantic_policy_recall: `{payload['semantic_policy_recall']:.4f}`",
        "",
        "## Recovered Rows",
        "",
        "| Sample | Macro | Area |",
        "|--------|-------|------|",
    ]
    for row in payload["recovered_rows"]:
        lines.append(f"| {row['sample']} | {row['macro']} | {row['area']:.4f} |")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply user-approved semantic review policy as an evaluation-only recall metric.")
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--area-tolerance", type=float, default=0.02)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = load_json(args.report_json)
    review_rows = read_review_rows(args.review_csv)
    payload = apply_semantic_policy(report, review_rows, args.area_tolerance)
    write_json(args.output_json, payload)
    write_markdown(args.output_md, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run tests**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_apply_semantic_review_policy_to_recall_report -v
```

Expected:

```text
OK
```

## 10. Task 6: Rerun Strict s3 Recall with Reviewed Truth

**Files:**

- Generate: `output_real_step_s3_v5_21_reviewed_truth_semantic_policy/reports/proposal_recall_s3_v5_21_reviewed_truth.json`
- Generate: `output_real_step_s3_v5_21_reviewed_truth_semantic_policy/reports/proposal_recall_s3_v5_21_reviewed_truth.md`

- [ ] **Step 1: Run strict recall report**

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
  --output-json output_real_step_s3_v5_21_reviewed_truth_semantic_policy\reports\proposal_recall_s3_v5_21_reviewed_truth.json `
  --output-md output_real_step_s3_v5_21_reviewed_truth_semantic_policy\reports\proposal_recall_s3_v5_21_reviewed_truth.md
```

Expected:

```text
Report written to output_real_step_s3_v5_21_reviewed_truth_semantic_policy\reports\proposal_recall_s3_v5_21_reviewed_truth.json
```

- [ ] **Step 2: Inspect strict recall**

Run:

```powershell
Get-Content -LiteralPath output_real_step_s3_v5_21_reviewed_truth_semantic_policy\reports\proposal_recall_s3_v5_21_reviewed_truth.md -TotalCount 80
```

Expected:

```text
The report uses configs\test_part_s3_area_truth_v5_21_reviewed.json.
```

## 11. Task 7: Apply Evaluation-Only Semantic Policy

**Files:**

- Generate: `output_real_step_s3_v5_21_reviewed_truth_semantic_policy/reports/semantic_policy_recall_s3_v5_21.json`
- Generate: `output_real_step_s3_v5_21_reviewed_truth_semantic_policy/reports/semantic_policy_recall_s3_v5_21.md`

- [ ] **Step 1: Run semantic policy adjustment**

Run:

```powershell
W:\python\envs\feagpt\python.exe apply_semantic_review_policy_to_recall_report.py `
  --report-json output_real_step_s3_v5_21_reviewed_truth_semantic_policy\reports\proposal_recall_s3_v5_21_reviewed_truth.json `
  --review-csv output_real_step_s3_v5_20_user_semantic_review\reports\semantic_review_template.csv `
  --output-json output_real_step_s3_v5_21_reviewed_truth_semantic_policy\reports\semantic_policy_recall_s3_v5_21.json `
  --output-md output_real_step_s3_v5_21_reviewed_truth_semantic_policy\reports\semantic_policy_recall_s3_v5_21.md `
  --area-tolerance 0.02
```

Expected:

```text
"policy_recovered_count": 2
```

If `policy_recovered_count` is not 2, inspect whether the reviewed-truth strict report already matched one of the accepted rows. Report the observed count and continue.

## 12. Task 8: Regenerate Proposal Miss Atlas with Reviewed Truth

**Files:**

- Generate: `output_real_step_s3_v5_21_reviewed_truth_semantic_policy/reports/v5_21_proposal_miss_atlas.json`
- Generate: `output_real_step_s3_v5_21_reviewed_truth_semantic_policy/reports/v5_21_proposal_miss_atlas.md`
- Generate: `output_real_step_s3_v5_21_reviewed_truth_semantic_policy/reports/v5_18_repair_gate.md`
- Generate: `output_real_step_s3_v5_21_reviewed_truth_semantic_policy/reports/v5_18_next_step_recommendation.md`

- [ ] **Step 1: Run proposal miss atlas**

Run:

```powershell
W:\python\envs\feagpt\python.exe proposal_miss_atlas.py `
  --report v5_21=output_real_step_s3_v5_21_reviewed_truth_semantic_policy\reports\proposal_recall_s3_v5_21_reviewed_truth.json `
  --output-dir output_real_step_s3_v5_21_reviewed_truth_semantic_policy\reports
```

Expected:

```text
The console prints a JSON summary with v5_21.
```

- [ ] **Step 2: Rename atlas files for clarity if needed**

If the script writes:

```text
v5_21_proposal_miss_atlas.json
v5_21_proposal_miss_atlas.md
```

leave them unchanged.

If it writes a different name, do not manually rename by shell move. Instead, report the exact generated filenames in the final response.

- [ ] **Step 3: Inspect atlas summary**

Run:

```powershell
Get-Content -LiteralPath output_real_step_s3_v5_21_reviewed_truth_semantic_policy\reports\v5_21_proposal_miss_atlas.md -TotalCount 120
```

Expected:

```text
The report lists miss-cause counts and top missed macros.
```

## 13. Task 9: Write v5.21 Decision Memo

**Files:**

- Create: `write_v5_21_decision.py`
- Create: `tests/test_write_v5_21_decision.py`
- Generate: `output_real_step_s3_v5_21_reviewed_truth_semantic_policy/reports/v5_21_decision.md`

- [ ] **Step 1: Create failing tests**

Create `tests/test_write_v5_21_decision.py`:

```python
from __future__ import annotations

import unittest

from write_v5_21_decision import choose_next_recommendation


class WriteV521DecisionTest(unittest.TestCase):
    def test_recommends_proposer_repair_for_no_same_macro_dominance(self):
        atlas_summary = {
            "by_miss_cause": {
                "no_same_macro_candidates": 20,
                "cross_macro_area_compatible": 5,
            },
            "by_macro": {
                "rib": 12,
                "fillet": 8,
            },
        }
        recommendation, reason = choose_next_recommendation(atlas_summary)
        self.assertEqual(recommendation, "targeted_proposer_repair")
        self.assertIn("no_same_macro_candidates", reason)

    def test_recommends_semantic_review_if_cross_macro_still_dominates(self):
        atlas_summary = {
            "by_miss_cause": {
                "cross_macro_area_compatible": 20,
                "no_same_macro_candidates": 5,
            },
            "by_macro": {},
        }
        recommendation, reason = choose_next_recommendation(atlas_summary)
        self.assertEqual(recommendation, "inspect_remaining_cross_macro_semantics")
        self.assertIn("cross_macro_area_compatible", reason)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_write_v5_21_decision -v
```

Expected before implementation:

```text
ERROR: ModuleNotFoundError: No module named 'write_v5_21_decision'
```

- [ ] **Step 3: Create implementation**

Create `write_v5_21_decision.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def choose_next_recommendation(atlas_summary: dict[str, Any]) -> tuple[str, str]:
    causes = atlas_summary.get("by_miss_cause", {})
    no_same = int(causes.get("no_same_macro_candidates", 0))
    cross = int(causes.get("cross_macro_area_compatible", 0))
    near = int(causes.get("same_macro_area_near_miss", 0)) + int(causes.get("cross_macro_area_near_miss", 0))
    if no_same >= max(cross, near) and no_same > 0:
        return "targeted_proposer_repair", "Reviewed truth and semantic policy leave no_same_macro_candidates as the dominant strict failure mode."
    if cross >= max(no_same, near) and cross > 0:
        return "inspect_remaining_cross_macro_semantics", "cross_macro_area_compatible remains dominant after v5.20 review, so inspect remaining macro semantics before coding."
    if near > 0:
        return "area_tolerance_or_boundary_refinement", "Near misses dominate, so inspect boundary grouping and area tolerance before new model work."
    return "hold_current_pipeline_and_write_results", "No dominant repair category remains."


def write_decision(path: Path, strict_report: dict[str, Any], policy_report: dict[str, Any], atlas: dict[str, Any]) -> None:
    best = strict_report.get("best_by_final_survival_recall") or {}
    stage = best.get("final_survival_stage") or {}
    atlas_summary = atlas.get("summary") or {}
    recommendation, reason = choose_next_recommendation(atlas_summary)
    lines = [
        "# v5.21 Decision",
        "",
        "## Recommendation",
        "",
        f"**{recommendation}**",
        "",
        "## Reasoning",
        "",
        reason,
        "",
        "## Strict Recall",
        "",
        f"- truth_count: `{best.get('truth_count', strict_report.get('truth_count', 'N/A'))}`",
        f"- matched_truth_count: `{stage.get('matched_truth_count', 'N/A')}`",
        f"- recall: `{float(stage.get('recall', 0.0)):.4f}`",
        "",
        "## Semantic Policy Recall",
        "",
        f"- policy_recovered_count: `{policy_report.get('policy_recovered_count', 0)}`",
        f"- semantic_policy_recall: `{float(policy_report.get('semantic_policy_recall', 0.0)):.4f}`",
        "",
        "## Remaining Miss Causes",
        "",
        "| Cause | Count |",
        "|-------|-------|",
    ]
    for cause, count in (atlas_summary.get("by_miss_cause") or {}).items():
        lines.append(f"| {cause} | {count} |")
    lines.extend(["", "## Remaining Missed Macros", "", "| Macro | Count |", "|-------|-------|"])
    for macro, count in (atlas_summary.get("by_macro") or {}).items():
        lines.append(f"| {macro} | {count} |")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write v5.21 decision memo from reviewed-truth strict recall, semantic policy recall, and proposal miss atlas.")
    parser.add_argument("--strict-report-json", type=Path, required=True)
    parser.add_argument("--semantic-policy-json", type=Path, required=True)
    parser.add_argument("--atlas-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    strict_report = load_json(args.strict_report_json)
    policy_report = load_json(args.semantic_policy_json)
    atlas = load_json(args.atlas_json)
    write_decision(args.output_md, strict_report, policy_report, atlas)
    print(f"Decision written to {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_write_v5_21_decision -v
```

Expected:

```text
OK
```

- [ ] **Step 5: Generate decision memo**

Run:

```powershell
W:\python\envs\feagpt\python.exe write_v5_21_decision.py `
  --strict-report-json output_real_step_s3_v5_21_reviewed_truth_semantic_policy\reports\proposal_recall_s3_v5_21_reviewed_truth.json `
  --semantic-policy-json output_real_step_s3_v5_21_reviewed_truth_semantic_policy\reports\semantic_policy_recall_s3_v5_21.json `
  --atlas-json output_real_step_s3_v5_21_reviewed_truth_semantic_policy\reports\v5_21_proposal_miss_atlas.json `
  --output-md output_real_step_s3_v5_21_reviewed_truth_semantic_policy\reports\v5_21_decision.md
```

Expected:

```text
Decision written to output_real_step_s3_v5_21_reviewed_truth_semantic_policy\reports\v5_21_decision.md
```

## 14. Task 10: Final Verification

**Files:**

- Verify all files created above.

- [ ] **Step 1: Run all new tests plus review-ingestion tests**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest `
  tests.test_export_area_truth_from_xlsx `
  tests.test_apply_semantic_review_policy_to_recall_report `
  tests.test_write_v5_21_decision `
  tests.test_semantic_review_decision_ingest `
  tests.test_semantic_cross_macro_audit -v
```

Expected:

```text
OK
```

- [ ] **Step 2: Verify expected output files**

Run:

```powershell
Get-ChildItem -LiteralPath output_real_step_s3_v5_21_reviewed_truth_semantic_policy\reports | Select-Object Name,Length,LastWriteTime
```

Expected files must include:

```text
truth_export_diff.md
proposal_recall_s3_v5_21_reviewed_truth.json
proposal_recall_s3_v5_21_reviewed_truth.md
semantic_policy_recall_s3_v5_21.json
semantic_policy_recall_s3_v5_21.md
v5_21_proposal_miss_atlas.json
v5_21_proposal_miss_atlas.md
v5_21_decision.md
```

- [ ] **Step 3: Final response requirements**

The implementing agent must report:

```text
1. Test command and result.
2. Whether truth JSON was refreshed from Feature_Comparison_Table_s3_mm2.xlsx.
3. Whether part12 now contains fillet rows at 125.6637 and 157.0796.
4. Strict v5.21 recall and truth count.
5. Semantic-policy recovered count and semantic-policy recall.
6. Top remaining miss causes from v5_21_proposal_miss_atlas.md.
7. The recommendation from v5_21_decision.md.
8. Confirmation that no training, inference, graph extraction, or generator files were modified.
```

## 15. How I Will Review This Work

I will check:

```text
export_area_truth_from_xlsx.py
tests/test_export_area_truth_from_xlsx.py
apply_semantic_review_policy_to_recall_report.py
tests/test_apply_semantic_review_policy_to_recall_report.py
write_v5_21_decision.py
tests/test_write_v5_21_decision.py
configs/test_part_s3_area_truth_v5_21_reviewed.json
output_real_step_s3_v5_21_reviewed_truth_semantic_policy\reports\truth_export_diff.md
output_real_step_s3_v5_21_reviewed_truth_semantic_policy\reports\proposal_recall_s3_v5_21_reviewed_truth.json
output_real_step_s3_v5_21_reviewed_truth_semantic_policy\reports\semantic_policy_recall_s3_v5_21.json
output_real_step_s3_v5_21_reviewed_truth_semantic_policy\reports\v5_21_proposal_miss_atlas.json
output_real_step_s3_v5_21_reviewed_truth_semantic_policy\reports\v5_21_decision.md
```

I will run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest `
  tests.test_export_area_truth_from_xlsx `
  tests.test_apply_semantic_review_policy_to_recall_report `
  tests.test_write_v5_21_decision `
  tests.test_semantic_review_decision_ingest `
  tests.test_semantic_cross_macro_audit -v
```

I will reject the implementation if:

- It modifies model training or inference files in v5.21.
- It changes prediction labels to make semantic-policy recall look better.
- It keeps using `configs/test_part_s3_area_truth_part20_mm2.json` after the user updated the Excel truth table.
- It merges strict macro recall and semantic-policy recall into one number.
- It ignores the 18 `keep_strict` rows from the user review.

## 16. Next Plan After v5.21

Choose the next version based on `v5_21_decision.md`:

- If recommendation is `targeted_proposer_repair`, write v5.22 against `utils/object_candidate_proposer.py` and related topology proposer helpers.
- If recommendation is `inspect_remaining_cross_macro_semantics`, create another small review template only for the newly remaining cross-macro rows.
- If recommendation is `area_tolerance_or_boundary_refinement`, diagnose area grouping and candidate boundary splitting/merging before model changes.
- If recommendation is `hold_current_pipeline_and_write_results`, stop engineering iterations and begin writing the method/experiment section for the paper.

## 17. English Handoff Summary

v5.20 showed that most cross-macro compatible misses must remain strict failures. v5.21 must first refresh the s3 truth JSON from the user's updated Excel file, then rerun strict recall, then apply the two accepted hole-family decisions as a separate evaluation-only metric. Do not train or change inference labels. The output of v5.21 is a clean strict baseline, a small auxiliary semantic-policy metric, and an updated proposal-miss atlas that determines whether v5.22 should repair proposers or stop engineering and write the paper results.
