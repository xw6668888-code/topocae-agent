"""Export area truth JSON from s3 Excel truth table via ZIP/XML parsing."""

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
