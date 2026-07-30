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
