from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path, PureWindowsPath
from unittest.mock import patch

import verify_p4_a1_frozen_assets
from utils.workspace_paths import resolve_workspace_path


ROOT = Path(__file__).resolve().parents[1]
FAKE_ORIGINAL_ROOT = PureWindowsPath(r"Z:\topocae\dataset_generator")


def fake_windows_path(path: Path) -> str:
    relative = path.resolve().relative_to(ROOT.resolve())
    return str(FAKE_ORIGINAL_ROOT.joinpath(*relative.parts))


class M1PathIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.environment = {
            "TOPOCAE_ORIGINAL_WORKSPACE_ROOT": str(FAKE_ORIGINAL_ROOT),
            "TOPOCAE_WORKSPACE_ROOT": str(ROOT),
        }

    def test_frozen_asset_verifier_maps_windows_asset_path(self) -> None:
        asset = ROOT / "M1_A1_PATH_PORTABILITY_FINDINGS.zh-CN.md"
        raw = asset.read_bytes()
        with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
            manifest_path = Path(temp_dir) / "freeze_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "assets": [
                            {
                                "path": fake_windows_path(asset),
                                "bytes": len(raw),
                                "sha256": hashlib.sha256(raw).hexdigest(),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, self.environment), patch.object(
                verify_p4_a1_frozen_assets,
                "MANIFESTS",
                (("TEST", manifest_path),),
            ):
                result = verify_p4_a1_frozen_assets.verify()
        self.assertEqual("PASS", result["status"])
        self.assertEqual(1, result["manifests"][0]["matching_count"])
        self.assertEqual(0, result["manifests"][0]["change_count"])

    def test_resolver_maps_real_p3_graph_and_geometry_without_rewriting_record(self) -> None:
        pilot_path = (
            ROOT / "output_cae_query_ground_v1" / "pilot" / "canonical" / "train.jsonl"
        )
        with pilot_path.open("r", encoding="utf-8") as handle:
            record = copy.deepcopy(json.loads(next(handle)))
        original_graph = resolve_workspace_path(str(record["source_ref"]["graph_path"]))
        original_geometry = resolve_workspace_path(str(record["source_ref"]["geometry_path"]))
        graph_before = original_graph.read_bytes()
        geometry_before = original_geometry.read_bytes()
        record["source_ref"]["graph_path"] = fake_windows_path(original_graph)
        record["source_ref"]["geometry_path"] = fake_windows_path(original_geometry)
        serialized_before = json.dumps(record, ensure_ascii=False, sort_keys=True)

        with patch.dict(os.environ, self.environment):
            resolved_graph = resolve_workspace_path(record["source_ref"]["graph_path"])
            resolved_geometry = resolve_workspace_path(record["source_ref"]["geometry_path"])

        self.assertEqual(original_graph.resolve(), resolved_graph)
        self.assertEqual(original_geometry.resolve(), resolved_geometry)
        self.assertEqual(serialized_before, json.dumps(record, ensure_ascii=False, sort_keys=True))
        self.assertEqual(graph_before, original_graph.read_bytes())
        self.assertEqual(geometry_before, original_geometry.read_bytes())


if __name__ == "__main__":
    unittest.main()
