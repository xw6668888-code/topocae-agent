from __future__ import annotations

import hashlib
import json
import os
import unittest
from pathlib import Path, PureWindowsPath
from unittest.mock import patch

from run_p4_a1_oracle_gate import _load_memory
from utils.cae_language_grounding.evaluation import MemoryCache
from utils.cae_language_grounding.records import iter_language_examples
from utils.workspace_paths import resolve_workspace_path


ROOT = Path(__file__).resolve().parents[1]
FAKE_ORIGINAL_ROOT = PureWindowsPath(r"Z:\topocae\dataset_generator")


def _fake_windows_path(path: Path) -> str:
    relative = path.resolve().relative_to(ROOT.resolve())
    return str(FAKE_ORIGINAL_ROOT.joinpath(*relative.parts))


class P4PathPortabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.example = next(iter(iter_language_examples("train")))
        cls.graph_path = resolve_workspace_path(str(cls.example.source_ref["graph_path"]))
        cls.raw_before = cls.graph_path.read_bytes()
        cls.fake_graph_path = _fake_windows_path(cls.graph_path)
        cls.environment = {
            "TOPOCAE_ORIGINAL_WORKSPACE_ROOT": str(FAKE_ORIGINAL_ROOT),
            "TOPOCAE_WORKSPACE_ROOT": str(ROOT),
        }

    def test_memory_cache_loads_real_p3_graph_through_windows_mapping(self) -> None:
        with patch.dict(os.environ, self.environment):
            memory, instance_map = MemoryCache().get(self.fake_graph_path)
        self.assertTrue(memory["objects"])
        self.assertEqual(len(memory["objects"]), len(instance_map))
        self.assertEqual(self.raw_before, self.graph_path.read_bytes())
        self.assertEqual(
            hashlib.sha256(self.raw_before).hexdigest(),
            hashlib.sha256(self.graph_path.read_bytes()).hexdigest(),
        )

    def test_oracle_reproduction_loader_uses_same_mapping(self) -> None:
        from utils.cae_intent import load_feature_role_policy

        with patch.dict(os.environ, self.environment):
            memory, graph, instance_map = _load_memory(
                self.fake_graph_path,
                policy=load_feature_role_policy(),
                cache={},
            )
        self.assertEqual(json.loads(self.raw_before.decode("utf-8")), graph)
        self.assertTrue(memory["objects"])
        self.assertEqual(len(memory["objects"]), len(instance_map))
        self.assertEqual(self.raw_before, self.graph_path.read_bytes())


if __name__ == "__main__":
    unittest.main()
