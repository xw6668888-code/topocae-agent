from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class CandidateAblationImportTests(unittest.TestCase):
    def test_balanced_module_is_imported_only_inside_balanced_branch(self) -> None:
        source = (ROOT / "train_p1_c_candidate_classifier_strict_v2.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        top_level_imports = [
            alias.name
            for node in tree.body
            if isinstance(node, ast.Import)
            for alias in node.names
        ]
        self.assertNotIn("train_p1_c_candidate_classifier_v2", top_level_imports)
        self.assertIn("if bool(spec[\"candidate_instance_balanced_loss\"])", source)
        self.assertIn("import train_p1_b_candidate_classifier as entrypoint_module", source)


if __name__ == "__main__":
    unittest.main()
