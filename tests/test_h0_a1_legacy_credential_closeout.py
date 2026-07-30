from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


LEGACY_SCRIPT = Path(r"W:\python\FEA\cae\step3_topology_agent.py")
ENVIRONMENT_VARIABLE = "TOPOCAE_LLM_API_KEY"


def _load_legacy_module_without_side_effects(*, credential: str | None):
    tree = ast.parse(LEGACY_SCRIPT.read_text(encoding="utf-8"))
    safe_body = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
            if "os" not in names:
                safe_body.append(node)
        elif isinstance(node, ast.ImportFrom):
            if node.module != "openai":
                safe_body.append(node)
        elif isinstance(node, (ast.Assign, ast.FunctionDef)):
            safe_body.append(node)

    def reject_client_construction(*args, **kwargs):
        raise AssertionError("client construction must not occur without a credential")

    namespace = {
        "__name__": "legacy_closeout_test_module",
        "OpenAI": reject_client_construction,
        "os": SimpleNamespace(
            getenv=lambda name: credential if name == ENVIRONMENT_VARIABLE else None
        ),
    }
    module = ast.fix_missing_locations(ast.Module(body=safe_body, type_ignores=[]))
    exec(compile(module, str(LEGACY_SCRIPT), "exec"), namespace)
    return namespace


class LegacyCredentialCloseoutTests(unittest.TestCase):
    def test_api_key_is_read_from_the_dedicated_environment_variable(self):
        tree = ast.parse(LEGACY_SCRIPT.read_text(encoding="utf-8"))
        assignments = [
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "API_KEY"
                for target in node.targets
            )
        ]
        self.assertEqual(len(assignments), 1)
        value = assignments[0].value
        self.assertIsInstance(value, ast.Call)
        self.assertIsInstance(value.func, ast.Attribute)
        self.assertIsInstance(value.func.value, ast.Name)
        self.assertEqual(value.func.value.id, "os")
        self.assertEqual(value.func.attr, "getenv")
        self.assertEqual(len(value.args), 1)
        self.assertIsInstance(value.args[0], ast.Constant)
        self.assertEqual(value.args[0].value, ENVIRONMENT_VARIABLE)
        self.assertEqual(value.keywords, [])

    def test_missing_credential_stops_before_client_construction(self):
        module = _load_legacy_module_without_side_effects(credential=None)
        with self.assertRaisesRegex(RuntimeError, ENVIRONMENT_VARIABLE):
            module["consult_ai"]("offline", "offline")


if __name__ == "__main__":
    unittest.main()
