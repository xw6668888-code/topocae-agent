from __future__ import annotations

import json
import unittest

from utils.cae_language_grounding.run_identity import build_run_manifest


class RunManifestRedactionTests(unittest.TestCase):
    def test_private_endpoint_credentials_are_hash_only_in_manifest(self):
        manifest = build_run_manifest(
            endpoint=(
                "https://fixture-user:fixture-pass@provider.invalid/v1"
                "?api_key=fixture-secret"
            ),
            backend="provider-backend",
            provider_profile="openai_json_schema",
            model="model",
            revision="revision",
            prompt="prompt",
            schema={"type": "object"},
            runtime_config={"temperature": 0},
            baseline="structured",
            split="val",
            offset=0,
            limit=1,
            shard_index=0,
            shard_count=1,
        )
        encoded = json.dumps(manifest)
        self.assertNotIn("fixture-user", encoded)
        self.assertNotIn("fixture-pass", encoded)
        self.assertNotIn("fixture-secret", encoded)
        self.assertRegex(
            manifest["public_endpoint"]["endpoint_sha256"], r"^[0-9a-f]{64}$"
        )
        self.assertEqual(
            manifest["public_endpoint"]["endpoint_origin_redacted"],
            "https://provider.invalid",
        )


if __name__ == "__main__":
    unittest.main()
