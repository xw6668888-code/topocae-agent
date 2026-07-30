from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from run_p4_r3a_hybrid_evidence_resolver import (
    REQUIRED_DELIVERY_FILES,
    acceptance_checks,
    compute_hybrid_metrics,
    load_cached_proposals,
    paired_comparison,
    publish_delivery,
    stop_state,
)
from utils.cae_hybrid_resolver import HybridCAEQueryFrameResolver


def frame():
    return {
        "frame_version": "cae-query-frame/1.0.0", "status": "valid", "language": "en-US",
        "operations": [{
            "operation_key": "op1", "action": "pressure", "feature": "through_hole_round",
            "face_role": "hole_wall", "selector": "all",
            "magnitude": {"value": 1.0, "unit": "MPa"}, "direction": {"kind": "surface_normal"},
        }],
    }


class OfflineReplayTests(unittest.TestCase):
    def test_cache_loader_reads_selected_response_without_transport(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            digest = hashlib.sha256(b"cached-request").hexdigest()
            namespace = root / "p4_api_r2b_1_first_round"
            namespace.mkdir()
            response = {
                "model": "offline",
                "choices": [{"message": {"content": json.dumps(frame())}}],
            }
            (namespace / f"{digest}.json").write_text(json.dumps({
                "schema_version": "topocae-p4-api-cache/1", "request_hash": digest,
                "namespace": namespace.name, "outcome": {"response": response},
            }), encoding="utf-8")
            progress = root / "progress.json"
            progress.write_text(json.dumps({"results": [{
                "requests": [{"stage": "first_round", "request_hash": digest, "cache_hit": True}],
            }]}), encoding="utf-8")

            loaded = load_cached_proposals(progress, root)
            self.assertEqual(loaded["proposals"], [frame()])
            self.assertEqual(loaded["backend_requests"], 0)
            self.assertEqual(loaded["cache_entries_read"], 1)

    def test_metrics_cover_contracts_accuracy_and_provenance(self):
        resolver = HybridCAEQueryFrameResolver()
        resolved = resolver.resolve(
            {"text": "Apply 5 MPa pressure normal to all hole inner walls.", "language": "en-US"},
            frame(),
        )
        expected = resolved["compiled_result"]
        metrics = compute_hybrid_metrics(
            [expected], [resolved], [{"object_exact": True, "face_exact": True, "illegal_count": 0}],
        )
        self.assertEqual(metrics["frame_valid_count"], 1)
        self.assertEqual(metrics["p2_valid_count"], 1)
        self.assertEqual(metrics["status_accuracy"], 1.0)
        self.assertEqual(metrics["action_accuracy"], 1.0)
        self.assertEqual(metrics["feature_accuracy"], 1.0)
        self.assertEqual(metrics["face_role_accuracy"], 1.0)
        self.assertEqual(metrics["selector_accuracy"], 1.0)
        self.assertEqual(metrics["operation_parameter_exact"], 1.0)
        self.assertEqual(metrics["target_contract_exact"], 1.0)
        self.assertEqual(metrics["canonical_exact"], 1.0)
        self.assertEqual(metrics["object_selection_accuracy"], 1.0)
        self.assertEqual(metrics["face_set_accuracy"], 1.0)
        self.assertGreater(metrics["override_count"], 0)
        self.assertEqual(metrics["override_provenance_coverage"], 1.0)

    def test_paired_comparison_uses_nonidentifying_case_keys(self):
        before = [{"canonical_semantic_exact": False, "categories": {"action_error": True}}]
        revised = [{"canonical_semantic_exact": False, "categories": {"action_error": False}}]
        hybrid = [{"canonical_semantic_exact": True, "categories": {"action_error": False}}]
        result = paired_comparison(before, revised, hybrid, ["synthetic text"])
        self.assertEqual(result["rows"][0]["transition"], "improved_over_both")
        self.assertRegex(result["rows"][0]["case_key"], r"^[0-9a-f]{16}$")
        self.assertNotIn("synthetic text", json.dumps(result))

    def test_stop_states_are_exactly_the_goal_states(self):
        passing = {
            "backend_requests": 0, "frame_valid_count": 20, "p2_valid_count": 20,
            "invalid_correct": True, "needs_clarification_correct": True,
            "action_accuracy": 1.0, "feature_accuracy": 1.0, "face_role_accuracy": 1.0,
            "selector_exact_count": 17, "selector_denominator": 18,
            "operation_parameter_exact_count": 18, "operation_parameter_denominator": 20,
            "target_contract_exact_count": 17, "target_contract_denominator": 18,
            "canonical_exact_count": 16, "canonical_denominator": 20,
            "multi_operation_exact_count": 4, "multi_operation_denominator": 4,
            "object_selection_accuracy": 1.0, "face_set_accuracy": 1.0,
            "illegal_id_rate": 0.0, "override_provenance_coverage": 1.0,
        }
        self.assertTrue(all(acceptance_checks(passing).values()))
        self.assertEqual(stop_state(passing), "READY_FOR_P4_R3B_DISJOINT_100_VAL_REVIEW")
        failing = dict(passing, canonical_exact_count=15)
        self.assertEqual(stop_state(failing), "HYBRID_RESOLVER_GATE_FAILED_WITH_EVIDENCE")
        self.assertEqual(stop_state(passing, external_blocker="missing cache"), "BLOCKED_WITH_EVIDENCE")

    def test_delivery_publisher_writes_every_required_asset_and_manifest_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "delivery"
            payload = {name: {"schema_version": "test/1"} for name in REQUIRED_DELIVERY_FILES if name.endswith(".json")}
            publish_delivery(output, payload, stop_status="HYBRID_RESOLVER_GATE_FAILED_WITH_EVIDENCE")
            self.assertEqual({path.name for path in output.iterdir()}, set(REQUIRED_DELIVERY_FILES))
            manifest = json.loads((output / "delivery_manifest.json").read_text(encoding="utf-8"))
            for row in manifest["assets"]:
                content = (output / row["path"]).read_bytes()
                self.assertEqual(hashlib.sha256(content).hexdigest(), row["sha256"])


if __name__ == "__main__":
    unittest.main()
