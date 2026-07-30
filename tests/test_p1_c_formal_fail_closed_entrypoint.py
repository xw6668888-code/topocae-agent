from __future__ import annotations

import unittest

import generate_p1_c_clean_context_7200_strict_concept as formal_protocol
from p1_c import all_features as feature_core
from p1_c import formal_generator as base_generator
from p1_c.formal_generator_v7 import (
    CANONICAL_ENTRYPOINT,
    GENERATOR_REVISION,
    RIB_WEB_GENERATOR_REVISION,
    SLOT_KERNEL_REVISION,
    SUPPORTED_WALL_ADAPTER_REVISION,
    build_planned_result,
)
from p1_c.formal_plan_v6 import (
    PLAN_REVISION,
    SEED_NAMESPACE,
    audit_plan,
    build_plan,
)
from p1_c.strict_generation_contract import SLOT_CLASSES


class FormalFailClosedEntrypointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = build_plan()

    def _first_success(self, predicate):
        failures = []
        for record in (item for item in self.plan if predicate(item)):
            try:
                return record, build_planned_result(record)
            except Exception as exc:  # formal production has the same deterministic retry policy
                failures.append(f"{record.sample_id}: {type(exc).__name__}: {exc}")
                if len(failures) >= 12:
                    break
        self.fail("no successful formal record in deterministic retry window: " + " | ".join(failures))

    def _assert_revision_contract(self, result) -> None:
        parameters = result.parameters
        self.assertEqual(parameters["generator_revision"], GENERATOR_REVISION)
        self.assertEqual(parameters["plan_revision"], PLAN_REVISION)
        self.assertEqual(parameters["canonical_generation_entrypoint"], CANONICAL_ENTRYPOINT)
        self.assertEqual(parameters["authoritative_slot_kernel_revision"], SLOT_KERNEL_REVISION)
        self.assertEqual(parameters["supported_wall_adapter_revision"], SUPPORTED_WALL_ADAPTER_REVISION)
        self.assertEqual(parameters["rib_web_generator_revision"], RIB_WEB_GENERATOR_REVISION)
        self.assertTrue(parameters["strict_generation_contract_audit"]["passed"])
        self.assertTrue(result.cleanup["passed"])

    def test_frozen_plan_has_exact_quota_and_new_seed_namespace(self) -> None:
        audit = audit_plan(self.plan)
        self.assertTrue(audit["passed"], audit["failures"])
        self.assertEqual(len(self.plan), 7200)
        self.assertEqual(audit["plan_revision"], PLAN_REVISION)
        self.assertEqual(audit["seed_namespace"], SEED_NAMESPACE)
        self.assertEqual(audit["split_counts"], {"train": 5760, "val": 720, "development_test": 720})
        self.assertEqual(audit["concept_incompatible_supported_wall_count"], 0)
        self.assertFalse(audit["semantic_relabel_permitted"])

    def test_human_gate_hashes_and_artifacts_are_fail_closed(self) -> None:
        gates = formal_protocol.verify_human_gates()
        self.assertTrue(gates["all_pass"])
        self.assertEqual(gates["H-C1"]["row_count"], 24)
        self.assertEqual(gates["H-C2"]["row_count"], 36)
        self.assertEqual(gates["H-C1"]["step_hashes_verified"], 24)
        self.assertEqual(gates["H-C2"]["graph_v2_hashes_verified"], 36)
        self.assertTrue(gates["H-C2"]["strict_contract_verified"])

    def test_every_slot_class_uses_continuous_sweep_in_formal_target_only(self) -> None:
        for macro in sorted(SLOT_CLASSES):
            with self.subTest(macro=macro):
                _, result = self._first_success(
                    lambda item, macro=macro: item.primary_target == macro
                    and item.context == "target_only"
                )
                self._assert_revision_contract(result)
                record = next(
                    value
                    for value in result.parameters["features"]
                    if value["macro_feature"] == macro
                )
                self.assertEqual(
                    record["parameters"]["construction_method"],
                    "p1_c_continuous_closed_profile_sweep_cut",
                )
                self.assertTrue(record["parameters"]["guide_on_existing_planar_face"])

    def test_approved_rib_and_web_hosts_are_source_concept_preserving(self) -> None:
        expected = {
            "rib": {"l_corner", "rect_cavity", "circular_cavity", "t_junction"},
            "web": {
                "two_supports_equal",
                "two_supports_asymmetric",
                "two_supports_tapered",
                "two_supports_buttressed",
            },
        }
        for macro, host_families in expected.items():
            for host_family in sorted(host_families):
                with self.subTest(macro=macro, host_family=host_family):
                    _, result = self._first_success(
                        lambda item, macro=macro, host_family=host_family: (
                            item.primary_target == macro
                            and item.context == "target_only"
                            and item.host_family == host_family
                        )
                    )
                    self._assert_revision_contract(result)
                    source = result.parameters["supported_wall"]["source_parameters"]
                    raw = source["source_parameters"]
                    self.assertEqual(raw["source_engineering_macro"], macro)
                    self.assertNotIn("semantic_relabel", source)
                    self.assertNotIn("semantic_relabel", raw)
                    self.assertTrue(raw["concept_contract"]["host_family_concept_compatible"])
                    if macro == "web":
                        self.assertTrue(raw["full_height_two_support_connection"])
                        self.assertEqual(raw["exposed_feature_end_face_count"], 0)

    def test_final_call_repairs_mutated_dependencies_instead_of_using_import_order(self) -> None:
        def forbidden(*args, **kwargs):
            raise AssertionError("mutated dependency was invoked")

        feature_core._feature_tool = forbidden
        base_generator.build_supported_wall_sample = forbidden
        _, slot_result = self._first_success(
            lambda item: item.primary_target == "slot_u" and item.context == "target_only"
        )
        self._assert_revision_contract(slot_result)

        base_generator.build_supported_wall_sample = forbidden
        _, web_result = self._first_success(
            lambda item: item.primary_target == "web" and item.context == "target_only"
        )
        self._assert_revision_contract(web_result)


if __name__ == "__main__":
    unittest.main()
