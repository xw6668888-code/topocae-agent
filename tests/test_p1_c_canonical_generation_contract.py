from __future__ import annotations

from copy import deepcopy
import unittest

from p1_c.all_features_h_c2_v3 import (
    CANONICAL_ENTRYPOINT,
    GENERATOR_REVISION,
    MACRO_CLASSES,
    SLOT_KERNEL_REVISION,
    build_feature_sample,
)
from p1_c.strict_generation_contract import (
    REQUIRED_SLOT_CONSTRUCTION_METHOD,
    SLOT_CLASSES,
    audit_generation_parameters,
)


class CanonicalGenerationContractTests(unittest.TestCase):
    def test_final_h_c2_entrypoint_uses_sweep_for_every_slot_and_context(self) -> None:
        for index, macro in enumerate(sorted(SLOT_CLASSES)):
            for context_offset, secondaries in enumerate(((), ("through_hole_round",))):
                with self.subTest(macro=macro, secondaries=secondaries):
                    result = build_feature_sample(
                        macro,
                        2026092000 + index * 2 + context_offset,
                        secondary_features=secondaries,
                    )
                    parameters = result.parameters
                    self.assertEqual(parameters["generator_revision"], GENERATOR_REVISION)
                    self.assertEqual(parameters["canonical_generation_entrypoint"], CANONICAL_ENTRYPOINT)
                    self.assertEqual(
                        parameters["authoritative_slot_kernel_revision"],
                        SLOT_KERNEL_REVISION,
                    )
                    self.assertTrue(parameters["strict_generation_contract_audit"]["passed"])
                    feature = next(
                        record
                        for record in parameters["features"]
                        if record["macro_feature"] == macro
                    )
                    self.assertEqual(
                        feature["parameters"]["construction_method"],
                        REQUIRED_SLOT_CONSTRUCTION_METHOD,
                    )
                    self.assertTrue(feature["parameters"]["guide_on_existing_planar_face"])

    def test_contract_rejects_a_legacy_planform_slot_even_if_other_audits_pass(self) -> None:
        result = build_feature_sample("slot_u", 2026092100)
        parameters = deepcopy(result.parameters)
        feature = next(record for record in parameters["features"] if record["macro_feature"] == "slot_u")
        feature["parameters"] = {"legacy_planform_extrusion": True}
        audit = audit_generation_parameters(parameters)
        self.assertFalse(audit["passed"])
        self.assertIn(
            "slot::slot_u::instance_1::continuous_sweep_method",
            audit["failures"],
        )

    def test_all_36_h_c2_concepts_pass_the_final_entrypoint_contract(self) -> None:
        from p1_c.all_features_h_c2_v3 import H_C2_SECONDARY_BY_PRIMARY

        for class_index, macro in enumerate(MACRO_CLASSES):
            for context_offset, secondaries in enumerate(
                ((), (H_C2_SECONDARY_BY_PRIMARY[macro],))
            ):
                with self.subTest(macro=macro, secondaries=secondaries):
                    result = build_feature_sample(
                        macro,
                        2026092200 + class_index * 2 + context_offset,
                        secondary_features=secondaries,
                    )
                    self.assertTrue(result.cleanup["passed"])
                    self.assertTrue(
                        result.parameters["strict_generation_contract_audit"]["passed"]
                    )


if __name__ == "__main__":
    unittest.main()
