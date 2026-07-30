from __future__ import annotations

import json
import re
import unittest

from utils.cae_intent import validate_intent_result
from utils.cae_language_grounding.llm_adapter_r0 import STRUCTURED_PROMPT_PATH


class R1CStructuredPromptContractTests(unittest.TestCase):
    def test_examples_cover_three_valid_mutually_exclusive_branches(self):
        prompt = STRUCTURED_PROMPT_PATH.read_text(encoding="utf-8")
        blocks = re.findall(r"```json\s*(.*?)\s*```", prompt, flags=re.DOTALL)
        self.assertEqual(len(blocks), 3)
        examples = [json.loads(block) for block in blocks]
        self.assertEqual(
            {example["status"] for example in examples},
            {"VALID", "NEEDS_CLARIFICATION", "INVALID"},
        )
        for example in examples:
            self.assertTrue(validate_intent_result(example), example)
            validation = validate_intent_result(example)
            self.assertTrue(validation["valid"], validation["errors"])
            self.assertEqual("intent" in example, example["status"] == "VALID")

    def test_prompt_is_general_and_states_level_separation(self):
        prompt = STRUCTURED_PROMPT_PATH.read_text(encoding="utf-8")
        self.assertNotIn("RF.e8bd1f616e8424683c1c.en-1", prompt)
        self.assertIn("Never place intent fields at the result root", prompt)
        self.assertIn("Only VALID may contain intent", prompt)


if __name__ == "__main__":
    unittest.main()
