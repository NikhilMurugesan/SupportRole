import unittest

from support_role.pipeline.coding_policy import (
    build_coding_instruction,
    classify_coding_request,
)


class CodingPolicyTests(unittest.TestCase):
    def test_defaults_to_python_for_unspecified_language(self):
        decision = classify_coding_request("Solve this using binary search and give code.")

        self.assertTrue(decision.is_coding)
        self.assertEqual(decision.language, "Python")
        self.assertTrue(decision.is_dsa)

    def test_detects_requested_language(self):
        decision = classify_coding_request("Implement two sum in Java.")

        self.assertTrue(decision.is_coding)
        self.assertEqual(decision.language, "Java")

    def test_code_only_instruction_suppresses_explanation(self):
        decision = classify_coding_request("Code only: write Python code for factorial.")
        instruction = build_coding_instruction(decision)

        self.assertTrue(decision.code_only)
        self.assertIn("Return only one complete runnable Python code block", instruction)
        self.assertIn("Do not include explanation", instruction)


if __name__ == "__main__":
    unittest.main()
