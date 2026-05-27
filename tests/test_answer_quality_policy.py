import unittest

from support_role.pipeline.answer_quality_policy import (
    build_answer_quality_instruction,
    classify_answer_quality,
    normalize_company_name,
)


class AnswerQualityPolicyTests(unittest.TestCase):
    def test_normalizes_abathon_to_avathon(self):
        self.assertEqual(
            normalize_company_name("What is Abathon doing in physical AI?"),
            "What is Avathon doing in physical AI?",
        )

    def test_avathon_private_fact_question_gets_no_fabrication_instruction(self):
        decision = classify_answer_quality(
            "What are Avathon's recent expansion plans and client partnerships?"
        )
        instruction = build_answer_quality_instruction(decision)

        self.assertTrue(decision.is_avathon)
        self.assertTrue(decision.asks_company_private_facts)
        self.assertIn("Do not fabricate Avathon partnerships", instruction)
        self.assertIn("I don't want to assume specific private partnership details", instruction)

    def test_project_question_defaults_to_gts_context_when_not_resource_specific(self):
        decision = classify_answer_quality("How did you implement your project?")
        instruction = build_answer_quality_instruction(decision)

        self.assertTrue(decision.is_project)
        self.assertTrue(decision.is_gts_project)
        self.assertIn("Agentic AI supervisor assistant", instruction)
        self.assertIn("Titan embeddings", instruction)

    def test_metrics_question_requires_calculation_usage(self):
        decision = classify_answer_quality("What metrics did you track for hallucination rate?")
        instruction = build_answer_quality_instruction(decision)

        self.assertTrue(decision.is_metrics)
        self.assertIn("unsupported generated claims / reviewed responses", instruction)


if __name__ == "__main__":
    unittest.main()
