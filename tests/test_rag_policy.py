import unittest

from support_role.knowledge.rag_policy import (
    GTS_TOPIC,
    RESOURCE_TOPIC,
    classify_rag_request,
)


class RagPolicyTests(unittest.TestCase):
    def test_casual_programming_language_question_does_not_use_rag(self):
        decision = classify_rag_request("What's your favorite programming language and why?")

        self.assertFalse(decision.rag_required)
        self.assertIn(RESOURCE_TOPIC, decision.blocked_topics)

    def test_generic_ai_ml_question_does_not_use_rag(self):
        decision = classify_rag_request(
            "What is the difference between AI, ML, Deep Learning, and Gen AI?"
        )

        self.assertFalse(decision.rag_required)
        self.assertIn("generic concept", decision.rag_reason)

    def test_resource_allocation_project_allows_resource_namespace(self):
        decision = classify_rag_request(
            "In your resource allocation project, how did Greedy and Hungarian differ?"
        )

        self.assertTrue(decision.rag_required)
        self.assertEqual(decision.allowed_topics, (RESOURCE_TOPIC,))
        self.assertNotIn(RESOURCE_TOPIC, decision.blocked_topics)

    def test_gts_project_blocks_resource_namespace(self):
        decision = classify_rag_request(
            "In your GTS Agentic AI project, how did you ground recommendations in policies?"
        )

        self.assertTrue(decision.rag_required)
        self.assertIn(GTS_TOPIC, decision.allowed_topics)
        self.assertIn(RESOURCE_TOPIC, decision.blocked_topics)

    def test_normal_coding_question_does_not_use_rag(self):
        decision = classify_rag_request(
            "Solve this DSA problem: find the first non-repeating character in a string."
        )

        self.assertFalse(decision.rag_required)
        self.assertIn("coding question", decision.rag_reason)

    def test_project_specific_coding_question_can_use_project_namespace(self):
        decision = classify_rag_request(
            "In your resource allocation project, implement the Greedy assignment logic."
        )

        self.assertTrue(decision.rag_required)
        self.assertEqual(decision.allowed_topics, (RESOURCE_TOPIC,))

    def test_repo_specific_coding_question_can_use_non_resource_rag(self):
        decision = classify_rag_request(
            "In this repo, implement the missing streaming retry logic."
        )

        self.assertTrue(decision.rag_required)
        self.assertIn("project-specific coding request", decision.rag_reason)
        self.assertIn(RESOURCE_TOPIC, decision.blocked_topics)

    def test_ambiguous_project_question_defaults_to_gts_not_resource(self):
        decision = classify_rag_request("How did you implement your project?")

        self.assertTrue(decision.rag_required)
        self.assertIn(GTS_TOPIC, decision.allowed_topics)
        self.assertIn(RESOURCE_TOPIC, decision.blocked_topics)


if __name__ == "__main__":
    unittest.main()
