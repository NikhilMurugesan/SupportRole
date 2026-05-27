import unittest

from support_role.knowledge.rag_policy import RESOURCE_TOPIC, classify_rag_request
from support_role.pipeline.latest_question import extract_latest_question_or_instruction


class LatestQuestionRagTests(unittest.TestCase):
    def test_repeated_old_resource_question_does_not_drive_retrieval(self):
        transcript = (
            "In your resource allocation project, how did Greedy and Hungarian differ? "
            "In your resource allocation project, how did Greedy and Hungarian differ? "
            "What is the difference between AI, ML, Deep Learning, and Gen AI?"
        )

        latest, _ = extract_latest_question_or_instruction(transcript)
        decision = classify_rag_request(latest or "", intent_value="question")

        self.assertEqual(
            latest,
            "What is the difference between AI, ML, Deep Learning, and Gen AI?",
        )
        self.assertFalse(decision.rag_required)
        self.assertIn(RESOURCE_TOPIC, decision.blocked_topics)


if __name__ == "__main__":
    unittest.main()
