import unittest

from support_role.knowledge.rag_policy import GTS_TOPIC, RESOURCE_TOPIC
from support_role.knowledge.retriever import RetrievedChunk, filter_and_rerank_candidates


class RetrieverFilteringTests(unittest.TestCase):
    def test_gts_query_rejects_resource_allocation_chunk(self):
        resource = RetrievedChunk(
            text="Greedy and Hungarian build a cost matrix for truck assignment.",
            source="resource.md",
            distance=0.1,
            metadata={"topic": RESOURCE_TOPIC},
        )
        gts = RetrievedChunk(
            text="GTS recommendations are grounded with SOP policy retrieval and reranking.",
            source="gts.md",
            distance=0.2,
            metadata={"topic": GTS_TOPIC},
        )

        result = filter_and_rerank_candidates(
            "In your GTS Agentic AI project, how did you ground recommendations in policies?",
            [resource, gts],
            top_k=3,
            min_similarity=0.52,
            max_distance=0.48,
            allowed_topics=(GTS_TOPIC,),
            blocked_topics=(RESOURCE_TOPIC,),
        )

        self.assertEqual([chunk.source for chunk in result.accepted_chunks], ["gts.md"])
        self.assertFalse(resource.accepted)
        self.assertEqual(resource.reject_reason, "topic mismatch")


if __name__ == "__main__":
    unittest.main()
