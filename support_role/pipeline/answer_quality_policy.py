"""Per-turn answer quality instructions for interview support."""

from __future__ import annotations

import re
from dataclasses import dataclass


GTS_PROJECT_CONTEXT = (
    "Agentic AI supervisor assistant for a GTS/timecard system: supervisor "
    "exception handling, policy retrieval, historical report retrieval, RAG "
    "architecture, Titan embeddings, vector DB, retriever/reranker, hallucination "
    "reduction, false-positive monitoring, CloudWatch/logging, and AWS orchestration."
)


@dataclass(frozen=True)
class AnswerQualityDecision:
    normalized_question: str
    is_interview: bool
    is_avathon: bool
    asks_company_private_facts: bool
    is_project: bool
    is_resource_allocation: bool
    is_gts_project: bool
    is_rag: bool
    is_metrics: bool
    is_architecture: bool
    is_multi_part: bool


def classify_answer_quality(latest_question: str) -> AnswerQualityDecision:
    normalized = normalize_company_name(latest_question)
    lower = normalized.lower()
    is_avathon = "avathon" in lower
    is_resource = any(
        term in lower
        for term in (
            "resource allocation",
            "greedy",
            "hungarian",
            "cost matrix",
            "truck",
            "fleet",
        )
    )
    is_gts = any(
        term in lower
        for term in (
            "gts",
            "supervisor",
            "timecard",
            "policy",
            "historical report",
            "exception",
            "payroll",
            "agentic ai",
        )
    )
    is_project = is_gts or is_resource or any(
        term in lower
        for term in ("my project", "your project", "candidate project", "project")
    )
    is_rag = any(term in lower for term in ("rag", "retrieval", "vector", "embedding", "rerank", "ground"))
    is_metrics = any(
        term in lower
        for term in (
            "metric",
            "metrics",
            "hallucination rate",
            "false positive",
            "precision",
            "recall",
            "groundedness",
            "latency",
        )
    )
    is_architecture = any(
        term in lower
        for term in ("architecture", "design", "implemented", "implement", "built", "flow", "pipeline")
    )
    asks_company_private = is_avathon and any(
        term in lower
        for term in (
            "recent",
            "expansion",
            "partnership",
            "client",
            "customer",
            "case study",
            "business result",
            "revenue",
            "percentage",
            "market share",
        )
    )
    is_interview = is_project or is_avathon or is_rag or is_metrics or is_architecture
    is_multi_part = normalized.count("?") >= 2 or bool(
        re.search(r"\b(and|also)\b.+\b(challenge|trade[- ]?off|why|how|what)\b", lower)
    )
    return AnswerQualityDecision(
        normalized_question=normalized,
        is_interview=is_interview,
        is_avathon=is_avathon,
        asks_company_private_facts=asks_company_private,
        is_project=is_project,
        is_resource_allocation=is_resource,
        is_gts_project=is_gts or (is_project and not is_resource),
        is_rag=is_rag,
        is_metrics=is_metrics,
        is_architecture=is_architecture,
        is_multi_part=is_multi_part,
    )


def build_answer_quality_instruction(decision: AnswerQualityDecision) -> str:
    parts = [
        "ANSWER QUALITY OVERRIDE:",
        "- Answer the latest active question directly first. Do not answer older transcript questions.",
        "- Do not use weak phrases like 'as mentioned earlier', 'as noted earlier', or vague claims like 'improves scalability and efficiency' without a concrete mechanism.",
        "- For interview answers, use 4-7 strong bullets or one compact paragraph plus 3 bullets.",
        "- Make every answer specific: include at least one architecture component, metric, trade-off, workflow step, failure mode, or business impact.",
    ]

    if decision.is_multi_part:
        parts.append("- The latest question is multi-part; answer each part explicitly.")

    if decision.is_project and decision.is_gts_project:
        parts.append(
            "- For project questions, use this project context unless the latest question explicitly asks about Resource Allocation/Greedy/Hungarian: "
            f"{GTS_PROJECT_CONTEXT}"
        )

    if decision.is_avathon:
        parts.append(
            "- For Avathon questions, answer strategically around physical AI, industrial operations, manufacturing/logistics/energy, real-time decisioning, legacy integration, human-in-the-loop decisions, operational constraints, reliability, safety, explainability, and measurable business impact."
        )
    if decision.asks_company_private_facts:
        parts.append(
            "- Do not fabricate Avathon partnerships, clients, expansion plans, percentages, or case studies. If exact confirmed information is not in context, say: \"I don't want to assume specific private partnership details, but based on Avathon's focus on physical AI and operational environments, I would think about it this way...\""
        )

    if decision.is_rag:
        parts.append(
            "- For RAG questions, include practical implementation details: query formation, retrieval/vector search, metadata filtering or hybrid retrieval if relevant, reranking, irrelevant-chunk rejection, grounding checks, and hallucination control."
        )

    if decision.is_metrics:
        parts.append(
            "- For metrics questions, do not just list metrics. Explain how each metric is calculated or used, for example hallucination rate = unsupported generated claims / reviewed responses, and false positives = incorrectly flagged exceptions or unsupported recommended supervisor actions."
        )

    if decision.is_architecture:
        parts.append(
            "- For architecture questions, answer as a flow: Input -> Processing -> Retrieval/Logic -> LLM/Decision layer -> Validation -> Output -> Monitoring."
        )

    parts.append(
        "- Strong interview structure: direct answer -> specific implementation/example -> challenge or trade-off -> outcome or business value."
    )
    return "\n".join(parts)


def normalize_company_name(text: str) -> str:
    return re.sub(r"\bAbathon\b", "Avathon", text or "", flags=re.IGNORECASE)
