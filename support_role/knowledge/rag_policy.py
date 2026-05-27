"""RAG routing policy, topic namespaces, and source classification.

The realtime assistant should only retrieve from the vector DB when the
latest user question needs private/project knowledge.  This module keeps
those rules out of the orchestrator so they can be tested directly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


RESOURCE_TOPIC = "resource_allocation"
GTS_TOPIC = "gts_agentic_ai"
RAG_TOPIC = "rag"
AWS_TOPIC = "aws"
AGENTIC_AI_TOPIC = "agentic_ai"
INTERVIEW_PREP_TOPIC = "interview_prep"
DATA_SCIENCE_TOPIC = "data_science"
RESUME_TOPIC = "resume"
JOB_DESCRIPTION_TOPIC = "job_description"
DSA_TOPIC = "dsa"
ML_GENERAL_TOPIC = "ml_general"
SYSTEM_DESIGN_TOPIC = "system_design"
PYSPARK_TOPIC = "pyspark_data_engineering"
GENERIC_REFERENCE_TOPIC = "generic_reference"
UNKNOWN_TOPIC = "unknown"

RESOURCE_TOPICS = frozenset({RESOURCE_TOPIC})
GTS_RELATED_TOPICS = frozenset({GTS_TOPIC, RAG_TOPIC, AWS_TOPIC, AGENTIC_AI_TOPIC})
PROJECT_TOPICS = frozenset({RESOURCE_TOPIC, GTS_TOPIC})

RESOURCE_TERMS = frozenset(
    {
        "resource allocation",
        "delivery fleet",
        "truck assignment",
        "truck",
        "trucks",
        "cost matrix",
        "greedy",
        "hungarian",
        "linear_sum_assignment",
        "cheapest insertion",
        "route optimization",
        "optimization",
        "fleet",
        "orders",
        "parcels",
    }
)

RESOURCE_STRONG_TERMS = frozenset(
    {
        "resource allocation",
        "delivery fleet",
        "truck assignment",
        "cost matrix",
        "linear_sum_assignment",
        "cheapest insertion",
        "route optimization",
    }
)

RESOURCE_CONTEXT_TERMS = frozenset(
    {
        "optimization",
        "greedy",
        "hungarian",
        "truck",
        "trucks",
        "fleet",
        "orders",
        "parcels",
        "route",
        "router",
    }
)

GTS_TERMS = frozenset(
    {
        "gts",
        "supervisor exception",
        "timecard",
        "timecards",
        "payroll",
        "sop",
        "policy",
        "policies",
        "business rule",
        "business rules",
        "bre",
        "opensearch",
        "bedrock",
        "titan embeddings",
        "glue",
        "step functions",
        "cloudwatch",
        "dynamodb",
        "human-in-the-loop",
        "grounded recommendations",
    }
)

GTS_STRONG_TERMS = frozenset(
    {
        "gts",
        "supervisor exception",
        "timecard",
        "timecards",
        "payroll",
        "workforce",
        "odometer",
        "package counts",
        "grounded recommendations",
    }
)

GTS_CONTEXT_TERMS = frozenset(
    {
        "policy",
        "policies",
        "sop",
        "business rule",
        "business rules",
        "bre",
        "opensearch",
        "bedrock",
        "titan embeddings",
        "glue",
        "step functions",
        "cloudwatch",
        "dynamodb",
        "human-in-the-loop",
    }
)

GENERIC_CONCEPT_TERMS = frozenset(
    {
        "ai",
        "artificial intelligence",
        "machine learning",
        "ml",
        "deep learning",
        "gen ai",
        "generative ai",
        "neural network",
        "transformer",
        "attention",
        "self-attention",
        "cnn",
        "rnn",
        "lstm",
        "gan",
        "diffusion",
        "llm",
        "nlp",
        "rag",
        "vector database",
        "embedding",
        "embeddings",
        "python",
        "programming",
        "sql",
        "database",
        "api",
        "docker",
        "kubernetes",
        "supervised learning",
        "unsupervised learning",
        "reinforcement learning",
        "regression",
        "classification",
        "overfitting",
        "underfitting",
        "precision",
        "recall",
        "f1",
    }
)

CASUAL_PATTERNS = tuple(
    re.compile(p)
    for p in (
        r"\bhi\b",
        r"\bhello\b",
        r"\bhey\b",
        r"\bhow are you\b",
        r"\bthank(s| you)\b",
        r"\bgood (morning|afternoon|evening)\b",
        r"\bwhat is your name\b",
        r"\bwho are you\b",
        r"\bmy name is\b",
    )
)

BEHAVIORAL_PATTERNS = tuple(
    re.compile(p)
    for p in (
        r"\btell me about yourself\b",
        r"\bintroduce yourself\b",
        r"\bfavorite programming language\b",
        r"\bfavourite programming language\b",
        r"\bwhy should we hire you\b",
        r"\bstrengths? and weaknesses?\b",
        r"\bdescribe a time\b",
        r"\btell me about a time\b",
        r"\bwhere do you see yourself\b",
    )
)

DOC_REFERENCE_TERMS = frozenset(
    {
        "uploaded document",
        "uploaded file",
        "document",
        "documents",
        "file",
        "files",
        "pdf",
        "docx",
        "readme",
        "knowledge base",
        "source material",
        "source document",
    }
)

PROJECT_REFERENCE_TERMS = frozenset(
    {
        "your project",
        "my project",
        "candidate project",
        "project",
        "assignment",
        "submission",
        "repo",
        "repository",
        "source code",
    }
)

ALL_TOPICS = frozenset(
    {
        RESOURCE_TOPIC,
        GTS_TOPIC,
        RAG_TOPIC,
        AWS_TOPIC,
        AGENTIC_AI_TOPIC,
        INTERVIEW_PREP_TOPIC,
        DATA_SCIENCE_TOPIC,
        RESUME_TOPIC,
        JOB_DESCRIPTION_TOPIC,
        DSA_TOPIC,
        ML_GENERAL_TOPIC,
        SYSTEM_DESIGN_TOPIC,
        PYSPARK_TOPIC,
        GENERIC_REFERENCE_TOPIC,
        UNKNOWN_TOPIC,
    }
)


@dataclass(frozen=True)
class RagDecision:
    rag_required: bool
    rag_reason: str
    retrieval_query: str
    allowed_topics: tuple[str, ...]
    blocked_topics: tuple[str, ...]


def classify_rag_request(
    latest_question: str,
    *,
    intent_value: str = "",
    knowledge_enabled: bool = True,
) -> RagDecision:
    """Classify whether the latest question should trigger vector search."""
    question = clean_query(latest_question)
    lower = question.lower()

    if not knowledge_enabled:
        return _no_rag(question, "knowledge disabled")
    if not question:
        return _no_rag(question, "empty latest question")
    if intent_value in {"filler", "acknowledgement", "incomplete"}:
        return _no_rag(question, f"intent is {intent_value}")
    if _matches_any(lower, CASUAL_PATTERNS):
        return _no_rag(question, "casual conversation")
    if _matches_any(lower, BEHAVIORAL_PATTERNS):
        return _no_rag(question, "behavioral/common interview question")

    is_resource = has_resource_signal(lower)
    is_gts = has_gts_signal(lower)
    has_doc_reference = _contains_any(lower, DOC_REFERENCE_TERMS)
    has_project_reference = _contains_any(lower, PROJECT_REFERENCE_TERMS)
    has_generic = _contains_any(lower, GENERIC_CONCEPT_TERMS)

    if is_resource:
        return RagDecision(
            rag_required=True,
            rag_reason="resource allocation project/topic requested",
            retrieval_query=question,
            allowed_topics=(RESOURCE_TOPIC,),
            blocked_topics=(),
        )

    if is_gts:
        return RagDecision(
            rag_required=True,
            rag_reason="GTS/agentic AI project or policy topic requested",
            retrieval_query=question,
            allowed_topics=tuple(sorted(GTS_RELATED_TOPICS | {INTERVIEW_PREP_TOPIC})),
            blocked_topics=(RESOURCE_TOPIC,),
        )

    if has_generic and not has_doc_reference and not has_project_reference:
        return _no_rag(question, "generic concept answerable without knowledge base")

    if has_doc_reference:
        return RagDecision(
            rag_required=True,
            rag_reason="uploaded/source document requested",
            retrieval_query=question,
            allowed_topics=tuple(sorted(ALL_TOPICS - RESOURCE_TOPICS)),
            blocked_topics=(RESOURCE_TOPIC,),
        )

    if has_project_reference:
        return RagDecision(
            rag_required=False,
            rag_reason="project reference is ambiguous; no specific project namespace detected",
            retrieval_query=question,
            allowed_topics=(),
            blocked_topics=(RESOURCE_TOPIC,),
        )

    return _no_rag(question, "latest question does not require private/project knowledge")


def infer_source_metadata(path: Path, text_sample: str = "") -> dict[str, str]:
    """Infer stable metadata for a source file before chunking."""
    name = path.name
    lower_name = name.lower()
    sample = f"{lower_name}\n{text_sample[:6000].lower()}"
    topic = UNKNOWN_TOPIC
    project_name = ""
    document_type = _document_type_for_suffix(path.suffix.lower())

    if "resume" in lower_name or "cv" in lower_name:
        topic = RESUME_TOPIC
        document_type = "resume"
    elif "jd_" in lower_name or "job description" in sample:
        topic = JOB_DESCRIPTION_TOPIC
        document_type = "job_description"
    elif "avathon_resource_allocation" in lower_name or lower_name.startswith("avathon_"):
        topic = RESOURCE_TOPIC
        project_name = "Resource Allocation Engine"
    elif "gts_agentic" in lower_name or _contains_any(sample, GTS_TERMS):
        topic = GTS_TOPIC
        project_name = "GTS Agentic AI Supervisor Exception Assistant"
    elif _contains_any(sample, RESOURCE_TERMS):
        topic = RESOURCE_TOPIC
        project_name = "Resource Allocation Engine"
    elif "pyspark" in lower_name or "databricks" in lower_name:
        topic = PYSPARK_TOPIC
        document_type = "interview_prep"
    elif "system design" in lower_name or "low-level design" in lower_name:
        topic = SYSTEM_DESIGN_TOPIC
        document_type = "interview_prep"
    elif "faang dsa" in lower_name or "dsa" in lower_name:
        topic = DSA_TOPIC
        document_type = "interview_prep"
    elif "machine learning" in lower_name or "complete ml" in lower_name:
        topic = ML_GENERAL_TOPIC
        document_type = "interview_prep"
    elif "data science" in lower_name or "data_science" in lower_name:
        topic = DATA_SCIENCE_TOPIC
        document_type = "interview_prep"
    elif "rag" in sample:
        topic = RAG_TOPIC
    elif "aws" in sample or "bedrock" in sample:
        topic = AWS_TOPIC
    elif "agentic ai" in sample or "agents" in sample:
        topic = AGENTIC_AI_TOPIC
    elif "interview" in sample or "prep" in lower_name:
        topic = INTERVIEW_PREP_TOPIC
        document_type = "interview_prep"

    if topic in {
        DATA_SCIENCE_TOPIC,
        DSA_TOPIC,
        ML_GENERAL_TOPIC,
        SYSTEM_DESIGN_TOPIC,
        PYSPARK_TOPIC,
    }:
        project_name = "General Interview Prep"

    return {
        "document_type": document_type,
        "topic": topic,
        "project_name": project_name,
    }


def clean_query(text: str) -> str:
    return " ".join((text or "").strip().split())


def has_resource_signal(lower_text: str) -> bool:
    if _contains_any(lower_text, RESOURCE_STRONG_TERMS):
        return True
    has_project_frame = _contains_any(lower_text, PROJECT_REFERENCE_TERMS)
    return has_project_frame and _contains_any(lower_text, RESOURCE_CONTEXT_TERMS)


def has_gts_signal(lower_text: str) -> bool:
    if _contains_any(lower_text, GTS_STRONG_TERMS):
        return True
    has_project_frame = _contains_any(lower_text, PROJECT_REFERENCE_TERMS) or "recommendation" in lower_text
    return has_project_frame and _contains_any(lower_text, GTS_CONTEXT_TERMS)


def topic_allowed(topic: str, allowed_topics: tuple[str, ...], blocked_topics: tuple[str, ...]) -> bool:
    if topic in blocked_topics:
        return False
    if allowed_topics and topic not in allowed_topics:
        return False
    return True


def is_probably_noisy_chat_log(path: Path, text: str) -> bool:
    """Reject raw transcripts unless explicitly marked as reference material."""
    lower_path = str(path).lower()
    lower = text[:5000].lower()
    if "training/reference" in lower or "reference material" in lower:
        return False
    if "rag ingestion" in lower or "interview prep" in lower:
        return False
    if "\\transcripts\\" in lower_path or "/transcripts/" in lower_path:
        return True
    speaker_markers = len(re.findall(r"\b(user|assistant|interviewer|candidate)\s*:", lower))
    timestamp_markers = len(re.findall(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", lower))
    return speaker_markers >= 8 or timestamp_markers >= 12


def _no_rag(question: str, reason: str) -> RagDecision:
    return RagDecision(
        rag_required=False,
        rag_reason=reason,
        retrieval_query=question,
        allowed_topics=(),
        blocked_topics=(RESOURCE_TOPIC,),
    )


def _contains_any(text: str, terms: frozenset[str]) -> bool:
    return any(term in text for term in terms)


def _matches_any(text: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def _document_type_for_suffix(suffix: str) -> str:
    return {
        ".pdf": "pdf",
        ".docx": "docx",
        ".md": "markdown",
        ".txt": "text",
        ".jsonl": "jsonl",
    }.get(suffix, "document")
