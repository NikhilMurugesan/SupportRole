"""Detection and prompt instructions for coding questions."""

from __future__ import annotations

import re
from dataclasses import dataclass


LANGUAGE_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Python", ("python", "py")),
    ("Java", ("java",)),
    ("C++", ("c++", "cpp")),
    ("C", (" c ", " c\n", " c.", " c,", " c?")),
    ("JavaScript", ("javascript", "js", "node")),
    ("TypeScript", ("typescript", "ts")),
    ("Go", ("golang", " go ")),
    ("Rust", ("rust",)),
    ("C#", ("c#", "csharp")),
    ("SQL", ("sql",)),
)

CODING_TERMS = frozenset(
    {
        "write code",
        "give code",
        "code only",
        "solve this",
        "implement",
        "implementation",
        "programming problem",
        "coding problem",
        "algorithm question",
        "dsa",
        "data structures",
        "debug",
        "debugging",
        "fix this code",
        "leetcode",
        "hackerrank",
        "geeksforgeeks",
        "time complexity",
        "space complexity",
        "complete working code",
    }
)

DSA_TERMS = frozenset(
    {
        "array",
        "linked list",
        "stack",
        "queue",
        "heap",
        "tree",
        "binary tree",
        "binary search",
        "graph",
        "bfs",
        "dfs",
        "dynamic programming",
        "dp",
        "sliding window",
        "two pointers",
        "recursion",
        "backtracking",
        "sort",
        "sorting",
        "search",
        "shortest path",
        "topological",
        "palindrome",
        "substring",
        "subarray",
    }
)

PROJECT_CODE_TERMS = frozenset(
    {
        "this repo",
        "this repository",
        "existing code",
        "codebase",
        "project file",
        "uploaded file",
        "source code",
        "in our project",
        "in my project",
        "in your project",
    }
)


@dataclass(frozen=True)
class CodingDecision:
    is_coding: bool
    language: str
    code_only: bool
    is_dsa: bool
    project_specific: bool
    reason: str


def classify_coding_request(latest_question: str) -> CodingDecision:
    text = f" {(latest_question or '').strip()} "
    lower = text.lower()
    code_only = "code only" in lower or "only code" in lower or "just code" in lower
    language = detect_language(lower)
    project_specific = any(term in lower for term in PROJECT_CODE_TERMS)

    has_explicit_coding = any(term in lower for term in CODING_TERMS)
    has_dsa_signal = any(term in lower for term in DSA_TERMS)
    has_question_shape = bool(re.search(r"\b(find|return|print|count|reverse|sort|merge|remove|insert)\b", lower))
    asks_for_function = bool(re.search(r"\b(function|method|class|program|script)\b", lower))

    is_coding = has_explicit_coding or (
        (has_dsa_signal or asks_for_function) and has_question_shape
    )
    is_dsa = has_dsa_signal or "dsa" in lower or "leetcode" in lower
    reason = "coding request detected" if is_coding else "not a coding request"
    return CodingDecision(
        is_coding=is_coding,
        language=language,
        code_only=code_only,
        is_dsa=is_dsa,
        project_specific=project_specific,
        reason=reason,
    )


def build_coding_instruction(decision: CodingDecision) -> str:
    if not decision.is_coding:
        return ""
    if decision.code_only:
        return (
            "CODING QUESTION OUTPUT OVERRIDE: The latest active question asks for code only. "
            f"Return only one complete runnable {decision.language} code block. "
            "Do not include explanation, bullets, prose, or complexity text."
        )

    complexity = (
        " Include time and space complexity, and mention key edge cases."
        if decision.is_dsa
        else ""
    )
    return (
        "CODING QUESTION OUTPUT OVERRIDE: The latest active question is a coding/programming request. "
        f"Use {decision.language}. Make actual code the main output. "
        "Use this structure: brief approach in 2-4 lines, complete runnable code, "
        "sample input/output if useful."
        f"{complexity} "
        "Do not answer with theory only. Do not follow the normal short bullet-only format for this turn."
    )


def detect_language(lower_text: str) -> str:
    for language, aliases in LANGUAGE_ALIASES:
        for alias in aliases:
            if alias.strip() != alias:
                if alias in lower_text:
                    return language
                continue
            if re.search(rf"\b{re.escape(alias)}\b", lower_text):
                return language
    return "Python"
