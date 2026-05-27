"""Helpers for isolating the latest meaningful user request."""

from __future__ import annotations

import re
from typing import Optional


def extract_latest_question_or_instruction(text: str) -> tuple[Optional[str], Optional[str]]:
    """Return the last complete meaningful question/instruction and prior feedback."""
    if not text:
        return None, None

    raw_sentences = re.split(r"([.?!])", text)
    sentences: list[str] = []
    i = 0
    while i < len(raw_sentences):
        sent = raw_sentences[i].strip()
        if not sent:
            i += 1
            continue
        punct = ""
        if i + 1 < len(raw_sentences) and raw_sentences[i + 1] in (".", "?", "!"):
            punct = raw_sentences[i + 1]
            i += 2
        else:
            i += 1
        sentences.append(sent + punct)

    question_starters = {
        "what",
        "why",
        "how",
        "when",
        "where",
        "who",
        "which",
        "is",
        "are",
        "do",
        "does",
        "did",
        "can",
        "could",
        "should",
        "would",
        "will",
        "shall",
        "tell",
        "explain",
        "describe",
        "define",
        "list",
        "compare",
        "show",
        "give",
        "summarize",
        "elaborate",
        "analyze",
    }

    active_idx = -1
    for idx in range(len(sentences) - 1, -1, -1):
        sent = sentences[idx].strip()
        sent_lower = sent.lower()
        is_question = sent.endswith("?")
        if not is_question:
            first_word = sent_lower.split(None, 1)[0].strip(".,!?:;") if sent_lower.split() else ""
            if first_word in question_starters:
                is_question = True
        if is_question and len(sent_lower.split()) >= 3:
            active_idx = idx
            break

    if active_idx != -1:
        active_question = sentences[active_idx]
        feedback = " ".join(sentences[:active_idx]).strip() or None
        return active_question, feedback

    non_empty = [sentence for sentence in sentences if sentence.strip()]
    if non_empty:
        return non_empty[-1], " ".join(non_empty[:-1]).strip() if len(non_empty) > 1 else None
    return None, None
