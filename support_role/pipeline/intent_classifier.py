"""Rule-based intent classifier for real-time conversational speech.

Classifies each user utterance into one of nine `Intent` categories
using lightweight pattern matching, word-overlap heuristics, and
context from :class:`ConversationMemory`.  No ML models are loaded —
classification is pure Python string logic, designed to run in
microseconds so it never blocks the streaming pipeline.

Classification priority (highest → lowest)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. FILLER        — very short noise ("um", "uh", "hmm")
2. ACKNOWLEDGEMENT — brief social tokens ("yes", "thank you", "I see")
3. CORRECTION    — explicit correction markers ("no I meant", "actually")
4. QUESTION      — interrogative syntax or question words
5. FOLLOW_UP     — follow-on markers + prior conversation exists
6. CONTINUATION  — high word overlap with the previous utterance
7. TOPIC_SWITCH  — low keyword overlap with recent context
8. INCOMPLETE    — mid-word / mid-phrase trailing fragment
9. STATEMENT     — declarative fallback
"""

from __future__ import annotations

import logging
import re
from enum import Enum

from .conversation_memory import ConversationMemory

log = logging.getLogger(__name__)


# ------------------------------------------------------------------- intent enum
class Intent(Enum):
    """Recognised conversational intents."""

    QUESTION = "question"
    STATEMENT = "statement"
    CONTINUATION = "continuation"
    FOLLOW_UP = "follow_up"
    CORRECTION = "correction"
    ACKNOWLEDGEMENT = "acknowledgement"
    TOPIC_SWITCH = "topic_switch"
    INCOMPLETE = "incomplete"
    FILLER = "filler"


# ---------------------------------------------------------------- pattern banks
_FILLER_PATTERNS: frozenset[str] = frozenset({
    "um", "uh", "uhh", "umm", "hmm", "hm", "hmm", "hmmm",
    "okay", "ok", "yeah", "yep", "yup", "right", "sure",
    "got it", "mhm", "mhmm", "ah", "ahh", "oh", "ohh",
    "er", "erm", "like", "so", "well",
})

_ACKNOWLEDGEMENT_PATTERNS: frozenset[str] = frozenset({
    "yes", "no", "yeah", "yep", "yup", "nope", "nah",
    "i see", "i understand", "i get it", "understood",
    "thank you", "thanks", "thanks a lot", "thank you so much",
    "alright", "all right", "okay so", "okay then",
    "got it", "fair enough", "makes sense", "that makes sense",
    "right", "sure", "of course", "absolutely", "exactly",
    "interesting", "cool", "great", "good", "nice", "perfect",
    "fine", "sounds good", "noted",
})

_CORRECTION_MARKERS: tuple[str, ...] = (
    "no i meant",
    "no i mean",
    "actually",
    "i said",
    "correction",
    "not that",
    "wait no",
    "let me rephrase",
    "let me correct",
    "what i meant was",
    "sorry i meant",
    "i meant",
    "what i mean is",
    "no no",
    "no wait",
)

_QUESTION_STARTERS: frozenset[str] = frozenset({
    "what", "why", "how", "when", "where", "who", "which",
    "is", "are", "do", "does", "did",
    "can", "could", "should", "would", "will", "shall",
    "tell", "explain", "describe", "define", "list", "compare",
})

_FOLLOW_UP_MARKERS: tuple[str, ...] = (
    "and",
    "also",
    "what about",
    "how about",
    "but what",
    "so",
    "then",
    "in that case",
    "regarding",
    "on that note",
    "speaking of",
    "along those lines",
    "related to that",
    "additionally",
    "furthermore",
    "moreover",
)

_TRAILING_CONJUNCTIONS: frozenset[str] = frozenset({
    "and", "but", "or", "because", "since", "so", "if",
    "when", "while", "although", "though", "that", "which",
    "the", "a", "an", "to", "for", "with", "of",
})

# Common English stop-words for keyword extraction.
_STOP_WORDS: frozenset[str] = frozenset({
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing",
    "a", "an", "the", "and", "but", "if", "or", "because", "as",
    "until", "while", "of", "at", "by", "for", "with", "about",
    "against", "between", "through", "during", "before", "after",
    "above", "below", "to", "from", "up", "down", "in", "out",
    "on", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "s", "t", "can", "will", "just", "don",
    "should", "now", "d", "ll", "m", "o", "re", "ve", "y",
    "ain", "aren", "couldn", "didn", "doesn", "hadn", "hasn",
    "haven", "isn", "ma", "mightn", "mustn", "needn", "shan",
    "shouldn", "wasn", "weren", "won", "wouldn",
})

# Regex to strip non-alphanumeric (keeps spaces for tokenisation).
_WORD_RE = re.compile(r"[a-z0-9]+")


# ---------------------------------------------------------------- classifier
class IntentClassifier:
    """Rule-based intent classifier for streaming conversational speech.

    All classification is deterministic and runs in microseconds — no
    network calls, no model loading, no GPU.
    """

    # ------------------------------------------------------------- classify
    def classify(self, text: str, memory: ConversationMemory) -> Intent:
        """Classify *text* into an :class:`Intent`.

        Parameters
        ----------
        text:
            The user utterance to classify.  May be a partial or
            finalised transcript.
        memory:
            Current conversation memory, used for context-dependent
            intents (FOLLOW_UP, CONTINUATION, TOPIC_SWITCH).

        Returns
        -------
        Intent
            The detected intent category.
        """
        if not text or not text.strip():
            return Intent.INCOMPLETE

        cleaned = text.strip()
        lower = cleaned.lower()

        # 1. FILLER — very short + matches filler bank.
        if len(cleaned) < 8 and self._is_filler(lower):
            log.debug("Intent: FILLER ← '%s'", cleaned)
            return Intent.FILLER

        # 2. ACKNOWLEDGEMENT — slightly longer social tokens.
        if self._is_acknowledgement(lower):
            log.debug("Intent: ACKNOWLEDGEMENT ← '%s'", cleaned)
            return Intent.ACKNOWLEDGEMENT

        # 3. CORRECTION — contains correction markers.
        if self._has_correction_marker(lower):
            log.debug("Intent: CORRECTION ← '%s'", cleaned)
            return Intent.CORRECTION

        # 4. QUESTION — interrogative syntax.
        if self._is_question(cleaned, lower):
            log.debug("Intent: QUESTION ← '%s'", cleaned[:60])
            return Intent.QUESTION

        # Context-dependent intents require prior exchanges.
        has_history = memory.turn_count > 0

        # 5. FOLLOW_UP — follow-on marker + history.
        if has_history and self._is_follow_up(lower):
            log.debug("Intent: FOLLOW_UP ← '%s'", cleaned[:60])
            return Intent.FOLLOW_UP

        # 6. CONTINUATION — high overlap with last utterance.
        if has_history and self._is_continuation(lower, memory):
            log.debug("Intent: CONTINUATION ← '%s'", cleaned[:60])
            return Intent.CONTINUATION

        # 7. TOPIC_SWITCH — low overlap with recent context + substantial.
        if has_history and len(cleaned) > 20 and self._is_topic_switch(lower, memory):
            log.debug("Intent: TOPIC_SWITCH ← '%s'", cleaned[:60])
            return Intent.TOPIC_SWITCH

        # 8. INCOMPLETE — trailing fragment / no sentence ender.
        if self._is_incomplete(cleaned, lower):
            log.debug("Intent: INCOMPLETE ← '%s'", cleaned[:60])
            return Intent.INCOMPLETE

        # 9. STATEMENT — declarative fallback.
        log.debug("Intent: STATEMENT ← '%s'", cleaned[:60])
        return Intent.STATEMENT

    # ---------------------------------------------------------- sub-checks
    @staticmethod
    def _is_filler(lower: str) -> bool:
        """Check if *lower* matches a known filler pattern."""
        stripped = lower.strip(".,!? ")
        return stripped in _FILLER_PATTERNS

    @staticmethod
    def _is_acknowledgement(lower: str) -> bool:
        """Check if *lower* is a pure acknowledgement phrase."""
        stripped = lower.strip(".,!? ")
        return stripped in _ACKNOWLEDGEMENT_PATTERNS

    @staticmethod
    def _has_correction_marker(lower: str) -> bool:
        """Return True if *lower* contains any correction marker."""
        return any(marker in lower for marker in _CORRECTION_MARKERS)

    @staticmethod
    def _is_question(cleaned: str, lower: str) -> bool:
        """Detect interrogative syntax."""
        if cleaned.rstrip().endswith("?"):
            return True
        first_word = lower.split(None, 1)[0].strip(".,!?:;") if lower.split() else ""
        return first_word in _QUESTION_STARTERS

    @staticmethod
    def _is_follow_up(lower: str) -> bool:
        """Check for follow-up sentence starters."""
        for marker in _FOLLOW_UP_MARKERS:
            if lower.startswith(marker) and (
                len(lower) == len(marker)
                or lower[len(marker)] in (" ", ",", ".")
            ):
                return True
        return False

    def _is_continuation(self, lower: str, memory: ConversationMemory) -> bool:
        """Return True if *lower* has >40 % word overlap with the last utterance."""
        recent = memory.get_recent_user_texts(1)
        if not recent:
            return False
        return self._word_overlap(lower, recent[-1].lower()) > 0.40

    def _is_topic_switch(self, lower: str, memory: ConversationMemory) -> bool:
        """Return True if keyword overlap with recent context is <15 %."""
        recent_texts = memory.get_recent_user_texts(3)
        if not recent_texts:
            return False
        combined = " ".join(recent_texts).lower()
        current_kw = self._extract_keywords(lower)
        context_kw = self._extract_keywords(combined)
        if not current_kw or not context_kw:
            return False
        overlap = len(current_kw & context_kw) / len(current_kw)
        return overlap < 0.15

    @staticmethod
    def _is_incomplete(cleaned: str, lower: str) -> bool:
        """Detect mid-word / mid-phrase trailing fragments."""
        # Very short text with no sentence-ending punctuation.
        if len(cleaned) < 10 and not cleaned.rstrip().endswith((".", "!", "?")):
            return True
        # Ends with a trailing conjunction / article.
        last_word = lower.split()[-1].strip(".,!?:;") if lower.split() else ""
        if last_word in _TRAILING_CONJUNCTIONS:
            return True
        return False

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _word_overlap(text1: str, text2: str) -> float:
        """Return the ratio of shared words between *text1* and *text2*.

        The ratio is computed as ``|intersection| / |union|``
        (Jaccard similarity) over lower-cased word tokens.

        Returns
        -------
        float
            Value in the range ``[0.0, 1.0]``.
        """
        words1 = set(_WORD_RE.findall(text1.lower()))
        words2 = set(_WORD_RE.findall(text2.lower()))
        if not words1 or not words2:
            return 0.0
        return len(words1 & words2) / len(words1 | words2)

    @staticmethod
    def _extract_keywords(text: str) -> set[str]:
        """Extract non-stopword tokens from *text*.

        Returns
        -------
        set[str]
            Lower-cased keyword tokens with stop-words removed.
        """
        tokens = _WORD_RE.findall(text.lower())
        return {t for t in tokens if t not in _STOP_WORDS and len(t) > 1}

    # --------------------------------------------------------- response gating
    @staticmethod
    def should_respond(intent: Intent, text: str) -> bool:
        """Decide whether the given *intent* warrants an LLM response.

        Parameters
        ----------
        intent:
            The classified intent.
        text:
            The original utterance text (used for length gating on
            STATEMENT and CONTINUATION intents).

        Returns
        -------
        bool
            ``True`` if the pipeline should forward this utterance to
            the LLM for a response.
        """
        if intent is Intent.QUESTION:
            return True
        if intent is Intent.STATEMENT:
            return len(text.strip()) > 15
        if intent is Intent.FOLLOW_UP:
            return True
        if intent is Intent.CORRECTION:
            return True
        if intent is Intent.TOPIC_SWITCH:
            return True
        if intent is Intent.CONTINUATION:
            return len(text.strip()) > 20
        if intent is Intent.ACKNOWLEDGEMENT:
            return False
        if intent is Intent.INCOMPLETE:
            return False
        if intent is Intent.FILLER:
            return False
        # Unknown / future intents default to responding.
        log.warning("should_respond: unhandled intent %s — defaulting to True", intent)
        return True

    # ------------------------------------------------------ prompt instruction
    @staticmethod
    def get_response_instruction(intent: Intent) -> str:
        """Return a short instruction to append to the LLM prompt.

        The instruction steers the model's tone / format based on the
        detected conversational intent.

        Parameters
        ----------
        intent:
            The classified intent.

        Returns
        -------
        str
            A brief natural-language instruction string.  May be empty
            for intents that should not reach the LLM.
        """
        instructions: dict[Intent, str] = {
            Intent.QUESTION: (
                "The user is asking a direct question. "
                "Provide a clear, concise answer."
            ),
            Intent.STATEMENT: (
                "The user made a statement. Acknowledge it and add "
                "relevant context or clarification if appropriate."
            ),
            Intent.FOLLOW_UP: (
                "The user is following up on the previous topic. "
                "Build on your last response and address the new angle."
            ),
            Intent.CORRECTION: (
                "The user is correcting something previously said. "
                "Acknowledge the correction and provide the updated answer."
            ),
            Intent.TOPIC_SWITCH: (
                "The user has switched to a new topic. "
                "Address the new subject without referencing the old one."
            ),
            Intent.CONTINUATION: (
                "The user is continuing their previous thought. "
                "Integrate this with what was said before and respond."
            ),
            Intent.ACKNOWLEDGEMENT: "",
            Intent.INCOMPLETE: "",
            Intent.FILLER: "",
        }
        return instructions.get(intent, "")
