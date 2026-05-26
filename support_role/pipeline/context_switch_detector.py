"""Topic-drift / context-switch detection for conversations.

Analyses keyword overlap between the current utterance and recent
conversation history to decide whether the user has switched topics.
Also recognises explicit topic-switch markers such as "moving on",
"different question", etc.

The detector is lightweight and purely lexical — no embeddings or ML
models required — making it suitable for real-time pipeline use.
"""

from __future__ import annotations

import logging
import re
import string
from typing import Sequence

log = logging.getLogger(__name__)

# ------------------------------------------------------------------ stopwords
# Comprehensive set of common English stopwords.  These are stripped before
# keyword extraction so that topic-relevant content words dominate the
# Jaccard overlap calculation.
STOPWORDS: frozenset[str] = frozenset({
    # Articles & determiners
    "a", "an", "the", "this", "that", "these", "those",
    # Pronouns
    "i", "me", "my", "myself", "mine",
    "we", "us", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves",
    "who", "whom", "whose", "which", "what",
    # Be / have / do
    "is", "are", "am", "was", "were", "be", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did", "doing",
    # Modals
    "will", "would", "could", "should", "may", "might", "shall", "can",
    "must", "need", "ought",
    # Prepositions
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "up", "down", "about",
    # Conjunctions & connectors
    "and", "but", "or", "nor", "if", "while", "because", "so",
    "yet", "both", "either", "neither", "not", "no",
    # Adverbs & misc
    "again", "further", "then", "once", "here", "there",
    "when", "where", "why", "how",
    "all", "each", "every", "few", "more", "most", "other", "some",
    "such", "only", "own", "same", "than", "too", "very", "just",
    "also", "now", "already", "still", "even", "really", "quite",
    # Common contractions (lowered, punctuation stripped)
    "dont", "doesnt", "didnt", "isnt", "arent", "wasnt", "werent",
    "wont", "wouldnt", "couldnt", "shouldnt", "cant", "cannot",
    "im", "ive", "id", "ill", "youre", "youve", "youd", "youll",
    "hes", "shes", "its", "weve", "theyve", "theyre", "theyd",
    "lets", "thats", "whats", "heres", "theres",
    # Filler / discourse
    "uh", "um", "like", "well", "okay", "ok", "yeah", "yes", "no",
    "right", "sure", "oh", "ah",
    # Short / trivial
    "get", "got", "go", "going", "went", "come", "came",
    "say", "said", "know", "known", "think", "thought",
    "see", "seen", "look", "make", "made", "take", "took",
    "want", "tell", "told", "give", "gave", "use", "used",
    "thing", "things",
})

# Explicit phrases that signal a deliberate topic switch.
_TOPIC_SWITCH_MARKERS: tuple[str, ...] = (
    "moving on",
    "let's talk about",
    "lets talk about",
    "next topic",
    "changing subject",
    "change the subject",
    "different question",
    "new question",
    "anyway",
    "on another note",
    "on a different note",
    "switching gears",
    "something else",
    "another thing",
    "by the way",
)

# Pre-compiled regex that matches any punctuation character.
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)

# Tokeniser: split on whitespace.
_TOKEN_RE = re.compile(r"\S+")


class ContextSwitchDetector:
    """Detects topic drift between the current utterance and recent history.

    Parameters
    ----------
    topic_drift_threshold:
        Jaccard similarity threshold.  When the keyword overlap between
        the current text and recent history drops below this value the
        detector flags a topic switch.  Lower values require more
        divergence; higher values are more sensitive.
    """

    def __init__(self, topic_drift_threshold: float = 0.4) -> None:
        self.topic_drift_threshold = topic_drift_threshold
        log.debug(
            "ContextSwitchDetector created (threshold=%.2f)",
            topic_drift_threshold,
        )

    # ------------------------------------------------------------------ public

    def detect(
        self,
        current_text: str,
        recent_texts: list[str],
    ) -> bool:
        """Determine whether *current_text* represents a topic switch.

        Parameters
        ----------
        current_text:
            The latest user utterance.
        recent_texts:
            A list of recent prior utterances (most-recent last) that
            form the conversational context.

        Returns
        -------
        bool
            ``True`` if the current text appears to be about a new
            topic, ``False`` otherwise.
        """
        current_stripped = current_text.strip()

        # Not enough history → can't detect drift.
        if not recent_texts:
            log.debug("detect: no recent_texts, returning False")
            return False

        # Very short input → not enough signal to judge.
        if len(current_stripped) < 10:
            log.debug(
                "detect: current_text too short (%d chars), returning False",
                len(current_stripped),
            )
            return False

        # --- Explicit marker check ---
        lower_text = current_stripped.lower()
        for marker in _TOPIC_SWITCH_MARKERS:
            if marker in lower_text:
                log.info(
                    "detect: explicit topic-switch marker '%s' found",
                    marker,
                )
                return True

        # --- Keyword overlap (Jaccard) ---
        current_kw = self.get_topic_keywords([current_stripped])
        history_kw = self.get_topic_keywords(recent_texts)

        if not current_kw or not history_kw:
            # One side has no meaningful keywords → can't reliably
            # compare; assume continuity.
            log.debug(
                "detect: empty keyword set (current=%d, history=%d), "
                "returning False",
                len(current_kw), len(history_kw),
            )
            return False

        intersection = current_kw & history_kw
        union = current_kw | history_kw
        jaccard = len(intersection) / len(union) if union else 0.0

        is_switch = jaccard < self.topic_drift_threshold

        log.debug(
            "detect: jaccard=%.3f (threshold=%.2f), |current|=%d, "
            "|history|=%d, |intersection|=%d -> switch=%s",
            jaccard,
            self.topic_drift_threshold,
            len(current_kw),
            len(history_kw),
            len(intersection),
            is_switch,
        )
        return is_switch

    @staticmethod
    def get_topic_keywords(texts: list[str]) -> set[str]:
        """Extract topic-relevant keywords from a list of texts.

        Tokens are lower-cased, stripped of punctuation, and filtered
        against :data:`STOPWORDS`.  Only tokens with two or more
        characters are retained to avoid noisy single-letter artefacts.

        Parameters
        ----------
        texts:
            One or more text strings to extract keywords from.

        Returns
        -------
        set[str]
            The set of content-bearing keyword tokens.
        """
        keywords: set[str] = set()
        for text in texts:
            for raw_token in _TOKEN_RE.findall(text):
                token = raw_token.lower().translate(_PUNCT_TABLE)
                if len(token) < 2:
                    continue
                if token in STOPWORDS:
                    continue
                keywords.add(token)
        return keywords
