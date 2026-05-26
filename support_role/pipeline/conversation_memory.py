"""Multi-turn rolling conversation memory.

Accumulates streaming transcript updates from Whisper into coherent
utterances, stores the conversation history as a list of `Exchange`
dataclasses, and produces formatted context windows that the LLM prompt
builder can consume.

Design notes
~~~~~~~~~~~~
* **Partial-replace semantics**: Whisper re-transcribes its rolling audio
  window on every tick, so each partial update *replaces* (not appends to)
  the current utterance buffer.  Only a ``is_partial=False`` update
  finalises the utterance.
* **Short-term / long-term split**: the most recent *short_term_turns*
  exchanges are rendered verbatim in the context window; older exchanges
  are compressed to a single-line summary to keep prompt length bounded.
* **Topic boundaries**: calling ``mark_topic_boundary()`` increments an
  internal counter so that context-window construction can deprioritise
  exchanges from earlier topics.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from .transcriber import TranscriptUpdate

log = logging.getLogger(__name__)

# ------------------------------------------------------------------ defaults
_DEFAULT_MAX_MEMORY_TURNS: int = 10
_DEFAULT_SHORT_TERM_TURNS: int = 3
_DEFAULT_MAX_MEMORY_CHARS: int = 6_000


# ----------------------------------------------------------------- data model
@dataclass
class Exchange:
    """A single conversational turn (user utterance + optional LLM response)."""

    user_text: str
    response_text: str = ""
    intent: str = ""
    timestamp: float = field(default_factory=time.time)
    topic_id: int = 0


# ---------------------------------------------------------------- main class
class ConversationMemory:
    """Thread-safe, rolling conversation memory.

    Parameters
    ----------
    max_memory_turns:
        Maximum number of `Exchange` objects retained.  When the limit is
        exceeded the oldest exchange is evicted.
    short_term_turns:
        The most recent *N* exchanges are included verbatim in the
        context window returned by :meth:`get_context_window`.  Older
        exchanges are summarised to a single line each.
    max_memory_chars:
        Hard character budget for :meth:`get_context_window`.  The
        builder stops adding exchanges once this limit would be exceeded.
    """

    def __init__(
        self,
        max_memory_turns: int = _DEFAULT_MAX_MEMORY_TURNS,
        short_term_turns: int = _DEFAULT_SHORT_TERM_TURNS,
        max_memory_chars: int = _DEFAULT_MAX_MEMORY_CHARS,
    ) -> None:
        self._max_turns = max_memory_turns
        self._short_term = short_term_turns
        self._max_chars = max_memory_chars

        self._exchanges: list[Exchange] = []
        self._current_utterance: str = ""
        self._topic_id: int = 0
        self._lock = threading.Lock()

        log.info(
            "ConversationMemory created (max_turns=%d, short_term=%d, "
            "max_chars=%d)",
            self._max_turns,
            self._short_term,
            self._max_chars,
        )

    # -------------------------------------------------------------- ingestion
    def ingest(self, update: TranscriptUpdate) -> None:
        """Accumulate transcript text into the current utterance buffer.

        * **Partial** updates (``update.is_partial is True``) *replace*
          the buffer — Whisper re-transcribes the entire rolling window
          on every tick, so later partials are strictly more complete
          than earlier ones.
        * **Final** updates (``update.is_partial is False``) replace the
          buffer with the final text and then finalise the utterance
          automatically (stored as a new `Exchange` with an empty intent
          label — call :meth:`finalize_utterance` explicitly if you need
          to attach an intent *before* the exchange is recorded).

        Parameters
        ----------
        update:
            A :class:`TranscriptUpdate` emitted by the transcriber.
        """
        text = update.text.strip()
        if not text:
            return

        with self._lock:
            if update.is_partial:
                # Replace — Whisper re-transcribes the whole window.
                self._current_utterance = text
                log.debug(
                    "Memory: partial utterance updated (%d chars)",
                    len(text),
                )
            else:
                # Final transcript: update the buffer. The orchestrator will
                # call finalize_utterance() explicitly after intent
                # classification so the intent label is preserved.
                self._current_utterance = text
                log.debug(
                    "Memory: final utterance received (%d chars)",
                    len(text),
                )

    def finalize_utterance(self, intent_label: str = "") -> str:
        """Close the current utterance and record it as a new Exchange.

        Parameters
        ----------
        intent_label:
            A label produced by the intent classifier (e.g. ``"QUESTION"``).
            Stored on the exchange for downstream analytics / filtering.

        Returns
        -------
        str
            The cleaned utterance text that was finalised, or an empty
            string if there was nothing to finalise.
        """
        with self._lock:
            return self._finalize_unlocked(intent_label)

    def _finalize_unlocked(self, intent_label: str) -> str:
        """Inner finalisation logic — caller must already hold ``_lock``."""
        text = self._current_utterance.strip()
        if not text:
            log.debug("Memory: finalize called with empty utterance, skipping")
            return ""

        exchange = Exchange(
            user_text=text,
            intent=intent_label,
            timestamp=time.time(),
            topic_id=self._topic_id,
        )
        self._exchanges.append(exchange)

        # Evict oldest if we exceed the cap.
        if len(self._exchanges) > self._max_turns:
            evicted = self._exchanges.pop(0)
            log.debug(
                "Memory: evicted oldest exchange (topic=%d, %d chars)",
                evicted.topic_id,
                len(evicted.user_text),
            )

        self._current_utterance = ""
        log.debug(
            "Memory: finalised exchange #%d intent=%s topic=%d (%d chars)",
            len(self._exchanges),
            intent_label or "<none>",
            self._topic_id,
            len(text),
        )
        return text

    # --------------------------------------------------------- response recording
    def record_response(self, response: str) -> None:
        """Associate an LLM response with the most recent Exchange.

        Parameters
        ----------
        response:
            The full text of the LLM response to attach.
        """
        with self._lock:
            if not self._exchanges:
                log.warning(
                    "Memory: record_response called with no exchanges — "
                    "response discarded (%d chars)",
                    len(response),
                )
                return
            self._exchanges[-1].response_text = response
            log.debug(
                "Memory: recorded response on exchange #%d (%d chars)",
                len(self._exchanges),
                len(response),
            )

    # --------------------------------------------------------- topic management
    def mark_topic_boundary(self) -> None:
        """Increment the internal topic counter.

        Exchanges created after this call carry a higher ``topic_id``,
        allowing :meth:`get_context_window` to deprioritise older-topic
        exchanges when the character budget is tight.
        """
        with self._lock:
            self._topic_id += 1
            log.debug("Memory: topic boundary — now topic_id=%d", self._topic_id)

    # -------------------------------------------------------------- accessors
    def ingest_raw_text(self, text: str) -> None:
        """Ingest typed text directly (no TranscriptUpdate wrapper).

        Used for text input from the UI where there is no speech pipeline.
        """
        text = text.strip()
        if not text:
            return
        with self._lock:
            self._current_utterance = text
            log.debug("Memory: raw text ingested (%d chars)", len(text))

    def get_current_utterance(self) -> str:
        """Return text accumulated since the last finalise."""
        with self._lock:
            return self._current_utterance

    @property
    def turn_count(self) -> int:
        """Number of finalised exchanges currently in memory."""
        with self._lock:
            return len(self._exchanges)

    def get_recent_user_texts(self, n: int) -> list[str]:
        """Return the last *n* user utterance strings.

        Parameters
        ----------
        n:
            Maximum number of utterances to return.  If fewer exchanges
            exist, all available utterances are returned.

        Returns
        -------
        list[str]
            Ordered oldest → newest.
        """
        with self._lock:
            tail = self._exchanges[-n:] if n > 0 else []
            return [ex.user_text for ex in tail]

    # ---------------------------------------------------------- context window
    def get_context_window(self, max_chars: int = 0) -> str:
        """Build a formatted context string from recent memory.

        The most recent ``short_term_turns`` exchanges are rendered
        verbatim (``USER: … / ASSISTANT: …``).  Older exchanges are
        compressed to a one-line summary each.  Construction stops once
        *max_chars* would be exceeded.

        Parameters
        ----------
        max_chars:
            Character budget.  Defaults to ``self._max_chars`` when
            ``0`` or negative.

        Returns
        -------
        str
            The formatted context block ready for injection into an LLM
            prompt.
        """
        budget = max_chars if max_chars > 0 else self._max_chars

        with self._lock:
            if not self._exchanges:
                return ""

            n = len(self._exchanges)
            short_start = max(0, n - self._short_term)
            current_topic = self._topic_id

            # We build the window in reverse (most-recent first) and then
            # flip so the final string reads chronologically.
            blocks: list[str] = []
            chars_used = 0

            for idx in range(n - 1, -1, -1):
                ex = self._exchanges[idx]
                is_short_term = idx >= short_start

                if is_short_term:
                    block = self._format_exchange_verbatim(ex)
                else:
                    block = self._format_exchange_summary(ex)

                # Add a topic-boundary marker if the topic changed.
                if idx > 0 and self._exchanges[idx - 1].topic_id != ex.topic_id:
                    block = "--- topic change ---\n" + block

                # Deprioritise exchanges from earlier topics: if we're
                # already tight on budget and this exchange is from a
                # previous topic, stop.
                if (
                    chars_used > budget * 0.6
                    and ex.topic_id < current_topic
                    and not is_short_term
                ):
                    break

                if chars_used + len(block) > budget:
                    break

                blocks.append(block)
                chars_used += len(block)

            blocks.reverse()
            context = "\n".join(blocks)
            log.debug(
                "Memory: context window built (%d exchanges, %d/%d chars)",
                len(blocks),
                len(context),
                budget,
            )
            return context

    # -------------------------------------------------------------- formatting
    @staticmethod
    def _format_exchange_verbatim(ex: Exchange) -> str:
        """Render an exchange with full text for short-term context."""
        lines = [f"USER: {ex.user_text}"]
        if ex.response_text:
            lines.append(f"ASSISTANT: {ex.response_text}")
        return "\n".join(lines)

    @staticmethod
    def _format_exchange_summary(ex: Exchange) -> str:
        """Render a one-line summary for older (long-term) exchanges."""
        # Take the first ~120 chars of the user text as the summary.
        user_preview = ex.user_text[:120]
        if len(ex.user_text) > 120:
            user_preview += "…"
        resp_snippet = ""
        if ex.response_text:
            resp_snippet = ex.response_text[:80]
            if len(ex.response_text) > 80:
                resp_snippet += "…"
            resp_snippet = f" → {resp_snippet}"
        return f"[earlier] USER: {user_preview}{resp_snippet}"

    # ------------------------------------------------------------------ reset
    def clear(self) -> None:
        """Reset all memory state."""
        with self._lock:
            self._exchanges.clear()
            self._current_utterance = ""
            self._topic_id = 0
            log.info("Memory: cleared all state")
