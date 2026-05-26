"""Adaptive pause detection for conversational turn-taking.

Classifies silence durations into tiers (MICRO / SHORT / MEDIUM / LONG)
and decides whether a pause should trigger downstream LLM generation.

The detector adapts thresholds based on:
* Linguistic context — sentence-ending punctuation lowers the bar,
  trailing conjunctions and very short utterances raise it.
* Speaking rate — fast speakers get slightly longer thresholds (they
  may just be taking a quick breath), slow speakers get shorter ones.
"""

from __future__ import annotations

import logging
import re
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

log = logging.getLogger(__name__)

# Conjunctions that strongly signal the speaker intends to continue.
_TRAILING_CONJUNCTIONS: frozenset[str] = frozenset({
    "and", "but", "or", "because", "so", "if", "when", "while",
})

# Regex used to count words in an utterance.
_WORD_RE = re.compile(r"\S+")

# Sentence-ending punctuation that implies semantic completeness.
_SENTENCE_ENDERS: frozenset[str] = frozenset({".", "?", "!"})


class PauseTier(str, Enum):
    """Categorised pause duration."""

    MICRO = "micro"      # 300–500 ms  — natural hesitation
    SHORT = "short"      # 500–1000 ms — brief pause
    MEDIUM = "medium"    # 1000–2000 ms — deliberate pause
    LONG = "long"        # >2000 ms    — extended silence


@dataclass
class _UtteranceSample:
    """A single data-point used to estimate speaking rate."""

    word_count: int
    duration_s: float


class AdaptivePauseDetector:
    """Context-aware pause classifier and trigger gate.

    Parameters
    ----------
    min_pause_ms:
        Absolute minimum silence (in ms) before any tier can be assigned.
    default_pause_ms:
        Base threshold between SHORT and MEDIUM tiers.
    max_pause_ms:
        Threshold above which a pause is always classified as LONG.
    rate_window:
        Number of recent utterances kept for speaking-rate estimation.
    """

    def __init__(
        self,
        min_pause_ms: int = 500,
        default_pause_ms: int = 1000,
        max_pause_ms: int = 2000,
        rate_window: int = 10,
    ) -> None:
        self.min_pause_ms = min_pause_ms
        self.default_pause_ms = default_pause_ms
        self.max_pause_ms = max_pause_ms
        self._rate_window = rate_window

        # Recent utterance samples for speaking-rate estimation.
        self._samples: deque[_UtteranceSample] = deque(maxlen=rate_window)

        log.debug(
            "AdaptivePauseDetector created (min=%d ms, default=%d ms, "
            "max=%d ms, rate_window=%d)",
            min_pause_ms, default_pause_ms, max_pause_ms, rate_window,
        )

    # ------------------------------------------------------------------ public

    def classify_pause(
        self,
        silence_duration_ms: float,
        utterance_text: str,
    ) -> PauseTier:
        """Classify a silence gap into a :class:`PauseTier`.

        The raw duration is compared against adaptive thresholds derived
        from the base configuration, linguistic context of
        *utterance_text*, and the estimated speaking rate.

        Parameters
        ----------
        silence_duration_ms:
            How long (in ms) the user has been silent.
        utterance_text:
            The most recent transcribed utterance text, used to shift
            thresholds up or down based on semantic completeness.

        Returns
        -------
        PauseTier
            The tier that best matches the adjusted silence duration.
        """
        short_thresh, medium_thresh, long_thresh = self._effective_thresholds(
            utterance_text,
        )

        if silence_duration_ms < short_thresh:
            tier = PauseTier.MICRO
        elif silence_duration_ms < medium_thresh:
            tier = PauseTier.SHORT
        elif silence_duration_ms < long_thresh:
            tier = PauseTier.MEDIUM
        else:
            tier = PauseTier.LONG

        log.debug(
            "classify_pause: %.0f ms -> %s (thresholds short=%.0f, "
            "medium=%.0f, long=%.0f, text=%.40s)",
            silence_duration_ms,
            tier.value,
            short_thresh,
            medium_thresh,
            long_thresh,
            utterance_text.strip()[-40:] if utterance_text else "",
        )
        return tier

    def should_trigger(
        self,
        tier: PauseTier,
        utterance_text: str,
        is_generating: bool,
    ) -> bool:
        """Decide whether a pause of the given *tier* should fire the LLM.

        Parameters
        ----------
        tier:
            The pause tier as returned by :meth:`classify_pause`.
        utterance_text:
            Latest transcribed text, used for completeness heuristics.
        is_generating:
            ``True`` when the LLM is already producing output.  In that
            case, only LONG pauses trigger (avoids restarting generation
            on short breathing pauses).

        Returns
        -------
        bool
            ``True`` if the pipeline should fire a new LLM request.
        """
        # When the LLM is already producing output, only a LONG pause
        # (likely a deliberate new turn) should interrupt.
        if is_generating:
            result = tier is PauseTier.LONG
            log.debug(
                "should_trigger: is_generating=True, tier=%s -> %s",
                tier.value, result,
            )
            return result

        text = utterance_text.strip()

        if tier is PauseTier.MICRO:
            # Natural hesitation — never trigger.
            log.debug("should_trigger: MICRO -> False")
            return False

        if tier is PauseTier.SHORT:
            # Only trigger when the text already looks semantically
            # complete: ends with sentence punctuation or is a clear
            # question.
            result = self._looks_complete(text)
            log.debug(
                "should_trigger: SHORT, complete=%s -> %s", result, result,
            )
            return result

        if tier is PauseTier.MEDIUM:
            # Always trigger *unless* the utterance is clearly
            # unfinished (trailing conjunction).
            if self._ends_with_conjunction(text):
                log.debug("should_trigger: MEDIUM but trailing conjunction -> False")
                return False
            log.debug("should_trigger: MEDIUM -> True")
            return True

        # PauseTier.LONG — always trigger.
        log.debug("should_trigger: LONG -> True")
        return True

    @staticmethod
    def compute_silence_duration_ms(pause_started_at: float) -> float:
        """Return elapsed silence in milliseconds since *pause_started_at*.

        Parameters
        ----------
        pause_started_at:
            A timestamp produced by :func:`time.monotonic` at the moment
            speech ended (or the last voiced frame was seen).

        Returns
        -------
        float
            Silence duration in milliseconds.
        """
        return (time.monotonic() - pause_started_at) * 1000.0

    def record_utterance(self, word_count: int, duration_s: float) -> None:
        """Record an utterance for speaking-rate estimation.

        Call this each time a finalised transcript is available so the
        detector can adapt thresholds to the user's pace.

        Parameters
        ----------
        word_count:
            Number of words in the finalised utterance.
        duration_s:
            Wall-clock duration (in seconds) of the utterance audio.
        """
        if duration_s <= 0 or word_count <= 0:
            return
        self._samples.append(
            _UtteranceSample(word_count=word_count, duration_s=duration_s),
        )
        wps = self._speaking_rate_wps()
        log.debug(
            "record_utterance: %d words / %.2f s, running rate=%.2f wps "
            "(%d samples)",
            word_count, duration_s, wps, len(self._samples),
        )

    def reset(self) -> None:
        """Clear all internal state (speaking-rate history, etc.)."""
        self._samples.clear()
        log.debug("AdaptivePauseDetector reset")

    # ---------------------------------------------------------------- internal

    def _speaking_rate_wps(self) -> float:
        """Compute the average speaking rate in words-per-second.

        Returns ``0.0`` when there are no recorded samples.
        """
        if not self._samples:
            return 0.0
        total_words = sum(s.word_count for s in self._samples)
        total_time = sum(s.duration_s for s in self._samples)
        if total_time <= 0:
            return 0.0
        return total_words / total_time

    def _rate_adjustment_ms(self) -> float:
        """Return a millisecond offset to apply to all thresholds based
        on the user's speaking rate.

        * Fast talkers (>3 wps): thresholds increase by up to +150 ms
          because their short pauses are just breathing gaps.
        * Slow talkers (<1.5 wps): thresholds decrease by up to −150 ms
          because their pauses are more deliberate.
        * Normal range (1.5–3 wps): no adjustment.
        """
        rate = self._speaking_rate_wps()
        if rate <= 0:
            return 0.0
        if rate > 3.0:
            # Linear ramp: at 3 wps -> 0 ms; at 5 wps -> +150 ms.
            adjustment = min((rate - 3.0) / 2.0, 1.0) * 150.0
            return adjustment
        if rate < 1.5:
            # Linear ramp: at 1.5 wps -> 0 ms; at 0.5 wps -> −150 ms.
            adjustment = min((1.5 - rate) / 1.0, 1.0) * -150.0
            return adjustment
        return 0.0

    def _effective_thresholds(
        self,
        utterance_text: str,
    ) -> tuple[float, float, float]:
        """Compute context-adjusted tier boundaries.

        Returns ``(short_threshold, medium_threshold, long_threshold)``
        in milliseconds.
        """
        text = utterance_text.strip()

        # Start from base values.
        short = float(self.min_pause_ms)      # 500
        medium = float(self.default_pause_ms)  # 1000
        long = float(self.max_pause_ms)        # 2000

        # --- Linguistic adjustments (applied to medium & long) ---
        offset = 0.0

        # Semantically complete → lower thresholds (user likely done).
        if self._looks_complete(text):
            offset -= 200.0

        # Very short utterance (<5 words) → raise thresholds (probably
        # incomplete; e.g. "I think…").
        word_count = len(_WORD_RE.findall(text)) if text else 0
        if word_count < 5 and word_count > 0:
            offset += 300.0

        # Trailing conjunction → raise thresholds (speaker clearly
        # intends to continue).
        if self._ends_with_conjunction(text):
            offset += 500.0

        # --- Speaking rate adjustment ---
        rate_adj = self._rate_adjustment_ms()
        offset += rate_adj

        # Apply offset to short and medium boundaries but keep them
        # properly ordered.  MICRO is always 0–short, LONG always
        # starts at the (adjusted) long boundary.
        short = max(300.0, short + offset * 0.5)   # dampen for short
        medium = max(short + 100.0, medium + offset)
        long = max(medium + 100.0, long + offset)

        return short, medium, long

    @staticmethod
    def _looks_complete(text: str) -> bool:
        """Return ``True`` if *text* appears to be a semantically
        complete sentence (ends with ``.``, ``?``, or ``!``)."""
        stripped = text.rstrip()
        if not stripped:
            return False
        return stripped[-1] in _SENTENCE_ENDERS

    @staticmethod
    def _ends_with_conjunction(text: str) -> bool:
        """Return ``True`` if *text* ends with a conjunction word that
        strongly signals the speaker intends to keep talking."""
        stripped = text.rstrip().rstrip(".,!?;:")
        if not stripped:
            return False
        last_word = stripped.rsplit(maxsplit=1)[-1].lower()
        return last_word in _TRAILING_CONJUNCTIONS
