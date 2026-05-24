"""Rolling context buffer + question-trigger + RAG retrieval.

Sits between the transcriber and the LLM. Responsibilities:

* Keep the most recent N characters of transcript (low prompt cost).
* Decide WHEN to fire the LLM:
    - on any VAD pause (configured to ~1 s of silence), OR
    - early when the partial transcript already ends in "?".
  When `LLMConfig.question_mode` is True, also requires the trailing
  clause to look like a question (what / why / how / is / can / ...).
* Pull relevant chunks from the document knowledge base (if enabled)
  and attach them to the prompt sent to the LLM.
* Surface a "should cancel inflight" signal — disabled by default in
  Q&A mode so the user can read the answer.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

from ..config import CONFIG, LLMConfig
from .util_queue import LatestWinsQueue
from .transcriber import TranscriptUpdate

log = logging.getLogger(__name__)


_QUESTION_STARTERS: frozenset[str] = frozenset({
    "what", "whats", "what's", "why", "how", "when", "where", "who", "whom",
    "whose", "which",
    "is", "are", "am", "was", "were",
    "do", "does", "did",
    "can", "could", "should", "would", "will", "shall", "may", "might",
    "has", "have", "had",
    "tell", "explain", "describe", "define", "list", "compare", "give",
    "show", "name",
})


def _looks_like_question(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    if t.endswith("?"):
        return True
    tail = t
    for sep in (". ", "! ", "? ", "; "):
        idx = tail.rfind(sep)
        if idx != -1:
            tail = tail[idx + len(sep):]
    first = tail.lstrip().split(" ", 1)[0].lower().strip(",.!?:;")
    return first in _QUESTION_STARTERS


@dataclass
class ContextPrompt:
    rolling_text: str
    seq: int
    produced_at: float
    # Optional retrieved knowledge to inject before the question.
    knowledge_block: str = ""
    knowledge_hits: int = 0


class RollingContextManager:
    def __init__(
        self,
        transcript_in: LatestWinsQueue[TranscriptUpdate],
        prompt_out: LatestWinsQueue[ContextPrompt],
        cancel_event: asyncio.Event,
        cfg: LLMConfig = CONFIG.llm,
        retriever=None,  # KnowledgeRetriever | None
    ) -> None:
        self.transcript_in = transcript_in
        self.prompt_out = prompt_out
        self.cancel_event = cancel_event
        self.cfg = cfg
        self.retriever = retriever
        self._last_sent_text = ""
        self._latest: Optional[TranscriptUpdate] = None
        self._latest_at: float = 0.0

    async def run(self, stop: asyncio.Event) -> None:
        cooldown_s = self.cfg.debounce_ms / 1000.0
        log.info(
            "Context manager started (question_mode=%s, cooldown=%.0fms, "
            "rag=%s)",
            self.cfg.question_mode, self.cfg.debounce_ms,
            "on" if self.retriever is not None else "off",
        )
        last_emit_at = 0.0
        last_heartbeat = time.monotonic()
        idle_since = time.monotonic()

        while not stop.is_set():
            try:
                update = await asyncio.wait_for(
                    self.transcript_in.get(), timeout=0.2
                )
            except asyncio.TimeoutError:
                now = time.monotonic()
                if now - last_heartbeat >= 5.0:
                    log.info(
                        "Context HEARTBEAT (idle %.1fs, transcript_qsize=%d, "
                        "prompt_qsize=%d/%d, latest_seq=%s, last_sent='%s')",
                        now - idle_since,
                        self.transcript_in.qsize(),
                        self.prompt_out.qsize(), self.prompt_out.maxsize(),
                        self._latest.seq if self._latest is not None else "none",
                        (self._last_sent_text[-40:] if self._last_sent_text else ""),
                    )
                    last_heartbeat = now
                continue
            # Got an update -> reset idle timer.
            idle_since = time.monotonic()
            last_heartbeat = idle_since
            # Partial spam goes to DEBUG; the pause (partial=False) is
            # the moment that actually matters, so log it LOUD.
            if update.is_partial:
                log.debug(
                    "Context got transcript seq=%d partial=True state=%s (qsize=%d)",
                    update.seq, update.speech_state.value,
                    self.transcript_in.qsize(),
                )
            else:
                log.info(
                    "*** Context got PAUSE transcript seq=%d state=%s (qsize=%d) ***",
                    update.seq, update.speech_state.value,
                    self.transcript_in.qsize(),
                )
            self._on_update(update)

            if self._latest is None:
                continue

            text = self._latest.text.strip()
            is_pause = not self._latest.is_partial
            ends_with_q = text.endswith("?")

            # Trigger gate:
            #  - VAD pause is the authoritative "I'm done" signal and
            #    always fires. The full utterance window is sent so the
            #    LLM can answer EVERY question contained in it.
            #  - Mid-partial '?' optionally fires early (off by default,
            #    see LLMConfig.fire_on_partial_question) — enabling it
            #    again causes the LLM to chase incomplete fragments.
            fire = False
            if is_pause:
                if not self.cfg.question_mode or _looks_like_question(text):
                    fire = True
                else:
                    log.debug(
                        "Q&A: pause but not a question, skipping: '%s'",
                        text[-80:],
                    )
            elif ends_with_q and self.cfg.fire_on_partial_question:
                fire = True
            if not fire:
                continue

            now = time.monotonic()
            if (now - last_emit_at) < cooldown_s:
                continue

            # Surface the trigger BEFORE the (potentially slow) RAG embed
            # call so we can tell from logs whether the gate fired at all.
            log.info(
                "Trigger fired (is_pause=%s, ends_with_q=%s, seq=%d, %d chars)",
                is_pause, ends_with_q, self._latest.seq, len(text),
            )

            # `is_pause` is the user's most authoritative "I'm done talking"
            # signal — it must always reach the LLM even if the rolling
            # tail looks identical to the last sent partial.
            try:
                emitted = await self._emit_with_rag(force=is_pause)
            except Exception:
                log.exception(
                    "_emit_with_rag CRASHED (seq=%d) — continuing loop",
                    self._latest.seq if self._latest is not None else -1,
                )
                emitted = False
            if emitted:
                last_emit_at = now

    # ----------------------------------------------------------------- helpers
    def _on_update(self, update: TranscriptUpdate) -> None:
        self._latest = update
        self._latest_at = time.monotonic()

    async def _emit_with_rag(self, *, force: bool = False) -> bool:
        if self._latest is None:
            return False
        text = self._latest.text.strip()
        if not text:
            return False
        rolling = text[-self.cfg.context_chars :]
        if not force:
            if rolling == self._last_sent_text:
                return False
            new_chars = sum(
                1 for a, b in zip(rolling[::-1], self._last_sent_text[::-1]) if a != b
            ) + max(0, len(rolling) - len(self._last_sent_text))
            if new_chars < self.cfg.min_new_chars:
                return False
        else:
            new_chars = max(0, len(rolling) - len(self._last_sent_text))
        self._last_sent_text = rolling

        # Pull RAG context off the event loop (embedding is a network call).
        knowledge_block = ""
        knowledge_hits = 0
        if self.retriever is not None:
            t0 = time.monotonic()
            try:
                loop = asyncio.get_running_loop()
                # Hard cap so a stuck embed server (e.g. nomic-embed-text
                # cold-loading) cannot freeze the whole answer pipeline.
                chunks = await asyncio.wait_for(
                    loop.run_in_executor(
                        None, self.retriever.retrieve, rolling
                    ),
                    timeout=8.0,
                )
                knowledge_hits = len(chunks)
                if chunks:
                    knowledge_block = self.retriever.format_for_prompt(
                        chunks, CONFIG.knowledge.max_context_chars,
                    )
                log.info(
                    "RAG retrieve OK in %.0f ms (%d chunks)",
                    (time.monotonic() - t0) * 1000, knowledge_hits,
                )
            except asyncio.TimeoutError:
                log.warning(
                    "RAG retrieve timed out after %.0f ms — sending prompt "
                    "without document context this turn.",
                    (time.monotonic() - t0) * 1000,
                )
            except Exception:
                log.exception("RAG retrieval failed (continuing without context)")

        preview = rolling if len(rolling) <= 80 else "..." + rolling[-77:]
        log.info(
            "Context -> LLM (seq=%d, %d chars, +%d new, rag_hits=%d, force=%s): '%s'",
            self._latest.seq, len(rolling), new_chars,
            knowledge_hits, force, preview,
        )

        if self.cfg.cancel_on_new_input and not self.cancel_event.is_set():
            self.cancel_event.set()
        self.prompt_out.put_latest(
            ContextPrompt(
                rolling_text=rolling,
                seq=self._latest.seq,
                produced_at=time.monotonic(),
                knowledge_block=knowledge_block,
                knowledge_hits=knowledge_hits,
            )
        )
        dropped = getattr(self.prompt_out, "last_drop_count", 0)
        log.info(
            "Context put prompt seq=%d -> prompt_out (qsize=%d/%d, dropped=%d)",
            self._latest.seq,
            self.prompt_out.qsize(), self.prompt_out.maxsize(), dropped,
        )
        self._latest = None
        return True
