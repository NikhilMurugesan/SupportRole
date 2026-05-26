"""Conversational orchestrator — the brain of SupportRole.

Replaces the old ``RollingContextManager`` with an event-driven state
machine that continuously evaluates incoming speech and decides *when*,
*whether*, and *how* to generate a response.

State machine
-------------

    LISTENING ──pause──► EVALUATING ──respond──► GENERATING
        ▲                     │                       │
        │                     ▼ (no response)         ▼ (done / interrupted)
        └─────────────────────┘───────────────────────┘

Responsibilities
----------------

* Receive ``TranscriptUpdate`` from the transcriber (partial + final).
* Maintain conversational memory across pauses and utterances.
* Classify intent of each utterance (question / statement / follow-up / …).
* Detect topic drift and manage context transitions.
* Decide when to fire the LLM and build a rich prompt.
* Pre-fetch RAG context speculatively during speech.
* Handle interruptions gracefully (soft-cancel policy).
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from ..config import CONFIG, ConversationConfig, LLMConfig, KnowledgeConfig
from ..interview_context import load_interview_context
from .conversation_memory import ConversationMemory
from .intent_classifier import Intent, IntentClassifier
from .pause_detector import AdaptivePauseDetector
from .context_switch_detector import ContextSwitchDetector
from .transcriber import TranscriptUpdate
from .util_queue import LatestWinsQueue
from .vad import SpeechState

log = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────── orchestrator states

class OrchestratorState(str, Enum):
    LISTENING = "listening"
    EVALUATING = "evaluating"
    GENERATING = "generating"


# ────────────────────────────────────────────────────────────── prompt dataclass

@dataclass
class ContextPrompt:
    """Prompt sent to the LLM streamer."""
    rolling_text: str
    seq: int
    produced_at: float
    knowledge_block: str = ""
    knowledge_hits: int = 0
    utterance_text: str = ""


# ────────────────────────────────────────────────────────────── prompt metadata

@dataclass
class PromptMetadata:
    """Extra information attached to a ContextPrompt for richer generation."""
    intent: Intent
    intent_instruction: str
    conversation_context: str
    topic_switched: bool


# ────────────────────────────────────────────────────────────── orchestrator

class ConversationOrchestrator:
    """Central brain replacing ``RollingContextManager``.

    Subscribes to the ``TranscriptUpdate`` stream, coordinates memory /
    intent / RAG, and emits ``ContextPrompt`` to the LLM streamer when a
    response is warranted.
    """

    def __init__(
        self,
        transcript_in: LatestWinsQueue[TranscriptUpdate],
        prompt_out: LatestWinsQueue[ContextPrompt],
        cancel_event: asyncio.Event,
        retriever=None,                           # KnowledgeRetriever | None
        *,
        conv_cfg: ConversationConfig = CONFIG.conversation,
        llm_cfg: LLMConfig = CONFIG.llm,
        knowledge_cfg: KnowledgeConfig = CONFIG.knowledge,
    ) -> None:
        self.transcript_in = transcript_in
        self.prompt_out = prompt_out
        self.cancel_event = cancel_event
        self.retriever = retriever
        self.conv_cfg = conv_cfg
        self.llm_cfg = llm_cfg
        self.knowledge_cfg = knowledge_cfg

        # Sub-components
        self.memory = ConversationMemory(
            max_memory_turns=conv_cfg.max_memory_turns,
            short_term_turns=conv_cfg.short_term_turns,
            max_memory_chars=conv_cfg.max_memory_chars,
        )
        self.intent_classifier = IntentClassifier()
        self.pause_detector = AdaptivePauseDetector(
            min_pause_ms=conv_cfg.min_pause_ms,
            default_pause_ms=conv_cfg.default_pause_ms,
            max_pause_ms=conv_cfg.max_pause_ms,
        )
        self.context_switch_detector = ContextSwitchDetector(
            topic_drift_threshold=conv_cfg.topic_drift_threshold,
        )

        # Internal state
        self._state = OrchestratorState.LISTENING
        self._last_emit_at: float = 0.0
        self._last_sent_text: str = ""
        self._pause_started_at: Optional[float] = None
        self._is_generating: bool = False
        self._last_pause_update: Optional[TranscriptUpdate] = None

        # Speculative RAG cache
        self._speculative_rag_text: str = ""
        self._speculative_rag_result: str = ""
        self._speculative_rag_hits: int = 0
        self._last_rag_fetch_at: float = 0.0

        # Utterance tracking (similar to old _utterance_partials)
        self._utterance_partials: list[str] = []

        # Sequence counter for prompts
        self._prompt_seq: int = 0

    # ────────────────────────────────────────────────────────── main loop

    async def run(self, stop: asyncio.Event) -> None:
        cooldown_s = self.llm_cfg.debounce_ms / 1000.0
        log.info(
            "ConversationOrchestrator started "
            "(memory=%d turns, pause=%d-%dms, rag=%s, soft_interrupt=%s)",
            self.conv_cfg.max_memory_turns,
            self.conv_cfg.min_pause_ms,
            self.conv_cfg.max_pause_ms,
            "on" if self.retriever is not None else "off",
            self.conv_cfg.soft_interrupt,
        )

        last_heartbeat = time.monotonic()
        idle_since = time.monotonic()

        while not stop.is_set():
            # ── consume next transcript update ──────────────────────────
            try:
                update = await asyncio.wait_for(
                    self.transcript_in.get(), timeout=0.1,
                )
            except asyncio.TimeoutError:
                # If we have a pending pause, evaluate it on timeout ticks
                if self._pause_started_at is not None and self._last_pause_update is not None:
                    await self._process_pause(self._last_pause_update)
                now = time.monotonic()
                if now - last_heartbeat >= 5.0:
                    log.debug(
                        "Orchestrator HEARTBEAT (idle %.1fs, state=%s, "
                        "memory_turns=%d, qsize=%d)",
                        now - idle_since,
                        self._state.value,
                        self.memory.turn_count,
                        self.transcript_in.qsize(),
                    )
                    last_heartbeat = now
                continue

            idle_since = time.monotonic()
            last_heartbeat = idle_since

            # ── ingest into memory ──────────────────────────────────────
            self.memory.ingest(update)
            self._track_partial(update)

            is_pause = not update.is_partial

            if update.is_partial:
                log.debug(
                    "Orchestrator got partial seq=%d state=%s (%d chars)",
                    update.seq, update.speech_state.value, len(update.text),
                )
                # During speech: kick off speculative RAG if enabled
                await self._maybe_speculative_rag(update.text)
                # Check for mid-speech interruption if we are generating
                if self._is_generating:
                    await self._evaluate_mid_speech_interruption(update.text)
                # Reset pause timer — the user is still speaking
                self._pause_started_at = None
                continue

            # ── we have a pause (final transcript) ──────────────────────
            log.debug(
                "Orchestrator got PAUSE seq=%d state=%s (%d chars)",
                update.seq, update.speech_state.value, len(update.text),
            )
            self._last_pause_update = update
            await self._process_pause(update)

    async def _process_pause(self, update: TranscriptUpdate) -> None:
        """Process a pause (final transcript update)."""
        # Record when the pause started if not set yet
        if self._pause_started_at is None:
            # Shift back by the VAD silence threshold (default 500 ms) because the user
            # has already been silent for that duration when the transcriber finishes.
            silence_offset = (CONFIG.audio.silence_frames_for_pause * 20) / 1000.0
            self._pause_started_at = time.monotonic() - silence_offset

        # Get the best version of the utterance text
        utterance_text = self._pick_best_utterance_text(update.text)
        if len(utterance_text.strip()) < 8:
            log.debug(
                "Orchestrator: skipping very short pause transcript "
                "(%d chars)",
                len(utterance_text.strip()),
            )
            self._utterance_partials.clear()
            self._pause_started_at = None
            return

        # ── adaptive pause evaluation ───────────────────────────────
        silence_ms = (time.monotonic() - self._pause_started_at) * 1000
        pause_tier = self.pause_detector.classify_pause(
            silence_ms, utterance_text,
        )
        should_trigger = self.pause_detector.should_trigger(
            pause_tier, utterance_text, self._is_generating,
        )

        if not should_trigger:
            log.debug(
                "Orchestrator: pause tier=%s but not triggering yet "
                "(silence=%.0fms, generating=%s)",
                pause_tier.value, silence_ms, self._is_generating,
            )
            # We don't reset self._pause_started_at because we want to keep counting
            return

        # ── cooldown check ──────────────────────────────────────────
        cooldown_s = self.llm_cfg.debounce_ms / 1000.0
        now = time.monotonic()
        if (now - self._last_emit_at) < cooldown_s:
            log.debug(
                "Orchestrator: within cooldown (%.0fms remaining)",
                (cooldown_s - (now - self._last_emit_at)) * 1000,
            )
            return

        # ── intent classification ───────────────────────────────────
        self._state = OrchestratorState.EVALUATING
        intent = self.intent_classifier.classify(
            utterance_text, self.memory,
        )
        should_respond = self.intent_classifier.should_respond(
            intent, utterance_text,
        )
        log.info(
            "Orchestrator: intent=%s, should_respond=%s, text='%s'",
            intent.value, should_respond,
            utterance_text[:80] + ("…" if len(utterance_text) > 80 else ""),
        )

        if not should_respond:
            # Still finalize the utterance in memory for context
            self.memory.finalize_utterance(intent.value)
            self._utterance_partials.clear()
            self._state = OrchestratorState.LISTENING
            self._pause_started_at = None
            return

        # ── context switch detection ────────────────────────────────
        recent_texts = self.memory.get_recent_user_texts(3)
        topic_switched = self.context_switch_detector.detect(
            utterance_text, recent_texts,
        )
        if topic_switched:
            log.info("Orchestrator: topic switch detected")
            self.memory.mark_topic_boundary()

        # ── finalize utterance in memory ────────────────────────────
        finalized_text = self.memory.finalize_utterance(intent.value)

        # ── build and emit prompt ───────────────────────────────────
        try:
            emitted = await self._build_and_emit(
                finalized_text, intent, topic_switched,
            )
        except Exception:
            log.exception(
                "Orchestrator _build_and_emit CRASHED (seq=%d)",
                update.seq,
            )
            emitted = False

        if emitted:
            self._last_emit_at = time.monotonic()
            self._state = OrchestratorState.GENERATING
            self._is_generating = True
        else:
            self._state = OrchestratorState.LISTENING

        self._utterance_partials.clear()
        self._pause_started_at = None

    # ────────────────────────────────────────────────────── prompt building

    async def _build_and_emit(
        self,
        utterance_text: str,
        intent: Intent,
        topic_switched: bool,
    ) -> bool:
        """Build the full LLM prompt and push it to prompt_out."""
        if not utterance_text.strip():
            return False

        # Duplicate suppression — avoid re-prompting identical text
        rolling = utterance_text[-self.llm_cfg.context_chars:]
        if rolling == self._last_sent_text and len(rolling.strip()) < 30:
            log.debug("Orchestrator: suppressing duplicate prompt")
            return False
        self._last_sent_text = rolling

        # ── RAG decision ────────────────────────────────────────────────
        should_rag = self._should_retrieve_rag(utterance_text, intent)
        knowledge_block = ""
        knowledge_hits = 0
        if should_rag:
            knowledge_block, knowledge_hits = await self._retrieve_rag(
                utterance_text,
            )
        else:
            log.debug("Orchestrator: bypassing RAG retrieval for this turn")

        # ── conversation context ────────────────────────────────────────
        conversation_context = self.memory.get_context_window(
            self.conv_cfg.max_memory_chars,
        )

        # ── build the user prompt ───────────────────────────────────────
        intent_instruction = self.intent_classifier.get_response_instruction(
            intent,
        )
        user_prompt = _build_conversational_prompt(
            utterance_text=utterance_text,
            conversation_context=conversation_context,
            knowledge_block=knowledge_block,
            knowledge_hits=knowledge_hits,
            intent=intent,
            intent_instruction=intent_instruction,
            topic_switched=topic_switched,
            include_interview_context=should_rag,
        )

        # ── signal cancellation if soft-interrupt is configured ─────────
        if self.conv_cfg.soft_interrupt and not self.cancel_event.is_set():
            # Let the LLM streamer know it can cancel a stale generation
            self.cancel_event.set()

        self._prompt_seq += 1
        self.prompt_out.put_latest(
            ContextPrompt(
                rolling_text=user_prompt,
                seq=self._prompt_seq,
                produced_at=time.monotonic(),
                knowledge_block=knowledge_block,
                knowledge_hits=knowledge_hits,
                utterance_text=utterance_text,
            ),
        )

        log.info(
            "Orchestrator -> LLM (seq=%d, intent=%s, %d chars, "
            "memory_turns=%d, rag_hits=%d, topic_switch=%s)",
            self._prompt_seq,
            intent.value,
            len(user_prompt),
            self.memory.turn_count,
            knowledge_hits,
            topic_switched,
        )
        return True

    def _should_retrieve_rag(self, text: str, intent: Intent) -> bool:
        """Apply strict rules to decide if we should trigger RAG retrieval."""
        if not self.knowledge_cfg.enabled:
            return False

        # Do NOT trigger RAG for fillers, acknowledgements, or incomplete text
        if intent in (Intent.FILLER, Intent.ACKNOWLEDGEMENT, Intent.INCOMPLETE):
            return False

        cleaned = text.strip().lower()

        # Rule 2: Explicitly check for document or project references
        doc_indicators = {"document", "file", "uploaded", "paper", "pdf", "txt", "docx", "readme", "context", "project", "repo", "repository", "source code"}
        if any(indicator in cleaned for indicator in doc_indicators):
            log.debug("RAG trigger: explicit document/project reference detected")
            return True

        # Rule 2: Check for general conversation or introductions
        casual_patterns = [
            r"\bhi\b", r"\bhello\b", r"\bhey\b", r"\bhow are you\b",
            r"\btell me about yourself\b", r"\bintroduce yourself\b",
            r"\bwho are you\b", r"\bwhat is your name\b", r"\bmy name is\b",
            r"\bgood morning\b", r"\bgood afternoon\b", r"\bgood evening\b",
            r"\bthank you\b", r"\bthanks\b",
        ]
        if any(re.search(pat, cleaned) for pat in casual_patterns):
            log.debug("RAG trigger bypassed: casual conversation / introduction")
            return False

        # Rule 2: Check for generic AI/ML/Coding questions that LLM can answer directly
        generic_concepts = {
            "transformer", "transformers", "self-attention", "attention mechanism",
            "neural network", "neural networks", "deep learning", "machine learning",
            "backpropagation", "gradient descent", "activation function", "cnn", "rnn",
            "lstm", "gan", "diffusion model", "llm", "large language model", "nlp",
            "python", "programming", "database", "sql", "api", "rest api", "docker",
            "kubernetes", "git", "github", "vector database", "embeddings", "embedding",
            "supervised", "unsupervised", "reinforcement learning", "regression", "classification",
            "overfitting", "underfitting", "bias variance", "precision recall", "f1 score",
        }
        
        # Tokenize user text into words to check exact matches
        words = set(re.findall(r"\b\w+\b", cleaned))
        
        # Extract keywords from active interview context to see if it matches domain
        interview_context = load_interview_context()
        domain_keywords = set()
        if interview_context:
            from .intent_classifier import _STOP_WORDS
            context_words = re.findall(r"\b\w+\b", interview_context.lower())
            domain_keywords = {w for w in context_words if w not in _STOP_WORDS and len(w) > 2}

        # If the user asks about a generic concept:
        # Only trigger RAG if they ALSO mention a domain-specific keyword from the interview context.
        has_generic = any(concept in cleaned for concept in generic_concepts)
        has_domain = any(word in domain_keywords for word in words)

        if has_generic and not has_domain:
            log.debug("RAG trigger bypassed: generic AI/ML/Coding concept query without domain keywords")
            return False

        # If the query has domain-specific keywords from the interview context, run RAG.
        if has_domain:
            log.debug("RAG trigger: domain keywords from interview context detected")
            return True

        # Fallback: if there are no domain keywords and no explicit question, don't run RAG
        if intent == Intent.STATEMENT and len(cleaned) < 30:
            return False

        # Default to True for other questions to be safe, but only if they contain substantive content
        return len(cleaned) >= 15

    # ────────────────────────────────────────────────────── RAG retrieval

    async def _retrieve_rag(self, text: str) -> tuple[str, int]:
        """Retrieve RAG context, using speculative cache if fresh enough."""
        if self.retriever is None:
            return "", 0

        # Use speculative cache if the text hasn't diverged much
        if (
            self._speculative_rag_result
            and self._speculative_rag_text
            and _text_similarity(text, self._speculative_rag_text) > 0.6
        ):
            log.debug(
                "Orchestrator: using speculative RAG cache "
                "(%d hits, similarity=%.2f)",
                self._speculative_rag_hits,
                _text_similarity(text, self._speculative_rag_text),
            )
            return self._speculative_rag_result, self._speculative_rag_hits

        return await self._fetch_rag(text)

    async def _fetch_rag(self, text: str) -> tuple[str, int]:
        """Actually call the retriever."""
        if self.retriever is None:
            return "", 0

        t0 = time.monotonic()
        knowledge_block = ""
        knowledge_hits = 0
        try:
            interview_context = load_interview_context()
            retrieval_query = text
            if interview_context:
                retrieval_query = (
                    "Active interview context:\n"
                    f"{interview_context}\n\n"
                    "Latest transcript:\n"
                    f"{text}"
                )
            loop = asyncio.get_running_loop()
            chunks = await asyncio.wait_for(
                loop.run_in_executor(
                    None, self.retriever.retrieve, retrieval_query,
                ),
                timeout=8.0,
            )
            
            # --- RAG Confidence Gating ---
            # Filter chunks based on cosine distance threshold
            chunks = [c for c in chunks if c.distance < 0.75]
            
            knowledge_hits = len(chunks)
            if chunks:
                knowledge_block = self.retriever.format_for_prompt(
                    chunks, self.knowledge_cfg.max_context_chars,
                )
            log.debug(
                "RAG retrieve OK in %.0f ms (%d chunks after confidence gating)",
                (time.monotonic() - t0) * 1000, knowledge_hits,
            )
        except asyncio.TimeoutError:
            log.warning(
                "RAG retrieve timed out after %.0f ms",
                (time.monotonic() - t0) * 1000,
            )
        except Exception:
            log.exception("RAG retrieval failed")

        return knowledge_block, knowledge_hits

    async def _maybe_speculative_rag(self, text: str) -> None:
        """Speculatively pre-fetch RAG during speech."""
        if not self.conv_cfg.speculative_rag or self.retriever is None:
            return
        if len(text.strip()) < 30:
            return

        # Check if RAG is actually warranted for this partial speech
        intent = self.intent_classifier.classify(text, self.memory)
        if not self._should_retrieve_rag(text, intent):
            # If not warranted, clean any cached speculative result so we don't use a stale one
            self._speculative_rag_text = ""
            self._speculative_rag_result = ""
            self._speculative_rag_hits = 0
            return

        now = time.monotonic()
        debounce_s = self.conv_cfg.rag_debounce_ms / 1000.0
        if (now - self._last_rag_fetch_at) < debounce_s:
            return
        # Only re-fetch if text has changed significantly
        if (
            self._speculative_rag_text
            and _text_similarity(text, self._speculative_rag_text) > 0.7
        ):
            return

        self._last_rag_fetch_at = now
        try:
            block, hits = await self._fetch_rag(text)
            self._speculative_rag_text = text
            self._speculative_rag_result = block
            self._speculative_rag_hits = hits
            log.debug(
                "Speculative RAG: cached %d hits for %d chars of speech",
                hits, len(text),
            )
        except Exception:
            log.debug("Speculative RAG failed (non-critical)")

    # ────────────────────────────────────────────────── utterance helpers

    def _track_partial(self, update: TranscriptUpdate) -> None:
        """Track partial transcripts for best-utterance selection."""
        text = update.text.strip()
        if text and (
            not self._utterance_partials
            or self._utterance_partials[-1] != text
        ):
            self._utterance_partials.append(text)
        if len(self._utterance_partials) > 200:
            self._utterance_partials = self._utterance_partials[-200:]

    def _pick_best_utterance_text(self, fallback: str) -> str:
        """On PAUSE, choose the cleanest transcript from the utterance
        history instead of trusting the final (often most-drifted) one.

        Reuses the scoring logic from the old context_buffer.
        """
        if not self._utterance_partials:
            return fallback

        def _score(text: str) -> tuple:
            q_count = text.count("?")
            ends_clean = 1 if text.rstrip().endswith(("?", ".", "!")) else 0
            return (q_count, ends_clean, len(text))

        best = max(self._utterance_partials, key=_score)
        if _score(best) > _score(fallback):
            return best
        return fallback

    # ────────────────────────────────────────────── response recording

    def record_response(self, response: str) -> None:
        """Called by LLM streamer when generation completes."""
        self.memory.record_response(response)
        self._is_generating = False
        self._state = OrchestratorState.LISTENING

    async def handle_typed_text(self, text: str) -> None:
        """Handle typed text from the UI directly as a completed utterance."""
        if not text.strip():
            return

        # 1. Ingest into memory
        self.memory.ingest_raw_text(text)

        # 2. Classify intent
        intent = self.intent_classifier.classify(text, self.memory)

        # 3. Check for topic switch
        recent_texts = self.memory.get_recent_user_texts(3)
        topic_switched = self.context_switch_detector.detect(text, recent_texts)
        if topic_switched:
            log.info("Orchestrator (typed): topic switch detected")
            self.memory.mark_topic_boundary()

        # 4. Finalize utterance in memory
        finalized_text = self.memory.finalize_utterance(intent.value)

        # 5. Build and emit prompt
        self._state = OrchestratorState.EVALUATING
        try:
            emitted = await self._build_and_emit(
                finalized_text, intent, topic_switched,
            )
        except Exception:
            log.exception("Orchestrator handle_typed_text _build_and_emit CRASHED")
            emitted = False

        if emitted:
            self._last_emit_at = time.monotonic()
            self._state = OrchestratorState.GENERATING
            self._is_generating = True
        else:
            self._state = OrchestratorState.LISTENING

    async def _evaluate_mid_speech_interruption(self, text: str) -> None:
        """Evaluate if the incoming partial user speech should interrupt the LLM."""
        if not self.conv_cfg.soft_interrupt:
            return

        # Get progress from streamer if available
        generated_chars = 0
        if hasattr(self, "streamer") and self.streamer is not None:
            generated_chars = getattr(self.streamer, "generated_char_count", 0)

        text_len = len(text.strip())
        if text_len < 10:
            # Too short to judge or interrupt
            return

        # Classify the tentative intent of the incoming speech
        intent = self.intent_classifier.classify(text, self.memory)

        # 1. If generation is >60% complete (approx >300 chars), let it finish.
        if generated_chars > 300:
            log.debug(
                "Interruption check: LLM is >60%% complete (%d chars), letting it finish.",
                generated_chars,
            )
            return

        # 2. If generation is <30% complete (approx <150 chars) and intent is substantial:
        if generated_chars < 150:
            if intent in (Intent.FILLER, Intent.ACKNOWLEDGEMENT, Intent.INCOMPLETE):
                return
            log.info(
                "Interruption check: LLM is <30%% complete (%d chars) and new intent is %s. Interrupted!",
                generated_chars,
                intent.value,
            )
            self.cancel_event.set()
            self._is_generating = False
            self._state = OrchestratorState.LISTENING
            return

        # 3. Between 30% and 60% (150-300 chars), evaluate continuation vs new topic/question
        if intent in (Intent.QUESTION, Intent.CORRECTION, Intent.TOPIC_SWITCH):
            log.info(
                "Interruption check: LLM is in-between (%d chars) and user raised new %s. Interrupted!",
                generated_chars,
                intent.value,
            )
            self.cancel_event.set()
            self._is_generating = False
            self._state = OrchestratorState.LISTENING


# ──────────────────────────────────────────────────── prompt construction

def _build_conversational_prompt(
    *,
    utterance_text: str,
    conversation_context: str,
    knowledge_block: str,
    knowledge_hits: int,
    intent: Intent,
    intent_instruction: str,
    topic_switched: bool,
    include_interview_context: bool = True,
) -> str:
    """Build the full user prompt with all available context."""
    parts: list[str] = []

    # 1. Interview context (static, high priority)
    if include_interview_context:
        interview_context = load_interview_context()
        if interview_context:
            parts.append(
                "Current interview context (use as domain background; "
                "disambiguate vague words using this context. If retrieved "
                "snippets conflict, trust this context):\n"
                f"{interview_context}"
            )

    # 2. Conversation memory
    if conversation_context:
        parts.append(
            "Recent conversation history (use for continuity; "
            "focus on the LATEST utterance):\n"
            f"{conversation_context}"
        )

    # 3. RAG knowledge
    if knowledge_block:
        parts.append(
            "Background reference material (use silently to ground your "
            "answer; DO NOT mention, cite, quote, or reference these "
            "snippets, their numbers, filenames, or that any document "
            "exists):\n"
            f"{knowledge_block}"
        )
    elif not CONFIG.knowledge.allow_general_knowledge:
        parts.append(
            "No background reference material was retrieved. "
            "If you cannot answer confidently, reply with one bullet: "
            "'- Not enough context to answer.'"
        )

    # 4. Topic switch notice
    if topic_switched:
        parts.append(
            "NOTE: The user has changed the topic. Focus entirely on "
            "the new topic below. Do not carry over unrelated context "
            "from previous exchanges."
        )

    # 5. Current utterance with intent label
    intent_label = intent.value.upper().replace("_", " ")
    parts.append(
        f"Current utterance (detected as {intent_label}):\n"
        f'"""{utterance_text}"""'
    )

    # 6. Intent-specific instruction
    if intent_instruction:
        parts.append(intent_instruction)

    return "\n\n".join(parts)


# ──────────────────────────────────────────────────────────── utilities

def _text_similarity(a: str, b: str) -> float:
    """Quick word-overlap similarity for cache freshness checks."""
    if not a or not b:
        return 0.0
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union) if union else 0.0
