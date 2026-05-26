"""Streaming LLM client with local and Bedrock backends.

Sends each fresh rolling-context prompt to the selected model source and
streams tokens back as soon as they are produced. Local mode uses Ollama.
Online mode uses Amazon Bedrock Runtime.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional

import httpx

from ..config import CONFIG, LLMConfig
from ..interview_context import load_interview_context
from ..llm_settings import LOCAL_SOURCE, LLMSettingsSnapshot, llm_settings
from .context_buffer import ContextPrompt
from .session_log import SessionLog
from .util_queue import LatestWinsQueue

log = logging.getLogger(__name__)


@dataclass
class HintToken:
    text: str          # incremental token text
    full: str          # full text accumulated so far
    seq: int           # prompt sequence number
    done: bool
    produced_at: float


class OllamaStreamer:
    def __init__(
        self,
        prompt_in: LatestWinsQueue[ContextPrompt],
        hint_out: LatestWinsQueue[HintToken],
        cancel_event: asyncio.Event,
        cfg: LLMConfig = CONFIG.llm,
        session_log: Optional[SessionLog] = None,
    ) -> None:
        self.prompt_in = prompt_in
        self.hint_out = hint_out
        self.cancel_event = cancel_event
        self.cfg = cfg
        self.session_log = session_log
        self._client: Optional[httpx.AsyncClient] = None
        self._local_ready = False

    async def run(self, stop: asyncio.Event) -> None:
        self._client = httpx.AsyncClient(timeout=self.cfg.request_timeout_s)
        log.info("LLM streamer started")
        last_heartbeat = time.monotonic()
        idle_since = time.monotonic()
        try:
            while not stop.is_set():
                try:
                    prompt = await asyncio.wait_for(self.prompt_in.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    now = time.monotonic()
                    if now - last_heartbeat >= 5.0:
                        log.debug(
                            "LLM HEARTBEAT (idle %.1fs, prompt_qsize=%d, waiting for prompt)",
                            now - idle_since, self.prompt_in.qsize(),
                        )
                        last_heartbeat = now
                    continue
                idle_since = time.monotonic()
                last_heartbeat = idle_since
                log.debug(
                    "LLM got prompt seq=%d from prompt_in (qsize=%d)",
                    prompt.seq, self.prompt_in.qsize(),
                )

                # Drain to the latest prompt — never run a stale request.
                drained = 0
                while True:
                    try:
                        prompt = self.prompt_in.get_nowait()
                        drained += 1
                    except asyncio.QueueEmpty:
                        break
                if drained:
                    log.debug(
                        "LLM drained %d stale prompts, running seq=%d",
                        drained, prompt.seq,
                    )

                # Reset the cancellation flag for this generation.
                self.cancel_event.clear()
                try:
                    settings = llm_settings.snapshot()
                    await self._stream_one(prompt, stop, settings)
                except Exception:
                    log.exception("LLM stream failed")
                    self._emit_error(
                        prompt,
                        "LLM request failed. Check the console log for details.",
                    )
        finally:
            if self._client is not None:
                await self._client.aclose()
            self._client = None

    async def _stream_one(
        self,
        prompt: ContextPrompt,
        stop: asyncio.Event,
        settings: LLMSettingsSnapshot,
    ) -> None:
        if settings.source == LOCAL_SOURCE:
            await self._stream_ollama(prompt, stop)
            return
        await self._stream_bedrock(prompt, stop, settings)

    async def _ensure_local_ready(self) -> None:
        if self._local_ready:
            return
        await self._healthcheck()
        await self._warmup()
        self._local_ready = True

    # ----------------------------------------------------------- health check
    async def _healthcheck(self) -> None:
        """Probe Ollama at startup so missing server / missing model is loud."""
        assert self._client is not None
        try:
            r = await self._client.get(f"{self.cfg.base_url}/api/tags", timeout=3.0)
            if r.status_code != 200:
                log.error(
                    "Ollama health check FAILED: HTTP %d at %s. "
                    "Start the server with `ollama serve`.",
                    r.status_code, self.cfg.base_url,
                )
                return
            data = r.json()
            models = [m.get("name", "") for m in data.get("models", [])]
            has_model = any(
                m == self.cfg.model or m.startswith(self.cfg.model + ":")
                for m in models
            )
            log.info(
                "Ollama OK at %s. Installed models: %s. Using '%s' (%s)",
                self.cfg.base_url,
                ", ".join(models) if models else "<none>",
                self.cfg.model,
                "FOUND" if has_model else "NOT FOUND — run `ollama pull " + self.cfg.model + "`",
            )
        except httpx.ConnectError as exc:
            log.error(
                "Ollama health check FAILED: cannot connect to %s (%s). "
                "Start the server with `ollama serve`.",
                self.cfg.base_url, exc,
            )
        except Exception:
            log.exception("Ollama health check raised")

    # ---------------------------------------------------------------- warmup
    async def _warmup(self) -> None:
        """Force Ollama to load the model into VRAM before the first real prompt.

        Without this, the first transcript-driven request blocks for ~15 s
        on an 8 B model cold-load, during which the cancel-on-new-input
        logic keeps aborting it and no tokens ever surface.
        """
        assert self._client is not None
        payload = {
            "model": self.cfg.model,
            "prompt": "ok",
            "stream": False,
            "think": False,
            "options": {"num_predict": 1, "num_ctx": self.cfg.num_ctx},
            "keep_alive": "30m",
        }
        t0 = time.monotonic()
        log.info("Warming up %s in Ollama VRAM\u2026", self.cfg.model)
        try:
            r = await self._client.post(
                f"{self.cfg.base_url}/api/generate", json=payload, timeout=60.0,
            )
            ms = (time.monotonic() - t0) * 1000
            if r.status_code == 200:
                log.info("Ollama warmup OK in %.0f ms", ms)
            else:
                log.error(
                    "Ollama warmup HTTP %d: %s",
                    r.status_code, r.text[:200],
                )
        except Exception as exc:
            log.error("Ollama warmup failed (%s) \u2014 first prompt may be slow", exc)

    # --------------------------------------------------------------- streaming
    async def _stream_ollama(self, prompt: ContextPrompt, stop: asyncio.Event) -> None:
        assert self._client is not None
        await self._ensure_local_ready()
        payload = {
            "model": self.cfg.model,
            "stream": True,
            "system": self.cfg.system_prompt,
            "prompt": _build_user_prompt(
                prompt.rolling_text,
                knowledge_block=prompt.knowledge_block,
                knowledge_hits=prompt.knowledge_hits,
            ),
            # qwen3 enables chain-of-thought ("<think>...</think>") by
            # default, which destroys realtime latency. Turn it off.
            "think": False,
            "options": {
                "temperature": self.cfg.temperature,
                "top_p": self.cfg.top_p,
                "num_predict": self.cfg.num_predict,
                "num_ctx": self.cfg.num_ctx,
            },
            "keep_alive": "30m",
        }
        url = f"{self.cfg.base_url}/api/generate"
        accumulated = ""
        first_token_logged = False
        t0 = time.monotonic()
        log.debug("LLM start seq=%d model=%s", prompt.seq, self.cfg.model)

        def _dump(reason: str) -> None:
            clean = _strip_thinking(accumulated).strip()
            log.debug(
                "LLM %s seq=%d in %.0fms (%d chars)",
                reason, prompt.seq, (time.monotonic() - t0) * 1000, len(clean),
            )
            for ln in (clean or "<empty>").splitlines() or ["<empty>"]:
                log.debug("LLM[%d] | %s", prompt.seq, ln)

        try:
            async with self._client.stream("POST", url, json=payload) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    message = body[:200].decode("utf-8", "replace")
                    log.error(
                        "Ollama HTTP %d: %s. Is `ollama serve` running and "
                        "is model '%s' pulled?",
                        resp.status_code,
                        message,
                        self.cfg.model,
                    )
                    self._emit_error(prompt, f"Local Ollama HTTP {resp.status_code}: {message[:120]}")
                    return

                async for line in resp.aiter_lines():
                    if stop.is_set():
                        return
                    if self.cancel_event.is_set():
                        log.info("LLM cancelled seq=%d (newer input arrived)", prompt.seq)
                        return
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    token = obj.get("response", "")
                    done = bool(obj.get("done"))
                    if token:
                        if not first_token_logged:
                            log.debug(
                                "LLM first token seq=%d after %.0fms",
                                prompt.seq, (time.monotonic() - t0) * 1000,
                            )
                            first_token_logged = True
                        accumulated += token
                        clean = _strip_thinking(accumulated).strip()
                        self.hint_out.put_latest(
                            HintToken(
                                text=token,
                                full=clean,
                                seq=prompt.seq,
                                done=False,
                                produced_at=time.monotonic(),
                            )
                        )
                    if done:
                        clean = _strip_thinking(accumulated).strip()
                        self.hint_out.put_latest(
                            HintToken(
                                text="",
                                full=clean,
                                seq=prompt.seq,
                                done=True,
                                produced_at=time.monotonic(),
                            )
                        )
                        log.debug(
                            "LLM done seq=%d in %.0fms (%d chars)",
                            prompt.seq,
                            (time.monotonic() - t0) * 1000,
                            len(clean),
                        )
                        for ln in (clean or "<empty>").splitlines() or ["<empty>"]:
                            log.debug("LLM[%d] | %s", prompt.seq, ln)
                        if self.session_log is not None and clean:
                            self.session_log.log_answer(clean)
                        return
        except httpx.ConnectError as exc:
            log.error(
                "Cannot reach Ollama at %s (%s). Is the Ollama service running?",
                self.cfg.base_url, exc,
            )
            self._emit_error(prompt, f"Cannot reach local Ollama at {self.cfg.base_url}.")
        except httpx.HTTPError as exc:
            log.exception("Ollama HTTP error")
            self._emit_error(prompt, f"Local Ollama request failed: {exc}")

    async def _stream_bedrock(
        self,
        prompt: ContextPrompt,
        stop: asyncio.Event,
        settings: LLMSettingsSnapshot,
    ) -> None:
        if settings.bedrock_api_key:
            os.environ["AWS_BEARER_TOKEN_BEDROCK"] = settings.bedrock_api_key

        event_q: asyncio.Queue[tuple[str, object]] = asyncio.Queue()
        cancel_thread = threading.Event()
        loop = asyncio.get_running_loop()
        accumulated = ""
        first_token_logged = False
        t0 = time.monotonic()
        log.debug(
            "LLM start seq=%d source=bedrock model=%s region=%s",
            prompt.seq, settings.bedrock_model_id, settings.bedrock_region,
        )

        def worker() -> None:
            try:
                for token in self._bedrock_token_iter(prompt, settings, cancel_thread):
                    loop.call_soon_threadsafe(event_q.put_nowait, ("token", token))
                loop.call_soon_threadsafe(event_q.put_nowait, ("done", None))
            except Exception as exc:
                loop.call_soon_threadsafe(event_q.put_nowait, ("error", exc))

        threading.Thread(target=worker, name="BedrockStream", daemon=True).start()

        while not stop.is_set():
            if self.cancel_event.is_set():
                cancel_thread.set()
                log.info("LLM cancelled seq=%d (newer input arrived)", prompt.seq)
                return
            try:
                kind, payload = await asyncio.wait_for(event_q.get(), timeout=0.2)
            except asyncio.TimeoutError:
                continue
            if kind == "token":
                token = str(payload)
                if not first_token_logged:
                    log.debug(
                        "LLM first token seq=%d after %.0fms",
                        prompt.seq, (time.monotonic() - t0) * 1000,
                    )
                    first_token_logged = True
                accumulated += token
                clean = _strip_thinking(accumulated).strip()
                self.hint_out.put_latest(
                    HintToken(
                        text=token,
                        full=clean,
                        seq=prompt.seq,
                        done=False,
                        produced_at=time.monotonic(),
                    )
                )
            elif kind == "done":
                clean = _strip_thinking(accumulated).strip()
                self._emit_done(prompt, clean, t0)
                return
            elif kind == "error":
                exc = payload
                log.error("Bedrock stream failed: %s", exc)
                self._emit_error(prompt, f"Bedrock request failed: {exc}")
                return

    def _bedrock_token_iter(
        self,
        prompt: ContextPrompt,
        settings: LLMSettingsSnapshot,
        cancel_thread: threading.Event,
    ):
        try:
            import boto3  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "boto3 is required for Amazon Bedrock. Install requirements.txt."
            ) from exc

        client = boto3.client(
            service_name="bedrock-runtime",
            region_name=settings.bedrock_region,
        )
        response = client.converse_stream(
            modelId=settings.bedrock_model_id,
            system=[{"text": self.cfg.system_prompt}],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "text": _build_user_prompt(
                                prompt.rolling_text,
                                knowledge_block=prompt.knowledge_block,
                                knowledge_hits=prompt.knowledge_hits,
                            )
                        }
                    ],
                }
            ],
            inferenceConfig={
                "maxTokens": self.cfg.num_predict,
                "temperature": self.cfg.temperature,
                "topP": self.cfg.top_p,
            },
        )
        for event in response.get("stream", []):
            if cancel_thread.is_set():
                break
            delta = event.get("contentBlockDelta", {}).get("delta", {})
            token = delta.get("text") or ""
            if token:
                yield token

    def _emit_done(self, prompt: ContextPrompt, clean: str, t0: float) -> None:
        self.hint_out.put_latest(
            HintToken(
                text="",
                full=clean,
                seq=prompt.seq,
                done=True,
                produced_at=time.monotonic(),
            )
        )
        log.debug(
            "LLM done seq=%d in %.0fms (%d chars)",
            prompt.seq,
            (time.monotonic() - t0) * 1000,
            len(clean),
        )
        for ln in (clean or "<empty>").splitlines() or ["<empty>"]:
            log.debug("LLM[%d] | %s", prompt.seq, ln)
        if self.session_log is not None and clean:
            self.session_log.log_answer(clean)

    def _emit_error(self, prompt: ContextPrompt, message: str) -> None:
        text = f"- {message}"
        self.hint_out.put_latest(
            HintToken(
                text="",
                full=text,
                seq=prompt.seq,
                done=True,
                produced_at=time.monotonic(),
            )
        )


def _build_user_prompt(
    rolling_text: str,
    *,
    knowledge_block: str = "",
    knowledge_hits: int = 0,
) -> str:
    parts: list[str] = []
    interview_context = load_interview_context()
    if interview_context:
        parts.append(
            "Current interview context (HIGHEST PRIORITY; use this to "
            "disambiguate vague words like assignment, submission, router.py, "
            "route, cost matrix, Greedy, Hungarian, truck, order, parcel, and "
            "candidate pair. If retrieved snippets conflict with this context, "
            "trust this context. Do not switch to unrelated AWS/GTS/UPS/RAG/ML "
            "topics unless the live question explicitly asks for them):\n"
            f"{interview_context}"
        )
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
    parts.append(
        "Live (possibly incomplete) transcript of the spoken utterance:\n"
        f"\"\"\"{rolling_text}\"\"\""
    )
    parts.append(
        "Answer the latest concrete interview question NOW as a direct "
        "interview-style explanation with **highlighted keywords**, "
        "following the system instructions exactly. Ignore filler such as "
        "'one second', 'okay', 'go ahead', 'continue', and repeated thanks "
        "unless it is attached to a concrete question. If there is no concrete "
        "question anywhere in the transcript, output exactly '- Waiting for "
        "the actual question.' If the utterance contains MULTIPLE questions "
        "or requests, answer EACH ONE in "
        "its own 'Qn:' group so nothing is skipped. Never mention "
        "source documents, filenames, PDFs, or snippet numbers."
    )
    return "\n\n".join(parts)


_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def _strip_thinking(text: str) -> str:
    """Remove any qwen3-style <think>...</think> block from streamed output.

    Handles partial streams where the closing tag hasn't arrived yet by
    truncating at the opening tag so the UI never shows raw reasoning.
    """
    if _THINK_OPEN not in text:
        return text
    out = []
    i = 0
    while i < len(text):
        start = text.find(_THINK_OPEN, i)
        if start == -1:
            out.append(text[i:])
            break
        out.append(text[i:start])
        end = text.find(_THINK_CLOSE, start)
        if end == -1:
            # Unterminated think block — drop the rest until it closes.
            break
        i = end + len(_THINK_CLOSE)
    return "".join(out)
