"""Streaming Ollama client.

Sends each fresh rolling-context prompt to the local Ollama server and
streams tokens back as soon as they are produced. Any in-flight request
is cancelled as soon as a newer prompt arrives, so the user always sees
hints derived from the freshest transcript.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Optional

import httpx

from ..config import CONFIG, LLMConfig
from .context_buffer import ContextPrompt
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
    ) -> None:
        self.prompt_in = prompt_in
        self.hint_out = hint_out
        self.cancel_event = cancel_event
        self.cfg = cfg
        self._client: Optional[httpx.AsyncClient] = None

    async def run(self, stop: asyncio.Event) -> None:
        self._client = httpx.AsyncClient(timeout=self.cfg.request_timeout_s)
        log.info("Ollama streamer started -> %s (%s)", self.cfg.base_url, self.cfg.model)
        await self._healthcheck()
        await self._warmup()
        last_heartbeat = time.monotonic()
        idle_since = time.monotonic()
        try:
            while not stop.is_set():
                try:
                    prompt = await asyncio.wait_for(self.prompt_in.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    now = time.monotonic()
                    if now - last_heartbeat >= 5.0:
                        log.info(
                            "LLM HEARTBEAT (idle %.1fs, prompt_qsize=%d, waiting for prompt)",
                            now - idle_since, self.prompt_in.qsize(),
                        )
                        last_heartbeat = now
                    continue
                idle_since = time.monotonic()
                last_heartbeat = idle_since
                log.info(
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
                    log.info(
                        "LLM drained %d stale prompts, running seq=%d",
                        drained, prompt.seq,
                    )

                # Reset the cancellation flag for this generation.
                self.cancel_event.clear()
                try:
                    await self._stream_one(prompt, stop)
                except Exception:
                    log.exception("LLM stream failed")
        finally:
            await self._client.aclose()
            self._client = None

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
    async def _stream_one(self, prompt: ContextPrompt, stop: asyncio.Event) -> None:
        assert self._client is not None
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
        log.info("LLM start seq=%d model=%s", prompt.seq, self.cfg.model)

        def _dump(reason: str) -> None:
            clean = _strip_thinking(accumulated).strip()
            log.info(
                "LLM %s seq=%d in %.0fms (%d chars)",
                reason, prompt.seq, (time.monotonic() - t0) * 1000, len(clean),
            )
            for ln in (clean or "<empty>").splitlines() or ["<empty>"]:
                log.info("LLM[%d] | %s", prompt.seq, ln)

        try:
            async with self._client.stream("POST", url, json=payload) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    log.error(
                        "Ollama HTTP %d: %s. Is `ollama serve` running and "
                        "is model '%s' pulled?",
                        resp.status_code,
                        body[:200].decode("utf-8", "replace"),
                        self.cfg.model,
                    )
                    return

                async for line in resp.aiter_lines():
                    if stop.is_set():
                        _dump("stopped")
                        return
                    if self.cancel_event.is_set():
                        log.info("LLM cancelled seq=%d (newer input arrived)", prompt.seq)
                        _dump("cancelled")
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
                            log.info(
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
                        # Full answer dumped over multiple log lines so it
                        # stays readable in the console / log file.
                        _dump("done")
                        return
        except httpx.ConnectError as exc:
            log.error(
                "Cannot reach Ollama at %s (%s). Is the Ollama service running?",
                self.cfg.base_url, exc,
            )
            if accumulated:
                _dump("connect-error")
        except httpx.HTTPError:
            log.exception("Ollama HTTP error")
            if accumulated:
                _dump("http-error")


def _build_user_prompt(
    rolling_text: str,
    *,
    knowledge_block: str = "",
    knowledge_hits: int = 0,
) -> str:
    parts: list[str] = []
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
        "Answer the most recent point or question NOW as a direct "
        "interview-style explanation with **highlighted keywords**, "
        "following the system instructions exactly. If the utterance "
        "contains MULTIPLE questions or requests, answer EACH ONE in "
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
