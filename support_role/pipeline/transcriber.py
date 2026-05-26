"""Realtime streaming transcription with Faster-Whisper.

Faster-Whisper is not a true streaming ASR — it transcribes whole audio
buffers. We approximate streaming by re-transcribing the rolling VAD
window every ~250 ms with `beam_size=1`, FP16, and a single decoder
worker, which on an RTX 4080 SUPER stays well under the chunk interval.

The transcribed text is emitted incrementally; downstream consumers
should diff against the previous transcript if they want only the new
suffix.
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import numpy as np

# IMPORTANT: must run before `faster_whisper` is imported so that the CUDA
# runtime DLLs shipped by the nvidia-* pip wheels are findable on Windows.
from . import cuda_bootstrap  # noqa: F401  (side-effect import)

from faster_whisper import WhisperModel

from ..config import CONFIG, WhisperConfig
from .session_log import SessionLog
from .util_queue import LatestWinsQueue
from .vad import SpeechWindow, SpeechState

log = logging.getLogger(__name__)


@dataclass
class TranscriptUpdate:
    text: str
    is_partial: bool
    seq: int
    speech_state: SpeechState
    produced_at: float


class StreamingTranscriber:
    def __init__(
        self,
        window_in: LatestWinsQueue[SpeechWindow],
        transcript_out: "LatestWinsQueue[TranscriptUpdate] | list[LatestWinsQueue[TranscriptUpdate]]",
        cfg: WhisperConfig = CONFIG.whisper,
        session_log: Optional[SessionLog] = None,
    ) -> None:
        self.window_in = window_in
        self.session_log = session_log
        # Allow fan-out: an asyncio.Queue can only deliver each item to ONE
        # waiter, so if both the context manager and the UI pump shared a
        # single transcript queue they would race — the PAUSE final could
        # be grabbed by the UI and never reach the LLM. Accept a list and
        # publish each transcript to every downstream consumer.
        if isinstance(transcript_out, list):
            self.transcript_outs: list[LatestWinsQueue[TranscriptUpdate]] = list(transcript_out)
        else:
            self.transcript_outs = [transcript_out]
        # Keep the legacy single-queue attr for any external readers.
        self.transcript_out = self.transcript_outs[0]
        self.cfg = cfg
        self._model: Optional[WhisperModel] = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper")
        self._last_text = ""
        self._last_emit_seq = -1

    # ------------------------------------------------------------------- init
    def _load(self) -> WhisperModel:
        if self._model is None:
            log.info(
                "Loading Faster-Whisper model=%s device=%s compute=%s",
                self.cfg.model_size, self.cfg.device, self.cfg.compute_type,
            )
            try:
                self._model = WhisperModel(
                    self.cfg.model_size,
                    device=self.cfg.device,
                    compute_type=self.cfg.compute_type,
                    cpu_threads=self.cfg.cpu_threads,
                    num_workers=self.cfg.num_workers,
                )
            except Exception as exc:
                if self.cfg.device != "cpu":
                    log.error(
                        "CUDA Whisper failed to load (%s). Falling back to CPU int8. "
                        "To fix CUDA: ensure `nvidia-cublas-cu12` and "
                        "`nvidia-cudnn-cu12` are installed in this venv.",
                        exc,
                    )
                    self._model = WhisperModel(
                        self.cfg.model_size,
                        device="cpu",
                        compute_type="int8",
                        cpu_threads=self.cfg.cpu_threads,
                        num_workers=self.cfg.num_workers,
                    )
                    log.info("Whisper running on CPU (int8) — slower than CUDA.")
                else:
                    raise
        return self._model

    # --------------------------------------------------------------- main loop
    async def run(self, stop: asyncio.Event) -> None:
        loop = asyncio.get_running_loop()
        # Warm the model in the worker thread.
        await loop.run_in_executor(self._executor, self._load)
        log.info("Transcriber ready (no throttle: every VAD window is transcribed)")

        while not stop.is_set():
            try:
                window = await asyncio.wait_for(self.window_in.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue

            # Drain to the latest window — never transcribe stale audio.
            # EXCEPT: a PAUSED window is the only signal downstream uses to
            # finalize an utterance and trigger the LLM. If we silently drop
            # it because a fresh SPEAKING window arrived a few ms later, the
            # answer never fires. So if any drained window is PAUSED, keep
            # that one and discard the trailing SPEAKING windows.
            paused_window: Optional[SpeechWindow] = (
                window if window.state == SpeechState.PAUSED else None
            )
            drained = 0
            while True:
                try:
                    nxt = self.window_in.get_nowait()
                except asyncio.QueueEmpty:
                    break
                drained += 1
                if nxt.state == SpeechState.PAUSED:
                    paused_window = nxt
                window = nxt
            if paused_window is not None:
                window = paused_window
            # Speaking-window pulls are extremely frequent. Quiet to DEBUG.
            # A PAUSED window pull is the critical end-of-utterance event.
            if window.state == SpeechState.PAUSED:
                log.info(
                    "*** Transcriber got PAUSE window seq=%d samples=%d (drained=%d, qsize=%d) ***",
                    window.seq, window.audio.size,
                    drained, self.window_in.qsize(),
                )
            else:
                log.debug(
                    "Transcriber got window seq=%d state=%s samples=%d (drained=%d, qsize=%d)",
                    window.seq, window.state.value, window.audio.size,
                    drained, self.window_in.qsize(),
                )

            if window.audio.size == 0:
                log.info("Transcriber: empty window seq=%d, skipping", window.seq)
                continue

            text = await loop.run_in_executor(
                self._executor, self._transcribe_sync, window.audio
            )

            if not text:
                log.debug("Transcriber: empty result (%d samples)", window.audio.size)
                continue

            is_partial = window.state != SpeechState.PAUSED
            update = TranscriptUpdate(
                text=text,
                is_partial=is_partial,
                seq=window.seq,
                speech_state=window.state,
                produced_at=time.monotonic(),
            )
            self._last_text = text
            self._last_emit_seq = window.seq
            preview = text if len(text) <= 90 else text[:87] + "..."
            log.info(
                "Whisper -> '%s'  (partial=%s, state=%s, %d samples)",
                preview, is_partial, window.state.value, window.audio.size,
            )
            for q in self.transcript_outs:
                q.put_latest(update)
            primary = self.transcript_outs[0]
            # Partial puts are extremely frequent — keep them at DEBUG.
            # Final (partial=False) puts mark utterance end; LOG LOUDLY.
            if is_partial:
                log.debug(
                    "Transcriber put transcript seq=%d partial=True -> %d queue(s) (primary qsize=%d/%d)",
                    window.seq, len(self.transcript_outs),
                    primary.qsize(), primary.maxsize(),
                )
            else:
                sizes = ", ".join(
                    f"{q.qsize()}/{q.maxsize()}" for q in self.transcript_outs
                )
                log.info(
                    "*** Transcriber put FINAL transcript seq=%d -> %d queue(s) [%s] ***",
                    window.seq, len(self.transcript_outs), sizes,
                )
                if self.session_log is not None:
                    self.session_log.log_transcript(text)
                log.info(
                    "Transcriber loop continuing after FINAL seq=%d (back to await window_in.get)",
                    window.seq,
                )

    # --------------------------------------------------------- inference call
    def _transcribe_sync(self, audio: np.ndarray) -> str:
        assert self._model is not None
        try:
            segments, _info = self._model.transcribe(
                audio,
                language=self.cfg.language,
                beam_size=self.cfg.beam_size,
                temperature=self.cfg.temperature,
                condition_on_previous_text=self.cfg.condition_on_previous_text,
                no_speech_threshold=self.cfg.no_speech_threshold,
                vad_filter=self.cfg.vad_filter,
                word_timestamps=False,
            )
            # Concatenating segments is cheap and keeps the generator drained.
            return " ".join(seg.text.strip() for seg in segments).strip()
        except RuntimeError as exc:
            msg = str(exc)
            if ("cublas" in msg.lower() or "cudnn" in msg.lower()
                    or "cuda" in msg.lower()):
                log.error(
                    "CUDA runtime error during transcription (%s). "
                    "Falling back to CPU int8 for the rest of this session.",
                    exc,
                )
                try:
                    self._model = WhisperModel(
                        self.cfg.model_size,
                        device="cpu",
                        compute_type="int8",
                        cpu_threads=self.cfg.cpu_threads,
                        num_workers=self.cfg.num_workers,
                    )
                    log.info("Whisper switched to CPU (int8). Retrying transcribe…")
                    segments, _info = self._model.transcribe(
                        audio,
                        language=self.cfg.language,
                        beam_size=self.cfg.beam_size,
                        temperature=self.cfg.temperature,
                        condition_on_previous_text=self.cfg.condition_on_previous_text,
                        no_speech_threshold=self.cfg.no_speech_threshold,
                        vad_filter=self.cfg.vad_filter,
                        word_timestamps=False,
                    )
                    return " ".join(seg.text.strip() for seg in segments).strip()
                except Exception:
                    log.exception("CPU fallback also failed")
                    return ""
            log.exception("Faster-Whisper transcribe failed")
            return ""
        except Exception:
            log.exception("Faster-Whisper transcribe failed")
            return ""

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
