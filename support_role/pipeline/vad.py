"""Streaming WebRTC VAD gate.

Consumes float32 mono audio chunks (any size) from an input queue, runs
WebRTC VAD over fixed-size 20 ms frames, and emits a rolling "active
speech" audio buffer to the downstream transcriber.

It also surfaces speech-state events (start/continue/pause) so the
transcriber and LLM can react without polling.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import webrtcvad

from ..config import CONFIG, AudioConfig
from .util_queue import LatestWinsQueue

log = logging.getLogger(__name__)


class SpeechState(str, Enum):
    IDLE = "idle"
    SPEAKING = "speaking"
    PAUSED = "paused"


@dataclass
class SpeechWindow:
    """Snapshot of the currently-active speech audio buffer."""
    audio: np.ndarray          # float32 mono, 16 kHz
    state: SpeechState
    seq: int                   # monotonically increasing


class StreamingVAD:
    def __init__(
        self,
        audio_in: LatestWinsQueue[np.ndarray],
        window_out: LatestWinsQueue[SpeechWindow],
        cfg: AudioConfig = CONFIG.audio,
    ) -> None:
        self.audio_in = audio_in
        self.window_out = window_out
        self.cfg = cfg
        self._vad = webrtcvad.Vad(cfg.vad_aggressiveness)
        self._frame_samples = int(cfg.sample_rate * cfg.vad_frame_ms / 1000)
        self._max_window_samples = int(cfg.sample_rate * cfg.max_audio_window_s)
        # Rolling speech buffer (only frames classified as speech are kept).
        self._buf: deque[np.ndarray] = deque()
        self._buf_len = 0
        self._tail = np.zeros(0, dtype=np.float32)  # leftover sub-frame samples
        self._state = SpeechState.IDLE
        self._speech_run = 0
        self._silence_run = 0
        self._seq = 0

    # --------------------------------------------------------------- main loop
    async def run(self, stop: asyncio.Event) -> None:
        log.info(
            "VAD started (aggressiveness=%d, frame=%dms, pause=%d frames)",
            self.cfg.vad_aggressiveness,
            self.cfg.vad_frame_ms,
            self.cfg.silence_frames_for_pause,
        )
        chunks_seen = 0
        while not stop.is_set():
            try:
                chunk = await asyncio.wait_for(self.audio_in.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            chunks_seen += 1
            # First chunk at INFO so we know audio is reaching VAD; rest at DEBUG.
            if chunks_seen == 1:
                log.info("VAD got chunk #%d (%d samples)", chunks_seen, chunk.size)
            else:
                log.debug("VAD got chunk #%d (%d samples)", chunks_seen, chunk.size)
            self._process_chunk(chunk)

    # ------------------------------------------------------------ frame engine
    def _process_chunk(self, chunk: np.ndarray) -> None:
        # Concatenate any leftover tail from the previous chunk.
        if self._tail.size:
            chunk = np.concatenate([self._tail, chunk])
        n_full = (chunk.size // self._frame_samples) * self._frame_samples
        frames = chunk[:n_full].reshape(-1, self._frame_samples)
        self._tail = chunk[n_full:].copy()

        any_new_speech = False
        state_changed = False

        for frame in frames:
            is_speech = self._is_speech(frame)
            if is_speech:
                self._speech_run += 1
                self._silence_run = 0
                self._append_speech(frame)
                any_new_speech = True
                if (
                    self._state in (SpeechState.IDLE, SpeechState.PAUSED)
                    and self._speech_run >= self.cfg.speech_frames_to_start
                ):
                    prev = self._state
                    self._state = SpeechState.SPEAKING
                    state_changed = True
                    log.info("VAD: state %s -> SPEAKING", prev.value)
            else:
                self._silence_run += 1
                self._speech_run = 0
                if (
                    self._state == SpeechState.SPEAKING
                    and self._silence_run >= self.cfg.silence_frames_for_pause
                ):
                    self._state = SpeechState.PAUSED
                    state_changed = True
                    buffered_ms = int(self._buf_len * 1000 / self.cfg.sample_rate)
                    log.info("VAD: state SPEAKING -> PAUSED (buffered %d ms speech)", buffered_ms)

        if any_new_speech or state_changed:
            self._emit()

    def _is_speech(self, frame: np.ndarray) -> bool:
        # webrtcvad expects 16-bit PCM bytes.
        pcm16 = np.clip(frame * 32767.0, -32768, 32767).astype(np.int16).tobytes()
        try:
            return self._vad.is_speech(pcm16, self.cfg.sample_rate)
        except Exception:
            return False

    def _append_speech(self, frame: np.ndarray) -> None:
        self._buf.append(frame)
        self._buf_len += frame.size
        # Trim from the left to honor the max window.
        while self._buf_len > self._max_window_samples and self._buf:
            old = self._buf.popleft()
            self._buf_len -= old.size

    def _emit(self) -> None:
        if not self._buf:
            audio = np.zeros(0, dtype=np.float32)
        else:
            audio = np.concatenate(list(self._buf))
        self._seq += 1
        win = SpeechWindow(audio=audio, state=self._state, seq=self._seq)
        self.window_out.put_latest(win)
        dropped = getattr(self.window_out, "last_drop_count", 0)
        # SPEAKING windows fire ~5x/s and flood the log; keep them at DEBUG.
        # PAUSED windows are rare and trigger downstream answers — LOG LOUD.
        if self._state == SpeechState.PAUSED:
            log.info(
                "*** VAD emit PAUSE seq=%d samples=%d -> window_out (qsize=%d/%d, dropped=%d) ***",
                self._seq, audio.size,
                self.window_out.qsize(), self.window_out.maxsize(), dropped,
            )
        else:
            log.debug(
                "VAD emit seq=%d state=%s samples=%d -> window_out (qsize=%d/%d, dropped=%d)",
                self._seq, self._state.value, audio.size,
                self.window_out.qsize(), self.window_out.maxsize(), dropped,
            )

    # Reset between utterances (called externally on long pauses if desired).
    def reset(self) -> None:
        self._buf.clear()
        self._buf_len = 0
        self._tail = np.zeros(0, dtype=np.float32)
        self._state = SpeechState.IDLE
        self._speech_run = 0
        self._silence_run = 0
