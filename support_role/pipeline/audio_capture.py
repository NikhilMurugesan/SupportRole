"""Windows loopback audio capture.

Uses the `soundcard` library to capture the default speaker as a loopback
input via WASAPI. Audio continues to play normally on the source PC
(non-exclusive shared mode).

A blocking capture loop runs in a dedicated worker thread; chunks are
pushed into an asyncio queue (latest-wins) for downstream consumers.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np

try:
    import soundcard as sc  # type: ignore
except Exception as exc:  # pragma: no cover - import-time
    sc = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

from ..config import CONFIG, AudioConfig
from .util_queue import LatestWinsQueue

log = logging.getLogger(__name__)


class LoopbackCapture:
    """Background thread that streams loopback audio chunks as float32 mono.

    Chunks have shape (n_samples,) at `cfg.sample_rate`.
    """

    def __init__(self, out_queue: LatestWinsQueue, cfg: AudioConfig = CONFIG.audio):
        if sc is None:
            raise RuntimeError(
                f"`soundcard` failed to import: {_IMPORT_ERROR!r}. "
                "Install it with `pip install soundcard`."
            )
        self.cfg = cfg
        self.out = out_queue
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    # ------------------------------------------------------------------ public
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="LoopbackCapture", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # ----------------------------------------------------------------- internal
    def _run(self) -> None:
        chunk_frames = int(self.cfg.sample_rate * self.cfg.capture_chunk_ms / 1000)
        try:
            mic = self._open_device()
        except Exception:
            log.exception("Failed to open audio device")
            return

        try:
            with mic.recorder(
                samplerate=self.cfg.sample_rate,
                channels=self.cfg.channels,
                blocksize=chunk_frames,
            ) as rec:
                while not self._stop.is_set():
                    data = rec.record(numframes=chunk_frames)
                    # data: (frames, channels) float32 in [-1, 1]
                    if data.ndim == 2 and data.shape[1] > 1:
                        mono = data.mean(axis=1)
                    else:
                        mono = data.reshape(-1)
                    # Use a contiguous copy so downstream consumers don't share
                    # the recorder's internal buffer.
                    chunk = np.ascontiguousarray(mono, dtype=np.float32)
                    self.out.put_latest(chunk)
        except Exception:
            log.exception("Audio capture crashed")

    # ----------------------------------------------------------- device picker
    def _open_device(self):
        """Resolve the configured input device.

        * mode="loopback" -> default (or named) speaker exposed as a recordable
          source via WASAPI loopback. Audio still plays normally.
        * mode="mic"      -> default (or named) microphone.
        """
        mode = self.cfg.input_mode
        name_filter = (self.cfg.device_name or "").lower().strip()

        if mode == "loopback":
            if name_filter:
                speakers = [s for s in sc.all_speakers() if name_filter in s.name.lower()]
                speaker = speakers[0] if speakers else sc.default_speaker()
            else:
                speaker = sc.default_speaker()
            log.info("Capture mode=loopback source=%s", speaker.name)
            return sc.get_microphone(id=str(speaker.name), include_loopback=True)

        if mode == "mic":
            if name_filter:
                mics = [
                    m for m in sc.all_microphones(include_loopback=False)
                    if name_filter in m.name.lower()
                ]
                mic = mics[0] if mics else sc.default_microphone()
            else:
                mic = sc.default_microphone()
            log.info("Capture mode=mic source=%s", mic.name)
            return mic

        raise ValueError(f"Unknown input_mode: {mode!r}")
