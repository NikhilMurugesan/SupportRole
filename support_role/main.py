"""Async orchestrator for SupportRole.

Threading model
---------------

* Main thread: PyQt6 event loop (windows + overlay rendering).
* Worker thread: asyncio event loop hosting the entire streaming pipeline.
* Capture thread: blocking `soundcard` recorder OR UDP receiver thread.
* Whisper thread: single worker thread inside the transcriber (blocking CT2).
* Tray thread: pystray icon loop.

Cross-thread communication
--------------------------

* asyncio -> Qt UI: `pyqtSignal.emit(...)` (thread-safe).
* threads -> asyncio: `LatestWinsQueue.put_latest` (uses `call_soon_threadsafe`).
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
import threading
from dataclasses import dataclass
from typing import Optional

from PyQt6.QtWidgets import QApplication

from .config import CONFIG
from .pipeline.context_buffer import ContextPrompt, RollingContextManager
from .pipeline.llm_streamer import HintToken, OllamaStreamer
from .pipeline.session_log import SessionLog
from .pipeline.transcriber import StreamingTranscriber, TranscriptUpdate
from .pipeline.udp_receiver import UdpAudioReceiver
from .pipeline.util_queue import LatestWinsQueue
from .pipeline.vad import SpeechWindow, StreamingVAD
from .ui.main_window import MainWindow
from .ui.overlay import OverlayWindow
from .ui.tray import TrayIcon

log = logging.getLogger(__name__)


@dataclass
class UIHandles:
    """Whichever windows we ended up creating. Either may be None."""
    main: Optional[MainWindow] = None
    overlay: Optional[OverlayWindow] = None


# --------------------------------------------------------------------- pipeline
async def _pipeline_main(ui: UIHandles, stop: asyncio.Event) -> None:
    audio_q: LatestWinsQueue = LatestWinsQueue(CONFIG.queues.audio_qsize)
    window_q: LatestWinsQueue[SpeechWindow] = LatestWinsQueue(CONFIG.queues.audio_qsize)
    # Two transcript queues so the context task and the UI pump don't
    # race over a single one (asyncio.Queue delivers each item to exactly
    # ONE waiter — sharing it would cause PAUSE finals to be eaten by the
    # UI and never reach the LLM).
    transcript_q: LatestWinsQueue[TranscriptUpdate] = LatestWinsQueue(
        CONFIG.queues.transcript_qsize
    )
    transcript_ui_q: LatestWinsQueue[TranscriptUpdate] = LatestWinsQueue(
        CONFIG.queues.transcript_qsize
    )
    prompt_q: LatestWinsQueue[ContextPrompt] = LatestWinsQueue(
        CONFIG.queues.transcript_qsize
    )
    hint_q: LatestWinsQueue[HintToken] = LatestWinsQueue(CONFIG.queues.hint_qsize)

    loop = asyncio.get_running_loop()
    for q in (audio_q, window_q, transcript_q, transcript_ui_q, prompt_q, hint_q):
        q.bind_loop(loop)

    cancel_event = asyncio.Event()

    if CONFIG.audio.input_mode == "udp":
        capture = UdpAudioReceiver(audio_q)
    else:
        # Lazy import: `soundcard` initializes COM, which conflicts with Qt
        # on the main thread when we don't actually need it.
        from .pipeline.audio_capture import LoopbackCapture
        capture = LoopbackCapture(audio_q)
    vad = StreamingVAD(audio_q, window_q)
    session_log = SessionLog.for_run()
    log.info("Session log -> %s", session_log.path)
    transcriber = StreamingTranscriber(
        window_q, [transcript_q, transcript_ui_q], session_log=session_log,
    )

    # ---- knowledge base (RAG) ------------------------------------------
    retriever = None
    indexer = None
    if CONFIG.knowledge.enabled:
        try:
            from .knowledge.indexer import DocumentIndexer
            from .knowledge.retriever import KnowledgeRetriever
            indexer = DocumentIndexer()
            retriever = KnowledgeRetriever()
            log.info(
                "  knowledge   = %s (collection=%s, embed=%s)",
                indexer.vector_dir, CONFIG.knowledge.collection_name,
                CONFIG.knowledge.embed_model,
            )
        except Exception:
            log.exception(
                "Knowledge base failed to initialize \u2014 continuing "
                "without RAG. Did you `pip install chromadb pypdf python-docx` "
                "and `ollama pull %s`?",
                CONFIG.knowledge.embed_model,
            )
            indexer = None
            retriever = None

    context = RollingContextManager(
        transcript_q, prompt_q, cancel_event, retriever=retriever,
    )
    llm = OllamaStreamer(prompt_q, hint_q, cancel_event, session_log=session_log)

    capture.start()
    try:
        tasks = [
            asyncio.create_task(vad.run(stop), name="vad"),
            asyncio.create_task(transcriber.run(stop), name="transcriber"),
            asyncio.create_task(context.run(stop), name="context"),
            asyncio.create_task(llm.run(stop), name="llm"),
            asyncio.create_task(_ui_pump(transcript_ui_q, hint_q, ui, stop), name="ui"),
        ]
        if indexer is not None:
            tasks.append(asyncio.create_task(indexer.run(stop), name="indexer"))
        done, pending = await asyncio.wait(
            tasks, return_when=asyncio.FIRST_EXCEPTION
        )
        for t in pending:
            t.cancel()
        for t in done:
            if (exc := t.exception()) is not None:
                log.error("Task %s crashed: %r", t.get_name(), exc)
    finally:
        capture.stop()
        transcriber.shutdown()


async def _ui_pump(
    transcript_q: LatestWinsQueue[TranscriptUpdate],
    hint_q: LatestWinsQueue[HintToken],
    ui: UIHandles,
    stop: asyncio.Event,
) -> None:
    """Fan transcript + hint updates out to Qt signals on every active window."""
    import time

    async def pump_transcript():
        last_warn = time.monotonic()
        seen = 0
        while not stop.is_set():
            try:
                upd = await asyncio.wait_for(transcript_q.get(), timeout=1.0)
            except asyncio.TimeoutError:
                if seen == 0 and time.monotonic() - last_warn > 10:
                    log.warning("UI pump: still no transcript after 10 s")
                    last_warn = time.monotonic()
                continue
            seen += 1
            # Per-update fanout is DEBUG — only the transcriber's finalized
            # "Whisper -> ..." lines stay at INFO.
            log.debug("UI <- transcript #%d (partial=%s, %d chars)",
                      seen, upd.is_partial, len(upd.text))
            if ui.main is not None:
                ui.main.transcript_signal.emit(upd.text, upd.is_partial)
            if ui.overlay is not None:
                ui.overlay.transcript_signal.emit(upd.text)

    async def pump_hint():
        seen = 0
        first_seq: Optional[int] = None
        while not stop.is_set():
            try:
                tok = await asyncio.wait_for(hint_q.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            seen += 1
            # First token per generation + done events at INFO so we can
            # confirm hints are reaching the UI.
            if tok.done or first_seq != tok.seq:
                first_seq = tok.seq
                log.info(
                    "UI <- hint seq=%d done=%s (%d chars): %r",
                    tok.seq, tok.done, len(tok.full),
                    (tok.full[:80] + "…") if len(tok.full) > 80 else tok.full,
                )
            else:
                log.debug("UI <- hint token seq=%d (%d chars total)",
                          tok.seq, len(tok.full))
            if ui.main is not None:
                ui.main.hint_signal.emit(tok.full, tok.done, tok.seq)
            if ui.overlay is not None:
                ui.overlay.hint_signal.emit(tok.full)

    await asyncio.gather(pump_transcript(), pump_hint())


# ------------------------------------------------------------------------ entry
def _create_ui() -> tuple[QApplication, UIHandles]:
    app = QApplication.instance() or QApplication(sys.argv)
    # Keep the process alive even if the user closes a single window —
    # the tray icon is the real lifecycle anchor.
    app.setQuitOnLastWindowClosed(False)

    ui = UIHandles()
    mode = CONFIG.ui.mode
    if mode in ("main", "both"):
        ui.main = MainWindow()
        ui.main.show()
    if mode in ("overlay", "both"):
        ui.overlay = OverlayWindow()
        ui.overlay.show()
    if ui.main is None and ui.overlay is None:
        ui.main = MainWindow()
        ui.main.show()
    return app, ui


def run(debug: bool = False) -> int:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    # Silence chatty third-party loggers so our pipeline logs stay readable.
    for noisy in (
        "PIL", "PIL.Image", "PIL.PngImagePlugin",
        "httpx", "httpcore", "httpcore.connection", "httpcore.http11",
        "urllib3", "asyncio", "matplotlib", "filelock",
        "huggingface_hub", "faster_whisper",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    app, ui = _create_ui()

    log.info("=" * 60)
    log.info("SupportRole starting")
    log.info("  input_mode = %s", CONFIG.audio.input_mode)
    if CONFIG.audio.input_mode == "udp":
        log.info(
            "  UDP listen  = %s:%d  (expecting %dHz / %dch int16 PCM)",
            CONFIG.udp.listen_ip, CONFIG.udp.listen_port,
            CONFIG.udp.source_sample_rate, CONFIG.udp.source_channels,
        )
    log.info("  whisper     = %s on %s (%s)",
             CONFIG.whisper.model_size, CONFIG.whisper.device, CONFIG.whisper.compute_type)
    log.info("  ollama      = %s @ %s", CONFIG.llm.model, CONFIG.llm.base_url)
    if CONFIG.knowledge.enabled:
        log.info(
            "  rag         = enabled (top_k=%d, embed=%s, inbox=%s)",
            CONFIG.knowledge.top_k, CONFIG.knowledge.embed_model,
            f"{CONFIG.knowledge.root_dir}/{CONFIG.knowledge.inbox_subdir}",
        )
    else:
        log.info("  rag         = disabled")
    log.info("  ui mode     = %s", CONFIG.ui.mode)
    log.info("=" * 60)

    stop_event_holder: dict[str, Optional[asyncio.Event]] = {"e": None}
    loop_holder: dict[str, Optional[asyncio.AbstractEventLoop]] = {"l": None}

    def _async_thread() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop_holder["l"] = loop
        stop = asyncio.Event()
        stop_event_holder["e"] = stop
        try:
            loop.run_until_complete(_pipeline_main(ui, stop))
        except Exception:
            log.exception("Pipeline crashed")
        finally:
            loop.close()
            app.quit()

    worker = threading.Thread(target=_async_thread, name="Pipeline", daemon=True)
    worker.start()

    def request_shutdown() -> None:
        log.info("Shutdown requested")
        stop = stop_event_holder["e"]
        loop = loop_holder["l"]
        if stop is not None and loop is not None:
            loop.call_soon_threadsafe(stop.set)
        app.quit()

    tray = TrayIcon(on_quit=request_shutdown)
    tray.start()

    try:
        signal.signal(signal.SIGINT, lambda *_: request_shutdown())
    except (ValueError, AttributeError):
        pass

    exit_code = app.exec()
    request_shutdown()
    worker.join(timeout=5.0)
    tray.stop()
    return exit_code


if __name__ == "__main__":
    sys.exit(run(debug=False))
