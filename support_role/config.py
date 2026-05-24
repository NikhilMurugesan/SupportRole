"""Central configuration for SupportRole.

All latency-sensitive tunables live here so they can be adjusted without
touching pipeline code. Defaults are tuned for RTX 4080 SUPER + Ryzen 9
7950X3D, targeting < 2 s perceived latency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass(frozen=True)
class AudioConfig:
    # Where to record from:
    #   "loopback" -> captures system speaker output (Zoom, YouTube, etc.)
    #   "mic"      -> captures your default microphone
    #   "udp"      -> receives int16 PCM packets over UDP (see UdpConfig)
    input_mode: Literal["loopback", "mic", "udp"] = "udp"
    # Optional: pick a specific device by (sub-)name. None = system default.
    # Examples: "Headset Microphone", "Speakers (Realtek", "BlackHole".
    device_name: Optional[str] = None

    # WebRTC VAD only supports 8/16/32/48 kHz mono int16. We resample to 16 kHz.
    sample_rate: int = 16_000
    channels: int = 1
    # Capture chunk in milliseconds. Smaller = lower latency, more overhead.
    # 200 ms is a good compromise; VAD frames inside are 20 ms.
    capture_chunk_ms: int = 200
    # VAD frame size (must be 10/20/30 ms for webrtcvad).
    vad_frame_ms: int = 20
    # 0 (least aggressive) ... 3 (most aggressive at filtering non-speech).
    vad_aggressiveness: int = 2
    # How many silent VAD frames before we declare a "pause".
    # How many silent VAD frames before we declare a "pause".
    # 50 frames * 20 ms = ~1000 ms — matches the "any pause >1 s should
    # trigger an answer" requirement.
    silence_frames_for_pause: int = 50
    # How many speech frames before we declare speech "started".
    speech_frames_to_start: int = 2  # ~40 ms
    # Max rolling audio kept for transcription, in seconds.
    max_audio_window_s: float = 8.0


@dataclass(frozen=True)
class UdpConfig:
    """Settings for `input_mode="udp"`.

    Matches the simple sender script: int16 PCM, stereo, 48 kHz.
    The receiver downmixes to mono and resamples to AudioConfig.sample_rate.
    """
    listen_ip: str = "0.0.0.0"
    # Matches the reference sender (LISTEN_PORT = 50007). If you get
    # WinError 10013 on startup, another process is already bound to this
    # port (often the standalone test receiver). Stop it first, or pick a
    # different free port. To check Windows' excluded dynamic ranges run:
    #   netsh int ipv4 show excludedportrange protocol=udp
    listen_port: int = 50007
    source_sample_rate: int = 48_000
    source_channels: int = 2
    # Max UDP datagram we expect (sender uses BLOCK_SIZE=1024 frames stereo int16
    # => 1024*2*2 = 4096 bytes). Give some headroom.
    recv_buffer: int = 8192
    # OS-level socket receive buffer (bytes). Bigger = more jitter tolerance.
    socket_rcvbuf: int = 1 << 20  # 1 MiB


@dataclass(frozen=True)
class WhisperConfig:
    # medium.en is roughly 3x more accurate than small.en on noisy / accented
    # speech and still fits comfortably in VRAM alongside qwen3:8b on the
    # RTX 4080 SUPER (medium fp16 ≈ 1.5 GB). Latency increase is ~150-200 ms
    # per partial transcription, still well under our pause-trigger window.
    model_size: str = "medium.en"
    device: Literal["cuda", "cpu", "auto"] = "cuda"
    compute_type: str = "float16"          # FP16 on RTX 4080 SUPER
    cpu_threads: int = 0                    # 0 = let CT2 decide
    num_workers: int = 1
    beam_size: int = 1
    temperature: float = 0.0
    condition_on_previous_text: bool = False
    no_speech_threshold: float = 0.5
    # Re-transcribe the rolling window this often. Lower = faster
    # partial updates (and more GPU usage). 180 ms keeps the 4080 SUPER
    # busy but still leaves headroom for the qwen3:8b decoder.
    transcribe_interval_ms: int = 180
    # VAD filter inside Whisper itself — we already gate, so off.
    vad_filter: bool = False
    language: str = "en"


@dataclass(frozen=True)
class LLMConfig:
    # Ollama HTTP endpoint (local).
    base_url: str = "http://127.0.0.1:11434"
    # qwen3:8b Q4_K_M ≈ 5.2 GB VRAM. With faster-whisper small.en FP16
    # (~1 GB) and KV cache @ num_ctx=2048 (~0.5 GB), total stays well
    # under the 14 GB budget on an RTX 4080 SUPER (16 GB).
    model: str = "qwen3:8b"
    # Allow a fuller answer — enough room to cover MULTIPLE questions in
    # the same utterance (e.g. "what is X and how does it differ from Y?").
    num_predict: int = 380
    # Bumped from 2048 → 4096 so a full 8-second multi-question utterance
    # (~1200 chars) PLUS the RAG snippets PLUS the answer all fit in ctx.
    num_ctx: int = 4096
    temperature: float = 0.2
    top_p: float = 0.9
    # Cooldown (ms) between two consecutive LLM emissions. In Q&A mode this
    # is the minimum spacing between answering two back-to-back questions,
    # so the previous answer stays readable.
    debounce_ms: int = 800
    # Minimum new chars in rolling buffer before we bother re-prompting.
    min_new_chars: int = 4
    # Rolling context window passed to the LLM (chars from end of transcript).
    # Sized to cover the full ~8 s VAD window so multi-question utterances
    # like "what is X? how does Y work? and explain Z" are sent intact —
    # otherwise only the trailing fragment reaches the LLM and earlier
    # questions in the same breath are silently dropped.
    context_chars: int = 1200
    # Trigger policy: ONLY fire on VAD pauses (~1 s of silence). Mid-partial
    # '?' triggers used to cause the LLM to answer garbled half-questions
    # like "The kind?" and burn 4-5 s of LLM time on noise, starving the
    # real questions that came right after. Set fire_on_partial_question=True
    # to restore the old behavior.
    fire_on_partial_question: bool = False
    question_mode: bool = False
    # In Q&A mode we keep cancellation OFF so the user can actually read the
    # answer before it gets replaced by the next one.
    cancel_on_new_input: bool = False
    request_timeout_s: float = 60.0

    system_prompt: str = (
        "You are a realtime assistant. Answer the user's most recent "
        "utterance directly and helpfully, in SIMPLE spoken English that "
        "a non-expert can follow. "
        "BUT the answer MUST also include the important technical keywords, "
        "terms, and concept names related to the topic (e.g. specific "
        "tools, APIs, algorithms, design patterns, library names). "
        "WRAP every important keyword in double asterisks like **this** so "
        "the UI can highlight it. Wrap at least 3 keywords per answer. "
        "Cover the small underlying concepts too, briefly. "
        "IF THE UTTERANCE CONTAINS MULTIPLE QUESTIONS OR REQUESTS "
        "(joined by 'and', 'also', commas, or separate sentences), "
        "YOU MUST ANSWER EVERY ONE OF THEM. Group bullets per question "
        "and prefix each group with a one-line header like "
        "'Q1: <short restatement>' on its own line, then 2-4 '- ' bullets "
        "under it, then 'Q2: <...>' and its bullets, and so on. "
        "For a single question, skip the 'Q1:' header and just emit "
        "3-6 '- ' bullets. "
        "Total length ~40-110 words depending on how many questions. "
        "First bullet of each group is the direct answer in plain words; "
        "remaining bullets each highlight 1-2 **keywords** in plain "
        "explanation. "
        "No markdown other than '- ', '**keyword**', and the 'Qn:' "
        "headers. No preface. No reasoning aloud. Do not repeat the "
        "question verbatim. "
        "NEVER mention or cite source documents, file names, PDFs, "
        "snippet numbers, or phrases like 'according to', 'based on "
        "the document', '(1)', '[2]'. Speak as if the knowledge is "
        "your own — a candidate answering an interview question."
    )


@dataclass(frozen=True)
class UIConfig:
    # Which window(s) to show:
    #   "main"    -> normal resizable window with transcript + hint history
    #   "overlay" -> small click-through always-on-top overlay
    #   "both"    -> show both
    mode: Literal["main", "overlay", "both"] = "main"

    width: int = 520
    height: int = 160
    margin: int = 24
    # Corner: "tr", "tl", "br", "bl"
    corner: str = "br"
    opacity: float = 0.82
    font_family: str = "Segoe UI"
    font_size: int = 14
    # Auto-clear hints after this many ms of no updates.
    stale_clear_ms: int = 12_000
    # Smooth re-render cap.
    max_fps: int = 30
    # In Q&A mode, only swap the visible answer once the new answer has
    # accumulated this many characters. Prevents the "delete & retype"
    # flicker between two questions. With ~40-word bullet answers we wait
    # for the first bullet line to be mostly formed before swapping.
    min_swap_chars: int = 24
    # Card-stack mode: keep at most this many answer cards on screen.
    # New answers are pushed onto the TOP and the older cards slide down.
    # When the stack would exceed `max_cards`, the oldest card at the
    # bottom is removed.
    max_cards: int = 6
    # Rotating palette used to color **keywords** in answers.
    keyword_colors: tuple[str, ...] = (
        "#ff6b6b",  # red
        "#7ee787",  # green
        "#79c0ff",  # cyan/blue
        "#ffa657",  # orange
        "#d2a8ff",  # magenta/purple
        "#f0c674",  # yellow
    )


@dataclass(frozen=True)
class QueueConfig:
    # Bounded queues, latest-wins semantics in producers.
    audio_qsize: int = 8
    transcript_qsize: int = 4
    hint_qsize: int = 8


@dataclass(frozen=True)
class KnowledgeConfig:
    """RAG (retrieval-augmented generation) document knowledge base.

    Drop PDF / DOCX / TXT / MD files into `inbox_dir`. They will be chunked,
    embedded with Ollama (`embed_model`), stored in a persistent ChromaDB
    under `vector_dir`, and moved to `processed_dir`. At query time, the
    rolling transcript is embedded and the top-K matching chunks are passed
    to the LLM alongside the question.
    """
    enabled: bool = True
    # Folders relative to the project root (overridden in main if needed).
    root_dir: str = "knowledge"
    inbox_subdir: str = "inbox_docs"
    processed_subdir: str = "processed_docs"
    vector_subdir: str = "vector_db"

    # ChromaDB collection name.
    collection_name: str = "support_role_docs"

    # Embedding model served by the same local Ollama instance. Pull once:
    #   ollama pull nomic-embed-text
    embed_model: str = "nomic-embed-text"

    # Chunking — small chunks favor precise retrieval over long context.
    chunk_chars: int = 700
    chunk_overlap_chars: int = 120

    # How often the watcher scans `inbox_dir` for new files (seconds).
    scan_interval_s: float = 5.0

    # Retrieval at query time.
    top_k: int = 3
    # Drop matches with distance above this (Chroma cosine distance, 0=best).
    max_distance: float = 1.2
    # If True, the LLM answers from general knowledge when no relevant
    # chunks are found. If False, it must say it doesn't know.
    allow_general_knowledge: bool = True
    # Max chars from retrieved chunks fed to the LLM (keeps prompts small
    # so latency stays low).
    max_context_chars: int = 1800


@dataclass(frozen=True)
class AppConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    udp: UdpConfig = field(default_factory=UdpConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    queues: QueueConfig = field(default_factory=QueueConfig)
    knowledge: KnowledgeConfig = field(default_factory=KnowledgeConfig)


CONFIG = AppConfig()
