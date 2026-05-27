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
    #   "udp"      -> receives int16 PCM packets over UDP (see UdpConfig)
    input_mode: Literal["loopback", "udp"] = "udp"
    # Optional: pick a specific device by (sub-)name. None = system default.
    # Examples: "Speakers (Realtek", "BlackHole".
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
    vad_aggressiveness: int = 3
    # How many silent VAD frames before we declare a "pause".
    # 25 frames * 20 ms = ~500 ms — the adaptive pause detector in the
    # ConversationOrchestrator applies real thresholds on top.  This low
    # value ensures the orchestrator sees pauses early and can decide
    # whether they are meaningful.
    silence_frames_for_pause: int = 25
    # How many speech frames before we declare speech "started".
    speech_frames_to_start: int = 8  # ~160 ms
    # Max rolling audio kept for transcription, in seconds.
    # 25 s is enough to cover a long multi-question stream like
    # "what is your RAG architecture? how did you design it? why a vector
    # store and not a regular DB? what embeddings did you use?" spoken in
    # one breath. Whisper medium.en on RTX 4080 SUPER transcribes 25 s in
    # well under 1 s, so latency stays acceptable.
    max_audio_window_s: float = 25.0


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
    num_predict: int = 700
    # Bumped from 4096 → 8192 so a full ~25 s multi-question utterance
    # (~3500 chars) PLUS the RAG snippets PLUS the answer all fit in ctx.
    num_ctx: int = 8192
    temperature: float = 0.2
    top_p: float = 0.9
    # Cooldown (ms) between two consecutive LLM emissions.
    debounce_ms: int = 800
    # Minimum new chars in rolling buffer before we bother re-prompting.
    min_new_chars: int = 4
    # Rolling context window passed to the LLM (chars from end of transcript).
    context_chars: int = 3500
    # --- Legacy flags (kept for backwards compat, ignored by orchestrator) ---
    fire_on_partial_question: bool = False
    question_mode: bool = False
    cancel_on_new_input: bool = False
    request_timeout_s: float = 60.0

    system_prompt: str = (
        "You are a realtime conversational assistant that listens continuously "
        "and responds naturally. Adapt your response style to the conversation:\n"
        "- For QUESTIONS: give a direct, helpful answer with key details.\n"
        "- For STATEMENTS: acknowledge and add relevant insight or explanation.\n"
        "- For FOLLOW-UPS: build on your previous response with additional detail.\n"
        "- For CORRECTIONS: acknowledge the correction and adjust your understanding.\n"
        "- For TOPIC CHANGES: smoothly transition to the new topic.\n\n"
        "Use SIMPLE spoken English that a non-expert can follow. "
        "Include important technical keywords, terms, and concept names. "
        "WRAP every important keyword in double asterisks like **this** so "
        "the UI can highlight it. Wrap at least 3 keywords per answer. "
        "Cover the small underlying concepts too, briefly.\n\n"
        "CRITICAL: LATEST QUESTION PRIORITY OVERRIDE RULES:\n"
        "1. ANSWER ONLY THE LATEST ACTIVE QUESTION. Do not answer older questions "
        "present in chat history or repeated in the transcript. If the user presents "
        "multiple questions in their transcript, answer only the last complete meaningful question.\n"
        "2. Treat repeated speech-to-text text as duplicate noise. Merge it into one intent "
        "and do NOT repeat the same answer.\n"
        "3. Chat history is only supporting context for background or continuity. Never let "
        "old context or older questions override or control the response to the newest user question.\n"
        "4. If the latest input contains feedback plus a new question, apply the feedback (acknowledge it) "
        "and answer the new question. Do not re-answer previous questions.\n"
        "5. If the latest question was already answered and no new information was added, do not repeat the full answer. "
        "Briefly acknowledge or wait for the next question.\n"
        "6. Before generating, silently decide: 'What is the newest question or instruction the user expects me to respond to now?' "
        "Then answer ONLY that active question/instruction.\n"
        "7. If the latest input says 'next question', 'let's continue', or 'let's try another', do not repeat the previous answer. "
        "Briefly acknowledge and move forward/wait for the next question.\n"
        "8. Do not use RAG or memory unless the latest question clearly requires it.\n\n"
        "CRITICAL ANSWER QUALITY OVERRIDE:\n"
        "Do not give weak, generic, repeated, or unsupported answers. "
        "Answer the latest active question directly first. "
        "For interview questions, use strong candidate-style wording: direct answer, specific implementation/example, challenge or trade-off, and outcome/business value. "
        "Include at least one concrete detail such as an architecture component, metric calculation, workflow step, failure mode, or business impact. "
        "If asked about Avathon partnerships, clients, expansion plans, recent private results, percentages, or case studies and confirmed context is not provided, do not invent facts. "
        "Say you do not want to assume private partnership details, then answer strategically from physical AI and operational environments. "
        "If the transcript says 'Abathon' and the surrounding context refers to Avathon, treat it as Avathon. "
        "For metrics, explain how metrics are calculated or used. "
        "For RAG, explain query formation, retrieval, metadata filtering or hybrid retrieval, reranking, rejection of irrelevant chunks, grounding, and hallucination reduction when relevant. "
        "For architecture, answer as Input -> Processing -> Retrieval/Logic -> LLM/Decision layer -> Validation -> Output -> Monitoring.\n\n"
        "CRITICAL CODING QUESTION OUTPUT OVERRIDE:\n"
        "If the latest active question is a coding question, programming problem, algorithm/DSA question, "
        "debugging request, or asks to write/solve/implement/give code, actual code must be the main output. "
        "Use the mentioned language; if none is mentioned, default to Python. "
        "For coding questions, do NOT follow the normal short bullet-only format. "
        "Give a brief approach, complete runnable code, sample input/output if useful, and time/space complexity for DSA. "
        "If the user asks for code only, output only the code block and no prose.\n\n"
        "For the single active non-coding question, emit 3-6 '- ' bullets. "
        "Total length ~40-120 words depending on complexity. "
        "First bullet is the direct answer in plain words; "
        "remaining bullets each highlight 1-2 **keywords** in plain explanation. "
        "No markdown other than '- ' and '**keyword**'. No preface. No reasoning aloud. "
        "Do not repeat the question verbatim.\n\n"
        "If a conversation history is provided, use it only for context and "
        "continuity but focus on the LATEST active utterance. "
        "If interview context is provided, use it as domain background "
        "but the live speech has priority. "
        "NEVER mention or cite source documents, file names, PDFs, "
        "snippet numbers, or phrases like 'according to', 'based on "
        "the document', '(1)', '[2]'. Speak as if the knowledge is "
        "your own."
    )


@dataclass(frozen=True)
class OnlineLLMConfig:
    # Runtime source selection is stored separately from this static config
    # so the UI can switch sources without editing tracked source files.
    default_source: Literal["local", "online"] = "local"

    # Amazon Bedrock Runtime. The API key is stored in .env as
    # AWS_BEARER_TOKEN_BEDROCK; AWS credentials/profile also continue to work.
    bedrock_region: str = "ap-south-1"
    bedrock_model_id: str = "openai.gpt-oss-120b-1:0"

    # Local, gitignored runtime files.
    env_file: str = ".env"
    runtime_config_file: str = "support_role/config.local.json"


@dataclass(frozen=True)
class UIConfig:
    # Which window(s) to show:
    #   "main"    -> normal resizable window with transcript + hint history
    #   "overlay" -> small click-through always-on-top overlay
    #   "both"    -> show both
    mode: Literal["main", "overlay", "both"] = "main"

    width: int = 1040
    height: int = 320
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
    max_distance: float = 0.48
    # Equivalent cosine-similarity floor used after retrieval reranking.
    min_similarity: float = 0.52
    # Pull a wider candidate set first so topic filters/reranking can reject
    # unrelated chunks without starving good matches.
    candidate_multiplier: int = 5
    # If True, the LLM answers from general knowledge when no relevant
    # chunks are found. If False, it must say it doesn't know.
    allow_general_knowledge: bool = True
    # Max chars from retrieved chunks fed to the LLM (keeps prompts small
    # so latency stays low).
    max_context_chars: int = 1800
    # Optional per-interview profile injected into every prompt and used to
    # bias retrieval. Edit this file before a new interview instead of
    # relying on vector search to infer the domain from noisy speech.
    interview_context_file: str = "knowledge/interview_context.md"
    max_interview_context_chars: int = 4000


@dataclass(frozen=True)
class ConversationConfig:
    """Conversational reasoning engine settings."""
    # -- Memory --
    max_memory_turns: int = 10           # Max exchanges to keep in memory
    short_term_turns: int = 3            # Turns kept verbatim
    max_memory_chars: int = 6000         # Total memory context budget

    # -- Adaptive pause detection --
    min_pause_ms: int = 500              # Ignore pauses shorter than this
    default_pause_ms: int = 1000         # Default trigger threshold
    max_pause_ms: int = 2000             # Always trigger above this

    # -- Intent classification --
    respond_to_statements: bool = True   # Respond to non-questions
    respond_to_incomplete: bool = False  # Respond to trailing thoughts
    ignore_filler_words: bool = True     # Skip "um", "okay" etc.

    # -- Context switching --
    topic_drift_threshold: float = 0.4   # Keyword overlap below this = new topic

    # -- Response management --
    soft_interrupt: bool = True          # Don't hard-cancel in-flight generation
    interrupt_completion_threshold: float = 0.6  # Let generation finish if >60% done

    # -- Speculative RAG --
    speculative_rag: bool = True         # Pre-fetch RAG during speech
    rag_debounce_ms: int = 500           # Min interval between speculative queries


@dataclass(frozen=True)
class AppConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    udp: UdpConfig = field(default_factory=UdpConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    online_llm: OnlineLLMConfig = field(default_factory=OnlineLLMConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    queues: QueueConfig = field(default_factory=QueueConfig)
    knowledge: KnowledgeConfig = field(default_factory=KnowledgeConfig)
    conversation: ConversationConfig = field(default_factory=ConversationConfig)


CONFIG = AppConfig()
