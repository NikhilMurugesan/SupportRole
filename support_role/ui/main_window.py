"""Card-stack realtime window for SupportRole.

Layout:

  ┌─ LIVE TRANSCRIPT  (single, overwritten in place) ─────────┐
  │ what you just said ...                                    │
  └────────────────────────────────────────────────────────────┘

  ┌─ ANSWERS  (newest on top, older pushed down) ─────────────┐
  │ ┌── #N (streaming) ────────────────────────────────────┐  │
  │ │ - direct answer with **keywords** in color          │  │
  │ │ - more bullets...                                   │  │
  │ └─────────────────────────────────────────────────────┘  │
  │ ┌── #N-1 ───────────────────────────────────────────────┐ │
  │ │ - previous answer                                    │ │
  │ └──────────────────────────────────────────────────────┘ │
  │ ... up to CONFIG.ui.max_cards ...                        │
  └────────────────────────────────────────────────────────────┘

Behavior:
* While streaming a given prompt `seq`, the TOP card updates in place.
* When a new `seq` arrives, a brand-new card is inserted at the TOP and
  the previous top card slides down (and is now "frozen").
* When the stack exceeds `max_cards`, the bottom card is removed.
* `**word**` in the LLM output is rendered as a colored span, cycling
  through `CONFIG.ui.keyword_colors`.
"""

from __future__ import annotations

import html
import re
import time
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from ..config import CONFIG


_STYLE = """
QMainWindow { background-color: #15171c; }
QScrollArea { border: none; background: #15171c; }
QWidget#cardHost { background: #15171c; }
QLabel#section { color: #9ab; font-size: 11px; letter-spacing: 1px; }
QLabel#liveTranscript {
    color: #e6e9ef; font-size: 16px; padding: 10px 12px;
    background: #1c1f26; border-radius: 8px;
}
QFrame#answerCard {
    background: #1c1f26; border-radius: 10px;
    border: 1px solid #262a33;
}
QFrame#answerCardActive {
    background: #1f2430; border-radius: 10px;
    border: 1px solid #3a4b66;
}
QLabel#cardHeader {
    color: #6f7a8c; font-size: 10px; letter-spacing: 1px;
    padding: 8px 12px 0 12px;
}
QLabel#cardBody {
    color: #e6e9ef; font-size: 15px; line-height: 1.45;
    padding: 4px 12px 10px 12px;
}
QPushButton {
    background: #232733; color: #e6e9ef; border: 1px solid #2e3340;
    border-radius: 6px; padding: 4px 12px;
}
QPushButton:hover { background: #2c3140; }
QStatusBar { color: #8a93a3; }
"""


# Match **anything** (non-greedy, no embedded newline).
_KEYWORD_RE = re.compile(r"\*\*([^*\n]+?)\*\*")


def _format_answer_html(text: str) -> str:
    """Convert raw LLM output (with **keyword** markers) to colored HTML.

    Keywords cycle through CONFIG.ui.keyword_colors so adjacent keywords
    get different colors. Plain text is HTML-escaped.
    """
    palette = CONFIG.ui.keyword_colors or ("#79c0ff",)
    out: list[str] = []
    i = 0
    color_idx = 0
    for m in _KEYWORD_RE.finditer(text):
        out.append(html.escape(text[i : m.start()]))
        color = palette[color_idx % len(palette)]
        color_idx += 1
        word = html.escape(m.group(1).strip())
        out.append(
            f"<span style='color:{color}; font-weight:600;'>{word}</span>"
        )
        i = m.end()
    out.append(html.escape(text[i:]))
    # Preserve line breaks (qt rich-text needs explicit <br/>).
    return "".join(out).replace("\n", "<br/>")


class _AnswerCard(QFrame):
    """One answer in the stack."""

    def __init__(self, seq: int, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.seq = seq
        self.created_at = time.monotonic()
        self.setObjectName("answerCardActive")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        self.header = QLabel(f"#{seq}   thinking\u2026", self)
        self.header.setObjectName("cardHeader")
        self.body = QLabel("", self)
        self.body.setObjectName("cardBody")
        self.body.setWordWrap(True)
        self.body.setMinimumHeight(80)
        self.body.setTextFormat(Qt.TextFormat.RichText)
        self.body.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        self.body.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding
        )
        self.body.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        v.addWidget(self.header)
        v.addWidget(self.body)

    def set_text(self, raw: str) -> None:
        self.body.setText(_format_answer_html(raw))

    def mark_done(self, latency_ms: int) -> None:
        self.setObjectName("answerCard")
        self.header.setText(f"#{self.seq}   {latency_ms} ms")
        # Re-apply stylesheet so the new object name takes effect.
        self.setStyleSheet(self.styleSheet())
        self.style().unpolish(self)
        self.style().polish(self)


class MainWindow(QMainWindow):
    transcript_signal = pyqtSignal(str, bool)     # (text, is_partial)
    hint_signal = pyqtSignal(str, bool, int)      # (full_text, done, seq)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SupportRole — Realtime Assistant")
        self.resize(720, 560)
        self.setStyleSheet(_STYLE)

        # Stack state.
        self._cards: list[_AnswerCard] = []  # newest first
        self._active_seq: Optional[int] = None
        self._active_started_at = 0.0
        self._last_transcript_at = 0.0

        self._build_ui()
        self._wire_signals()
        self._update_status("Waiting for audio…")

    # ------------------------------------------------------------------ build
    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(14, 14, 14, 8)
        root.setSpacing(8)

        # ---- transcript region
        root.addWidget(self._section_label("LIVE TRANSCRIPT"))
        self.live_transcript = QLabel("Waiting for audio…", self)
        self.live_transcript.setObjectName("liveTranscript")
        self.live_transcript.setWordWrap(True)
        self.live_transcript.setMinimumHeight(60)
        self.live_transcript.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        self.live_transcript.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        root.addWidget(self.live_transcript)

        # ---- answers region (scrollable stack)
        root.addWidget(self._section_label(
            f"ANSWERS  (newest on top, up to {CONFIG.ui.max_cards})"
        ))
        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.card_host = QWidget(self.scroll)
        self.card_host.setObjectName("cardHost")
        self.cards_layout = QVBoxLayout(self.card_host)
        self.cards_layout.setContentsMargins(0, 0, 0, 0)
        self.cards_layout.setSpacing(8)
        self.cards_layout.addStretch(1)  # pushes existing cards to the top
        self.scroll.setWidget(self.card_host)
        root.addWidget(self.scroll, 1)

        # ---- action bar
        actions = QHBoxLayout()
        actions.setSpacing(8)
        self.clear_btn = QPushButton("Clear", self)
        self.clear_btn.clicked.connect(self._clear)
        self.pin_btn = QPushButton("Pin on top: OFF", self)
        self.pin_btn.setCheckable(True)
        self.pin_btn.toggled.connect(self._toggle_on_top)
        actions.addWidget(self.clear_btn)
        actions.addWidget(self.pin_btn)
        actions.addStretch(1)
        root.addLayout(actions)

        # ---- status bar
        self.status = QStatusBar(self)
        self.setStatusBar(self.status)

    def _section_label(self, text: str) -> QLabel:
        lbl = QLabel(text, self)
        lbl.setObjectName("section")
        return lbl

    def _wire_signals(self) -> None:
        self.transcript_signal.connect(self._on_transcript)
        self.hint_signal.connect(self._on_hint)

    # ------------------------------------------------------------------ slots
    def _on_transcript(self, text: str, is_partial: bool) -> None:
        text = text.strip()
        if not text:
            return
        self.live_transcript.setText(text[-600:])
        self._last_transcript_at = time.monotonic()
        self._update_status(
            "Listening…" if is_partial else "Sentence finalized"
        )

    def _on_hint(self, full_text: str, done: bool, seq: int) -> None:
        full_text = full_text.strip()
        now = time.monotonic()

        # New generation? Push a fresh card on top.
        if self._active_seq != seq:
            self._active_seq = seq
            self._active_started_at = now
            self._push_card(seq)

        card = self._cards[0] if self._cards else None
        if card is None:
            return

        if not done:
            # Update top card in place during streaming.
            if full_text:
                card.set_text(full_text)
            latency_ms = int((now - self._active_started_at) * 1000)
            card.header.setText(f"#{seq}   thinking… ({latency_ms} ms)")
            self._update_status(f"Thinking…  ({latency_ms} ms)")
            return

        # done=True: finalize this card; the next seq will push a new one.
        if full_text:
            card.set_text(full_text)
        latency_ms = int((now - self._active_started_at) * 1000)
        card.mark_done(latency_ms)
        self._active_seq = None
        self._update_status(f"Answer ready  ({latency_ms} ms)")

    # ----------------------------------------------------------- card stack
    def _push_card(self, seq: int) -> None:
        card = _AnswerCard(seq, self.card_host)
        # Insert at the TOP (index 0). The trailing stretch keeps cards
        # anchored to the top of the scroll area as they accumulate.
        self.cards_layout.insertWidget(0, card)
        self._cards.insert(0, card)
        self._trim_overflow()
        # Auto-scroll to the top so the new card is visible.
        bar = self.scroll.verticalScrollBar()
        if bar is not None:
            bar.setValue(0)

    def _trim_overflow(self) -> None:
        max_cards = max(1, CONFIG.ui.max_cards)
        while len(self._cards) > max_cards:
            old = self._cards.pop()  # remove oldest (bottom)
            self.cards_layout.removeWidget(old)
            old.deleteLater()

    def _clear(self) -> None:
        self.live_transcript.setText("Waiting for audio…")
        for card in self._cards:
            self.cards_layout.removeWidget(card)
            card.deleteLater()
        self._cards.clear()
        self._active_seq = None

    def _toggle_on_top(self, on: bool) -> None:
        flags = self.windowFlags()
        if on:
            flags |= Qt.WindowType.WindowStaysOnTopHint
            self.pin_btn.setText("Pin on top: ON")
        else:
            flags &= ~Qt.WindowType.WindowStaysOnTopHint
            self.pin_btn.setText("Pin on top: OFF")
        self.setWindowFlags(flags)
        self.show()  # re-show is required after changing flags

    def _update_status(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.status.showMessage(
            f"{msg}   ·   {ts}   ·   {CONFIG.llm.model}   ·   "
            f"input: {CONFIG.audio.input_mode}",
            4000,
        )


def create_main_app() -> tuple[QApplication, "MainWindow"]:
    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    win.show()
    return app, win  # type: ignore[return-value]
