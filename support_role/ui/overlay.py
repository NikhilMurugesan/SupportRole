"""Transparent, always-on-top, non-focus-stealing overlay (PyQt6).

Two text regions:

* Top: latest partial transcript (dim).
* Bottom: streaming LLM hints (bright).

A Qt signal funnels updates from the asyncio side onto the GUI thread.
"""

from __future__ import annotations

import logging
from typing import Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QGuiApplication, QPainter, QPainterPath
from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

from ..config import CONFIG, UIConfig

log = logging.getLogger(__name__)


class OverlayWindow(QWidget):
    # Signals are thread-safe; emit from asyncio side.
    transcript_signal = pyqtSignal(str)
    hint_signal = pyqtSignal(str)

    def __init__(self, cfg: UIConfig = CONFIG.ui) -> None:
        super().__init__()
        self.cfg = cfg
        self._build_window()
        self._build_layout()
        self._wire_signals()
        self._position()

        self._stale_timer = QTimer(self)
        self._stale_timer.setSingleShot(True)
        self._stale_timer.timeout.connect(self._clear_hint)

    # ------------------------------------------------------------------ setup
    def _build_window(self) -> None:
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool                # no taskbar entry
            | Qt.WindowType.WindowTransparentForInput  # click-through
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setWindowOpacity(self.cfg.opacity)
        self.resize(self.cfg.width, self.cfg.height)

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(6)

        font_t = QFont(self.cfg.font_family, self.cfg.font_size - 2)
        font_h = QFont(self.cfg.font_family, self.cfg.font_size, QFont.Weight.DemiBold)

        self.transcript_label = QLabel("…", self)
        self.transcript_label.setFont(font_t)
        self.transcript_label.setStyleSheet("color: rgba(200,200,210,210);")
        self.transcript_label.setWordWrap(True)
        self.transcript_label.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)

        self.hint_label = QLabel("", self)
        self.hint_label.setFont(font_h)
        self.hint_label.setStyleSheet("color: rgba(120,220,255,255);")
        self.hint_label.setWordWrap(True)

        layout.addWidget(self.transcript_label, 1)
        layout.addWidget(self.hint_label, 2)

    def _wire_signals(self) -> None:
        self.transcript_signal.connect(self._on_transcript)
        self.hint_signal.connect(self._on_hint)

    def _position(self) -> None:
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        geo = screen.availableGeometry()
        m = self.cfg.margin
        if self.cfg.corner == "br":
            x = geo.right() - self.width() - m
            y = geo.bottom() - self.height() - m
        elif self.cfg.corner == "tr":
            x = geo.right() - self.width() - m
            y = geo.top() + m
        elif self.cfg.corner == "bl":
            x = geo.left() + m
            y = geo.bottom() - self.height() - m
        else:
            x = geo.left() + m
            y = geo.top() + m
        self.move(x, y)

    # ------------------------------------------------------------------ paint
    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt API
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(self.rect().toRectF(), 14, 14)
        p.fillPath(path, QColor(15, 17, 22, 210))
        p.setPen(QColor(80, 90, 110, 180))
        p.drawPath(path)

    # ------------------------------------------------------------------ slots
    def _on_transcript(self, text: str) -> None:
        # Show the tail so users see "what was just said" rather than the start.
        tail = text[-220:]
        self.transcript_label.setText(tail)

    def _on_hint(self, text: str) -> None:
        self.hint_label.setText(text)
        self._stale_timer.start(self.cfg.stale_clear_ms)

    def _clear_hint(self) -> None:
        self.hint_label.setText("")


def create_overlay_app() -> tuple[QApplication, OverlayWindow]:
    app = QApplication.instance() or QApplication([])
    win = OverlayWindow()
    win.show()
    return app, win  # type: ignore[return-value]
