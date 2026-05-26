"""Plain-text session log.

Append-only single-file record of everything spoken (final transcripts
only — partials are skipped to keep the file readable) and every LLM
answer produced during the session.

One file per app launch under ``transcripts/`` at the project root,
named ``session_YYYYMMDD-HHMMSS.txt``. Entries are timestamped but not
otherwise structured.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


class SessionLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # Touch the file with a session header so an empty session is
        # still distinguishable from a missing log.
        header = (
            f"=== SupportRole session started "
            f"{datetime.now().isoformat(timespec='seconds')} ===\n"
        )
        try:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(header)
        except Exception:
            log.exception("Failed to initialize session log at %s", self.path)

    @classmethod
    def for_run(cls, root_dir: Optional[Path] = None) -> "SessionLog":
        root = root_dir or Path("transcripts")
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        return cls(root / f"session_{ts}.txt")

    def _write(self, kind: str, text: str) -> None:
        text = text.strip()
        if not text:
            return
        stamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{stamp}] {kind}: {text}\n"
        try:
            with self._lock, self.path.open("a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            log.exception("Session log write failed (%s)", kind)

    def log_transcript(self, text: str) -> None:
        """Record a finalized (post-PAUSE) transcript."""
        self._write("USER", text)

    def log_answer(self, text: str) -> None:
        """Record a complete LLM answer."""
        self._write("LLM ", text)
