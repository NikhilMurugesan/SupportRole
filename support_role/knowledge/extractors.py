"""Text extraction for the knowledge base.

Each supported format returns a plain UTF-8 string. Empty / unreadable
files return "". No third-party imports beyond what's listed in
requirements.txt for the matching extension.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


SUPPORTED_SUFFIXES: tuple[str, ...] = (".pdf", ".docx", ".txt", ".md")


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            return _extract_pdf(path)
        if suffix == ".docx":
            return _extract_docx(path)
        if suffix in (".txt", ".md"):
            return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        log.exception("Failed to extract text from %s", path)
        return ""
    log.warning("Unsupported file type: %s", path)
    return ""


def _extract_pdf(path: Path) -> str:
    # pypdf is pure-Python and small. Lazy-import so the rest of the app
    # doesn't pay the import cost when no PDFs are present.
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        if txt:
            parts.append(txt)
    return "\n\n".join(parts)


def _extract_docx(path: Path) -> str:
    from docx import Document  # python-docx

    doc = Document(str(path))
    parts: list[str] = [p.text for p in doc.paragraphs if p.text]
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text]
            if cells:
                parts.append(" | ".join(cells))
    return "\n".join(parts)


def chunk_text(
    text: str,
    chunk_chars: int,
    overlap_chars: int,
) -> list[str]:
    """Split text into overlapping char windows on sentence-ish boundaries."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_chars, n)
        # Try to break at the nearest sentence boundary going backwards.
        if end < n:
            for sep in ("\n\n", ". ", "\n", "? ", "! "):
                idx = text.rfind(sep, start + chunk_chars // 2, end)
                if idx != -1:
                    end = idx + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(0, end - overlap_chars)
    return chunks
