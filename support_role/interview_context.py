"""Optional active interview context shared by retrieval and prompting."""

from __future__ import annotations

import logging
from pathlib import Path

from .config import CONFIG, KnowledgeConfig

log = logging.getLogger(__name__)


def load_interview_context(cfg: KnowledgeConfig = CONFIG.knowledge) -> str:
    """Return the active interview context, if configured and readable."""
    path = Path(cfg.interview_context_file)
    if not path.is_absolute():
        cwd_path = Path.cwd() / path
        project_path = Path(__file__).resolve().parents[1] / path
        path = cwd_path if cwd_path.exists() else project_path
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except FileNotFoundError:
        return ""
    except Exception:
        log.exception("Failed to read interview context from %s", path)
        return ""
    if not text:
        return ""
    return text[: cfg.max_interview_context_chars]
