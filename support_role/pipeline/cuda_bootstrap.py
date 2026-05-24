"""Make the CUDA runtime DLLs shipped by the `nvidia-*` pip wheels visible
to CTranslate2 / faster-whisper on Windows.

This must be imported BEFORE `faster_whisper` (or any module that loads
CT2). It uses `os.add_dll_directory` for each `nvidia\<pkg>\bin` folder
found inside the active venv.

Safe to import on Linux / macOS — it's a no-op there.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

log = logging.getLogger(__name__)


def register_cuda_dll_dirs() -> list[str]:
    """Register nvidia wheel `bin/` dirs with the Windows DLL loader.

    Returns the list of paths that were registered, for logging.
    """
    registered: list[str] = []
    if sys.platform != "win32":
        return registered
    if not hasattr(os, "add_dll_directory"):
        return registered

    # site-packages folder of the current interpreter.
    try:
        import site
        candidates = [Path(p) for p in site.getsitepackages()]
        candidates.append(Path(site.getusersitepackages()))
    except Exception:
        candidates = []

    # Also try sys.prefix/Lib/site-packages (venv case).
    candidates.append(Path(sys.prefix) / "Lib" / "site-packages")

    seen: set[Path] = set()
    for sp in candidates:
        if not sp.exists() or sp in seen:
            continue
        seen.add(sp)
        nvidia_root = sp / "nvidia"
        if not nvidia_root.is_dir():
            continue
        for pkg_dir in nvidia_root.iterdir():
            bin_dir = pkg_dir / "bin"
            if bin_dir.is_dir():
                try:
                    os.add_dll_directory(str(bin_dir))
                    # Also extend PATH so child processes / late loaders see it.
                    os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
                    registered.append(str(bin_dir))
                except OSError:
                    log.debug("Could not register DLL dir: %s", bin_dir)
    return registered


# Run on import so callers just `import cuda_bootstrap` once.
_REGISTERED = register_cuda_dll_dirs()
if _REGISTERED:
    log.info("Registered %d CUDA DLL dirs: %s", len(_REGISTERED), _REGISTERED)
