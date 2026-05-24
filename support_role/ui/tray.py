"""System tray icon (pystray) running on its own thread."""

from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

from PIL import Image, ImageDraw
import pystray

log = logging.getLogger(__name__)


def _make_icon() -> Image.Image:
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((4, 4, 60, 60), fill=(20, 24, 32, 255), outline=(120, 220, 255, 255), width=3)
    d.ellipse((24, 24, 40, 40), fill=(120, 220, 255, 255))
    return img


class TrayIcon:
    def __init__(self, on_quit: Callable[[], None]) -> None:
        self._on_quit = on_quit
        self._icon: Optional[pystray.Icon] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._icon = pystray.Icon(
            "SupportRole",
            _make_icon(),
            "SupportRole — realtime assistant",
            menu=pystray.Menu(
                pystray.MenuItem("Quit", self._quit),
            ),
        )
        self._thread = threading.Thread(target=self._icon.run, name="Tray", daemon=True)
        self._thread.start()

    def _quit(self, _icon: pystray.Icon, _item) -> None:
        try:
            self._on_quit()
        finally:
            if self._icon is not None:
                self._icon.stop()

    def stop(self) -> None:
        if self._icon is not None:
            self._icon.stop()
