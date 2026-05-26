"""Queue helpers used across the async pipeline.

`LatestWinsQueue` is a small wrapper over `asyncio.Queue` that drops older
items when a newer one is produced. This is critical for low latency:
a slow consumer must never cause the producer to back up real-time audio.
"""

from __future__ import annotations

import asyncio
from typing import Generic, Optional, TypeVar

T = TypeVar("T")


class LatestWinsQueue(Generic[T]):
    def __init__(self, maxsize: int = 4) -> None:
        self._q: asyncio.Queue[T] = asyncio.Queue(maxsize=maxsize)
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Bind the event loop. Required so `put_latest` works from a thread."""
        self._loop = loop

    # ------------------------------------------------------------------ consumer
    async def get(self) -> T:
        return await self._q.get()

    def get_nowait(self) -> T:
        return self._q.get_nowait()

    def qsize(self) -> int:
        return self._q.qsize()

    def maxsize(self) -> int:
        return self._q.maxsize

    # ------------------------------------------------------------------ producer
    def put_latest(self, item: T) -> None:
        """Push the freshest item. Drops the oldest entry if the queue is full.

        Safe to call from any thread once `bind_loop()` has been called.
        """
        loop = self._loop
        if loop is None:
            # Fall back to direct put if we're on the same loop already.
            self._direct_put_latest(item)
            return
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is loop:
            self._direct_put_latest(item)
            return
        loop.call_soon_threadsafe(self._direct_put_latest, item)

    def _direct_put_latest(self, item: T) -> None:
        q = self._q
        dropped = 0
        while q.full():
            try:
                q.get_nowait()
                dropped += 1
            except asyncio.QueueEmpty:
                break
        try:
            q.put_nowait(item)
        except asyncio.QueueFull:
            # Highly unlikely after the drain above; just drop.
            pass
        self.last_drop_count = dropped

    # Number of items evicted by the most recent put (for diagnostics).
    last_drop_count: int = 0
