from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

from committee.models import TraceEvent

Subscriber = Callable[[TraceEvent], Awaitable[None] | None]


class EventBus:
    """In-process pub/sub. Subscribers: trace writer, CLI renderer, SSE queues."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._subscribers: list[Subscriber] = []

    # attach a consumer (trace writer, CLI renderer, SSE queue)
    def subscribe(self, fn: Subscriber) -> None:
        self._subscribers.append(fn)

    # queue-shaped subscription for the SSE stream
    def subscribe_queue(self) -> asyncio.Queue[TraceEvent]:
        queue: asyncio.Queue[TraceEvent] = asyncio.Queue()

        def push(event: TraceEvent) -> None:
            queue.put_nowait(event)

        self.subscribe(push)
        return queue

    # fan one event out to every subscriber
    async def publish(self, type: str, round: int | None = None, lens: str | None = None,
                      **payload: object) -> None:
        event = TraceEvent(run_id=self.run_id, type=type, round=round, lens=lens, payload=dict(payload))
        for fn in self._subscribers:
            result = fn(event)
            if asyncio.iscoroutine(result):
                await result
