"""
Small WebSocket broadcaster for pose events.

Usage:
    from ws_bridge import WSBroadcaster
    ws = WSBroadcaster(port=8765)
    ws.start()
    ws.broadcast({"event": "down"})
    ws.broadcast({"event": "up", "count": 5})
    ws.stop()

Requires: pip install websockets
"""

import asyncio
import json
import threading
from typing import Any, Dict, Set

import websockets


class WSBroadcaster:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765):
        self.host = host
        self.port = port
        self.loop = None
        self.thread = None
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self._started = threading.Event()
        self._failed: Exception | None = None

    async def _handler(self, websocket, path=None):
        self.clients.add(websocket)
        try:
            async for _ in websocket:  # Keep alive; ignore inbound messages
                pass
        finally:
            self.clients.discard(websocket)

    def start(self):
        if self.thread:
            return

        def run():
            try:
                async def main():
                    async with websockets.serve(self._handler, self.host, self.port):
                        self.loop = asyncio.get_running_loop()
                        self._started.set()
                        await asyncio.Future()  # run forever
                asyncio.run(main())
            except Exception as e:  # pragma: no cover - runtime safety
                self._failed = e
                self._started.set()

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        self._started.wait(timeout=2.0)
        if self._failed:
            raise self._failed
        if not self.loop:
            raise RuntimeError("WebSocket loop did not start")

    def broadcast(self, payload: Dict[str, Any]):
        if not self.loop:
            return
        message = json.dumps(payload)

        async def _send():
            if not self.clients:
                return
            await asyncio.gather(
                *[client.send(message) for client in list(self.clients)],
                return_exceptions=True,
            )

        asyncio.run_coroutine_threadsafe(_send(), self.loop)

    def stop(self):
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None
            self.loop = None
