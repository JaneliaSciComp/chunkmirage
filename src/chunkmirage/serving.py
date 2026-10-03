"""Serving an app from Python as ``chunkmirage serve`` does: the port bound before anything
announces it, a callback once the server accepts connections, in this thread or another.

For applications that build their own app (``create_app``, or one that mounts it) and run
it themselves::

    server = chunkmirage.serve(app, port=0, on_ready=lambda s: print(s.url))
    ...
    server.stop()
"""

from __future__ import annotations

import socket
import threading
from collections.abc import Callable

from chunkmirage.netutil import bind_socket, public_host_for


class Server:
    """``app`` served by uvicorn on ``sock``, a socket already bound and listening
    (``netutil.bind_socket``), so ``port`` and ``url`` are known before it runs.

    ``run`` serves in this thread until stopped (in the main thread, Ctrl-C stops it);
    ``start`` serves in a background thread and returns once connections are accepted.
    ``on_ready(server)`` is called then, from the server's event loop, so it should be
    quick. ``ssl`` is ``(certfile, keyfile)``.
    """

    def __init__(
        self,
        app,
        sock: socket.socket,
        *,
        ssl: tuple[str, str] | None = None,
        log_level: str = "info",
        on_ready: Callable[[Server], None] | None = None,
    ):
        import uvicorn

        self.sock = sock
        self.host, self.port = sock.getsockname()[:2]
        self.https = ssl is not None
        tls = {"ssl_certfile": ssl[0], "ssl_keyfile": ssl[1]} if ssl else {}
        config = uvicorn.Config(app, host=self.host, port=self.port, log_level=log_level, **tls)
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: BaseException | None = None
        outer = self

        class _Uvicorn(uvicorn.Server):
            async def startup(self, sockets=None):
                await super().startup(sockets=sockets)
                if self.started:  # listening on its socket, the app's startup done
                    outer._ready.set()
                    if on_ready is not None:
                        on_ready(outer)

        self._server = _Uvicorn(config)

    @property
    def url(self) -> str:
        """Where other machines reach it: this machine's network address when bound to
        every interface, ``localhost`` for a loopback address."""
        return f"{'https' if self.https else 'http'}://{public_host_for(self.host)}:{self.port}"

    @property
    def running(self) -> bool:
        return bool(self._server.started) and not self._server.should_exit

    def run(self) -> None:
        """Serve in this thread until ``stop`` (or Ctrl-C, in the main thread)."""
        self._server.run(sockets=[self.sock])

    def start(self, timeout: float = 60) -> Server:
        """Serve in a background thread; returns once it accepts connections."""

        def target():
            try:
                self.run()
            except BaseException as e:  # noqa: BLE001 - reported by start
                self.error = e
            finally:
                self._ready.set()

        self._thread = threading.Thread(target=target, name=f"chunkmirage:{self.port}", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout) or not self._server.started:
            self.stop()
            raise RuntimeError(f"the server on port {self.port} did not start: {self.error!r}")
        return self

    def stop(self, timeout: float = 10) -> None:
        """Stop serving: requests in progress finish, then the port is released."""
        self._server.should_exit = True
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout)
            if not self._thread.is_alive():
                self.sock.close()

    def wait(self) -> None:
        """Block until a server started in the background stops."""
        if self._thread is not None:
            self._thread.join()

    def __enter__(self) -> Server:
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


def serve(
    app,
    host: str = "0.0.0.0",
    port: int | None = 0,
    *,
    on_ready: Callable[[Server], None] | None = None,
    ssl: tuple[str, str] | None = None,
    log_level: str = "info",
    block: bool = False,
) -> Server:
    """Serve ``app`` (``create_app``'s, or any ASGI app) on ``host``: on ``port``, 0 for any
    free one, None for the first free one from 8000. The port is bound before this returns
    or calls anything, so no other process can take it. ``on_ready(server)`` is called once
    connections are accepted. By default the server runs in a background thread and is
    returned running (``server.port``, ``server.url``, ``server.stop()``); with ``block``
    it runs in this thread until stopped, as ``chunkmirage serve`` does."""
    server = Server(app, bind_socket(host, port), ssl=ssl, log_level=log_level, on_ready=on_ready)
    if block:
        server.run()
        return server
    return server.start()


__all__ = ["Server", "serve"]
