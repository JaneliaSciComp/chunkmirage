"""chunkmirage.serve: an app served from Python on a port bound first, announced once it
accepts connections, in a background thread that stops cleanly."""

import json
import socket
import urllib.request

import chunkmirage
from chunkmirage import Pipeline, create_app, open_source

SRC = "synthetic://blobs?shape=16,16,16&chunk=8,8,8&levels=1"


def test_serve_in_a_thread_announces_answers_and_stops():
    ready = []
    app = create_app({"d": Pipeline(open_source(SRC), [])})
    server = chunkmirage.serve(app, host="127.0.0.1", port=0, on_ready=ready.append, log_level="warning")
    try:
        assert ready == [server] and server.port > 0 and server.running
        assert server.url == f"http://localhost:{server.port}"
        with urllib.request.urlopen(f"http://127.0.0.1:{server.port}/api/datasets", timeout=10) as r:
            assert json.loads(r.read()) == {"datasets": ["d"]}
        with urllib.request.urlopen(f"http://127.0.0.1:{server.port}/d/zarr3/s0/c/0/0/0", timeout=10) as r:
            assert r.status == 200
    finally:
        server.stop()
    assert not server.running
    with socket.socket() as s:  # the port is free again, for a server (connections it closed linger)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", server.port))
