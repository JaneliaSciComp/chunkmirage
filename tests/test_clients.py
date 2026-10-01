"""The server as real clients see it: a running server, over real HTTP.

tensorstore reads every format the way any chunked-array library does; directory listings
are what Fiji's N5 viewer browses (n5's HTTP access, checked by hand against n5 4.0.1 and
n5-zarr 2.0.1); and a request its client abandons while it waits for a compute slot is
never computed.
"""

import socket
import threading
import time
from typing import ClassVar

import numpy as np
import pytest
import tensorstore as ts
import uvicorn
from starlette.testclient import TestClient

from chunkmirage import Pipeline, create_app, open_source
from chunkmirage.core import Box
from chunkmirage.netutil import free_port
from chunkmirage.ops.base import Op

SOURCE = "synthetic://blobs+noise?shape=64,64,96&chunk=32,32,32&levels=2&seed=4"


class _Slow(Op):
    """Test op (not registered, so the op list stays the shipped one): waits, then passes its
    block through, counting the blocks it computed."""

    name = "test_slow"
    seconds: float = 0.0
    computed: ClassVar[list] = []

    def apply_at(self, block, box):
        type(self).computed.append(tuple(box.start))
        time.sleep(self.seconds)
        return block


class _Server:
    def __init__(self, app):
        self.port = free_port("127.0.0.1", 18765)
        config = uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning")
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self):
        self.thread.start()
        while not self.server.started:
            time.sleep(0.02)
        return f"http://127.0.0.1:{self.port}"

    def __exit__(self, *exc):
        self.server.should_exit = True
        self.thread.join(5)


@pytest.fixture(scope="module")
def pipeline():
    return Pipeline(open_source(SOURCE), [{"op": "gaussian", "sigma": 1, "cache": True}])


@pytest.fixture(scope="module")
def served(pipeline):
    with _Server(create_app({"d": pipeline})) as base:
        yield base


@pytest.mark.parametrize(
    "fmt, spec",
    [
        ("zarr", lambda root: {"driver": "zarr", "kvstore": f"{root}/s0/"}),
        ("zarr3", lambda root: {"driver": "zarr3", "kvstore": f"{root}/s0/"}),
        ("n5", lambda root: {"driver": "n5", "kvstore": f"{root}/s0/"}),
        (
            "precomputed",
            lambda root: {"driver": "neuroglancer_precomputed", "kvstore": f"{root}/"},
        ),
    ],
)
def test_tensorstore_reads_every_format_over_http(served, pipeline, fmt, spec):
    arr = ts.open(spec(f"{served}/d/{fmt}"), read=True).result()
    expected = pipeline.read(0, Box((0, 0, 0), pipeline.info(0).shape))
    got = np.asarray(arr.read().result())
    if fmt == "n5":
        got = got.transpose()  # N5 lists axes x first
    elif fmt == "precomputed":
        got = got[..., 0].transpose()  # x, y, z, channel
    np.testing.assert_array_equal(got, expected)


def test_groups_and_levels_list_as_directories(pipeline):
    client = TestClient(create_app({"d": pipeline}))
    html = {"accept": "text/html, image/gif, image/jpeg, *; q=.2, */*; q=.2"}  # Java's default
    for url, headers in (("/d/n5/", {}), ("/d/n5", html)):
        r = client.get(url, headers=headers)
        assert r.headers["content-type"].startswith("text/html")
        assert '<a href="s0/">s0/</a>' in r.text and '<a href="s1/">s1/</a>' in r.text
    assert '<a href=".zarray">' in client.get("/d/zarr/s1/").text
    assert '<a href="zarr.json">' in client.get("/d/zarr3/s0", headers=html).text
    # metadata clients ask for */*, and get metadata as before
    assert client.get("/d/n5", headers={"accept": "*/*"}).json()["multiScale"] is True
    assert client.get("/d/n5/s7/").status_code == 404  # no such level
    assert client.get("/d/precomputed/").status_code == 404  # precomputed has no directories


def test_a_request_abandoned_while_it_waits_is_never_computed():
    _Slow.computed.clear()
    slow = Pipeline(open_source(SOURCE), [_Slow(seconds=1.5)])
    with _Server(create_app({"d": slow}, threads=1)) as base:  # one request computes at once
        port = int(base.rsplit(":", 1)[1])

        def get(path):
            s = socket.create_connection(("127.0.0.1", port))
            s.sendall(f"GET {path} HTTP/1.1\r\nHost: x\r\n\r\n".encode())
            return s

        busy = get("/d/zarr/s0/0/0/0")  # holds the one slot for 1.5 s
        time.sleep(0.3)
        waiting = get("/d/zarr/s0/1/1/1")  # waits for the slot...
        time.sleep(0.3)
        waiting.close()  # ...and its client gives up
        busy.settimeout(10)
        assert b"200 OK" in busy.recv(64)
        time.sleep(1.0)  # the slot came free after the client left
        busy.close()
    assert len(_Slow.computed) == 1, _Slow.computed  # only the first chunk was computed
