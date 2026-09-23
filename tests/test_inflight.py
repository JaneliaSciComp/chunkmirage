import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from chunkmirage.cache import LRUCache
from chunkmirage.core import ArrayInfo
from chunkmirage.sources.base import ChunkedSource


def test_concurrent_requests_share_one_computation():
    calls = []
    lock = threading.Lock()

    def compute(idx):
        with lock:
            calls.append(idx)
        time.sleep(0.2)
        return np.full((8, 8, 8), idx[0], dtype=np.uint8)

    info = ArrayInfo((32, 32, 32), np.uint8, (8, 8, 8), (1, 1, 1), ("nm",) * 3, ("z", "y", "x"))
    src = ChunkedSource(info, compute, LRUCache(), "t")
    with ThreadPoolExecutor(8) as ex:
        results = list(ex.map(lambda _: src.chunk((1, 0, 0)), range(8)))
    assert len(calls) == 1  # eight concurrent requests, one computation
    for r in results:
        assert r[0, 0, 0] == 1

    # a failure must not wedge waiters
    def bad(idx):
        raise RuntimeError("boom")

    src2 = ChunkedSource(info, bad, LRUCache(), "t2")
    errors = []

    def go():
        try:
            src2.chunk((0, 0, 0))
        except RuntimeError as e:
            errors.append(e)

    ts = [threading.Thread(target=go) for _ in range(4)]
    [t.start() for t in ts]
    [t.join(timeout=5) for t in ts]
    assert all(not t.is_alive() for t in ts)
    assert 1 <= len(errors) <= 4
