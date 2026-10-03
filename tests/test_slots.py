"""An op with slots runs at most that many apply calls at once, through a demand queue."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from chunkmirage import demand, open_source
from chunkmirage.ops.base import Op
from chunkmirage.pipeline import Pipeline

SRC = "synthetic://blobs?shape=32,32,32&chunk=8,8,8&levels=1"
state = {"now": 0, "max": 0}
lock = threading.Lock()


class _Gpu(Op):
    name = "test_gpu"
    slots = 2

    def apply(self, block):
        with lock:
            state["now"] += 1
            state["max"] = max(state["max"], state["now"])
        time.sleep(0.05)
        with lock:
            state["now"] -= 1
        return block


def test_no_more_than_slots_apply_calls_run_at_once():
    state.update(now=0, max=0)
    p = Pipeline(open_source(SRC), [{"op": "threshold", "low": 1}, _Gpu(), {"op": "threshold", "low": 2}])
    chunks = [(z, y, x) for z in range(4) for y in range(2) for x in range(2)]
    with ThreadPoolExecutor(16) as pool:
        list(pool.map(lambda idx: p.chunk(0, idx), chunks))
    assert state["max"] == 2
    queue = demand.queues["op test_gpu"]
    assert queue.slots == 2 and queue.stats()["done"][0] >= len(chunks)
