"""Work done because clients asked for it: in the order they want it, dropped when they stop
waiting (chunkmirage.demand)."""

import threading
import time

import numpy as np
import pytest
from starlette.testclient import TestClient

from chunkmirage import demand
from chunkmirage.core import ArrayInfo, Box
from chunkmirage.server import create_app
from chunkmirage.sources.base import ChunkedSource


def _blocked(queue: demand.Queue):
    """Occupy the queue's one slot until the returned event is set."""
    release, started = threading.Event(), threading.Event()

    def hold():
        started.set()
        release.wait()

    threading.Thread(target=queue.run, args=("hold", 0, hold), daemon=True).start()
    started.wait()
    return release


def _ask(queue, key, level, order, claim=None):
    def work():
        order.append(key)
        return key

    def ask():
        if claim is not None:
            demand.current_claim.set(claim)
        try:
            queue.run(key, level, work)
        except demand.Cancelled:
            order.append(f"{key} dropped")

    t = threading.Thread(target=ask, daemon=True)
    t.start()
    return t


def _settle():
    time.sleep(0.05)


def test_the_finest_level_goes_first_then_the_first_asked():
    queue, order = demand.Queue(slots=1), []
    release = _blocked(queue)
    threads = []
    for key, level in (("level 2", 2), ("level 1", 1), ("level 0", 0), ("level 1, later", 1)):
        threads.append(_ask(queue, key, level, order))
        _settle()  # asked in this order
    release.set()
    for t in threads:
        t.join(5)
    assert order == ["level 0", "level 1", "level 1, later", "level 2"]


def test_work_nobody_waits_for_is_dropped_unrun():
    queue, order = demand.Queue(slots=1), []
    release = _blocked(queue)
    gone, kept = demand.Claim(), demand.Claim()
    threads = [
        _ask(queue, "abandoned", 0, order, gone),
        _ask(queue, "shared", 0, order, gone),
        _ask(queue, "shared", 0, order, kept),  # someone else still wants this one
        _ask(queue, "from python", 1, order),  # no claim: never dropped
    ]
    _settle()
    gone.cancel()
    release.set()
    for t in threads:
        t.join(5)
    assert "abandoned dropped" in order and "abandoned" not in order
    assert order.count("shared") == 1 and "from python" in order
    assert queue.stats()["dropped"] == 1


def test_a_request_waiting_on_queued_work_frees_its_slot():
    # one slot, one queue slot taken: a request that needs queued work waits for it without
    # holding the slot, so a request needing nothing expensive runs meanwhile
    slots, queue = demand.Slots(1), demand.Queue(slots=1)
    release = _blocked(queue)
    done = []

    def expensive():
        demand.claimed(
            demand.Claim(), queue.run, "fit", 0, lambda: done.append("expensive"), slots=slots
        )

    def cheap():
        demand.claimed(demand.Claim(), done.append, "cheap", slots=slots)

    a = threading.Thread(target=expensive, daemon=True)
    a.start()
    _settle()
    b = threading.Thread(target=cheap, daemon=True)
    b.start()
    b.join(2)
    assert done == ["cheap"] and slots.stats()["waiting on queued work"] == 1
    release.set()
    a.join(5)
    assert done == ["cheap", "expensive"] and slots.stats()["computing"] == 0


def test_running_work_gives_up_once_no_request_waits_for_it():
    # a long job asks demand.cancelled() between its steps: still wanted while any request
    # waits on it, given up once the last of them is
    queue, started, results = demand.Queue(slots=1), threading.Event(), []

    def long_job():
        started.set()
        while not demand.cancelled():
            time.sleep(0.01)
        raise demand.Cancelled

    def ask(claim):
        demand.current_claim.set(claim)
        try:
            results.append(queue.run("fit", 0, long_job))
        except demand.Cancelled:
            results.append("gave up")

    first, second = demand.Claim(), demand.Claim()
    a = threading.Thread(target=ask, args=(first,), daemon=True)
    a.start()
    started.wait(2)
    b = threading.Thread(target=ask, args=(second,), daemon=True)
    b.start()
    _settle()
    first.cancel()
    _settle()
    assert results == [], "the second request still waits on it"
    second.cancel()
    a.join(2)
    b.join(2)
    assert results == ["gave up", "gave up"] and not demand.cancelled()


def test_a_cancelled_claim_stops_work_before_it_starts():
    claim = demand.Claim()
    claim.cancel()
    with pytest.raises(demand.Cancelled):
        demand.claimed(claim, lambda: None)


def test_the_queues_are_reported():
    body = TestClient(create_app({})).get("/api/queue").json()
    assert {"computing", "waiting on queued work", "slots"} <= set(body["requests"])


def test_a_request_keeps_its_place_however_long_its_input_takes_to_read():
    """Two requests for one level: the first arrives first but its input reads slowly, so its
    job reaches the queue second. With the one slot busy meanwhile, it still runs first: its
    place in line is when its request arrived, not when its job was queued."""
    queue = demand.Queue(slots=1)
    release = _blocked(queue)
    early, late = demand.Claim(), demand.Claim()  # the requests, in the order they arrived
    order = []
    late_thread = _ask(queue, "late", 0, order, late)  # its input was quick to read
    time.sleep(0.1)
    early_thread = _ask(queue, "early", 0, order, early)  # its slow read done only now
    time.sleep(0.1)
    release.set()
    for t in (late_thread, early_thread):
        t.join(5)
    assert order == ["early", "late"]


@pytest.fixture
def gather_threads():
    yield demand.set_gather_threads
    demand.set_gather_threads(max(4, demand._cores()))


def _stage(shape, chunk, compute, key):
    info = ArrayInfo(shape=shape, dtype=np.float32, chunk_shape=chunk, voxel_size=(1.0,) * len(shape),
                     units=("",) * len(shape), axes=("z", "y", "x")[-len(shape):])
    return ChunkedSource(info, compute, None, key)


def test_a_read_computes_its_chunks_at_once_for_its_request(gather_threads):
    gather_threads(8)
    seen = []

    def slow(idx):
        seen.append((demand.current_claim.get(), demand.current_slots.get()))
        time.sleep(0.1)
        return np.full((2, 2, 2), sum(idx), np.float32)

    src, claim = _stage((4, 4, 4), (2, 2, 2), slow, "slow"), demand.Claim()
    started = time.perf_counter()
    out = demand.claimed(claim, src.read, Box((0, 0, 0), (4, 4, 4)), slots=demand.Slots(1))
    took = time.perf_counter() - started
    assert out[3, 3, 3] == 3 and out[0, 0, 0] == 0
    assert took < 0.5, f"8 chunks of 0.1 s took {took:.2f} s: one after another"
    assert all(c is claim for c, _ in seen), "the pool's threads work for the request"
    assert sum(s is None for _, s in seen) >= 6, "and do not hold its slot"


def test_inputs_gathered_by_the_pools_threads_in_turn_cannot_deadlock(gather_threads):
    # each of 8 chunks reads 8 chunks of the stage before it, on a pool of 2 threads
    gather_threads(2)
    inner = _stage((8, 8, 8), (2, 2, 2), lambda idx: (time.sleep(0.01), np.ones((2, 2, 2), np.float32))[1], "inner")

    def outer_chunk(idx):
        box = Box(tuple(4 * i for i in idx), tuple(4 * i + 4 for i in idx))
        return inner.read(box)[::2, ::2, ::2]

    outer = _stage((4, 4, 4), (2, 2, 2), outer_chunk, "outer")
    result = []
    t = threading.Thread(target=lambda: result.append(outer.read(Box((0, 0, 0), (4, 4, 4)))), daemon=True)
    t.start()
    t.join(10)
    assert result and result[0].sum() == 64, "the read finished"
