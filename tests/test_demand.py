"""Work done because clients asked for it: in the order they want it, dropped when they stop
waiting (chunkmirage.demand)."""

import threading
import time

import pytest
from starlette.testclient import TestClient

from chunkmirage import demand
from chunkmirage.server import create_app


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


def test_the_finest_level_goes_first_then_the_latest_burst():
    queue, order = demand.Queue(slots=1), []
    release = _blocked(queue)
    threads = [_ask(queue, "level 2", 2, order), _ask(queue, "level 1", 1, order)]
    _settle()
    time.sleep(demand.BURST_S + 0.05)  # a later burst of requests, at levels 0 and 1
    threads += [_ask(queue, "level 0", 0, order), _ask(queue, "level 1, later", 1, order)]
    _settle()
    release.set()
    for t in threads:
        t.join(5)
    assert order == ["level 0", "level 1, later", "level 1", "level 2"]


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


def test_a_cancelled_claim_stops_work_before_it_starts():
    claim = demand.Claim()
    claim.cancel()
    with pytest.raises(demand.Cancelled):
        demand.claimed(claim, lambda: None)


def test_the_queues_are_reported():
    body = TestClient(create_app({})).get("/api/queue").json()
    assert {"computing", "waiting on queued work", "slots"} <= set(body["requests"])
