"""Expensive work done because a client asked for it, in the order clients want it, and
dropped when no client waits for it any more.

A request holds a :class:`Claim` on the work it needs (the server sets one per chunk request,
as :data:`current_claim`, and cancels it when the client disconnects). Expensive work goes
through a :class:`Queue`: a few jobs run at once; among those waiting, the finest level goes
first (a client asking for one place at two levels, as a viewer does to show a coarse
placeholder while the fine chunk computes, wants the finer), then the first asked for. A waiting job whose every claim is
cancelled is dropped without running, and whoever waits on it gets :class:`Cancelled`; one
already running finishes, and its result is kept. Work asked for outside a request (from
Python, with no claim) is never dropped. The server bounds how many requests compute at once
with :class:`Slots`, and a request waiting on a queue's work gives its slot up meanwhile, so
cheap requests never wait behind expensive ones. The browser engine's ``web/src/demand.ts``
keeps the same rules for its expensive work.
"""

from __future__ import annotations

import contextlib
import itertools
import threading
import time
from collections import Counter
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any


class Cancelled(Exception):
    """No request wants this work any more."""


class Claim:
    """A request's hold on the work it needs, cancelled when its client stops waiting."""

    def __init__(self):
        self.cancelled = False
        self._hooks: list[Callable[[], None]] = []
        self._lock = threading.Lock()

    def on_cancel(self, hook: Callable[[], None]) -> None:
        with self._lock:
            if not self.cancelled:
                self._hooks.append(hook)
                return
        hook()

    def cancel(self) -> None:
        with self._lock:
            if self.cancelled:
                return
            self.cancelled = True
            hooks, self._hooks = self._hooks, []
        for hook in hooks:
            hook()


current_claim: ContextVar[Claim | None] = ContextVar("chunkmirage_claim", default=None)


class Slots:
    """At most ``n`` requests computing at once, each in its own thread, which bounds the
    memory and CPU chunk work takes. A request waiting on a queue's work gives its slot up
    meanwhile (its thread only waits), so requests that need nothing expensive are never held
    up behind ones that do."""

    def __init__(self, n: int):
        self.n = n
        self._sem = threading.Semaphore(n)
        self._lock = threading.Lock()
        self.computing = 0
        self.waiting = 0  # on queued work, their slot given up

    def _count(self, attr: str, d: int) -> None:
        with self._lock:
            setattr(self, attr, getattr(self, attr) + d)

    @contextlib.contextmanager
    def held(self):
        self._sem.acquire()
        self._count("computing", 1)
        try:
            yield
        finally:
            self._count("computing", -1)
            self._sem.release()

    @contextlib.contextmanager
    def given_up(self):
        self._count("computing", -1)
        self._count("waiting", 1)
        self._sem.release()
        try:
            yield
        finally:
            self._sem.acquire()
            self._count("waiting", -1)
            self._count("computing", 1)

    def stats(self) -> dict:
        return {
            "computing": self.computing,
            "waiting on queued work": self.waiting,
            "slots": self.n,
        }


current_slots: ContextVar[Slots | None] = ContextVar("chunkmirage_slots", default=None)


def claimed(claim: Claim, fn: Callable[..., Any], *args, slots: Slots | None = None) -> Any:
    """``fn(*args)`` for a request holding ``claim``, in one of ``slots`` (run in a server
    thread); a request given up on before its turn comes costs nothing."""
    with slots.held() if slots is not None else contextlib.nullcontext():
        if claim.cancelled:
            raise Cancelled
        tokens = current_claim.set(claim), current_slots.set(slots)
        try:
            return fn(*args)
        finally:
            current_slots.reset(tokens[1])
            current_claim.reset(tokens[0])


def _waiting():
    """While a request's thread waits on queued work, its slot is free for others."""
    slots = current_slots.get()
    return slots.given_up() if slots is not None else contextlib.nullcontext()


class _Job:
    def __init__(self, level: int):
        self.level = level
        self.seq = 0  # when it was first asked for
        self.claims: set[Claim] = set()
        self.kept = False  # asked for without a claim: never dropped
        self.state = "new"  # new, waiting, running, done, dropped
        self.result: Any = None
        self.error: BaseException | None = None
        self.done = threading.Event()


class Queue:
    """Jobs keyed by what they compute; see the module's description for the order."""

    def __init__(self, slots: int = 3):
        self.slots = slots
        self._cond = threading.Condition()
        self._jobs: dict[Any, _Job] = {}
        self._running = 0
        self._seq = itertools.count()
        self.dropped = 0
        self.done_per_level: Counter[int] = Counter()
        self.seconds = 0.0

    def run(self, key: Any, level: int, compute: Callable[[], Any]) -> Any:
        """``compute()``'s result for ``key``, computed once however many ask at once."""
        claim = current_claim.get()
        if claim is not None and claim.cancelled:
            raise Cancelled  # its request was given up on: nothing more for it
        with self._cond:
            job = self._jobs.get(key)
            owner = job is None
            if owner:
                job = self._jobs[key] = _Job(level)
                job.seq = next(self._seq)
            if claim is None:
                job.kept = True
            else:
                job.claims.add(claim)
        if claim is not None:
            claim.on_cancel(lambda: self._release(key, job, claim))
        if not owner:
            if not job.done.is_set():
                with _waiting():
                    job.done.wait()
            if job.error is not None:
                raise job.error
            return job.result
        return self._own(key, job, compute, claim)

    def _own(self, key, job: _Job, compute, claim: Claim | None):
        with self._cond:
            if job.state == "new":
                job.state = "waiting"
            self._pump()
            waits = job.state == "waiting"
        if waits:  # not holding the lock while the slot is taken back
            with _waiting():
                with self._cond:
                    while job.state == "waiting":
                        self._cond.wait()
        with self._cond:
            if job.state == "dropped":
                raise Cancelled
        started = time.perf_counter()
        try:
            job.result = compute()
            return job.result
        except BaseException as e:
            job.error = e
            raise
        finally:
            with self._cond:
                self._jobs.pop(key, None)
                self._running -= 1
                job.state = "done"
                if job.error is None:
                    self.done_per_level[job.level] += 1
                    self.seconds += time.perf_counter() - started
                self._pump()
            job.done.set()

    def _pump(self) -> None:  # holding the lock
        while self._running < self.slots:
            waiting = [j for j in self._jobs.values() if j.state == "waiting"]
            if not waiting:
                break
            best = min(waiting, key=lambda j: (j.level, j.seq))  # finest, then first asked
            best.state = "running"
            self._running += 1
        self._cond.notify_all()

    def _release(self, key, job: _Job, claim: Claim) -> None:
        with self._cond:
            job.claims.discard(claim)
            if job.state not in ("new", "waiting") or job.claims or job.kept:
                return
            job.state = "dropped"
            self._jobs.pop(key, None)
            self.dropped += 1
            job.error = Cancelled()
            self._cond.notify_all()
        job.done.set()

    def stats(self) -> dict:
        """Jobs running and waiting per level, dropped, done per level, and the time done took."""
        with self._cond:
            count = lambda state: dict(  # noqa: E731
                sorted(Counter(j.level for j in self._jobs.values() if j.state == state).items())
            )
            return {
                "running": count("running"),
                "waiting": count("waiting"),
                "dropped": self.dropped,
                "done": dict(sorted(self.done_per_level.items())),
                "seconds": round(self.seconds, 3),
            }


queues: dict[str, Queue] = {}  # the process's queues of expensive work, by name, for /api/queue
