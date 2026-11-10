#!/usr/bin/env python3
"""Keep bounded CPU compilation workers fed without rescanning on completion."""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import fcntl
import glob
import heapq
import os
from pathlib import Path
import signal
import sys
import time

import semantic_compile_broker as broker

SCHEMA = "cubutterfly-semantic-compile-producer-v1"


class Queue:
    def __init__(self, directory):
        self.directory = directory
        self.states = {}
        self.counts = dict(pending=0, running=0, ready=0, failed=0)
        self.queued = set()
        self.background = deque()
        self.priority = []
        self.deferred = {}
        self.dispatches = 0
        self.scans = 0
        self.report_cache = {}

    def state(self, identifier, value):
        if value not in self.counts:
            raise ValueError("unknown broker record status: " + str(value))
        old = self.states.get(identifier)
        if old is not None:
            self.counts[old] -= 1
        self.states[identifier] = value
        self.counts[value] += 1

    def read_state(self, identifier):
        path = self.directory / "records" / (identifier + ".json")
        record = broker._read(path) if path.exists() else {}
        if record and (record.get("id") != identifier or
                       record.get("broker_id") != self.broker_id):
            raise ValueError("record identity mismatch")
        return record.get("status", "pending")

    def enqueue(self, identifier):
        if identifier not in self.queued:
            self.queued.add(identifier)
            self.background.append(identifier)

    def discover(self, retry_failed=False):
        self.scans += 1
        with os.scandir(self.directory / "requests") as entries:
            for entry in entries:
                if not entry.name.endswith(".json"):
                    continue
                identifier = entry.name[:-5]
                if identifier in self.states:
                    # A retryable compiler interruption may remain failed
                    # after one attempt. Revisit it on every discovery pass;
                    # otherwise the in-memory queue permanently suppresses
                    # the broker's --retry-failed policy.
                    if retry_failed and self.states[identifier] == "failed":
                        self.enqueue(identifier)
                    continue
                state = self.read_state(identifier)
                self.state(identifier, state)
                if state in ("pending", "running") or (retry_failed and state == "failed"):
                    self.enqueue(identifier)

    def priorities(self, reports):
        groups = []
        errors = []
        cases = {}
        paths = set()
        for pattern in reports:
            paths.update(Path(path) for path in glob.glob(str(pattern)))
        for path in sorted(paths):
            try:
                stat = path.stat()
                signature = (stat.st_mtime_ns, stat.st_size)
                cached = self.report_cache.get(path)
                if cached is None or cached[0] != signature:
                    cached = (signature, broker._read(path).get("waiting_compilation", {}))
                    self.report_cache[path] = cached
                waiting = cached[1]
                if isinstance(waiting, dict):
                    for case, group in waiting.items():
                        if case not in cases or cases[case][0] < stat.st_mtime_ns:
                            cases[case] = (stat.st_mtime_ns, group)
                    continue
                if not isinstance(waiting, list):
                    raise ValueError("waiting_compilation must be dict or list")
                if waiting and all(isinstance(value, str) for value in waiting):
                    waiting = [waiting]
                for index, group in enumerate(waiting):
                    cases[(str(path), index)] = (stat.st_mtime_ns, group)
            except (OSError, ValueError, TypeError) as error:
                errors.append({"path": str(path), "error": str(error)})
        for _, group in cases.values():
            try:
                if not isinstance(group, list) or not all(isinstance(i, str) for i in group):
                    raise ValueError("waiting case must be a list of request IDs")
                # Failed or unknown members cannot close a case in this run.
                if any(self.states.get(i) in (None, "failed") for i in group):
                    continue
                remaining = sorted(set(i for i in group if self.states[i] != "ready"))
                if remaining:
                    groups.append(remaining)
            except (OSError, ValueError, TypeError) as error:
                errors.append({"error": str(error)})
        ranked = {}
        for rank, group in enumerate(sorted(groups, key=lambda g: (len(g), g))):
            for identifier in group:
                ranked.setdefault(identifier, rank)
        self.priority = [(rank, i) for i, rank in ranked.items() if i in self.queued]
        heapq.heapify(self.priority)
        return errors

    def pop(self):
        identifier = None
        if self.dispatches % 4 != 3:
            while self.priority:
                _, candidate = heapq.heappop(self.priority)
                if candidate in self.queued:
                    identifier = candidate
                    break
        if identifier is None:
            while self.background:
                candidate = self.background.popleft()
                if candidate in self.queued:
                    identifier = candidate
                    break
        if identifier is not None:
            self.queued.remove(identifier)
            self.dispatches += 1
        return identifier


def run(directory, jobs=1, watch=False, priority_reports=(), *, retry_failed=False,
        discovery_seconds=5.0, heartbeat_seconds=1.0, _stop_requested=None):
    if jobs < 1 or discovery_seconds <= 0 or heartbeat_seconds <= 0:
        raise ValueError("jobs and intervals must be positive")
    directory = Path(directory)
    previous_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    previous_signals = {}
    stopping = False
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
    def requested():
        return stopping or bool(_stop_requested and _stop_requested())
    try:
        with (directory / "standalone_producer.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("another standalone compile producer owns this broker") from None
            document, compiler = broker._validated_worker(directory)
            if _stop_requested is None:
                for sig in (signal.SIGINT, signal.SIGTERM):
                    previous_signals[sig] = signal.signal(sig, stop)
            queue = Queue(directory)
            queue.broker_id = document["id"]
            started_ns = time.time_ns()
            started = time.monotonic()
            completed = 0
            active = {}
            errors = []
            def heartbeat(state):
                result = dict(schema=SCHEMA, broker_id=document["id"], pid=os.getpid(),
                              jobs=jobs, counts=queue.counts.copy(), **queue.counts,
                              status=state, started_ns=started_ns, heartbeat_ns=time.time_ns(),
                              active=len(active), queued=len(queue.queued),
                              deferred=len(queue.deferred), completed=completed,
                              completions_per_second=completed / max(time.monotonic() - started, 1e-9),
                              discovery_scans=queue.scans, priority_report_errors=errors)
                broker._atomic(directory / "producer_status.json", result)
                return result
            next_discovery = next_heartbeat = 0.0
            try:
                with ThreadPoolExecutor(max_workers=jobs) as executor:
                    while True:
                        now = time.monotonic()
                        if not requested() and now >= next_discovery:
                            queue.discover(retry_failed=retry_failed)
                            for identifier, due in list(queue.deferred.items()):
                                if now >= due:
                                    state = queue.read_state(identifier)
                                    queue.state(identifier, state)
                                    if state in ("running", "pending") or (retry_failed and state == "failed"):
                                        queue.enqueue(identifier)
                                    del queue.deferred[identifier]
                            errors = queue.priorities(priority_reports)
                            next_discovery = time.monotonic() + discovery_seconds
                        while not requested() and len(active) < jobs:
                            identifier = queue.pop()
                            if identifier is None:
                                break
                            queue.state(identifier, "running")
                            future = executor.submit(broker._compile_one, directory, document,
                                directory / "requests" / (identifier + ".json"), compiler, retry_failed)
                            active[future] = identifier
                        if now >= next_heartbeat:
                            heartbeat("stopping" if requested() else "running")
                            next_heartbeat = time.monotonic() + heartbeat_seconds
                        if not active and (requested() or (not watch and not queue.queued and not queue.deferred)):
                            break
                        if active:
                            done, _ = wait(active, timeout=0.1, return_when=FIRST_COMPLETED)
                            for future in done:
                                identifier = active.pop(future)
                                future.result()
                                state = queue.read_state(identifier)
                                queue.state(identifier, state)
                                if state in ("pending", "running"):
                                    queue.deferred[identifier] = time.monotonic() + discovery_seconds
                                else:
                                    completed += 1
                        else:
                            time.sleep(min(0.1, discovery_seconds, heartbeat_seconds))
                result = heartbeat("stopped" if requested() else "complete")
                result["interrupted"] = requested()
                return result
            except BaseException:
                heartbeat("failed")
                raise
    finally:
        for sig, handler in previous_signals.items():
            signal.signal(sig, handler)
        if previous_cuda is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous_cuda


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--broker", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--retry-failed", action="store_true",
                        help="retry broker records previously marked failed")
    parser.add_argument("--priority-report", action="append", default=[], help="report path or quoted glob; repeatable")
    args = parser.parse_args(argv)
    try:
        result = run(args.broker, args.jobs, args.watch, args.priority_report,
                     retry_failed=args.retry_failed)
        print(broker._canonical(result))
        return 130 if result["interrupted"] else int(bool(result["failed"] or result["pending"] or result["running"]))
    except Exception as error:
        print("semantic compile producer: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
