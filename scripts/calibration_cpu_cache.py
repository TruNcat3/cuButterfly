"""Durable, content-addressed CPU search scores without executable serialization.

The caller supplies algorithm/build identities in ``context`` and binds the
complete scoring model. Candidates must be immutable under their candidate ID;
descriptor contents are hashed on every access, including in-place updates.
Writes stay in a small local buffer until a flush, 256 scores, or one second
of subsequent writes. SQLite transactions are held only during flushes so an
independent preparation process can read committed partial results.
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import sqlite3
import time
from types import MappingProxyType


def _serialized(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def _digest(serialized):
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class DiskScoreCache:
    """Exact scores scoped by model, context, candidate ID and descriptor.

    ``get`` returns ``(False, None)`` on a miss. Storage/validation failures
    raise rather than returning a cache hit. Models are snapshots at ``bind``;
    call ``bind`` again after any in-place model mutation. Descriptor mutation
    is detected without rebinding. ``stats`` is a read-only snapshot.
    """

    def __init__(self, directory, context):
        self._context = _serialized(context)
        directory = pathlib.Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "scores-v1.sqlite3"
        self._connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.executescript("""
            CREATE TABLE IF NOT EXISTS epochs (
                identity TEXT PRIMARY KEY,
                context_json TEXT NOT NULL,
                model_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS scores (
                epoch TEXT NOT NULL,
                candidate TEXT NOT NULL,
                descriptor TEXT NOT NULL,
                score_hex TEXT NOT NULL,
                PRIMARY KEY (epoch, candidate, descriptor)
            ) WITHOUT ROWID;
        """)
        self._epoch = None
        self._pending = {}
        self._last_flush = time.monotonic()
        self._closed = False
        self._stats = dict(hits=0, misses=0, puts=0, flushes=0, committed_scores=0, binds=0)

    @property
    def stats(self):
        return MappingProxyType(dict(self._stats, pending_scores=len(self._pending)))

    def _check_open(self):
        if self._closed:
            raise RuntimeError("score cache is closed")

    def bind(self, model):
        self._check_open()
        serialized = _serialized(model)
        epoch = _digest(_serialized([self._context, serialized]))
        if epoch != self._epoch:
            self.flush()
            self._connection.execute(
                "INSERT OR IGNORE INTO epochs VALUES (?, ?, ?)",
                (epoch, self._context, serialized))
            self._epoch = epoch
        self._stats["binds"] += 1
        return epoch

    def _key(self, candidate_id, descriptor):
        self._check_open()
        if self._epoch is None:
            raise RuntimeError("bind a model before accessing scores")
        if not isinstance(candidate_id, str):
            raise TypeError("candidate_id must be a string")
        return self._epoch, candidate_id, _digest(_serialized(descriptor))

    def get(self, candidate_id, descriptor):
        key = self._key(candidate_id, descriptor)
        encoded = self._pending.get(key)
        if encoded is None:
            row = self._connection.execute(
                "SELECT score_hex FROM scores WHERE epoch=? AND candidate=? AND descriptor=?", key).fetchone()
            encoded = row[0] if row is not None else None
        if encoded is None:
            self._stats["misses"] += 1
            return False, None
        value = float.fromhex(encoded)
        if math.isnan(value):
            raise ValueError("score cache contains NaN")
        self._stats["hits"] += 1
        return True, value

    def put(self, candidate_id, descriptor, score):
        key = self._key(candidate_id, descriptor)
        value = float(score)
        if math.isnan(value):
            raise ValueError("NaN scores cannot be cached")
        self._pending[key] = value.hex()
        self._stats["puts"] += 1
        if len(self._pending) >= 256 or time.monotonic() - self._last_flush >= 1.0:
            self.flush()

    def flush(self):
        self._check_open()
        if self._pending:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                self._connection.executemany(
                    "INSERT OR REPLACE INTO scores VALUES (?, ?, ?, ?)",
                    ((*key, value) for key, value in self._pending.items()))
                self._connection.execute("COMMIT")
            except BaseException:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise
            self._stats["committed_scores"] += len(self._pending)
            self._stats["flushes"] += 1
            self._pending.clear()
        self._last_flush = time.monotonic()

    def close(self):
        if not self._closed:
            self.flush()
            self._connection.close()
            self._closed = True

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *unused):
        self.close()
