"""Bounded CPU/I/O comparison against the original 256-row score cache.

Run with the project Python, CUDA_VISIBLE_DEVICES='', and optionally
--candidates 65536. Temporary databases stay beneath the repository root.
Reported /proc/self/io write_bytes cover this process, not whole-device traffic.
"""
import argparse
import json
import pathlib
import random
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from calibration_cpu_cache import DiskScoreCache
from calibration_space import candidate_id


class OriginalBatchScoreCache(DiskScoreCache):
    """Reproduce the original batch size and insertion-order flush."""

    _BATCH_SIZE = 256

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


def _write_bytes():
    values = dict(line.split(":") for line in pathlib.Path("/proc/self/io").read_text().splitlines())
    return int(values["write_bytes"])


def _rows(count):
    rows = []
    for index in range(count):
        point = {
            "operator": "fft", "precision": "fp32", "logN": 12 + index % 13,
            "batch": 1 << ((index // 13) % 10), "backend": "online-reorder",
            "prefix_threads": 32 << ((index // 130) % 4),
            "suffix_threads": 32 << ((index // 520) % 4),
            "prefix_ept": 1 << ((index // 2080) % 4),
            "suffix_ept": 1 << ((index // 8320) % 4),
            "local_stages": 5 + index % 7,
            "shared_layout": "linear" if index % 2 else "xor-swizzle",
            "cross_twiddle": "table" if index % 3 else "recurrence",
        }
        descriptor = {"groups": [{"registers": 16 + index % 97,
                                   "shared_bytes": 256 * (index % 17)}]}
        rows.append((candidate_id(point), descriptor, (index + 1) / 7))
    assert len({key for key, _, _ in rows}) == len(rows), "candidate identities must be unique"
    random.Random(77).shuffle(rows)
    return rows


def _model(epoch):
    return {"profile": {"rate": epoch + 1},
            "stage_service": {"curves": {"a": [1, 2]}}, "parameters": {}}


def _benchmark(cache_type, rows, epochs):
    context = {"build": "bounded-io-comparison", "feature_version": "v1"}
    with tempfile.TemporaryDirectory(prefix="score-cache-io-", dir=ROOT) as directory:
        cache = cache_type(directory, context)
        initial_bytes = _write_bytes()
        start = time.perf_counter()
        rounds = []
        for epoch in range(epochs):
            round_bytes = _write_bytes()
            round_start = time.perf_counter()
            cache.bind(_model(epoch))
            for key, descriptor, score in rows:
                assert cache.get(key, descriptor) == (False, None)
                cache.put(key, descriptor, score)
            cache.flush()
            rounds.append({"seconds": time.perf_counter() - round_start,
                           "write_bytes": _write_bytes() - round_bytes})
        result = {"seconds": time.perf_counter() - start,
                  "write_bytes": _write_bytes() - initial_bytes,
                  "stats": dict(cache.stats), "rounds": rounds}
        cache.close()
        result["write_bytes_including_close"] = _write_bytes() - initial_bytes
        result["database_bytes"] = cache.path.stat().st_size
        with DiskScoreCache(directory, context) as reader:
            for epoch in range(epochs):
                reader.bind(_model(epoch))
                for key, descriptor, expected in rows:
                    found, score = reader.get(key, descriptor)
                    assert found and score.hex() == expected.hex()
        result["bit_exact_scores_checked"] = len(rows) * epochs
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=int, default=16384)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.candidates <= 65536 or not 1 <= args.epochs <= 4:
        parser.error("bounded comparison requires 1..65536 candidates and 1..4 epochs")
    rows = _rows(args.candidates)
    for name, cache_type in (("baseline", OriginalBatchScoreCache), ("sorted", DiskScoreCache)):
        print(json.dumps({"variant": name, "candidates_per_epoch": len(rows),
                          "epochs": args.epochs, **_benchmark(cache_type, rows, args.epochs)},
                         sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
