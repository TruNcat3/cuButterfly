import copy
import math
import pathlib
import sqlite3
import subprocess
import sys

import pytest

SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from calibration_cpu_cache import DiskScoreCache


MODEL = {"profile": {"rate": 1}, "stage_service": {"curves": {"a": [1, 2]}}, "parameters": {}}
CONTEXT = {"build": "abc", "feature_version": "v1"}


def test_cross_process_and_partial_epoch_reuse(tmp_path):
    script = """
import sys
sys.path.insert(0, sys.argv[1])
from calibration_cpu_cache import DiskScoreCache
with DiskScoreCache(sys.argv[2], {'build': 'abc', 'feature_version': 'v1'}) as cache:
    cache.bind({'profile': {'rate': 1}, 'stage_service': {'curves': {'a': [1, 2]}}, 'parameters': {}})
    cache.put('a', None, 0.10000000000000002)
"""
    subprocess.run([sys.executable, "-c", script, str(SCRIPTS), str(tmp_path)], check=True)
    with DiskScoreCache(tmp_path, CONTEXT) as cache:
        cache.bind(MODEL)
        assert cache.get("a", None) == (True, 0.10000000000000002)
        assert cache.get("unfinished", None) == (False, None)


def test_model_context_and_mutable_descriptor_invalidation(tmp_path):
    descriptor = {"groups": [{"registers": 16}]}
    with DiskScoreCache(tmp_path, CONTEXT) as cache:
        cache.bind(MODEL)
        cache.put("a", descriptor, .25)
        descriptor["groups"][0]["registers"] = 32
        assert cache.get("a", descriptor) == (False, None)
        assert cache.get("a", None) == (False, None)
        changed = copy.deepcopy(MODEL)
        changed["profile"]["rate"] = 2
        cache.bind(changed)
        assert cache.get("a", {"groups": [{"registers": 16}]}) == (False, None)
        cache.bind(MODEL)
        assert cache.get("a", {"groups": [{"registers": 16}]}) == (True, .25)
        changed = copy.deepcopy(MODEL)
        changed["stage_service"]["curves"]["a"].append(3)
        cache.bind(changed)
        assert cache.get("a", {"groups": [{"registers": 16}]}) == (False, None)
    with DiskScoreCache(tmp_path, dict(CONTEXT, build="new")) as cache:
        cache.bind(MODEL)
        assert cache.get("a", {"groups": [{"registers": 16}]}) == (False, None)


@pytest.mark.parametrize("value", [0.0, -0.0, 5e-324, .1, 1.7976931348623157e308,
                                   float("inf"), float("-inf")])
def test_float_bits_and_infinity_survive_reopen(tmp_path, value):
    with DiskScoreCache(tmp_path, CONTEXT) as cache:
        cache.bind(MODEL)
        cache.put("a", None, value)
    with DiskScoreCache(tmp_path, CONTEXT) as cache:
        cache.bind(MODEL)
        found, actual = cache.get("a", None)
        assert found and actual.hex() == value.hex()


def test_nan_and_bad_inputs_never_become_hits(tmp_path):
    with DiskScoreCache(tmp_path, CONTEXT) as cache:
        with pytest.raises(RuntimeError, match="bind"):
            cache.get("a", None)
        cache.bind(MODEL)
        with pytest.raises(ValueError, match="NaN"):
            cache.put("a", None, math.nan)
        assert cache.get("a", None) == (False, None)
        with pytest.raises(ValueError):
            cache.bind({"profile": math.nan})
        with pytest.raises(ValueError):
            cache.put("a", {"resource": math.nan}, 1.0)
        assert cache.get("a", None) == (False, None)
        with pytest.raises(TypeError):
            cache.stats["hits"] = 100


def test_concurrent_reader_and_writers_see_only_committed_partial_results(tmp_path):
    with DiskScoreCache(tmp_path, CONTEXT) as writer, DiskScoreCache(tmp_path, CONTEXT) as reader:
        writer.bind(MODEL)
        reader.bind(MODEL)
        writer.put("a", None, 1.0)
        assert writer.get("a", None) == (True, 1.0)
        assert reader.get("a", None) == (False, None)
        reader.put("b", None, 2.0)
        reader.flush()
        writer.flush()
        assert reader.get("a", None) == (True, 1.0)
        assert writer.get("b", None) == (True, 2.0)


def test_flush_batches_and_elapsed_time(tmp_path, monkeypatch):
    with DiskScoreCache(tmp_path, CONTEXT) as cache:
        cache.bind(MODEL)
        for index in range(256):
            cache.put(str(index), None, float(index))
        assert cache.stats["flushes"] == 1
        assert cache.stats["pending_scores"] == 0
        monkeypatch.setattr("calibration_cpu_cache.time.monotonic", lambda: cache._last_flush + 1.1)
        cache.put("timed", None, .5)
        assert cache.stats["flushes"] == 2
        assert cache.stats["pending_scores"] == 0


def test_corrupt_score_raises_instead_of_hit(tmp_path):
    with DiskScoreCache(tmp_path, CONTEXT) as cache:
        cache.bind(MODEL)
        cache.put("a", None, 1.0)
        cache.flush()
        cache._connection.execute("UPDATE scores SET score_hex='not-a-number'")
        with pytest.raises(ValueError):
            cache.get("a", None)
        assert cache.stats["hits"] == 0


def test_failed_flush_rolls_back_batch_and_keeps_pending_for_retry(tmp_path):
    with DiskScoreCache(tmp_path, CONTEXT) as writer, DiskScoreCache(tmp_path, CONTEXT) as reader:
        writer.bind(MODEL)
        reader.bind(MODEL)
        writer.put("a", None, 1.0)
        writer.put("b", None, 2.0)
        writer._connection.execute("""
            CREATE TRIGGER fail_second BEFORE INSERT ON scores
            WHEN NEW.candidate = 'b' BEGIN SELECT RAISE(ABORT, 'injected failure'); END
        """)
        with pytest.raises(sqlite3.IntegrityError, match="injected failure"):
            writer.flush()
        assert not writer._connection.in_transaction
        assert writer.stats["pending_scores"] == 2
        assert reader.get("a", None) == (False, None)
        writer._connection.execute("DROP TRIGGER fail_second")
        writer.flush()
        assert reader.get("a", None) == (True, 1.0)
        assert reader.get("b", None) == (True, 2.0)
