import hashlib
import json
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from generate_design_points import (load_points, register_legacy_threads,
                                    register_thread_options, write_fwht, write_structured)


@pytest.mark.parametrize("writer,log_ns,requested,expected_sha256", [
    (write_fwht, set(), None,
     "658295bf50a168f74c52509def2d3b0fd0ff73e302151cecdf0d8a42c32312e1"),
    (write_fwht, {3, 12, 15}, None,
     "b3facb3ac38a9422995eb284989fa4c92e8b4159a6eed77c2eac2eaa4efcf927"),
    (write_fwht, {3, 12, 15}, {32, 64},
     "16d9f776e4f6e7c4d1fcb5818321bb96678530e99d134927fc917daf188db2e1"),
    (write_structured, set(), None,
     "1c3bae1d2dafa8521d5ae745d7918c6852745714062f6d93406b9facfb05a608"),
    (write_structured, {3, 12, 15}, None,
     "7dfa0cf5184d85f492ba5b120077c465f866ac10b83f1b584a26ecd51407c502"),
    (write_structured, {3, 12, 15}, {32, 64},
     "89d4c464a70e10172b2094c4db1f97aad721cb0ef479c7cdf627856ab2e389ee"),
])
def test_register_generation_preserves_bytes_and_unchanged_mtime(
        tmp_path, writer, log_ns, requested, expected_sha256):
    path = tmp_path / "generated.cu"
    writer(path, log_ns, requested)
    # Digests capture the output before removing intermediate legacy writes.
    original = path.read_bytes()
    assert hashlib.sha256(original).hexdigest() == expected_sha256
    timestamp = 1_600_000_000_123_456_789
    os.utime(path, ns=(timestamp, timestamp))
    original_mtime = path.stat().st_mtime_ns

    writer(path, log_ns, requested)

    assert path.read_bytes() == original
    assert path.stat().st_mtime_ns == original_mtime


def test_register_inventory_matches_actual_template_dispatch(tmp_path):
    for family, writer in (("fwht", write_fwht), ("structured", write_structured)):
        path = tmp_path / (family + ".cu")
        writer(path, set(range(3, 16)))
        source = path.read_text()
        for log_n in range(3, 16):
            options = register_thread_options(log_n)
            assert options[0] == register_legacy_threads(log_n)
            assert len(options) == len(set(options))
            for threads in options:
                assert (1 << log_n) % (threads * 4) == 0
                assert f"launch_register_{family}<{log_n}, {threads}>" in source
                assert f"launch_register_{family}_zero_extended<{log_n}, {threads}>" in source
            assert f"case {log_n}: return {{{', '.join(map(str, options))}}};" in source
        assert "if (threads == 0)" in source
        assert "switch ((log_n << 16) | threads)" in source
    assert register_thread_options(3) == [1]
    assert register_thread_options(15) == [256, 64, 128]


def test_manifest_selection_and_fixed_items(tmp_path):
    entry = dict(operator="fwht", core="warp-register", precision="fp32",
                 log_n=[8], tile_threads=[32, 64], items=4)
    path = tmp_path / "spec.json"
    for invalid in ({"items": 8}, {"tile_threads": [48]}, {"tile_threads": [32.0]}):
        path.write_text(json.dumps({"design_points": [dict(entry, **invalid)]}))
        with pytest.raises(ValueError):
            load_points(path, set(), set(), set())
    path.write_text(json.dumps({"design_points": [entry]}))
    points, _ = load_points(path, set(), set(), set())
    assert points[("fwht", "warp-register", "fp32")] == {8}
    assert register_thread_options(8, [32]) == [32]
