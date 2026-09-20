"""Public runtime contracts for selectable FFT leaf/layout/I-O policies.

Set CUBUTTERFLY_TEST_BINARY and pin CUDA_VISIBLE_DEVICES to an idle device.
These are correctness checks, never performance samples.
"""
import csv
import io
import json
import os
import subprocess

import pytest


def run(mapping, *, precision="fp32", inverse=False, mode="research", expect_error=False):
    binary = os.environ.get("CUBUTTERFLY_TEST_BINARY")
    if not binary:
        pytest.skip("set CUBUTTERFLY_TEST_BINARY on an idle CUDA device")
    n = sum(mapping["stage_partition"])
    args = [binary, "--operator", "fft", "--precision", precision, "--logN", str(n),
        "--batch", "3", "--element-stride", "2", "--batch-stride", str((1 << n)*2+16),
        "--placement", "in-place" if inverse else "out-of-place", "--normalization", "inverse",
        "--mapping-json", json.dumps(mapping), "--warmup", "0", "--repeat", "1",
        "--verify", "--verify-batches", "0", "--csv"]
    if inverse:
        args.append("--inverse")
    env = dict(os.environ, CUBUTTERFLY_COMPILE_MODE=mode)
    completed = subprocess.run(args, capture_output=True, text=True, env=env)
    if expect_error:
        assert completed.returncode != 0, "unsupported strategy silently executed"
        return completed.stderr
    assert completed.returncode == 0, completed.stderr
    row = next(csv.DictReader(io.StringIO(completed.stdout)))
    assert row["correct"] == "1" and int(row["verified_batches"]) == 3
    return json.loads(row["mapping_json"]), json.loads(row["execution_groups_json"])


def prefix(core="native", layout="linear"):
    return dict(backend="online-reorder", fft_core="register-tile", local_stages=6,
        stage_partition=[6,6], prefix_threads=32, prefix_ept=8, suffix_threads=32,
        suffix_ept=8, shared_layout="writer-aligned", cross_twiddle="recurrence",
        reorder_columns=1, prefix_codelet=core, prefix_shared_layout=layout)


@pytest.mark.parametrize("core,layout", [("native","linear"), ("native","xor"),
                                         ("cufftdx-thread","linear"), ("cufftdx-thread","xor")])
@pytest.mark.parametrize("inverse", [False, True])
def test_prefix_strategy_executes_without_changing_semantics(core, layout, inverse):
    mapping = prefix(core, layout)
    # Exercise both bulk and the existing batch pipeline with the same leaf.
    if inverse:
        mapping.update(stage_overlap=True, batch_tile_count=2)
    resolved, groups = run(mapping, inverse=inverse)
    assert resolved["prefix_codelet"] == core
    assert resolved["prefix_shared_layout"] == layout
    assert len(groups) == 2
    assert groups[0]["codelet"] == core


def test_fp64_native_xor_preserves_inverse_stride_contract():
    resolved, _ = run(prefix(layout="xor"), precision="fp64", inverse=True)
    assert resolved["prefix_codelet"] == "native"


@pytest.mark.parametrize("precision,core,layout,lanes,inverse", [
    ("fp32", "native", "linear", 2, False),
    ("fp32", "native", "xor", 4, True),
    ("fp32", "cufftdx-thread", "xor", 2, False),
    ("fp32", "cufftdx-thread", "xor", 4, True),
    ("fp32", "cufftdx-thread", "xor", 8, False),
    ("fp64", "native", "xor", 2, True),
    ("fp64", "native", "xor", 4, True),
    ("fp64", "native", "linear", 8, False),
])
def test_cooperative_prefix_preserves_semantics_and_physical_shape(precision, core, layout, lanes, inverse):
    mapping = dict(prefix(core, layout), prefix_codelet_lanes=lanes,
                   prefix_threads=32*lanes, prefix_ept=8//lanes)
    if inverse:
        mapping.update(stage_overlap=True, batch_tile_count=2)
    resolved, groups = run(mapping, precision=precision, inverse=inverse)
    assert resolved["prefix_codelet_lanes"] == lanes
    assert groups[0]["threads"] == 32*lanes
    assert groups[0]["elements_per_thread"] == 8//lanes
    assert resolved["prefix_units_per_cta"] == 4
    # G changes threads/EPT, not the independent columns or CTA grid. Runtime
    # resource queries can replace shared-byte metadata, so also assert grid
    # and data folding rather than testing only the compiler resource fields.
    launch_batch = 2 if inverse else 3
    assert groups[0]["grid_ctas"] == launch_batch*64//4
    assert groups[0]["data_time"] == max(1, 4//lanes)
    assert groups[0]["live_shared_bytes"] == 64*4*(16 if precision == "fp64" else 8)


@pytest.mark.parametrize("changes,mode", [
    (dict(prefix_codelet_lanes=2, prefix_threads=64, prefix_ept=4), "precompiled"),
    (dict(prefix_codelet_lanes=3), "research"),
    (dict(prefix_codelet_lanes=2), "research"),
    (dict(prefix_codelet_lanes=16, prefix_threads=512, prefix_ept=1), "research"),
])
def test_invalid_or_uncompiled_cooperative_prefix_is_rejected(changes, mode):
    run(dict(prefix(), **changes), mode=mode, expect_error=True)


def test_explicit_segment_codelet_conflict_is_rejected():
    mapping = prefix("cufftdx-thread", "xor")
    mapping["segment_mappings"] = [
        dict(core="register-tile", exchange="shared", threads=32, ept=8, codelet="native"),
        dict(core="cufftdx-block", exchange="shared", threads=32, ept=8, codelet="native"),
    ]
    error = run(mapping, expect_error=True)
    assert "codelets disagree" in error


@pytest.mark.parametrize("precision,core,depth,policies", [
    ("fp32", "cufftdx-block", 0, ["static-unrolled", "dynamic", "static-unrolled"]),
    ("fp32", "cufftdx-block", 1, ["dynamic", "static-unrolled", "dynamic"]),
    ("fp64", "cufftdx-block", 2, ["static-unrolled"]*3),
    ("fp32", "register-tile", 1, ["static-unrolled", "dynamic", "static-unrolled"]),
])
def test_factor_policy_is_per_physical_group(precision, core, depth, policies):
    mapping = dict(backend="factor-streamed", fft_core=core, stage_partition=[18],
        factor_partition=[6,6,6], factor_ept=8, factor_columns=4,
        data_tiles_per_cta=3, prefetch_depth=depth, factor_io_policies=policies,
        shared_layout="writer-aligned", cross_twiddle="recurrence",
        factor_slices=2, factor_overlap=True)
    resolved, groups = run(mapping, precision=precision, inverse=True)
    assert resolved["factor_io_policies"] == policies
    assert [g["io_policy"] for g in groups] == policies
    assert all(g["compiler_local_resources_known"] for g in groups)


@pytest.mark.parametrize("mapping,precision,mode", [
    (prefix("cufftdx-thread", "xor"), "fp64", "research"),
    (prefix("cufftdx-thread", "xor"), "fp32", "precompiled"),
    (prefix("invalid", "xor"), "fp32", "research"),
    ({**prefix(), "prefix_shared_layout": "invalid"}, "fp32", "research"),
    ({**prefix(), "factor_io_policies": ["static-unrolled"]}, "fp32", "research"),
])
def test_unsupported_strategy_does_not_fall_back(mapping, precision, mode):
    run(mapping, precision=precision, mode=mode, expect_error=True)
