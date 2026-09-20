"""Optional GPU regression for compiled-unit versus scalar range validation."""
import csv
import io
import os
import subprocess

import pytest


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("placement", ["in-place", "out-of-place"])
def test_compiled_small_prefix_executes_with_large_suffix(inverse, placement):
    binary = os.environ.get("CUBUTTERFLY_TEST_BINARY")
    if not binary:
        pytest.skip("set CUBUTTERFLY_TEST_BINARY on an idle CUDA device")
    result = subprocess.run([binary, "--list-processing-units"], capture_output=True, text=True, check=True)
    units = {(r["precision"], r["logN"], r["threads"], r["ept"]) for r in csv.DictReader(io.StringIO(result.stdout))}
    if not {("fp32", "4", "256", "8"), ("fp32", "12", "256", "16")} <= units:
        pytest.skip("build does not contain the regression's two local units")
    command = [binary, "--operator", "fft", "--precision", "fp32", "--backend", "online-reorder",
               "--fft-core", "cufftdx-block", "--logN", "16", "--local-stages", "4", "--batch", "2",
               "--prefix-threads", "256", "--suffix-threads", "256", "--prefix-ept", "8", "--suffix-ept", "16",
               "--placement", placement, "--cross-twiddle", "recurrence", "--verify", "--csv", "--warmup", "1", "--repeat", "2"]
    if inverse:
        command.append("--inverse")
    actual = subprocess.run(command, capture_output=True, text=True)
    assert actual.returncode == 0, actual.stderr
    row = next(csv.DictReader(io.StringIO(actual.stdout)))
    assert row["correct"] == "1"
    assert row["fft_core"] == "cufftdx-block" and row["stages_per_decomposition"] == "4x12"
