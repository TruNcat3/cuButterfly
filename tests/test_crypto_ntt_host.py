"""Host-only contract tests for the isolated cryptographic NTT benchmark.

These tests intentionally pass ``--self-test-host`` and hide all CUDA devices.
They validate semantic admission/rejection and the independent CPU/reference
oracles; they are not GPU correctness or performance evidence.
"""

import json
import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
BINARY = Path(os.environ.get(
    "CRYPTO_NTT_HOST_BINARY",
    str(ROOT / "results/crypto_application_20260920/build/crypto_ntt_bench"),
))

P23 = 8380417
P31_BABYBEAR = 2013265921
P31_KOALABEAR = 2130706433
P40 = 550829555713
P62 = 2391411402133733377
GOLDILOCKS = 18446744069414584321


def _base_command(*, log_n, batch, word_bits, moduli, mode, direction,
                  pattern, seed, coset_generator=7):
    return [
        str(BINARY),
        "--log-n", str(log_n),
        "--batch", str(batch),
        "--word-bits", str(word_bits),
        "--moduli", ",".join(str(modulus) for modulus in moduli),
        "--mode", mode,
        "--direction", direction,
        "--coset-generator", str(coset_generator),
        "--warmup", "3",
        "--repeat", "5",
        "--seed", str(seed),
        "--input-pattern", pattern,
        "--json",
        "--self-test-host",
    ]


def _run(command):
    if not BINARY.is_file():
        pytest.skip("crypto_ntt_bench is not built")
    env = os.environ.copy()
    # The host path does not call device_info(), and this guard prevents an
    # accidental CUDA context if the benchmark changes its initialization.
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["CUBUTTERFLY_COMPILE_MODE"] = "research"
    result = subprocess.run(command, env=env, cwd=str(ROOT), text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            timeout=60)
    return result


def _host_json(result):
    assert result.returncode == 0, result.stderr
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise AssertionError("host benchmark did not emit JSON: " + result.stdout) from error
    assert payload["correct"] is True
    # A host oracle result must never be presented as a GPU measurement.
    assert payload["gpu_uuid"] == "host"
    assert payload["device_name"] == "host"
    assert payload["device_memory_bytes"] == 0
    assert payload["compile_mode"] == "research"
    assert payload["plan_ms"] == 0.0
    assert payload["kernel_ms"] == 0.0
    return payload


@pytest.mark.parametrize(
    "case",
    [
        # p23/word32, cyclic, forward, modular-boundary values.
        dict(name="p23-cyclic-boundary", log_n=4, batch=1, word_bits=32,
             moduli=(P23,), mode="cyclic", direction="forward",
             pattern="boundary"),
        # p31/word32, negacyclic, inverse, random values.
        dict(name="p31-negacyclic-random", log_n=8, batch=1, word_bits=32,
             moduli=(P31_BABYBEAR,), mode="negacyclic", direction="inverse",
             pattern="random"),
        # p31/word32, coset with g=1: the subgroup coset is a valid contract.
        dict(name="p31-coset-one", log_n=4, batch=1, word_bits=32,
             moduli=(P31_KOALABEAR,), mode="coset", direction="forward",
             pattern="boundary", coset_generator=1),
        # p40/word64, cyclic, inverse.
        dict(name="p40-cyclic-inverse", log_n=4, batch=1, word_bits=64,
             moduli=(P40,), mode="cyclic", direction="inverse",
             pattern="random"),
        # p62/word64, negacyclic, forward.
        dict(name="p62-negacyclic-forward", log_n=4, batch=1, word_bits=64,
             moduli=(P62,), mode="negacyclic", direction="forward",
             pattern="boundary"),
        # One larger O(N log N) host oracle case, still small enough for CI.
        dict(name="p40-cyclic-log12", log_n=12, batch=1, word_bits=64,
             moduli=(P40,), mode="cyclic", direction="forward",
             pattern="random"),
        # Four independent residues and a non-power-of-two polynomial batch.
        dict(name="rns-four-channel-batch3", log_n=4, batch=3, word_bits=32,
             moduli=(P23, 1073479681, P31_BABYBEAR, P31_KOALABEAR),
             mode="negacyclic", direction="inverse", pattern="random"),
    ],
    ids=lambda case: case["name"],
)
def test_host_contracts(case):
    command = _base_command(seed=20260920, **{
        key: value for key, value in case.items() if key != "name"
    })
    payload = _host_json(_run(command))
    assert payload["logN"] == case["log_n"]
    assert payload["batch"] == case["batch"]
    assert payload["word_bits"] == case["word_bits"]
    assert payload["mode"] == case["mode"]
    assert payload["direction"] == case["direction"]
    assert payload["input_pattern"] == case["pattern"]
    assert payload["moduli"] == list(case["moduli"])
    assert payload["verified_batches"] == case["batch"]
    assert payload["verified_channels"] == len(case["moduli"])
    assert payload["seed"] == 20260920
    if case["mode"] == "coset":
        assert payload["coset_generator"] == case.get("coset_generator", 7)


@pytest.mark.parametrize(
    "case",
    [
        dict(name="composite", log_n=4, batch=1, word_bits=32,
             moduli=(15,), mode="cyclic", direction="forward",
             pattern="boundary"),
        dict(name="word32-large-prime", log_n=4, batch=1, word_bits=32,
             moduli=(P40,), mode="cyclic", direction="forward",
             pattern="random"),
        dict(name="goldilocks-runtime-bound", log_n=4, batch=1, word_bits=64,
             moduli=(GOLDILOCKS,), mode="cyclic", direction="forward",
             pattern="random"),
        # Kyber q supports cyclic N=256 but not the 2N root required here;
        # its protocol NTT is incomplete and must not be silently relabeled.
        dict(name="kyber-incomplete-negacyclic", log_n=8, batch=1,
             word_bits=32, moduli=(3329,), mode="negacyclic",
             direction="forward", pattern="boundary"),
        dict(name="invalid-log-n", log_n=0, batch=1, word_bits=32,
             moduli=(P31_BABYBEAR,), mode="cyclic", direction="forward",
             pattern="random"),
        dict(name="zero-coset", log_n=4, batch=1, word_bits=32,
             moduli=(P31_BABYBEAR,), mode="coset", direction="forward",
             pattern="boundary", coset_generator=0),
    ],
    ids=lambda case: case["name"],
)
def test_host_rejects_invalid_contracts(case):
    command = _base_command(seed=20260920, **{
        key: value for key, value in case.items() if key != "name"
    })
    result = _run(command)
    assert result.returncode != 0
    assert result.stdout.strip(), result.stderr
    payload = json.loads(result.stdout)
    assert payload["correct"] is False
    assert "gpu_uuid" not in payload
