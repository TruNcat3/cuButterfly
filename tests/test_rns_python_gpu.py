"""Opt-in real ABI/Graph acceptance, including fresh-plan cache replay."""
import os

import pytest


pytestmark = pytest.mark.skipif(
    not os.environ.get("CUBUTTERFLY_TEST_LIBRARY"),
    reason="set CUBUTTERFLY_TEST_LIBRARY and select an idle CUDA device explicitly",
)


def reference(values, config):
    n = 1 << config["log_n"]
    bits = config["log_n"]
    reverse = [int(f"{j:0{bits}b}"[::-1], 2) for j in range(n)]
    result = list(values)
    for polynomial in config["polynomials"]:
        d = config["domains"][polynomial["domain"]]
        q, psi = d["modulus"], d["root"]
        offset = polynomial["input_offset"]
        a = [values[offset + (reverse[j] if config["inverse"] else j)] for j in range(n)]
        if not config["inverse"]:
            a = [x * pow(psi, j, q) % q for j, x in enumerate(a)]
        a = [a[reverse[j]] for j in range(n)]
        omega = pow(psi, 2, q)
        if config["inverse"]:
            omega = pow(omega, -1, q)
        length = 2
        while length <= n:
            step = pow(omega, n // length, q)
            for base in range(0, n, length):
                w = 1
                for j in range(length // 2):
                    x, y = a[base + j], a[base + j + length // 2] * w % q
                    a[base + j] = (x + y) % q
                    a[base + j + length // 2] = (x - y) % q
                    w = w * step % q
            length *= 2
        if config["inverse"]:
            inv_n, inv_psi = pow(n, -1, q), pow(psi, -1, q)
            a = [x * inv_n * pow(inv_psi, j, q) % q for j, x in enumerate(a)]
        start = polynomial["output_offset"]
        for j, x in enumerate(a):
            result[start + (j if config["inverse"] else reverse[j])] = x
    return result


@pytest.mark.parametrize("inverse", [False, True])
def test_prepared_python_graph_exact(tmp_path, inverse):
    import cupy as cp
    import numpy as np
    from cubutterfly.rns import CachePolicy, Library, Placement, Plan

    n = 4096
    config = dict(schema_version=1, log_n=12, inverse=inverse, ring="negacyclic",
                  input_order="bit_reversed" if inverse else "natural",
                  output_order="natural" if inverse else "bit_reversed",
                  domains=[dict(modulus=q, root=pow(g, (q - 1) // (2 * n), q))
                           for q, g in [(2013265921, 31), (469762049, 3)]],
                  polynomials=[dict(domain=0, input_offset=n, output_offset=n),
                               dict(domain=1, input_offset=0, output_offset=0)])
    stream = cp.cuda.Stream(non_blocking=True)
    device = cp.empty(2 * n, dtype=cp.uint64)
    library = Library(os.environ["CUBUTTERFLY_TEST_LIBRARY"])
    mapping = None
    for policy in (CachePolicy.PREFER, CachePolicy.REQUIRE):
        with Plan(library, config) as plan:
            plan.prepare(placement=Placement.IN_PLACE, cache_directory=tmp_path,
                         policy=policy, repetitions=1, stream=stream.ptr)
            info = plan.info()
            assert info["cache_hit"] == (policy == CachePolicy.REQUIRE)
            if mapping is not None:
                assert mapping == info["mapping"]
            mapping = info["mapping"]
            with stream:
                stream.begin_capture()
                plan.execute(device.data.ptr, device.data.ptr, stream=stream.ptr)
                graph = stream.end_capture()
            for seed in (1, 13, 1):
                values = [0] * (2 * n)
                for p in config["polynomials"]:
                    q = config["domains"][p["domain"]]["modulus"]
                    for j in range(n):
                        values[p["input_offset"] + j] = (j * j * seed + 17 * j + q - 1) % q
                expected = reference(values, config)
                device.set(np.asarray(values, dtype=np.uint64), stream=stream)
                graph.launch(stream)
                stream.synchronize()
                assert device.get().tolist() == expected
            # CUDA graphs borrow the plan and must be released before close.
            del graph
