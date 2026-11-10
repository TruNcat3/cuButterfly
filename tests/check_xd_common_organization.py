#!/usr/bin/env python3
"""Public six-operator resident organization checks; floating error reporting only."""
import argparse
import copy
import ctypes as ct
import json
import pathlib
import sys

import numpy as np
import scipy.fft
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))
import check_xd as api


def reference(source, op, shape, inverse, matrices, modulus):
    if op == api.OP_FFT:
        fn = scipy.fft.ifftn if inverse else scipy.fft.fftn
        return fn(source, axes=tuple(range(1, len(shape) + 1)))
    if op == api.OP_NTT:
        return np.array(api._ntt_reference(source.ravel(), shape, source.shape[0], modulus, inverse),
                        dtype=source.dtype).reshape(source.shape)
    out = source.copy()
    for axis, length in enumerate(shape, 1):
        moved = np.moveaxis(out, axis, -1).copy()
        rows = moved.reshape(-1, length)
        stages = list(range(length.bit_length() - 1))
        if inverse:
            stages.reverse()
        for stage in stages:
            half = 1 << stage
            for start in range(0, length, 2 * half):
                a = rows[:, start:start + half].copy()
                b = rows[:, start + half:start + 2 * half].copy()
                if op == api.OP_FWHT:
                    left, right = a + b, a - b
                elif op == api.OP_SUBSET_ZETA:
                    left, right = a, b - a if inverse else b + a
                elif op == api.OP_SUPERSET_ZETA:
                    left, right = a - b if inverse else a + b, b
                else:
                    m00, m01, m10, m11 = np.asarray(matrices[axis - 1][stage], dtype=source.dtype)
                    if inverse:
                        det = m00 * m11 - m01 * m10
                        m00, m01, m10, m11 = m11 / det, -m01 / det, -m10 / det, m00 / det
                    left, right = m00 * a + m01 * b, m10 * a + m11 * b
                rows[:, start:start + half], rows[:, start + half:start + 2 * half] = left, right
        out = np.moveaxis(rows.reshape(moved.shape), -1, axis)
    if op == api.OP_FWHT and inverse:
        out *= np.array(1 / np.prod(shape), dtype=out.dtype)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    interface = api.Api(args.library.resolve())
    handle = api.Handle(interface, torch.cuda.current_stream().cuda_stream)
    rng = np.random.default_rng(321)
    records = []
    try:
        for shape in [(4, 8), (2, 4, 2, 4), (2,) * 8]:
            for op in range(6):
                for inverse in (False, True):
                    batch = 3
                    floating = op not in (api.OP_NTT, api.OP_SUBSET_ZETA, api.OP_SUPERSET_ZETA)
                    dtype = np.complex64 if op == api.OP_FFT else np.float32 if floating else np.uint32
                    source = (rng.normal(size=(batch, *shape)).astype(dtype) if floating else
                              rng.integers(0, 97, size=(batch, *shape), dtype=dtype))
                    if op == api.OP_FFT:
                        source += 1j * rng.normal(size=source.shape).astype(np.float32)
                    matrices = [[(1.0, (axis + 1) / 64, -(stage + 1) / 32, 1.0)
                                 for stage in range(n.bit_length() - 1)] for axis, n in enumerate(shape)]
                    expected = reference(source, op, shape, inverse, matrices, 97)
                    spec = dict(op=op, storage=api.DATA_COMPLEX_FP32 if op == api.OP_FFT else
                                api.DATA_FP32 if floating else api.DATA_UINT32,
                                compute=api.COMPUTE_FP32 if floating else api.COMPUTE_UINT32,
                                shape=shape, batch=batch, mode=api.EXEC_XD, direction=int(inverse))
                    if op == api.OP_NTT:
                        spec.update(modulus=97, word_bits=32)
                    if op == api.OP_STRUCTURED_2X2:
                        spec["matrices"] = matrices
                    inp = torch.from_numpy(source if floating else source.view(np.int32)).cuda()
                    original = inp.clone()
                    output = torch.empty_like(inp)
                    for threads in (32, 96, 512):
                        plan = api.make_plan(interface, handle, torch, **spec, mapping=f"xd:resident-t{threads}")
                        mapping = json.loads(plan.metadata["mapping_json"])
                        plan.close()
                        # Nonlegacy order, with strict public mapping replay.
                        order = list(range(len(shape)))
                        order = order[1:] + order[:1]
                        mapping["physical"]["axis_order"] = order
                        if op not in (api.OP_FFT, api.OP_NTT):
                            legacy = list(reversed(range(len(shape)))) if inverse else list(range(len(shape)))
                            mapping["physical"]["order"] = (
                                "reverse-axis-reverse-stage-inverse" if inverse else "natural-axis-natural-stage"
                            ) if order == legacy else (
                                "explicit-axis-reverse-stage-inverse" if inverse else "explicit-axis-natural-stage")
                        plan = api.make_plan(interface, handle, torch, **spec, mapping=json.dumps(mapping))
                        try:
                            assert json.loads(plan.metadata["mapping_json"]) == mapping
                            plan.execute(inp, output)
                            torch.cuda.synchronize()
                            actual = output.cpu().numpy()
                            if not floating:
                                actual = actual.view(np.uint32)
                                assert np.array_equal(actual, expected), (op, shape, inverse, threads)
                                error = {"exact": True}
                            else:
                                assert np.isfinite(actual).all()
                                diff = np.abs(actual.astype(np.complex128) - expected.astype(np.complex128))
                                error = {"max_abs": float(diff.max()),
                                         "relative_l2": float(np.linalg.norm(diff.ravel()) /
                                             max(np.linalg.norm(expected.ravel()), np.finfo(float).tiny)),
                                         "abs_quantiles": np.quantile(diff, [0, .5, .9, .99, 1]).tolist(),
                                         "accuracy_acceptance": "pending-user-threshold"}
                            assert torch.equal(inp, original)
                            # Graph output must be byte-identical to eager for a changed input and restoration.
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):
                                interface.check("set capture stream", interface.cubutterflySetStream(
                                    handle.value, ct.c_void_p(torch.cuda.current_stream().cuda_stream)))
                                plan.execute(inp, output)
                            interface.check("restore stream", interface.cubutterflySetStream(
                                handle.value, ct.c_void_p(torch.cuda.current_stream().cuda_stream)))
                            for value in (torch.zeros_like(inp), original):
                                inp.copy_(value)
                                plan.execute(inp, output)
                                eager = output.clone()
                                graph.replay()
                                torch.cuda.synchronize()
                                assert torch.equal(output, eager)
                            record = dict(op=op, shape=shape, inverse=inverse, threads=threads,
                                          axis_order=order, mapping=mapping, graph_replay=True, **error)
                            records.append(record)
                            print(f"op={op} rank={len(shape)} inverse={inverse} threads={threads}: contract passed", flush=True)
                        finally:
                            plan.close()
                    # A duplicate axis is never normalized into a different request.
                    bad = copy.deepcopy(mapping)
                    bad["physical"]["axis_order"] = [0] * len(shape)
                    try:
                        invalid = api.make_plan(interface, handle, torch, **spec, mapping=json.dumps(bad))
                    except (api.CApiError, RuntimeError):
                        pass
                    else:
                        invalid.close()
                        raise AssertionError("duplicate axis accepted")
    finally:
        args.output.write_text(json.dumps(records, indent=2) + "\n")
        handle.close()
    print(f"xD common organization: {len(records)} contracts passed; floating errors reported without threshold")


if __name__ == "__main__":
    main()
