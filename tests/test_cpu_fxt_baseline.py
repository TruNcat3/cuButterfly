"""Exact analytic cases for the independent FXT adapter (no tolerance passes)."""
import importlib.util
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("cpu_fxt_baseline", ROOT / "scripts/cpu_fxt_baseline.py")
fxt = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fxt)


@pytest.fixture(scope="module")
def library(tmpdir_factory):
    source = Path(os.environ.get("FXT_ROOT", ROOT / "results/install_compare_20261013/external/fxt"))
    if not (source / "src/walsh/arithtransform.h").exists() or not shutil.which("g++"):
        pytest.skip("FXT source and g++ are required for the external adapter tests")
    return fxt.build_library(source, Path(str(tmpdir_factory.mktemp("fxt"))) / "libfxt.so")


def semantics(op, dtype, shape, batch, inverse=False, normalize=False):
    storage, compute = {np.float32: (2, 1), np.float64: (3, 2),
                        np.uint32: (8, 3), np.uint64: (9, 4)}[dtype]
    strides, stride = [0] * len(shape), 1
    for axis in reversed(range(len(shape))):
        strides[axis] = stride
        stride *= shape[axis]
    return dict(operator=op, storage=storage, compute=compute, shape=list(shape), batch=batch,
                inverse=inverse, normalize_inverse=normalize, placement=0, length_mode=0,
                input_orders=[0] * len(shape), output_orders=[0] * len(shape),
                input_strides=strides, output_strides=strides,
                input_batch_stride=stride, output_batch_stride=stride)


def analytic(source, op, inverse, normalize):
    """Direct subset enumeration / Walsh parity, independent of butterfly order."""
    rows = source.reshape(source.shape[0], -1)
    result = np.empty_like(rows)
    n = rows.shape[1]
    unsigned = source.dtype.kind == "u"
    mask = (1 << (8 * source.dtype.itemsize)) - 1
    for b, row in enumerate(rows):
        for i in range(n):
            total = 0
            for j in range(n):
                if op == 2:
                    sign = -1 if bin(i & j).count("1") % 2 else 1
                elif (op == 3 and j & i == j) or (op == 4 and j & i == i):
                    sign = -1 if inverse and bin(i ^ j).count("1") % 2 else 1
                else:
                    continue
                total += sign * (int(row[j]) if unsigned else float(row[j]))
            if unsigned:
                total &= mask
            elif op == 2 and inverse and normalize:
                total /= n
            result[b, i] = total
    return result.reshape(source.shape)


@pytest.mark.parametrize("dtype", [np.uint32, np.uint64])
@pytest.mark.parametrize("op", [3, 4])
@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("shape,batch", [((1,), 1), ((8,), 3), ((2, 4), 2), ((2, 2, 2, 2), 2)])
def test_unsigned_overflow_and_tensor_rank(library, dtype, op, inverse, shape, batch):
    n = int(np.prod(shape))
    maximum = np.iinfo(dtype).max
    source = np.array([maximum - (i % 7) if i % 2 else 3 + i * 3 for i in range(n * batch)],
                      dtype=dtype).reshape(batch, *shape)
    original = source.copy()
    core = fxt.FXTCore(semantics(op, dtype, shape, batch, inverse, True), source, library)
    np.testing.assert_array_equal(core.execute(), analytic(source, op, inverse, True))
    np.testing.assert_array_equal(source, original)
    np.testing.assert_array_equal(core.execute(), analytic(source, op, inverse, True))
    recovered = fxt.FXTCore(semantics(op, dtype, shape, batch, not inverse), core.output, library)
    np.testing.assert_array_equal(recovered.execute(), source)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("inverse,normalize", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("shape,batch", [((1,), 1), ((8,), 3), ((2, 4), 2), ((2, 2, 2, 2), 2)])
def test_fwht_analytic_exact(library, dtype, inverse, normalize, shape, batch):
    n = int(np.prod(shape))
    source = (((np.arange(n * batch) * 7) % 19 - 9) / 4).astype(dtype).reshape(batch, *shape)
    original = source.copy()
    core = fxt.FXTCore(semantics(2, dtype, shape, batch, inverse, normalize), source, library)
    np.testing.assert_array_equal(core.execute(), analytic(source, 2, inverse, normalize))
    np.testing.assert_array_equal(source, original)
    assert core.output.dtype == source.dtype


def test_direction_not_interchanged(library):
    source = np.array([[1, 2, 4, 8]], dtype=np.uint32)
    subset = fxt.FXTCore(semantics(3, np.uint32, (4,), 1), source, library)
    superset = fxt.FXTCore(semantics(4, np.uint32, (4,), 1), source, library)
    np.testing.assert_array_equal(subset.execute(), [[1, 3, 5, 15]])
    np.testing.assert_array_equal(superset.execute(), [[15, 10, 12, 8]])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_fwht_recursive_core_exact_impulses(library, dtype):
    n = 16384  # FXT loc_dif2 switches to its recursive core above log_n=13.
    source = np.zeros((2, n), dtype=dtype)
    source[0, 7], source[1, 129] = 3, -2
    core = fxt.FXTCore(semantics(2, dtype, (16, 32, 32), 2, True, True), source, library)
    expected = np.array([[amplitude * (-1 if bin(i & index).count("1") % 2 else 1) / n
                          for i in range(n)] for index, amplitude in [(7, 3), (129, -2)]], dtype=dtype)
    np.testing.assert_array_equal(core.execute(), expected)


@pytest.mark.parametrize("change", [dict(compute=2), dict(storage=3), dict(placement=1),
                                     dict(input_orders=[1]), dict(output_strides=[2]),
                                     dict(input_batch_stride=9), dict(length_mode=1), dict(shape=[3])])
def test_rejects_semantic_mismatch(library, change):
    s = semantics(2, np.float32, (8,), 1)
    s.update(change)
    with pytest.raises(ValueError):
        fxt.FXTCore(s, np.arange(8, dtype=np.float32).reshape(1, 8), library)


def test_no_cast_no_python_batch_loop_and_manifest(library):
    s = semantics(2, np.float32, (8,), 3, True, True)
    source = np.arange(24, dtype=np.float32).reshape(3, 8)
    with pytest.raises(ValueError, match="dtype"):
        fxt.FXTCore(s, source.astype(np.float64), library)
    with pytest.raises(ValueError, match="threads"):
        fxt.FXTCore(s, source, library, threads=8)
    with pytest.raises(ValueError, match="contiguous"):
        fxt.FXTCore(s, source[:, ::-1], library)
    core = fxt.FXTCore(s, source, library)
    original = core.fn
    calls = []
    def counted(*args):
        calls.append(args)
        return original(*args)
    core.fn = counted
    core.execute()
    assert len(calls) == 1 and calls[0][3] == 3
    metadata = core.metadata()
    assert "copy" in metadata["timing_scope"] and "scaling" in metadata["timing_scope"]
    manifest = json.loads(Path(str(library) + ".manifest.json").read_text())
    assert manifest["library_sha256"] == fxt._hash(library)
    assert manifest["compiler_sha256"] and manifest["source_file_sha256"]["src/walsh/arithtransform.h"]
    assert manifest["declared_source_url"] == fxt.SOURCE_URL
    core.close()
    with pytest.raises(RuntimeError, match="closed"):
        core.execute()
