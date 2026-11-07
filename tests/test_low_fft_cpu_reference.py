import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from low_fft_cpu_reference import rounded, butterfly, transform

@pytest.mark.parametrize('storage,step', [(4, 2**-10), (5, 2**-7)])
def test_round_nearest_even_ties(storage, step):
    assert rounded(1 + step/2, storage) == 1
    assert rounded(1 + 3*step/2, storage) == 1 + 2*step

@pytest.mark.parametrize('storage', [4, 5])
def test_product_rounds_before_complex_accumulation(storage):
    root = rounded(np.array([.7, .7]), storage)
    x = rounded(np.array([1.1, .9]), storage)
    lo, hi = butterfly(np.zeros(2), x, root, storage)
    expected = rounded(rounded(x[0]*root[0], storage) - rounded(x[1]*root[1], storage), storage)
    assert lo[0] == expected and hi[0] == -expected

@pytest.mark.parametrize('storage', [4, 5])
def test_impulse_and_inverse_normalization(storage):
    s = dict(shape=[8], batch=1, storage=storage, compute=0, inverse=0, normalize_inverse=True)
    x = np.zeros(8, np.complex64); x[0] = 1
    np.testing.assert_array_equal(transform(x,s), np.ones((1,8)))
    s['inverse'] = 1
    np.testing.assert_array_equal(transform(x,s), np.full((1,8),.125))

def test_no_unproven_arithmetic_or_axis_substitution():
    s = dict(shape=[2,2], batch=1, storage=4, compute=0)
    with pytest.raises(ValueError, match='axis_order'):
        transform(np.zeros(4),s)
    s['compute'] = 2
    with pytest.raises(ValueError, match='compute'):
        transform(np.zeros(4),s)

def test_fp16_normalized_large_pair_scales_before_overflow():
    lo, hi = butterfly(np.array([65504.,0]), np.array([65504.,0]), np.array([1.,0]),4,True)
    assert lo[0] == 65504 and hi[0] == 0

def test_review_mapping_uses_fixed_rank2_row_then_column_order():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from benchmark.review_fft_l_cpu_reference import axis_order, metrics
    assert axis_order({'physical': {'algorithm': 'rank2-transpose[temporal-radix4,temporal-radix4]+direct-output', 'nd_composite':False}},2) == [1,0]
    assert axis_order({},2) is None
    result = metrics(np.array([1+1j, np.inf]), np.array([1+0j, 1]))
    assert result['actual_nonfinite'] == 1
    assert result['max_abs_finite'] == 1
    assert result['acceptance_threshold'] is None


def test_error_metrics_import_does_not_select_a_python_frontend():
    import subprocess
    root = Path(__file__).resolve().parents[1]
    # An installed-package validator must be able to reuse CPU error metrics
    # without the historical campaign loader selecting the source frontend.
    subprocess.run([sys.executable, '-c', """
import sys
from pathlib import Path
root = Path.cwd()
sys.path.insert(0, str(root / 'scripts'))
from benchmark.review_fft_l_cpu_reference import metrics
assert 'cubutterfly' not in sys.modules
assert str(root / 'python') not in sys.path
"""], cwd=root, check=True)

@pytest.mark.parametrize('storage', [4, 5])
def test_fp32_compute_retains_products_until_complex_sum(storage):
    root = rounded(np.array([.7, .7]), storage)
    x = rounded(np.array([1.1, .9]), storage)
    lo, hi = butterfly(np.zeros(2), x, root, storage, native=False)
    expected = rounded(np.float32(x[0]*root[0]) - np.float32(x[1]*root[1]), storage)
    assert lo[0] == expected and hi[0] == -expected
    s = dict(shape=[8], batch=1, storage=storage, compute=1, inverse=0, normalize_inverse=True)
    impulse = np.zeros(8, np.complex64); impulse[0] = 1
    np.testing.assert_array_equal(transform(impulse, s), np.ones((1,8)))
