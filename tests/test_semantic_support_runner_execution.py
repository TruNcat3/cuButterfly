"""Opt-in complete runner executions for newly supported semantic contracts.

Enable CUBUTTERFLY_GPU_TESTS=1 and set CUBUTTERFLY_TEST_LIBRARY. Rows and full
error artifacts remain in pytest's temporary directory, including on failure.
"""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from test_semantic_execution import runtime

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_semantic_campaign as runner


TARGETS = [
    pytest.param(dict(op=op, storage=9, compute=4, extents=[4, 8], batch=2,
                      inverse=inverse), id=f"uint64-{name}-{'inverse' if inverse else 'forward'}")
    for op, name in ((3, 'subset'), (4, 'superset')) for inverse in (False, True)
]
TARGETS += [
    pytest.param(dict(storage=6, compute=2, extents=[4, 8], batch=2), id='fft-fp32-compute-fp64'),
    pytest.param(dict(storage=0, compute=0, fft_type=1, extents=[4, 8], batch=2), id='fp16-r2c-rank2'),
    pytest.param(dict(storage=1, compute=0, fft_type=2, extents=[4, 8], batch=2,
                      inverse=True), id='bf16-c2r-rank2'),
    pytest.param(dict(storage=2, compute=1, fft_type=1, extents=[2, 2, 2, 4], batch=2), id='fp32-r2c-rank4'),
    pytest.param(dict(storage=3, compute=2, fft_type=2, extents=[2, 2, 2, 4], batch=2,
                      inverse=True), id='fp64-c2r-rank4'),
    pytest.param(dict(storage=6, compute=1, extents=[3, 5, 7], batch=2), id='fp32-fft-odd-rank3'),
    pytest.param(dict(storage=7, compute=2, extents=[3, 5, 7], batch=2), id='fp64-fft-odd-rank3'),
    pytest.param(dict(op=1, storage=8, compute=3, extents=[65536], batch=1,
                      modulus=998244353, word_bits=32), id='ntt-word32-n65536'),
    pytest.param(dict(op=1, storage=9, compute=4, extents=[65536], batch=1,
                      modulus=576460756061519873, word_bits=64), id='ntt-word64-n65536'),
]


@pytest.mark.parametrize('kwargs', TARGETS)
def test_new_semantics_complete_runner_execution(runtime, kwargs, tmp_path):
    api, library, torch, _, _ = runtime
    transform = api.Transform(**kwargs)
    with api.Descriptor(library, transform) as description:
        description.validate()
        descriptor = json.loads(description.json())
        capabilities = json.loads(description.capabilities_json())
    case = dict(id=runner.digest(descriptor['semantics']), descriptor=descriptor,
                capabilities=capabilities, populations=['support-runner-integration'],
                requested_variants=[])
    row = runner.run_case(case, library, torch, torch.cuda.current_stream(),
                          timing=False, vkfft_library=None, gpuntt_library=None,
                          float_verdict='error-report', error_dir=tmp_path / 'errors')
    (tmp_path / 'row.json').write_text(json.dumps(row, indent=2, allow_nan=False) + '\n')
    assert row['capabilities']['status'] == 'eligible', row
    integer = descriptor['semantics']['storage'] in (8, 9)
    assert row['status'] == ('pass' if integer else 'accuracy-review-required'), row
    assert row['descriptor']['semantics'] == row['plan_descriptor']['semantics']
    expected_category = 'adapter' if row['mapping_kind'] in ('semantic-boundary', 'numeric-adapter') else 'native-contract'
    assert row['category'] == expected_category
    if (kwargs.get('fft_type', 0) and kwargs['storage'] in (0, 1)) or (kwargs['storage'] == 6 and kwargs['compute'] == 2):
        assert row['category'] == 'adapter', row
    assert 'timing' not in row
    if integer:
        assert row['correctness']['exact'] is True
        assert row['mapping_replay_correctness']['exact'] is True
    else:
        assert row['correctness']['verdict'] == 'accuracy-review-required'
        assert row['mapping_replay_correctness']['verdict'] == 'accuracy-review-required'
        assert 'error_distributions' in row['correctness']
        assert Path(row['error_artifact']['path']).is_file()
    assert all(baseline['status'] in ('pass', 'accuracy-review-required', 'missing-baseline', 'not-applicable')
               and baseline.get('support_status') != 'external-binding-error'
               for baseline in row['baselines']), row['baselines']


@pytest.mark.parametrize('storage', [0, 1])
def test_low_real_r2c_reference_preserves_imaginary_components(storage):
    semantics = dict(operator=0, storage=storage, fft_type=1, shape=[4, 8], batch=2,
                     inverse=False, normalize_inverse=True, length_mode=0,
                     ntt_input_layout=0, ntt_output_layout=0, fft_output_scale=1.0)
    source = runner.make_source(semantics)
    actual = runner.reference(semantics, source)
    expected = np.fft.rfftn(source, axes=(1, 2))
    assert actual.dtype == np.complex128
    assert np.any(actual.imag != 0)
    np.testing.assert_array_equal(actual, expected)
