"""CPU-only semantic boundaries and unchanged upstream source audit."""
import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('hadacore', ROOT/'scripts/hadacore_baseline.py')
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


def semantic(**changes):
    s = dict(schema='cubutterfly-semantic-v1', operator=2, storage=0, compute=0,
             shape=[256], batch=1, length_mode=0, placement=0,
             inverse=0, normalize_inverse=False, input_orders=[0], output_orders=[0])
    s.update(changes)
    return s


@pytest.mark.parametrize('change', [dict(storage=2), dict(compute=1), dict(storage=1),
    dict(shape=[1]), dict(shape=[65536]), dict(shape=[True]), dict(batch=True),
    dict(shape=[128]), dict(input_orders=[1]), dict(output_strides=[2]),
    dict(output_rounding=1), dict(inverse=True), dict(normalize_inverse=1), dict(placement=2)])
def test_reject_semantics_before_loading_library(change):
    assert h.support(semantic(**change), '/does/not/exist')['status'] == 'unsupported-semantic'


def test_dependency_and_full_batch_scale(tmp_path):
    assert h.support(semantic(), '/does/not/exist')['status'] == 'missing-external-dependency'
    lib = tmp_path/'mock.so'
    lib.write_bytes(b'not loaded by support')
    s = semantic(shape=[128, 256], batch=37, input_orders=[0, 0], output_orders=[0, 0],
                 inverse=1, normalize_inverse=True)
    report = h.support(s, lib)
    assert report['supported']
    assert report['metadata']['batch'] == 37
    assert report['metadata']['adapter_output_scale'] == 1/(32768**.5)
    assert report['metadata']['arithmetic_equivalence'] == 'not bitwise/stage-rounding equivalent'
    report['metadata']['semantic']['shape'][0] = 1
    assert s['shape'] == [128, 256]


def test_power_of_two_large_product_and_no_padding(tmp_path):
    lib = tmp_path/'mock.so'
    lib.touch()
    assert h.support(semantic(shape=[2], batch=128), lib)['supported']
    assert not h.support(semantic(shape=[2], batch=127), lib)['supported']
    assert not h.support(semantic(batch=2**24), lib)['supported']


def test_source_hash_gate():
    with pytest.raises(ValueError, match='hash mismatch'):
        h.transformed_source('// changed upstream')


def test_pointer_placement_and_stream_validation_without_cuda():
    p = h.Plan.__new__(h.Plan)
    p.report = dict(metadata=dict(flattened_points=256, batch=1, adapter_output_scale=16,
                                 semantic=dict(placement=0)))
    calls = []
    p.fn = lambda *args: calls.append(args) or 0
    p.execute(4096, 8192, stream=123)
    assert calls[-1] == (4096, 8192, 256, 256, 16, 123)
    for source, destination, stream in [(4096,4096,0), (4096,4100,0), (4096,8192,None)]:
        with pytest.raises(ValueError):
            p.execute(source, destination, stream=stream)
    p.close()
    with pytest.raises(RuntimeError, match='closed'):
        p.execute(4096, 8192, stream=0)


@pytest.mark.parametrize('requested,aliased', [(1, False), (0, True), (0, False), (1, True)])
def test_build_core_projects_actual_scratch_placement(monkeypatch, tmp_path, requested, aliased):
    from types import SimpleNamespace
    import copy

    lib = tmp_path/'mock.so'
    lib.touch()
    calls = []

    class Tensor:
        dtype, device, is_cuda = 'fp16', 'cuda:0', True

        def __init__(self, pointer):
            self.pointer = pointer

        def is_contiguous(self):
            return True

        def numel(self):
            return 256

        def data_ptr(self):
            return self.pointer

    class FakePlan:
        def __init__(self, library, s):
            self.s = copy.deepcopy(s)

        def metadata(self):
            return {'semantic': copy.deepcopy(self.s)}

        def execute(self, source, destination, *, stream):
            assert self.s['placement'] == int(source == destination)
            calls.append((source, destination, stream))

        def close(self):
            pass

    monkeypatch.setattr(h, 'Plan', FakePlan)
    s = semantic(placement=requested)
    core = h.build_core(SimpleNamespace(float16='fp16'), s, Tensor(4096),
                        Tensor(4096 if aliased else 8192), stream=123, librarypath=lib)
    core()
    assert calls == [(4096, 4096 if aliased else 8192, 123)]
    m = core.metadata()
    assert m['semantic']['placement'] == requested
    assert m['core_semantic']['placement'] == int(aliased)
    assert m['placement_projection']['requested'] == requested
    assert m['placement_projection']['core'] == int(aliased)
    m['semantic']['placement'] = 99
    assert core.metadata()['semantic']['placement'] == requested
    assert s == semantic(placement=requested)
    core.close()
    with pytest.raises(RuntimeError, match='closed'):
        core()
