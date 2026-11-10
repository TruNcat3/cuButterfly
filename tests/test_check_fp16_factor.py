"""CPU contracts for the finite public FP16 factor correctness harness."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('fp16_factor_check',Path(__file__).with_name('check_fp16_factor.py'))
check=importlib.util.module_from_spec(spec);spec.loader.exec_module(check)


def test_finite_matrix_covers_each_requested_mechanism():
    tasks=check.matrix()
    assert len(tasks)==56 and len({t['id'] for t in tasks})==56
    assert {(t['n'],tuple(t['mapping']['factor_partition'])) for t in tasks}=={
        (4096,(6,6)),(32768,(7,8)),(262144,(6,6,6))}
    assert {t['mapping']['prefetch_depth'] for t in tasks}=={0,1,2}
    for t in tasks:
        assert t['batch']==3 and t['element_stride']==2 and t['batch_stride']==2*t['n']+16
        assert t['inverse']==t['in_place']
        assert check.module_point(t)['packing']=='half2-transform-pair-v1'
    assert any(t['mapping']['factor_overlap'] for t in tasks)
    assert any(t['schedule']=='slices' and not t['mapping']['factor_overlap'] for t in tasks)


def test_prepare_module_identity_omits_runtime_direction_and_overlap():
    tasks=check.matrix()
    points={}
    for t in tasks:
        p=check.module_point(t)
        assert 'inverse' not in p and 'normalize_inverse' not in p and 'factor_overlap' not in p
        points.setdefault((t['logN'],t['schedule']),[]).append(p)
    assert all(all(p==ps[0] for p in ps) for ps in points.values())
    assert points[(18,'slices')]==points[(18,'overlap')]


@pytest.mark.parametrize('inverse',[False,True])
def test_batch_distinct_impulse_cpu_reference_is_exact(inverse):
    source=np.zeros((3,128,2),np.float16);source[:,0,0]=[1,2,3]
    result=check.reference(source,inverse)
    assert np.array_equal(result.real,np.broadcast_to(np.arange(1,4)[:,None],(3,128)))
    assert np.all(result.imag==0)
