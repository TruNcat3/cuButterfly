"""CPU-only native-half factor module preparation contracts."""
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from research_compile_requests import compile_request


def point(**changes):
    p=dict(operator='fft',precision='fp16',accumulation='native',logN=18,
           direction='forward',normalization='none',mapping_json=dict(
               backend='factor-streamed',fft_core='cufftdx-block',stage_partition=[18],
               factor_partition=[6,6,6],factor_columns=8,factor_ept=8,
               data_tiles_per_cta=4,prefetch_depth=2,factor_slices=2,factor_overlap=True,
               factor_io_policies=['dynamic','static-unrolled','dynamic']))
    p.update(changes)
    return p


def test_native_half_factor_cache_and_schedule_identity():
    r=compile_request(point(),'research')
    assert r['precision']=='fp16' and r['accumulation']=='native'
    assert r['packing']=='half2-transform-pair-v1'
    assert r['factor_columns']==8 and r['factor_ept']==8
    assert r['factor_slices']==2 and r['prefetch_depth']==2
    assert r['factor_io_policies']==['dynamic','static-unrolled','dynamic']
    assert 'direction' not in r and 'normalization' not in r


def test_unscaled_inverse_compiles_same_two_direction_module():
    assert compile_request(point(),'research')==compile_request(point(direction='inverse'),'research')


@pytest.mark.parametrize('changes',[dict(direction='inverse',normalization='inverse'),
                                    dict(inverse=True,normalize_inverse=True),dict(accumulation='fp32')])
def test_unsupported_arithmetic_contracts_rejected(changes):
    with pytest.raises(ValueError): compile_request(point(**changes),'research')


@pytest.mark.parametrize('columns',[2,4,8,16,32])
def test_logical_even_columns_preserved(columns):
    p=point();p['mapping_json']['factor_columns']=columns
    assert compile_request(p,'research')['factor_columns']==columns


@pytest.mark.parametrize('columns',[1,3,64])
def test_invalid_columns_rejected(columns):
    p=point();p['mapping_json']['factor_columns']=columns
    with pytest.raises(ValueError): compile_request(p,'research')


def test_native_register_core_not_widened_to_half():
    p=point();p['mapping_json']['fft_core']='register-tile'
    with pytest.raises(ValueError): compile_request(p,'research')


def test_precompiled_policy_cannot_silently_accept_half_factor():
    with pytest.raises(ValueError): compile_request(point(),'precompiled')


@pytest.mark.parametrize('precision',['fp32','fp64'])
def test_existing_factor_cache_fields_are_unchanged(precision):
    r=compile_request(point(precision=precision),'research')
    assert 'accumulation' not in r and 'packing' not in r


@pytest.mark.parametrize('policy',['research','precompiled'])
def test_bf16_is_an_explicit_provider_gap(policy):
    with pytest.raises(ValueError,match='BF16'):
        compile_request(point(precision='bf16'),policy)
