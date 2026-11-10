import pathlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).parents[1]/'scripts'))
import pytest
from cufftdx_low_module import validate_point,render

BASE=dict(backend='temporal-tile',fft_core='cufftdx-block',precision='fp16',accumulation='native',packing='half2-transform-pair-v1',logN=7,tile_threads=32,prefix_ept=8)

def test_valid_contract_without_provider():
    x=validate_point(BASE,80)
    assert x['size']==128 and x['ffts_per_block']==4 and x['pairs_per_block']==2
    assert x['normalization']=='final-store-fp16'

def test_reject_non_single_group():
    with pytest.raises(ValueError,match='one local'):
        validate_point(dict(BASE,stage_partition=[6,1]),80)

def test_reject_bad_packing_and_geometry():
    with pytest.raises(ValueError): validate_point(dict(BASE,packing='aos'),80)
    with pytest.raises(ValueError): validate_point(dict(BASE,tile_threads=48),80)

def test_render_has_odd_batch_mask_and_pair_contract():
    source=render(BASE,80,'test-id')
    assert 'FFTsPerBlock<4>' in source
    header=(pathlib.Path(__file__).parents[1]/'src/cufftdx_low_block.cuh').read_text()
    assert 't1<batch' in header and '__halves2half2(a.real,b.real)' in header
    assert '__high2half(re)' in header


def test_provider_unsupported_is_not_advertised():
    class Missing:
        def check_both(self,*args): return 'unsupported'
    with pytest.raises(ValueError,match='provider capability'):
        validate_point(BASE,80,Missing())

def test_installed_provider_and_large_tile():
    from cufftdx_capabilities import load_database
    root=pathlib.Path(__file__).parents[1]/'external/mathdx/nvidia/mathdx'
    if not root.exists(): pytest.skip('MathDX not installed')
    db=load_database(root)
    p=dict(BASE,logN=12,tile_threads=256,prefix_ept=16)
    assert validate_point(p,80,db)['ffts_per_block']==2


def test_unknown_provider_can_be_compiled_for_confirmation():
    class Unknown:
        def check_both(self,*args): return 'unknown'
    assert validate_point(BASE,80,Unknown())['size']==128


@pytest.mark.parametrize("key,value", [
    ("operator", "fwht"), ("compute_unit", "radix4"),
    ("shared_layout", "writer-aligned"), ("local_exchange", "warp-register"),
    ("complex_multiply", "three-mul"), ("prefix_codelet", "radix4"),
    ("prefix_shared_layout", "xor"), ("prefix_twiddle", "table"),
    ("prefix_codelet_lanes", 2), ("suffix_core", "native"),
    ("data_tiles_per_cta", 2), ("prefetch_depth", 1),
    ("factor_slices", 2), ("factor_overlap", True),
    ("boundaries", [{}]), ("local_stage_partitions", [[]]),
    ("exchange_chunks", [1]), ("segment_mappings", [{}]),
    ("execution_group_mappings", [{}]), ("factor_partition", [7]),
])
def test_projection_rejects_unimplemented_requested_axes(key,value):
    with pytest.raises(ValueError, match=key):
        validate_point(dict(BASE,**{key:value}),80)


def test_default_public_axes_and_unrelated_metadata_remain_valid():
    point=dict(BASE,operator="fft",compute_unit="radix2",shared_layout="linear",
               local_exchange="shared",complex_multiply="four-mul",
               prefix_codelet="native",prefix_shared_layout="linear",
               prefix_twiddle="runtime",prefix_codelet_lanes=1,
               suffix_core="cufftdx-block",data_tiles_per_cta=1,
               prefetch_depth=0,factor_slices=1,factor_overlap=False,
               boundaries=[],local_stage_partitions=[],exchange_chunks=[],
               segment_mappings=[],execution_group_mappings=[],factor_partition=[],
               batch=3,provenance="retained by caller")
    assert validate_point(point,80)["ffts_per_block"]==4


@pytest.mark.parametrize("threads,fpb", [(512, 2), (1024, 4)])
def test_wide_cta_uses_installed_fp16_provider(threads, fpb):
    from cufftdx_capabilities import load_database
    root = pathlib.Path(__file__).parents[1] / 'external/mathdx/nvidia/mathdx'
    if not root.exists():
        pytest.skip('MathDX not installed')
    db = load_database(root)
    point = dict(BASE, logN=12, tile_threads=threads, prefix_ept=8)
    shape = validate_point(point, 80, db)
    assert shape['threads'] == threads
    assert shape['ffts_per_block'] == fpb
    source = render(point, 80, 'wide-cpu-test', db)
    assert f'FFTsPerBlock<{fpb}>()' in source
    assert f'=={threads}' in source


@pytest.mark.parametrize("threads", [0, 48, 1056, 2048])
def test_cta_launch_bounds_remain_enforced(threads):
    with pytest.raises(ValueError):
        validate_point(dict(BASE, tile_threads=threads), 80)


def test_wide_cta_does_not_bypass_provider_capability_check():
    class Missing:
        def check_both(self, *args):
            return 'unsupported'
    with pytest.raises(ValueError, match='provider capability'):
        validate_point(dict(BASE, logN=12, tile_threads=512, prefix_ept=8), 80, Missing())
