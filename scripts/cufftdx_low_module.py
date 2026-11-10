"""Native-half cuFFTDx single-local-group lowering; paired transform packing.

Normalization is performed once on final output in FP16. It differs from the
scalar core's stagewise normalized inverse and needs its own error measurements.
"""
from __future__ import annotations
import json

PACKING = "half2-transform-pair-v1"

def validate_point(point, sm, capabilities=None):
    for key, expected in (("backend", "temporal-tile"), ("fft_core", "cufftdx-block"),
                          ("precision", "fp16"), ("accumulation", "native"), ("packing", PACKING)):
        if point.get(key) != expected:
            raise ValueError(f"native FP16 block requires {key}={expected}")
    # A projection must not silently discard requested implementation axes.
    # Omitted defaults remain compatible with compact compiler requests.
    defaults = {
        "operator": "fft", "shared_layout": "linear", "local_exchange": "shared",
        "complex_multiply": "four-mul", "prefix_codelet": "native",
        "prefix_shared_layout": "linear", "prefix_twiddle": "runtime",
        "prefix_codelet_lanes": 1, "suffix_core": "cufftdx-block",
        "data_tiles_per_cta": 1, "prefetch_depth": 0, "factor_slices": 1,
        "factor_overlap": False,
    }
    for key, expected in defaults.items():
        if point.get(key, expected) != expected:
            raise ValueError(f"native FP16 block requires {key}={expected}")
    if point.get("compute_unit", "auto") not in ("auto", "radix2"):
        raise ValueError("native FP16 block requires auto or radix2 compute_unit")
    for key in ("boundaries", "local_stage_partitions", "exchange_chunks",
                "segment_mappings", "execution_group_mappings", "factor_partition"):
        if point.get(key):
            raise ValueError(f"native FP16 block does not implement {key}")
    def integer(key):
        value=point.get(key)
        if isinstance(value,bool) or not isinstance(value,int) or value<=0:
            raise ValueError(f"{key} must be a positive integer")
        return value
    logn,threads,ept=integer("logN"),integer("tile_threads"),integer("prefix_ept")
    # CUDA's architectural CTA ceiling is 1024; enumeration additionally
    # applies the selected device limit, and compiled resources remain decisive.
    if logn>20 or threads<32 or threads>1024 or threads%32 or ept not in (2,4,8,16,32):
        raise ValueError("unsupported FP16 block size or geometry")
    n=1<<logn
    if n%ept or threads%(n//ept):
        raise ValueError("tile_threads must contain integral paired FFT rows")
    fpb=2*threads//(n//ept)
    if capabilities is not None:
        status=capabilities.check_both(sm*10,"fp16",n,ept)
        if status == "unsupported":
            raise ValueError(f"FP16 cuFFTDx provider capability is {status}")
    if point.get("stage_partition",[logn]) != [logn]:
        raise ValueError("FP16 block module implements one local group")
    if point.get("stage_overlap",False):
        raise ValueError("FP16 block module does not implement range overlap")
    return dict(size=n,threads=threads,ept=ept,ffts_per_block=fpb,
                pairs_per_block=fpb//2,packing=PACKING,normalization="final-store-fp16")

def render(point,sm,identity,capabilities=None):
    g=validate_point(point,sm,capabilities)
    return f'''#include "cufftdx_low_block.cuh"
using namespace cufftdx;
template<bool Inverse> using FFT=decltype(Block()+Size<{g['size']}>()+Precision<__half>()+
    Type<fft_type::c2c>()+Direction<Inverse ? fft_direction::inverse : fft_direction::forward>()+
    ElementsPerThread<{g['ept']}>()+FFTsPerBlock<{g['ffts_per_block']}>()+SM<{sm*10}>());
static_assert(FFT<false>::block_dim.x*FFT<false>::block_dim.y=={g['threads']});
static int resources(int inverse,unsigned group,cubutterflyModuleResourcesV1* out) {{
    if(group || !out) return cudaErrorInvalidValue;
    return inverse ? cuntt::detail::low_block_resources<FFT<true>,true>(out)
                   : cuntt::detail::low_block_resources<FFT<false>,false>(out);
}}
static int launch(const cubutterflyModuleInvocationV1* io) {{
    if(!io) return cudaErrorInvalidValue;
    return io->inverse ? cuntt::detail::low_block_launch<FFT<true>,true>(io)
                       : cuntt::detail::low_block_launch<FFT<false>,false>(io);
}}
extern "C" int cubutterfly_module_launch_group_v1(unsigned group,const cubutterflyModuleInvocationV1* io) {{
    return group ? cudaErrorInvalidValue : launch(io);
}}
extern "C" int cubutterfly_module_local_bytes_v1(int inverse,unsigned group,uint64_t* out) {{
    if(group || !out) return cudaErrorInvalidValue;
    return inverse ? cuntt::detail::low_block_resources<FFT<true>,true>(nullptr,out)
                   : cuntt::detail::low_block_resources<FFT<false>,false>(nullptr,out);
}}
extern "C" const cubutterflyModuleV1* cubutterfly_module_v1() {{
    static const cubutterflyModuleV1 api{{1,sizeof(cubutterflyModuleV1),{json.dumps(identity)},{sm},1,resources,launch}};
    return &api;
}}
'''
