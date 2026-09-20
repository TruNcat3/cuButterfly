#include "fft_register_dispatch.hpp"
#include "fft_register_tile.cuh"
#include "fft_grouped_suffix.cuh"
#include "jit_module.hpp"
#include "resident_mapping.hpp"
#include <algorithm>
#include <stdexcept>

namespace cuntt::detail {
namespace {
bool compiled_mapping(const ButterflyConfig& c) {
    // The precompiled inventory contains the historical native prefix and
    // linear shared layout only.  Any new codelet/layout identity must go
    // through the JIT path and must never silently execute this fallback.
    if (c.prefix_codelet != "native" || c.prefix_shared_layout != "linear") return false;
    // Cooperative prefix codelets are deliberately JIT-only.  Keep the
    // legacy inventory exact so an old native module cannot be reused with a
    // different physical thread geometry.
    if (c.prefix_codelet_lanes != 1) return false;
    if (resident_axes_requested(c.local_stage_partitions,c.exchange_chunks)) return false;
    if(c.precision!=ButterflyPrecision::Fp32) return false;
    const auto local=c.local_stages, suffix=c.log_n-local;
    if ((local!=6 && local!=8 && local!=10) || (suffix!=10 && suffix!=12)) return false;
    const auto r=1U<<(local/2), pc=c.prefix_threads/r;
    const auto product=std::uint64_t(c.suffix_threads)*c.suffix_ept;
    const auto sn=1U<<suffix;
    const auto sc=product/sn;
    return (pc==4 || pc==8 || pc==16 || (pc==32 && local<=8)) && product%sn==0 &&
        (suffix==10 ? ((c.suffix_ept==8 && sc==4) || (c.suffix_ept==16 && (sc==4 || sc==8))) :
         ((c.suffix_ept==16 || c.suffix_ept==32) && (sc==2 || sc==4)));
}
template <unsigned Local, bool Inverse>
void prefix_dispatch(const ButterflyConfig& c, const Complex32* input, Complex32* scratch, std::uint64_t batch, cudaStream_t stream) {
    const auto columns = c.prefix_threads / (1U << (Local/2));
#define PREFIX(C) register_tile::launch_prefix<Local,C,Inverse>(input,scratch,c.log_n,batch,c.batch_stride,c.element_stride,stream)
    switch(columns) {
        case 4: PREFIX(4); return;
        case 8: PREFIX(8); return;
        case 16: PREFIX(16); return;
        case 32: if constexpr(Local<=8) { PREFIX(32); return; }
    }
#undef PREFIX
    throw std::invalid_argument("register-tile prefix mapping is not compiled");
}
template <unsigned Suffix, bool Inverse>
void suffix_dispatch(const ButterflyConfig& c, const Complex32* scratch, Complex32* output, std::uint64_t batch, cudaStream_t stream) {
    const auto columns = c.suffix_threads * c.suffix_ept / (1U << Suffix);
#define SUFFIX(E,C) launch_grouped_suffix<Suffix,E,C,Inverse>(scratch,output,c.log_n,batch,stream,c.batch_stride,c.element_stride,c.normalize_inverse)
    if constexpr (Suffix==10) {
        if(c.suffix_ept==8 && columns==4) { SUFFIX(8,4); return; }
        if(c.suffix_ept==16 && columns==4) { SUFFIX(16,4); return; }
        if(c.suffix_ept==16 && columns==8) { SUFFIX(16,8); return; }
    } else {
        if(c.suffix_ept==16 && columns==2) { SUFFIX(16,2); return; }
        if(c.suffix_ept==16 && columns==4) { SUFFIX(16,4); return; }
        if(c.suffix_ept==32 && columns==2) { SUFFIX(32,2); return; }
        if(c.suffix_ept==32 && columns==4) { SUFFIX(32,4); return; }
    }
#undef SUFFIX
    throw std::invalid_argument("register-tile suffix mapping is not compiled");
}
template <bool Inverse>
void dispatch_group(const ButterflyConfig& c,unsigned group,const Complex32* input,Complex32* output,std::uint64_t batch,cudaStream_t stream) {
    if (group==0) {
        switch(c.local_stages) {
            case 6: prefix_dispatch<6,Inverse>(c,input,output,batch,stream); return;
            case 8: prefix_dispatch<8,Inverse>(c,input,output,batch,stream); return;
            case 10: prefix_dispatch<10,Inverse>(c,input,output,batch,stream); return;
            default: throw std::invalid_argument("uncompiled register-tile prefix dimension");
        }
    }
    if (group!=1) throw std::invalid_argument("register-tile group must be 0 or 1");
    switch(c.log_n-c.local_stages) {
        case 10: suffix_dispatch<10,Inverse>(c,input,output,batch,stream); break;
        case 12: suffix_dispatch<12,Inverse>(c,input,output,batch,stream); break;
        default: throw std::invalid_argument("uncompiled register-tile suffix dimension");
    }
}
}

void validate_register_tile(const ButterflyConfig& c) {
    if (c.prefix_codelet != "native" && c.prefix_codelet != "cufftdx-thread")
        throw std::invalid_argument("register-tile prefix_codelet must be native or cufftdx-thread");
    if (c.prefix_shared_layout != "linear" && c.prefix_shared_layout != "xor")
        throw std::invalid_argument("register-tile prefix_shared_layout must be linear or xor");
    if (c.precision == ButterflyPrecision::Fp64 && c.prefix_codelet == "cufftdx-thread")
        throw std::invalid_argument("cufftdx-thread prefix codelet is unavailable for FP64");
    if ((c.prefix_codelet != "native" || c.prefix_shared_layout != "linear") && !on_demand_compilation_enabled())
        throw std::invalid_argument("non-default register prefix strategy requires CUBUTTERFLY_COMPILE_MODE=research or auto");
    const auto is_power_of_two = [](std::uint32_t value) {
        return value != 0 && (value & (value - 1)) == 0;
    };
    const auto lanes = c.prefix_codelet_lanes;
    if (!is_power_of_two(lanes))
        throw std::invalid_argument("register-tile prefix_codelet_lanes must be a positive power of two");
    if (lanes > 1 && (!on_demand_compilation_enabled() ||
                      c.backend != ButterflyBackend::OnlineReorder || c.fft_core != FftCore::RegisterTile))
        throw std::invalid_argument("cooperative prefix codelet lanes require an online register-tile mapping with research/auto compilation");
    if(c.op!=ButterflyOperator::Fft || (c.precision!=ButterflyPrecision::Fp32 && c.precision!=ButterflyPrecision::Fp64) ||
       c.backend!=ButterflyBackend::OnlineReorder || c.stage_partition.size()!=2 ||
       (c.stage_overlap && !c.batch_tile_count) ||
       c.local_exchange!=LocalExchange::SharedMemory || c.direct_boundary!=DirectBoundary::Strided ||
       c.shared_layout!=SharedLayout::WriterAligned || c.cross_twiddle!=CrossTwiddleMode::Recurrence ||
       c.complex_multiply!=ComplexMultiply::FourMul ||
       (c.compute_unit!=ComputeUnit::Auto && c.compute_unit!=ComputeUnit::Radix2) ||
       c.boundaries.size()!=1 || c.boundaries[0].residency!=FftBoundaryResidency::GlobalScratch)
        throw std::invalid_argument("register-tile needs FP32/64 online FFT, two global groups, writer-aligned placement, recurrence and a positive pipeline tile");
    const unsigned local=c.local_stages, suffix=c.log_n-local;
    if(local<2 || local>12 || suffix<3 || suffix>14)
        throw std::invalid_argument("register-tile template supports prefix logN=2..12 and suffix logN=3..14");
    validate_resident_axis_counts(c.local_stage_partitions,c.exchange_chunks,2);
    unsigned log_a=local/2,log_b=local/2;
    if(!c.local_stage_partitions.empty() && !c.local_stage_partitions[0].empty()) {
        const auto& p=c.local_stage_partitions[0];
        if(p.size()!=2 || !p[0] || !p[1] || p[0]>=local || p[1]!=local-p[0])
            throw std::invalid_argument("register prefix needs two positive local factors covering local_stages");
        log_a=p[0];log_b=p[1];
    } else if(local%2) throw std::invalid_argument("odd register prefix needs an explicit local stage partition");
    if(!c.local_stage_partitions.empty() && !c.local_stage_partitions[1].empty())
        throw std::invalid_argument("opaque suffix core does not expose its internal local stage partition");
    if(resident_axes_requested(c.local_stage_partitions,c.exchange_chunks) && !on_demand_compilation_enabled())
        throw std::invalid_argument("resident factor and exchange strategies require on-demand compilation");
    const unsigned a=1U<<log_a,b=1U<<log_b,pn=1U<<local,sn=1U<<suffix;
    if (lanes > a || a % lanes)
        throw std::invalid_argument("register-tile prefix_codelet_lanes must divide the local register FFT width");
    const unsigned expected_ept=a/lanes;
    if (c.prefix_ept != expected_ept)
        throw std::invalid_argument("register-tile prefix EPT must equal first factor size/prefix_codelet_lanes");
    if(c.prefix_ept>b || b%c.prefix_ept || !c.prefix_threads ||
       (std::uint64_t(c.prefix_threads)*c.prefix_ept)%pn)
        throw std::invalid_argument("register prefix requires common EPT dividing both local factors and integral columns");
    const unsigned pc=std::uint64_t(c.prefix_threads)*c.prefix_ept/pn,other_lanes=b/c.prefix_ept;
    if (!is_power_of_two(pc) || pc > sn)
        throw std::invalid_argument("register-tile prefix columns must be a power of two dividing the suffix dimension");
    if ((lanes>1 && lanes*pc>32) || (other_lanes>1 && other_lanes*pc>32))
        throw std::invalid_argument("cooperative prefix lanes and columns exceed the 32-thread shuffle group");
    const auto suffix_product=std::uint64_t(c.suffix_threads)*c.suffix_ept;
    const unsigned sc=static_cast<unsigned>(suffix_product/sn);
    if(suffix_product%sn || !sc || (sc & (sc-1)) || sc>16 || sc>pn ||
       !c.suffix_ept || c.suffix_ept>sn || (c.suffix_ept & (c.suffix_ept-1)))
        throw std::invalid_argument("register-tile grouped suffix requires power-of-two EPT and columns <=16 dividing the prefix dimension");
    const unsigned chunk=c.exchange_chunks.empty()?0:c.exchange_chunks[1];
    if((!c.exchange_chunks.empty() && c.exchange_chunks[0]) ||
       (chunk && (sc==1 || !is_power_of_two(chunk) || chunk>c.suffix_ept || c.suffix_ept%chunk)))
        throw std::invalid_argument("suffix exchange chunk must divide EPT; prefix and single-column exchange use zero");
    int device; cudaDeviceProp props{};
    if(cudaGetDevice(&device)!=cudaSuccess || cudaGetDeviceProperties(&props,device)!=cudaSuccess)
        throw std::runtime_error("cannot inspect register-tile device resources");
    const auto width=c.precision==ButterflyPrecision::Fp64 ? sizeof(Complex64) : sizeof(Complex32);
    if(log_a!=log_b && c.prefix_shared_layout=="xor") {
        const auto bank_rows=std::min<std::size_t>(b,128/(width*pc));
        if(lanes>bank_rows || other_lanes>bank_rows)
            throw std::invalid_argument("rectangular XOR lane digits exceed the shared transaction rows");
    }
    if(c.prefix_threads>static_cast<unsigned>(props.maxThreadsPerBlock) ||
       c.suffix_threads>static_cast<unsigned>(props.maxThreadsPerBlock) ||
       std::uint64_t(pn)*pc*width>props.sharedMemPerBlockOptin ||
       (sc>1 && std::uint64_t(sn)*sc*width*(chunk?chunk:c.suffix_ept)/c.suffix_ept>props.sharedMemPerBlockOptin))
        throw std::invalid_argument("register-tile shared state or thread count exceeds active GPU capacity");
    const auto launch_batch=c.stage_overlap ? std::min<std::uint64_t>(c.batch,c.batch_tile_count) : c.batch;
    if((launch_batch*sn/pc)>static_cast<unsigned>(props.maxGridSize[0]) ||
       (launch_batch*pn/sc)>static_cast<unsigned>(props.maxGridSize[0]))
        throw std::invalid_argument("register-tile grid exceeds active GPU capacity");
    if(!compiled_mapping(c)) prepare_register_module(c);
}
void launch_register_tile(const ButterflyConfig& c,const void* input,void* output,void* scratch,cudaStream_t stream) {
    if(!compiled_mapping(c)) {
        if(!launch_prepared_register_module(c,input,output,scratch,stream))
            throw std::logic_error("register FFT module was not prepared during plan creation");
        return;
    }
    launch_register_tile_group(c,0,input,scratch,c.batch,stream);
    launch_register_tile_group(c,1,scratch,output,c.batch,stream);
}
void launch_register_tile_group(const ButterflyConfig& c,unsigned group,const void* input,void* output,
                                std::uint64_t batch,cudaStream_t stream) {
    // JIT plans retain their group entry pointer; this adapter is precompiled
    // only and performs no cache lookup or configuration allocation at launch.
    if(!compiled_mapping(c)) throw std::logic_error("register group requires its prepared module entry");
    if(c.inverse) dispatch_group<true>(c,group,static_cast<const Complex32*>(input),static_cast<Complex32*>(output),batch,stream);
    else dispatch_group<false>(c,group,static_cast<const Complex32*>(input),static_cast<Complex32*>(output),batch,stream);
}
std::vector<ButterflyConfig> register_tile_mappings() {
    std::vector<ButterflyConfig> result;
    for (unsigned local : {6U,8U,10U}) for (unsigned suffix : {10U,12U}) {
        for(unsigned pc : {4U,8U,16U,32U}) for(unsigned ept : {8U,16U,32U}) for(unsigned sc : {2U,4U,8U}) {
            ButterflyConfig c;
            c.op=ButterflyOperator::Fft; c.precision=ButterflyPrecision::Fp32;
            c.backend=ButterflyBackend::OnlineReorder; c.fft_core=FftCore::RegisterTile;
            c.log_n=local+suffix; c.local_stages=local; c.batch=1;
            c.stage_partition={local,suffix}; c.shared_layout=SharedLayout::WriterAligned;
            c.cross_twiddle=CrossTwiddleMode::Recurrence; c.reorder_columns=1;
            c.prefix_ept=1U<<(local/2); c.prefix_threads=pc*c.prefix_ept;
            c.suffix_ept=ept; c.suffix_threads=(1U<<suffix)/ept*sc;
            c.boundaries={{CrossTwiddleMode::Recurrence,DirectBoundary::Strided,FftBoundaryResidency::GlobalScratch}};
            if (!compiled_mapping(c)) continue;
            try { validate_register_tile(c); result.push_back(c); }
            catch(const std::invalid_argument&) { /* uncompiled or resource-infeasible */ }
        }
    }
    return result;
}
}
