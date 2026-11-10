#pragma once
#include "fft_register_tile.cuh"
#include <cufftdx.hpp>
#include <algorithm>

namespace cuntt::detail::factor_streamed {
// cuFFTDx native-half values pack two independent logical column transforms.
// Global/shared storage keeps the public Complex16 layout and logical columns.
template<class Complex> constexpr unsigned packed_transforms =
    std::is_same_v<Complex, Complex16> ? 2U : 1U;

__device__ __forceinline__ cufftdx::complex<__half2> half_pair_multiply(
    cufftdx::complex<__half2> a, cufftdx::complex<__half2> b) {
    return {__hsub2_rn(__hmul2_rn(a.x,b.x),__hmul2_rn(a.y,b.y)),
            __hadd2_rn(__hmul2_rn(a.x,b.y),__hmul2_rn(a.y,b.x))};
}
template<bool Inverse,unsigned Denominator>
__device__ __forceinline__ cufftdx::complex<__half2> half_pair_root(unsigned row,unsigned s0,unsigned s1) {
    // Phase calculation is coefficient generation; arithmetic products below
    // still round in FP16. Never convert a large index/denominator to half.
    const auto a=register_tile::root<Inverse,Complex32>(float(row*s0)/float(Denominator));
    const auto b=register_tile::root<Inverse,Complex32>(float(row*s1)/float(Denominator));
    return {__floats2half2_rn(a.real,b.real),__floats2half2_rn(a.imag,b.imag)};
}
template<bool Partial> __device__ __forceinline__ std::uint64_t physical_tile(
    std::uint64_t i,std::uint64_t first,std::uint64_t period,std::uint64_t span) {
    if constexpr(Partial) return first+(i/span)*period+i%span;
    else return i;
}
template<unsigned Columns,class Complex>
__device__ __forceinline__ unsigned index(unsigned row,unsigned col) {
    constexpr unsigned values_per_bank_cycle=128/sizeof(Complex);
    return row*Columns+(col^((row/(values_per_bank_cycle/Columns))&(Columns-1)));
}
template<unsigned Depth> __device__ __forceinline__ void commit() {
#if __CUDA_ARCH__ >= 800
    if constexpr(Depth) asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}
template<unsigned Depth> __device__ __forceinline__ void wait() {
#if __CUDA_ARCH__ >= 800
    // The input ring contains current + Depth future groups.  Leave the
    // newest Depth groups pending, waiting only for the current tile.  Empty
    // tail commits preserve that FIFO distance when fewer tiles remain.
    if constexpr(Depth) asm volatile("cp.async.wait_group %0;" :: "n"(Depth) : "memory");
#endif
}
template<unsigned Depth,class Complex>
__device__ __forceinline__ void copy(Complex* dst,const Complex* src) {
#if __CUDA_ARCH__ >= 800
    if constexpr(Depth) {
        const auto address=static_cast<unsigned>(__cvta_generic_to_shared(dst));
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2;" ::
            "r"(address),"l"(src),"n"(sizeof(Complex)) : "memory");
    } else *dst=*src;
#else
    *dst=*src;
#endif
}

template<unsigned Total,unsigned Log,unsigned Done,unsigned Columns,unsigned Threads,unsigned Depth,
         class Complex,bool Partial=false,bool StaticIO=false>
__device__ __forceinline__ void load_tile(const Complex* input,Complex* tile,
    std::uint64_t tile_id,std::uint64_t tile_count,std::uint64_t distance,std::uint64_t stride,
    std::uint64_t range_first=0,std::uint64_t period=1,std::uint64_t span=1) {
    constexpr unsigned F=1U<<Log, Independent=1U<<(Total-Log);
    if constexpr(StaticIO) {
        static_assert(Threads > 0 && Columns > 0 && Threads % Columns == 0,
                      "static factor I/O requires a positive thread count divisible by columns");
        static_assert((F * Columns) % Threads == 0,
                      "static factor I/O requires an integral load iteration count");
        constexpr unsigned LoadIterations=F*Columns/Threads;
        const auto tid=(Threads/(Columns/packed_transforms<Complex>))*threadIdx.y+threadIdx.x;
        if(tile_id<tile_count) {
            #pragma unroll
            for(unsigned element=0;element<LoadIterations;++element) {
                const auto i=tid+element*Threads;
                const auto row=i/Columns,col=i%Columns;
                const auto transform=physical_tile<Partial>(tile_id,range_first,period,span)*Columns+col;
                const auto j=transform&(Independent-1), base=(transform>>(Total-Log))*distance;
                copy<Depth>(&tile[index<Columns,Complex>(row,col)],input+base+(row*std::uint64_t(Independent)+j)*stride);
            }
        }
    } else {
        const auto tid=threadIdx.y*blockDim.x+threadIdx.x;
        if(tile_id<tile_count) for(unsigned i=tid;i<F*Columns;i+=blockDim.x*blockDim.y) {
            const auto row=i/Columns,col=i%Columns;
            const auto transform=physical_tile<Partial>(tile_id,range_first,period,span)*Columns+col;
            const auto j=transform&(Independent-1), base=(transform>>(Total-Log))*distance;
            copy<Depth>(&tile[index<Columns,Complex>(row,col)],input+base+(row*std::uint64_t(Independent)+j)*stride);
        }
    }
    // Empty commits at the tail retain a uniform FIFO distance across threads.
    commit<Depth>();
}

template<class FFT,unsigned Total,unsigned Log,unsigned Done,unsigned Columns,unsigned Tiles,unsigned Depth,
         bool Inverse,class Complex,bool Partial=false,bool StaticIO=false>
__launch_bounds__(FFT::max_threads_per_block)
__global__ void kernel(const Complex* input,Complex* output,std::uint64_t batch,
    std::uint64_t distance,std::uint64_t stride,bool normalize,
    std::uint64_t range_first=0,std::uint64_t range_count=0,std::uint64_t period=1,std::uint64_t span=1) {
    using Real=decltype(Complex{}.real);
    constexpr unsigned F=1U<<Log,P=1U<<Done,S=1U<<(Total-Done-Log),Rows=F/FFT::elements_per_thread;
    constexpr unsigned Threads=FFT::max_threads_per_block;
    constexpr unsigned Packed=packed_transforms<Complex>;
    static_assert(Columns%Packed==0, "packed FFT columns must contain complete transform pairs");
    // prefetch_depth is the number of tiles ahead of the current tile.  The
    // current tile therefore needs one additional ring slot: depth=1 is
    // explicitly a current+next double buffer, while depth=0 keeps a single
    // synchronous tile slot.
    // A CTA with one tile has no successor tile to overlap.  Keep the public
    // mapping expressible (so an explicit t1-d1 request remains valid), but
    // lower this degenerate schedule to the synchronous single-slot path
    // instead of paying for an unused async ring and reducing occupancy.
    constexpr unsigned EffectiveDepth=(Tiles>1 ? Depth : 0);
    constexpr unsigned TileValues=F*Columns, Slots=EffectiveDepth+1;
    if constexpr(StaticIO) {
        constexpr unsigned StoreIterations=TileValues/Threads;
        static_assert(FFT::input_length == F, "static factor I/O FFT input length must match tile");
        static_assert(Threads == FFT::block_dim.x*FFT::block_dim.y,
                      "static factor I/O thread count must match the FFT block shape");
        static_assert(FFT::block_dim.y == Columns/Packed && FFT::block_dim.x == Rows,
                      "static factor I/O block shape must match columns and EPT");
        static_assert(TileValues % Threads == 0,
                      "static factor I/O requires an integral store iteration count");
    }
    const auto tid=threadIdx.y*blockDim.x+threadIdx.x;
    const auto total_tiles=Partial ? range_count : (batch<<(Total-Log))/Columns;
    const auto first=std::uint64_t(blockIdx.x)*Tiles;
    const unsigned count=static_cast<unsigned>(min(std::uint64_t(Tiles),total_tiles-first));
    extern __shared__ __align__(16) unsigned char memory[];
    auto* ring=reinterpret_cast<Complex*>(memory);
    // Depth=0 retains the original in-place tile/work storage. Buffered
    // variants need a separate work tile after the depth+1 input ring.
    auto* work=memory+(EffectiveDepth ? Slots*TileValues*sizeof(Complex) : 0);
    auto* tile=reinterpret_cast<Complex*>(work);
    if constexpr(EffectiveDepth) {
        #pragma unroll
        for(unsigned d=0;d<Slots;++d) load_tile<Total,Log,Done,Columns,Threads,EffectiveDepth,Complex,Partial,StaticIO>(
            input,ring+d*TileValues,first+d,first+count,distance,stride,range_first,period,span);
    }
    for(unsigned iteration=0;iteration<count;++iteration) {
        auto* current=ring+(iteration%Slots)*TileValues;
        if constexpr(!EffectiveDepth) load_tile<Total,Log,Done,Columns,Threads,0,Complex,Partial,StaticIO>(input,current,first+iteration,first+count,distance,stride,range_first,period,span);
        wait<EffectiveDepth>(); __syncthreads();
        typename FFT::value_type values[FFT::storage_size];
        #pragma unroll
        for(unsigned i=0;i<FFT::storage_size;++i) {
            const auto row=threadIdx.x+i*Rows;
            const auto v=current[index<Columns,Complex>(row,Packed*threadIdx.y)];
            if constexpr(Packed==2) {
                const auto b=current[index<Columns,Complex>(row,2*threadIdx.y+1)];
                values[i]={__halves2half2(v.real,b.real),__halves2half2(v.imag,b.imag)};
            } else values[i]=typename FFT::value_type{v.real,v.imag};
        }
        __syncthreads(); // Every reader releases the ring slot before reuse.
        if constexpr(EffectiveDepth) load_tile<Total,Log,Done,Columns,Threads,EffectiveDepth,Complex,Partial,StaticIO>(
            input,current,first+iteration+Slots,first+count,distance,stride,range_first,period,span);
        FFT().execute(values,work);
        __syncthreads(); // cuFFTDx work area is now available for output layout.
        const auto physical=physical_tile<Partial>(first+iteration,range_first,period,span);
        if constexpr(Packed==2) {
            const auto transform=physical*Columns+2*threadIdx.y;
            const auto j0=transform&((1ULL<<(Total-Log))-1);
            const auto j1=(transform+1)&((1ULL<<(Total-Log))-1);
            auto cross=half_pair_root<Inverse,F*S>(threadIdx.x,j0>>Done,j1>>Done);
            const auto step=half_pair_root<Inverse,F*S>(Rows,j0>>Done,j1>>Done);
            #pragma unroll
            for(unsigned i=0;i<FFT::storage_size;++i) {
                const auto row=threadIdx.x+i*Rows;
                auto v=values[i];
                if constexpr(S>1) {
                    if(i && i%8==0) cross=half_pair_root<Inverse,F*S>(row,j0>>Done,j1>>Done);
                    v=half_pair_multiply(v,cross);
                    cross=half_pair_multiply(cross,step);
                }
                // Native-half normalized inverse is explicitly rejected by
                // the public contract and module entry points, not rescaled here.
                tile[index<Columns,Complex>(row,2*threadIdx.y)]={__low2half(v.x),__low2half(v.y)};
                tile[index<Columns,Complex>(row,2*threadIdx.y+1)]={__high2half(v.x),__high2half(v.y)};
            }
        } else {
        const auto transform=physical*Columns+threadIdx.y;
        const auto j=transform&((1ULL<<(Total-Log))-1);
        const auto s=j>>Done;
        auto cross=register_tile::root<Inverse,Complex>(Real(threadIdx.x*s)/Real(F*S));
        const auto step=register_tile::root<Inverse,Complex>(Real(Rows*s)/Real(F*S));
        const Real scale=Inverse && normalize && Done+Log==Total ? Real(1)/Real(1ULL<<Total) : Real(1);
        #pragma unroll
        for(unsigned i=0;i<FFT::storage_size;++i) {
            const auto row=threadIdx.x+i*Rows;
            Complex v{values[i].x,values[i].y};
            if constexpr(S>1) {
                if(i && i%8==0) cross=register_tile::root<Inverse,Complex>(Real(row*s)/Real(F*S));
                v=register_tile::multiply(v,cross); cross=register_tile::multiply(cross,step);
            }
            tile[index<Columns,Complex>(row,threadIdx.y)]={v.real*scale,v.imag*scale};
        }
        }
        __syncthreads();
        if constexpr(StaticIO) {
            constexpr unsigned StoreIterations=TileValues/Threads;
            const auto static_tid=(Threads/(Columns/Packed))*threadIdx.y+threadIdx.x;
            #pragma unroll
            for(unsigned element=0;element<StoreIterations;++element) {
                const auto i=static_tid+element*Threads;
                const auto row=Done==0 ? i%F : i/Columns,col=Done==0 ? i/F : i%Columns;
                const auto t=physical*Columns+col;
                const auto k=t&((1ULL<<(Total-Log))-1);
                const auto base=(t>>(Total-Log))*distance;
                output[base+(((k>>Done)*F+row)*P+(k&(P-1)))*stride]=tile[index<Columns,Complex>(row,col)];
            }
        } else for(unsigned i=tid;i<TileValues;i+=FFT::max_threads_per_block) {
            const auto row=Done==0 ? i%F : i/Columns,col=Done==0 ? i/F : i%Columns;
            const auto t=physical*Columns+col;
            const auto k=t&((1ULL<<(Total-Log))-1);
            const auto base=(t>>(Total-Log))*distance;
            output[base+(((k>>Done)*F+row)*P+(k&(P-1)))*stride]=tile[index<Columns,Complex>(row,col)];
        }
        __syncthreads(); // Do not reuse output work until all stores consumed it.
    }
    // Tail commits contain no writes; all real copies were waited on above.
}
template<class FFT,unsigned Columns,unsigned Tiles,unsigned Depth,class Complex>
constexpr unsigned shared_bytes() {
    constexpr unsigned tile=FFT::input_length*Columns*sizeof(Complex);
    constexpr unsigned effective_depth=Tiles>1 ? Depth : 0;
    return (effective_depth ? (effective_depth+1)*tile : 0)+std::max(tile,FFT::shared_memory_size);
}
}
