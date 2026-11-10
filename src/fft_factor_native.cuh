#pragma once
#include "fft_factor_streamed.cuh"

namespace cuntt::detail::factor_streamed {
// Two register codelets with one explicitly mapped shared handoff. The
// external factor input/output order is identical to the imported block core.
// This adapter currently supports balanced even-log factors; other shapes
// remain candidates of the imported core rather than being silently changed.
template<unsigned Log,unsigned Ept,unsigned Columns,bool Inverse,class Complex>
struct NativeFactor {
    static_assert(Log%2==0 && Ept==(1U<<(Log/2)));
    using Real=decltype(Complex{}.real);
    using value_type=cufftdx::complex<Real>;
    static constexpr unsigned R=Ept, storage_size=R, elements_per_thread=R;
    static constexpr unsigned input_length=1U<<Log, max_threads_per_block=R*Columns;
    static constexpr unsigned shared_memory_size=input_length*Columns*sizeof(Complex);
    static constexpr dim3 block_dim{R,Columns,1};

    __device__ static unsigned handoff_index(unsigned major,unsigned minor,unsigned col) {
        constexpr unsigned values_per_bank_cycle=128/sizeof(Complex);
        const auto q=minor^major;
        return (major*R+q)*Columns+(col^((q/(values_per_bank_cycle/Columns))&(Columns-1)));
    }
    __device__ __forceinline__ void execute(value_type* values,void* shared) const {
        auto* tile=static_cast<Complex*>(shared);
        const auto lane=threadIdx.x, col=threadIdx.y;
        Complex v[R];
        #pragma unroll
        for(unsigned i=0;i<R;++i) v[i]={values[i].x,values[i].y};
        register_tile::fft<Log/2,Inverse>(v);
        const auto step=register_tile::root<Inverse,Complex>(Real(lane)/Real(input_length));
        Complex cross{Real(1),Real(0)};
        #pragma unroll
        for(unsigned i=0;i<R;++i) {
            if(i && i%8==0) cross=register_tile::root<Inverse,Complex>(Real(i*lane)/Real(input_length));
            tile[handoff_index(i,lane,col)]=register_tile::multiply(v[i],cross);
            cross=register_tile::multiply(cross,step);
        }
        __syncthreads();
        #pragma unroll
        for(unsigned i=0;i<R;++i) v[i]=tile[handoff_index(lane,i,col)];
        register_tile::fft<Log/2,Inverse>(v);
        #pragma unroll
        for(unsigned i=0;i<R;++i) values[i]=value_type{v[i].real,v[i].imag};
    }
};
}
