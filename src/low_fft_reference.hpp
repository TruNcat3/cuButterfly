#pragma once
// Declared unfused CPU arithmetic, matching scripts/low_fft_cpu_reference.py.
// GPU contraction and vendor internal arithmetic may differ.
#include "cuntt/butterfly.hpp"
#include <cmath>
#include <vector>
#include <type_traits>
#include <stdexcept>
namespace cuntt::detail {
template<class Scalar> float low_fft_widen(Scalar x) {
    if constexpr(std::is_same_v<Scalar,Fp16>) return __half2float(x);
    else return __bfloat162float(x);
}
template<class Scalar> Scalar low_fft_narrow(float x) {
    if constexpr(std::is_same_v<Scalar,Fp16>) return __float2half_rn(x);
    else return __float2bfloat16_rn(x);
}
template<class Value> void reference_low_fft(std::vector<Value>& values,bool inverse,bool normalize,bool native) {
    using Scalar=decltype(Value{}.real);
    const auto n=values.size();
    if(n<2 || (n&(n-1))) throw std::invalid_argument("low FFT reference requires power of two >=2");
    auto q=[](float x) { return low_fft_widen(low_fft_narrow<Scalar>(x)); };
    // Volatile intermediate prevents host compiler FMA contraction.
    auto mul=[](float a,float b) { volatile float x=a*b; return x; };
    auto add=[](float a,float b) { volatile float x=a+b; return x; };
    auto sub=[](float a,float b) { volatile float x=a-b; return x; };
    for(std::size_t i=1,j=0;i<n;++i) {
        auto bit=n>>1; for(;j&bit;bit>>=1)j^=bit; j^=bit;
        if(i<j)std::swap(values[i],values[j]);
    }
    const float limit=std::is_same_v<Scalar,Fp16> ? 16376.0F : low_fft_widen(Bf16(__nv_bfloat16_raw{0x7e7f}));
    for(std::size_t half=1;half<n;half*=2) {
        for(std::size_t j=0;j<half;++j) {
            const double angle=(inverse ? 1.0 : -1.0)*std::acos(-1.0)*double(j)/double(half);
            const float wr=q(static_cast<float>(std::cos(angle))),wi=q(static_cast<float>(std::sin(angle)));
            for(std::size_t base=0;base<n;base+=2*half) {
                auto a=values[base+j],b=values[base+j+half];
                float ar=low_fft_widen(a.real),ai=low_fft_widen(a.imag),br=low_fft_widen(b.real),bi=low_fft_widen(b.imag);
                const bool scale=inverse&&normalize;
                const bool after=scale&&std::abs(ar)<=limit&&std::abs(ai)<=limit&&std::abs(br)<=limit&&std::abs(bi)<=limit;
                if(scale&&!after) { ar=q(ar*.5F);ai=q(ai*.5F);br=q(br*.5F);bi=q(bi*.5F); }
                float rr=mul(br,wr),ii=mul(bi,wi),ri=mul(br,wi),ir=mul(bi,wr);
                if(native) { rr=q(rr);ii=q(ii);ri=q(ri);ir=q(ir); }
                const float re=q(sub(rr,ii)),im=q(add(ri,ir));
                float lr=q(add(ar,re)),li=q(add(ai,im)),hr=q(sub(ar,re)),hi=q(sub(ai,im));
                if(after) { lr=q(lr*.5F);li=q(li*.5F);hr=q(hr*.5F);hi=q(hi*.5F); }
                values[base+j]={low_fft_narrow<Scalar>(lr),low_fft_narrow<Scalar>(li)};
                values[base+j+half]={low_fft_narrow<Scalar>(hr),low_fft_narrow<Scalar>(hi)};
            }
        }
    }
}
}
