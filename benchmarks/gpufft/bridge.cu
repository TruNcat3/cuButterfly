// Restricted complete C2C adapter. Upstream arithmetic is linked unchanged.
#include "gpufft/fft.cuh"
#include "gpufft/twiddles.cuh"
#include <stdexcept>
using namespace gpufft;
struct Base { virtual ~Base() = default; virtual int run(void*,void*,cudaStream_t)=0; };
inline void check(cudaError_t e) { if(e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
template<class T> __global__ void reorder(const COMPLEX<T>* a, COMPLEX<T>* b, int logn, size_t count) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count) { size_t n=size_t(1)<<logn; unsigned r=__brev(unsigned(i&(n-1)))>>(32-logn); b[i]=a[(i&~(n-1))+r]; }
}
template<class T> struct Plan: Base {
    int logn,batch; bool inverse; size_t count; COMPLEX<T>* roots=nullptr; COMPLEX<T>* scratch=nullptr; fft_configuration<T> cfg{};
    Plan(int l,int b,bool inv,bool norm):logn(l),batch(b),inverse(inv),count((size_t(1)<<l)*b) {
        try {
            auto host=make_fft_twiddles_host<T>(l,X_N_minus,inv?FORWARD:INVERSE);
            check(cudaMalloc(&roots,host.size()*sizeof(COMPLEX<T>)));
            check(cudaMalloc(&scratch,count*sizeof(COMPLEX<T>)));
            check(cudaMemcpy(roots,host.data(),host.size()*sizeof(COMPLEX<T>),cudaMemcpyHostToDevice));
            cfg.n_power=l; cfg.fft_type=inv?INVERSE:FORWARD; cfg.reduction_poly=X_N_minus;
            cfg.mod_inverse=COMPLEX<T>(inv&&norm?T(1)/T(size_t(1)<<l):T(1),T(0));
        } catch(...) { cudaFree(roots); cudaFree(scratch); throw; }
    }
    ~Plan() override { cudaFree(scratch); cudaFree(roots); }
    int run(void* a,void* b,cudaStream_t stream) override {
        cfg.stream=stream;
        if(inverse) reorder<<<(count+255)/256,256,0,stream>>>((COMPLEX<T>*)a,scratch,logn,count);
        else check(cudaMemcpyAsync(scratch,a,count*sizeof(COMPLEX<T>),cudaMemcpyDeviceToDevice,stream));
        check(cudaGetLastError());
        GPU_FFT(scratch,roots,cfg,batch,false);
        check(cudaGetLastError());
        if(inverse) check(cudaMemcpyAsync(b,scratch,count*sizeof(COMPLEX<T>),cudaMemcpyDeviceToDevice,stream));
        else reorder<<<(count+255)/256,256,0,stream>>>(scratch,(COMPLEX<T>*)b,logn,count);
        return int(cudaGetLastError());
    }
};
extern "C" int gpufft_create(int l,int b,int bits,int inv,int norm,void** out) {
    if(!out || l<1 || l>28 || b<1 || (l>9 && b>65535) || (uint64_t(b)<<l)>=(1ull<<31) || (bits!=32&&bits!=64) || (inv!=0&&inv!=1) || (norm!=0&&norm!=1)) return -1;
    *out=nullptr;
    try { *out=bits==32?static_cast<Base*>(new Plan<float>(l,b,inv,norm)):static_cast<Base*>(new Plan<double>(l,b,inv,norm)); return 0; } catch(...) { return -2; }
}
extern "C" int gpufft_execute(void* p,void* a,void* b,void* stream) {
    if(!p||!a||!b) return -1;
    try { return static_cast<Base*>(p)->run(a,b,(cudaStream_t)stream); } catch(...) { return -2; }
}
extern "C" int gpufft_destroy(void* p) { delete static_cast<Base*>(p); return 0; }
