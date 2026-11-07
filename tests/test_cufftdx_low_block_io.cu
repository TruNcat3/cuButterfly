// CPU compile; GPU execution must be scheduled separately.
// Usage: test_io OLD_MODULE NEW_MODULE N
// OLD_MODULE must be a frozen, same-geometry pre-I/O-change module.
#include <cubutterfly/module.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <dlfcn.h>
#include <vector>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <cmath>
static void check(cudaError_t status) {
    if(status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
struct Module {
    void* handle;
    const cubutterflyModuleV1* api;
    explicit Module(const char* path) {
        handle=dlopen(path,RTLD_NOW|RTLD_LOCAL);
        if(!handle) throw std::runtime_error(dlerror());
        auto entry=reinterpret_cast<cubutterflyModuleEntryV1>(dlsym(handle,"cubutterfly_module_v1"));
        if(!entry) throw std::runtime_error("missing module entry");
        api=entry();
    }
    ~Module(){dlclose(handle);}
};
int main(int argc,char** argv) {try {
    if(argc!=4) throw std::runtime_error("usage: test_io OLD_MODULE NEW_MODULE N");
    Module old(argv[1]), current(argv[2]);
    const size_t n=std::stoul(argv[3]); unsigned cases=0;
    for(size_t batch:{1U,3U,5U}) for(size_t stride:{1U,2U,3U})
    for(unsigned offset:{0U,1U,2U,3U}) for(bool inplace:{false,true})
    for(bool inverse:{false,true}) for(bool normalize:{false,true}) {
        const size_t distance=n*stride+7, bytes=batch*distance*4, pad=64;
        std::vector<unsigned char> source(bytes+2*pad+offset,0xa5);
        for(size_t t=0;t<batch;++t) for(size_t j=0;j<n;++j) {
            __half pair[2]={__float2half(float(std::sin(j*.177+t)*.03125)),
                            __float2half(float(std::cos(j*.137+t)*.03125))};
            std::memcpy(source.data()+pad+offset+4*(t*distance+j*stride),pair,4);
        }
        // Baseline old adapter always receives naturally aligned storage;
        // the new adapter matches it for half-aligned pointers; odd bytes reject.
        unsigned char *old_x,*old_y,*new_x,*new_y;
        check(cudaMalloc(&old_x,bytes)); check(cudaMalloc(&old_y,bytes));
        check(cudaMalloc(&new_x,source.size())); check(cudaMalloc(&new_y,source.size()));
        check(cudaMemcpy(old_x,source.data()+pad+offset,bytes,cudaMemcpyHostToDevice));
        check(cudaMemset(old_y,0xa5,bytes));
        check(cudaMemcpy(new_x,source.data(),source.size(),cudaMemcpyHostToDevice));
        check(cudaMemset(new_y,0xa5,source.size()));
        cubutterflyModuleInvocationV1 a{old_x,inplace?old_x:old_y,nullptr,batch,distance,stride,inverse,normalize,nullptr};
        auto b=a; b.input=new_x+pad+offset; b.output=(inplace?new_x:new_y)+pad+offset;
        if(offset&1U) {
            if(current.api->launch(&b)!=cudaErrorInvalidValue)
                throw std::runtime_error("odd-byte pointer was not rejected");
            check(cudaDeviceSynchronize());
            std::vector<unsigned char> unchanged(source.size());
            check(cudaMemcpy(unchanged.data(),new_x,unchanged.size(),cudaMemcpyDeviceToHost));
            if(unchanged!=source) throw std::runtime_error("rejected invocation changed input");
            check(cudaFree(old_x));check(cudaFree(old_y));check(cudaFree(new_x));check(cudaFree(new_y));
            ++cases; continue;
        }
        check(static_cast<cudaError_t>(old.api->launch(&a)));
        check(static_cast<cudaError_t>(current.api->launch(&b))); check(cudaDeviceSynchronize());
        std::vector<unsigned char> expected(bytes), actual(source.size()), input_after(source.size());
        check(cudaMemcpy(expected.data(),a.output,bytes,cudaMemcpyDeviceToHost));
        check(cudaMemcpy(actual.data(),inplace?new_x:new_y,actual.size(),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(input_after.data(),new_x,input_after.size(),cudaMemcpyDeviceToHost));
        if(!inplace && input_after!=source) throw std::runtime_error("input/guard changed");
        auto wanted=inplace?source:std::vector<unsigned char>(source.size(),0xa5);
        for(size_t t=0;t<batch;++t) for(size_t j=0;j<n;++j) {
            const size_t index=4*(t*distance+j*stride);
            std::memcpy(wanted.data()+pad+offset+index,expected.data()+index,4);
        }
        if(actual!=wanted) throw std::runtime_error("old/new output bytes or guards/gaps differ");
        check(cudaFree(old_x));check(cudaFree(old_y));check(cudaFree(new_x));check(cudaFree(new_y)); ++cases;
    }
    std::cout<<"old/new equality and odd-pointer rejection cases="<<cases<<'\n'; return 0;
} catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}}
