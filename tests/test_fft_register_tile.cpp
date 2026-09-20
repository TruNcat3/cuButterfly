#include "cuntt/butterfly.hpp"
#include "../src/fft_register_dispatch.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
void check(cudaError_t s) { if(s!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(s)); }
struct Buffer {
    cuntt::Complex32* p=nullptr;
    explicit Buffer(std::size_t n) { check(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(*p))); }
    ~Buffer() { cudaFree(p); }
};
void verify(cuntt::ButterflyConfig c, unsigned pattern) {
    const std::size_t n=std::size_t{1}<<c.log_n;
    c.batch=3;
    c.element_stride=pattern==2 ? 3 : 1;
    c.batch_stride=n*c.element_stride+13;
    c.inverse=pattern!=0; c.normalize_inverse=pattern==2;
    c.placement=pattern==1 ? cuntt::ButterflyPlacement::InPlace : cuntt::ButterflyPlacement::OutOfPlace;
    c.auto_allocate_workspace=false;
    cuntt::ButterflyPlan plan(c);
    auto reference_config=c;
    reference_config.backend=cuntt::ButterflyBackend::CuFft;
    reference_config.fft_core=cuntt::FftCore::Scalar;
    reference_config.stage_partition.clear(); reference_config.boundaries.clear();
    reference_config.shared_layout=cuntt::SharedLayout::Linear;
    reference_config.cross_twiddle=cuntt::CrossTwiddleMode::Table;
    reference_config.placement=cuntt::ButterflyPlacement::OutOfPlace;
    reference_config.auto_allocate_workspace=true;
    cuntt::ButterflyPlan reference(reference_config);
    const auto count=plan.data_size()/sizeof(cuntt::Complex32);
    Buffer input(count), output(count), refout(count), workspace(plan.workspace_size()/sizeof(cuntt::Complex32)+128);
    std::vector<cuntt::Complex32> values(count,{0,0}),actual(count),expected(count);
    std::mt19937 rng(817);
    std::uniform_real_distribution<float> random(-1,1);
    for(std::size_t b=0;b<c.batch;++b) for(std::size_t i=0;i<n;++i) {
        auto& v=values[b*c.batch_stride+i*c.element_stride];
        if(pattern==0) v={random(rng),random(rng)};
        if(pattern==1) v={i==17 ? float(b+1) : 0.0f,0};
        if(pattern==2) { const double a=6.283185307179586*double(i*(b+7)%n)/n; v={float(std::cos(a)),float(std::sin(a))}; }
    }
    check(cudaMemcpy(input.p,values.data(),count*sizeof(values[0]),cudaMemcpyHostToDevice));
    check(cudaMemset(workspace.p,0xa5,plan.workspace_size()+128*sizeof(cuntt::Complex32)));
    check(cudaDeviceSynchronize());
    plan.set_workspace(workspace.p+64,plan.workspace_size());
    cudaStream_t stream; check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    reference.set_stream(stream);
    plan.set_stream(stream);
    reference.execute_async(input.p,refout.p);
    auto* destination=c.placement==cuntt::ButterflyPlacement::InPlace ? input.p : output.p;
    plan.execute_async(input.p,destination);
    check(cudaStreamSynchronize(stream));
    check(cudaMemcpy(actual.data(),destination,count*sizeof(values[0]),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(expected.data(),refout.p,count*sizeof(values[0]),cudaMemcpyDeviceToHost));
    double max_error=0,max_reference=0,error2=0,reference2=0;
    for(std::size_t b=0;b<c.batch;++b) for(std::size_t i=0;i<n;++i) {
        const auto index=b*c.batch_stride+i*c.element_stride;
        const double e=std::hypot(double(actual[index].real)-expected[index].real,double(actual[index].imag)-expected[index].imag);
        const double v=std::hypot(double(expected[index].real),double(expected[index].imag));
        if(!std::isfinite(e)) throw std::runtime_error("nonfinite register-tile output");
        max_error=std::max(max_error,e); max_reference=std::max(max_reference,v);
        error2+=e*e; reference2+=v*v;
    }
    if(max_error/max_reference>2e-6 || std::sqrt(error2/reference2)>2e-6 ||
       (pattern==0 && max_error>2e-4*std::sqrt(double(n)/1024)))
        throw std::runtime_error("register-tile disagrees with cuFFT");
    std::vector<unsigned char> guard(64*sizeof(cuntt::Complex32));
    for(auto* p : {workspace.p,workspace.p+64+plan.workspace_size()/sizeof(cuntt::Complex32)}) {
        check(cudaMemcpy(guard.data(),p,guard.size(),cudaMemcpyDeviceToHost));
        if(!std::all_of(guard.begin(),guard.end(),[](unsigned char x){return x==0xa5;}))
            throw std::runtime_error("register-tile workspace guard overwritten");
    }
    check(cudaStreamDestroy(stream));
    std::cout << "verified logN=" << c.log_n << " prefix=" << c.local_stages << " pattern=" << pattern
              << " relative_max=" << max_error/max_reference << " relative_l2=" << std::sqrt(error2/reference2) << '\n';
}
}
int main() {
  try {
    const auto inventory=cuntt::detail::register_tile_mappings();
    if(inventory.empty()) throw std::runtime_error("no register-tile mappings available");
    unsigned tested=0;
    // Cover all compiled prefix dimensions and suffix register layouts, plus
    // both directions, in-place output, odd distances and strided data.
    for(const auto& c:inventory) {
        const auto pc=c.prefix_threads/c.prefix_ept;
        const auto sc=c.suffix_threads*c.suffix_ept/(1U<<(c.log_n-c.local_stages));
        if(pc!=8 || (sc!=4 && !(c.suffix_ept==32 && sc==2))) continue;
        for(unsigned pattern=0;pattern<3;++pattern) { verify(c,pattern); ++tested; }
    }
    auto invalid=inventory.front(); invalid.prefix_ept=1;
    bool rejected=false;
    try { cuntt::ButterflyPlan plan(invalid); } catch(const std::invalid_argument&) { rejected=true; }
    if(!rejected) throw std::runtime_error("uncompiled prefix mapping accepted");
    invalid=inventory.front(); invalid.shared_layout=cuntt::SharedLayout::Linear; rejected=false;
    try { cuntt::ButterflyPlan plan(invalid); } catch(const std::invalid_argument&) { rejected=true; }
    if(!rejected) throw std::runtime_error("unimplemented layout accepted");
    std::cout << tested << " correctness cases passed\n";
  } catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
