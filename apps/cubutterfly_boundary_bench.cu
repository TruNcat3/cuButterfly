// Controlled organization experiment. These explicit shared tile-group
// instantiations are not public AUTO candidates or an installation calibration.
#include "operator_traits.cuh"
#include "shared_iterative.cuh"
#include "cuntt/mixed_dataflow.hpp"
#include <cufft.h>
#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {
void check(cudaError_t s) {
    if (s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s));
}
void fft_check(cufftResult s) {
    if (s != CUFFT_SUCCESS) throw std::runtime_error("cuFFT status " + std::to_string(s));
}
template<class T> struct Buffer {
    T* p = nullptr;
    explicit Buffer(std::size_t n) { if (n) check(cudaMalloc(&p, n * sizeof(T))); }
    ~Buffer() { if (p) cudaFree(p); }
    Buffer(const Buffer&) = delete;
};
struct Stream {
    cudaStream_t s{};
    Stream() { check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(s); }
};
struct FftPlan { cufftHandle p{}; ~FftPlan() { if (p) cufftDestroy(p); } };
struct Graph {
    cudaGraph_t graph{}; cudaGraphExec_t exec{}; std::size_t nodes{};
    ~Graph() { if (exec) cudaGraphExecDestroy(exec); if (graph) cudaGraphDestroy(graph); }
    void capture(cudaStream_t stream, const std::function<void()>& launch) {
        launch(); check(cudaStreamSynchronize(stream)); // lazy initialization is untimed
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        launch(); check(cudaStreamEndCapture(stream, &graph));
        check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        check(cudaGraphGetNodes(graph, nullptr, &nodes));
    }
};
struct Options {
    std::string op, precision, only;
    unsigned log_n, batch, split, tiles, threads, inverse, repeat, trials;
};
unsigned number(const char* arg) {
    const std::string s(arg);
    if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("expected nonnegative integer");
    const auto x = std::stoull(s);
    if (x > std::numeric_limits<unsigned>::max()) throw std::invalid_argument("integer too large");
    return static_cast<unsigned>(x);
}
template<class T> double norm(T x) { return double(x) * double(x); }
template<> double norm(cuntt::Complex32 x) { return double(x.real)*x.real + double(x.imag)*x.imag; }
template<> double norm(cuntt::Complex64 x) { return x.real*x.real + x.imag*x.imag; }
template<class T> double distance(T a, T b) { return std::abs(double(a)-double(b)); }
template<> double distance(cuntt::Complex32 a, cuntt::Complex32 b) {
    return std::hypot(double(a.real)-b.real, double(a.imag)-b.imag);
}
template<> double distance(cuntt::Complex64 a, cuntt::Complex64 b) {
    return std::hypot(a.real-b.real, a.imag-b.imag);
}
template<class T> bool finite(T x) { return std::isfinite(x); }
template<> bool finite(cuntt::Complex32 x) { return std::isfinite(x.real) && std::isfinite(x.imag); }
template<> bool finite(cuntt::Complex64 x) { return std::isfinite(x.real) && std::isfinite(x.imag); }

template<class V, class Real> void cpu_fft(std::vector<V>& values, const std::vector<V>& roots,
                                         unsigned log_n) {
    // Arithmetic and stored coefficients use Real. In particular, FP32 does
    // not inherit the legacy reference's FP64 complex-product intermediates.
    std::size_t reversed=0;
    for (std::size_t i=1;i<values.size();++i) {
        auto bit=values.size()/2;
        while (reversed & bit) { reversed^=bit; bit>>=1; }
        reversed^=bit;
        if (i<reversed) std::swap(values[i],values[reversed]);
    }
    for (unsigned stage=0;stage<log_n;++stage) {
        const auto half=std::size_t{1}<<stage;
        for (std::size_t base=0;base<values.size();base+=2*half)
            for (std::size_t j=0;j<half;++j) {
                const V a=values[base+j], b=values[base+j+half], w=roots[half-1+j];
                const Real vr=b.real*w.real-b.imag*w.imag;
                const Real vi=b.real*w.imag+b.imag*w.real;
                values[base+j]={a.real+vr,a.imag+vi};
                values[base+j+half]={a.real-vr,a.imag-vi};
            }
    }
}

template<class Op, unsigned Tiles> void stage(const Options& o, unsigned first, unsigned stages,
        const typename Op::Value* input, typename Op::Value* output, Op op,
        bool transposed_input, bool transposed_output, cudaStream_t stream) {
    using V = typename Op::Value;
    const auto bytes = 2ULL * (1ULL << stages) * Tiles * sizeof(V);
    const auto kernel = cuntt::detail::shared_iterative_kernel<Op,-1,-1,-1,-1,0,Tiles>;
    if (bytes > 48 * 1024) check(cudaFuncSetAttribute(kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes)));
    const auto tile_count = 1U << (o.log_n - stages);
    const auto grid = o.batch * ((tile_count + Tiles - 1) / Tiles);
    cuntt::detail::shared_iterative_kernel<Op,-1,-1,-1,-1,0,Tiles><<<grid,o.threads,bytes,stream>>>(input,output,o.log_n,first,stages,
        std::size_t{1} << o.log_n,1,true,o.batch,1,op,false,transposed_input,
        transposed_output,o.split,o.log_n-o.split);
    check(cudaGetLastError());
}
template<class Op> void dispatch(const Options& o, unsigned tiles, unsigned first, unsigned stages,
        const typename Op::Value* input, typename Op::Value* output, Op op,
        bool ti, bool to, cudaStream_t stream) {
    switch (tiles) {
        case 1: stage<Op,1>(o,first,stages,input,output,op,ti,to,stream); break;
        case 2: stage<Op,2>(o,first,stages,input,output,op,ti,to,stream); break;
        case 4: stage<Op,4>(o,first,stages,input,output,op,ti,to,stream); break;
        case 8: stage<Op,8>(o,first,stages,input,output,op,ti,to,stream); break;
        default: throw std::invalid_argument("tile group must be 1,2,4,8");
    }
}
template<class V, class Real, bool IsFft> int run(const Options& o) {
    using Op = std::conditional_t<IsFft,cuntt::traits::FftOperator<V,Real>,cuntt::traits::FwhtOperator<Real>>;
    const auto n = std::size_t{1} << o.log_n, count = n * o.batch;
    const auto hardware = cuntt::query_hardware_resource_model();
    const auto shared = 2ULL * (1ULL << std::max(o.split,o.log_n-o.split)) * o.tiles * sizeof(V);
    if (shared > hardware.shared_bytes_per_block || o.threads > hardware.max_threads_per_block)
        throw std::invalid_argument("requested tile group exceeds static hardware resources");
    if (count > hardware.memory_bytes / (8 * sizeof(V)))
        throw std::invalid_argument("experiment working set exceeds conservative memory budget");
    Stream stream;
    Buffer<V> input(count), output(count), scratch(count), auxiliary(count), twiddles(IsFft ? n-1 : 0);
    std::vector<V> host(count), reference(count), canonical(count), actual(count);
    std::mt19937 rng(1237);
    std::uniform_real_distribution<Real> dist(-1,1);
    for (auto& v : host) {
        if constexpr (IsFft) v = {dist(rng),dist(rng)};
        else v = dist(rng);
    }
    Op op{};
    op.inverse = bool(o.inverse); op.normalize_inverse = false; // equal unnormalized inverse for cuFFT
    std::vector<V> roots(IsFft ? n-1 : 0);
    if constexpr (IsFft) {
        for (unsigned s=0; s<o.log_n; ++s) for (unsigned k=0; k<(1U<<s); ++k) {
            const double angle = (o.inverse ? 1 : -1)*3.14159265358979323846*k/(1U<<s);
            roots[(1U<<s)-1+k] = {Real(std::cos(angle)),Real(std::sin(angle))};
        }
        check(cudaMemcpy(twiddles.p,roots.data(),(n-1)*sizeof(V),cudaMemcpyHostToDevice));
        op.twiddles=twiddles.p; op.multiply=cuntt::ComplexMultiply::FourMul;
    }
    check(cudaMemcpy(input.p,host.data(),count*sizeof(V),cudaMemcpyHostToDevice));
    // Same-storage, same-arithmetic-width CPU reference, all batches, untimed.
    for (unsigned b=0;b<o.batch;++b) {
        std::vector<V> ref(host.begin()+b*n,host.begin()+(b+1)*n);
        if constexpr (IsFft) cpu_fft<V,Real>(ref,roots,o.log_n);
        else cuntt::reference_fwht(ref,bool(o.inverse),false);
        std::copy(ref.begin(),ref.end(),reference.begin()+b*n);
    }
    auto launch = [&](const std::string& mode) {
        const bool tx = mode != "strided";
        const bool fused = mode == "producer";
        // Group width is fixed across all three organization controls.
        dispatch(o,o.tiles,0,o.split,input.p,scratch.p,op,false,fused,stream.s);
        const V* boundary = scratch.p;
        if (mode == "materialized") {
            const dim3 grid(((1U<<o.split)+31)/32,((1U<<(o.log_n-o.split))+31)/32,o.batch);
            cuntt::detail::iterative_boundary_transpose_kernel<V><<<grid,dim3(32,8),0,stream.s>>>(
                scratch.p,auxiliary.p,o.log_n,o.split,o.log_n-o.split,n,1,o.batch);
            check(cudaGetLastError()); boundary=auxiliary.p;
        }
        dispatch(o,o.tiles,o.split,o.log_n-o.split,boundary,output.p,op,tx,false,stream.s);
    };
    // Compare to the unchanged, ungrouped scalar core as an independent
    // organization check. The floating reference error is reported separately.
    dispatch(o,1,0,o.split,input.p,scratch.p,op,false,false,stream.s);
    dispatch(o,1,o.split,o.log_n-o.split,scratch.p,output.p,op,false,false,stream.s);
    check(cudaStreamSynchronize(stream.s));
    check(cudaMemcpy(canonical.data(),output.p,count*sizeof(V),cudaMemcpyDeviceToHost));
    FftPlan fft;
    if constexpr (IsFft) {
        int length=static_cast<int>(n);
        fft_check(cufftPlanMany(&fft.p,1,&length,nullptr,1,length,nullptr,1,length,
                               sizeof(Real)==8 ? CUFFT_Z2Z : CUFFT_C2C,o.batch));
        fft_check(cufftSetStream(fft.p,stream.s));
    }
    auto invoke = [&](const std::string& mode) {
        if (mode != "cufft") launch(mode);
        else if constexpr (IsFft && sizeof(Real)==8) fft_check(cufftExecZ2Z(fft.p,
            reinterpret_cast<cufftDoubleComplex*>(input.p),reinterpret_cast<cufftDoubleComplex*>(output.p),
            o.inverse ? CUFFT_INVERSE : CUFFT_FORWARD));
        else if constexpr (IsFft) fft_check(cufftExecC2C(fft.p,
            reinterpret_cast<cufftComplex*>(input.p),reinterpret_cast<cufftComplex*>(output.p),
            o.inverse ? CUFFT_INVERSE : CUFFT_FORWARD));
    };
    std::vector<std::string> modes = {"strided","materialized","producer"};
    if constexpr (IsFft) modes.push_back("cufft");
    if (o.only != "all") {
        if (std::find(modes.begin(),modes.end(),o.only)==modes.end())
            throw std::invalid_argument("unknown mode or missing external comparator");
        modes={o.only};
    }
    std::vector<std::unique_ptr<Graph>> graphs;
    std::vector<std::vector<double>> times(modes.size());
    for (const auto& mode : modes) {
        auto g=std::make_unique<Graph>();
        g->capture(stream.s,[&]{ invoke(mode); }); graphs.push_back(std::move(g));
    }
    cudaEvent_t start{}, stop{};
    check(cudaEventCreate(&start)); check(cudaEventCreate(&stop));
    for (unsigned trial=0;trial<o.trials;++trial) for (unsigned offset=0;offset<modes.size();++offset) {
        const auto k=(trial+offset)%modes.size();
        for (unsigned w=0;w<5;++w) check(cudaGraphLaunch(graphs[k]->exec,stream.s));
        check(cudaEventRecord(start,stream.s));
        for (unsigned i=0;i<o.repeat;++i) check(cudaGraphLaunch(graphs[k]->exec,stream.s));
        check(cudaEventRecord(stop,stream.s)); check(cudaEventSynchronize(stop));
        float ms{}; check(cudaEventElapsedTime(&ms,start,stop));
        times[k].push_back(ms/o.repeat);
    }
    check(cudaEventDestroy(start)); check(cudaEventDestroy(stop));
    bool valid=true;
    std::cout << std::setprecision(12)
        << "operator,precision,logN,batch,split,tiles_per_cta,threads,inverse,mode,graph_nodes,shared_bytes_per_cta,median_ms,min_ms,max_ms,max_abs_cpu,relative_l2_cpu,nonfinite,max_abs_same_core,verified_batches,trials_ms\n";
    for (unsigned k=0;k<modes.size();++k) {
        check(cudaGraphLaunch(graphs[k]->exec,stream.s)); check(cudaStreamSynchronize(stream.s));
        check(cudaMemcpy(actual.data(),output.p,count*sizeof(V),cudaMemcpyDeviceToHost));
        double max_error=0, max_core=0; long double err2=0, ref2=0; std::size_t nonfinite=0;
        for (std::size_t i=0;i<count;++i) {
            if (!finite(actual[i]) || !finite(reference[i]) || !finite(canonical[i])) { ++nonfinite; continue; }
            const auto error=distance(actual[i],reference[i]);
            max_error=std::max(max_error,error); err2+=static_cast<long double>(error)*error;
            ref2+=norm(reference[i]); max_core=std::max(max_core,distance(actual[i],canonical[i]));
        }
        valid=valid && nonfinite==0 && (modes[k]=="cufft" || max_core==0);
        auto t=times[k]; std::sort(t.begin(),t.end());
        const auto med=(t[(t.size()-1)/2]+t[t.size()/2])/2;
        std::cout << o.op << ',' << o.precision << ',' << o.log_n << ',' << o.batch << ',' << o.split
            << ',' << o.tiles << ',' << o.threads << ',' << o.inverse << ',' << modes[k] << ','
            << graphs[k]->nodes << ',' << (modes[k]=="cufft" ? 0 : shared) << ',' << med << ','
            << t.front() << ',' << t.back() << ',' << max_error << ',' << std::sqrt(err2/std::max(ref2,1e-300L))
            << ',' << nonfinite << ',' << max_core << ',' << o.batch << ',';
        for (unsigned trial=0;trial<times[k].size();++trial)
            std::cout << (trial ? ";" : "") << times[k][trial];
        std::cout << '\n';
    }
    return valid ? 0 : 2;
}
}
int main(int argc,char** argv) {
    try {
        if (argc != 12) throw std::invalid_argument(
            "usage: cubutterfly_boundary_bench fft|fwht fp32|fp64 logN batch split tiles threads inverse repeat trials all|strided|materialized|producer|cufft");
        const Options o{argv[1],argv[2],argv[11],number(argv[3]),number(argv[4]),number(argv[5]),
            number(argv[6]),number(argv[7]),number(argv[8]),number(argv[9]),number(argv[10])};
        if (o.log_n<2 || o.log_n>24 || !o.split || o.split>=o.log_n || !o.batch || o.batch>65535 ||
            o.inverse>1 || !o.repeat || o.repeat>10000 || !o.trials || o.trials>31 ||
            (o.tiles!=1 && o.tiles!=2 && o.tiles!=4 && o.tiles!=8) || !o.threads || o.threads%32)
            throw std::invalid_argument("invalid bounded experiment configuration");
        if (o.op=="fft" && o.precision=="fp64") return run<cuntt::Complex64,double,true>(o);
        if (o.op=="fft" && o.precision=="fp32") return run<cuntt::Complex32,float,true>(o);
        if (o.op=="fwht" && o.precision=="fp64") return run<double,double,false>(o);
        if (o.op=="fwht" && o.precision=="fp32") return run<float,float,false>(o);
        throw std::invalid_argument("unknown operator/precision");
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
