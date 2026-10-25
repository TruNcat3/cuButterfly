#include "xd_stockham_pair_fft.hpp"
#include "xd_pair_fft.hpp"
#include <cuda_runtime.h>
#include <cufft.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(cudaError_t s) { if(s!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(s)); }
void check_fft(cufftResult s) {
    if(s!=CUFFT_SUCCESS) throw std::runtime_error("cuFFT status "+std::to_string(s));
}
struct Buf {
    cufftComplex* p{};
    explicit Buf(std::size_t n) { if(n) check(cudaMalloc(&p,n*sizeof(*p))); }
    ~Buf() { if(p) cudaFree(p); }
    Buf(const Buf&)=delete;
    Buf& operator=(const Buf&)=delete;
};
struct Stream {
    cudaStream_t p{};
    Stream() { check(cudaStreamCreate(&p)); }
    ~Stream() { cudaStreamDestroy(p); }
};
struct Plan { cufftHandle p{}; ~Plan() { if(p) cufftDestroy(p); } };
struct Event {
    cudaEvent_t p{};
    Event() { check(cudaEventCreate(&p)); }
    ~Event() { cudaEventDestroy(p); }
};
struct Graph {
    cudaGraph_t graph{};
    cudaGraphExec_t exec{};
    std::size_t nodes{};
    ~Graph() { if(exec) cudaGraphExecDestroy(exec); if(graph) cudaGraphDestroy(graph); }
    void capture(cudaStream_t stream,unsigned chain,const std::function<void()>& launch) {
        check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        for(unsigned i=0;i<chain;++i) launch();
        check(cudaStreamEndCapture(stream,&graph));
        check(cudaGraphGetNodes(graph,nullptr,&nodes));
        check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    }
};
std::size_t number(const char* value) {
    const std::string text(value);
    if(text.empty() || text.find_first_not_of("0123456789")!=std::string::npos)
        throw std::invalid_argument("expected an unsigned integer");
    return std::stoull(text);
}
void dump(const std::string& prefix,const char* name,const std::vector<cufftComplex>& data) {
    if(prefix=="-") return;
    std::ofstream file(prefix+"."+name+".c64",std::ios::binary);
    file.write(reinterpret_cast<const char*>(data.data()),data.size()*sizeof(data[0]));
    if(!file) throw std::runtime_error("cannot write output dump");
}
double median(std::vector<double> values) {
    std::sort(values.begin(),values.end());
    return (values[(values.size()-1)/2]+values[values.size()/2])/2;
}
}

int main(int argc,char** argv) {
    try {
        // Serial graph chains amortize CPU submission gaps equally for all
        // providers. Each transform uses immutable input and overwrites output.
        // Compilation, allocation, plan creation, CPU checking and I/O are untimed.
        const std::size_t planes=argc>1 ? number(argv[1]):64;
        const std::size_t warmup=argc>2 ? number(argv[2]):5;
        const std::size_t repeat=argc>3 ? number(argv[3]):20;
        const std::size_t direction=argc>4 ? number(argv[4]):0;
        const std::size_t variant_id=argc>5 ? number(argv[5]):0;
        const std::string prefix=argc>6 ? argv[6]:"-";
        const std::size_t trials=argc>7 ? number(argv[7]):5;
        const std::size_t chain=argc>8 ? number(argv[8]):32;
        const std::string pattern=argc>9 ? argv[9]:"random";
        constexpr std::size_t points=32*32;
        if(argc>10 || !planes || planes>std::numeric_limits<int>::max() ||
           planes>std::numeric_limits<std::size_t>::max()/(points*sizeof(cufftComplex)) ||
           !repeat || repeat>100000 || warmup>100000 || !trials || trials>101 ||
           !chain || chain>1024 || direction>1 || variant_id>3 ||
           (pattern!="random" && pattern!="impulse" && pattern!="tone"))
            throw std::invalid_argument("usage: stockham_subgraph_bench [planes] [warmup] [repeat] "
                "[inverse=0|1] [variant=0..3] [dump-prefix|-] [trials] [graph-chain] [random|impulse|tone]");
        const bool inverse=direction==1;
        const auto variant=static_cast<cuntt::detail::XdStagePairVariant>(variant_id);
        const std::array<const char*,4> variants={
            "radix2_stage","radix4_stage","radix4_register","radix4_register_split"};
        const std::array<const char*,3> names={"candidate","cufftdx_pair","cufft"};
        if(!cuntt::detail::xd_pair_fft_available(32,32,false))
            throw std::runtime_error("this experiment requires the cuFFTDx pair provider");
        const std::size_t count=points*planes;
        Stream stream;
        Buf input(count),candidate(count),pair(count),reference(count);
        Buf scratch(variant_id==3 ? count:0);
        std::vector<cufftComplex> host(count);
        std::mt19937 rng(1237);
        std::uniform_real_distribution<float> dist(-1.0f,1.0f);
        for(std::size_t i=0;i<count;++i) {
            if(pattern=="random") host[i]={dist(rng),dist(rng)};
            else if(pattern=="impulse") {
                const auto pos=((i/points)*37+97)%points;
                host[i]=i%points==pos ? cufftComplex{1.0f,-0.5f}:cufftComplex{};
            } else {
                const auto row=(i%points)/32,col=i%32;
                const double phase=2*3.14159265358979323846*(3*row+7*col)/32;
                host[i]={static_cast<float>(std::cos(phase)),static_cast<float>(std::sin(phase))};
            }
        }
        check(cudaMemcpyAsync(input.p,host.data(),count*sizeof(host[0]),cudaMemcpyHostToDevice,stream.p));
        check(cudaStreamSynchronize(stream.p));
        Plan plan;
        int dims[2]={32,32};
        check_fft(cufftPlanMany(&plan.p,2,dims,nullptr,1,points,nullptr,1,points,CUFFT_C2C,
                               static_cast<int>(planes)));
        check_fft(cufftSetStream(plan.p,stream.p));
        const std::array<std::function<void()>,3> launch={{
            [&] { cuntt::detail::launch_xd_stockham_pair_fft(input.p,candidate.p,32,32,planes,
                      false,inverse,1.0,stream.p,variant,scratch.p); },
            [&] { cuntt::detail::launch_xd_pair_fft(input.p,pair.p,32,32,planes,false,inverse,1.0,stream.p); },
            [&] { check_fft(cufftExecC2C(plan.p,input.p,reference.p,inverse ? CUFFT_INVERSE:CUFFT_FORWARD)); }
        }};
        for(const auto& fn:launch) fn();
        check(cudaStreamSynchronize(stream.p));
        std::array<Graph,3> graphs;
        for(unsigned k=0;k<3;++k) graphs[k].capture(stream.p,static_cast<unsigned>(chain),launch[k]);
        Event start,stop;
        std::array<std::vector<double>,3> times;
        for(std::size_t trial=0;trial<trials;++trial) {
            for(unsigned offset=0;offset<3;++offset) {
                const unsigned k=(trial+offset)%3;
                for(std::size_t i=0;i<warmup;++i) check(cudaGraphLaunch(graphs[k].exec,stream.p));
                check(cudaStreamSynchronize(stream.p));
                check(cudaEventRecord(start.p,stream.p));
                for(std::size_t i=0;i<repeat;++i) check(cudaGraphLaunch(graphs[k].exec,stream.p));
                check(cudaEventRecord(stop.p,stream.p));
                check(cudaEventSynchronize(stop.p));
                float ms{}; check(cudaEventElapsedTime(&ms,start.p,stop.p));
                times[k].push_back(1000.0*ms/(repeat*chain));
            }
        }
        std::array<std::vector<cufftComplex>,3> outputs;
        const std::array<cufftComplex*,3> pointers={candidate.p,pair.p,reference.p};
        for(unsigned k=0;k<3;++k) {
            outputs[k].resize(count);
            check(cudaMemcpy(outputs[k].data(),pointers[k],count*sizeof(host[0]),cudaMemcpyDeviceToHost));
            dump(prefix,names[k],outputs[k]);
        }
        dump(prefix,"input",host);
        bool finite=true;
        std::cout<<std::setprecision(12)<<"{\"protocol\":\"serial_graph_chain_v2\",\"planes\":"<<planes
                 <<",\"inverse\":"<<(inverse ? "true":"false")
                 <<",\"variant\":\""<<variants[variant_id]<<"\",\"pattern\":\""<<pattern
                 <<"\",\"graph_chain\":"<<chain<<",\"replays\":"<<repeat
                 <<",\"warmup\":"<<warmup<<",\"trials\":"<<trials<<",\"providers\":{";
        for(unsigned k=0;k<3;++k) {
            bool provider_finite=true,comparison_finite=true;
            double max_error=0;
            for(std::size_t i=0;i<count;++i) {
                const auto a=outputs[k][i],b=outputs[2][i];
                if(!std::isfinite(a.x) || !std::isfinite(a.y)) provider_finite=false;
                if(!std::isfinite(b.x) || !std::isfinite(b.y)) comparison_finite=false;
                max_error=std::max(max_error,std::hypot(double(a.x)-b.x,double(a.y)-b.y));
            }
            finite=finite && provider_finite;
            if(k) std::cout<<',';
            std::cout<<'\"'<<names[k]<<"\":{\"median_us\":"<<median(times[k])
                     <<",\"nodes\":"<<graphs[k].nodes<<",\"finite\":"<<(provider_finite ? "true":"false")
                     <<",\"max_abs_vs_cufft\":";
            if(provider_finite && comparison_finite) std::cout<<max_error; else std::cout<<"null";
            std::cout<<",\"trial_us\":[";
            for(std::size_t j=0;j<times[k].size();++j) { if(j) std::cout<<','; std::cout<<times[k][j]; }
            std::cout<<"]}";
        }
        std::cout<<"},\"speedup_vs_cufft\":"<<median(times[2])/median(times[0])<<"}\n";
        // Numerical distributions are reported without inventing an acceptance
        // tolerance. The runner compares all providers to a CPU FP32 reference.
        return finite ? 0:2;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
