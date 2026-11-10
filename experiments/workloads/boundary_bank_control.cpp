// Controlled prefix shared-staging bank map; global inter-stage layout is fixed.
#include <cuntt/butterfly.hpp>
#include <cubutterfly/mapping.hpp>
#include <cuda_profiler_api.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
using Json=nlohmann::json;
using V=cuntt::Complex32;
void check(cudaError_t s) { if(s!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(s)); }
struct Buffer { V* p=nullptr; explicit Buffer(size_t n) { check(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(V))); } ~Buffer(){if(p)cudaFree(p);} };
struct Stream { cudaStream_t s{}; Stream(){check(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));} ~Stream(){cudaStreamDestroy(s);} };
struct Graph { cudaGraph_t g{}; cudaGraphExec_t e{}; size_t nodes=0; ~Graph(){if(e)cudaGraphExecDestroy(e);if(g)cudaGraphDestroy(g);} };
struct Event { cudaEvent_t e{}; Event(){check(cudaEventCreate(&e));} ~Event(){cudaEventDestroy(e);} };
unsigned integer(const std::string& s) {
    if(s.empty()||s.find_first_not_of("0123456789")!=std::string::npos) throw std::invalid_argument("expected unsigned integer");
    auto n=std::stoull(s); if(n>1000000)throw std::invalid_argument("integer out of bounds"); return unsigned(n);
}
struct Options { unsigned log=20,batch=16,repeat=20,trials=7; std::string only="both"; bool profile=false; };
Options parse(int argc,char**argv){
    Options o;
    for(int i=1;i<argc;++i){
        std::string k=argv[i];
        if(k=="--profile"){o.profile=true;continue;}
        if(k=="--help") { std::cout<<"--logN 20 --batch N --repeat 1..10000 --trials 1..31 --only linear|xor --profile\n"; std::exit(0); }
        if(k!="--logN"&&k!="--batch"&&k!="--repeat"&&k!="--trials"&&k!="--only")throw std::invalid_argument("unknown option: "+k);
        if(++i==argc)throw std::invalid_argument("missing option value");
        if(k=="--only")o.only=argv[i];
        else {auto v=integer(argv[i]); if(k=="--logN")o.log=v; if(k=="--batch")o.batch=v; if(k=="--repeat")o.repeat=v; if(k=="--trials")o.trials=v;}
    }
    if(o.log!=20||!o.batch||o.batch>256||!o.repeat||o.repeat>10000||!o.trials||o.trials>31||(o.only!="both"&&o.only!="linear"&&o.only!="xor"))throw std::invalid_argument("invalid bounded experiment configuration");
    if(o.profile&&o.only=="both")throw std::invalid_argument("--profile requires --only linear or xor");
    // Conservative limit for device input/output/two plan workspaces and CPU reference arrays.
    if((size_t(1)<<o.log)*o.batch*sizeof(V)>512ULL*1024*1024)throw std::invalid_argument("working set exceeds 512 MiB per array");
    return o;
}
void cpu_fft(std::vector<V>& v,const std::vector<V>& roots,unsigned log){
    size_t reversed=0;
    for(size_t i=1;i<v.size();++i){auto bit=v.size()/2;while(reversed&bit){reversed^=bit;bit>>=1;}reversed^=bit;if(i<reversed)std::swap(v[i],v[reversed]);}
    for(unsigned stage=0;stage<log;++stage){const auto half=size_t(1)<<stage;
        for(size_t base=0;base<v.size();base+=2*half)for(size_t j=0;j<half;++j){
            const V a=v[base+j],b=v[base+j+half],w=roots[half-1+j];
            const float vr=b.real*w.real-b.imag*w.imag,vi=b.real*w.imag+b.imag*w.real;
            v[base+j]={a.real+vr,a.imag+vi};v[base+j+half]={a.real-vr,a.imag-vi};
        }
    }
}
Json error(const std::vector<V>& a,const std::vector<V>& b){
    double maxabs=0;long double e2=0,r2=0;size_t nonfinite=0;
    for(size_t i=0;i<a.size();++i){
        if(!std::isfinite(a[i].real)||!std::isfinite(a[i].imag)||!std::isfinite(b[i].real)||!std::isfinite(b[i].imag)){++nonfinite;continue;}
        double e=std::hypot(double(a[i].real)-b[i].real,double(a[i].imag)-b[i].imag);
        maxabs=std::max(maxabs,e);e2+=static_cast<long double>(e)*e;r2+=static_cast<long double>(b[i].real)*b[i].real+static_cast<long double>(b[i].imag)*b[i].imag;
    }
    return {{"max_abs",maxabs},{"relative_l2",double(std::sqrt(e2/std::max(r2,1e-300L)))},{"nonfinite",nonfinite}};
}
int main(int argc,char**argv){try{
    const auto o=parse(argc,argv); // All CLI rejects occur before any CUDA call.
    const size_t n=size_t(1)<<o.log,count=n*o.batch;
    std::vector<V> host(count),reference(count),linear(count),actual(count),roots(n-1);
    std::mt19937 rng(1237);std::uniform_real_distribution<float>d(-1,1);
    for(auto&v:host)v={d(rng),d(rng)};
    for(unsigned s=0;s<o.log;++s)for(unsigned k=0;k<(1U<<s);++k){double a=-3.14159265358979323846*k/(1U<<s);roots[(1U<<s)-1+k]={float(std::cos(a)),float(std::sin(a))};}
    for(unsigned b=0;b<o.batch;++b){std::vector<V> v(host.begin()+b*n,host.begin()+(b+1)*n);cpu_fft(v,roots,o.log);std::copy(v.begin(),v.end(),reference.begin()+b*n);}
    Stream stream; Buffer input(count),output(count);check(cudaMemcpy(input.p,host.data(),count*sizeof(V),cudaMemcpyHostToDevice));
    std::vector<std::unique_ptr<cuntt::ButterflyPlan>> plans;
    Json result={{"experiment","prefix-shared-staging-bank-map"},{"global_boundary_layout_fixed",true},{"logN",o.log},{"batch",o.batch},{"seed",1237},{"operator","fft"},{"precision","fp32"},{"inverse",false},{"normalize_inverse",false},{"batch_stride",n},{"element_stride",1},{"placement","out-of-place"},{"capture_calls",o.repeat},{"warmup_graph_launches",10},{"trials",o.trials},{"only",o.only},{"profile",o.profile},{"timing","CUDA events around one graph of capture_calls complete calls; ms divided by capture_calls"},{"cpu_reference","FP32 storage, arithmetic and coefficient table; unnormalized forward; all batches"},{"runtime_fingerprint",cubutterfly::runtime_fingerprint()},{"cases",Json::array()}};
    bool valid=true,identical=false;
    for(unsigned k=0;k<2;++k){
        cuntt::ButterflyConfig c;c.op=cuntt::ButterflyOperator::Fft;c.backend=cuntt::ButterflyBackend::OnlineReorder;c.fft_core=cuntt::FftCore::CufftDxBlock;
        c.precision=cuntt::ButterflyPrecision::Fp32;c.log_n=o.log;c.batch=o.batch;c.inverse=false;c.normalize_inverse=false;c.stage_partition={8,o.log-8};
        c.prefix_threads=c.suffix_threads=256;c.prefix_ept=c.suffix_ept=16;c.cross_twiddle=cuntt::CrossTwiddleMode::Recurrence;c.direct_boundary=cuntt::DirectBoundary::TiledTranspose;c.stage_overlap=false;c.batch_tile_count=1;
        c.shared_layout=k?cuntt::SharedLayout::XorSwizzle:cuntt::SharedLayout::Linear;
        plans.push_back(std::make_unique<cuntt::ButterflyPlan>(c));auto&p=*plans.back();p.set_stream(stream.s);
        p.execute_async(input.p,output.p);check(cudaStreamSynchronize(stream.s));check(cudaMemcpy(actual.data(),output.p,count*sizeof(V),cudaMemcpyDeviceToHost));
        auto cpu=error(actual,reference);valid=valid&&cpu["nonfinite"].get<size_t>()==0;
        if(!k)linear=actual;else identical=std::memcmp(linear.data(),actual.data(),count*sizeof(V))==0;
        result["cases"].push_back({{"mode",k?"xor":"linear"},{"mapping_json",Json::parse(cubutterfly::serialize_mapping(p.config()))},{"execution_groups",Json::parse(cubutterfly::execution_groups_json(p.dataflow_plan()))},{"cpu_error",cpu},{"linear_output_error",error(actual,linear)},{"verified_batches",o.batch},{"trials_ms",Json::array()}});
    }
    result["byte_identical_outputs"]=identical;result["strict_control_valid"]=valid&&identical;
    if(!valid||!identical){result["timing_skipped"]="nonfinite values or nonidentical organization outputs";std::cout<<result.dump(2)<<'\n';return 2;}
    std::vector<unsigned> selected;for(unsigned k=0;k<2;++k)if(o.only=="both"||o.only==(k?"xor":"linear"))selected.push_back(k);
    if(o.profile){ // Validation is outside the profiler interval; exactly one complete ordinary call inside.
        check(cudaStreamSynchronize(stream.s));check(cudaProfilerStart());plans[selected[0]]->execute_async(input.p,output.p);check(cudaStreamSynchronize(stream.s));check(cudaProfilerStop());
        result["profile_complete_calls"]=1;result["ordinary_timing_performed"]=false;
    }else{
        std::unique_ptr<Graph> graphs[2];
        for(auto k:selected){graphs[k]=std::make_unique<Graph>();auto&g=*graphs[k];
            check(cudaStreamBeginCapture(stream.s,cudaStreamCaptureModeThreadLocal));for(unsigned i=0;i<o.repeat;++i)plans[k]->execute_async(input.p,output.p);
            check(cudaStreamEndCapture(stream.s,&g.g));check(cudaGraphInstantiateWithFlags(&g.e,g.g,0));check(cudaGraphGetNodes(g.g,nullptr,&g.nodes));result["cases"][k]["graph_nodes"]=g.nodes;
        }
        for(unsigned w=0;w<10;++w)for(auto k:selected)check(cudaGraphLaunch(graphs[k]->e,stream.s));check(cudaStreamSynchronize(stream.s));
        Event start,stop;
        for(unsigned t=0;t<o.trials;++t)for(unsigned offset=0;offset<selected.size();++offset){auto k=selected[(t+offset)%selected.size()];
            check(cudaEventRecord(start.e,stream.s));check(cudaGraphLaunch(graphs[k]->e,stream.s));check(cudaEventRecord(stop.e,stream.s));check(cudaEventSynchronize(stop.e));float ms=0;check(cudaEventElapsedTime(&ms,start.e,stop.e));result["cases"][k]["trials_ms"].push_back(ms/o.repeat);
        }
        for(auto k:selected){auto v=result["cases"][k]["trials_ms"].get<std::vector<double>>();std::sort(v.begin(),v.end());result["cases"][k]["median_ms"]=(v[(v.size()-1)/2]+v[v.size()/2])/2;}
    }
    std::cout<<result.dump(2)<<'\n';return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
