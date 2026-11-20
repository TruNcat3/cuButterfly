// Bounded Figure 11 mechanism control. This is not a vendor-library benchmark.
// Vendor arithmetic functions are included unchanged from pinned source trees.
// All organizations instantiate cuButterfly's existing shared_iterative kernel.
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include "ntt_traits.cuh"
#include "shared_iterative.cuh"
#include "gpuntt/ntt_merge/ntt.cuh"
#include "butterfly.cuh"

using Word = std::uint64_t;
namespace fs = std::filesystem;
#define CHECK(call) do { auto e=(call); if(e!=cudaSuccess) throw std::runtime_error(std::string(#call)+": "+cudaGetErrorString(e)); } while(0)

Word mul_cpu(Word a, Word b, Word q) { return static_cast<Word>((__uint128_t(a)*b)%q); }
Word power_cpu(Word a, Word e, Word q) {
    Word r=1; for(;e;e>>=1,a=mul_cpu(a,a,q)) if(e&1) r=mul_cpu(r,a,q); return r;
}
bool prime_cpu(Word q) {
    if(q<2) return false;
    for(Word p: {2ULL,3ULL,5ULL,7ULL,11ULL,13ULL,17ULL,19ULL,23ULL,29ULL,31ULL,37ULL})
        if(q%p==0) return q==p;
    Word d=q-1; unsigned s=0; while(!(d&1)){d>>=1;++s;}
    for(Word a: {2ULL,325ULL,9375ULL,28178ULL,450775ULL,9780504ULL,1795265022ULL}) {
        if(a%q==0) continue;
        Word x=power_cpu(a%q,d,q); if(x==1||x==q-1) continue;
        bool ok=false; for(unsigned j=1;j<s;++j){x=mul_cpu(x,x,q); if(x==q-1){ok=true;break;}}
        if(!ok) return false;
    }
    return true;
}
unsigned reverse_cpu(unsigned x, unsigned logn) {
    unsigned y=0; for(unsigned i=0;i<logn;++i){y=(y<<1)|(x&1);x>>=1;} return y;
}
void reference(std::vector<Word>& x, unsigned logn, unsigned batch, Word root, Word q) {
    const unsigned n=1U<<logn;
    for(unsigned b=0;b<batch;++b) {
        Word* a=x.data()+std::size_t(b)*n;
        for(unsigned i=0;i<n;++i){auto j=reverse_cpu(i,logn);if(i<j)std::swap(a[i],a[j]);}
        for(unsigned stage=0;stage<logn;++stage) {
            unsigned h=1U<<stage;
            Word step=power_cpu(root,n/(2*h),q);
            for(unsigned base=0;base<n;base+=2*h) {
                Word w=1;
                for(unsigned j=0;j<h;++j,w=mul_cpu(w,step,q)) {
                    Word u=a[base+j],v=mul_cpu(a[base+j+h],w,q);
                    Word sum=u+v; a[base+j]=sum>=q?sum-q:sum;
                    a[base+j+h]=u>=v?u-v:q+u-v;
                }
            }
        }
    }
}
void save_words(const fs::path& p,const std::vector<Word>& x) {
    std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<const char*>(x.data()),x.size()*sizeof(Word));
    if(!f)throw std::runtime_error("write failed: "+p.string());
}
std::vector<Word> load_words(const fs::path& p,std::size_t n) {
    if(fs::file_size(p)!=n*sizeof(Word))throw std::runtime_error("size mismatch: "+p.string());
    std::vector<Word>x(n);std::ifstream f(p,std::ios::binary);f.read(reinterpret_cast<char*>(x.data()),n*sizeof(Word));
    if(!f)throw std::runtime_error("read failed: "+p.string());return x;
}
void prepare(const fs::path& dir,unsigned logn,unsigned batch,Word q) {
    if(logn!=16||batch!=16)throw std::runtime_error("frozen mechanism request requires logN=16, B=16");
    if(!prime_cpu(q)||q>=(Word(1)<<60)||(q-1)%(1U<<logn))throw std::runtime_error("invalid shared field");
    const unsigned n=1U<<logn; Word root=0;
    for(Word a=2;a<1000;++a){Word w=power_cpu(a,(q-1)/n,q);if(power_cpu(w,n,q)==1&&power_cpu(w,n/2,q)!=1){root=w;break;}}
    if(!root)throw std::runtime_error("primitive root not found");
    // Independently validate the host FFT-style reference by direct evaluation.
    std::vector<Word> small(16);for(unsigned j=0;j<16;++j)small[j]=(j%3==0)?q-1:j*j+1;
    auto fast=small;Word smallroot=power_cpu(root,n/16,q);reference(fast,4,1,smallroot,q);
    for(unsigned k=0;k<16;++k){Word v=0;for(unsigned j=0;j<16;++j)v=(v+mul_cpu(small[j],power_cpu(smallroot,j*k,q),q))%q;
        if(v!=fast[k])throw std::runtime_error("direct CPU oracle disagrees");}
    std::vector<Word>input(std::size_t(n)*batch),tw(n-1),sh(n-1);Word seed=20261118;
    for(auto&v:input){seed^=seed<<13;seed^=seed>>7;seed^=seed<<17;v=seed%q;}
    for(unsigned j=0;j<64;++j)input[j]=j%3==0?q-1:(j%3==1?0:1);
    for(unsigned s=0;s<logn;++s){unsigned h=1U<<s;Word step=power_cpu(root,n/(2*h),q),w=1;
        for(unsigned j=0;j<h;++j,w=mul_cpu(w,step,q)){tw[h-1+j]=w;sh[h-1+j]=Word((__uint128_t(w)<<64)/q);}}
    auto ref=input;reference(ref,logn,batch,root,q);
    fs::create_directories(dir);save_words(dir/"params.u64",{logn,batch,q,root,20261118});
    save_words(dir/"input.u64",input);save_words(dir/"reference.u64",ref);
    save_words(dir/"twiddles.u64",tw);save_words(dir/"shoup.u64",sh);
    std::ofstream out(dir/"metadata.json");
    out<<"{\"schema\":\"ntt-fixed-core-input-v1\",\"log_n\":"<<logn<<",\"N\":"<<n<<",\"batch\":"<<batch
       <<",\"modulus\":"<<q<<",\"root\":"<<root<<",\"seed\":20261118,\"word_bits\":64,"
       <<"\"direction\":\"forward\",\"ring\":\"cyclic\",\"input_order\":\"natural\",\"output_order\":\"natural\","
       <<"\"normalization\":\"none\",\"placement\":\"out-of-place\",\"reference\":\"CPU uint128 exact modular DIT; direct N16 check\","
       <<"\"prime_verified\":true,\"root_order_verified\":true,\"cpu_direct_check\":true,\"gpu_used\":false}\n";
    std::cout<<"CPU preparation complete: "<<dir<<"\n";
}

template<int Core>struct Arithmetic {
    using Value=Word;static constexpr bool kBitReverseInput=true;
    const Word* tw;const Word* sh;Word q;gpuntt::Modulus<Word> modulus;
    __device__ __forceinline__ void apply(unsigned stage,unsigned offset,Word& a,Word& b)const {
        const auto i=(1U<<stage)-1+offset;
        if constexpr(Core==0) {
            cuntt::traits::NttStageOperatorT<Word> op{tw,sh,q,1,0};
            op.apply_coefficient(tw[i],sh[i],a,b);
        } else if constexpr(Core==1) {
            gpuntt::CooleyTukeyUnit<Word>(a,b,tw[i],modulus);
        } else {
            phantom::arith::ct_butterfly(a,b,tw[i],sh[i],q);
            // Common canonical stage representation. Same wrapper in both organizations.
            if(a>=2*q)a-=2*q;if(a>=q)a-=q;
            if(b>=2*q)b-=2*q;if(b>=q)b-=q;
        }
    }
    __device__ __forceinline__ Word finalize(Word v,unsigned)const{return v;}
};

struct Resource {unsigned first,stages,blocks,threads,shared;int registers,local,active_blocks;};
template<int Core>Resource resource(unsigned first,unsigned stages,unsigned batch) {
    cudaFuncAttributes attr{};
    auto kernel=cuntt::detail::shared_iterative_kernel<Arithmetic<Core>,16,-1,-1,1,0>;
    CHECK(cudaFuncGetAttributes(&attr,kernel));
    int active=0;unsigned shared=2*(1U<<stages)*sizeof(Word);
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,kernel,128,shared));
    return{first,stages,batch*(1U<<(16-stages)),128,shared,attr.numRegs,int(attr.localSizeBytes),active};
}
template<int Core>void launch(const Word* input,Word* output,Arithmetic<Core> op,unsigned first,unsigned stages,
        unsigned batch,bool trans_in,bool trans_out,unsigned local_log,cudaStream_t stream) {
    const unsigned n=1U<<16;
    cuntt::detail::shared_iterative_kernel<Arithmetic<Core>,16,-1,-1,1,0>
      <<<batch*(n>>stages),128,2*(1U<<stages)*sizeof(Word),stream>>>
      (input,output,16,first,stages,n,1,true,batch,1,op,false,trans_in,trans_out,local_log,16-local_log);
    CHECK(cudaGetLastError());
}
struct DeviceWords {Word* p=nullptr;explicit DeviceWords(std::size_t n){CHECK(cudaMalloc(&p,n*sizeof(Word)));}
    ~DeviceWords(){if(p)cudaFree(p);}void put(const std::vector<Word>&x){CHECK(cudaMemcpy(p,x.data(),x.size()*sizeof(Word),cudaMemcpyHostToDevice));}};

template<int Core>void measure(const fs::path& prepared,const fs::path& path,unsigned local_log,bool resident,
        unsigned warmup,unsigned repeat,unsigned samples,bool profile) {
    auto params=load_words(prepared/"params.u64",5);
    if(params[0]!=16||params[1]!=16||(local_log!=8&&local_log!=10))throw std::runtime_error("unexpected frozen request");
    unsigned n=1U<<params[0],batch=params[1];Word q=params[2];
    auto input=load_words(prepared/"input.u64",std::size_t(n)*batch),ref=load_words(prepared/"reference.u64",input.size());
    auto tw=load_words(prepared/"twiddles.u64",n-1),sh=load_words(prepared/"shoup.u64",n-1);
    DeviceWords din(input.size()),dout(input.size()),tmp(input.size()),mid(input.size()),dtw(tw.size()),dsh(sh.size());
    din.put(input);dtw.put(tw);dsh.put(sh);
    Arithmetic<Core>op{dtw.p,dsh.p,q,gpuntt::Modulus<Word>(q)};
    cudaDeviceProp prop{};CHECK(cudaGetDeviceProperties(&prop,0));
    if(std::string(prop.name).find("A100")==std::string::npos)throw std::runtime_error("A100-only evidence");
    cudaStream_t stream;CHECK(cudaStreamCreate(&stream));
    cudaGraph_t graph;cudaGraphExec_t exec;
    CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
    if(resident)launch<Core>(din.p,tmp.p,op,0,local_log,batch,false,true,local_log,stream);
    else {
        launch<Core>(din.p,mid.p,op,0,local_log/2,batch,false,false,local_log,stream);
        launch<Core>(mid.p,tmp.p,op,local_log/2,local_log/2,batch,false,true,local_log,stream);
    }
    launch<Core>(tmp.p,dout.p,op,local_log,16-local_log,batch,true,false,local_log,stream);
    CHECK(cudaStreamEndCapture(stream,&graph));CHECK(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    std::size_t node_count=0;CHECK(cudaGraphGetNodes(graph,nullptr,&node_count));
    if(node_count!=(resident?2:3))throw std::runtime_error("unexpected graph node count");
    CHECK(cudaGraphLaunch(exec,stream));CHECK(cudaStreamSynchronize(stream));
    std::vector<Word>actual(input.size());CHECK(cudaMemcpy(actual.data(),dout.p,actual.size()*sizeof(Word),cudaMemcpyDeviceToHost));
    std::size_t mismatches=0;for(std::size_t i=0;i<actual.size();++i)mismatches+=actual[i]!=ref[i];
    if(mismatches) {
        std::ofstream out(path);out<<"{\"correctness_exact_mismatches\":"<<mismatches<<",\"samples_us\":[],\"status\":\"correctness-failed\"}\n";
        for(std::size_t i=0,count=0;i<actual.size()&&count<6;++i)if(actual[i]!=ref[i]){std::cerr<<i<<": "<<actual[i]<<" expected "<<ref[i]<<"\n";++count;}
        throw std::runtime_error("exact check failed");
    }
    for(unsigned i=0;i<warmup;++i)CHECK(cudaGraphLaunch(exec,stream));CHECK(cudaStreamSynchronize(stream));
    std::vector<double>times;
    if(profile) {
        CHECK(cudaProfilerStart());CHECK(cudaGraphLaunch(exec,stream));CHECK(cudaStreamSynchronize(stream));CHECK(cudaProfilerStop());
    }else {
        cudaEvent_t start,stop;CHECK(cudaEventCreate(&start));CHECK(cudaEventCreate(&stop));
        for(unsigned s=0;s<samples;++s){CHECK(cudaEventRecord(start,stream));for(unsigned r=0;r<repeat;++r)CHECK(cudaGraphLaunch(exec,stream));
            CHECK(cudaEventRecord(stop,stream));CHECK(cudaEventSynchronize(stop));float ms;CHECK(cudaEventElapsedTime(&ms,start,stop));times.push_back(ms*1000.0/repeat);}
        CHECK(cudaEventDestroy(start));CHECK(cudaEventDestroy(stop));
    }
    CHECK(cudaMemcpy(actual.data(),dout.p,actual.size()*sizeof(Word),cudaMemcpyDeviceToHost));
    for(std::size_t i=0;i<actual.size();++i)mismatches+=actual[i]!=ref[i];
    std::vector<Resource>resources;
    if(resident)resources.push_back(resource<Core>(0,local_log,batch));
    else{resources.push_back(resource<Core>(0,local_log/2,batch));resources.push_back(resource<Core>(local_log/2,local_log/2,batch));}
    resources.push_back(resource<Core>(local_log,16-local_log,batch));
    std::ofstream out(path);out<<std::setprecision(12);
    out<<"{\"schema\":\"ntt-fixed-core-measurement-v1\",\"status\":\"measured\",\"core\":\""<<(Core==0?"native":Core==1?"gpuntt":"phantom")
       <<"\",\"organization\":\""<<(resident?"resident-prefix":"split-prefix")<<"\",\"local_log\":"<<local_log<<",\"N\":"<<n<<",\"batch\":"<<batch
       <<",\"modulus\":"<<q<<",\"root\":"<<params[3]<<",\"gpu_name\":\""<<prop.name<<"\",\"gpu_total_bytes\":"<<prop.totalGlobalMem
       <<",\"correctness_exact_mismatches\":"<<mismatches<<",\"checked_values\":"<<actual.size()<<",\"kernel_count\":"<<node_count
       <<",\"warmup\":"<<warmup<<",\"repeat\":"<<repeat<<",\"profile_only\":"<<(profile?"true":"false")<<",\"samples_us\":[";
    for(unsigned i=0;i<times.size();++i){if(i)out<<',';out<<times[i];}out<<"],\"median_us\":";
    if(times.empty())out<<"null";else{auto sorted=times;std::sort(sorted.begin(),sorted.end());out<<sorted[sorted.size()/2];}
    out<<",\"kernels\":[";for(unsigned i=0;i<resources.size();++i){if(i)out<<',';const auto&r=resources[i];
        out<<"{\"first_stage\":"<<r.first<<",\"stages\":"<<r.stages<<",\"blocks\":"<<r.blocks<<",\"threads\":"<<r.threads
           <<",\"shared_bytes\":"<<r.shared<<",\"registers_per_thread\":"<<r.registers<<",\"local_bytes_per_thread\":"<<r.local
           <<",\"theoretical_active_blocks_per_sm\":"<<r.active_blocks<<'}';}out<<"]}\n";
    CHECK(cudaGraphExecDestroy(exec));CHECK(cudaGraphDestroy(graph));CHECK(cudaStreamDestroy(stream));
    if(mismatches)throw std::runtime_error("post-timing exact check failed");
    std::cout<<"exact="<<actual.size()<<"/"<<actual.size()<<" graph_nodes="<<node_count<<" result="<<path<<"\n";
}
int main(int argc,char**argv) {
    try {
        std::map<std::string,std::string>a;for(int i=1;i<argc;++i){std::string k=argv[i];if(k=="--profile-one-call")a[k]="1";
            else{if(i+1==argc)throw std::runtime_error("missing value "+k);a[k]=argv[++i];}}
        auto val=[&](const std::string&k,const std::string&d){auto it=a.find(k);return it==a.end()?d:it->second;};
        if(a.count("--prepare")){prepare(a.at("--prepare"),std::stoul(val("--log-n","16")),std::stoul(val("--batch","16")),std::stoull(val("--modulus","576460756061519873")));return 0;}
        fs::path dir=a.at("--prepared"),out=a.at("--output");fs::create_directories(out.parent_path());
        auto core=a.at("--core"),org=a.at("--organization");if(org!="split-prefix"&&org!="resident-prefix")throw std::runtime_error("unknown organization");
        unsigned local=std::stoul(a.at("--local-log")),warmup=std::stoul(val("--warmup","100")),repeat=std::stoul(val("--repeat","100")),samples=std::stoul(val("--samples","7"));
        if(!repeat||!samples)throw std::runtime_error("invalid measurement count");bool resident=org=="resident-prefix",profile=a.count("--profile-one-call");
        if(core=="native")measure<0>(dir,out,local,resident,warmup,repeat,samples,profile);
        else if(core=="gpuntt")measure<1>(dir,out,local,resident,warmup,repeat,samples,profile);
        else if(core=="phantom")measure<2>(dir,out,local,resident,warmup,repeat,samples,profile);
        else throw std::runtime_error("unknown core");
        return 0;
    }catch(const std::exception&e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;}
}
