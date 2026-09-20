#include <cubutterfly/mapping.hpp>
#include <cuda.h>
#include <nlohmann/json.hpp>
#include "stage_probe.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <type_traits>

using Json = nlohmann::json;
using namespace cuntt;

namespace {
void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
void check(CUresult error) {
    if(error!=CUDA_SUCCESS) {
        const char* message=nullptr;
        cuGetErrorString(error,&message);
        throw std::runtime_error(message ? message : "CUDA driver query failed");
    }
}
struct Buffer {
    void* data=nullptr;
    explicit Buffer(std::size_t bytes) { if(bytes) check(cudaMalloc(&data,bytes)); }
    ~Buffer() { if(data) cudaFree(data); }
    Buffer(const Buffer&)=delete;
};
struct Stream {
    cudaStream_t value=nullptr;
    Stream() { check(cudaStreamCreateWithFlags(&value,cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(value); }
};
struct Event {
    cudaEvent_t value=nullptr;
    Event() { check(cudaEventCreate(&value)); }
    ~Event() { cudaEventDestroy(value); }
};
struct Protocol { unsigned warmup=10, repeat=20, trials=3; bool describe=false; double warmup_ms=0; };

Json hardware() {
    const auto h=query_hardware_resource_model();
    return {{"device",h.device_name},{"sm_count",h.sm_count},{"global_memory_bytes",h.memory_bytes},
        {"compute_capability",std::to_string(h.compute_major)+"."+std::to_string(h.compute_minor)},
        {"max_blocks_per_sm",h.max_blocks_per_sm},{"max_threads_per_sm",h.max_threads_per_sm},
        {"registers_per_sm",h.registers_per_sm},{"shared_bytes_per_sm",h.shared_bytes_per_sm}};
}

std::vector<double> measure(const std::function<void()>& launch,const std::function<void()>& reset,
                            cudaStream_t stream,const Protocol& p,double& warmup_ms) {
    Event start,stop;
    reset();
    for(unsigned i=0;i<p.warmup;++i) launch();
    check(cudaStreamSynchronize(stream));
    warmup_ms=0;
    while(warmup_ms<p.warmup_ms) {
        // Accumulate device time, not compilation, initialization or host sleep.
        reset();
        check(cudaEventRecord(start.value,stream));
        for(unsigned i=0;i<100;++i) launch();
        check(cudaEventRecord(stop.value,stream)); check(cudaEventSynchronize(stop.value));
        float ms=0; check(cudaEventElapsedTime(&ms,start.value,stop.value));
        warmup_ms+=ms;
    }
    std::vector<double> trials;
    for(unsigned t=0;t<p.trials;++t) {
        // Match Plan::execute's direct-launch throughput protocol. Per-launch
        // synchronization drains the queue and adds event/host submission
        // latency to every small kernel. Reset between trials, outside the
        // interval; correctness is checked separately before any timing.
        reset();
        check(cudaEventRecord(start.value,stream));
        for(unsigned r=0;r<p.repeat;++r) launch();
        check(cudaEventRecord(stop.value,stream)); check(cudaEventSynchronize(stop.value));
        float ms=0; check(cudaEventElapsedTime(&ms,start.value,stop.value));
        trials.push_back(ms/p.repeat);
    }
    return trials;
}

// Capture only discovers actual launch attributes. Timings always invoke the
// original adapter directly; a CUDA graph is never the measured execution path.
void inspect(Json& group,const std::function<void()>& launch,cudaStream_t stream) {
    cudaGraph_t graph=nullptr;
    bool capturing=false;
    try {
        check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal)); capturing=true;
        launch();
        const auto status=cudaStreamEndCapture(stream,&graph); capturing=false; check(status);
        std::size_t count=0; check(cudaGraphGetNodes(graph,nullptr,&count));
        std::vector<cudaGraphNode_t> nodes(count); check(cudaGraphGetNodes(graph,nodes.data(),&count));
        Json kernels=Json::array();
        for(auto node:nodes) {
            cudaGraphNodeType type; check(cudaGraphNodeGetType(node,&type));
            if(type!=cudaGraphNodeTypeKernel) continue;
            // The driver handle is valid for both linked kernels and kernels
            // registered by a JIT-loaded shared object's own CUDA runtime.
            CUDA_KERNEL_NODE_PARAMS params{}; check(cuGraphKernelNodeGetParams(node,&params));
            Json kernel={{"grid_ctas",std::uint64_t(params.gridDimX)*params.gridDimY*params.gridDimZ},
                {"threads",params.blockDimX*params.blockDimY*params.blockDimZ},
                {"dynamic_shared_bytes",params.sharedMemBytes}};
            int registers=0,local_bytes=0,shared_bytes=0;
            if(cuFuncGetAttribute(&registers,CU_FUNC_ATTRIBUTE_NUM_REGS,params.func)==CUDA_SUCCESS &&
               cuFuncGetAttribute(&local_bytes,CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,params.func)==CUDA_SUCCESS &&
               cuFuncGetAttribute(&shared_bytes,CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,params.func)==CUDA_SUCCESS) {
                kernel["compiler_registers_per_thread"]=registers;
                kernel["compiler_local_bytes_per_thread"]=local_bytes;
                kernel["live_shared_bytes"]=shared_bytes+params.sharedMemBytes;
                kernel["compiler_resources_known"]=true;
                kernel["compiler_local_resources_known"]=true;
            } else { cudaGetLastError(); kernel["compiler_resources_known"]=false; }
            kernels.push_back(kernel);
        }
        group["actual_kernels"]=kernels;
        group["kernel_launch_count"]=kernels.size();
        if(kernels.empty()) {
            group["independent"]=false;
            group["reason"]="selected physical group submitted no kernel";
        }
        if(kernels.size()==1) {
            for(auto it=kernels[0].begin();it!=kernels[0].end();++it) {
                // A driver-loaded module already has authoritative attributes;
                // an unsupported runtime query must not erase those attributes.
                if(it.key()=="compiler_resources_known" && !it.value().get<bool>() &&
                   group.value("compiler_resources_known",false)) continue;
                group[it.key()]=it.value();
            }
            group["resource_source"]=group.value("compiler_resources_known",false) ?
                "actual-kernel" : "unknown";
            group["work_blocks"]=group["grid_ctas"];
        } else {
            group["resource_source"]="multiple-physical-launches";
            group["compiler_resources_known"]=false;
        }
        cudaGraphDestroy(graph);
    } catch(const std::exception& e) {
        if(capturing) cudaStreamEndCapture(stream,&graph);
        if(graph) cudaGraphDestroy(graph);
        cudaGetLastError(); group["resource_query_error"]=e.what();
        group["independent"]=false;
        group["reason"]="physical launch inspection failed";
    }
}

template<class Value> double distance(Value a,Value b) {
    return std::abs(static_cast<double>(a)-static_cast<double>(b));
}
template<> double distance(Complex32 a,Complex32 b) { return std::hypot(double(a.real)-b.real,double(a.imag)-b.imag); }
template<> double distance(Complex64 a,Complex64 b) { return std::hypot(a.real-b.real,a.imag-b.imag); }
template<class Value> Value initial(std::size_t i) {
    if constexpr(std::is_integral_v<Value>) return static_cast<Value>((i*17+3)%101);
    else return static_cast<Value>((int((i*17+3)%101)-50)/101.0);
}
template<> Complex32 initial(std::size_t i) { return {initial<float>(i),initial<float>(i+7)}; }
template<> Complex64 initial(std::size_t i) { return {initial<double>(i),initial<double>(i+7)}; }

template<class PlanType,class Value,class Reference>
Json execute_probe(PlanType& plan,Json point,const Protocol& p,Reference reference,
                   std::size_t n,std::size_t batch,std::size_t stride,std::size_t element_stride,bool inplace) {
    Stream stream;
    plan.set_stream(stream.value);
    detail::StageProbeAdapter adapter(plan);
    const auto bytes=plan.data_size(),elements=bytes/sizeof(Value);
    std::size_t free_bytes=0,total_bytes=0; check(cudaMemGetInfo(&free_bytes,&total_bytes));
    const auto count=adapter.groups().size();
    if((bytes && (count+4)>free_bytes/bytes) ||
       plan.workspace_size()>free_bytes-(count+4)*bytes)
        throw std::runtime_error("stage probe endpoint buffers exceed currently available device memory");
    std::vector<std::unique_ptr<Buffer>> endpoints;
    for(std::size_t i=0;i<=count;++i) endpoints.push_back(std::make_unique<Buffer>(bytes));
    Buffer input_seed(bytes),full_input(bytes),full_output(bytes),workspace(plan.workspace_size());
    auto physical=plan.dataflow_plan();
    physical.execution_groups.clear();
    for(const auto& entry:adapter.groups()) physical.execution_groups.push_back(entry.execution);
    Json groups=Json::parse(cubutterfly::execution_groups_json(physical));
    bool all_independent=count>0;
    for(std::size_t i=0;i<count;++i) {
        auto& group=groups[i]; const auto& entry=adapter.groups()[i];
        group["index"]=i; group["exchange"]=static_cast<unsigned>(entry.execution.exchange);
        group["independent"]=entry.independent; group["reason"]=entry.reason;
        group["in_place"]=entry.in_place || (inplace && count==1);
        group["work_blocks"]=entry.execution.grid_ctas;
        if(entry.independent) inspect(group,[&] {
            adapter.launch_group(i,group["in_place"].get<bool>() ? endpoints[i+1]->data : endpoints[i]->data,endpoints[i+1]->data,
                                 workspace.data,stream.value);
        },stream.value);
        all_independent &= group["independent"].get<bool>();
    }
    const auto normalized=cubutterfly::serialize_mapping(plan.config());
    const auto mapping=Json::parse(normalized);
    for(auto it=mapping.begin();it!=mapping.end();++it)
        point[it.key()]=it.value();
    point["direction"]=plan.config().inverse ? "inverse" : "forward";
    point["inverse"]=plan.config().inverse;
    if constexpr(std::is_same_v<PlanType,Plan>) {
        point["modulus"]=plan.config().modulus;
        point["word_bits"]=plan.config().word_bits;
        point["input_order"]=input_order_name(plan.config().input_order);
        point["output_order"]=output_order_name(plan.config().output_order);
    } else {
        point["operator"]=butterfly_operator_name(plan.config().op);
        point["precision"]=butterfly_precision_name(plan.config().precision);
        point["accumulation"]=butterfly_accumulation_name(plan.config().accumulation);
        point["placement"]=butterfly_placement_name(plan.config().placement);
        point["normalization"]=plan.config().normalize_inverse ? "inverse" : "none";
    }
    point["mapping_json"]=normalized;
    point["runtime_fingerprint"]=cubutterfly::runtime_fingerprint();
    const auto device=hardware();
    for(const auto* key : {"device","compute_capability","global_memory_bytes"}) point[key]=device.at(key);
    point["execution_groups_json"]=groups.dump();
    point["N"]=n; point["batch_stride"]=stride; point["element_stride"]=element_stride;
    Json result={{"schema","cubutterfly-stage-probe-v1"},{"status","resolved"},{"sample",point},
        {"hardware",device},{"groups",groups},{"pairs",Json::array()},
        {"workspace_bytes",plan.workspace_size()},{"probe_payload_buffers",count+4},
        {"free_memory_bytes",free_bytes},
        {"protocol",{{"timing","direct-launch-batched-events-reset-between-trials-v2"},
         {"warmup",p.warmup},{"warmup_min_ms",p.warmup_ms},{"repeat",p.repeat},{"trials",p.trials}}}};
    if(p.describe) return result;
    if(!all_independent) { result["status"]="unavailable"; result["correct"]=false; return result; }
    std::vector<Value> host(elements),expected(elements),actual(elements);
    for(std::size_t i=0;i<elements;++i) host[i]=initial<Value>(i);
    expected=host;
    for(std::size_t b=0;b<batch;++b) {
        std::vector<Value> values(n);
        for(std::size_t i=0;i<n;++i) values[i]=host[b*stride+i*element_stride];
        reference(values);
        for(std::size_t i=0;i<n;++i) expected[b*stride+i*element_stride]=values[i];
    }
    check(cudaMemcpyAsync(input_seed.data,host.data(),bytes,cudaMemcpyHostToDevice,stream.value));
    auto reset_full=[&] { check(cudaMemcpyAsync(full_input.data,input_seed.data,bytes,cudaMemcpyDeviceToDevice,stream.value)); };
    auto full=[&] { adapter.launch_plan(full_input.data,inplace ? full_input.data : full_output.data,workspace.data,stream.value); };
    auto verify=[&](void* output) {
        check(cudaMemcpyAsync(actual.data(),output,bytes,cudaMemcpyDeviceToHost,stream.value));
        check(cudaStreamSynchronize(stream.value));
        double max_error=0,max_value=0;
        for(std::size_t b=0;b<batch;++b) for(std::size_t i=0;i<n;++i) {
            const auto k=b*stride+i*element_stride;
            if constexpr(std::is_integral_v<Value>) {
                if(actual[k]!=expected[k]) return false;
                continue;
            }
            const auto error=distance(actual[k],expected[k]);
            if(!std::isfinite(error)) return false;
            max_error=std::max(max_error,error); max_value=std::max(max_value,distance(expected[k],Value{}));
        }
        if constexpr(std::is_integral_v<Value>) return max_error==0;
        else return max_error<= (sizeof(Value)==sizeof(float) || std::is_same_v<Value,Complex32> ? 5e-4 : 1e-9)*std::max(1.0,max_value);
    };
    reset_full(); full();
    if(!verify(inplace ? full_input.data : full_output.data)) throw std::runtime_error("complete-plan reference verification failed");
    check(cudaMemcpyAsync(endpoints[0]->data,input_seed.data,bytes,cudaMemcpyDeviceToDevice,stream.value));
    for(std::size_t i=0;i<count;++i) {
        auto* destination=endpoints[i+1]->data;
        const bool alias=groups[i]["in_place"];
        if(alias) check(cudaMemcpyAsync(destination,endpoints[i]->data,bytes,cudaMemcpyDeviceToDevice,stream.value));
        adapter.launch_group(i,alias ? destination : endpoints[i]->data,destination,workspace.data,stream.value);
    }
    if(!verify(endpoints[count]->data))
        throw std::runtime_error("independent groups do not reproduce the reference transform");
    for(std::size_t i=0;i<count;++i) {
        const bool alias=groups[i]["in_place"];
        auto reset=[&] { if(alias) check(cudaMemcpyAsync(endpoints[i+1]->data,endpoints[i]->data,bytes,cudaMemcpyDeviceToDevice,stream.value)); };
        auto launch=[&] { adapter.launch_group(i,alias ? endpoints[i+1]->data : endpoints[i]->data,endpoints[i+1]->data,
                                               workspace.data,stream.value); };
        double warmup_ms=0;
        result["groups"][i]["trial_kernel_ms"]=measure(launch,reset,stream.value,p,warmup_ms);
        result["groups"][i]["warmup_ms"]=warmup_ms;
        result["groups"][i]["role"]="train";
    }
    for(std::size_t i=0;i+1<count;++i) {
        auto reset=[&] { check(cudaMemcpyAsync(full_input.data,endpoints[i]->data,bytes,cudaMemcpyDeviceToDevice,stream.value)); };
        auto launch=[&] {
            void* intermediate=groups[i]["in_place"].get<bool>() ? full_input.data : full_output.data;
            void* output=groups[i+1]["in_place"].get<bool>() ? intermediate :
                (intermediate==full_input.data ? full_output.data : full_input.data);
            adapter.launch_group(i,full_input.data,intermediate,workspace.data,stream.value);
            adapter.launch_group(i+1,intermediate,output,workspace.data,stream.value);
        };
        double warmup_ms=0;
        const auto trials=measure(launch,reset,stream.value,p,warmup_ms);
        result["pairs"].push_back({{"producer_group",i},{"consumer_group",i+1},{"mode","serial"},
            {"trial_kernel_ms",trials},{"warmup_ms",warmup_ms}});
    }
    double warmup_ms=0;
    result["plan_trial_kernel_ms"]=measure(full,reset_full,stream.value,p,warmup_ms);
    result["plan_warmup_ms"]=warmup_ms;
    result["status"]="measured"; result["correct"]=true;
    return result;
}

Json probe(Json point,const Protocol& p) {
    const auto mapping=point.at("mapping_json").get<std::string>();
    if(point.value("operator","fft")=="ntt") {
        PlanConfig c;
        c.log_n=point.at("logN"); c.batch=point.at("batch");
        c.word_bits=std::stoi(point.value("precision","word32").substr(4));
        c.modulus=point.value("modulus",kDefaultModulus);
        c.inverse=point.value("direction","forward")=="inverse";
        c.input_order=parse_input_order(point.value("input_order","natural"));
        auto placement=point.value("placement","natural");
        if(placement=="in-place" || placement=="out-of-place") placement="natural";
        c.output_order=parse_output_order(point.value("output_order",placement));
        cubutterfly::apply_serialized_mapping(mapping,c);
        Plan plan(c); c=plan.config();
        const auto n=std::size_t{1}<<c.log_n;
        auto reference=[&](auto& values) {
            const auto layout=(c.input_order==InputOrder::ApptStatic || c.output_order==OutputOrder::ApptStatic)
                ? plan.appt_layout_info() : ApptLayoutInfo{};
            std::vector<std::uint64_t> natural(n);
            for(std::size_t i=0;i<n;++i) natural[i]=values[c.input_order==InputOrder::ApptStatic ? appt_static_index(i,layout) : i];
            reference_ntt(natural,c.modulus,c.inverse);
            for(std::size_t i=0;i<n;++i) {
                auto index=i;
                if(c.output_order==OutputOrder::BitReversed) {
                    index=0;
                    for(unsigned bit=0;bit<c.log_n;++bit) index=(index<<1)|((i>>bit)&1U);
                } else if(c.output_order==OutputOrder::ApptStatic) index=appt_static_index(i,layout);
                values[index]=natural[i];
            }
        };
        if(c.word_bits==32) return execute_probe<Plan,std::uint32_t>(plan,point,p,reference,n,c.batch,n,1,false);
        return execute_probe<Plan,std::uint64_t>(plan,point,p,reference,n,c.batch,n,1,false);
    }
    ButterflyConfig c;
    c.op=parse_butterfly_operator(point.value("operator","fft"));
    c.precision=parse_butterfly_precision(point.value("precision","fp32"));
    c.accumulation=parse_butterfly_accumulation(point.value("accumulation","native"));
    c.placement=parse_butterfly_placement(point.value("placement","out-of-place"));
    c.log_n=point.at("logN"); c.batch=point.at("batch");
    c.inverse=point.value("direction","forward")=="inverse";
    c.normalize_inverse=point.value("normalization","inverse")!="none";
    c.element_stride=point.value("element_stride",std::size_t{1}); c.batch_stride=point.value("batch_stride",std::size_t{0});
    if(point.contains("stage_matrix")) {
        auto text=point.at("stage_matrix").get<std::string>(); std::replace(text.begin(),text.end(),',',' ');
        std::istringstream input(text); ButterflyMatrix2x2 matrix;
        if(!(input>>matrix.m00>>matrix.m01>>matrix.m10>>matrix.m11)) throw std::invalid_argument("invalid stage_matrix");
        c.stage_matrices.push_back(matrix);
    }
    cubutterfly::apply_serialized_mapping(mapping,c); ButterflyPlan plan(c); c=plan.config();
    const auto n=std::size_t{1}<<c.log_n; const bool inplace=c.placement==ButterflyPlacement::InPlace;
    auto real_reference=[&](auto& v) {
        if(c.op==ButterflyOperator::Structured2x2) reference_structured_2x2(v,c.stage_matrices,c.inverse);
        else reference_fwht(v,c.inverse,c.normalize_inverse);
    };
    if(c.op==ButterflyOperator::Fft) {
        auto reference=[&](auto& v) { reference_fft(v,c.inverse,c.normalize_inverse); };
        if(c.precision==ButterflyPrecision::Fp32) return execute_probe<ButterflyPlan,Complex32>(plan,point,p,reference,n,c.batch,c.batch_stride,c.element_stride,inplace);
        if(c.precision==ButterflyPrecision::Fp64) return execute_probe<ButterflyPlan,Complex64>(plan,point,p,reference,n,c.batch,c.batch_stride,c.element_stride,inplace);
    } else if(c.precision==ButterflyPrecision::Uint32) {
        return execute_probe<ButterflyPlan,std::uint32_t>(plan,point,p,[&](auto& v) {
            if(c.op==ButterflyOperator::SupersetZeta) reference_superset_zeta(v,c.inverse);
            else reference_subset_zeta(v,c.inverse);
        },n,c.batch,c.batch_stride,c.element_stride,inplace);
    } else {
        if(c.precision==ButterflyPrecision::Fp32) return execute_probe<ButterflyPlan,float>(plan,point,p,real_reference,n,c.batch,c.batch_stride,c.element_stride,inplace);
        if(c.precision==ButterflyPrecision::Fp64) return execute_probe<ButterflyPlan,double>(plan,point,p,real_reference,n,c.batch,c.batch_stride,c.element_stride,inplace);
    }
    // The plan was constructed successfully. A missing numeric reference is
    // an adapter coverage gap, not evidence that the design is unexecutable.
    auto groups=Json::parse(cubutterfly::execution_groups_json(plan.dataflow_plan()));
    const std::string reason="stage probe reference for this numeric regime is not available";
    for(auto& group:groups) { group["independent"]=false; group["reason"]=reason; }
    point["mapping_json"]=cubutterfly::serialize_mapping(c);
    point["runtime_fingerprint"]=cubutterfly::runtime_fingerprint();
    point["execution_groups_json"]=groups.dump();
    return {{"schema","cubutterfly-stage-probe-v1"},{"status",p.describe ? "resolved" : "unavailable"},
        {"correct",false},{"reason",reason},{"sample",point},{"groups",groups},
        {"hardware",hardware()},{"workspace_bytes",plan.workspace_size()}};
}
}

int main(int argc,char** argv) {
    try {
        Protocol protocol; Json point;
        for(int i=1;i<argc;++i) {
            const std::string arg=argv[i];
            if(arg=="--describe-only") protocol.describe=true;
            else if(i+1<argc && arg=="--point-json") point=Json::parse(argv[++i]);
            else if(i+1<argc && arg=="--warmup") protocol.warmup=std::stoul(argv[++i]);
            else if(i+1<argc && arg=="--warmup-ms") protocol.warmup_ms=std::stod(argv[++i]);
            else if(i+1<argc && arg=="--repeat") protocol.repeat=std::stoul(argv[++i]);
            else if(i+1<argc && arg=="--trials") protocol.trials=std::stoul(argv[++i]);
            else throw std::invalid_argument("usage: cubutterfly_stage_microbench --point-json JSON [--describe-only] [--warmup N --warmup-ms MS --repeat N --trials N]");
        }
        if(point.is_null() || !protocol.repeat || !protocol.trials) throw std::invalid_argument("point, repeat and trials must be positive/present");
        if(!std::isfinite(protocol.warmup_ms) || protocol.warmup_ms<0)
            throw std::invalid_argument("warmup-ms must be finite and non-negative");
        auto result=probe(point,protocol); std::cout<<result<<'\n';
        return result.value("status","")=="unavailable" ? 2 : 0;
    } catch(const std::exception& e) {
        std::cout<<Json{{"schema","cubutterfly-stage-probe-v1"},{"status","unavailable"},{"correct",false},{"reason",e.what()}}<<'\n';
        return 1;
    }
}
