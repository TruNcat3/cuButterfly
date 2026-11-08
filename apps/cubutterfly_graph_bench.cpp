#include "cuntt/butterfly.hpp"
#include "cubutterfly/mapping.hpp"
#include <nlohmann/json.hpp>

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <fstream>
#include <memory>
#include <set>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

struct DeviceBuffer {
    cuntt::Complex64* value = nullptr;
    explicit DeviceBuffer(std::size_t count) {
        check(cudaMalloc(reinterpret_cast<void**>(&value), count * sizeof(cuntt::Complex64)), "cudaMalloc");
    }
    ~DeviceBuffer() { cudaFree(value); }
    DeviceBuffer(const DeviceBuffer&) = delete;
};

struct CudaStream {
    cudaStream_t value = nullptr;
    CudaStream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking), "cudaStreamCreateWithFlags"); }
    ~CudaStream() { cudaStreamDestroy(value); }
    CudaStream(const CudaStream&) = delete;
};

template <typename Launch>
float measure_replays(cudaStream_t stream, std::uint32_t warmup, std::uint32_t repeat,
                      Launch&& launch) {
    for (std::uint32_t i = 0; i < warmup; ++i)
        launch();
    check(cudaStreamSynchronize(stream), "replay warmup synchronize");
    cudaEvent_t start = nullptr, stop = nullptr;
    check(cudaEventCreate(&start), "cudaEventCreate start");
    check(cudaEventCreate(&stop), "cudaEventCreate stop");
    check(cudaEventRecord(start, stream), "cudaEventRecord start");
    for (std::uint32_t i = 0; i < repeat; ++i)
        launch();
    check(cudaEventRecord(stop, stream), "cudaEventRecord stop");
    check(cudaEventSynchronize(stop), "replay timing synchronize");
    float total_ms = 0.0F;
    check(cudaEventElapsedTime(&total_ms, start, stop), "cudaEventElapsedTime");
    check(cudaEventDestroy(start), "cudaEventDestroy start");
    check(cudaEventDestroy(stop), "cudaEventDestroy stop");
    return total_ms / repeat;
}

// Explicit mapping comparisons use the public plan/lowering and graph capture.
// They are distinct from AUTO selection; preparation and references are untimed.
int mapping_suite(const char* path, bool prepare_reference=false, const std::string& profile_case="", unsigned trial_offset=0) {
    using Json=nlohmann::json;
    std::ifstream file(path);
    if (!file) throw std::invalid_argument("cannot read graph suite configuration");
    Json suite; file >> suite;
    if (suite.at("operator")!="fft" || suite.at("precision")!="fp64")
        throw std::invalid_argument("this graph suite currently validates FFT FP64 only");
    const auto log_n=suite.at("logN").get<unsigned>();
    const auto batch=suite.at("batch").get<std::size_t>();
    const auto warmup=suite.at("warmup").get<unsigned>();
    const auto repeat=suite.at("repeat").get<unsigned>();
    const auto trials=suite.at("trials").get<unsigned>();
    const bool inverse=suite.at("inverse").get<bool>();
    if(log_n<1 || log_n>24 || !batch || !repeat || repeat>10000 || !trials || trials>31 || warmup>1000)
        throw std::invalid_argument("invalid bounded graph suite dimensions/protocol");
    const std::size_t n=std::size_t{1}<<log_n, count=n*batch;
    if(count/n!=batch) throw std::invalid_argument("graph suite extent overflow");
    const auto cases=suite.at("cases").size();
    if(!profile_case.empty()) {
        unsigned matches=0;
        for(const auto& item:suite.at("cases")) matches+=item.at("id")==profile_case;
        if(matches!=1 || !suite.contains("reference_file"))
            throw std::invalid_argument("profile requires one prepared case and reference");
    }
    if(!cases || cases>64 || count>(std::size_t{1}<<28))
        throw std::invalid_argument("graph suite exceeds bounded host reference size");
    std::vector<cuntt::Complex64> host(count), expected(count), actual(count);
    std::mt19937 generator(suite.at("seed").get<unsigned>());
    std::uniform_real_distribution<double> distribution(-1,1);
    for(auto& value:host) value={distribution(generator),distribution(generator)};
    if(!prepare_reference && suite.contains("reference_file")) {
        std::ifstream ref(suite.at("reference_file").get<std::string>(),std::ios::binary|std::ios::ate);
        if(!ref || ref.tellg()!=static_cast<std::streamoff>(count*sizeof(expected[0])))
            throw std::invalid_argument("prepared reference length mismatch");
        ref.seekg(0); ref.read(reinterpret_cast<char*>(expected.data()),count*sizeof(expected[0]));
        if(!ref) throw std::runtime_error("cannot read prepared reference");
    } else {
        for(std::size_t b=0;b<batch;++b) {
            std::vector<cuntt::Complex64> reference(host.begin()+b*n,host.begin()+(b+1)*n);
            cuntt::reference_fft(reference,inverse,false);
            std::copy(reference.begin(),reference.end(),expected.begin()+b*n);
        }
    }
    if(prepare_reference) {
        std::ofstream ref(suite.at("reference_file").get<std::string>(),std::ios::binary);
        ref.write(reinterpret_cast<const char*>(expected.data()),count*sizeof(expected[0]));
        if(!ref) throw std::runtime_error("cannot save prepared reference");
        return 0;
    }
    const auto hardware=cuntt::query_hardware_resource_model();
    if(count>hardware.memory_bytes/sizeof(cuntt::Complex64)/(4+2*(profile_case.empty()?cases:1)))
        throw std::invalid_argument("graph suite exceeds conservative workspace budget");
    CudaStream stream;
    DeviceBuffer input(count), output(count);
    check(cudaMemcpy(input.value,host.data(),count*sizeof(host[0]),cudaMemcpyHostToDevice),"suite H2D");
    struct Entry {
        std::unique_ptr<cuntt::ButterflyPlan> plan;
        cuntt::ButterflyGraph graph;
        Json result;
        std::vector<double> samples;
    };
    std::vector<std::unique_ptr<Entry>> entries;
    std::set<std::string> identifiers;
    for(const auto& requested:suite.at("cases")) {
        const auto id=requested.at("id").get<std::string>();
        if(!identifiers.insert(id).second) throw std::invalid_argument("duplicate graph suite id");
        if(!profile_case.empty() && id!=profile_case) continue;
        auto entry=std::make_unique<Entry>();
        cuntt::ButterflyConfig config;
        config.op=cuntt::ButterflyOperator::Fft;
        config.precision=cuntt::ButterflyPrecision::Fp64;
        config.log_n=log_n; config.batch=batch;
        config.inverse=inverse; config.normalize_inverse=false;
        cubutterfly::apply_serialized_mapping(requested.at("mapping").dump(),config);
        entry->plan=std::make_unique<cuntt::ButterflyPlan>(config);
        entry->plan->set_stream(stream.value);
        const auto actual_mapping=Json::parse(cubutterfly::serialize_mapping(entry->plan->config()));
        for(auto it=requested.at("mapping").begin();it!=requested.at("mapping").end();++it) {
            if(it.key()=="compute_unit" && it.value()=="auto") continue;
            if(!actual_mapping.contains(it.key()) || actual_mapping.at(it.key())!=it.value())
                throw std::runtime_error("actual mapping differs from request: "+id+"/"+it.key());
        }
        // Any lazy provider work precedes graph capture and all timing.
        entry->plan->execute_async(input.value,output.value);
        check(cudaStreamSynchronize(stream.value),"suite initialization");
        entry->graph=entry->plan->capture_graph(input.value,output.value);
        if(!entry->graph.valid() || !entry->graph.node_count()) throw std::runtime_error("empty suite graph");
        entry->result={{"id",id},{"mapping",Json::parse(cubutterfly::serialize_mapping(entry->plan->config()))},
            {"execution_groups",Json::parse(cubutterfly::execution_groups_json(entry->plan->dataflow_plan()))},
            {"graph_nodes",entry->graph.node_count()},{"kernel_nodes",entry->graph.kernel_node_count()},
            {"workspace_bytes",entry->plan->workspace_size()}};
        entries.push_back(std::move(entry));
        std::cerr << "prepared " << id << '\n';
    }
    for(unsigned t=0;t<(profile_case.empty()?trials:0);++t) for(std::size_t offset=0;offset<entries.size();++offset) {
        auto& entry=*entries[(offset+t+trial_offset)%entries.size()];
        entry.samples.push_back(measure_replays(stream.value,warmup,repeat,[&]{entry.graph.launch(stream.value);}));
    }
    Json rows=Json::array();
    bool finite=true;
    for(auto& pointer:entries) {
        auto& entry=*pointer;
        entry.graph.launch(stream.value); check(cudaStreamSynchronize(stream.value),"suite verification");
        check(cudaMemcpy(actual.data(),output.value,count*sizeof(actual[0]),cudaMemcpyDeviceToHost),"suite D2H");
        double maximum=0; long double error2=0,reference2=0; std::size_t nonfinite=0;
        for(std::size_t i=0;i<count;++i) {
            const auto a=actual[i],r=expected[i];
            if(!std::isfinite(a.real)||!std::isfinite(a.imag)||!std::isfinite(r.real)||!std::isfinite(r.imag)) {
                ++nonfinite; continue;
            }
            const auto error=std::hypot(a.real-r.real,a.imag-r.imag);
            maximum=std::max(maximum,error); error2+=static_cast<long double>(error)*error;
            reference2+=static_cast<long double>(r.real)*r.real+static_cast<long double>(r.imag)*r.imag;
        }
        finite &= nonfinite==0;
        if(profile_case.empty()) {
            auto sorted=entry.samples; std::sort(sorted.begin(),sorted.end());
            entry.result["trial_ms"]=entry.samples;
            entry.result["median_ms"]=(sorted[(sorted.size()-1)/2]+sorted[sorted.size()/2])/2;
            entry.result["min_ms"]=sorted.front(); entry.result["max_ms"]=sorted.back();
        }
        entry.result["max_abs_cpu"]=maximum;
        entry.result["relative_l2_cpu"]=std::sqrt(error2/std::max(reference2,1e-300L));
        entry.result["nonfinite"]=nonfinite; entry.result["verified_batches"]=batch;
        rows.push_back(entry.result);
    }
    if(!profile_case.empty()) {
        if(!finite) throw std::runtime_error("nonfinite profile verification");
        auto& entry=*entries.front();
        for(unsigned i=0;i<100;++i) entry.graph.launch(stream.value);
        check(cudaStreamSynchronize(stream.value),"profile warmup synchronize");
        check(cudaProfilerStart(),"cudaProfilerStart");
        try {
            entry.graph.launch(stream.value);
            check(cudaStreamSynchronize(stream.value),"profile replay synchronize");
        } catch(...) { cudaProfilerStop(); throw; }
        check(cudaProfilerStop(),"cudaProfilerStop");
        auto result=rows.front();
        result["schema"]="cubutterfly-factor-graph-profile-v1";
        result["status"]="ok"; result["profile_case"]=profile_case;
        result["logical_calls"]=1; result["warmup"]=100;
        result["actual_mapping"]=result.at("mapping");
        result["normal_timing_accepted"]=false;
        result["runtime_fingerprint"]=cubutterfly::runtime_fingerprint();
        std::cout << result.dump(2) << '\n';
        return 0;
    }
    std::cout << Json({{"schema","cubutterfly-explicit-graph-suite-v1"},{"config",suite},
        {"runtime_fingerprint",cubutterfly::runtime_fingerprint()},
        {"verification","same-precision CPU reference, report errors without a new tolerance"},
        {"trial_offset",trial_offset},{"all_outputs_finite",finite},{"results",rows}}).dump(2) << '\n';
    return finite ? 0 : 2;
}
}

int main(int argc, char** argv) {
    try {
        if(argc==5 && std::string(argv[1])=="--profile-suite" && std::string(argv[3])=="--case") {
            if(!*argv[4]) throw std::invalid_argument("empty profile case");
            return mapping_suite(argv[2],false,argv[4]);
        }
        if(argc==5 && std::string(argv[1])=="--suite" && std::string(argv[3])=="--trial-offset")
            return mapping_suite(argv[2],false,"",std::stoul(argv[4]));
        if(argc==3 && std::string(argv[1])=="--suite") return mapping_suite(argv[2]);
        if(argc==3 && std::string(argv[1])=="--prepare-reference") return mapping_suite(argv[2],true);
        const std::uint32_t log_n = argc > 1 ? std::stoul(argv[1]) : 20;
        const std::size_t batch = argc > 2 ? std::stoull(argv[2]) : 16;
        const std::uint32_t warmup = argc > 3 ? std::stoul(argv[3]) : 5;
        const std::uint32_t repeat = argc > 4 ? std::stoul(argv[4]) : 20;
        const std::string backend_name = argc > 5 ? argv[5] : "factor-streamed";
        if (log_n != 20 || batch == 0 || warmup == 0 || repeat == 0 ||
            (backend_name != "factor-streamed" && backend_name != "shared-iterative" &&
             backend_name != "shared-iterative-transpose"))
            throw std::invalid_argument("usage: cubutterfly_graph_bench [logN=20] [batch=16] [warmup=5] [repeat=20] [factor-streamed|shared-iterative|shared-iterative-transpose]");

        const std::size_t n = std::size_t{1} << log_n;
        const std::size_t count = n * batch;
        std::vector<cuntt::Complex64> host_input(count), host_output(count), expected(count);
        std::mt19937 generator(0x43554246U + static_cast<unsigned>(batch));
        std::uniform_real_distribution<double> distribution(-1.0, 1.0);
        for (auto& value : host_input)
            value = {distribution(generator), distribution(generator)};
        // CUDA graph capture is intentionally performed on an explicit
        // non-blocking stream.  Capturing the legacy default stream can be
        // rejected when unrelated work exists in the process or device.
        CudaStream stream;
        DeviceBuffer input(count), output(count);
        check(cudaMemcpy(input.value, host_input.data(), count * sizeof(host_input[0]), cudaMemcpyHostToDevice), "cudaMemcpy H2D");

        cuntt::ButterflyConfig config;
        config.op = cuntt::ButterflyOperator::Fft;
        config.backend = backend_name == "factor-streamed"
            ? cuntt::ButterflyBackend::FactorStreamed : cuntt::ButterflyBackend::SharedIterative;
        config.fft_core = backend_name == "factor-streamed" ? cuntt::FftCore::CufftDxBlock : cuntt::FftCore::Scalar;
        config.precision = cuntt::ButterflyPrecision::Fp64;
        config.normalize_inverse = false;
        config.log_n = log_n;
        config.batch = batch;
        config.stage_partition = {10, 10};
        config.tile_threads = 256;
        config.batch_tile_count = 1;
        config.cross_twiddle = backend_name == "factor-streamed"
            ? cuntt::CrossTwiddleMode::Recurrence : cuntt::CrossTwiddleMode::Table;
        config.direct_boundary = backend_name == "shared-iterative-transpose"
            ? cuntt::DirectBoundary::TiledTranspose : cuntt::DirectBoundary::Strided;
        config.shared_layout = cuntt::SharedLayout::WriterAligned;
        if (backend_name == "factor-streamed") {
            config.stage_partition = {log_n};
            config.factor_partition = {10, 10};
            config.factor_columns = 4;
            config.factor_ept = 8;
            config.data_tiles_per_cta = 4;
            config.prefetch_depth = 0;
        }
        cuntt::ButterflyPlan plan(config);
        plan.set_stream(stream.value);
        auto graph = plan.capture_graph(input.value, output.value);
        if (!graph.valid() || graph.node_count() == 0)
            throw std::runtime_error(backend_name + " plan did not produce a non-empty CUDA Graph");

        const auto kernel_nodes = graph.kernel_node_count();
        const auto non_kernel_nodes = graph.non_kernel_node_count();
        const float graph_kernel_ms = measure_replays(stream.value, warmup, repeat,
                                                       [&] { graph.launch(stream.value); });
        const float plan_kernel_ms = measure_replays(stream.value, warmup, repeat,
                                                     [&] { plan.execute_async(input.value, output.value); });

        check(cudaMemcpy(host_output.data(), output.value, count * sizeof(host_output[0]), cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
        // The reference helper operates on one transform, so check every batch
        // explicitly while retaining the graph replay protocol.
        double max_error = 0.0;
        for (std::size_t transform = 0; transform < batch; ++transform) {
            std::vector<cuntt::Complex64> reference(n);
            for (std::size_t index = 0; index < n; ++index)
                reference[index] = host_input[transform * n + index];
            cuntt::reference_fft(reference, false, false);
            for (std::size_t index = 0; index < n; ++index) {
                const auto& actual = host_output[transform * n + index];
                max_error = std::max(max_error, std::hypot(actual.real - reference[index].real,
                                                            actual.imag - reference[index].imag));
            }
        }
        const bool correct = std::isfinite(max_error) && max_error <= 1.0e-10;
        std::cout << "operator=fft,precision=fp64,backend=" << backend_name << ",graph=compiled"
                  << ",logN=" << log_n << ",batch=" << batch
                  << ",graph_nodes=" << graph.node_count()
                  << ",graph_kernel_nodes=" << kernel_nodes
                  << ",graph_non_kernel_nodes=" << non_kernel_nodes
                  << ",graph_kernel_ms=" << graph_kernel_ms
                  << ",plan_kernel_ms=" << plan_kernel_ms
                  << ",graph_speedup_vs_plan=" << (plan_kernel_ms / graph_kernel_ms)
                  << ",max_error=" << max_error << ",correct=" << (correct ? 1 : 0) << '\n';
        return correct ? 0 : 1;
    } catch (const std::exception& error) {
        if(argc>1 && std::string(argv[1])=="--profile-suite")
            std::cout << nlohmann::json({{"status","error"},{"error",error.what()},
                {"normal_timing_accepted",false}}).dump() << '\n';
        std::cerr << error.what() << '\n';
        return 1;
    }
}
