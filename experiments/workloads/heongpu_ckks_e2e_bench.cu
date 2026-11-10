// Complete public CKKS calls; setup and CPU validation are outside timing.
#include <heongpu/heongpu.hpp>
#include "ckks_bootstrap_parameters.hpp"
#include "ntt_adapter.hpp"
#include "graph_memory.hpp"
#include "ntt_capture.hpp"
#include "ntt_graph_batch.hpp"
#include "graph_schedule.hpp"
#include "gpu_thermal_guard.hpp"
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
#ifndef CUBUTTERFLY_HEONGPU_STREAM_ORDERING_FIX
#define CUBUTTERFLY_HEONGPU_STREAM_ORDERING_FIX 0
#endif
constexpr auto Scheme = heongpu::Scheme::CKKS;
using Cipher = heongpu::Ciphertext<Scheme>;
struct Options {
    std::vector<std::size_t> sizes{4096, 16384}, batches{1, 8};
    int warmup = 3, repetitions = 10;
    int graph_replays = 1;
    int schedule_rounds = 7, schedule_replays = 5;
    int aux_primes = 1;
    float pool_max_device_fraction = 0.95f;
    std::string parameter_set = "short-chain";
    unsigned seed = 20261018;
    std::string output;
    cubutterfly_heongpu::Execution ntt_execution = cubutterfly_heongpu::Execution::NativeRns;
    bool rotate = false;
    bool graph = false;
    bool stage_batch = false;
    bool stage_batch_only = false;
    cubutterfly_heongpu::StageBackend stage_backend = cubutterfly_heongpu::StageBackend::CuButterfly;
    bool auto_schedule = false;
    bool trace_measurement = false;
    std::vector<std::string> scenes;
    bool thermal_guard = false;
    bool prepare_only = false;
    std::string prepare_backend = "both";
    std::string stream_mode = "nonblocking";
    bool stream_ordering_fix = false;
};
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
std::vector<std::size_t> numbers(const std::string& text) {
    std::vector<std::size_t> result;
    std::stringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ',')) {
        const auto n = std::stoull(token);
        if (!n) throw std::invalid_argument("sizes and batches must be positive");
        result.push_back(n);
    }
    if (result.empty()) throw std::invalid_argument("empty size or batch list");
    return result;
}
std::vector<int> bit_sizes(const std::string& text) {
    std::vector<int> result;
    std::stringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ':')) {
        const auto bits = std::stoi(token);
        if (bits < 2 || bits > 62) throw std::invalid_argument("invalid coefficient modulus bit size");
        result.push_back(bits);
    }
    if (result.empty()) throw std::invalid_argument("empty coefficient modulus bit-size list");
    return result;
}
Options options(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&]() -> std::string {
            if (++i >= argc) throw std::invalid_argument("missing value for " + arg);
            return argv[i];
        };
        if (arg == "--sizes") o.sizes = numbers(value());
        else if (arg == "--batches") o.batches = numbers(value());
        else if (arg == "--warmup") o.warmup = std::stoi(value());
        else if (arg == "--repetitions") o.repetitions = std::stoi(value());
        else if (arg == "--graph-replays") o.graph_replays = std::stoi(value());
        else if (arg == "--schedule-rounds") o.schedule_rounds = std::stoi(value());
        else if (arg == "--schedule-replays") o.schedule_replays = std::stoi(value());
        else if (arg == "--scenes") {
            std::stringstream names(value());
            std::string name;
            while (std::getline(names, name, ',')) {
                if (name != "keyswitch" && name != "square_relinearize_rescale" &&
                    name != "square_rescale_keyswitch" && name != "rotate_rows_shift1" &&
                    name != "regular_bootstrapping" && name != "regular_bootstrapping_v2")
                    throw std::invalid_argument("unknown scene: " + name);
                if (std::find(o.scenes.begin(), o.scenes.end(), name) != o.scenes.end())
                    throw std::invalid_argument("duplicate scene: " + name);
                o.scenes.push_back(name);
            }
            if (o.scenes.empty()) throw std::invalid_argument("empty scenes");
        }
        else if (arg == "--aux-primes") o.aux_primes = std::stoi(value());
        else if (arg == "--pool-max-fraction") o.pool_max_device_fraction = std::stof(value());
        else if (arg == "--parameter-set") o.parameter_set = value();
        else if (arg == "--seed") o.seed = std::stoul(value());
        else if (arg == "--output") o.output = value();
        else if (arg == "--ntt-execution") {
            const auto mode = value();
            if (mode == "native-rns") o.ntt_execution = cubutterfly_heongpu::Execution::NativeRns;
            else if (mode == "grouped") o.ntt_execution = cubutterfly_heongpu::Execution::Grouped;
            else throw std::invalid_argument("ntt-execution must be native-rns or grouped");
        }
        else if (arg == "--rotate") o.rotate = true;
        else if (arg == "--graph") o.graph = true;
        else if (arg == "--stage-batch") o.stage_batch = true;
        else if (arg == "--stage-batch-only") o.stage_batch_only = true;
        else if (arg == "--stage-backend") {
            const auto backend = value();
            if (backend == "native") o.stage_backend = cubutterfly_heongpu::StageBackend::Native;
            else if (backend == "cubutterfly") o.stage_backend = cubutterfly_heongpu::StageBackend::CuButterfly;
            else throw std::invalid_argument("stage-backend must be native or cubutterfly");
        }
        else if (arg == "--auto-schedule") o.auto_schedule = true;
        else if (arg == "--trace-measurement") o.trace_measurement = true;
        else if (arg == "--thermal-guard") o.thermal_guard = true;
        else if (arg == "--prepare-only") o.prepare_only = true;
        else if (arg == "--prepare-backend") o.prepare_backend = value();
        else if (arg == "--stream-mode") o.stream_mode = value();
        else if (arg == "--stream-ordering-fix") o.stream_ordering_fix = true;
        else if (arg == "--help") {
            std::cout << "[--parameter-set short-chain|bootstrap-v1-small|bootstrap-v2-small|bootstrap-v2-mid|bootstrap-v2] [--prepare-only --prepare-backend native|cubutterfly|both --stream-mode nonblocking|default]\n";
            std::cout << "--sizes 4096,16384 --batches 1,8 --warmup 3 --repetitions 10 --output FILE [--rotate] [--graph] [--graph-replays 1] [--stage-batch [--stage-batch-only --stage-backend native|cubutterfly]] [--auto-schedule --schedule-rounds 7 --schedule-replays 5] [--scenes keyswitch,...] [--thermal-guard] [--aux-primes 1|2] [--pool-max-fraction 0.95] [--seed N] [--ntt-execution native-rns|grouped]\n";
            std::exit(0);
        } else throw std::invalid_argument("unknown option: " + arg);
    }
    if (o.warmup < 0 || o.repetitions < 1 || o.graph_replays < 1) throw std::invalid_argument("invalid iteration count");
    if (!(o.pool_max_device_fraction > 0.0f && o.pool_max_device_fraction <= 1.0f))
        throw std::invalid_argument("pool-max-fraction must be in (0,1]");
    if (o.parameter_set != "short-chain" && o.parameter_set != "bootstrap-v1-small" &&
        o.parameter_set != "bootstrap-v2" &&
        o.parameter_set != "bootstrap-v2-small" && o.parameter_set != "bootstrap-v2-mid")
        throw std::invalid_argument("unknown parameter-set");
    if (o.parameter_set == "bootstrap-v2" || o.parameter_set == "bootstrap-v2-small" || o.parameter_set == "bootstrap-v2-mid") {
        if (o.sizes != std::vector<std::size_t>{ckks_bootstrap_profile::N} ||
            o.aux_primes != int(ckks_bootstrap_profile::p_moduli.size()) || o.rotate)
            throw std::invalid_argument("bootstrap-v2 profile requires its generated N/P5 contract and no --rotate");
        if (o.scenes.empty()) o.scenes = {"regular_bootstrapping_v2"};
        if (o.scenes != std::vector<std::string>{"regular_bootstrapping_v2"})
            throw std::invalid_argument("bootstrap-v2 requires the complete bootstrap scene");
    } else if (o.parameter_set == "bootstrap-v1-small") {
        if (o.sizes != std::vector<std::size_t>{ckks_bootstrap_profile::N} ||
            o.aux_primes != 3 || o.rotate)
            throw std::invalid_argument("bootstrap-v1-small requires N4096/P3 and no --rotate");
        if (o.scenes.empty()) o.scenes = {"regular_bootstrapping"};
        if (o.scenes != std::vector<std::string>{"regular_bootstrapping"})
            throw std::invalid_argument("bootstrap-v1-small requires the complete bootstrap scene");
    } else {
        if (o.aux_primes != 1 && o.aux_primes != 2)
            throw std::invalid_argument("short-chain aux-primes must be 1 or 2");
        if (std::find(o.scenes.begin(), o.scenes.end(), "regular_bootstrapping") != o.scenes.end() ||
            std::find(o.scenes.begin(), o.scenes.end(), "regular_bootstrapping_v2") != o.scenes.end())
            throw std::invalid_argument("bootstrap scene requires its explicit parameter-set");
    }
    if (o.stage_batch && (!o.graph || o.ntt_execution != cubutterfly_heongpu::Execution::NativeRns))
        throw std::invalid_argument("stage-batch requires --graph --ntt-execution native-rns");
    if (o.stage_batch_only && !o.stage_batch)
        throw std::invalid_argument("stage-batch-only requires --stage-batch");
    if (o.stage_backend == cubutterfly_heongpu::StageBackend::Native && !o.stage_batch_only)
        throw std::invalid_argument("stage-backend native requires --stage-batch-only");
    if (o.stage_batch_only && o.auto_schedule)
        throw std::invalid_argument("stage-batch-only does not support auto-schedule; there is no loop candidate");
    if (o.thermal_guard && !o.graph)
        throw std::invalid_argument("thermal-guard currently requires --graph");
    if (o.auto_schedule && (!o.stage_batch || o.output.empty()))
        throw std::invalid_argument("auto-schedule requires --stage-batch and --output for selection evidence");
    if (o.trace_measurement && (!o.graph || o.scenes.size() != 1 || o.sizes.size() != 1 || o.batches.size() != 1))
        throw std::invalid_argument("trace-measurement requires one explicit Graph scene, size and batch");
    if (o.schedule_rounds < 1 || o.schedule_replays < 1)
        throw std::invalid_argument("invalid schedule measurement budget");
    if (o.prepare_backend != "native" && o.prepare_backend != "cubutterfly" && o.prepare_backend != "both")
        throw std::invalid_argument("prepare-backend must be native, cubutterfly or both");
    if (!o.prepare_only && o.prepare_backend != "both")
        throw std::invalid_argument("prepare-backend requires prepare-only");
    if (o.stream_mode != "nonblocking" && o.stream_mode != "default")
        throw std::invalid_argument("stream-mode must be nonblocking or default");
    if (o.stream_ordering_fix != bool(CUBUTTERFLY_HEONGPU_STREAM_ORDERING_FIX))
        throw std::invalid_argument("stream-ordering-fix does not match the compiled binary; rebuild or use matching flags");
    if (o.stream_mode == "default" && !o.prepare_only)
        throw std::invalid_argument("default stream is a prepare-only diagnostic control");
    if (o.prepare_only && (o.graph || o.stage_batch || o.auto_schedule || o.trace_measurement || o.thermal_guard))
        throw std::invalid_argument("prepare-only cannot be combined with Graph/timing options");
    if (std::find(o.scenes.begin(), o.scenes.end(), "rotate_rows_shift1") != o.scenes.end() && !o.rotate)
        throw std::invalid_argument("rotate_rows_shift1 requires --rotate");
    return o;
}
struct Errors {
    double maxabs = 0, sumsq = 0, reference_sumsq = 0;
    void add(const std::vector<double>& actual, const std::vector<double>& expected) {
        if (actual.size() != expected.size()) throw std::runtime_error("decoded slot count mismatch");
        for (std::size_t i = 0; i < actual.size(); ++i) {
            if (!std::isfinite(actual[i])) throw std::runtime_error("nonfinite decoded value");
            const double e = actual[i] - expected[i];
            maxabs = std::max(maxabs, std::abs(e));
            sumsq += e * e;
            reference_sumsq += expected[i] * expected[i];
        }
    }
    double relative() const {
        return reference_sumsq ? std::sqrt(sumsq / reference_sumsq)
                              : (sumsq ? std::numeric_limits<double>::infinity() : 0);
    }
};
std::size_t mismatches(Cipher& a, Cipher& b) {
    // memory_size() is private in the pinned API; get_data exports all logical
    // ciphertext coefficients without including unused allocator capacity.
    std::vector<Data64> ha, hb;
    a.get_data(ha);
    b.get_data(hb);
    check(cudaDeviceSynchronize());
    const std::size_t na = ha.size(), nb = hb.size();
    std::size_t count = na > nb ? na - nb : nb - na;
    for (std::size_t i = 0; i < std::min(na, nb); ++i) count += ha[i] != hb[i];
    return count;
}
template<class F> double elapsed(F&& call) {
    check(cudaDeviceSynchronize());
    const auto start = std::chrono::steady_clock::now();
    call();
    check(cudaDeviceSynchronize());
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

struct CapturedCall {
    // All RMM buffers, including temporaries destroyed during the public call,
    // remain reserved until after graph destruction. Native and treatment have
    // separate arenas; validation/another graph cannot reuse their scratch.
    cubutterfly_heongpu::GraphMemory memory;
    std::vector<Cipher> outputs;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    cudaStream_t stream;
    std::size_t calls = 0, polynomials = 0;
    std::vector<cubutterfly_heongpu::CapturedNttStage> stages;
    // Native and cuButterfly stage-batch graphs may share the same marked
    // scalar captures.  The graph nodes retain pointers into these arenas, so
    // the owners must outlive both BatchedNttGraph instances.  Sharing avoids
    // recapturing and retaining a duplicate B-call GraphMemory set.
    std::shared_ptr<std::vector<std::unique_ptr<CapturedCall>>> scalar_calls;
    std::unique_ptr<cubutterfly_heongpu::BatchedNttGraph> batched;
    std::unique_ptr<cubutterfly_heongpu::GraphSchedule> schedule;
    CapturedCall* selected = nullptr; // Borrowed; candidate calls outlive this wrapper.
    explicit CapturedCall(cudaStream_t value) : stream(value) {}
    ~CapturedCall() {
        cudaStreamSynchronize(stream);
        batched.reset();
        if (executable) cudaGraphExecDestroy(executable);
        if (graph) cudaGraphDestroy(graph);
        outputs.clear();
    }
    template<class F> void prepare(F&& run, bool enabled, bool mark_stages = false) {
        cubutterfly_heongpu::set_enabled(enabled);
        memory.recording();
        cubutterfly_heongpu::GraphMemory::activate(&memory);
        bool capturing = false;
        try {
            run(outputs);
            check(cudaStreamSynchronize(stream));
            outputs.clear();
            memory.replaying();
            cubutterfly_heongpu::reset_counters();
            check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
            capturing = true;
            if (mark_stages) cubutterfly_heongpu::record_ntt_stages(&stages);
            run(outputs);
            cubutterfly_heongpu::record_ntt_stages(nullptr);
            check(cudaStreamEndCapture(stream, &graph));
            capturing = false;
            memory.verify_replay_complete();
            cubutterfly_heongpu::GraphMemory::activate(nullptr);
            if (!memory.allocations()) throw std::runtime_error("Graph arena hook did not intercept allocations");
            calls = cubutterfly_heongpu::replacement_calls();
            polynomials = cubutterfly_heongpu::replacement_polynomials();
            // A marked graph omits all NTT work and is only a lowering input.
            if (!mark_stages) check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        } catch (...) {
            cubutterfly_heongpu::record_ntt_stages(nullptr);
            cubutterfly_heongpu::GraphMemory::activate(nullptr);
            if (capturing) cudaStreamEndCapture(stream, &graph);
            throw;
        }
    }
    void prepare_batch(const std::vector<cubutterfly_heongpu::MarkedNttGraph>& marked,
                       const std::shared_ptr<std::vector<std::unique_ptr<CapturedCall>>>& owners,
                       cubutterfly_heongpu::StageBackend backend) {
        if (!owners || owners->size() != marked.size())
            throw std::logic_error("marked scalar capture owner count");
        scalar_calls = owners;
        batched = std::make_unique<cubutterfly_heongpu::BatchedNttGraph>(marked, backend, stream);
        calls = batched->stage_count();
        polynomials = batched->polynomial_count();
    }
    template<class F> void prepare_batch(F&& run_one, std::size_t batch,
                                         cubutterfly_heongpu::StageBackend backend) {
        auto owners = std::make_shared<std::vector<std::unique_ptr<CapturedCall>>>();
        std::vector<cubutterfly_heongpu::MarkedNttGraph> marked;
        owners->reserve(batch);
        marked.reserve(batch);
        for (std::size_t b = 0; b < batch; ++b) {
            auto scalar = std::make_unique<CapturedCall>(stream);
            scalar->prepare([&](std::vector<Cipher>& result) { run_one(b, result); }, true, true);
            if (scalar->outputs.size() != 1) throw std::logic_error("scalar capture output count");
            marked.push_back({scalar->graph, scalar->stages});
            owners->push_back(std::move(scalar));
        }
        prepare_batch(marked, owners, backend);
    }
    Cipher& output(std::size_t b) {
        if (selected) return selected->output(b);
        return batched ? scalar_calls->at(b)->outputs.at(0) : outputs.at(b);
    }
    void launch(cudaStream_t execution_stream = nullptr) {
        if (!execution_stream) execution_stream = stream;
        if (schedule) schedule->launch(execution_stream);
        else if (batched) batched->launch(execution_stream);
        else {
            if (!executable) throw std::logic_error("cannot execute a graph with unresolved NTT markers");
            check(cudaGraphLaunch(executable, execution_stream));
        }
    }
    std::size_t kernel_nodes() const {
        if (selected) return selected->kernel_nodes();
        if (batched) return batched->kernel_nodes();
        std::size_t size = 0, kernels = 0;
        check(cudaGraphGetNodes(graph, nullptr, &size));
        std::vector<cudaGraphNode_t> nodes(size);
        check(cudaGraphGetNodes(graph, nodes.data(), &size));
        for (auto node : nodes) {
            cudaGraphNodeType type;
            check(cudaGraphNodeGetType(node, &type));
            kernels += type == cudaGraphNodeTypeKernel;
        }
        return kernels;
    }
    double measure(int replays = 1) {
        cudaEvent_t start, stop;
        check(cudaEventCreate(&start));
        check(cudaEventCreate(&stop));
        check(cudaEventRecord(start, stream));
        for (int i = 0; i < replays; ++i) launch();
        check(cudaEventRecord(stop, stream));
        check(cudaEventSynchronize(stop));
        float ms = 0;
        check(cudaEventElapsedTime(&ms, start, stop));
        check(cudaEventDestroy(start));
        check(cudaEventDestroy(stop));
        return ms / replays;
    }
};
} // namespace

int main(int argc, char** argv) {
    try {
        const auto o = options(argc, argv);
        const bool bootstrap_v2 = o.parameter_set == "bootstrap-v2" ||
                                  o.parameter_set == "bootstrap-v2-small" ||
                                  o.parameter_set == "bootstrap-v2-mid";
        const bool bootstrap_v1 = o.parameter_set == "bootstrap-v1-small";
        const bool bootstrap = bootstrap_v1 || bootstrap_v2;
        cubutterfly_heongpu::set_execution(o.ntt_execution);
        const char* ntt_execution = cubutterfly_heongpu::execution_name(o.ntt_execution);
        const std::string q_bits = bootstrap ? ckks_bootstrap_profile::q_bits : "50:40:40";
        const std::string p_bits = bootstrap ? ckks_bootstrap_profile::p_bits : (o.aux_primes == 1 ? "50" : "50:50");
        const std::string method = o.aux_primes == 1 ? "I" : "II";
        std::ofstream file;
        std::ofstream selection_file, samples_file;
        std::ostream* out = &std::cout;
        if (!o.output.empty()) {
            file.open(o.output);
            if (!file) throw std::runtime_error("cannot open output: " + o.output);
            out = &file;
            if (o.auto_schedule) {
                selection_file.open(o.output + ".selection.csv");
                samples_file.open(o.output + ".samples.csv");
                if (!selection_file || !samples_file) throw std::runtime_error("cannot open selection evidence");
                selection_file << "scene,N,batch,keyswitch_method,candidate,round,ms,median_ms,selected,selection_wall_ms,replays,confidence\n";
                samples_file << "scene,N,batch,keyswitch_method,variant,round,ms,replays\n";
            }
        }
        *out << "scene,type,baseline,N,batch,mode,native_ms,cubutterfly_ms,speedup,exact_ciphertext,mismatch_count,maxabs_native,maxabs_cu,relL2_native,relL2_cu,replacement_calls,polynomials,repetitions,sec_level,Q_bits,P_bits,scale,keyswitch_method,batch_submission,graph_status,native_kernel_nodes,cubutterfly_kernel_nodes,ntt_execution,graph_replays_per_sample,parameter_set,stage_backend\n" << std::flush;
        int device = 0;
        check(cudaGetDevice(&device));
        cudaDeviceProp prop{};
        check(cudaGetDeviceProperties(&prop, device));
        GpuThermalGuard thermal_guard(o.thermal_guard);
        std::cerr << "gpu_name=" << prop.name << " memory_bytes=" << prop.totalGlobalMem << " uuid=";
        const char* hex = "0123456789abcdef";
        for (unsigned char byte : prop.uuid.bytes) std::cerr << hex[byte >> 4] << hex[byte & 15];
        std::cerr << "\nmode=" << (o.prepare_only ? "preparation_diagnostic_no_timing" :
                                  o.graph ? "cuda_graph_gpu_events_reserved_scratch" : "eager_wall_complete_call") << '\n';
        std::cerr << "ntt_execution=" << ntt_execution << '\n';
        std::cerr << "stream_mode=" << o.stream_mode << '\n';
        cudaStream_t stream = nullptr;
        if (o.stream_mode == "nonblocking") check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        const auto execution = heongpu::ExecutionOptions().set_stream(stream);
        auto memory_phase = [&](const char* phase) {
            check(cudaDeviceSynchronize());
            std::size_t free_bytes = 0, total_bytes = 0;
            check(cudaMemGetInfo(&free_bytes, &total_bytes));
            std::cerr << "memory_phase,phase=" << phase << ",free_bytes=" << free_bytes
                      << ",total_bytes=" << total_bytes
                      << ",pool_live_bytes=" << heongpu::MemoryPool::instance().get_current_device_pool_memory_usage()
                      << ",pool_available_bytes=" << heongpu::MemoryPool::instance().get_free_device_pool_memory() << '\n';
        };
        std::mt19937 rng(o.seed);
        std::uniform_real_distribution<double> distribution(-0.5, 0.5);
        for (const auto n : o.sizes) {
            cubutterfly_heongpu::set_enabled(false);
            cubutterfly_heongpu::freeze_plans(false);
            auto context = heongpu::GenHEContext<Scheme>(heongpu::sec_level_type::none);
            context->set_poly_modulus_degree(n);
            if (bootstrap_v2)
                context->set_coeff_modulus_values(ckks_bootstrap_profile::q_moduli, ckks_bootstrap_profile::p_moduli);
            else if (bootstrap_v1)
                context->set_coeff_modulus_bit_sizes(bit_sizes(ckks_bootstrap_profile::q_bits),
                                                     bit_sizes(ckks_bootstrap_profile::p_bits));
            else
                context->set_coeff_modulus_bit_sizes({50, 40, 40}, std::vector<int>(o.aux_primes, 50));
            heongpu::MemoryPoolConfig pool_config;
            pool_config.max_device_fraction = o.pool_max_device_fraction;
            std::cerr << "memory_pool,max_device_fraction=" << o.pool_max_device_fraction << '\n';
            context->generate(pool_config);
            memory_phase("context_ready");
            heongpu::HEKeyGenerator<Scheme> keygen(context);
            auto secret = bootstrap ? std::make_unique<heongpu::Secretkey<Scheme>>(context, ckks_bootstrap_profile::secret_weight)
                                    : std::make_unique<heongpu::Secretkey<Scheme>>(context);
            auto& sk = *secret;
            heongpu::Secretkey<Scheme> sk2(context);
            if (bootstrap_v2) keygen.generate_secret_key_v2(sk);
            else if (bootstrap_v1) keygen.generate_secret_key(sk);
            else { keygen.generate_secret_key(sk); keygen.generate_secret_key(sk2); }
            heongpu::Publickey<Scheme> pk(context);
            keygen.generate_public_key(pk, sk);
            heongpu::Relinkey<Scheme> relin(context);
            keygen.generate_relin_key(relin, sk);
            heongpu::Switchkey<Scheme> switchkey(context);
            if (!bootstrap) keygen.generate_switch_key(switchkey, sk2, sk);
            std::unique_ptr<heongpu::Switchkey<Scheme>> dense_to_sparse, sparse_to_dense;
            if (bootstrap_v2) {
                heongpu::Secretkey<Scheme> sparse(context, ckks_bootstrap_profile::ephemeral_secret_weight);
                keygen.generate_secret_key_v2(sparse);
                dense_to_sparse = std::make_unique<heongpu::Switchkey<Scheme>>(context);
                sparse_to_dense = std::make_unique<heongpu::Switchkey<Scheme>>(context);
                keygen.generate_switch_key(*dense_to_sparse, sparse, sk);
                keygen.generate_switch_key(*sparse_to_dense, sk, sparse);
            }
            heongpu::HEEncoder<Scheme> encoder(context);
            heongpu::HEEncryptor<Scheme> encryptor(context, pk);
            heongpu::HEDecryptor<Scheme> decryptor(context, sk), decryptor2(context, sk2);
            heongpu::HEArithmeticOperator<Scheme> op(context, encoder);
            std::vector<int> shifts{1};
            if (bootstrap_v2) {
                std::cerr << "bootstrap_setup,phase=encoding_matrices\n";
                heongpu::BootstrappingConfigV2 config(
                    heongpu::EncodingMatrixConfig(heongpu::LinearTransformType::SLOTS_TO_COEFFS, ckks_bootstrap_profile::slots_to_coeffs_level),
                    heongpu::EvalModConfig(ckks_bootstrap_profile::evalmod_log_message_ratio),
                    heongpu::EncodingMatrixConfig(heongpu::LinearTransformType::COEFFS_TO_SLOTS, ckks_bootstrap_profile::coeffs_to_slots_level));
                op.generate_bootstrapping_params_v2(std::ldexp(1.0, ckks_bootstrap_profile::scale_bits), config);
                shifts = op.bootstrapping_key_indexs();
                std::cerr << "bootstrap_setup,phase=galois_keys,count=" << shifts.size() << '\n';
            } else if (bootstrap_v1) {
                std::cerr << "bootstrap_setup,phase=legacy_bootstrap\n";
                heongpu::BootstrappingConfig config(
                    ckks_bootstrap_profile::bootstrap_ctos_piece,
                    ckks_bootstrap_profile::bootstrap_stoc_piece,
                    ckks_bootstrap_profile::bootstrap_taylor_number,
                    ckks_bootstrap_profile::bootstrap_less_key_mode);
                op.generate_bootstrapping_params(
                    std::ldexp(1.0, ckks_bootstrap_profile::scale_bits), config,
                    heongpu::arithmetic_bootstrapping_type::REGULAR_BOOTSTRAPPING);
                shifts = op.bootstrapping_key_indexs();
                std::cerr << "bootstrap_setup,phase=galois_keys,count=" << shifts.size() << '\n';
            }
            heongpu::Galoiskey<Scheme> galois(context, shifts);
            memory_phase("before_galois_keys");
            if (o.rotate || bootstrap) keygen.generate_galois_key(galois, sk);
            memory_phase("keys_ready");
            auto prepare_input = [&](Cipher& input) {
                const std::size_t q_count = bootstrap_v2 ? ckks_bootstrap_profile::q_moduli.size()
                                                          : (bootstrap_v1 ? bit_sizes(ckks_bootstrap_profile::q_bits).size() : 0);
                for (std::size_t level = 1; level < q_count; ++level)
                    op.mod_drop_inplace(input, execution);
                input.switch_stream(stream);
            };
            for (const auto batch : o.batches) {
                std::vector<std::vector<double>> messages(batch, std::vector<double>(n / 2));
                std::vector<Cipher> inputs;
                inputs.reserve(batch);
                for (std::size_t b = 0; b < batch; ++b) {
                    for (auto& x : messages[b]) x = distribution(rng);
                    heongpu::Plaintext<Scheme> plain(context);
                    encoder.encode(plain, messages[b], std::ldexp(1.0, bootstrap ? ckks_bootstrap_profile::scale_bits : 40));
                    inputs.emplace_back(context);
                    encryptor.encrypt(inputs.back(), plain);
                    check(cudaDeviceSynchronize());
                    prepare_input(inputs.back());
                }
                check(cudaDeviceSynchronize());
                std::vector<std::string> workloads{"keyswitch", "square_relinearize_rescale", "square_rescale_keyswitch"};
                if (o.rotate) workloads.push_back("rotate_rows_shift1");
                if (!o.scenes.empty()) workloads = o.scenes;
                for (const auto& scene : workloads) {
                    auto run_one = [&](std::size_t b, std::vector<Cipher>& result) {
                        if (scene == "regular_bootstrapping_v2") {
                            result.push_back(op.regular_bootstrapping_v2(inputs[b], galois, relin,
                                dense_to_sparse.get(), sparse_to_dense.get(), execution));
                            return;
                        }
                        if (scene == "regular_bootstrapping") {
                            result.push_back(op.regular_bootstrapping(inputs[b], galois, relin, execution));
                            return;
                        }
                        result.emplace_back(context);
                        if (scene == "keyswitch") op.keyswitch(inputs[b], result.back(), switchkey, execution);
                        else if (scene == "rotate_rows_shift1") op.rotate_rows(inputs[b], result.back(), galois, 1, execution);
                        else {
                            op.multiply(inputs[b], inputs[b], result.back(), execution);
                            op.relinearize_inplace(result.back(), relin, execution);
                            op.rescale_inplace(result.back(), execution);
                            if (scene == "square_rescale_keyswitch")
                                op.keyswitch(result.back(), result.back(), switchkey, execution);
                        }
                    };
                    auto run_batch = [&](std::vector<Cipher>& result) {
                        // Independent ciphertexts submitted through the public scalar API,
                        // with one completion barrier for the entire batch.
                        result.reserve(batch);
                        for (std::size_t b = 0; b < batch; ++b) run_one(b, result);
                    };
                    if (o.prepare_only) {
                        // Keep full ciphertext metadata as well as words. This
                        // probe distinguishes input mutation, native instability,
                        // and provider divergence before any performance claim.
                        std::vector<Cipher> pristine = inputs;
                        check(cudaDeviceSynchronize());
                        std::vector<std::vector<Data64>> original(batch), reference_output(batch);
                        for (std::size_t b = 0; b < batch; ++b) pristine[b].get_data(original[b]);
                        check(cudaDeviceSynchronize());
                        bool have_reference = false, exact = true;
                        for (bool enabled : {false, true}) {
                            if ((o.prepare_backend == "native" && enabled) ||
                                (o.prepare_backend == "cubutterfly" && !enabled)) continue;
                            cubutterfly_heongpu::set_enabled(enabled);
                            for (int trial = 0; trial < std::max(2, o.warmup); ++trial) {
                                for (std::size_t b = 0; b < batch; ++b) inputs[b] = pristine[b];
                                check(cudaDeviceSynchronize());
                                std::cerr << "preparation_probe,phase=begin,N=" << n << ",B=" << batch
                                          << ",backend=" << (enabled ? "cubutterfly" : "native")
                                          << ",trial=" << trial << '\n';
                                std::vector<Cipher> result;
                                run_batch(result);
                                check(cudaDeviceSynchronize());
                                std::size_t input_changes = 0, output_changes = 0, metadata_changes = 0;
                                Errors error;
                                for (std::size_t b = 0; b < batch; ++b) {
                                    std::vector<Data64> after, output;
                                    inputs[b].get_data(after);
                                    result[b].get_data(output);
                                    check(cudaDeviceSynchronize());
                                    auto difference = [](const auto& x, const auto& y) {
                                        std::size_t d = x.size() > y.size() ? x.size()-y.size() : y.size()-x.size();
                                        for (std::size_t j = 0; j < std::min(x.size(), y.size()); ++j) d += x[j] != y[j];
                                        return d;
                                    };
                                    input_changes += difference(after, original[b]);
                                    metadata_changes += inputs[b].depth() != pristine[b].depth() ||
                                        inputs[b].scale() != pristine[b].scale() || inputs[b].size() != pristine[b].size();
                                    if (have_reference) output_changes += difference(output, reference_output[b]);
                                    else reference_output[b] = output;
                                    heongpu::Plaintext<Scheme> plain(context);
                                    ((scene == "keyswitch" || scene == "square_rescale_keyswitch") ? decryptor2 : decryptor).decrypt(plain, result[b]);
                                    std::vector<double> actual;
                                    encoder.decode(actual, plain);
                                    check(cudaDeviceSynchronize());
                                    auto expected = messages[b];
                                    if (scene == "square_relinearize_rescale" || scene == "square_rescale_keyswitch")
                                        for (auto& x : expected) x *= x;
                                    if (scene == "rotate_rows_shift1") std::rotate(expected.begin(), expected.begin()+1, expected.end());
                                    error.add(actual, expected);
                                }
                                have_reference = true;
                                exact = exact && output_changes == 0;
                                std::cerr << "preparation_probe,phase=result,N=" << n << ",B=" << batch
                                          << ",backend=" << (enabled ? "cubutterfly" : "native") << ",trial=" << trial
                                          << ",input_word_changes=" << input_changes << ",metadata_changes=" << metadata_changes
                                          << ",reference_output_changes=" << output_changes << ",maxabs=" << error.maxabs
                                          << ",relL2=" << error.relative() << '\n';
                                memory_phase("probe_call_complete");
                            }
                        }
                        if (!exact) throw std::runtime_error("preparation probe output is not repeatable/exact across selected backends");
                        continue;
                    }
                    cubutterfly_heongpu::freeze_plans(false);
                    // Always perform one preparation call, even with --warmup 0.
                    for (bool enabled : {false, true}) {
                        cubutterfly_heongpu::set_enabled(enabled);
                        std::cerr << "application_prepare,scene=" << scene << ",B=" << batch
                                  << ",backend=" << (enabled ? "cubutterfly" : "native") << '\n';
                        for (int i = 0; i < std::max(1, o.warmup); ++i) {
                            std::vector<Cipher> scratch;
                            run_batch(scratch);
                            check(cudaDeviceSynchronize());
                        }
                        std::size_t free_bytes = 0, total_bytes = 0;
                        check(cudaMemGetInfo(&free_bytes, &total_bytes));
                        std::cerr << "application_prepared,scene=" << scene << ",free_bytes=" << free_bytes << '\n';
                    }
                    cubutterfly_heongpu::freeze_plans(true);
                    if (o.graph) {
                        CapturedCall native_graph(stream), cu_graph(stream);
                        CapturedCall native_batch_graph(stream), cu_batch_graph(stream);
                        CapturedCall auto_graph(stream);
                        std::vector<CapturedCall*> variants;
                        if (!o.stage_batch_only) {
                            native_graph.prepare(run_batch, false);
                            cu_graph.prepare(run_batch, true);
                            if (native_graph.calls || !cu_graph.calls)
                                throw std::runtime_error("Graph NTT replacement attribution failed");
                            variants.push_back(&native_graph);
                            variants.push_back(&cu_graph);
                        }
                        if (o.stage_batch) {
                            // Capture the marked scalar graphs once.  Both
                            // stage backends consume the same graph structure;
                            // only their replacement NTT workspaces differ.
                            auto shared_scalar_calls = std::make_shared<
                                std::vector<std::unique_ptr<CapturedCall>>>();
                            std::vector<cubutterfly_heongpu::MarkedNttGraph> marked;
                            shared_scalar_calls->reserve(batch);
                            marked.reserve(batch);
                            for (std::size_t b = 0; b < batch; ++b) {
                                auto scalar = std::make_unique<CapturedCall>(stream);
                                scalar->prepare([&](std::vector<Cipher>& result) {
                                    run_one(b, result);
                                }, true, true);
                                if (scalar->outputs.size() != 1)
                                    throw std::logic_error("scalar capture output count");
                                marked.push_back({scalar->graph, scalar->stages});
                                shared_scalar_calls->push_back(std::move(scalar));
                            }
                            if (!o.stage_batch_only)
                                native_batch_graph.prepare_batch(
                                    marked, shared_scalar_calls, cubutterfly_heongpu::StageBackend::Native);
                            cu_batch_graph.prepare_batch(
                                marked, shared_scalar_calls, o.stage_backend);
                            if ((!o.stage_batch_only &&
                                 (native_batch_graph.polynomials != cu_graph.polynomials ||
                                  cu_batch_graph.polynomials != cu_graph.polynomials ||
                                  native_batch_graph.calls * batch != cu_graph.calls ||
                                  cu_batch_graph.calls * batch != cu_graph.calls)) ||
                                (o.stage_batch_only && cu_batch_graph.calls == 0))
                                throw std::runtime_error("stage batching changed transform work count");
                            if (!o.stage_batch_only) variants.push_back(&native_batch_graph);
                            variants.push_back(&cu_batch_graph);
                            std::cerr << "stage_batch_graph,backend="
                                      << (o.stage_backend == cubutterfly_heongpu::StageBackend::Native ? "native" : "cubutterfly")
                                      << ",scene=" << scene << ",N=" << n << ",B=" << batch
                                      << ",stages=" << cu_batch_graph.calls << ",polynomials=" << cu_batch_graph.polynomials
                                      << ",native_loop_nodes=" << (o.stage_batch_only ? 0 : native_graph.kernel_nodes())
                                      << ",cu_loop_nodes=" << (o.stage_batch_only ? 0 : cu_graph.kernel_nodes())
                                      << ",native_batch_nodes=" << (o.stage_batch_only ? 0 : native_batch_graph.kernel_nodes())
                                      << ",cu_batch_nodes=" << cu_batch_graph.kernel_nodes() << '\n';
                        }
                        cubutterfly_heongpu::set_enabled(false);
                        std::vector<std::vector<Data64>> original(batch);
                        for (std::size_t b = 0; b < batch; ++b) inputs[b].get_data(original[b]);
                        check(cudaDeviceSynchronize());
                        auto restore_original = [&] {
                            for (std::size_t b = 0; b < batch; ++b)
                                check(cudaMemcpyAsync(inputs[b].data(), original[b].data(),
                                                      original[b].size() * sizeof(Data64),
                                                      cudaMemcpyHostToDevice, stream));
                            check(cudaStreamSynchronize(stream));
                        };
                        auto validate_graph = [&] {
                            // Full bootstrap may consume ciphertext storage
                            // in place. Snapshot the current logical inputs
                            // and restore them before every eager/Graph
                            // comparison so the validation compares the same
                            // call, rather than successive calls on mutated
                            // ciphertexts.
                            std::vector<std::vector<Data64>> current(batch);
                            for (std::size_t b = 0; b < batch; ++b) inputs[b].get_data(current[b]);
                            check(cudaDeviceSynchronize());
                            auto restore_current = [&] {
                                for (std::size_t b = 0; b < batch; ++b)
                                    check(cudaMemcpyAsync(inputs[b].data(), current[b].data(),
                                                          current[b].size() * sizeof(Data64),
                                                          cudaMemcpyHostToDevice, stream));
                                check(cudaStreamSynchronize(stream));
                            };
                            restore_current();
                            std::vector<Cipher> eager;
                            run_batch(eager);
                            check(cudaStreamSynchronize(stream));
                            std::vector<std::vector<std::vector<Data64>>> saved(variants.size(),
                                std::vector<std::vector<Data64>>(batch));
                            for (std::size_t v = 0; v < variants.size(); ++v) {
                                restore_current();
                                variants[v]->launch();
                                check(cudaStreamSynchronize(stream));
                                for (std::size_t b = 0; b < batch; ++b) {
                                    if (mismatches(eager[b], variants[v]->output(b)))
                                        throw std::runtime_error("Graph exact check failed: variant=" + std::to_string(v) +
                                                                 ", ciphertext=" + std::to_string(b));
                                    variants[v]->output(b).get_data(saved[v][b]);
                                }
                                check(cudaDeviceSynchronize());
                            }
                            for (std::size_t v = 0; v < variants.size(); ++v) {
                                for (std::size_t b = 0; b < batch; ++b) {
                                    std::vector<Data64> after;
                                    variants[v]->output(b).get_data(after);
                                    check(cudaDeviceSynchronize());
                                    if (after != saved[v][b])
                                        throw std::runtime_error("Graph output lifetime/arena isolation check failed");
                                }
                            }
                        };
                        validate_graph();
                        if (o.auto_schedule) {
                            auto_graph.schedule = std::make_unique<cubutterfly_heongpu::GraphSchedule>(
                                std::vector<cubutterfly_heongpu::GraphScheduleCandidate>{
                                    {"loop", [&](cudaStream_t s) { cu_graph.launch(s); }},
                                    {"stage_batch", [&](cudaStream_t s) { cu_batch_graph.launch(s); }}});
                            cubutterfly_heongpu::GraphScheduleOptions budget;
                            budget.warmup = o.warmup;
                            budget.rounds = o.schedule_rounds;
                            budget.replays = o.schedule_replays;
                            thermal_guard.ready();
                            const auto selection_start = std::chrono::steady_clock::now();
                            auto_graph.schedule->tune(stream, budget,
                                [&] { thermal_guard.ready(); },
                                [&] { thermal_guard.check_sample(); });
                            const double selection_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - selection_start).count();
                            const auto winner = auto_graph.schedule->selected_index();
                            auto_graph.selected = winner == 0 ? &cu_graph : &cu_batch_graph;
                            auto_graph.calls = auto_graph.selected->calls;
                            auto_graph.polynomials = auto_graph.selected->polynomials;
                            for (std::size_t c = 0; c < auto_graph.schedule->measurements().size(); ++c) {
                                const auto& m = auto_graph.schedule->measurements()[c];
                                for (std::size_t r = 0; r < m.round_ms.size(); ++r)
                                    selection_file << scene << ',' << n << ',' << batch << ',' << method << ','
                                        << m.name << ',' << r << ',' << std::setprecision(12) << m.round_ms[r] << ','
                                        << m.median_ms << ',' << (c == winner) << ',' << selection_ms << ','
                                        << budget.replays << ",measured-full-graph-request\n";
                            }
                            selection_file.flush();
                            std::cerr << "graph_schedule,scene=" << scene << ",N=" << n << ",B=" << batch
                                      << ",selected=" << auto_graph.schedule->selected_candidate().name
                                      << ",confidence=measured-full-graph-request,calibrated=0,selection_ms="
                                      << selection_ms << '\n';
                            variants.push_back(&auto_graph);
                            validate_graph();
                        }
                        // Change real encrypted inputs at the same addresses, then
                        // restore them. Eager recomputation is the independent
                        // graph-lifetime control; it uses the original NTT library.
                        for (std::size_t b = 0; b < batch; ++b) {
                            auto changed = messages[b];
                            for (auto& x : changed) x = -x + 0.125;
                            heongpu::Plaintext<Scheme> plain(context);
                            encoder.encode(plain, changed, std::ldexp(1.0, 40));
                            Cipher alternate(context);
                            encryptor.encrypt(alternate, plain);
                            check(cudaDeviceSynchronize());
                            prepare_input(alternate);
                            check(cudaStreamSynchronize(stream));
                            check(cudaMemcpyAsync(inputs[b].data(), alternate.data(), original[b].size() * sizeof(Data64), cudaMemcpyDeviceToDevice, stream));
                            check(cudaStreamSynchronize(stream));
                        }
                        validate_graph();
                        for (std::size_t b = 0; b < batch; ++b)
                            check(cudaMemcpyAsync(inputs[b].data(), original[b].data(), original[b].size() * sizeof(Data64), cudaMemcpyHostToDevice, stream));
                        check(cudaStreamSynchronize(stream));
                        validate_graph();
                        std::cerr << "measurement_ready,scene=" << scene << ",N=" << n << ",B=" << batch << '\n';
                        thermal_guard.ready();
                        for (int i = 0; i < o.warmup; ++i)
                            for (auto* variant : variants) variant->launch();
                        check(cudaStreamSynchronize(stream));
                        thermal_guard.log("after_warmup");
                        thermal_guard.check_sample();
                        std::vector<double> times(variants.size(), 0);
                        const char* isolated_stage_name =
                            o.stage_backend == cubutterfly_heongpu::StageBackend::Native
                                ? "native_stage_batch" : "cu_stage_batch";
                        const std::vector<const char*> variant_names = o.stage_batch_only
                            ? std::vector<const char*>{isolated_stage_name}
                            : std::vector<const char*>{"native_loop", "cu_loop", "native_stage_batch", "cu_stage_batch", "cu_auto"};
                        double eager_total_ms = 0;
                        if (o.trace_measurement) check(cudaProfilerStart());
                        for (int rep = 0; rep < o.repetitions; ++rep) {
                            // Cool between complete rotated rounds, outside every
                            // CUDA event interval. Keep the full batch and replay
                            // count; a throttled sample still invalidates the run.
                            thermal_guard.ready();
                            if (o.stage_batch_only) {
                                restore_original();
                                std::vector<Cipher> eager;
                                eager_total_ms += elapsed([&] { run_batch(eager); });
                                thermal_guard.check_sample();
                            }
                            for (std::size_t offset = 0; offset < variants.size(); ++offset) {
                                const std::size_t v = (rep + offset) % variants.size();
                                const char* name = variant_names.at(v);
                                const std::string label = "ckks_measure/" + scene + "/" + name + "/" + std::to_string(rep);
                                if (o.trace_measurement) nvtxRangePushA(label.c_str());
                                if (o.stage_batch_only) restore_original();
                                const double sample_ms = variants[v]->measure(o.graph_replays);
                                if (o.trace_measurement) nvtxRangePop();
                                times[v] += sample_ms;
                                if (o.auto_schedule) {
                                    samples_file << scene << ',' << n << ',' << batch << ',' << method << ','
                                        << name << ',' << rep << ',' << std::setprecision(12) << sample_ms
                                        << ',' << o.graph_replays << '\n';
                                }
                                thermal_guard.check_sample();
                            }
                        }
                        if (o.trace_measurement) check(cudaProfilerStop());
                        thermal_guard.log("after_measurement");
                        if (o.auto_schedule) samples_file.flush();
                        std::vector<Errors> errors(variants.size());
                        for (std::size_t b = 0; b < batch; ++b) {
                            auto expected = messages[b];
                            if (scene == "square_relinearize_rescale" || scene == "square_rescale_keyswitch") for (auto& x : expected) x *= x;
                            if (scene == "rotate_rows_shift1") std::rotate(expected.begin(), expected.begin() + 1, expected.end());
                            auto validate = [&](Cipher& cipher, Errors& error) {
                                heongpu::Plaintext<Scheme> plain(context);
                                ((scene == "keyswitch" || scene == "square_rescale_keyswitch") ? decryptor2 : decryptor).decrypt(plain, cipher);
                                std::vector<double> actual;
                                encoder.decode(actual, plain);
                                check(cudaDeviceSynchronize());
                                error.add(actual, expected);
                            };
                            for (std::size_t v = 0; v < variants.size(); ++v)
                                validate(variants[v]->output(b), errors[v]);
                        }
                        Errors eager_error;
                        std::size_t stage_only_mismatch = 0;
                        if (o.stage_batch_only) {
                            restore_original();
                            std::vector<Cipher> eager;
                            run_batch(eager);
                            check(cudaStreamSynchronize(stream));
                            restore_original();
                            variants.front()->launch();
                            check(cudaStreamSynchronize(stream));
                            for (std::size_t b = 0; b < batch; ++b) {
                                stage_only_mismatch += mismatches(eager[b], variants.front()->output(b));
                                auto expected = messages[b];
                                if (scene == "square_relinearize_rescale" || scene == "square_rescale_keyswitch")
                                    for (auto& x : expected) x *= x;
                                if (scene == "rotate_rows_shift1")
                                    std::rotate(expected.begin(), expected.begin() + 1, expected.end());
                                auto validate_eager = [&](Cipher& cipher) {
                                    heongpu::Plaintext<Scheme> plain(context);
                                    ((scene == "keyswitch" || scene == "square_rescale_keyswitch")
                                         ? decryptor2 : decryptor).decrypt(plain, cipher);
                                    std::vector<double> actual;
                                    encoder.decode(actual, plain);
                                    check(cudaDeviceSynchronize());
                                    eager_error.add(actual, expected);
                                };
                                validate_eager(eager[b]);
                            }
                        }
                        auto report = [&](std::size_t reference, std::size_t treatment,
                                          const char* baseline, const char* submission) {
                            std::size_t mismatch_count = 0;
                            for (std::size_t b = 0; b < batch; ++b)
                                mismatch_count += mismatches(variants[reference]->output(b), variants[treatment]->output(b));
                            *out << scene << ",NTT," << baseline << ',' << n << ',' << batch << ",cuda_graph," << std::setprecision(12)
                                 << times[reference] / o.repetitions << ',' << times[treatment] / o.repetitions << ','
                                 << times[reference] / times[treatment] << ',' << (mismatch_count == 0) << ',' << mismatch_count << ','
                                 << errors[reference].maxabs << ',' << errors[treatment].maxabs << ','
                                 << errors[reference].relative() << ',' << errors[treatment].relative() << ','
                                 << variants[treatment]->calls * o.repetitions * o.graph_replays << ','
                                 << variants[treatment]->polynomials * o.repetitions * o.graph_replays << ',' << o.repetitions
                                 << ",none," << q_bits << ',' << p_bits << ",1099511627776," << method << ',' << submission
                                 << ",reserved_scratch_changed_restored_input_exact,"
                                 << variants[reference]->kernel_nodes() << ',' << variants[treatment]->kernel_nodes() << ','
                                 << ntt_execution << ',' << o.graph_replays << ',' << o.parameter_set << ','
                                 << (o.stage_backend == cubutterfly_heongpu::StageBackend::Native ? "native" : "cubutterfly")
                                 << '\n' << std::flush;
                            if (mismatch_count) throw std::runtime_error("Graph ciphertext mismatch after timing");
                        };
                        if (o.stage_batch_only) {
                            const auto* treatment = variants.front();
                            *out << scene << ",NTT,HEonGPU_native_eager," << n << ',' << batch
                                 << ",cuda_graph_vs_eager," << std::setprecision(12)
                                 << eager_total_ms / o.repetitions << ',' << times.front() / o.repetitions << ','
                                 << eager_total_ms / times.front() << ',' << (stage_only_mismatch == 0) << ','
                                 << stage_only_mismatch << ',' << eager_error.maxabs << ',' << errors.front().maxabs << ','
                                 << eager_error.relative() << ',' << errors.front().relative() << ','
                                 << treatment->calls * o.repetitions * o.graph_replays << ','
                                 << treatment->polynomials * o.repetitions * o.graph_replays << ',' << o.repetitions
                                 << ",none," << q_bits << ',' << p_bits << ",1099511627776," << method
                                 << (o.stage_backend == cubutterfly_heongpu::StageBackend::Native
                                         ? ",stage_batch_only_native_vs_eager_native"
                                         : ",stage_batch_only_cu_vs_eager_native")
                                 << ",reserved_scratch_changed_restored_input_exact,0,"
                                 << treatment->kernel_nodes() << ',' << ntt_execution << ',' << o.graph_replays << ','
                                 << o.parameter_set << ','
                                 << (o.stage_backend == cubutterfly_heongpu::StageBackend::Native ? "native" : "cubutterfly")
                                 << '\n' << std::flush;
                            if (stage_only_mismatch)
                                throw std::runtime_error("stage-only graph ciphertext mismatch after timing");
                        } else {
                            report(0, 1, "HEonGPU_native", "independent_public_API_loop");
                            if (o.stage_batch) {
                                report(0, 3, "HEonGPU_native_loop", "independent_graphs_stage_batch_pack_ntt_scatter");
                                report(2, 3, "HEonGPU_native_stage_batch", "matched_stage_batch_pack_ntt_scatter");
                            }
                        }
                        if (o.auto_schedule) {
                            report(0, 4, "HEonGPU_native_loop", "measured_auto_loop_or_stage_batch");
                            report(2, 4, "HEonGPU_native_stage_batch", "measured_auto_loop_or_stage_batch");
                        }
                        cubutterfly_heongpu::freeze_plans(false);
                        continue;
                    }
                    double native_ms = 0, cu_ms = 0;
                    std::size_t mismatch_count = 0, calls = 0, polynomials = 0;
                    int completed = 0;
                    Errors native_error, cu_error;
                    for (int rep = 0; rep < o.repetitions; ++rep) {
                        // Some complete HE workflows consume their input
                        // ciphertext in place (the bootstrap v2 path is one
                        // such workflow).  Snapshot and restore outside the
                        // timed region so each backend sees exactly the same
                        // ciphertext on every repetition.  Comparing the
                        // outputs without this barrier would compare native
                        // against treatment-after-native, rather than two
                        // implementations of one call.
                        std::vector<std::vector<Data64>> input_snapshot(batch);
                        for (std::size_t b = 0; b < batch; ++b)
                            inputs[b].get_data(input_snapshot[b]);
                        check(cudaDeviceSynchronize());
                        auto restore_inputs = [&] {
                            for (std::size_t b = 0; b < batch; ++b)
                                check(cudaMemcpyAsync(inputs[b].data(), input_snapshot[b].data(),
                                                      input_snapshot[b].size() * sizeof(Data64),
                                                      cudaMemcpyHostToDevice, stream));
                            check(cudaStreamSynchronize(stream));
                        };
                        std::vector<Cipher> native, treatment;
                        auto measure = [&](bool enabled) {
                            cubutterfly_heongpu::set_enabled(enabled);
                            cubutterfly_heongpu::reset_counters();
                            const double ms = elapsed([&] { run_batch(enabled ? treatment : native); });
                            if (enabled) {
                                cu_ms += ms;
                                calls += cubutterfly_heongpu::replacement_calls();
                                polynomials += cubutterfly_heongpu::replacement_polynomials();
                            } else {
                                native_ms += ms;
                                if (cubutterfly_heongpu::replacement_calls()) throw std::runtime_error("native path replaced NTT");
                            }
                        };
                        restore_inputs();
                        measure(rep % 2 != 0);
                        restore_inputs();
                        measure(rep % 2 == 0);
                        ++completed;
                        cubutterfly_heongpu::set_enabled(false);
                        for (std::size_t b = 0; b < batch; ++b) {
                            mismatch_count += mismatches(native[b], treatment[b]);
                            auto expected = messages[b];
                            if (scene == "square_relinearize_rescale" || scene == "square_rescale_keyswitch") for (auto& x : expected) x *= x;
                            if (scene == "rotate_rows_shift1") std::rotate(expected.begin(), expected.begin() + 1, expected.end());
                            auto validate = [&](Cipher& cipher, Errors& error) {
                                heongpu::Plaintext<Scheme> plain(context);
                                ((scene == "keyswitch" || scene == "square_rescale_keyswitch") ? decryptor2 : decryptor).decrypt(plain, cipher);
                                std::vector<double> actual;
                                encoder.decode(actual, plain);
                                check(cudaDeviceSynchronize());
                                error.add(actual, expected);
                            };
                            validate(native[b], native_error);
                            validate(treatment[b], cu_error);
                        }
                        if (mismatch_count) break;
                    }
                    const bool exact = mismatch_count == 0;
                    *out << scene << ",NTT,HEonGPU_native," << n << ',' << batch << ",eager," << std::setprecision(12)
                         << native_ms / completed << ',' << cu_ms / completed << ',' << native_ms / cu_ms << ','
                         << exact << ',' << mismatch_count << ',' << native_error.maxabs << ',' << cu_error.maxabs << ','
                         << native_error.relative() << ',' << cu_error.relative() << ',' << calls << ',' << polynomials << ','
                         << completed << ",none," << q_bits << ',' << p_bits << ",1099511627776," << method
                         << ",independent_public_API_loop,not_requested,,," << ntt_execution << ",0," << o.parameter_set << ",none\n" << std::flush;
                    if (!exact) throw std::runtime_error("raw ciphertext mismatch in " + scene);
                    if (!calls) throw std::runtime_error("treatment did not replace any NTT calls in " + scene);
                    cubutterfly_heongpu::freeze_plans(false);
                }
            }
            cubutterfly_heongpu::clear_cache();
        }
        if (stream) check(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 2;
    }
}
