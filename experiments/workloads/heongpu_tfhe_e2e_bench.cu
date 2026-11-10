// Complete-call TFHE benchmark for the pinned HEonGPU public API.
//
// This runner deliberately times the public gate/circuit call from the host
// entry point through CUDA completion. Key generation, encryption, decryption,
// and host transfers are outside the timed region and are reported separately.
// Graph mode captures the same complete public call with reserved scratch.
// The optional overlay fixes native/template kernel identity before replay.

#include <heongpu/heongpu.hpp>
#include <cuda_runtime.h>
#include "tfhe_graph_call.hpp"
#ifdef CUBUTTERFLY_TFHE_SHARED
#include "../../adapters/heongpu/tfhe_shared_ntt.cuh"
#endif

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr auto kScheme = heongpu::Scheme::TFHE;

struct Options {
    std::vector<std::size_t> shapes{1, 8, 64, 512};
    std::vector<std::string> gates{"NAND", "AND", "NOR", "OR", "XNOR",
                                   "XOR", "NOT", "MUX"};
    int warmup = 5;
    int repetitions = 20;
    std::uint32_t seed = 20261018u;
    std::string output;
    std::string expected_backend;
    std::string execution = "eager";
    unsigned graph_replays = 1;
    std::vector<std::size_t> circuit_depths{2, 4, 8};
};

std::vector<std::string> split(const std::string& value) {
    std::vector<std::string> result;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (!item.empty()) result.push_back(item);
    }
    return result;
}

std::vector<std::size_t> parse_shapes(const std::string& value) {
    std::vector<std::size_t> result;
    for (const auto& item : split(value)) {
        const auto parsed = std::stoull(item);
        if (parsed == 0) throw std::invalid_argument("shape must be positive");
        result.push_back(static_cast<std::size_t>(parsed));
    }
    if (result.empty()) throw std::invalid_argument("at least one shape is required");
    return result;
}

void print_help(const char* name) {
    std::cout << "Usage: " << name
              << " [--shapes 1,8,64,512] [--gates NAND,XOR,MUX]"
                 " [--warmup N] [--repetitions N] [--seed N] [--output FILE]\n";
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument(argv[i]);
        auto value = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument(std::string(flag) + " needs a value");
            return argv[++i];
        };
        if (argument == "--help" || argument == "-h") {
            print_help(argv[0]);
            std::exit(0);
        } else if (argument == "--shapes") {
            options.shapes = parse_shapes(value("--shapes"));
        } else if (argument == "--gates") {
            options.gates = split(value("--gates"));
        } else if (argument == "--warmup") {
            options.warmup = std::stoi(value("--warmup"));
        } else if (argument == "--repetitions") {
            options.repetitions = std::stoi(value("--repetitions"));
        } else if (argument == "--seed") {
            options.seed = static_cast<std::uint32_t>(std::stoul(value("--seed")));
        } else if (argument == "--output") {
            options.output = value("--output");
        } else if (argument == "--expected-backend") {
            options.expected_backend = value("--expected-backend");
        } else if (argument == "--execution") {
            options.execution = value("--execution");
        } else if (argument == "--graph-replays") {
            options.graph_replays = std::stoul(value("--graph-replays"));
        } else if (argument == "--circuit-depths") {
            const auto depths = value("--circuit-depths");
            options.circuit_depths = depths.empty() ? std::vector<std::size_t>{} : parse_shapes(depths);
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }
    if (options.warmup < 0 || options.repetitions <= 0)
        throw std::invalid_argument("warmup must be non-negative and repetitions positive");
    if ((options.execution != "eager" && options.execution != "graph") || !options.graph_replays)
        throw std::invalid_argument("execution must be eager/graph and graph-replays must be positive");
    for (auto depth : options.circuit_depths)
        if (depth < 2) throw std::invalid_argument("circuit depths must be >=2");
    return options;
}

bool bool_result(const std::string& gate, bool a, bool b, bool control) {
    if (gate == "NAND") return !(a && b);
    if (gate == "AND") return a && b;
    if (gate == "NOR") return !(a || b);
    if (gate == "OR") return a || b;
    if (gate == "XNOR") return !(a ^ b);
    if (gate == "XOR") return a ^ b;
    if (gate == "NOT") return !a;
    if (gate == "MUX") return control ? a : b;
    throw std::invalid_argument("unsupported gate: " + gate);
}

template <typename Callable>
double elapsed_complete_call(Callable&& callable) {
    // Host wall time plus an explicit synchronize is intentional: it captures
    // launches, host-side adapter work, allocations, and all CUDA work.
    const auto start = std::chrono::steady_clock::now();
    callable();
    const auto status = cudaDeviceSynchronize();
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
#ifdef CUBUTTERFLY_TFHE_SHARED
        const std::string backend = "cubutterfly-shared";
        std::cerr << "backend,cubutterfly-shared;core,pinned-gpuntt-ct-gs;organization,cta-shared-stage-barriers\n";
        if (options.execution != "graph")
            throw std::invalid_argument("fixed cubutterfly-shared backend requires --execution graph");
#else
        const std::string backend = "native";
        std::cerr << "backend,native\n";
#endif
        if (!options.expected_backend.empty() && options.expected_backend != backend)
            throw std::runtime_error("binary backend differs from requested backend; rebuild required");
        std::ofstream file;
        std::ostream* output = &std::cout;
        if (!options.output.empty()) {
            file.open(options.output);
            if (!file) throw std::runtime_error("cannot open output: " + options.output);
            output = &file;
        }
        *output << "workload,gate,batch,repetitions,avg_ms,correct,backend,execution,graph_nodes,bootstrap_nodes,validation\n";

        const auto setup_start = std::chrono::steady_clock::now();
        heongpu::HEContext<kScheme> context = heongpu::GenHEContext<kScheme>();
        heongpu::HEKeyGenerator<kScheme> keygen(context);
        heongpu::Secretkey<kScheme> secret_key(context);
        keygen.generate_secret_key(secret_key);
        heongpu::Bootstrappingkey<kScheme> boot_key(context);
        keygen.generate_bootstrapping_key(boot_key, secret_key);
        heongpu::HEEncryptor<kScheme> encryptor(context, secret_key);
        heongpu::HEDecryptor<kScheme> decryptor(context, secret_key);
        heongpu::HELogicOperator<kScheme> logic(context);
        const auto setup_stop = std::chrono::steady_clock::now();
        const double setup_ms = std::chrono::duration<double, std::milli>(setup_stop - setup_start).count();
        std::cerr << "setup_ms," << std::setprecision(10) << setup_ms << "\n";

        std::mt19937 generator(options.seed);
        std::bernoulli_distribution bit(0.5);
        for (const std::size_t shape : options.shapes) {
            std::vector<bool> input1(shape), input2(shape), control(shape);
            for (std::size_t i = 0; i < shape; ++i) {
                input1[i] = bit(generator);
                input2[i] = bit(generator);
                control[i] = bit(generator);
            }
            heongpu::Ciphertext<kScheme> ct1(context), ct2(context), ct3(context);
            const double encrypt_ms = elapsed_complete_call([&] {
                encryptor.encrypt(ct1, input1);
                encryptor.encrypt(ct2, input2);
                encryptor.encrypt(ct3, control);
            });
            std::cerr << "encrypt_ms," << shape << ',' << std::setprecision(10) << encrypt_ms << "\n";

            if (options.execution == "graph") {
                using namespace cubutterfly::heongpu_tfhe;
                Stream graph_stream;
                heongpu::ExecutionOptions execution;
                execution.set_stream(graph_stream.value);
                using Cipher = heongpu::Ciphertext<kScheme>;
                std::vector<std::vector<std::vector<bool>>> messages{{input1, input2, control}};
                messages.push_back({std::vector<bool>(shape, false), std::vector<bool>(shape, false), std::vector<bool>(shape, false)});
                messages.push_back({std::vector<bool>(shape, true), std::vector<bool>(shape, false), std::vector<bool>(shape, true)});
                messages.push_back({std::vector<bool>(shape, true), std::vector<bool>(shape, true), std::vector<bool>(shape, false)});
                std::vector<std::vector<std::unique_ptr<Cipher>>> encrypted(messages.size());
                encrypted[0].emplace_back(std::make_unique<Cipher>(ct1));
                encrypted[0].emplace_back(std::make_unique<Cipher>(ct2));
                encrypted[0].emplace_back(std::make_unique<Cipher>(ct3));
                for (std::size_t v = 1; v < messages.size(); ++v)
                    for (unsigned j = 0; j < 3; ++j) {
                        encrypted[v].emplace_back(std::make_unique<Cipher>(context));
                        encryptor.encrypt(*encrypted[v].back(), messages[v][j]);
                    }
                check(cudaDeviceSynchronize());
                std::vector<std::pair<std::string, unsigned>> scenes;
                for (const auto& gate : options.gates) scenes.emplace_back(gate, 0);
                for (auto depth : options.circuit_depths) scenes.emplace_back("nand_xor_chain_d" + std::to_string(depth), depth);
                auto gate_call = [&](const std::string& gate, Cipher& a, Cipher& b, Cipher& c, Cipher& result) {
                    if (gate == "NAND") logic.NAND(a, b, result, boot_key, execution);
                    else if (gate == "AND") logic.AND(a, b, result, boot_key, execution);
                    else if (gate == "NOR") logic.NOR(a, b, result, boot_key, execution);
                    else if (gate == "OR") logic.OR(a, b, result, boot_key, execution);
                    else if (gate == "XNOR") logic.XNOR(a, b, result, boot_key, execution);
                    else if (gate == "XOR") logic.XOR(a, b, result, boot_key, execution);
                    else if (gate == "NOT") logic.NOT(a, result, execution);
                    else if (gate == "MUX") logic.MUX(a, b, c, result, boot_key, execution);
                    else throw std::invalid_argument("unsupported gate: " + gate);
                };
                for (const auto& scene : scenes) {
                    const auto& gate = scene.first;
                    const unsigned depth = scene.second;
                    const std::size_t bootstraps = depth ? depth : gate == "NOT" ? 0 : gate == "MUX" ? 2 : 1;
                    auto run = [&](Outputs& nodes) {
                        for (unsigned d = 0; d < (depth ? depth : 1); ++d)
                            nodes.emplace_back(std::make_unique<Cipher>(context, execution));
                        if (!depth) gate_call(gate, ct1, ct2, ct3, *nodes[0]);
                        else {
                            gate_call("NAND", ct1, ct2, ct3, *nodes[0]);
                            gate_call("XOR", *nodes[0], ct1, ct3, *nodes[1]);
                            for (unsigned d = 2; d < depth; ++d)
                                gate_call("NAND", *nodes[d - 1], ct2, ct3, *nodes[d]);
                        }
                    };
                    auto expected = [&](std::size_t v) {
                        std::vector<bool> result(shape);
                        for (std::size_t i = 0; i < shape; ++i) {
                            const bool a = messages[v][0][i], b = messages[v][1][i], c = messages[v][2][i];
                            bool value = depth ? !(a && b) : bool_result(gate, a, b, c);
                            if (depth) value ^= a;
                            for (unsigned d = 2; d < depth; ++d) value = !(value && b);
                            result[i] = value;
                        }
                        return result;
                    };
                    const std::vector<std::string> names =
#ifdef CUBUTTERFLY_TFHE_SHARED
                        {"native-original-kernel", "cubutterfly-shared"};
#else
                        {"native-original-library"};
#endif
                    std::vector<std::unique_ptr<GraphCall>> graphs;
                    for (std::size_t g = 0; g < names.size(); ++g) {
                        graphs.emplace_back(std::make_unique<GraphCall>(graph_stream.value));
                        graphs.back()->prepare(run, [&](cudaGraph_t graph) -> std::size_t {
#ifdef CUBUTTERFLY_TFHE_SHARED
                            return heongpu::cubutterfly_tfhe_fix_graph(graph, g != 0, bootstraps);
#else
                            return 0;
#endif
                        });
                    }
                    auto validate = [&](std::size_t g, const std::vector<bool>& reference) {
                        std::vector<bool> actual;
                        decryptor.decrypt(*graphs[g]->outputs.back(), actual);
                        check(cudaDeviceSynchronize());
                        if (actual != reference) throw std::runtime_error("TFHE Graph exact/lifetime validation failed: " + gate + "/" + names[g]);
                    };
                    std::vector<std::vector<bool>> last(names.size());
                    for (const unsigned v : {0, 1, 2, 3, 0}) {
                        // Upstream same-size copy assignment retains the captured
                        // DeviceVector addresses while replacing their contents.
                        ct1 = *encrypted[v][0]; ct2 = *encrypted[v][1]; ct3 = *encrypted[v][2];
                        check(cudaDeviceSynchronize());
                        for (std::size_t g = 0; g < graphs.size(); ++g) {
                            graphs[g]->launch(); check(cudaStreamSynchronize(graph_stream.value));
                            last[g] = expected(v);
                            for (std::size_t other = 0; other < graphs.size(); ++other)
                                if (!last[other].empty()) validate(other, last[other]);
                        }
                    }
                    for (int w = 0; w < options.warmup; ++w)
                        for (auto& graph : graphs) graph->launch();
                    check(cudaStreamSynchronize(graph_stream.value));
                    std::vector<double> times(graphs.size());
                    for (int rep = 0; rep < options.repetitions; ++rep)
                        for (std::size_t offset = 0; offset < graphs.size(); ++offset) {
                            const std::size_t g = (rep + offset) % graphs.size();
                            times[g] += graphs[g]->measure(options.graph_replays);
                            validate(g, expected(0));
                        }
                    for (std::size_t g = 0; g < graphs.size(); ++g)
                        *output << (depth ? "tfhe_circuit," : "tfhe_gate,") << gate << ',' << shape << ','
                                << options.repetitions << ',' << std::setprecision(10) << times[g] / options.repetitions
                                << ",1," << names[g] << ",cuda_graph," << graphs[g]->nodes << ',' << graphs[g]->bootstrap_nodes
                                << ",exact_changed_restored_isolated_outputs\n" << std::flush;
                }
                continue;
            }

            const std::vector<std::string> backends =
#ifdef CUBUTTERFLY_TFHE_SHARED
                {"native-helper", "cubutterfly-shared"};
#else
                {"native"};
#endif
            for (const auto& measured_backend : backends) {
            for (const auto& gate : options.gates) {
                std::vector<bool> expected(shape);
                for (std::size_t i = 0; i < shape; ++i)
                    expected[i] = bool_result(gate, input1[i], input2[i], control[i]);

                auto run = [&](heongpu::Ciphertext<kScheme>& result) {
                    if (gate == "NAND") logic.NAND(ct1, ct2, result, boot_key);
                    else if (gate == "AND") logic.AND(ct1, ct2, result, boot_key);
                    else if (gate == "NOR") logic.NOR(ct1, ct2, result, boot_key);
                    else if (gate == "OR") logic.OR(ct1, ct2, result, boot_key);
                    else if (gate == "XNOR") logic.XNOR(ct1, ct2, result, boot_key);
                    else if (gate == "XOR") logic.XOR(ct1, ct2, result, boot_key);
                    else if (gate == "NOT") logic.NOT(ct1, result);
                    else if (gate == "MUX") logic.MUX(ct1, ct2, ct3, result, boot_key);
                    else throw std::invalid_argument("unsupported gate: " + gate);
                };
                for (int i = 0; i < options.warmup; ++i) {
                    heongpu::Ciphertext<kScheme> warmup_result(context);
                    run(warmup_result);
                }
                cudaDeviceSynchronize();
                double total = 0.0;
                bool correct = true;
                for (int i = 0; i < options.repetitions; ++i) {
                    heongpu::Ciphertext<kScheme> result(context);
                    total += elapsed_complete_call([&] { run(result); });
                    {
                        std::vector<bool> decrypted;
                        decryptor.decrypt(result, decrypted);
                        cudaDeviceSynchronize();
                        correct = correct && decrypted == expected;
                    }
                }
                *output << "tfhe_gate," << gate << ',' << shape << ','
                        << options.repetitions << ','
                        << std::setprecision(10)
                        << total / static_cast<double>(options.repetitions) << ','
                        << (correct ? 1 : 0) << ',' << measured_backend << ",eager,0,0,exact_decrypt\n";
                if (!correct)
                    throw std::runtime_error("exact TFHE correctness check failed for " + gate);
            }

            // A depth-2/4/8 NAND chain is a complete application-level circuit,
            // rather than a bootstrap-only attribution. Each depth starts from
            // the same encrypted inputs and compares the final decrypted bit
            // vector against the CPU Boolean recurrence.
            for (const int depth : options.circuit_depths) {
                double total = 0.0;
                bool correct = true;
                for (int i = 0; i < options.warmup + options.repetitions; ++i) {
                    std::vector<std::unique_ptr<heongpu::Ciphertext<kScheme>>> nodes;
                    nodes.reserve(static_cast<std::size_t>(depth));
                    for (int node = 0; node < depth; ++node)
                        nodes.emplace_back(std::make_unique<heongpu::Ciphertext<kScheme>>(context));
                    const bool timed = i >= options.warmup;
                    const auto call = [&] {
                        logic.NAND(ct1, ct2, *nodes[0], boot_key);
                        logic.XOR(*nodes[0], ct1, *nodes[1], boot_key);
                        for (int d = 2; d < depth; ++d)
                            logic.NAND(*nodes[static_cast<std::size_t>(d - 1)], ct2,
                                       *nodes[static_cast<std::size_t>(d)], boot_key);
                    };
                    if (timed) total += elapsed_complete_call(call);
                    else { call(); cudaDeviceSynchronize(); }
                    if (timed) {
                        std::vector<bool> expected(shape), decrypted;
                        for (std::size_t j = 0; j < shape; ++j) {
                            bool value = bool_result("NAND", input1[j], input2[j], control[j]);
                            value = bool_result("XOR", value, input1[j], control[j]);
                            for (int d = 2; d < depth; ++d)
                                value = bool_result("NAND", value, input2[j], control[j]);
                            expected[j] = value;
                        }
                        decryptor.decrypt(*nodes[static_cast<std::size_t>(depth - 1)], decrypted);
                        cudaDeviceSynchronize();
                        correct = correct && decrypted == expected;
                    }
                }
                *output << "tfhe_circuit,nand_xor_chain_d" << depth << ','
                        << shape << ',' << options.repetitions << ','
                        << std::setprecision(10)
                        << total / static_cast<double>(options.repetitions) << ','
                        << (correct ? 1 : 0) << ',' << measured_backend << ",eager,0,0,exact_decrypt\n";
                if (!correct) throw std::runtime_error("exact TFHE circuit check failed");
            }
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    }
}
