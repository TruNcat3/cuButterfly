#include "cuntt/butterfly.hpp"
#include "external_fft_units.cuh"
#include "fft_register_tile.cuh"
#include "fft_grouped_suffix.cuh"
#include "fft_thread_codelet.cuh"
#include "fft_register_dispatch.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
void check(cudaError_t s) { if (s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s)); }
struct Buffer {
    cuntt::Complex32* data = nullptr;
    explicit Buffer(std::size_t count) { check(cudaMalloc(&data, count * sizeof(*data))); }
    ~Buffer() { cudaFree(data); }
};
}

int main(int argc, char** argv) {
  try {
    const unsigned log = argc > 1 ? std::stoul(argv[1]) : 20;
    const unsigned batch = argc > 2 ? std::stoul(argv[2]) : 16;
    const unsigned rounds = argc > 3 ? std::stoul(argv[3]) : 5;
    const unsigned repeat = argc > 4 ? std::stoul(argv[4]) : 100;
    const bool verify_only = argc > 5 && std::string(argv[5]) == "--verify-only";
    const bool production_only = argc > 5 && std::string(argv[5]) == "--production-only";
    if ((log != 18 && log != 20) || batch == 0 || rounds == 0 || repeat == 0)
        throw std::invalid_argument("usage: fft_resident_experiment {18|20} batch rounds repeat");
    const std::size_t n = std::size_t{1} << log, count = n * batch;
    Buffer input(count), output(count), scratch(count), auxiliary(count);
    std::vector<cuntt::Complex32> values(count), expected(count), actual(count);
    std::mt19937 rng(127);
    std::uniform_real_distribution<float> distribution(-1, 1);
    for (auto& v : values) v = {distribution(rng), distribution(rng)};
    check(cudaMemcpy(input.data, values.data(), count * sizeof(values[0]), cudaMemcpyHostToDevice));
    cuntt::ButterflyConfig config;
    config.op = cuntt::ButterflyOperator::Fft;
    config.log_n = log; config.batch = batch;
    config.normalize_inverse = false;
    config.backend = cuntt::ButterflyBackend::CuFft;
    cuntt::ButterflyPlan reference(config);
    reference.execute_async(input.data, output.data);
    check(cudaMemcpy(expected.data(), output.data, count * sizeof(values[0]), cudaMemcpyDeviceToHost));
    std::vector<std::unique_ptr<cuntt::ButterflyPlan>> plans;
    std::vector<std::pair<std::string, std::function<void()>>> cases;
    cases.emplace_back("cufft", [&]{ reference.execute_async(input.data, output.data); });
    config.backend = cuntt::ButterflyBackend::OnlineReorder;
    config.fft_core = cuntt::FftCore::CufftDxBlock;
    config.cross_twiddle = cuntt::CrossTwiddleMode::Recurrence;
    config.direct_boundary = cuntt::DirectBoundary::TiledTranspose;
    config.prefix_threads = 128; config.prefix_ept = 16;
    config.suffix_threads = 256; config.suffix_ept = 16;
    config.stage_partition = {log - 12, 12};
    for (auto layout : {cuntt::SharedLayout::Linear, cuntt::SharedLayout::XorSwizzle}) {
        config.shared_layout = layout;
        plans.push_back(std::make_unique<cuntt::ButterflyPlan>(config));
        auto* plan = plans.back().get();
        cases.emplace_back(layout == cuntt::SharedLayout::Linear ? "dx-linear" : "dx-xor",
                           [&, plan]{ plan->execute_async(input.data, output.data); });
    }
    for (unsigned columns : {4U, 8U, 16U, 32U}) {
        cases.emplace_back("register-c" + std::to_string(columns), [&, columns]{
            using namespace cuntt::detail::register_tile;
#define PREFIX(C) if (log == 20) launch_prefix<8,C>(input.data,scratch.data,log,batch,n); \
                  else launch_prefix<6,C>(input.data,scratch.data,log,batch,n)
            switch (columns) {
                case 4: PREFIX(4); break;
                case 8: PREFIX(8); break;
                case 16: PREFIX(16); break;
                case 32: PREFIX(32); break;
            }
#undef PREFIX
            cuntt::detail::launch_cufftdx_online_reorder(log, log-12, input.data, output.data,
                scratch.data, auxiliary.data, nullptr, batch, n, 1, false, false,
                cuntt::CrossTwiddleMode::Recurrence, cuntt::SharedLayout::Linear,
                128, 256, 16, 16, cuntt::DirectBoundary::TiledTranspose, nullptr, 2);
        });
    }
    for (unsigned ept : {16U, 32U})
      for (unsigned columns : {2U, 4U}) {
        cases.emplace_back("register-c16-grouped-e" + std::to_string(ept) + "-c" + std::to_string(columns),
          [&, ept, columns]{
            using namespace cuntt::detail;
            if (log == 20) register_tile::launch_prefix<8,16>(input.data,scratch.data,log,batch,n);
            else register_tile::launch_prefix<6,16>(input.data,scratch.data,log,batch,n);
            if (ept==16 && columns==2) launch_grouped_suffix<12,16,2>(scratch.data,output.data,log,batch);
            if (ept==16 && columns==4) launch_grouped_suffix<12,16,4>(scratch.data,output.data,log,batch);
            if (ept==32 && columns==2) launch_grouped_suffix<12,32,2>(scratch.data,output.data,log,batch);
            if (ept==32 && columns==4) launch_grouped_suffix<12,32,4>(scratch.data,output.data,log,batch);
        });
    }
    for (unsigned columns : {8U,16U,32U}) {
      cases.emplace_back("dx-thread-c" + std::to_string(columns), [&, columns]{
        using namespace cuntt::detail::register_tile;
#define DX_PREFIX(C) if (log == 20) launch_prefix<8,C,false,DxCodelet>(input.data,scratch.data,log,batch,n); \
                     else launch_prefix<6,C,false,DxCodelet>(input.data,scratch.data,log,batch,n)
        switch(columns) {
            case 8: DX_PREFIX(8); break;
            case 16: DX_PREFIX(16); break;
            case 32: DX_PREFIX(32); break;
        }
#undef DX_PREFIX
        cuntt::detail::launch_cufftdx_online_reorder(log,log-12,input.data,output.data,
            scratch.data,auxiliary.data,nullptr,batch,n,1,false,false,
            cuntt::CrossTwiddleMode::Recurrence,cuntt::SharedLayout::Linear,
            128,256,16,16,cuntt::DirectBoundary::TiledTranspose,nullptr,2);
      });
    }
    for (unsigned columns : {4U,8U,16U})
      for (unsigned ept : {8U,16U})
        for (unsigned suffix_columns : {4U,8U}) {
          if (ept==8 && suffix_columns==8) continue;
          cases.emplace_back("balanced-register-c" + std::to_string(columns) + "-e" + std::to_string(ept) +
            "-sc" + std::to_string(suffix_columns), [&,columns,ept,suffix_columns]{
              using namespace cuntt::detail;
#define BALANCED_PREFIX(C) if (log == 20) register_tile::launch_prefix<10,C>(input.data,scratch.data,log,batch,n); \
                          else register_tile::launch_prefix<8,C>(input.data,scratch.data,log,batch,n)
              switch(columns) {
                case 4: BALANCED_PREFIX(4); break;
                case 8: BALANCED_PREFIX(8); break;
                case 16: BALANCED_PREFIX(16); break;
              }
#undef BALANCED_PREFIX
              if(ept==8) launch_grouped_suffix<10,8,4>(scratch.data,output.data,log,batch);
              else if(suffix_columns==4) launch_grouped_suffix<10,16,4>(scratch.data,output.data,log,batch);
              else launch_grouped_suffix<10,16,8>(scratch.data,output.data,log,batch);
          });
        }
    if(production_only) {
        // Freeze two mappings before the confirmation run. They are exposed
        // through ButterflyPlan; no raw experimental launch is measured here.
        cases.resize(3);
        for(const auto& mapping : cuntt::detail::register_tile_mappings()) {
            const auto pc=mapping.prefix_threads/mapping.prefix_ept;
            if(mapping.log_n!=log || mapping.local_stages!=log-10 ||
               (pc!=8 && pc!=16) || mapping.suffix_threads!=256 || mapping.suffix_ept!=16) continue;
            auto selected=mapping; selected.batch=batch; selected.normalize_inverse=false;
            plans.push_back(std::make_unique<cuntt::ButterflyPlan>(selected));
            auto* plan=plans.back().get();
            cases.emplace_back("plan-register-c"+std::to_string(pc),[&,plan]{plan->execute_async(input.data,output.data);});
        }
    }
    double max_reference = 0;
    for (const auto& v : expected) max_reference = std::max(max_reference, std::hypot(double(v.real), double(v.imag)));
    for (const auto& c : cases) {
        c.second(); check(cudaGetLastError());
        check(cudaMemcpy(actual.data(), output.data, count*sizeof(actual[0]), cudaMemcpyDeviceToHost));
        double error = 0;
        for (std::size_t i = 0; i < count; ++i) {
            double e = std::hypot(double(actual[i].real)-expected[i].real, double(actual[i].imag)-expected[i].imag);
            if (!std::isfinite(e)) throw std::runtime_error("nonfinite output: " + c.first);
            error = std::max(error, e);
        }
        std::cerr << "verified " << c.first << " relative_max_error=" << error/max_reference << '\n';
        const double absolute_tolerance=2e-4*std::sqrt(double(n)/1024);
        if (error/max_reference > 2e-4 || error > absolute_tolerance)
            throw std::runtime_error("incorrect: " + c.first);
    }
    if (verify_only) return 0;
    cudaEvent_t start, stop; check(cudaEventCreate(&start)); check(cudaEventCreate(&stop));
    std::vector<unsigned> order(cases.size()); std::iota(order.begin(), order.end(), 0);
    std::cout << "logN,batch,round,mode,kernel_ms\n";
    for (unsigned r = 0; r < rounds; ++r) {
        std::shuffle(order.begin(), order.end(), rng);
        for (auto index : order) {
            const auto& c = cases[index];
            for (unsigned j=0; j<20; ++j) c.second();
            check(cudaDeviceSynchronize()); check(cudaEventRecord(start));
            for (unsigned j=0; j<repeat; ++j) c.second();
            check(cudaEventRecord(stop)); check(cudaEventSynchronize(stop));
            float ms; check(cudaEventElapsedTime(&ms,start,stop));
            std::cout << log << ',' << batch << ',' << r << ',' << c.first << ',' << ms/repeat << '\n';
        }
    }
    check(cudaEventDestroy(start)); check(cudaEventDestroy(stop));
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
