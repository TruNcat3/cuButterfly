#include "fft_local_core_contract.hpp"
#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template<class F> void rejects(F check) {
    try { check(); } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("invalid FP16 local core contract was accepted");
}
cuntt::ButterflyConfig native_block() {
    cuntt::ButterflyConfig c;
    c.op = cuntt::ButterflyOperator::Fft;
    c.backend = cuntt::ButterflyBackend::TemporalTile;
    c.precision = cuntt::ButterflyPrecision::Fp16;
    c.fft_core = cuntt::FftCore::CufftDxBlock;
    c.compute_unit = cuntt::ComputeUnit::Radix2;
    c.log_n = 12;
    c.prefix_ept = 16;
    c.tile_threads = 256;
    c.batch = 5;
    c.stage_partition = {12};
    return c;
}
}
int main() try {
    using namespace cuntt;
    using namespace cuntt::detail;
    const auto c = native_block();
    require(cufftdx_arithmetic_precision_bits(c) == 16U, "native FP16 provider classification");
    for (const auto precision : {ButterflyPrecision::Fp32, ButterflyPrecision::Fp64}) {
        auto value = c; value.precision = precision;
        require(cufftdx_arithmetic_precision_bits(value) ==
                    (precision == ButterflyPrecision::Fp64 ? 64U : 32U),
                "wide provider arithmetic classification");
    }
    for (const auto precision : {ButterflyPrecision::Bf16, ButterflyPrecision::Fp16Fp32,
                                 ButterflyPrecision::Uint32, ButterflyPrecision::Uint64}) {
        auto value = c; value.precision = precision;
        require(!cufftdx_arithmetic_precision_bits(value),
                "unsupported storage must not be classified as FP32 provider arithmetic");
    }
    for (const auto precision : {ButterflyPrecision::Fp16, ButterflyPrecision::Bf16,
                                 ButterflyPrecision::Fp32, ButterflyPrecision::Fp64}) {
        auto value = c; value.precision = precision; value.accumulation = ButterflyAccumulation::Fp32;
        require(!cufftdx_arithmetic_precision_bits(value), "unsupported accumulation must not reach provider");
    }
    auto unsupported_provider = c; unsupported_provider.inverse = true; unsupported_provider.normalize_inverse = true;
    require(!cufftdx_arithmetic_precision_bits(unsupported_provider), "normalized native half provider is not implemented");
    unsupported_provider = c; unsupported_provider.op = ButterflyOperator::Fwht;
    require(!cufftdx_arithmetic_precision_bits(unsupported_provider), "FFT provider must reject other operators");
    const auto shape = fp16_local_block_shape(c);
    require(shape.lanes == 256, "N4096 EPT16 requires 256 lanes");
    require(shape.transforms_per_cta == 2, "half2 must pack two logical transforms");
#ifdef FP16_LOCAL_CORE_TEST_DATAFLOW
    const auto plan = make_dataflow_plan(c);
    require(plan.execution_groups.size() == 1, "local core must remain one group");
    const auto& g = plan.execution_groups.front();
    require(g.grid_ctas == 3 && plan.grid_ctas == 3, "odd batch5 requires three CTAs");
    require(g.units_per_cta == 2 && g.batch_space == 2 && g.batch_time == 3,
            "physical paired geometry must retain odd batch tail");
    require(g.threads == 256 && g.elements_per_thread == 16 && g.data_space == 256,
            "projected geometry disagrees with local core");
#endif
    for (const auto batch : {1U, 2U, 5U, 6U}) {
        auto value = c; value.batch = batch;
        require(fp16_local_block_shape(value).transforms_per_cta == 2,
                "batch must not change physical packing");
    }
    for (const auto threads : {512U, 1024U}) {
        auto wide = c;
        wide.tile_threads = threads;
        wide.prefix_ept = 8;
        const auto packed = fp16_local_block_shape(wide);
        require(packed.lanes == 512 && packed.transforms_per_cta == 2 * (threads / 512),
                "wide CTA must preserve integral half2 transform rows");
#ifdef FP16_LOCAL_CORE_TEST_DATAFLOW
        const auto wide_plan = make_dataflow_plan(wide);
        const auto& wide_group = wide_plan.execution_groups.front();
        require(wide_group.threads == threads && wide_group.elements_per_thread == 8,
                "wide geometry was lost during dataflow projection");
        require(wide_group.grid_ctas == (5 + packed.transforms_per_cta - 1) / packed.transforms_per_cta,
                "wide CTA grid disagrees with logical batch");
#endif
    }
    for (const auto threads : {0U, 48U, 1056U, 2048U}) {
        auto bad_threads = c; bad_threads.tile_threads = threads;
        rejects([&] { fp16_local_block_shape(bad_threads); });
    }
    auto bad = c; bad.accumulation = ButterflyAccumulation::Fp32;
    rejects([&] { fp16_local_block_shape(bad); });
    bad = c; bad.tile_threads = 128;
    rejects([&] { fp16_local_block_shape(bad); });
    bad = c; bad.stage_partition = {6, 6};
    rejects([&] { fp16_local_block_shape(bad); });
    bad = c; bad.stage_overlap = true;
    rejects([&] { fp16_local_block_shape(bad); });
    bad = c; bad.shared_layout = SharedLayout::XorSwizzle;
    rejects([&] { fp16_local_block_shape(bad); });
    bad = c; bad.compute_unit = ComputeUnit::Radix4;
    rejects([&] { fp16_local_block_shape(bad); });
    bad = c; bad.inverse = true; bad.normalize_inverse = true;
    rejects([&] { fp16_local_block_shape(bad); });
    std::cout << "FP16 local core CPU contract passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
