#include "semantic_contract.hpp"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace cubutterfly::detail;
namespace {
void check(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
template <typename Function> void invalid(Function function) {
    try { function(); }
    catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("expected invalid_argument");
}
cubutterflyDescriptor shape(std::initializer_list<std::size_t> extents) {
    cubutterflyDescriptor d;
    d.extents = extents;
    d.rank = static_cast<std::uint32_t>(d.extents.size());
    return d;
}
void layout_tests() {
    auto d = shape({4, 8});
    d.batch = 3;
    d.input_batch_stride = 100;
    auto n = normalize_descriptor(d);
    check(n.input_strides == std::vector<std::size_t>({8, 1}), "default input strides");
    check(n.input_batch_stride == 100 && n.output_batch_stride == 32, "independent batch defaults");
    d.input_strides = {16, 2};
    d.input_batch_stride = 200;
    d.output_batch_stride = 70;
    n = normalize_descriptor(d);
    check(n.input_batch_stride == 200 && n.output_batch_stride == 70, "preserve explicit batch strides");
    d.input_batch_stride = 0;
    check(normalize_descriptor(d).input_batch_stride == 63, "strided side span default");
    auto interleaved = shape({8});
    interleaved.batch = 2;
    interleaved.input_strides = {2};
    interleaved.input_batch_stride = 1;
    normalize_descriptor(interleaved);
    auto sparse = shape({3, 2});
    sparse.input_strides = {2, 3};
    auto sparse_normalized = normalize_descriptor(sparse);
    check(layout_is_nonoverlapping(sparse.extents, sparse.input_strides, 1,
                                  sparse_normalized.input_batch_stride), "exact sparse layout proof");
    sparse = shape({2, 2, 2});
    sparse.input_strides = {2, 3, 5};
    invalid([&] { normalize_descriptor(sparse); });
    auto unproven = shape({300, 300});
    unproven.input_strides = {300, 301};
    auto unproven_normalized = normalize_descriptor(unproven);
    check(!layout_is_nonoverlapping(unproven.extents, unproven.input_strides, 1,
                                   unproven_normalized.input_batch_stride), "bounded unknown layout proof");
    descriptor_json(unproven);
    d.input_strides = {1, 1};
    invalid([&] { normalize_descriptor(d); });
    d.input_strides = {8, 0};
    invalid([&] { normalize_descriptor(d); });
    d.input_strides = {1};
    invalid([&] { normalize_descriptor(d); });
    d.input_strides = {std::numeric_limits<std::size_t>::max(), 1};
    invalid([&] { normalize_descriptor(d); });
    d = shape({8}); d.batch = 2; d.input_batch_stride = 7;
    invalid([&] { normalize_descriptor(d); });
    d.input_batch_stride = std::numeric_limits<std::size_t>::max();
    invalid([&] { normalize_descriptor(d); });
}
void extent_and_order_tests() {
    auto d = shape({4, 8});
    d.storage = CUBUTTERFLY_DATA_FP16;
    d.compute = CUBUTTERFLY_COMPUTE_NATIVE;
    d.fft_type = CUBUTTERFLY_FFT_R2C;
    auto n = normalize_descriptor(d);
    check(semantic_input_extents(n) == std::vector<std::size_t>({4, 8}), "R2C input extent");
    check(semantic_output_extents(n) == std::vector<std::size_t>({4, 5}), "R2C output extent");
    check(n.input_batch_stride == 32 && n.output_batch_stride == 20, "R2C side defaults");
    check(n.output_strides == std::vector<std::size_t>({5, 1}), "R2C output strides");
    d.fft_type = CUBUTTERFLY_FFT_C2R;
    d.direction = CUBUTTERFLY_DIRECTION_INVERSE;
    n = normalize_descriptor(d);
    check(semantic_input_extents(n) == std::vector<std::size_t>({4, 5}), "C2R input extent");
    check(semantic_output_extents(n) == std::vector<std::size_t>({4, 8}), "C2R output extent");
    check(n.input_batch_stride == 20 && n.output_batch_stride == 32, "C2R side defaults");
    d.input_orders = {CUBUTTERFLY_ORDER_NATURAL, CUBUTTERFLY_ORDER_BIT_REVERSED};
    invalid([&] { normalize_descriptor(d); });
    d = shape({3, 5});
    d.length_mode = CUBUTTERFLY_LENGTH_ZERO_EXTENDED_EMBEDDING;
    d.output_orders = {CUBUTTERFLY_ORDER_BIT_REVERSED, CUBUTTERFLY_ORDER_BIT_REVERSED};
    n = normalize_descriptor(d);
    check(semantic_input_extents(n) == std::vector<std::size_t>({3, 5}), "embedding input extent");
    check(semantic_output_extents(n) == std::vector<std::size_t>({4, 8}), "embedding output extent");
    check(n.input_batch_stride == 15 && n.output_batch_stride == 32, "embedding side defaults");
    d.placement = CUBUTTERFLY_PLACEMENT_IN_PLACE;
    normalize_descriptor(d);
    d.direction = CUBUTTERFLY_DIRECTION_INVERSE;
    d.input_orders = d.output_orders; d.output_orders.clear();
    n = normalize_descriptor(d);
    check(semantic_input_extents(n) == std::vector<std::size_t>({4, 8}) &&
          semantic_output_extents(n) == std::vector<std::size_t>({3, 5}), "inverse embedding extents");
    d = shape({1, 2, 4, 8});
    d.input_orders.assign(4, CUBUTTERFLY_ORDER_BIT_REVERSED);
    n = normalize_descriptor(d);
    check(n.rank == 4 && n.stage_matrices.size() >= 4, "rank four expression");
    for (std::size_t axis = 0; axis < 4; ++axis) {
        auto changed = d;
        changed.input_orders[axis] = CUBUTTERFLY_ORDER_NATURAL;
        check(semantic_json(changed) != semantic_json(d), "per-axis input order identity");
        changed = d;
        changed.output_orders.assign(4, CUBUTTERFLY_ORDER_NATURAL);
        changed.output_orders[axis] = CUBUTTERFLY_ORDER_BIT_REVERSED;
        check(semantic_json(changed) != semantic_json(d), "per-axis output order identity");
    }
    d.rank = 9; d.extents.assign(9, 1); d.input_orders.clear();
    check(normalize_descriptor(d).rank == 9, "semantic rank is independent of lowering maximum");
    d = shape({3}); d.input_orders = {CUBUTTERFLY_ORDER_BIT_REVERSED};
    invalid([&] { normalize_descriptor(d); });
}
void normalization_tests() {
    auto d = shape({8});
    auto explicit_defaults = normalize_descriptor(d);
    d.compute = CUBUTTERFLY_COMPUTE_NATIVE;
    check(semantic_json(d) == semantic_json(explicit_defaults), "native FP32 effective compute");
    d.normalize_inverse = false;
    check(semantic_json(d) == semantic_json(explicit_defaults), "forward normalization irrelevant");
    d.execution_mode = CUBUTTERFLY_EXECUTION_COMPOSITE;
    d.algorithm_policy = CUBUTTERFLY_ALGORITHM_EXPLICIT;
    d.explicit_algorithm = "test-lowering";
    check(semantic_json(d) == semantic_json(explicit_defaults), "selection excluded from semantics");
    check(descriptor_json(d) != descriptor_json(explicit_defaults), "selection included in descriptor");
    check(semantic_json(d).at("modulus").is_null(), "non-NTT modulus omitted");
    d.storage = CUBUTTERFLY_DATA_COMPLEX_FP64;
    check(normalize_descriptor(d).compute == CUBUTTERFLY_COMPUTE_FP64, "native FP64 effective compute");
    d.storage = CUBUTTERFLY_DATA_COMPLEX_FP16;
    check(normalize_descriptor(d).compute == CUBUTTERFLY_COMPUTE_NATIVE, "low precision native retained");
    d.compute = CUBUTTERFLY_COMPUTE_FP32; normalize_descriptor(d);
    d.compute = CUBUTTERFLY_COMPUTE_UINT64;
    invalid([&] { normalize_descriptor(d); });
    d = shape({8}); d.op = CUBUTTERFLY_OPERATOR_NTT; d.storage = CUBUTTERFLY_DATA_UINT32;
    d.compute = CUBUTTERFLY_COMPUTE_NATIVE; d.word_bits = 32; d.modulus = 17;
    check(normalize_descriptor(d).compute == CUBUTTERFLY_COMPUTE_UINT32, "native uint32 effective compute");
    d.direction = CUBUTTERFLY_DIRECTION_INVERSE; d.normalize_inverse = false;
    check(normalize_descriptor(d).normalize_inverse, "inverse NTT normalization");
    d.output_ntt_layout = CUBUTTERFLY_NTT_LAYOUT_BIT_REVERSED;
    const auto legacy = semantic_json(d);
    d.output_orders = {CUBUTTERFLY_ORDER_NATURAL};
    invalid([&] { normalize_descriptor(d); });
    d.output_orders = {CUBUTTERFLY_ORDER_BIT_REVERSED};
    check(semantic_json(d) == legacy, "consistent legacy and generic bit reversal");
    auto contradictory = descriptor_json(d);
    contradictory["semantics"]["ntt_output_layout"] = CUBUTTERFLY_NTT_LAYOUT_BIT_REVERSED;
    contradictory["semantics"]["output_orders"] = {CUBUTTERFLY_ORDER_NATURAL};
    invalid([&] { descriptor_from_json(contradictory); });
    contradictory["semantics"]["output_orders"] = {CUBUTTERFLY_ORDER_BIT_REVERSED};
    check(semantic_json(descriptor_from_json(contradictory)) == legacy,
          "consistent legacy and generic JSON import");
    d.output_ntt_layout = CUBUTTERFLY_NTT_LAYOUT_NATURAL;
    d.output_orders = {CUBUTTERFLY_ORDER_BIT_REVERSED};
    check(semantic_json(d) == legacy, "legacy bit reversal canonical identity");
    d.input_ntt_layout = CUBUTTERFLY_NTT_LAYOUT_APPT_STATIC;
    invalid([&] { normalize_descriptor(d); });
    d.input_ntt_layout = CUBUTTERFLY_NTT_LAYOUT_NATURAL;
    d.word_bits = 64; invalid([&] { normalize_descriptor(d); });
    d = shape({8}); d.op = CUBUTTERFLY_OPERATOR_STRUCTURED_2X2; d.storage = CUBUTTERFLY_DATA_FP32;
    invalid([&] { normalize_descriptor(d); });
    d.stage_matrices[0] = {{1, 2, 3, 4}};
    normalize_descriptor(d);
    const auto structured_json = descriptor_json(d);
    check(descriptor_json(descriptor_from_json(structured_json)) == structured_json,
          "structured matrix roundtrip");
    auto different_matrix = d;
    different_matrix.stage_matrices[0][0].m00 = 2;
    check(semantic_json(different_matrix) != semantic_json(d), "matrix identity");
    d.direction = CUBUTTERFLY_DIRECTION_INVERSE;
    check(!normalize_descriptor(d).normalize_inverse, "structured inverse normalization");
    d.stage_matrices[0] = {{1, 1, 1, 1}};
    invalid([&] { normalize_descriptor(d); });
    d.direction = CUBUTTERFLY_DIRECTION_FORWARD;
    d.stage_matrices[0] = {{1, 0, 0, 1}, {1, 0, 0, 1}};
    invalid([&] { normalize_descriptor(d); });
    d.stage_matrices[0] = {{1, 0, 0, std::numeric_limits<double>::infinity()}};
    invalid([&] { normalize_descriptor(d); });
}
void invalid_descriptor_tests() {
    auto d = shape({8});
    d.op = static_cast<cubutterflyOperator_t>(-1); invalid([&] { normalize_descriptor(d); });
    d = shape({8}); d.storage = static_cast<cubutterflyDataType_t>(99); invalid([&] { normalize_descriptor(d); });
    d = shape({8}); d.direction = static_cast<cubutterflyDirection_t>(2); invalid([&] { normalize_descriptor(d); });
    d = shape({8}); d.compute = static_cast<cubutterflyComputeType_t>(5); invalid([&] { normalize_descriptor(d); });
    d = shape({8}); d.placement = static_cast<cubutterflyPlacement_t>(2); invalid([&] { normalize_descriptor(d); });
    d = shape({8}); d.input_orders = {static_cast<cubutterflyLogicalOrder_t>(2)}; invalid([&] { normalize_descriptor(d); });
    d = shape({8}); d.input_ntt_layout = CUBUTTERFLY_NTT_LAYOUT_APPT_STATIC; invalid([&] { normalize_descriptor(d); });
    d = shape({8}); d.modulus = 17; invalid([&] { normalize_descriptor(d); });
    d = shape({8}); d.storage = CUBUTTERFLY_DATA_FP32; invalid([&] { normalize_descriptor(d); });
    d = shape({8}); d.rank = 2; invalid([&] { normalize_descriptor(d); });
    d = shape({0}); invalid([&] { normalize_descriptor(d); });
    d = shape({8}); d.batch = 0; invalid([&] { normalize_descriptor(d); });
    d = shape({8}); d.storage = CUBUTTERFLY_DATA_FP32; d.fft_type = CUBUTTERFLY_FFT_C2R;
    invalid([&] { normalize_descriptor(d); });
}
void semantic_type_domain_tests() {
    for (auto op : {CUBUTTERFLY_OPERATOR_FWHT, CUBUTTERFLY_OPERATOR_SUBSET_ZETA,
                    CUBUTTERFLY_OPERATOR_SUPERSET_ZETA, CUBUTTERFLY_OPERATOR_STRUCTURED_2X2}) {
        for (auto storage : {CUBUTTERFLY_DATA_FP64, CUBUTTERFLY_DATA_COMPLEX_FP32,
                             CUBUTTERFLY_DATA_UINT32, CUBUTTERFLY_DATA_UINT64}) {
            auto d = shape({8});
            d.op = op;
            d.storage = storage;
            d.compute = CUBUTTERFLY_COMPUTE_NATIVE;
            if (op == CUBUTTERFLY_OPERATOR_STRUCTURED_2X2)
                d.stage_matrices[0] = {{1, 2, 3, 4}};
            const auto normalized = normalize_descriptor(d);
            check(normalized.storage == storage, "butterfly semantic storage domain");
            const auto serialized = descriptor_json(d);
            check(descriptor_json(descriptor_from_json(serialized)) == serialized,
                  "butterfly semantic type roundtrip");
            d.compute = storage >= CUBUTTERFLY_DATA_UINT32 ? CUBUTTERFLY_COMPUTE_FP32
                                                         : CUBUTTERFLY_COMPUTE_UINT64;
            invalid([&] { normalize_descriptor(d); });
        }
    }
    for (auto storage : {CUBUTTERFLY_DATA_FP16, CUBUTTERFLY_DATA_FP32,
                         CUBUTTERFLY_DATA_FP64, CUBUTTERFLY_DATA_COMPLEX_FP32,
                         CUBUTTERFLY_DATA_COMPLEX_FP64, CUBUTTERFLY_DATA_UINT32,
                         CUBUTTERFLY_DATA_UINT64}) {
        auto d = shape({8});
        d.op = CUBUTTERFLY_OPERATOR_FWHT;
        d.storage = storage;
        d.compute = storage >= CUBUTTERFLY_DATA_UINT32
            ? (storage == CUBUTTERFLY_DATA_UINT32 ? CUBUTTERFLY_COMPUTE_UINT64 : CUBUTTERFLY_COMPUTE_UINT32)
            : (storage == CUBUTTERFLY_DATA_FP64 || storage == CUBUTTERFLY_DATA_COMPLEX_FP64
               ? CUBUTTERFLY_COMPUTE_FP32 : CUBUTTERFLY_COMPUTE_FP64);
        check(normalize_descriptor(d).compute == d.compute, "explicit mixed compute retained");
        const auto serialized = descriptor_json(d);
        check(descriptor_json(descriptor_from_json(serialized)) == serialized,
              "mixed compute semantic roundtrip");
    }
}
void json_tests() {
    auto d = shape({2, 4, 8, 16});
    d.batch = 2; d.input_batch_stride = 1100;
    d.input_orders = {CUBUTTERFLY_ORDER_NATURAL, CUBUTTERFLY_ORDER_BIT_REVERSED,
                      CUBUTTERFLY_ORDER_NATURAL, CUBUTTERFLY_ORDER_BIT_REVERSED};
    const auto original = descriptor_json(d);
    check(descriptor_json(descriptor_from_json(original)) == original, "descriptor JSON roundtrip");
    auto changed = d; changed.fft_output_scale = 0.5;
    check(semantic_json(changed) != semantic_json(d), "scale identity");
    changed = d; changed.direction = CUBUTTERFLY_DIRECTION_INVERSE;
    check(semantic_json(changed) != semantic_json(d), "direction identity");
    changed = d; changed.input_batch_stride += 1;
    check(semantic_json(changed) != semantic_json(d), "batch stride identity");
    changed = d; changed.batch += 1;
    check(semantic_json(changed) != semantic_json(d), "batch identity");
    changed = d; changed.extents[0] = 4;
    changed.input_batch_stride = 2200;
    check(semantic_json(changed) != semantic_json(d), "shape identity");
    changed = d; changed.placement = CUBUTTERFLY_PLACEMENT_IN_PLACE;
    check(semantic_json(changed) != semantic_json(d), "placement identity");
    auto j = original; j["extra"] = 1; invalid([&] { descriptor_from_json(j); });
    j = original; j["semantics"]["extra"] = 1; invalid([&] { descriptor_from_json(j); });
    j = original; j["selection"]["extra"] = 1; invalid([&] { descriptor_from_json(j); });
    j = original; j["semantics"].erase("input_orders"); invalid([&] { descriptor_from_json(j); });
    j = original; j["semantics"]["batch"] = 1.5; invalid([&] { descriptor_from_json(j); });
    j = original; j["semantics"]["shape"][0] = -1; invalid([&] { descriptor_from_json(j); });
    j = original; j["semantics"]["input_strides"][0] = 1.0; invalid([&] { descriptor_from_json(j); });
    j = original; j["semantics"]["operator"] = true; invalid([&] { descriptor_from_json(j); });
    j = original; j["semantics"]["inverse"] = 0.0; invalid([&] { descriptor_from_json(j); });
    j = original; j["semantics"]["input_orders"][0] = -1; invalid([&] { descriptor_from_json(j); });
    j = original; j["semantics"]["normalize_inverse"] = 0; invalid([&] { descriptor_from_json(j); });
    j = original; j["semantics"]["fft_output_scale"] = "1"; invalid([&] { descriptor_from_json(j); });
    j = original; j["semantics"]["modulus"] = 17; invalid([&] { descriptor_from_json(j); });
    j = original; j["semantics"]["matrices"][0] = {{1, 2, 3}}; invalid([&] { descriptor_from_json(j); });
    j = original; j["selection"]["execution_mode"] = -1; invalid([&] { descriptor_from_json(j); });
    j = original; j["selection"]["explicit_algorithm"] = nullptr; invalid([&] { descriptor_from_json(j); });
    j = original; j["selection"]["algorithm_policy"] = CUBUTTERFLY_ALGORITHM_EXPLICIT;
    invalid([&] { descriptor_from_json(j); });
    for (auto policy : {CUBUTTERFLY_ALGORITHM_DEFAULT, CUBUTTERFLY_ALGORITHM_MEASURE}) {
        j = original;
        j["selection"]["algorithm_policy"] = policy;
        j["selection"]["explicit_algorithm"] = "ignored-mapping";
        invalid([&] { descriptor_from_json(j); });
    }
    j = original; j["schema"] = "unknown"; invalid([&] { descriptor_from_json(j); });
}
}
int main() {
    try {
        layout_tests(); extent_and_order_tests(); normalization_tests();
        invalid_descriptor_tests(); semantic_type_domain_tests(); json_tests();
        auto legacy = shape({32});
        const auto old_json = descriptor_json(legacy);
        check(!old_json.at("semantics").contains("output_scale") &&
                  !old_json.at("semantics").contains("output_rounding"), "legacy JSON identity");
        check(descriptor_json(descriptor_from_json(old_json)) == old_json, "legacy JSON round trip");
        auto scaled = legacy;
        scaled.op = CUBUTTERFLY_OPERATOR_FWHT;
        scaled.storage = CUBUTTERFLY_DATA_FP16;
        scaled.compute = CUBUTTERFLY_COMPUTE_FP32;
        scaled.output_rounding = CUBUTTERFLY_ROUNDING_FINAL_STORAGE;
        for (double scale : {0.0, -0.5, 1.0, 0.125}) {
            scaled.output_scale = scale;
            const auto encoded = descriptor_json(scaled);
            const auto decoded = descriptor_from_json(encoded);
            check(decoded.output_scale == scale && decoded.output_rounding == scaled.output_rounding,
                  "output epilogue round trip");
            check(encoded.at("semantics").contains("output_rounding"), "unit scale preserves final rounding");
            check(encoded != old_json, "output epilogue changes semantic identity");
        }
        auto malformed = descriptor_json(scaled);
        malformed["semantics"]["output_scale"] = true;
        invalid([&] { descriptor_from_json(malformed); });
        malformed = descriptor_json(scaled);
        malformed["semantics"]["output_rounding"] = 2;
        invalid([&] { descriptor_from_json(malformed); });
        scaled.output_scale = std::numeric_limits<double>::infinity();
        invalid([&] { normalize_descriptor(scaled); });
        std::cout << "semantic contract CPU tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
