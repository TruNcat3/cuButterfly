#include <cubutterfly/plan.hpp>
#include <cassert>
#include <stdexcept>

// CPU-only: descriptor APIs must not create a context or probe a device.
int main() {
    cubutterfly::Transform transform;
    transform.op = CUBUTTERFLY_OPERATOR_FWHT;
    transform.storage = CUBUTTERFLY_DATA_FP32;
    transform.extents = {8, 8, 8, 8};
    transform.input_batch_stride = 8192;
    transform.output_strides = {512, 64, 8, 1};
    transform.output_batch_stride = 9000;
    transform.input_orders = {CUBUTTERFLY_ORDER_BIT_REVERSED,
                              CUBUTTERFLY_ORDER_NATURAL,
                              CUBUTTERFLY_ORDER_NATURAL,
                              CUBUTTERFLY_ORDER_NATURAL};
    cubutterfly::Descriptor descriptor(transform);
    descriptor.validate();
    const auto json = descriptor.json();
    assert(json.find("8192") != std::string::npos);
    assert(json.find("9000") != std::string::npos);
    auto restored = cubutterfly::Descriptor::from_json(json);
    restored.validate();
    assert(restored.json() == json);
    assert(!restored.capabilities_json().empty());
    transform.output_order = CUBUTTERFLY_NTT_LAYOUT_BIT_REVERSED;
    bool rejected = false;
    try { cubutterfly::Descriptor invalid(transform); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
}
