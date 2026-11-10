#include "xd_axis_resources.hpp"

#include <cassert>
#include <set>
#include <tuple>
#include <type_traits>

using namespace cuntt::detail;
using Key = std::tuple<int, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned>;

Key key(const XdAxisResources& r) {
    return {static_cast<int>(r.core), r.ept, r.register_lanes, r.block_cap,
            r.exchange, r.local_twiddle_policy, r.ffts_per_block};
}

template <class Function>
void expect_invalid(Function function) {
    bool rejected = false;
    try { function(); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
}

void reject(XdAxisResources resource) {
    expect_invalid([&] { validate_xd_axis_resource(resource); });
}

int main() {
    static_assert(std::is_aggregate<XdAxisResources>::value, "preserve aggregate API");
    const XdAxisResources defaults;
    assert(key(defaults) == key({XdAxisCore::CufftdxBlock,8,4,0,0,0,0}));
    validate_xd_axis_resource(defaults);
    for (const auto core : {XdPairCore::Thread, XdPairCore::CooperativeRegister,
                           XdPairCore::Radix4Register})
        assert(xd_pair_core_from_name(xd_pair_core_name(core)) == core);
    expect_invalid([] { xd_pair_core_from_name("radix4-regitser"); });
    expect_invalid([] { xd_pair_core_from_name(""); });
    expect_invalid([] { xd_pair_core_name(static_cast<XdPairCore>(-1)); });
    for (const auto core : {XdAxisCore::CufftdxBlock, XdAxisCore::CooperativeRegister,
                            XdAxisCore::ThreadShared, XdAxisCore::Vkfft}) {
        assert(xd_axis_core_from_name(xd_axis_core_name(core)) == core);
        validate_xd_axis_resource({core,8,4,0,0});
    }
    for (const bool include_vkfft : {false, true}) {
        const auto candidates = xd_axis_resource_candidates(include_vkfft);
        assert(candidates.size() == (include_vkfft ? 71U : 70U));
        std::set<Key> seen;
        for (const auto& resource : candidates) {
            validate_xd_axis_resource(resource);
            assert(seen.insert(key(resource)).second);
        }
        std::size_t index = 0;
        for (unsigned ept : {4U,8U,16U})
            for (unsigned cap : {0U,8U,16U})
                for (unsigned exchange : {0U,1U,2U,3U})
                    assert(key(candidates.at(index++)) == key({XdAxisCore::CufftdxBlock,ept,4,cap,exchange}));
        for (unsigned ept : {8U,16U})
            for (unsigned exact : {2U,4U})
                for (unsigned exchange : {0U,2U,3U})
                    assert(key(candidates.at(index++)) == key({XdAxisCore::CufftdxBlock,ept,4,0,exchange,0,exact}));
        for (unsigned lanes : {4U,8U,16U})
            assert(key(candidates.at(index++)) == key({XdAxisCore::CooperativeRegister,8,lanes,0,0}));
        for (unsigned roots : {0U,1U})
            for (unsigned cap : {0U,16U})
                assert(key(candidates.at(index++)) == key({XdAxisCore::ThreadShared,8,4,cap,0,roots}));
        assert(index == 55);
        assert(seen.count(key({XdAxisCore::Vkfft,8,4,0,0})) == (include_vkfft ? 1U : 0U));
        if (include_vkfft) assert(candidates.at(index++).core == XdAxisCore::Vkfft);
        assert(candidates.at(index).exchange == 4);
        for (unsigned ept : {4U,8U,16U})
            for (unsigned cap : {0U,8U,16U})
                assert(seen.count(key({XdAxisCore::CufftdxBlock,ept,4,cap,4})) == 1);
        for (unsigned ept : {8U,16U})
            for (unsigned exact : {2U,4U})
                assert(seen.count(key({XdAxisCore::CufftdxBlock,ept,4,0,4,0,exact})) == 1);
        for (unsigned roots : {0U,1U})
            assert(seen.count(key({XdAxisCore::ThreadShared,8,4,32,0,roots})) == 1);
    }

    // Shape-specific exclusions must not narrow this core contract.
    for (unsigned exact : {2U,4U})
        validate_xd_axis_resource({XdAxisCore::CufftdxBlock,4,4,0,4,0,exact});
    validate_xd_axis_resource({XdAxisCore::ThreadShared,8,4,8,0,1});
    reject({XdAxisCore::CufftdxBlock,2,4,0,0});
    reject({XdAxisCore::CufftdxBlock,8,8,0,0});
    reject({XdAxisCore::CufftdxBlock,8,4,32,0});
    reject({XdAxisCore::CufftdxBlock,8,4,0,5});
    reject({XdAxisCore::CufftdxBlock,8,4,0,0,1});
    reject({XdAxisCore::CufftdxBlock,8,4,0,0,0,1});
    reject({XdAxisCore::CufftdxBlock,8,4,8,0,0,2});
    reject({XdAxisCore::CufftdxBlock,8,4,0,1,0,4});
    for (const auto core : {XdAxisCore::CooperativeRegister, XdAxisCore::ThreadShared, XdAxisCore::Vkfft}) {
        reject({core,4,4,0,0});
        reject({core,8,4,0,1});
        reject({core,8,4,0,0,0,2});
        if (core != XdAxisCore::ThreadShared) {
            reject({core,8,4,8,0});
            reject({core,8,4,0,0,1});
        }
        if (core != XdAxisCore::CooperativeRegister) reject({core,8,8,0,0});
    }
    reject({XdAxisCore::CooperativeRegister,8,2,0,0});
    reject({XdAxisCore::ThreadShared,8,4,4,0});
    reject({XdAxisCore::ThreadShared,8,4,0,0,2});
    const auto invalid_core = static_cast<XdAxisCore>(-1);
    reject({invalid_core,8,4,0,0});
    expect_invalid([&] { xd_axis_core_name(invalid_core); });
    expect_invalid([] { xd_axis_core_from_name("unknown"); });
}
