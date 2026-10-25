#include "tfhe_graph_contract.hpp"
#include <cassert>

int main() {
    using namespace cubutterfly::heongpu_tfhe;
    int first, second, unknown;
    const std::array<const void*, 2> identities{&first, &second};
    assert(identity_index(&first, identities) == 0);
    assert(identity_index(&second, identities) == 1);
    assert(identity_index(&unknown, identities) == identities.size());
    for (std::size_t bootstraps : {0, 1, 2, 4, 8}) {
        const auto expected = expected_bootstrap_counts(bootstraps);
        validate_bootstrap_counts(expected, bootstraps);
        auto missing = expected;
        ++missing[1];
        bool rejected = false;
        try { validate_bootstrap_counts(missing, bootstraps); }
        catch (const std::runtime_error&) { rejected = true; }
        assert(rejected);
    }
    bool rejected = false;
    try { expected_bootstrap_counts(std::numeric_limits<std::size_t>::max()); }
    catch (const std::overflow_error&) { rejected = true; }
    assert(rejected);
}
