#include <algorithm>
#include <cmath>
#include <vector>

// Keep double arithmetic and active/dependent iteration order identical to
// stage_cost_model.compose; compilation deliberately disables contraction.
extern "C" double cubutterfly_compose(const double* stages, int groups, int tiles,
                                     int factor, int overlap, double last_fraction,
                                     const double* tail) {
    struct Node {
        double remaining, sm, hbm, dispatch;
        int pending = 0;
        std::vector<int> dependents;
    };
    const int repeated = groups - (factor ? 1 : 0);
    const int count = repeated * tiles + (factor ? 1 : 0);
    std::vector<Node> nodes(count);
    auto node_id = [=](int group, int tile) {
        return factor && group == groups - 1 ? count - 1 : tile * repeated + group;
    };
    for (int i = 0; i < count; ++i) {
        const int group = factor && i == count - 1 ? groups - 1 : i % repeated;
        const int tile = factor && i == count - 1 ? 0 : i / repeated;
        int deps[4], size = 0;
        auto add = [&](int dep) {
            if (std::find(deps, deps + size, dep) == deps + size) deps[size++] = dep;
        };
        if (tile) add(node_id(group, tile - 1));
        if (group) add(node_id(group - 1, factor && group == groups - 1 ? tiles - 1 : tile));
        if (!factor && tile >= 2 && group + 1 < groups) add(node_id(group + 1, tile - 2));
        if (!overlap && i) add(i - 1);
        const double* stage = stages + 4 * group;
        double fraction = !factor && tile == tiles - 1 ? last_fraction : 1.0;
        if (tail && !factor && tile == tiles - 1) {
            stage = tail + 4 * group;
            fraction = 1.0;
        }
        Node& node = nodes[i];
        node.remaining = stage[0] + stage[1] * fraction;
        node.sm = std::min(1.0, std::max(stage[2], 0.0));
        node.hbm = std::min(1.0, std::max(stage[3], 0.0));
        node.dispatch = std::min(1.0, stage[0] / node.remaining);
        node.pending = size;
        for (int j = 0; j < size; ++j) nodes[deps[j]].dependents.push_back(i);
    }
    std::vector<int> active, done, remaining;
    active.reserve(groups * 3);
    done.reserve(groups * 3);
    remaining.reserve(groups * 3);
    for (int i = 0; i < count; ++i) if (!nodes[i].pending) active.push_back(i);
    double elapsed = 0.0;
    int completed = 0;
    while (!active.empty()) {
        double sm = 0.0, hbm = 0.0, dispatch = 0.0;
        double work = nodes[active[0]].remaining;
        for (int i : active) {
            sm += nodes[i].sm;
            hbm += nodes[i].hbm;
            dispatch += nodes[i].dispatch;
            work = std::min(work, nodes[i].remaining);
        }
        const double pressure = std::max(std::max(std::max(1.0, sm), hbm), dispatch);
        elapsed += work * pressure;
        done.clear();
        remaining.clear();
        for (int i : active) {
            if (nodes[i].remaining <= work + 1e-12) done.push_back(i);
            else {
                nodes[i].remaining -= work;
                remaining.push_back(i);
            }
        }
        active.swap(remaining);
        for (int i : done) {
            ++completed;
            for (int consumer : nodes[i].dependents) {
                if (!--nodes[consumer].pending) active.push_back(consumer);
            }
        }
    }
    return completed == count ? elapsed : std::nan("");
}
