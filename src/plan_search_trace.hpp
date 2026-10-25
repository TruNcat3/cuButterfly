#pragma once

#include <nlohmann/json.hpp>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <exception>
#include <mutex>
#include <utility>

namespace cuntt::detail {

// Diagnostics must never turn a legal measurement into a rejected candidate.
class PlanSearchTrace {
    using Clock = std::chrono::steady_clock;
    std::FILE* sink_ = nullptr;
    bool owns_sink_ = false;
    nlohmann::json context_;
    std::uint64_t search_id_ = 0;
    inline static std::atomic<std::uint64_t> next_id_{0};
    inline static std::mutex output_mutex_;

    void fail(const char* reason) noexcept {
        std::fprintf(stderr, "cuButterfly: CUBUTTERFLY_PLAN_TRACE disabled: %s\n", reason);
        if (owns_sink_ && sink_) std::fclose(sink_);
        sink_ = nullptr;
        owns_sink_ = false;
    }

public:
    explicit PlanSearchTrace(const char* search) noexcept {
        const char* destination = std::getenv("CUBUTTERFLY_PLAN_TRACE");
        if (!destination || !*destination || std::strcmp(destination, "0") == 0) return;
        try {
            context_ = {{"search", search}};
            search_id_ = ++next_id_;
            if (std::strcmp(destination, "1") == 0 || std::strcmp(destination, "stderr") == 0)
                sink_ = stderr;
            else {
                sink_ = std::fopen(destination, "a");
                owns_sink_ = true;
                if (!sink_) fail("cannot open trace destination");
            }
        } catch (...) { fail("cannot initialize trace"); }
    }
    ~PlanSearchTrace() { if (owns_sink_ && sink_) std::fclose(sink_); }
    PlanSearchTrace(const PlanSearchTrace&) = delete;
    PlanSearchTrace& operator=(const PlanSearchTrace&) = delete;
    bool enabled() const noexcept { return sink_ != nullptr; }

    template<class Fields>
    void context(Fields&& fields) noexcept {
        if (!enabled()) return;
        try { context_ = fields(); }
        catch (...) { fail("cannot serialize trace context"); }
    }

    template<class Fields>
    void event(const char* phase, Fields&& fields) noexcept {
        if (!enabled()) return;
        try {
            auto row = context_;
            row.update(fields());
            row["schema"] = "cubutterfly-plan-search-trace-v1";
            row["search_id"] = search_id_;
            row["phase"] = phase;
            const auto line = row.dump() + '\n';
            std::lock_guard<std::mutex> guard(output_mutex_);
            if (std::fwrite(line.data(), 1, line.size(), sink_) != line.size() || std::fflush(sink_) != 0)
                fail("cannot write trace destination");
        } catch (...) { fail("cannot serialize trace event"); }
    }

    class Phase {
        PlanSearchTrace* trace_;
        const char* name_;
        Clock::time_point start_;
        int exceptions_ = 0;
    public:
        Phase(PlanSearchTrace* trace, const char* name) noexcept : trace_(trace), name_(name) {
            if (trace_ && trace_->enabled()) {
                start_ = Clock::now();
                exceptions_ = std::uncaught_exceptions();
            } else trace_ = nullptr;
        }
        ~Phase() {
            if (!trace_) return;
            const auto elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start_).count();
            trace_->event(name_, [&] {
                return nlohmann::json{{"wall_ms", elapsed},
                    {"outcome", std::uncaught_exceptions() > exceptions_ ? "exception" : "complete"}};
            });
        }
        Phase(const Phase&) = delete;
        Phase& operator=(const Phase&) = delete;
    };

    template<class Cleanup>
    class OnExit {
        PlanSearchTrace* trace_;
        const char* phase_;
        Cleanup cleanup_;
    public:
        OnExit(PlanSearchTrace* trace, const char* phase, Cleanup cleanup)
            : trace_(trace), phase_(phase), cleanup_(std::move(cleanup)) {}
        ~OnExit() {
            Phase phase(trace_, phase_);
            cleanup_();
        }
        OnExit(const OnExit&) = delete;
        OnExit& operator=(const OnExit&) = delete;
    };
    template<class Cleanup>
    static OnExit<Cleanup> on_exit(PlanSearchTrace* trace, const char* phase, Cleanup cleanup) {
        return OnExit<Cleanup>(trace, phase, std::move(cleanup));
    }
};

} // namespace cuntt::detail
