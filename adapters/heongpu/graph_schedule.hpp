#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cubutterfly_heongpu {

struct GraphScheduleCandidate {
    std::string name;
    std::function<void(cudaStream_t)> launch;
};

struct GraphScheduleOptions {
    std::size_t warmup = 3;
    std::size_t rounds = 7;
    std::size_t replays = 5;
};

struct GraphScheduleMeasurement {
    std::string name;
    // Each sample is the event interval divided by replays, in milliseconds.
    std::vector<double> round_ms;
    double median_ms = std::numeric_limits<double>::quiet_NaN();
};

// Pure helpers: no CUDA context or device is needed to use these functions.
inline double graph_schedule_median_ms(std::vector<double> samples) {
    if (samples.empty()) throw std::invalid_argument("graph schedule has no samples");
    for (double value : samples)
        if (!std::isfinite(value) || value <= 0)
            throw std::invalid_argument("graph schedule sample must be finite and positive");
    std::sort(samples.begin(), samples.end());
    const std::size_t middle = samples.size() / 2;
    return samples.size() % 2 ? samples[middle]
                             : samples[middle - 1] + (samples[middle] - samples[middle - 1]) / 2;
}

inline std::size_t graph_schedule_select(const std::vector<double>& medians) {
    std::size_t selected = medians.size();
    double best = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < medians.size(); ++i) {
        if (std::isfinite(medians[i]) && medians[i] > 0 && medians[i] < best) {
            best = medians[i];
            selected = i;
        }
    }
    if (selected == medians.size())
        throw std::invalid_argument("graph schedule has no valid measured candidate");
    return selected;
}

namespace graph_schedule_detail {
inline void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

class Event {
public:
    Event() { check(cudaEventCreate(&event_), "cudaEventCreate"); }
    ~Event() { if (event_) cudaEventDestroy(event_); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    cudaEvent_t get() const { return event_; }
private:
    cudaEvent_t event_ = nullptr;
};
} // namespace graph_schedule_detail

// Callers must validate every candidate's mathematics before tuning. This class
// only schedules launches; it owns neither graphs nor buffers. Resources borrowed
// by the stored callables must outlive this instance and all pending launches.
// Candidates must enqueue the complete equivalent operation on the given explicit
// stream, with no CPU work or other-stream work required for completion. Replays
// must be legal without resetting inputs. Calls on one instance must not overlap.
// Results are measurements for this instance, not a persistent tuning cache.
class GraphSchedule {
public:
    explicit GraphSchedule(std::vector<GraphScheduleCandidate> candidates)
        : candidates_(std::move(candidates)) {
        if (candidates_.empty()) throw std::invalid_argument("graph schedule has no candidates");
        for (const auto& candidate : candidates_)
            if (candidate.name.empty() || !candidate.launch)
                throw std::invalid_argument("graph schedule needs named callable candidates");
    }

    // before_round runs once before each measured round; after_sample runs once
    // after each synchronized sample. Neither callback is timed. Work enqueued by
    // guards on stream is drained before the next sample. Warmup is unmeasured.
    // A failed tuning attempt invalidates any previous selection.
    void tune(cudaStream_t stream, const GraphScheduleOptions& options = {},
              const std::function<void()>& before_round = {},
              const std::function<void()>& after_sample = {}) {
        using graph_schedule_detail::check;
        selected_ = no_selection();
        measurements_.clear();
        if (!stream || stream == cudaStreamLegacy || stream == cudaStreamPerThread)
            throw std::invalid_argument("graph schedule tuning requires an explicit nondefault stream");
        if (!options.rounds || !options.replays)
            throw std::invalid_argument("graph schedule rounds and replays must be positive");
        cudaStreamCaptureStatus capture;
        check(cudaStreamIsCapturing(stream, &capture), "cudaStreamIsCapturing");
        if (capture != cudaStreamCaptureStatusNone)
            throw std::logic_error("graph schedule cannot tune during stream capture");

        graph_schedule_detail::Event start, stop;
        std::vector<GraphScheduleMeasurement> measured;
        measured.reserve(candidates_.size());
        for (const auto& candidate : candidates_)
            measured.push_back({candidate.name, {}, std::numeric_limits<double>::quiet_NaN()});
        for (auto& result : measured) result.round_ms.reserve(options.rounds);
        check(cudaStreamSynchronize(stream), "graph schedule initial synchronization");
        for (const auto& candidate : candidates_)
            for (std::size_t i = 0; i < options.warmup; ++i) candidate.launch(stream);
        check(cudaStreamSynchronize(stream), "graph schedule warmup synchronization");

        for (std::size_t round = 0; round < options.rounds; ++round) {
            if (before_round) before_round();
            // Rotate the first candidate to reduce consistent order bias.
            for (std::size_t offset = 0; offset < candidates_.size(); ++offset) {
                const std::size_t index = (round % candidates_.size() + offset) % candidates_.size();
                check(cudaStreamSynchronize(stream), "graph schedule guard synchronization");
                check(cudaEventRecord(start.get(), stream), "graph schedule start event");
                for (std::size_t replay = 0; replay < options.replays; ++replay)
                    candidates_[index].launch(stream);
                check(cudaEventRecord(stop.get(), stream), "graph schedule stop event");
                check(cudaEventSynchronize(stop.get()), "graph schedule sample synchronization");
                float elapsed = 0;
                check(cudaEventElapsedTime(&elapsed, start.get(), stop.get()), "cudaEventElapsedTime");
                const double sample = static_cast<double>(elapsed) / options.replays;
                if (!std::isfinite(sample) || sample <= 0)
                    throw std::runtime_error("graph schedule measured a nonpositive or nonfinite interval");
                measured[index].round_ms.push_back(sample);
                if (after_sample) after_sample();
            }
        }
        std::vector<double> medians;
        medians.reserve(measured.size());
        for (auto& result : measured) {
            result.median_ms = graph_schedule_median_ms(result.round_ms);
            medians.push_back(result.median_ms);
        }
        const auto selected = graph_schedule_select(medians);
        measurements_ = std::move(measured);
        selected_ = selected;
    }

    // The steady-state path adds no events, synchronization, or measurement.
    // The caller supplies a stream compatible with the borrowed graph resources.
    void launch(cudaStream_t stream) const {
        if (!tuned()) throw std::logic_error("graph schedule must be tuned before launch");
        candidates_[selected_].launch(stream);
    }

    bool tuned() const { return selected_ != no_selection(); }
    std::size_t selected_index() const {
        if (!tuned()) throw std::logic_error("graph schedule has no selection");
        return selected_;
    }
    const GraphScheduleCandidate& selected_candidate() const {
        return candidates_[selected_index()];
    }
    const std::vector<GraphScheduleMeasurement>& measurements() const { return measurements_; }

private:
    static constexpr std::size_t no_selection() { return std::numeric_limits<std::size_t>::max(); }
    std::vector<GraphScheduleCandidate> candidates_;
    std::vector<GraphScheduleMeasurement> measurements_;
    std::size_t selected_ = no_selection();
};

} // namespace cubutterfly_heongpu
