#pragma once

#include <cuntt/rns_tuning.hpp>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

namespace cubutterfly_heongpu {

// Preparation-only integration shared by the scalar and stage adapters. No
// environment lookup, file access or tuning is performed by execute_async.
inline cuntt::RnsPreparation prepare_rns(cuntt::RnsPlan& plan,
                                       cuntt::RnsPlacement placement,
                                       cudaStream_t stream,
                                       const char* origin) {
    cuntt::RnsPrepareOptions options;
    if (const auto* directory = std::getenv("CUBUTTERFLY_RNS_CACHE"))
        options.cache_directory = directory;
    if (const auto* policy = std::getenv("CUBUTTERFLY_RNS_CACHE_POLICY")) {
        const std::string value(policy);
        if (value == "prefer") options.policy = cuntt::RnsCachePolicy::Prefer;
        else if (value == "require") options.policy = cuntt::RnsCachePolicy::Require;
        else if (value == "refresh") options.policy = cuntt::RnsCachePolicy::Refresh;
        else throw std::invalid_argument("CUBUTTERFLY_RNS_CACHE_POLICY must be prefer, require or refresh");
    }
    // The default keeps the lightweight installation path.  A deployment or
    // research run may request a larger bounded confirmation budget without
    // changing the operator semantics or introducing an application-specific
    // mapping.  The repetition count is part of the cache identity.
    if (const auto* repetitions = std::getenv("CUBUTTERFLY_RNS_TUNE_REPETITIONS")) {
        char* end = nullptr;
        errno = 0;
        const auto value = std::strtoul(repetitions, &end, 10);
        if (errno || end == repetitions || *end != '\0' || value == 0 || value > 1000)
            throw std::invalid_argument("CUBUTTERFLY_RNS_TUNE_REPETITIONS must be an integer in [1,1000]");
        options.repetitions = static_cast<unsigned>(value);
    }
    auto prepared = cuntt::prepare_rns_plan(plan, placement, options, stream);
    if (const auto* path = std::getenv("CUBUTTERFLY_RNS_REQUEST_EXPORT")) {
        if (!*path) throw std::invalid_argument("CUBUTTERFLY_RNS_REQUEST_EXPORT must be nonempty");
        // origin is an internal literal, never caller-supplied JSON.
        if (std::strcmp(origin, "scalar") && std::strcmp(origin, "stage"))
            throw std::invalid_argument("invalid RNS export origin");
        const std::string line = std::string("{\"schema\":\"cubutterfly-rns-request-v1\",\"origin\":\"") +
            origin + "\",\"placement\":\"" +
            (placement == cuntt::RnsPlacement::InPlace ? "in_place" : "separate") +
            "\",\"config\":" + cuntt::rns_config_json(plan.config()) +
            ",\"mapping\":" + cuntt::rns_mapping_json(plan.mapping()) +
            ",\"cache_hit\":" + (prepared.cache_hit ? "true" : "false") + "}\n";
        const int fd = ::open(path, O_WRONLY | O_CREAT | O_APPEND, 0666);
        if (fd < 0) throw std::runtime_error("cannot open RNS request export: " + std::string(std::strerror(errno)));
        try {
            if (::flock(fd, LOCK_EX)) throw std::runtime_error("cannot lock RNS request export");
            std::size_t written = 0;
            while (written < line.size()) {
                const auto n = ::write(fd, line.data() + written, line.size() - written);
                if (n < 0 && errno == EINTR) continue;
                if (n <= 0) throw std::runtime_error("cannot write RNS request export");
                written += static_cast<std::size_t>(n);
            }
            ::flock(fd, LOCK_UN);
            ::close(fd);
        } catch (...) {
            ::close(fd);
            throw;
        }
    }
    return prepared;
}

} // namespace cubutterfly_heongpu
