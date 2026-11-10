#pragma once
#include <nlohmann/json.hpp>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;
namespace cuntt::detail {

// A CPU-only process boundary. No shell, device initialization, compilation or
// measurement is permitted in the ranking helper. Files live only for this call.
inline nlohmann::json run_static_ranker(const nlohmann::json& request,
                                      const std::string& python, const std::string& script) {
    struct Temporary {
        std::filesystem::path path;
        Temporary() {
            auto pattern = (std::filesystem::temp_directory_path() / "cubutterfly-rank-XXXXXX").string();
            auto* created = mkdtemp(pattern.data());
            if (!created) throw std::runtime_error("cannot create static ranking workspace");
            path = created;
        }
        ~Temporary() { std::error_code error; std::filesystem::remove_all(path, error); }
    } temporary;
    const auto input = temporary.path / "request.json", output = temporary.path / "ranking.json";
    { std::ofstream stream(input); stream << request;
      if (!stream) throw std::runtime_error("cannot write static ranking request"); }
    std::vector<std::string> command{python, script, "--request", input.string(), "--output", output.string()};
    std::vector<char*> argv;
    for (auto& value : command) argv.push_back(value.data());
    argv.push_back(nullptr);
    pid_t process = -1;
    if (posix_spawnp(&process, argv[0], nullptr, nullptr, argv.data(), environ))
        throw std::runtime_error("cannot start CPU static ranking helper");
    int status = 0;
    while (waitpid(process, &status, 0) < 0)
        if (errno != EINTR) throw std::runtime_error("cannot wait for static ranking helper");
    if (!WIFEXITED(status) || WEXITSTATUS(status))
        throw std::runtime_error("static ranking helper failed; inspect its diagnostic above");
    std::ifstream stream(output);
    if (!stream) throw std::runtime_error("static ranking helper did not produce a result");
    nlohmann::json result; stream >> result;
    if (result.value("schema", "") != "cubutterfly-static-ranking-v1")
        throw std::runtime_error("unsupported static ranking response schema");
    return result;
}

// A partition of the submitted inventory is required: a helper must neither
// lose candidates nor introduce indices that refer to a different request.
inline void validate_static_ranking(const nlohmann::json& result, const std::set<std::size_t>& submitted) {
    std::set<std::size_t> seen;
    double previous = -1;
    for (const auto* category : {"ranked", "unranked", "rejected"}) {
        const auto& rows = result.at(category);
        if (!rows.is_array()) throw std::runtime_error("static ranking category must be an array");
        for (const auto& row : rows) {
            const auto& index = row.at("index");
            if (!index.is_number_unsigned() && (!index.is_number_integer() || index.get<long long>() < 0))
                throw std::runtime_error("static ranking index must be nonnegative");
            const auto i = index.get<std::size_t>();
            if (!submitted.count(i) || !seen.insert(i).second)
                throw std::runtime_error("static ranking response has an unknown or duplicate candidate");
            if (std::string(category) == "ranked") {
                const auto score = row.at("predicted_kernel_ms").get<double>();
                if (!std::isfinite(score) || score <= 0 || score < previous)
                    throw std::runtime_error("static ranking scores must be positive, finite and ordered");
                previous = score;
            }
        }
    }
    if (seen != submitted) throw std::runtime_error("static ranking response lost candidates");
}
} // namespace cuntt::detail
