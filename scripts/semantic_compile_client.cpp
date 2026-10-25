// CPU-only transport: c++ -O2 -std=c++17 semantic_compile_client.cpp -o CLIENT
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/file.h>
#include <unistd.h>
#include <fcntl.h>
#include <cerrno>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr size_t max_frame = 1024 * 1024;
std::string env(const char* key) { const char* value = std::getenv(key); return value ? value : ""; }
void integer(std::string& data, uint32_t value) {
    value = htonl(value); data.append(reinterpret_cast<const char*>(&value), 4);
}
void field(std::string& data, const std::string& value) { integer(data, value.size()); data += value; }
uint32_t number(const std::string& data, size_t& offset) {
    if (offset + 4 > data.size()) throw std::runtime_error("truncated service response");
    uint32_t value; std::memcpy(&value, data.data() + offset, 4); offset += 4; return ntohl(value);
}
std::string field(const std::string& data, size_t& offset) {
    auto size = number(data, offset);
    if (size > data.size() - offset) throw std::runtime_error("truncated service response");
    auto value = data.substr(offset, size); offset += size; return value;
}
void transfer(int fd, char* data, size_t size, bool writing) {
    while (size) {
        auto count = writing ? send(fd, data, size, MSG_NOSIGNAL) : recv(fd, data, size, 0);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) throw std::runtime_error("compile service connection failed");
        data += count; size -= count;
    }
}
std::string quoted(const std::string& text) {
    const char* hex = "0123456789abcdef";
    std::string result = "\"";
    for (unsigned char c : text) {
        if (c == '"' || c == '\\') { result += '\\'; result += c; }
        else if (c < 32 || c >= 127) {
            result += "\\u00"; result += hex[c >> 4]; result += hex[c & 15];
        } else result += c;
    }
    return result + '"';
}
void record_error(const std::string& error) {
    auto path = env("CUBUTTERFLY_PREPARE_EVENTS");
    if (path.empty()) throw std::runtime_error("CUBUTTERFLY_PREPARE_EVENTS is missing");
    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
    int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
    if (fd < 0) throw std::runtime_error("cannot record compile transport failure");
    if (flock(fd, LOCK_EX)) { close(fd); throw std::runtime_error("cannot lock compile event file"); }
    auto line = "{\"kind\":\"broker_error\",\"error\":" + quoted(error) + "}\n";
    size_t offset = 0;
    while (offset < line.size()) {
        auto count = write(fd, line.data() + offset, line.size() - offset);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) { close(fd); throw std::runtime_error("cannot write compile event file"); }
        offset += count;
    }
    if (fsync(fd)) { close(fd); throw std::runtime_error("cannot flush compile event file"); }
    close(fd);
}
uintmax_t event_size() {
    auto path = env("CUBUTTERFLY_PREPARE_EVENTS");
    return std::filesystem::exists(path) ? std::filesystem::file_size(path) : 0;
}
}

int main(int argc, char** argv) {
    int fd = -1;
    try {
        auto endpoint = env("CUBUTTERFLY_COMPILE_SOCKET");
        if (endpoint.empty()) throw std::runtime_error("CUBUTTERFLY_COMPILE_SOCKET is required");
        if (env("CUBUTTERFLY_PREPARE_BROKER").empty() || env("CUBUTTERFLY_PREPARE_EVENTS").empty())
            throw std::runtime_error("compile transport requires broker and preparation events");
        const auto previous_event_size = event_size();
        sockaddr_un address{}; address.sun_family = AF_UNIX;
        if (endpoint.size() >= sizeof(address.sun_path)) throw std::runtime_error("compile socket path too long");
        std::memcpy(address.sun_path, endpoint.c_str(), endpoint.size() + 1);
        fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
        if (fd < 0) throw std::runtime_error("cannot create compile service socket");
        timeval timeout{15, 0};
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
        if (connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)))
            throw std::runtime_error("compile service unavailable: " + std::string(std::strerror(errno)));
        std::string data;
        field(data, "cubutterfly-compile-service-v1"); field(data, std::to_string(getppid()));
        field(data, std::filesystem::current_path().string());
        for (auto name : {"CUBUTTERFLY_PREPARE_BROKER", "CUBUTTERFLY_PREPARE_EVENTS",
                          "CUBUTTERFLY_WORKER_MAX_VMAS", "CUBUTTERFLY_WORKER_MAX_RSS_BYTES", "PATH"}) field(data, env(name));
        for (int i = 1; i < argc; ++i) field(data, argv[i]);
        if (data.size() > max_frame) throw std::runtime_error("compile request exceeds frame limit");
        std::string header; integer(header, data.size());
        transfer(fd, header.data(), header.size(), true); transfer(fd, data.data(), data.size(), true);
        header.resize(4); transfer(fd, header.data(), 4, false);
        size_t offset = 0; auto size = number(header, offset);
        if (size > max_frame) throw std::runtime_error("compile response exceeds frame limit");
        data.resize(size); transfer(fd, data.data(), size, false);
        offset = 0; auto code = number(data, offset); auto out = field(data, offset); auto err = field(data, offset);
        if (offset != data.size() || code > 255) throw std::runtime_error("invalid compile service response");
        close(fd); fd = -1;
        // MEASURE may catch a failed candidate; its attempt must retain evidence.
        if (code && event_size() <= previous_event_size)
            record_error("compile service returned nonzero without preparation event: " + err);
        std::cout << out; std::cerr << err;
        return code;
    } catch (const std::exception& error) {
        if (fd >= 0) close(fd);
        try { record_error(error.what()); } catch (const std::exception& event_error) {
            std::cerr << "semantic compile client: preparation event recording failed: " << event_error.what() << '\n';
        }
        std::cerr << "semantic compile client: " << error.what() << '\n';
        return 1;
    }
}
