// Isolated cryptographic NTT composition benchmark.
//
// The benchmark deliberately composes independent single-word NTT plans.  It
// does not implement CRT/RNS fusion: channels are laid out as
// [channel][batch][coefficient] and are submitted sequentially on one stream.

#include <cuda_runtime.h>

#include <cubutterfly/mapping.hpp>
#include <cuntt/ntt.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;

void check_cuda(cudaError_t status, const char* context) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
    }
}

template <typename T>
class DeviceBuffer {
  public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t count) { allocate(count); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept : pointer_(other.pointer_), count_(other.count_) {
        other.pointer_ = nullptr;
        other.count_ = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            release();
            pointer_ = other.pointer_;
            count_ = other.count_;
            other.pointer_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }
    ~DeviceBuffer() { release(); }

    void allocate(std::size_t count) {
        if (count == 0) return;
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::overflow_error("device allocation size overflows size_t");
        }
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&pointer_), count * sizeof(T)), "cudaMalloc");
        count_ = count;
    }
    void release() noexcept {
        if (pointer_ != nullptr) cudaFree(pointer_);
        pointer_ = nullptr;
        count_ = 0;
    }
    T* get() const noexcept { return pointer_; }
    std::size_t size() const noexcept { return count_; }

  private:
    T* pointer_ = nullptr;
    std::size_t count_ = 0;
};

class Stream {
  public:
    Stream() { check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreateWithFlags"); }
    ~Stream() { if (stream_ != nullptr) cudaStreamDestroy(stream_); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    cudaStream_t get() const noexcept { return stream_; }

  private:
    cudaStream_t stream_ = nullptr;
};

class Event {
  public:
    Event() { check_cuda(cudaEventCreateWithFlags(&event_, cudaEventDefault), "cudaEventCreate"); }
    ~Event() { if (event_ != nullptr) cudaEventDestroy(event_); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    cudaEvent_t get() const noexcept { return event_; }

  private:
    cudaEvent_t event_ = nullptr;
};

enum class Mode { Cyclic, Negacyclic, Coset };
enum class Direction { Forward, Inverse };
enum class InputPattern { Random, Boundary };

struct Options {
    std::uint32_t log_n = 12;
    std::size_t batch = 1;
    std::uint32_t word_bits = 64;
    std::vector<std::uint64_t> moduli;
    Mode mode = Mode::Cyclic;
    Direction direction = Direction::Forward;
    std::uint64_t coset_generator = 7;
    std::uint32_t warmup = 5;
    std::uint32_t repeat = 20;
    std::uint64_t seed = 1;
    InputPattern input_pattern = InputPattern::Random;
    std::string mapping_json;
    bool json = false;
    bool self_test_host = false;
};

struct RootInfo {
    std::uint64_t root = 0;
    std::uint64_t default_root_2n = 0;
    std::uint64_t default_root_2n_squared = 0;
    std::uint64_t psi = 0;
    bool root_match = true;
};

struct ChannelInfo {
    std::uint64_t modulus = 0;
    RootInfo roots;
    std::vector<std::uint64_t> twist;
    std::vector<std::uint64_t> twist_shoup;
};

std::string mode_name(Mode mode) {
    switch (mode) {
        case Mode::Cyclic: return "cyclic";
        case Mode::Negacyclic: return "negacyclic";
        case Mode::Coset: return "coset";
    }
    return "unknown";
}

std::string direction_name(Direction direction) {
    return direction == Direction::Forward ? "forward" : "inverse";
}

std::string pattern_name(InputPattern pattern) {
    return pattern == InputPattern::Random ? "random" : "boundary";
}

std::uint64_t parse_u64(const std::string& text, const char* name) {
    if (text.empty() || text.front() == '-') throw std::invalid_argument(std::string("invalid ") + name);
    std::size_t used = 0;
    const auto value = std::stoull(text, &used, 0);
    if (used != text.size()) throw std::invalid_argument(std::string("invalid ") + name + ": " + text);
    return static_cast<std::uint64_t>(value);
}

std::string take_arg(int& index, int argc, char** argv) {
    if (index + 1 >= argc) throw std::invalid_argument(std::string("missing value for ") + argv[index]);
    return argv[++index];
}

std::vector<std::uint64_t> parse_moduli(const std::string& text) {
    std::vector<std::uint64_t> values;
    std::istringstream stream(text);
    for (std::string token; std::getline(stream, token, ',');) {
        if (token.empty()) throw std::invalid_argument("empty modulus in --moduli");
        values.push_back(parse_u64(token, "modulus"));
    }
    if (values.empty()) throw std::invalid_argument("--moduli must not be empty");
    return values;
}

std::string read_mapping(const std::string& value) {
    if (value.empty()) return {};
    if (value.front() == '{' || value.front() == '[') return value;
    std::ifstream input(value);
    if (!input) throw std::invalid_argument("cannot open --mapping-json: " + value);
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

std::vector<std::string> mapping_for_channels(const std::string& raw, std::size_t channels) {
    if (raw.empty()) return std::vector<std::string>(channels);
    const auto parsed = Json::parse(raw);
    if (!parsed.is_array()) return std::vector<std::string>(channels, raw);
    if (parsed.size() != channels) {
        throw std::invalid_argument("mapping array length must equal the number of modulus channels");
    }
    std::vector<std::string> result(channels);
    for (std::size_t i = 0; i < channels; ++i) {
        if (!parsed[i].is_null()) result[i] = parsed[i].dump();
    }
    return result;
}

std::size_t checked_mul(std::size_t left, std::size_t right, const char* context) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::overflow_error(std::string(context) + " overflows size_t");
    }
    return left * right;
}

std::vector<std::uint32_t> default_partition(std::uint32_t log_n) {
    // Eight stages fit comfortably in the shared-iterative 128-thread
    // codelet for both supported word widths.  More groups are introduced as
    // N grows, but the logical partition remains bounded and explicit.
    std::vector<std::uint32_t> partition;
    while (log_n != 0) {
        const auto stages = std::min<std::uint32_t>(8, log_n);
        partition.push_back(stages);
        log_n -= stages;
    }
    if (partition.empty()) partition.push_back(1);
    return partition;
}

std::uint64_t mod_mul(std::uint64_t left, std::uint64_t right, std::uint64_t modulus) {
    return static_cast<std::uint64_t>((static_cast<unsigned __int128>(left) * right) % modulus);
}

std::uint64_t mod_pow(std::uint64_t base, std::uint64_t exponent, std::uint64_t modulus) {
    std::uint64_t result = 1 % modulus;
    base %= modulus;
    while (exponent != 0) {
        if (exponent & 1U) result = mod_mul(result, base, modulus);
        base = mod_mul(base, base, modulus);
        exponent >>= 1U;
    }
    return result;
}

std::uint64_t shoup_precompute(std::uint64_t value, std::uint64_t modulus, std::uint32_t word_bits) {
    if (word_bits == 32) {
        return static_cast<std::uint64_t>((static_cast<unsigned __int128>(value) << 32) / modulus);
    }
    return static_cast<std::uint64_t>((static_cast<unsigned __int128>(value) << 64) / modulus);
}

std::vector<std::uint64_t> powers(std::uint64_t base, std::size_t count, std::uint64_t modulus) {
    std::vector<std::uint64_t> result(count, 1);
    base %= modulus;
    for (std::size_t index = 1; index < count; ++index) result[index] = mod_mul(result[index - 1], base, modulus);
    return result;
}

std::uint64_t find_psi_for_root(std::uint32_t log_n, std::uint64_t modulus, std::uint64_t root,
                                std::uint64_t& default_psi, std::uint64_t& default_squared) {
    if (log_n >= 62) throw std::invalid_argument("negacyclic root order overflows uint64_t");
    default_psi = cuntt::find_primitive_power_of_two_root(log_n + 1, modulus);
    default_squared = mod_mul(default_psi, default_psi, modulus);
    if (default_squared == root) return default_psi;

    const auto order = std::uint64_t{1} << (log_n + 1);
    if ((modulus - 1) % order != 0) throw std::invalid_argument("modulus does not support a 2N negacyclic root");
    for (std::uint64_t candidate = 2; candidate < (std::uint64_t{1} << 20); ++candidate) {
        const auto psi = mod_pow(candidate, (modulus - 1) / order, modulus);
        if (mod_pow(psi, order, modulus) == 1 && mod_pow(psi, order >> 1U, modulus) != 1 &&
            mod_mul(psi, psi, modulus) == root) {
            return psi;
        }
    }
    throw std::runtime_error("could not select a 2N root whose square equals the plan root");
}

ChannelInfo make_channel(std::uint32_t log_n, std::uint64_t modulus, Mode mode, std::uint64_t coset_generator,
                         Direction direction) {
    const auto n = std::size_t{1} << log_n;
    ChannelInfo channel;
    channel.modulus = modulus;
    channel.roots.root = cuntt::find_primitive_power_of_two_root(log_n, modulus);
    if (mode == Mode::Negacyclic) {
        if ((modulus - 1) % (std::uint64_t{2} * n) != 0) {
            throw std::invalid_argument("negacyclic mode requires 2N to divide modulus-1");
        }
        channel.roots.psi = find_psi_for_root(log_n, modulus, channel.roots.root,
                                               channel.roots.default_root_2n,
                                               channel.roots.default_root_2n_squared);
        channel.roots.root_match = mod_mul(channel.roots.psi, channel.roots.psi, modulus) == channel.roots.root;
        if (!channel.roots.root_match) throw std::logic_error("selected negacyclic root does not match Plan root");
        channel.twist = powers(direction == Direction::Forward ? channel.roots.psi :
                               mod_pow(channel.roots.psi, modulus - 2, modulus), n, modulus);
    } else if (mode == Mode::Coset) {
        const auto generator = coset_generator % modulus;
        if (generator == 0) throw std::invalid_argument("coset generator must be nonzero modulo the field");
        const auto base = direction == Direction::Forward ? generator : mod_pow(generator, modulus - 2, modulus);
        channel.twist = powers(base, n, modulus);
    }
    if (!channel.twist.empty()) {
        channel.twist_shoup.resize(channel.twist.size());
        for (std::size_t index = 0; index < channel.twist.size(); ++index) {
            channel.twist_shoup[index] = shoup_precompute(channel.twist[index], modulus, 64);
        }
    }
    return channel;
}

void validate_options(const Options& options) {
    if (options.log_n == 0 || options.log_n > 30) throw std::invalid_argument("--log-n must be in [1,30]");
    if (options.batch == 0) throw std::invalid_argument("--batch must be positive");
    if (options.word_bits != 32 && options.word_bits != 64) throw std::invalid_argument("--word-bits must be 32 or 64");
    if (options.warmup > 100000 || options.repeat == 0 || options.repeat > 100000) {
        throw std::invalid_argument("warmup/repeat are outside the supported finite range");
    }
    if (options.moduli.empty()) throw std::invalid_argument("at least one modulus is required");
    for (std::size_t index = 0; index < options.moduli.size(); ++index) {
        const auto modulus = options.moduli[index];
        for (std::size_t previous = 0; previous < index; ++previous)
            if (options.moduli[previous] == modulus)
                throw std::invalid_argument("RNS modulus channels must be pairwise distinct");
        if (modulus < 3 || modulus >= (std::uint64_t{1} << 63)) throw std::invalid_argument("modulus must be in [3,2^63)");
        if (options.word_bits == 32 && modulus >= (std::uint64_t{1} << 31)) {
            throw std::invalid_argument("32-bit words require modulus < 2^31");
        }
        if (!cuntt::is_prime(modulus)) throw std::invalid_argument("every modulus must be prime");
        const auto n = std::uint64_t{1} << options.log_n;
        if ((modulus - 1) % n != 0) throw std::invalid_argument("modulus-1 must be divisible by N");
        if (options.mode == Mode::Negacyclic && (modulus - 1) % (2 * n) != 0) {
            throw std::invalid_argument("negacyclic mode requires modulus-1 divisible by 2N");
        }
    }
    const auto n = std::size_t{1} << options.log_n;
    const auto points = checked_mul(checked_mul(n, options.batch, "input point count"), options.moduli.size(), "RNS point count");
    const auto input_bytes = checked_mul(points, options.word_bits / 8, "RNS input bytes");
    if (input_bytes > (std::size_t{2} << 30)) {
        throw std::invalid_argument("RNS input envelope exceeds the 2 GiB benchmark limit");
    }
}

Options parse_options(int argc, char** argv) {
    Options options;
    std::string modulus_text;
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--help" || arg == "-h") {
            std::cout << "crypto_ntt_bench --log-n K --batch B --word-bits 32|64 --moduli p[,p...]\n"
                         "  [--mode cyclic|negacyclic|coset] [--direction forward|inverse]\n"
                         "  [--coset-generator g] [--mapping-json JSON-or-file-or-array]\n"
                         "  [--warmup K] [--repeat K] [--seed K] [--input-pattern random|boundary]\n"
                         "  [--self-test-host] [--json]\n";
            std::exit(0);
        } else if (arg == "--log-n") {
            options.log_n = static_cast<std::uint32_t>(parse_u64(take_arg(index, argc, argv), "--log-n"));
        } else if (arg == "--batch") {
            options.batch = static_cast<std::size_t>(parse_u64(take_arg(index, argc, argv), "--batch"));
        } else if (arg == "--word-bits") {
            options.word_bits = static_cast<std::uint32_t>(parse_u64(take_arg(index, argc, argv), "--word-bits"));
        } else if (arg == "--moduli") {
            modulus_text = take_arg(index, argc, argv);
        } else if (arg == "--mode") {
            const auto value = take_arg(index, argc, argv);
            if (value == "cyclic") options.mode = Mode::Cyclic;
            else if (value == "negacyclic") options.mode = Mode::Negacyclic;
            else if (value == "coset") options.mode = Mode::Coset;
            else throw std::invalid_argument("unknown --mode: " + value);
        } else if (arg == "--direction") {
            const auto value = take_arg(index, argc, argv);
            if (value == "forward") options.direction = Direction::Forward;
            else if (value == "inverse") options.direction = Direction::Inverse;
            else throw std::invalid_argument("unknown --direction: " + value);
        } else if (arg == "--coset-generator") {
            options.coset_generator = parse_u64(take_arg(index, argc, argv), "--coset-generator");
        } else if (arg == "--mapping-json") {
            options.mapping_json = read_mapping(take_arg(index, argc, argv));
        } else if (arg == "--warmup") {
            options.warmup = static_cast<std::uint32_t>(parse_u64(take_arg(index, argc, argv), "--warmup"));
        } else if (arg == "--repeat") {
            options.repeat = static_cast<std::uint32_t>(parse_u64(take_arg(index, argc, argv), "--repeat"));
        } else if (arg == "--seed") {
            options.seed = parse_u64(take_arg(index, argc, argv), "--seed");
        } else if (arg == "--input-pattern") {
            const auto value = take_arg(index, argc, argv);
            if (value == "random") options.input_pattern = InputPattern::Random;
            else if (value == "boundary") options.input_pattern = InputPattern::Boundary;
            else throw std::invalid_argument("unknown --input-pattern: " + value);
        } else if (arg == "--json") {
            options.json = true;
        } else if (arg == "--self-test-host") {
            options.self_test_host = true;
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    if (modulus_text.empty()) {
        modulus_text = options.word_bits == 32 ? "2013265921" : "576460756061519873";
    }
    options.moduli = parse_moduli(modulus_text);
    validate_options(options);
    return options;
}

std::vector<std::uint64_t> make_input(std::uint32_t log_n, std::size_t batch, std::uint64_t modulus,
                                      std::uint64_t seed, InputPattern pattern, std::size_t channel) {
    const auto n = std::size_t{1} << log_n;
    const auto count = checked_mul(batch, n, "host input size");
    std::vector<std::uint64_t> values(count);
    std::mt19937_64 random(seed ^ (0x9e3779b97f4a7c15ULL * (channel + 1)));
    const std::uint64_t boundary[] = {0, 1 % modulus, modulus - 1, modulus - 2};
    for (std::size_t index = 0; index < count; ++index) {
        values[index] = pattern == InputPattern::Boundary ? boundary[index & 3U] :
                         (index < 4 ? boundary[index] : random() % modulus);
    }
    return values;
}

std::uint64_t root_for_direction(const ChannelInfo& channel, std::uint64_t modulus, Direction direction) {
    return direction == Direction::Forward ? channel.roots.root : mod_pow(channel.roots.root, modulus - 2, modulus);
}

std::vector<std::uint64_t> independent_ntt(const std::vector<std::uint64_t>& values, std::uint64_t modulus,
                                            std::uint64_t root, bool inverse) {
    const auto n = values.size();
    if (n <= 256) {
        std::vector<std::uint64_t> output(n, 0);
        for (std::size_t k = 0; k < n; ++k) {
            std::uint64_t sum = 0;
            for (std::size_t j = 0; j < n; ++j) {
                const auto exponent = static_cast<std::uint64_t>((static_cast<unsigned __int128>(j) * k) % n);
                const auto term = mod_mul(values[j] % modulus, mod_pow(root, exponent, modulus), modulus);
                sum += term;
                if (sum >= modulus) sum -= modulus;
            }
            output[k] = sum;
        }
        if (inverse) {
            const auto inverse_n = mod_pow(static_cast<std::uint64_t>(n), modulus - 2, modulus);
            for (auto& value : output) value = mod_mul(value, inverse_n, modulus);
        }
        return output;
    }

    auto output = values;
    auto reverse_bits = [](std::uint64_t value, std::uint32_t bits) {
        std::uint64_t result = 0;
        for (std::uint32_t bit = 0; bit < bits; ++bit) result = (result << 1U) | ((value >> bit) & 1U);
        return result;
    };
    std::uint32_t log_n = 0;
    for (auto size = n; size > 1; size >>= 1U) ++log_n;
    for (std::size_t index = 0; index < n; ++index) {
        const auto reversed = static_cast<std::size_t>(reverse_bits(index, log_n));
        if (index < reversed) std::swap(output[index], output[reversed]);
    }
    for (std::size_t length = 2; length <= n; length <<= 1U) {
        const auto half = length >> 1U;
        const auto step = mod_pow(root, n / length, modulus);
        for (std::size_t start = 0; start < n; start += length) {
            std::uint64_t omega = 1;
            for (std::size_t offset = 0; offset < half; ++offset) {
                const auto left = start + offset;
                const auto u = output[left];
                const auto v = mod_mul(output[left + half], omega, modulus);
                output[left] = u + v >= modulus ? u + v - modulus : u + v;
                output[left + half] = u >= v ? u - v : modulus + u - v;
                omega = mod_mul(omega, step, modulus);
            }
        }
    }
    if (inverse) {
        const auto inverse_n = mod_pow(static_cast<std::uint64_t>(n), modulus - 2, modulus);
        for (auto& value : output) value = mod_mul(value, inverse_n, modulus);
    }
    return output;
}

std::vector<std::uint64_t> expected_transform(const std::vector<std::uint64_t>& input, const ChannelInfo& channel,
                                              std::uint64_t modulus, std::size_t batch, Direction direction) {
    const auto n = channel.roots.root == 0 ? 0 : channel.twist.empty() ?
        (input.size() / batch) : channel.twist.size();
    std::vector<std::uint64_t> output(input.size());
    for (std::size_t b = 0; b < batch; ++b) {
        std::vector<std::uint64_t> slice(input.begin() + b * n, input.begin() + (b + 1) * n);
        if (direction == Direction::Forward && !channel.twist.empty()) {
            for (std::size_t j = 0; j < n; ++j) slice[j] = mod_mul(slice[j], channel.twist[j], modulus);
        }
        slice = independent_ntt(slice, modulus, root_for_direction(channel, modulus, direction), direction == Direction::Inverse);
        if (direction == Direction::Inverse && !channel.twist.empty()) {
            for (std::size_t j = 0; j < n; ++j) slice[j] = mod_mul(slice[j], channel.twist[j], modulus);
        }
        std::copy(slice.begin(), slice.end(), output.begin() + b * n);
    }
    return output;
}

std::vector<std::uint64_t> library_reference_transform(const std::vector<std::uint64_t>& input, const ChannelInfo& channel,
                                                       std::uint64_t modulus, std::size_t batch, Direction direction) {
    const auto n = channel.twist.empty() ? input.size() / batch : channel.twist.size();
    std::vector<std::uint64_t> output(input.size());
    for (std::size_t b = 0; b < batch; ++b) {
        std::vector<std::uint64_t> slice(input.begin() + b * n, input.begin() + (b + 1) * n);
        if (direction == Direction::Forward && !channel.twist.empty()) {
            for (std::size_t j = 0; j < n; ++j) slice[j] = mod_mul(slice[j], channel.twist[j], modulus);
        }
        cuntt::reference_ntt(slice, modulus, direction == Direction::Inverse);
        if (direction == Direction::Inverse && !channel.twist.empty()) {
            for (std::size_t j = 0; j < n; ++j) slice[j] = mod_mul(slice[j], channel.twist[j], modulus);
        }
        std::copy(slice.begin(), slice.end(), output.begin() + b * n);
    }
    return output;
}

bool equal_values(const std::vector<std::uint64_t>& left, const std::vector<std::uint64_t>& right, std::size_t* mismatch = nullptr) {
    if (left.size() != right.size()) return false;
    const auto result = std::mismatch(left.begin(), left.end(), right.begin());
    if (result.first == left.end()) return true;
    if (mismatch != nullptr) *mismatch = static_cast<std::size_t>(result.first - left.begin());
    return false;
}

template <typename Word>
__device__ Word shoup_multiply(Word value, Word factor, Word factor_shoup, Word modulus);

template <>
__device__ std::uint32_t shoup_multiply(std::uint32_t value, std::uint32_t factor, std::uint32_t factor_shoup,
                                        std::uint32_t modulus) {
    const auto quotient = __umulhi(value, factor_shoup);
    auto reduced = value * factor - quotient * modulus;
    if (reduced >= modulus) reduced -= modulus;
    return reduced;
}

template <>
__device__ std::uint64_t shoup_multiply(std::uint64_t value, std::uint64_t factor, std::uint64_t factor_shoup,
                                        std::uint64_t modulus) {
    // Match the Shoup trait used by cuNTT.  The wrapped low word is valid
    // because the quotient differs from floor(value*factor/p) by at most one.
    const auto quotient = __umul64hi(value, factor_shoup);
    auto reduced = value * factor - quotient * modulus;
    if (reduced >= modulus) reduced -= modulus;
    return reduced;
}

template <typename Word>
__global__ void twist_out_kernel(const Word* source, Word* destination, const Word* factors,
                                 const Word* shoup, Word modulus, std::size_t count, std::size_t n) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto offset = index & (n - 1);  // N is validated to be a power of two.
    if (index < count) destination[index] = shoup_multiply(source[index], factors[offset], shoup[offset], modulus);
}

template <typename Word>
__global__ void twist_inplace_kernel(Word* values, const Word* factors, const Word* shoup,
                                     Word modulus, std::size_t count, std::size_t n) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto offset = index & (n - 1);
    if (index < count) values[index] = shoup_multiply(values[index], factors[offset], shoup[offset], modulus);
}

template <typename Word>
void launch_twist_out(const Word* source, Word* destination, const Word* factors, const Word* shoup,
                      Word modulus, std::size_t count, std::size_t n, cudaStream_t stream, int max_grid) {
    constexpr std::size_t threads = 256;
    const auto grid = (count + threads - 1) / threads;
    if (grid > static_cast<std::size_t>(max_grid)) throw std::invalid_argument("twist launch exceeds device grid limits");
    twist_out_kernel<Word><<<static_cast<unsigned>(grid), threads, 0, stream>>>(source, destination, factors, shoup,
                                                                                  modulus, count, n);
    check_cuda(cudaGetLastError(), "twist_out_kernel launch");
}

template <typename Word>
void launch_twist_inplace(Word* values, const Word* factors, const Word* shoup, Word modulus,
                          std::size_t count, std::size_t n, cudaStream_t stream, int max_grid) {
    constexpr std::size_t threads = 256;
    const auto grid = (count + threads - 1) / threads;
    if (grid > static_cast<std::size_t>(max_grid)) throw std::invalid_argument("twist launch exceeds device grid limits");
    twist_inplace_kernel<Word><<<static_cast<unsigned>(grid), threads, 0, stream>>>(values, factors, shoup,
                                                                                     modulus, count, n);
    check_cuda(cudaGetLastError(), "twist_inplace_kernel launch");
}

template <typename Word>
struct ChannelRun {
    std::unique_ptr<cuntt::Plan> plan;
    DeviceBuffer<Word> source;
    DeviceBuffer<Word> output;
    DeviceBuffer<Word> work;
    DeviceBuffer<Word> factors;
    DeviceBuffer<Word> shoup;
    DeviceBuffer<std::byte> workspace;
    std::vector<std::uint64_t> host_values;
    std::string mapping_json;
    bool mapping_fallback = false;
    double plan_ms = 0.0;
    std::size_t workspace_bytes = 0;
};

template <typename Word>
ChannelRun<Word> prepare_channel(const Options& options, const ChannelInfo& channel, const std::string& mapping_spec,
                                 std::size_t channel_index, cudaStream_t stream) {
    const auto n = std::size_t{1} << options.log_n;
    const auto count = checked_mul(n, options.batch, "device input size");
    const auto host_values = make_input(options.log_n, options.batch, channel.modulus, options.seed, options.input_pattern, channel_index);
    std::vector<Word> input(count);
    for (std::size_t index = 0; index < count; ++index) input[index] = static_cast<Word>(host_values[index]);

    cuntt::PlanConfig config;
    config.log_n = options.log_n;
    config.batch = options.batch;
    config.modulus = channel.modulus;
    config.inverse = options.direction == Direction::Inverse;
    config.backend = cuntt::Backend::SharedIterative;
    config.compute_unit = cuntt::ComputeUnit::Radix2;
    config.modular_multiply = cuntt::ModularMultiply::Shoup;
    config.dataflow_layout = cuntt::DataflowLayout::HermesXor;
    config.dataflow_state_mode = cuntt::DataflowStateMode::InPlace;
    config.stage_partition = default_partition(options.log_n);
    config.threads_per_block = 128;
    config.word_bits = options.word_bits;
    config.input_order = cuntt::InputOrder::Natural;
    config.output_order = cuntt::OutputOrder::Natural;
    config.auto_select = false;
    config.auto_allocate_workspace = false;
    const auto semantic_log_n = config.log_n;
    const auto semantic_batch = config.batch;
    const auto semantic_modulus = config.modulus;
    const auto semantic_inverse = config.inverse;
    const auto semantic_word_bits = config.word_bits;
    const auto semantic_input_order = config.input_order;
    const auto semantic_output_order = config.output_order;
    const bool mapping_fallback = mapping_spec.empty();
    if (!mapping_spec.empty()) cubutterfly::apply_serialized_mapping(mapping_spec, config);
    if (config.log_n != semantic_log_n || config.batch != semantic_batch || config.modulus != semantic_modulus ||
        config.inverse != semantic_inverse || config.word_bits != semantic_word_bits ||
        config.input_order != semantic_input_order || config.output_order != semantic_output_order) {
        throw std::invalid_argument("mapping-json changed the NTT semantic contract");
    }

    const auto plan_start = Clock::now();
    auto plan = std::make_unique<cuntt::Plan>(config);
    plan->set_stream(stream);
    const auto plan_ms = std::chrono::duration<double, std::milli>(Clock::now() - plan_start).count();
    const auto normalized_mapping = cubutterfly::serialize_mapping(plan->config());

    ChannelRun<Word> run;
    run.plan = std::move(plan);
    run.host_values = host_values;
    run.mapping_json = normalized_mapping;
    run.mapping_fallback = mapping_fallback;
    run.plan_ms = plan_ms;
    run.workspace_bytes = run.plan->workspace_size();
    run.source.allocate(count);
    run.output.allocate(count);
    run.work.allocate(channel.twist.empty() || options.direction == Direction::Inverse ? 0 : count);
    run.factors.allocate(channel.twist.size());
    run.shoup.allocate(channel.twist_shoup.size());
    run.workspace.allocate(run.plan->workspace_size());
    if (run.plan->workspace_size() != 0) run.plan->set_workspace(run.workspace.get(), run.workspace.size());
    check_cuda(cudaMemcpyAsync(run.source.get(), input.data(), count * sizeof(Word), cudaMemcpyHostToDevice, stream), "copy NTT input");
    if (!channel.twist.empty()) {
        std::vector<Word> factors_host(channel.twist.size());
        std::vector<Word> shoup_host(channel.twist_shoup.size());
        for (std::size_t index = 0; index < channel.twist.size(); ++index) {
            factors_host[index] = static_cast<Word>(channel.twist[index]);
            shoup_host[index] = static_cast<Word>(shoup_precompute(channel.twist[index], channel.modulus, options.word_bits));
        }
        check_cuda(cudaMemcpyAsync(run.factors.get(), factors_host.data(), factors_host.size() * sizeof(Word), cudaMemcpyHostToDevice, stream), "copy twist factors");
        check_cuda(cudaMemcpyAsync(run.shoup.get(), shoup_host.data(), shoup_host.size() * sizeof(Word), cudaMemcpyHostToDevice, stream), "copy Shoup factors");
    }
    check_cuda(cudaStreamSynchronize(stream), "synchronize setup copies");
    return run;
}

template <typename Word>
void launch_channel(const Options& options, const ChannelInfo& channel, ChannelRun<Word>& run,
                    std::size_t count, std::size_t n, cudaStream_t stream, int max_grid) {
    if (options.direction == Direction::Forward && !channel.twist.empty()) {
        launch_twist_out(run.source.get(), run.work.get(), run.factors.get(), run.shoup.get(), static_cast<Word>(channel.modulus),
                             count, n, stream, max_grid);
        run.plan->execute_async(run.work.get(), run.output.get());
    } else {
        run.plan->execute_async(run.source.get(), run.output.get());
        if (options.direction == Direction::Inverse && !channel.twist.empty()) {
            launch_twist_inplace(run.output.get(), run.factors.get(), run.shoup.get(), static_cast<Word>(channel.modulus),
                                     count, n, stream, max_grid);
        }
    }
    check_cuda(cudaGetLastError(), "NTT launch");
}

template <typename Word>
void verify_channel(const Options& options, const ChannelInfo& channel, ChannelRun<Word>& run,
                    std::size_t channel_index, cudaStream_t stream, std::size_t count) {
    std::vector<Word> output_host(count);
    check_cuda(cudaMemcpyAsync(output_host.data(), run.output.get(), count * sizeof(Word), cudaMemcpyDeviceToHost, stream), "copy NTT output");
    check_cuda(cudaStreamSynchronize(stream), "synchronize NTT output");
    std::vector<std::uint64_t> output_u64(count);
    for (std::size_t index = 0; index < count; ++index) output_u64[index] = static_cast<std::uint64_t>(output_host[index]);
    const auto expected = expected_transform(run.host_values, channel, channel.modulus, options.batch, options.direction);
    std::size_t mismatch = 0;
    if (!equal_values(expected, output_u64, &mismatch)) {
        throw std::runtime_error("GPU exact verification failed in channel " + std::to_string(channel_index) +
                                 " at element " + std::to_string(mismatch));
    }
}

template <typename Word>
struct CompositionResult {
    double plan_ms = 0.0;
    double kernel_ms = 0.0;
    std::size_t workspace_bytes = 0;
    std::size_t working_set_bytes = 0;
    std::size_t allocation_delta_bytes = 0;
    std::vector<Json> records;
};

struct DeviceInfo {
    std::string uuid;
    std::string name;
    std::size_t memory_bytes = 0;
    int max_grid_x = 0;
};

template <typename Word>
CompositionResult<Word> run_composition(const Options& options, const std::vector<ChannelInfo>& channels,
                                        const std::vector<std::string>& mapping_specs, const DeviceInfo& device,
                                        cudaStream_t stream) {
    const auto n = std::size_t{1} << options.log_n;
    const auto count = checked_mul(n, options.batch, "device input size");
    const auto bytes_per_channel = checked_mul(count, sizeof(Word), "device channel bytes");
    std::size_t estimated_working_bytes = 0;
    const auto estimated_workspace = checked_mul(bytes_per_channel, 2, "estimated workspace bytes");
    for (const auto& channel : channels) {
        estimated_working_bytes += 2 * bytes_per_channel + estimated_workspace;
        if (options.direction == Direction::Forward && !channel.twist.empty()) estimated_working_bytes += bytes_per_channel;
        estimated_working_bytes += 2 * channel.twist.size() * sizeof(Word);
    }
    std::size_t free_before = 0;
    std::size_t total_before = 0;
    check_cuda(cudaMemGetInfo(&free_before, &total_before), "cudaMemGetInfo before composition allocation");
    if (estimated_working_bytes > free_before * 9 / 10) {
        throw std::invalid_argument("estimated RNS composition working set exceeds 90% of free GPU memory");
    }
    std::vector<ChannelRun<Word>> runs;
    runs.reserve(channels.size());
    CompositionResult<Word> result;
    for (std::size_t index = 0; index < channels.size(); ++index) {
        runs.push_back(prepare_channel<Word>(options, channels[index], mapping_specs[index], index, stream));
        result.plan_ms += runs.back().plan_ms;
        result.workspace_bytes += runs.back().workspace_bytes;
    }
    // All channel plans and buffers are live before warmup.  Thus every timed
    // iteration represents one complete [channel][batch][N] composition.
    std::size_t working_bytes = 0;
    for (std::size_t index = 0; index < runs.size(); ++index) {
        working_bytes += 2 * bytes_per_channel;
        if (options.direction == Direction::Forward && !channels[index].twist.empty()) working_bytes += bytes_per_channel;
        working_bytes += 2 * channels[index].twist.size() * sizeof(Word) + runs[index].workspace_bytes;
    }
    std::size_t free_after = 0;
    std::size_t total_after = 0;
    check_cuda(cudaMemGetInfo(&free_after, &total_after), "cudaMemGetInfo after composition allocation");
    const auto allocation_delta = free_before > free_after ? free_before - free_after : 0;
    if (allocation_delta > free_before * 9 / 10) {
        throw std::invalid_argument("RNS composition allocation consumes more than 90% of free GPU memory");
    }
    result.working_set_bytes = working_bytes;
    result.allocation_delta_bytes = allocation_delta;
    for (std::uint32_t warmup = 0; warmup < options.warmup; ++warmup) {
        for (std::size_t index = 0; index < runs.size(); ++index)
            launch_channel(options, channels[index], runs[index], count, n, stream, device.max_grid_x);
    }
    check_cuda(cudaStreamSynchronize(stream), "synchronize composition warmup");
    Event start;
    Event stop;
    check_cuda(cudaEventRecord(start.get(), stream), "record composition start event");
    for (std::uint32_t repeat = 0; repeat < options.repeat; ++repeat) {
        for (std::size_t index = 0; index < runs.size(); ++index)
            launch_channel(options, channels[index], runs[index], count, n, stream, device.max_grid_x);
    }
    check_cuda(cudaEventRecord(stop.get(), stream), "record composition stop event");
    check_cuda(cudaEventSynchronize(stop.get()), "synchronize composition stop event");
    float elapsed_ms = 0.0F;
    check_cuda(cudaEventElapsedTime(&elapsed_ms, start.get(), stop.get()), "elapsed composition event");
    result.kernel_ms = static_cast<double>(elapsed_ms) / options.repeat;
    for (std::size_t index = 0; index < runs.size(); ++index) {
        verify_channel(options, channels[index], runs[index], index, stream, count);
        result.records.push_back(Json{{"channel", index}, {"modulus", channels[index].modulus},
                                      {"modulus_bits", 64U - static_cast<unsigned>(__builtin_clzll(channels[index].modulus))},
                                      {"mapping_json", runs[index].mapping_json}, {"mapping_fallback", runs[index].mapping_fallback},
                                      {"mapping_provenance", runs[index].mapping_fallback ? "shared-iterative-default" : "provided-mapping"},
                                      {"root", channels[index].roots.root},
                                      {"default_root_2n", channels[index].roots.default_root_2n},
                                      {"default_root_2n_squared", channels[index].roots.default_root_2n_squared},
                                      {"psi", channels[index].roots.psi}, {"root_match", channels[index].roots.root_match},
                                      {"workspace_bytes", runs[index].workspace_bytes}});
    }
    return result;
}

DeviceInfo device_info() {
    int device = 0;
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
    std::ostringstream uuid;
    uuid << "GPU-" << std::hex << std::setfill('0');
    for (int index = 0; index < 16; ++index) {
        if (index == 4 || index == 6 || index == 8 || index == 10) uuid << '-';
        uuid << std::setw(2) << (static_cast<unsigned>(static_cast<unsigned char>(properties.uuid.bytes[index])));
    }
    return {uuid.str(), properties.name, properties.totalGlobalMem, properties.maxGridSize[0]};
}

void run_host_self_test(const Options& options, const std::vector<ChannelInfo>& channels) {
    const auto n = std::size_t{1} << options.log_n;
    const auto per_channel = checked_mul(options.batch, n, "host self-test size");
    const auto total = checked_mul(per_channel, channels.size(), "host self-test channel size");
    if (checked_mul(total, sizeof(std::uint64_t), "host self-test bytes") > (std::size_t{1} << 34)) {
        throw std::invalid_argument("host self-test input exceeds the 16 GiB safety limit");
    }
    for (std::size_t channel_index = 0; channel_index < channels.size(); ++channel_index) {
        const auto& channel = channels[channel_index];
        const auto input = make_input(options.log_n, options.batch, channel.modulus, options.seed, options.input_pattern, channel_index);
        const auto expected = expected_transform(input, channel, channel.modulus, options.batch, options.direction);
        const auto reference = library_reference_transform(input, channel, channel.modulus, options.batch, options.direction);
        std::size_t mismatch = 0;
        if (!equal_values(expected, reference, &mismatch)) {
            throw std::runtime_error("host independent/reference verification failed in channel " +
                                     std::to_string(channel_index) + " at element " + std::to_string(mismatch));
        }
    }
}

Json output_base(const Options& options, const DeviceInfo* device, bool correct, double plan_ms, double kernel_ms,
                 std::size_t workspace_bytes, std::size_t working_set_bytes, std::size_t allocation_delta_bytes,
                 const std::vector<Json>& channel_records) {
    Json output;
    output["logN"] = options.log_n;
    output["batch"] = options.batch;
    output["word_bits"] = options.word_bits;
    output["mode"] = mode_name(options.mode);
    output["direction"] = direction_name(options.direction);
    output["moduli"] = options.moduli;
    output["gpu_uuid"] = device == nullptr ? "host" : device->uuid;
    output["device_name"] = device == nullptr ? "host" : device->name;
    output["device_memory_bytes"] = device == nullptr ? 0 : device->memory_bytes;
    output["compile_mode"] = std::getenv("CUBUTTERFLY_COMPILE_MODE") == nullptr ? "default" : std::getenv("CUBUTTERFLY_COMPILE_MODE");
    output["correct"] = correct;
    output["verified_batches"] = options.batch;
    output["verified_channels"] = options.moduli.size();
    output["warmup"] = options.warmup;
    output["repeat"] = options.repeat;
    output["kernel_ms"] = kernel_ms;
    output["plan_ms"] = plan_ms;
    output["workspace_bytes"] = workspace_bytes;
    output["working_set_bytes"] = working_set_bytes;
    output["allocation_delta_bytes"] = allocation_delta_bytes;
    output["transforms_per_second"] = kernel_ms > 0 ? (1000.0 * options.batch * options.moduli.size() / kernel_ms) : 0.0;
    output["ring_batches_per_second"] = kernel_ms > 0 ? (1000.0 * options.batch / kernel_ms) : 0.0;
    output["coset_generator"] = options.mode == Mode::Coset ? options.coset_generator : 0;
    output["input_pattern"] = pattern_name(options.input_pattern);
    output["seed"] = options.seed;
    output["normalization"] = options.direction == Direction::Inverse ? "inverse" : "none";
    output["layout"] = "channel-batch-coefficient";
    output["rns_stream_order"] = "single-stream";
    if (options.mapping_json.empty()) output["mapping_request"] = nullptr;
    else {
        try { output["mapping_request"] = Json::parse(options.mapping_json); }
        catch (...) { output["mapping_request"] = options.mapping_json; }
    }
    output["channels"] = channel_records;
    output["composition"] = "sequential single-word plans; no cross-modulus fusion";
    return output;
}

int execute(const Options& options) {
    const auto mapping_specs = mapping_for_channels(options.mapping_json, options.moduli.size());
    std::vector<ChannelInfo> channels;
    channels.reserve(options.moduli.size());
    for (const auto modulus : options.moduli) channels.push_back(make_channel(options.log_n, modulus, options.mode,
                                                                                options.coset_generator, options.direction));
    if (options.self_test_host) {
        run_host_self_test(options, channels);
        const auto result = output_base(options, nullptr, true, 0.0, 0.0, 0, 0, 0, {});
        std::cout << result.dump() << '\n';
        return 0;
    }

    const auto device = device_info();
    Stream stream;
    double plan_ms = 0.0;
    double kernel_ms = 0.0;
    std::size_t workspace_bytes = 0;
    std::size_t working_set_bytes = 0;
    std::size_t allocation_delta_bytes = 0;
    std::vector<Json> records;
    if (options.word_bits == 32) {
        auto result = run_composition<std::uint32_t>(options, channels, mapping_specs, device, stream.get());
        plan_ms = result.plan_ms;
        kernel_ms = result.kernel_ms;
        workspace_bytes = result.workspace_bytes;
        working_set_bytes = result.working_set_bytes;
        allocation_delta_bytes = result.allocation_delta_bytes;
        records = std::move(result.records);
    } else {
        auto result = run_composition<std::uint64_t>(options, channels, mapping_specs, device, stream.get());
        plan_ms = result.plan_ms;
        kernel_ms = result.kernel_ms;
        workspace_bytes = result.workspace_bytes;
        working_set_bytes = result.working_set_bytes;
        allocation_delta_bytes = result.allocation_delta_bytes;
        records = std::move(result.records);
    }
    const auto result = output_base(options, &device, true, plan_ms, kernel_ms, workspace_bytes,
                                    working_set_bytes, allocation_delta_bytes, records);
    std::cout << result.dump() << '\n';
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_options(argc, argv);
        return execute(options);
    } catch (const std::exception& error) {
        std::cerr << "crypto_ntt_bench: " << error.what() << '\n';
        if (argc > 1) {
            for (int index = 1; index < argc; ++index) {
                if (std::string(argv[index]) == "--json") {
                    std::cout << Json{{"correct", false}, {"error", error.what()}}.dump() << '\n';
                    break;
                }
            }
        }
        return 2;
    }
}
