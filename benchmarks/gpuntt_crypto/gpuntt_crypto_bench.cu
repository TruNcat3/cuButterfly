// Standalone natural-order crypto NTT adapter for the pinned GPU-NTT Merge
// implementation. The external library is intentionally composed one
// modulus at a time; no CRT/RNS fusion is claimed here.

#include <cuda_runtime.h>

#include <cuntt/ntt.hpp>
#include <gpuntt/ntt_merge/ntt.cuh>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
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
using U64 = std::uint64_t;

#ifndef GPUNTT_SOURCE_REVISION
#define GPUNTT_SOURCE_REVISION "unknown"
#endif

constexpr U64 kMaxWord32Modulus = U64{1} << 30;
constexpr U64 kMaxWord64Modulus = U64{1} << 62;
constexpr std::size_t kMaxBatch = 65535;
constexpr std::uint32_t kMinLogN = 1;
constexpr std::uint32_t kMaxLogN = 28;

class Unsupported : public std::runtime_error {
  public:
    explicit Unsupported(const std::string& message) : std::runtime_error("unsupported: " + message) {}
};

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
    ~Stream() {
        if (stream_ != nullptr) cudaStreamDestroy(stream_);
    }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    cudaStream_t get() const noexcept { return stream_; }

  private:
    cudaStream_t stream_ = nullptr;
};

class Event {
  public:
    Event() { check_cuda(cudaEventCreateWithFlags(&event_, cudaEventDefault), "cudaEventCreate"); }
    ~Event() {
        if (event_ != nullptr) cudaEventDestroy(event_);
    }
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
    std::vector<U64> moduli;
    Mode mode = Mode::Cyclic;
    Direction direction = Direction::Forward;
    U64 coset_generator = 7;
    std::uint32_t warmup = 5;
    std::uint32_t repeat = 20;
    U64 seed = 1;
    InputPattern input_pattern = InputPattern::Random;
    bool json = false;
};

struct ChannelInfo {
    U64 modulus = 0;
    U64 root = 0;
    U64 default_root_2n = 0;
    U64 default_root_2n_squared = 0;
    U64 psi = 0;
    bool root_match = true;
    bool native_x_n_plus = false;
    std::vector<U64> oracle_twist;
    std::vector<U64> timed_twist;
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

U64 parse_u64(const std::string& text, const char* name) {
    if (text.empty() || text.front() == '-') throw std::invalid_argument(std::string("invalid ") + name);
    std::size_t used = 0;
    const auto value = std::stoull(text, &used, 0);
    if (used != text.size()) throw std::invalid_argument(std::string("invalid ") + name + ": " + text);
    return static_cast<U64>(value);
}

std::string take_arg(int& index, int argc, char** argv) {
    if (index + 1 >= argc) throw std::invalid_argument(std::string("missing value for ") + argv[index]);
    return argv[++index];
}

std::vector<U64> parse_moduli(const std::string& text) {
    std::vector<U64> values;
    std::istringstream stream(text);
    for (std::string token; std::getline(stream, token, ',');) {
        if (token.empty()) throw std::invalid_argument("empty modulus in --moduli");
        values.push_back(parse_u64(token, "modulus"));
    }
    if (values.empty()) throw std::invalid_argument("--moduli must not be empty");
    return values;
}

std::size_t checked_mul(std::size_t left, std::size_t right, const char* context) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::overflow_error(std::string(context) + " overflows size_t");
    }
    return left * right;
}

std::size_t checked_add(std::size_t left, std::size_t right, const char* context) {
    if (right > std::numeric_limits<std::size_t>::max() - left) {
        throw std::overflow_error(std::string(context) + " overflows size_t");
    }
    return left + right;
}

U64 mod_mul(U64 left, U64 right, U64 modulus) {
    return static_cast<U64>((static_cast<unsigned __int128>(left) * right) % modulus);
}

U64 mod_pow(U64 base, U64 exponent, U64 modulus) {
    U64 result = 1 % modulus;
    base %= modulus;
    while (exponent != 0) {
        if (exponent & 1U) result = mod_mul(result, base, modulus);
        base = mod_mul(base, base, modulus);
        exponent >>= 1U;
    }
    return result;
}

U64 shoup_precompute(U64 value, U64 modulus, std::uint32_t word_bits) {
    return static_cast<U64>((static_cast<unsigned __int128>(value) << word_bits) / modulus);
}

std::vector<U64> powers(U64 base, std::size_t count, U64 modulus) {
    std::vector<U64> result(count, 1);
    base %= modulus;
    for (std::size_t index = 1; index < count; ++index) result[index] = mod_mul(result[index - 1], base, modulus);
    return result;
}

std::uint32_t modulus_bits(U64 modulus) {
    std::uint32_t bits = 0;
    while (modulus != 0) {
        ++bits;
        modulus >>= 1U;
    }
    return bits;
}

__host__ __device__ U64 reverse_bits(U64 value, std::uint32_t bits) {
    U64 result = 0;
    for (std::uint32_t bit = 0; bit < bits; ++bit) {
        result = (result << 1U) | ((value >> bit) & 1U);
    }
    return result;
}

U64 find_psi_for_root(std::uint32_t log_n, U64 modulus, U64 root,
                      U64& default_psi, U64& default_squared) {
    if (log_n >= 62) throw Unsupported("negacyclic 2N root order overflows uint64_t");
    default_psi = cuntt::find_primitive_power_of_two_root(log_n + 1, modulus);
    default_squared = mod_mul(default_psi, default_psi, modulus);
    if (default_squared == root) return default_psi;

    const U64 order = U64{1} << (log_n + 1);
    if ((modulus - 1) % order != 0) throw Unsupported("modulus does not support a 2N negacyclic root");
    for (U64 candidate = 2; candidate < (U64{1} << 20); ++candidate) {
        const U64 psi = mod_pow(candidate, (modulus - 1) / order, modulus);
        if (mod_pow(psi, order, modulus) == 1 &&
            mod_pow(psi, order >> 1U, modulus) != 1 &&
            mod_mul(psi, psi, modulus) == root) {
            return psi;
        }
    }
    throw Unsupported("could not select a 2N root matching cuNTT's deterministic N root");
}

ChannelInfo make_channel(const Options& options, U64 modulus) {
    const std::size_t n = std::size_t{1} << options.log_n;
    ChannelInfo channel;
    channel.modulus = modulus;
    channel.root = cuntt::find_primitive_power_of_two_root(options.log_n, modulus);

    if (options.mode == Mode::Negacyclic) {
        if ((modulus - 1) % (U64{2} * n) != 0) {
            throw Unsupported("negacyclic mode requires 2N to divide modulus-1");
        }
        channel.psi = find_psi_for_root(options.log_n, modulus, channel.root,
                                        channel.default_root_2n, channel.default_root_2n_squared);
        channel.root_match = mod_mul(channel.psi, channel.psi, modulus) == channel.root;
        if (!channel.root_match) throw std::logic_error("selected negacyclic root does not match cuNTT root");
        channel.native_x_n_plus = true;
        channel.oracle_twist = powers(options.direction == Direction::Forward ? channel.psi
                                                                         : mod_pow(channel.psi, modulus - 2, modulus),
                                      n, modulus);
    } else if (options.mode == Mode::Coset) {
        const U64 generator = options.coset_generator % modulus;
        if (generator == 0) throw std::invalid_argument("coset generator must be nonzero modulo the field");
        const U64 base = options.direction == Direction::Forward ? generator : mod_pow(generator, modulus - 2, modulus);
        channel.oracle_twist = powers(base, n, modulus);
    }

    // Native X_N_plus already incorporates the negacyclic twist. Coset and
    // cyclic paths use the explicit wrapper, which remains inside timing.
    channel.timed_twist = channel.native_x_n_plus ? std::vector<U64>{} : channel.oracle_twist;
    return channel;
}

void validate_options(const Options& options) {
    if (options.log_n < kMinLogN || options.log_n > kMaxLogN) {
        throw Unsupported("--log-n must be in [1,28] for GPU-NTT Merge");
    }
    if (options.batch == 0 || options.batch > kMaxBatch) {
        throw Unsupported("--batch must be in [1,65535] because GPU-NTT maps batch to grid.z");
    }
    if (options.word_bits != 32 && options.word_bits != 64) {
        throw Unsupported("--word-bits must be 32 or 64");
    }
    if (options.warmup > 100000 || options.repeat == 0 || options.repeat > 100000) {
        throw std::invalid_argument("warmup/repeat are outside the supported finite range");
    }
    if (options.moduli.empty()) throw std::invalid_argument("at least one modulus is required");

    const U64 max_modulus = options.word_bits == 32 ? kMaxWord32Modulus : kMaxWord64Modulus;
    const U64 n = U64{1} << options.log_n;
    for (std::size_t index = 0; index < options.moduli.size(); ++index) {
        const U64 modulus = options.moduli[index];
        for (std::size_t previous = 0; previous < index; ++previous) {
            if (options.moduli[previous] == modulus) {
                throw std::invalid_argument("RNS modulus channels must be pairwise distinct");
            }
        }
        if (modulus <= 2 || modulus >= max_modulus) {
            std::ostringstream message;
            message << "modulus " << modulus << " is outside the supported "
                    << (options.word_bits == 32 ? "prime > 2 and < 2^30" : "prime > 2 and < 2^62")
                    << " arithmetic range";
            throw Unsupported(message.str());
        }
        if (!cuntt::is_prime(modulus)) throw std::invalid_argument("every modulus must be prime");
        if ((modulus - 1) % n != 0) throw std::invalid_argument("modulus-1 must be divisible by N");
        if (options.mode == Mode::Negacyclic && (modulus - 1) % (U64{2} * n) != 0) {
            throw std::invalid_argument("negacyclic mode requires modulus-1 divisible by 2N");
        }
    }

    const auto points = checked_mul(checked_mul(std::size_t{1} << options.log_n, options.batch, "input point count"),
                                   options.moduli.size(), "RNS point count");
    const auto input_bytes = checked_mul(points, options.word_bits / 8, "RNS input bytes");
    if (input_bytes > (std::size_t{2} << 30)) {
        throw Unsupported("RNS input envelope exceeds the 2 GiB benchmark limit");
    }
}

Options parse_options(int argc, char** argv) {
    Options options;
    std::string modulus_text;
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--help" || arg == "-h") {
            std::cout << "gpuntt_crypto_bench --log-n K --batch B --word-bits 32|64 --moduli p[,p...]\n"
                         "  [--mode cyclic|negacyclic|coset] [--direction forward|inverse]\n"
                         "  [--coset-generator g] [--warmup K] [--repeat K] [--seed K]\n"
                         "  [--input-pattern random|boundary] [--json]\n";
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
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    if (modulus_text.empty()) modulus_text = options.word_bits == 32 ? "469762049" : "576460756061519873";
    options.moduli = parse_moduli(modulus_text);
    validate_options(options);
    return options;
}

std::vector<U64> make_input(std::uint32_t log_n, std::size_t batch, U64 modulus,
                            U64 seed, InputPattern pattern, std::size_t channel) {
    const std::size_t n = std::size_t{1} << log_n;
    const std::size_t count = checked_mul(batch, n, "host input size");
    std::vector<U64> values(count);
    std::mt19937_64 random(seed ^ (U64{0x9e3779b97f4a7c15ULL} * (channel + 1)));
    const U64 boundary[] = {0, 1 % modulus, modulus - 1, modulus - 2};
    for (std::size_t index = 0; index < count; ++index) {
        values[index] = pattern == InputPattern::Boundary ? boundary[index & 3U]
                     : (index < 4 ? boundary[index] : random() % modulus);
    }
    return values;
}

U64 root_for_direction(const ChannelInfo& channel, Direction direction) {
    return direction == Direction::Forward ? channel.root : mod_pow(channel.root, channel.modulus - 2, channel.modulus);
}

std::vector<U64> independent_ntt(const std::vector<U64>& values, U64 modulus, U64 root, bool inverse) {
    const std::size_t n = values.size();
    auto output = values;
    const std::uint32_t log_n = [&]() {
        std::uint32_t result = 0;
        for (std::size_t size = n; size > 1; size >>= 1U) ++result;
        return result;
    }();
    for (std::size_t index = 0; index < n; ++index) {
        const auto reversed = static_cast<std::size_t>(reverse_bits(index, log_n));
        if (index < reversed) std::swap(output[index], output[reversed]);
    }
    for (std::size_t length = 2; length <= n; length <<= 1U) {
        const std::size_t half = length >> 1U;
        const U64 step = mod_pow(root, n / length, modulus);
        for (std::size_t start = 0; start < n; start += length) {
            U64 omega = 1;
            for (std::size_t offset = 0; offset < half; ++offset) {
                const auto left = start + offset;
                const U64 u = output[left];
                const U64 v = mod_mul(output[left + half], omega, modulus);
                output[left] = u + v >= modulus ? u + v - modulus : u + v;
                output[left + half] = u >= v ? u - v : modulus + u - v;
                omega = mod_mul(omega, step, modulus);
            }
        }
    }
    if (inverse) {
        const U64 inverse_n = mod_pow(static_cast<U64>(n), modulus - 2, modulus);
        for (auto& value : output) value = mod_mul(value, inverse_n, modulus);
    }
    return output;
}

std::vector<U64> expected_transform(const std::vector<U64>& input, const ChannelInfo& channel,
                                    std::size_t batch, Direction direction) {
    const std::size_t n = input.size() / batch;
    std::vector<U64> output(input.size());
    for (std::size_t b = 0; b < batch; ++b) {
        std::vector<U64> slice(input.begin() + b * n, input.begin() + (b + 1) * n);
        if (direction == Direction::Forward && !channel.oracle_twist.empty()) {
            for (std::size_t index = 0; index < n; ++index) {
                slice[index] = mod_mul(slice[index], channel.oracle_twist[index], channel.modulus);
            }
        }
        slice = independent_ntt(slice, channel.modulus, root_for_direction(channel, direction), direction == Direction::Inverse);
        if (direction == Direction::Inverse && !channel.oracle_twist.empty()) {
            for (std::size_t index = 0; index < n; ++index) {
                slice[index] = mod_mul(slice[index], channel.oracle_twist[index], channel.modulus);
            }
        }
        std::copy(slice.begin(), slice.end(), output.begin() + b * n);
    }
    return output;
}

template <typename Word>
__device__ Word shoup_multiply(Word value, Word factor, Word factor_shoup, Word modulus);

template <>
__device__ std::uint32_t shoup_multiply(std::uint32_t value, std::uint32_t factor,
                                        std::uint32_t factor_shoup, std::uint32_t modulus) {
    const std::uint64_t product = static_cast<std::uint64_t>(value) * factor;
    const std::uint32_t quotient = __umulhi(value, factor_shoup);
    auto reduced = static_cast<std::uint32_t>(product - quotient * modulus);
    if (reduced >= modulus) reduced -= modulus;
    return reduced;
}

template <>
__device__ std::uint64_t shoup_multiply(std::uint64_t value, std::uint64_t factor,
                                        std::uint64_t factor_shoup, std::uint64_t modulus) {
    const auto quotient = __umul64hi(value, factor_shoup);
    auto reduced = value * factor - quotient * modulus;
    if (reduced >= modulus) reduced -= modulus;
    return reduced;
}

template <typename Word>
__global__ void twist_kernel(const Word* source, Word* destination, const Word* factors,
                             const Word* shoup, Word modulus, std::size_t count, std::size_t n) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        destination[index] = shoup_multiply(source[index], factors[index & (n - 1)],
                                            shoup[index & (n - 1)], modulus);
    }
}

template <typename Word>
__global__ void twist_inplace_kernel(Word* values, const Word* factors, const Word* shoup,
                                     Word modulus, std::size_t count, std::size_t n) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        values[index] = shoup_multiply(values[index], factors[index & (n - 1)],
                                       shoup[index & (n - 1)], modulus);
    }
}

template <typename Word>
__global__ void bit_reverse_copy_kernel(const Word* source, Word* destination,
                                        std::size_t n, std::uint32_t log_n,
                                        std::size_t count, bool source_is_bit_reversed) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const std::size_t batch = index >> log_n;
    const std::size_t coefficient = index & (n - 1);
    const std::size_t reversed = static_cast<std::size_t>(
        __brev(static_cast<unsigned>(coefficient)) >> (32U - log_n));
    if (source_is_bit_reversed) {
        destination[batch * n + coefficient] = source[batch * n + reversed];
    } else {
        destination[batch * n + reversed] = source[index];
    }
}

template <typename Word>
void launch_twist(const Word* source, Word* destination, const Word* factors, const Word* shoup,
                  Word modulus, std::size_t count, std::size_t n, cudaStream_t stream, int max_grid) {
    constexpr std::size_t threads = 256;
    const auto grid = (count + threads - 1) / threads;
    if (grid > static_cast<std::size_t>(max_grid)) throw Unsupported("twist launch exceeds device grid limits");
    twist_kernel<Word><<<static_cast<unsigned>(grid), threads, 0, stream>>>(source, destination, factors, shoup,
                                                                              modulus, count, n);
    check_cuda(cudaGetLastError(), "twist kernel launch");
}

template <typename Word>
void launch_twist_inplace(Word* values, const Word* factors, const Word* shoup, Word modulus,
                          std::size_t count, std::size_t n, cudaStream_t stream, int max_grid) {
    constexpr std::size_t threads = 256;
    const auto grid = (count + threads - 1) / threads;
    if (grid > static_cast<std::size_t>(max_grid)) throw Unsupported("twist launch exceeds device grid limits");
    twist_inplace_kernel<Word><<<static_cast<unsigned>(grid), threads, 0, stream>>>(values, factors, shoup,
                                                                                     modulus, count, n);
    check_cuda(cudaGetLastError(), "twist in-place kernel launch");
}

template <typename Word>
void launch_bit_reverse(const Word* source, Word* destination, std::size_t count, std::size_t n,
                        std::uint32_t log_n, bool source_is_bit_reversed,
                        cudaStream_t stream, int max_grid) {
    constexpr std::size_t threads = 256;
    const auto grid = (count + threads - 1) / threads;
    if (grid > static_cast<std::size_t>(max_grid)) throw Unsupported("bit-reversal launch exceeds device grid limits");
    bit_reverse_copy_kernel<Word><<<static_cast<unsigned>(grid), threads, 0, stream>>>(
        source, destination, n, log_n, count, source_is_bit_reversed);
    check_cuda(cudaGetLastError(), "bit-reversal kernel launch");
}

template <typename Word>
std::vector<Word> make_gpu_root_table(U64 root, std::size_t table_size, std::uint32_t table_log,
                                      U64 modulus) {
    const auto host_powers = powers(root, table_size, modulus);
    std::vector<Word> table(table_size);
    for (std::size_t index = 0; index < table_size; ++index) {
        table[index] = static_cast<Word>(host_powers[reverse_bits(index, table_log)]);
    }
    return table;
}

template <typename Word>
struct ChannelRun {
    ChannelInfo channel;
    std::vector<U64> host_values;
    std::size_t count = 0;
    std::size_t n = 0;
    std::size_t root_count = 0;
    DeviceBuffer<Word> source;
    DeviceBuffer<Word> work;
    DeviceBuffer<Word> output;
    DeviceBuffer<Word> factors;
    DeviceBuffer<Word> shoup;
    DeviceBuffer<Root<Word>> roots;
    DeviceBuffer<Modulus<Word>> modulus;
    DeviceBuffer<Ninverse<Word>> n_inverse;
    Modulus<Word> modulus_host;
    Ninverse<Word> n_inverse_host = 0;
    std::size_t device_bytes = 0;

    Word* natural_output() noexcept {
        return channel_direction_forward() ? work.get() : output.get();
    }
    const Word* natural_output() const noexcept {
        return channel_direction_forward() ? work.get() : output.get();
    }
    bool direction_forward = true;
    bool channel_direction_forward() const noexcept { return direction_forward; }
};

template <typename Word>
ChannelRun<Word> prepare_channel(const Options& options, const ChannelInfo& channel,
                                 std::size_t channel_index, cudaStream_t stream) {
    ChannelRun<Word> run;
    run.channel = channel;
    run.direction_forward = options.direction == Direction::Forward;
    run.n = std::size_t{1} << options.log_n;
    run.count = checked_mul(run.n, options.batch, "device input size");
    run.host_values = make_input(options.log_n, options.batch, channel.modulus,
                                 options.seed, options.input_pattern, channel_index);
    std::vector<Word> input(run.count);
    for (std::size_t index = 0; index < run.count; ++index) input[index] = static_cast<Word>(run.host_values[index]);

    const U64 transform_root = channel.native_x_n_plus ? channel.psi : channel.root;
    const U64 inverse_root = mod_pow(transform_root, channel.modulus - 2, channel.modulus);
    const U64 selected_root = options.direction == Direction::Forward ? transform_root : inverse_root;
    run.root_count = channel.native_x_n_plus ? run.n : run.n / 2;
    const std::uint32_t table_log = channel.native_x_n_plus ? options.log_n : options.log_n - 1;
    const auto root_host = make_gpu_root_table<Word>(selected_root, run.root_count, table_log, channel.modulus);

    run.modulus_host = Modulus<Word>(static_cast<Word>(channel.modulus));
    run.n_inverse_host = static_cast<Ninverse<Word>>(mod_pow(static_cast<U64>(run.n), channel.modulus - 2,
                                                             channel.modulus));
    run.source.allocate(run.count);
    run.work.allocate(run.count);
    run.output.allocate(run.count);
    run.factors.allocate(channel.timed_twist.size());
    run.shoup.allocate(channel.timed_twist.size());
    run.roots.allocate(run.root_count);
    run.modulus.allocate(1);
    run.n_inverse.allocate(1);

    std::vector<Word> factors_host(channel.timed_twist.size());
    std::vector<Word> shoup_host(channel.timed_twist.size());
    for (std::size_t index = 0; index < channel.timed_twist.size(); ++index) {
        factors_host[index] = static_cast<Word>(channel.timed_twist[index]);
        shoup_host[index] = static_cast<Word>(shoup_precompute(channel.timed_twist[index], channel.modulus,
                                                               options.word_bits));
    }
    check_cuda(cudaMemcpyAsync(run.source.get(), input.data(), run.count * sizeof(Word), cudaMemcpyHostToDevice, stream),
               "copy NTT input");
    check_cuda(cudaMemcpyAsync(run.roots.get(), root_host.data(), run.root_count * sizeof(Word), cudaMemcpyHostToDevice, stream),
               "copy GPU-NTT roots");
    check_cuda(cudaMemcpyAsync(run.modulus.get(), &run.modulus_host, sizeof(run.modulus_host), cudaMemcpyHostToDevice, stream),
               "copy GPU-NTT modulus");
    check_cuda(cudaMemcpyAsync(run.n_inverse.get(), &run.n_inverse_host, sizeof(run.n_inverse_host), cudaMemcpyHostToDevice, stream),
               "copy GPU-NTT normalization");
    if (!factors_host.empty()) {
        check_cuda(cudaMemcpyAsync(run.factors.get(), factors_host.data(), factors_host.size() * sizeof(Word),
                                   cudaMemcpyHostToDevice, stream), "copy twist factors");
        check_cuda(cudaMemcpyAsync(run.shoup.get(), shoup_host.data(), shoup_host.size() * sizeof(Word),
                                   cudaMemcpyHostToDevice, stream), "copy Shoup factors");
    }
    check_cuda(cudaStreamSynchronize(stream), "synchronize GPU-NTT setup copies");

    run.device_bytes = checked_add(checked_mul(run.count, sizeof(Word), "channel input bytes"),
                                   checked_add(checked_mul(run.count, sizeof(Word), "channel work bytes"),
                                               checked_mul(run.count, sizeof(Word), "channel output bytes"),
                                               "channel data bytes"), "channel data bytes");
    run.device_bytes = checked_add(run.device_bytes, checked_mul(run.root_count, sizeof(Word), "root table bytes"),
                                   "channel allocation bytes");
    run.device_bytes = checked_add(run.device_bytes, checked_mul(channel.timed_twist.size(), sizeof(Word) * 2,
                                                                  "twist table bytes"), "channel allocation bytes");
    run.device_bytes = checked_add(run.device_bytes, sizeof(run.modulus_host) + sizeof(run.n_inverse_host),
                                   "channel allocation bytes");
    return run;
}

template <typename Word>
void launch_channel(const Options& options, ChannelRun<Word>& run, cudaStream_t stream, int max_grid) {
    gpuntt::ntt_rns_configuration<Word> config{};
    config.n_power = static_cast<int>(options.log_n);
    config.ntt_type = options.direction == Direction::Forward ? gpuntt::FORWARD : gpuntt::INVERSE;
    config.ntt_layout = gpuntt::PerPolynomial;
    config.reduction_poly = run.channel.native_x_n_plus ? gpuntt::ReductionPolynomial::X_N_plus
                                                        : gpuntt::ReductionPolynomial::X_N_minus;
    config.zero_padding = false;
    config.mod_inverse = run.n_inverse.get();
    config.stream = stream;

    if (options.direction == Direction::Forward) {
        Word* ntt_input = run.source.get();
        if (!run.channel.timed_twist.empty()) {
            launch_twist(run.source.get(), run.work.get(), run.factors.get(), run.shoup.get(),
                         static_cast<Word>(run.channel.modulus), run.count, run.n, stream, max_grid);
            ntt_input = run.work.get();
        }
        gpuntt::GPU_NTT<Word>(ntt_input, run.output.get(), run.roots.get(), run.modulus.get(),
                              config, static_cast<int>(options.batch), 1);
        check_cuda(cudaGetLastError(), "GPU-NTT forward launch");
        // GPU-NTT's Merge forward result is bit-reversed. Naturalization is
        // part of the measured external baseline contract.
        launch_bit_reverse(run.output.get(), run.work.get(), run.count, run.n, options.log_n,
                           true, stream, max_grid);
    } else {
        // GPU-NTT's inverse accepts bit-reversed input and emits natural
        // coefficients (including its required N inverse normalization).
        launch_bit_reverse(run.source.get(), run.work.get(), run.count, run.n, options.log_n,
                           false, stream, max_grid);
        gpuntt::GPU_INTT<Word>(run.work.get(), run.output.get(), run.roots.get(), run.modulus.get(),
                               config, static_cast<int>(options.batch), 1);
        check_cuda(cudaGetLastError(), "GPU-NTT inverse launch");
        if (!run.channel.timed_twist.empty()) {
            launch_twist_inplace(run.output.get(), run.factors.get(), run.shoup.get(),
                                 static_cast<Word>(run.channel.modulus), run.count, run.n, stream, max_grid);
        }
    }
}

template <typename Word>
void verify_channel(const Options& options, const ChannelRun<Word>& run,
                    std::size_t channel_index, cudaStream_t stream) {
    std::vector<Word> output_host(run.count);
    check_cuda(cudaMemcpyAsync(output_host.data(), run.natural_output(), run.count * sizeof(Word),
                               cudaMemcpyDeviceToHost, stream), "copy GPU-NTT output");
    check_cuda(cudaStreamSynchronize(stream), "synchronize GPU-NTT output");
    const auto expected = expected_transform(run.host_values, run.channel, options.batch, options.direction);
    for (std::size_t index = 0; index < run.count; ++index) {
        if (static_cast<U64>(output_host[index]) != expected[index]) {
            throw std::runtime_error("GPU exact verification failed in channel " + std::to_string(channel_index) +
                                     " at element " + std::to_string(index) +
                                     " (expected " + std::to_string(expected[index]) +
                                     ", got " + std::to_string(static_cast<U64>(output_host[index])) + ")");
        }
    }
}

struct DeviceInfo {
    std::string uuid;
    std::string name;
    std::size_t memory_bytes = 0;
    int max_grid_x = 0;
};

DeviceInfo device_info() {
    int device = 0;
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
    std::ostringstream uuid;
    uuid << "GPU-" << std::hex << std::setfill('0');
    for (int index = 0; index < 16; ++index) {
        if (index == 4 || index == 6 || index == 8 || index == 10) uuid << '-';
        uuid << std::setw(2) << static_cast<unsigned>(static_cast<unsigned char>(properties.uuid.bytes[index]));
    }
    return {uuid.str(), properties.name, properties.totalGlobalMem, properties.maxGridSize[0]};
}

template <typename Word>
Json run_benchmark(const Options& options, const DeviceInfo& device, cudaStream_t stream) {
    const auto setup_start = Clock::now();
    std::vector<ChannelInfo> channels;
    channels.reserve(options.moduli.size());
    for (const auto modulus : options.moduli) channels.push_back(make_channel(options, modulus));

    std::size_t estimated_bytes = 0;
    for (const auto& channel : channels) {
        const auto points = checked_mul(std::size_t{1} << options.log_n, options.batch, "estimated point count");
        const auto data_bytes = checked_mul(points, sizeof(Word) * 3, "estimated working set");
        estimated_bytes = checked_add(estimated_bytes, data_bytes, "estimated working set");
        estimated_bytes = checked_add(estimated_bytes, checked_mul(channel.timed_twist.size(), sizeof(Word) * 2,
                                                                  "estimated twist tables"), "estimated working set");
        estimated_bytes = checked_add(estimated_bytes,
                                      checked_mul(channel.native_x_n_plus ? points / options.batch : points / (options.batch * 2),
                                                  sizeof(Word), "estimated root table"),
                                      "estimated working set");
    }
    std::size_t free_before = 0;
    std::size_t total_memory = 0;
    check_cuda(cudaMemGetInfo(&free_before, &total_memory), "cudaMemGetInfo before GPU-NTT allocation");
    if (estimated_bytes > free_before * 9 / 10) throw Unsupported("estimated working set exceeds 90% of free GPU memory");

    std::vector<ChannelRun<Word>> runs;
    runs.reserve(channels.size());
    std::size_t working_set = 0;
    for (std::size_t index = 0; index < channels.size(); ++index) {
        runs.push_back(prepare_channel<Word>(options, channels[index], index, stream));
        working_set = checked_add(working_set, runs.back().device_bytes, "working set");
    }
    std::size_t free_after = 0;
    check_cuda(cudaMemGetInfo(&free_after, &total_memory), "cudaMemGetInfo after GPU-NTT allocation");
    const std::size_t allocation_delta = free_before > free_after ? free_before - free_after : 0;
    if (allocation_delta > free_before * 9 / 10) throw Unsupported("GPU-NTT allocation consumes more than 90% of free GPU memory");
    const double plan_ms = std::chrono::duration<double, std::milli>(Clock::now() - setup_start).count();

    for (std::uint32_t warmup = 0; warmup < options.warmup; ++warmup) {
        for (auto& run : runs) launch_channel(options, run, stream, device.max_grid_x);
    }
    check_cuda(cudaStreamSynchronize(stream), "synchronize GPU-NTT warmup");

    Event start;
    Event stop;
    check_cuda(cudaEventRecord(start.get(), stream), "record GPU-NTT start event");
    for (std::uint32_t repeat = 0; repeat < options.repeat; ++repeat) {
        for (auto& run : runs) launch_channel(options, run, stream, device.max_grid_x);
    }
    check_cuda(cudaEventRecord(stop.get(), stream), "record GPU-NTT stop event");
    check_cuda(cudaEventSynchronize(stop.get()), "synchronize GPU-NTT stop event");
    float elapsed_ms = 0.0F;
    check_cuda(cudaEventElapsedTime(&elapsed_ms, start.get(), stop.get()), "elapsed GPU-NTT event");
    const double kernel_ms = static_cast<double>(elapsed_ms) / options.repeat;

    std::vector<Json> records;
    records.reserve(runs.size());
    for (std::size_t index = 0; index < runs.size(); ++index) {
        verify_channel(options, runs[index], index, stream);
        const auto& channel = runs[index].channel;
        records.push_back(Json{{"channel", index},
                               {"modulus", channel.modulus},
                               {"modulus_bits", modulus_bits(channel.modulus)},
                               {"root", channel.root},
                               {"default_root_2n", channel.default_root_2n},
                               {"default_root_2n_squared", channel.default_root_2n_squared},
                               {"psi", channel.psi},
                               {"root_match", channel.root_match},
                               {"native_reduction_polynomial", channel.native_x_n_plus ? "X_N_plus" : "X_N_minus"},
                               {"twist_timing", channel.timed_twist.empty() ? "none" : "explicit_cuda"},
                               {"root_table_entries", runs[index].root_count}});
    }

    std::string method = "gpuntt_merge_x_n_minus_naturalized";
    if (options.mode == Mode::Negacyclic) method = "gpuntt_merge_x_n_plus_naturalized";
    else if (options.mode == Mode::Coset) method = "gpuntt_merge_x_n_minus_explicit_coset_naturalized";

    Json output;
    output["logN"] = options.log_n;
    output["batch"] = options.batch;
    output["word_bits"] = options.word_bits;
    output["mode"] = mode_name(options.mode);
    output["direction"] = direction_name(options.direction);
    output["moduli"] = options.moduli;
    output["gpu_uuid"] = device.uuid;
    output["device_name"] = device.name;
    output["device_memory_bytes"] = device.memory_bytes;
    output["compile_mode"] = std::getenv("CUBUTTERFLY_COMPILE_MODE") == nullptr
                                  ? "research" : std::getenv("CUBUTTERFLY_COMPILE_MODE");
    output["correct"] = true;
    output["verified_batches"] = options.batch;
    output["verified_channels"] = options.moduli.size();
    output["warmup"] = options.warmup;
    output["repeat"] = options.repeat;
    output["kernel_ms"] = kernel_ms;
    output["setup_ms"] = plan_ms;
    output["workspace_bytes"] = 0;
    output["working_set_bytes"] = working_set;
    output["allocation_delta_bytes"] = allocation_delta;
    output["transforms_per_second"] = kernel_ms > 0 ? (1000.0 * options.batch * options.moduli.size() / kernel_ms) : 0.0;
    output["ring_batches_per_second"] = kernel_ms > 0 ? (1000.0 * options.batch / kernel_ms) : 0.0;
    output["coset_generator"] = options.mode == Mode::Coset ? options.coset_generator : 0;
    output["input_pattern"] = pattern_name(options.input_pattern);
    output["seed"] = options.seed;
    output["normalization"] = options.direction == Direction::Inverse ? "inverse" : "none";
    output["layout"] = "channel-batch-coefficient";
    output["rns_stream_order"] = "single-stream";
    output["mapping_request"] = nullptr;
    output["baseline"] = "GPU-NTT";
    output["source"] = std::string("GPU-NTT official repository ") + GPUNTT_SOURCE_REVISION;
    output["method"] = method;
    output["implementation"] = "GPU-NTT Merge";
    output["input_order"] = "natural";
    output["output_order"] = "natural";
    output["timing_protocol"] = "CUDA events include bit reversal, twists, normalization; exclude setup and CPU validation";
    output["composition"] = "sequential per-channel GPU_NTT calls; no native fused RNS";
    output["channels"] = records;
    return output;
}

int execute(const Options& options) {
    const auto device = device_info();
    Stream stream;
    if (options.word_bits == 32) {
        const auto result = run_benchmark<std::uint32_t>(options, device, stream.get());
        std::cout << result.dump() << '\n';
    } else {
        const auto result = run_benchmark<std::uint64_t>(options, device, stream.get());
        std::cout << result.dump() << '\n';
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    bool json_requested = false;
    for (int index = 1; index < argc; ++index) {
        if (std::string(argv[index]) == "--json") json_requested = true;
    }
    try {
        const auto options = parse_options(argc, argv);
        return execute(options);
    } catch (const std::exception& error) {
        std::cerr << "gpuntt_crypto_bench: " << error.what() << '\n';
        if (json_requested) {
            std::cout << Json{{"correct", false}, {"status", "error"}, {"error", error.what()}}.dump() << '\n';
        }
        return 2;
    }
}
