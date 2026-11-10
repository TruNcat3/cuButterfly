#pragma once

#include <cuda_runtime_api.h>
#include <nvml.h>
#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

class GpuThermalGuard {
public:
    explicit GpuThermalGuard(bool enabled, int target_temperature = 75)
        : enabled_(enabled), target_temperature_(target_temperature) {
        if (!enabled_) return;
        if (target_temperature_ < 0)
            throw std::invalid_argument("GPU thermal target must be nonnegative");
        try {
            library_ = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
            if (!library_)
                throw std::runtime_error(std::string("load NVML: ") + dlerror());
            init_ = symbol<decltype(init_)>("nvmlInit_v2");
            shutdown_ = symbol<decltype(shutdown_)>("nvmlShutdown");
            error_string_ = symbol<decltype(error_string_)>("nvmlErrorString");
            get_handle_ = symbol<decltype(get_handle_)>("nvmlDeviceGetHandleByPciBusId_v2");
            get_temperature_ = symbol<decltype(get_temperature_)>("nvmlDeviceGetTemperature");
            get_clock_ = symbol<decltype(get_clock_)>("nvmlDeviceGetClockInfo");
            get_throttle_ = symbol<decltype(get_throttle_)>("nvmlDeviceGetCurrentClocksThrottleReasons");
            check(init_(), "initialize NVML");
            initialized_ = true;
            int device = 0;
            check_cuda(cudaGetDevice(&device), "get CUDA device");
            char pci_bus_id[32]{};
            check_cuda(cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), device),
                       "get CUDA PCI bus ID");
            pci_bus_id_ = pci_bus_id;
            check(get_handle_(pci_bus_id, &device_), "resolve CUDA device in NVML");
        } catch (...) {
            release();
            throw;
        }
    }

    ~GpuThermalGuard() { release(); }
    GpuThermalGuard(const GpuThermalGuard&) = delete;
    GpuThermalGuard& operator=(const GpuThermalGuard&) = delete;
    GpuThermalGuard(GpuThermalGuard&&) = delete;
    GpuThermalGuard& operator=(GpuThermalGuard&&) = delete;

    void ready() {
        if (!enabled_) return;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
        bool waiting = false;
        for (;;) {
            const auto state = snapshot();
            if (state.temperature <= static_cast<unsigned>(target_temperature_) &&
                !(state.throttle & thermal_flags_)) {
                emit(waiting ? "ready_after_cooling" : "ready", state);
                return;
            }
            if (!waiting) {
                emit("waiting_for_cooling", state);
                waiting = true;
            }
            const auto now = std::chrono::steady_clock::now();
            if (now >= deadline)
                throw std::runtime_error("GPU did not reach thermal target " +
                                         std::to_string(target_temperature_) +
                                         " C within 120 seconds: " + describe(state));
            std::this_thread::sleep_until(std::min(now + std::chrono::seconds(1), deadline));
        }
    }

    void check_sample() {
        if (!enabled_) return;
        const auto state = snapshot();
        if (state.throttle & thermal_flags_)
            throw std::runtime_error("GPU thermal throttling invalidates measurement: " +
                                     describe(state));
    }

    void log(const char* phase) {
        if (!enabled_) return;
        emit(phase ? phase : "sample", snapshot());
    }

private:
    struct Snapshot {
        unsigned temperature = 0;
        unsigned sm_clock = 0;
        unsigned long long throttle = 0;
    };
    static constexpr unsigned long long thermal_flags_ =
        nvmlClocksThrottleReasonSwThermalSlowdown | nvmlClocksThrottleReasonHwThermalSlowdown;

    template<class Function> Function symbol(const char* name) {
        dlerror();
        void* address = dlsym(library_, name);
        const char* error = dlerror();
        if (error || !address)
            throw std::runtime_error(std::string("load NVML symbol ") + name + ": " +
                                     (error ? error : "null address"));
        return reinterpret_cast<Function>(address);
    }

    void check(nvmlReturn_t status, const char* operation) const {
        if (status != NVML_SUCCESS)
            throw std::runtime_error(std::string(operation) + ": " + error_string_(status));
    }

    static void check_cuda(cudaError_t status, const char* operation) {
        if (status != cudaSuccess)
            throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }

    Snapshot snapshot() const {
        Snapshot state;
        check(get_temperature_(device_, NVML_TEMPERATURE_GPU, &state.temperature), "read GPU temperature");
        check(get_clock_(device_, NVML_CLOCK_SM, &state.sm_clock), "read GPU SM clock");
        check(get_throttle_(device_, &state.throttle), "read GPU throttle flags");
        return state;
    }

    std::string describe(const Snapshot& state) const {
        std::ostringstream text;
        text << "pci_bus_id=" << pci_bus_id_ << ",temperature_C=" << state.temperature
             << ",sm_clock_MHz=" << state.sm_clock << ",throttle_flags=0x"
             << std::hex << state.throttle;
        return text.str();
    }

    void emit(const char* phase, const Snapshot& state) const {
        std::cerr << "gpu_thermal_guard,phase=" << phase << ',' << describe(state) << '\n';
    }

    void release() noexcept {
        if (initialized_) {
            shutdown_();
            initialized_ = false;
        }
        if (library_) {
            dlclose(library_);
            library_ = nullptr;
        }
    }

    bool enabled_ = false;
    int target_temperature_ = 75;
    void* library_ = nullptr;
    bool initialized_ = false;
    nvmlDevice_t device_ = nullptr;
    std::string pci_bus_id_;
    decltype(&nvmlInit_v2) init_ = nullptr;
    decltype(&nvmlShutdown) shutdown_ = nullptr;
    decltype(&nvmlErrorString) error_string_ = nullptr;
    decltype(&nvmlDeviceGetHandleByPciBusId_v2) get_handle_ = nullptr;
    decltype(&nvmlDeviceGetTemperature) get_temperature_ = nullptr;
    decltype(&nvmlDeviceGetClockInfo) get_clock_ = nullptr;
    decltype(&nvmlDeviceGetCurrentClocksThrottleReasons) get_throttle_ = nullptr;
};
