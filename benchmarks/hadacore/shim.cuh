#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
namespace torch { enum class ScalarType { Half, BFloat16 }; }
#define C10_CUDA_CHECK(expr) do { if ((expr) != cudaSuccess) throw std::runtime_error("CUDA error"); } while (0)
#define C10_CUDA_KERNEL_LAUNCH_CHECK() C10_CUDA_CHECK(cudaGetLastError())
