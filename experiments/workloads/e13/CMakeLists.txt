cmake_minimum_required(VERSION 3.26)

project(heongpu_ntt_adapter_probe LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "Probe CUDA architecture")

get_filename_component(CUBUTTERFLY_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
set(HEONGPU_SOURCE_DIR "${CUBUTTERFLY_ROOT}/results/e13_assets/sources/HEonGPU" CACHE PATH
    "Pinned HEonGPU source tree")
set(HEONGPU_BUILD_DIR "${CUBUTTERFLY_ROOT}/results/e13_assets/heongpu-build-e13" CACHE PATH
    "Pinned HEonGPU build tree")
set(CUBUTTERFLY_BUILD_DIR "${CUBUTTERFLY_ROOT}/build-e13-shared" CACHE PATH
    "Existing cuButterfly shared build")

find_package(CUDAToolkit REQUIRED)

set(GPUNTT_ARCHIVE
    "${HEONGPU_BUILD_DIR}/thirdparty/GPU-NTT/src/libntt-1.0.a")
set(CUNTT_SHARED "${CUBUTTERFLY_BUILD_DIR}/libcuntt.so")
if(NOT EXISTS "${GPUNTT_ARCHIVE}")
    message(FATAL_ERROR "Pinned GPU-NTT archive not found: ${GPUNTT_ARCHIVE}")
endif()
if(NOT EXISTS "${CUNTT_SHARED}")
    message(FATAL_ERROR "cuButterfly shared library not found: ${CUNTT_SHARED}")
endif()

add_library(pinned_gpuntt STATIC IMPORTED GLOBAL)
set_target_properties(pinned_gpuntt PROPERTIES
    IMPORTED_LOCATION "${GPUNTT_ARCHIVE}"
    INTERFACE_INCLUDE_DIRECTORIES
        "${HEONGPU_SOURCE_DIR}/thirdparty/GPU-NTT/src/include")
target_link_libraries(pinned_gpuntt INTERFACE CUDA::cudart)

add_library(cuntt_existing SHARED IMPORTED GLOBAL)
set_target_properties(cuntt_existing PROPERTIES
    IMPORTED_LOCATION "${CUNTT_SHARED}"
    INTERFACE_INCLUDE_DIRECTORIES "${CUBUTTERFLY_ROOT}/include"
    INTERFACE_LINK_DIRECTORIES "${CUBUTTERFLY_BUILD_DIR}")

add_executable(heongpu_ntt_adapter_probe heongpu_ntt_adapter_probe.cu)
target_include_directories(heongpu_ntt_adapter_probe PRIVATE
    "${CUBUTTERFLY_ROOT}/include"
    "${HEONGPU_SOURCE_DIR}/thirdparty/GPU-NTT/src/include")
target_link_libraries(heongpu_ntt_adapter_probe PRIVATE
    pinned_gpuntt
    cuntt_existing
    CUDA::cudart
    CUDA::cufft)
target_compile_options(heongpu_ntt_adapter_probe PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>)
set_target_properties(heongpu_ntt_adapter_probe PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    BUILD_RPATH "${CUBUTTERFLY_BUILD_DIR}")

