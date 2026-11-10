cmake_minimum_required(VERSION 3.20)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    find_program(CUBUTTERFLY_BOOTSTRAP_PYTHON NAMES python3 REQUIRED)
    execute_process(COMMAND "${CUBUTTERFLY_BOOTSTRAP_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/detect_cuda_arch.py"
        RESULT_VARIABLE CUBUTTERFLY_ARCH_RESULT OUTPUT_VARIABLE CUBUTTERFLY_DETECTED_ARCH
        ERROR_VARIABLE CUBUTTERFLY_ARCH_ERROR OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT CUBUTTERFLY_ARCH_RESULT EQUAL 0)
        message(FATAL_ERROR "GPU detection failed; set CMAKE_CUDA_ARCHITECTURES explicitly for cross compilation. ${CUBUTTERFLY_ARCH_ERROR}")
    endif()
    set(CMAKE_CUDA_ARCHITECTURES "${CUBUTTERFLY_DETECTED_ARCH}" CACHE STRING "CUDA architectures to compile for")
endif()

project(cuButterfly VERSION 0.9.3 LANGUAGES C CXX CUDA)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

find_package(CUDAToolkit REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

include(FetchContent)
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()
FetchContent_Declare(nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d)
FetchContent_MakeAvailable(nlohmann_json)

option(CUBUTTERFLY_BUILD_SHARED "Build the cuButterfly C API as a shared library" OFF)
option(CUBUTTERFLY_ENABLE_VKFFT "Build the optional VkFFT comparison benchmark" OFF)
option(CUBUTTERFLY_ENABLE_XD_VKFFT "Enable the optional VkFFT xD axis processing unit" ${CUBUTTERFLY_ENABLE_VKFFT})
set(CUBUTTERFLY_VKFFT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/external/VkFFT" CACHE PATH
    "Path to a VkFFT source checkout")
option(CUBUTTERFLY_ENABLE_CUFFTDX "Enable the optional cuFFTDx block FFT processing unit" OFF)
option(CUBUTTERFLY_ENABLE_XD_CUFFTDX "Enable the cuFFTDx axis-tiled xD processing unit" ${CUBUTTERFLY_ENABLE_CUFFTDX})
option(CUBUTTERFLY_ENABLE_APPT_LAYOUT "Enable experimental APPT scratch layout" OFF)
set(CUBUTTERFLY_APPT_PARALLELISM 1 CACHE STRING "APPT boundary parallelism")
option(CUBUTTERFLY_APPT_STAGE_CONTRACT "Declare that both FFT stages share the APPT layout contract" OFF)
set(CUBUTTERFLY_MATHDX_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/external/mathdx/nvidia/mathdx" CACHE PATH
    "Path to an NVIDIA MathDx package")
option(CUBUTTERFLY_ENABLE_TURBOFFT "Enable the optional TurboFFT generated processing unit" OFF)
set(CUBUTTERFLY_TURBOFFT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/external/TurboFFT/TurboFFT" CACHE PATH
    "Path to a TurboFFT source checkout")
set(CUBUTTERFLY_ARCHITECTURE_MANIFEST "${CMAKE_CURRENT_SOURCE_DIR}/config/v09_architecture_manifest.json" CACHE FILEPATH
    "v0.9 layer and provenance manifest")

set(CUNTT_DESIGN_SPEC "${CMAKE_CURRENT_SOURCE_DIR}/config/v100_design_points.json" CACHE FILEPATH
    "JSON file selecting generated cuButterfly design points")
set(CUBUTTERFLY_FFT_SPACE_SPEC "${CMAKE_CURRENT_SOURCE_DIR}/config/fft_architecture_space.json" CACHE FILEPATH
    "Canonical two-dimensional FFT architecture-space specification")
set(CUBUTTERFLY_ARCHITECTURE_SPACE_SPEC "${CMAKE_CURRENT_SOURCE_DIR}/config/butterfly_architecture_space.json" CACHE FILEPATH
    "Canonical operator-independent butterfly architecture-space specification")
set(CUBUTTERFLY_FFT_CODEGEN_SPEC "${CMAKE_CURRENT_SOURCE_DIR}/config/v100_fft_codegen.json" CACHE FILEPATH
    "Selected FFT template points compiled into this build")
set(CUBUTTERFLY_FFT_CODEGEN_MODE "precompiled" CACHE STRING
    "FFT template policy: precompiled or on-demand (research builds)")
set_property(CACHE CUBUTTERFLY_FFT_CODEGEN_MODE PROPERTY STRINGS precompiled on-demand)
if(NOT CUBUTTERFLY_FFT_CODEGEN_MODE STREQUAL "precompiled" AND
   NOT CUBUTTERFLY_FFT_CODEGEN_MODE STREQUAL "on-demand")
    message(FATAL_ERROR "CUBUTTERFLY_FFT_CODEGEN_MODE must be precompiled or on-demand")
endif()
set(CUBUTTERFLY_FFT_PIPELINE_SPEC "${CMAKE_CURRENT_SOURCE_DIR}/config/v100_fft_pipeline.json" CACHE FILEPATH
    "FFT processing-unit and mapping search specification")
set(CUBUTTERFLY_FFT_PIPELINE_CANDIDATES "${CMAKE_CURRENT_SOURCE_DIR}/config/v100_fft_pipeline_candidates.json" CACHE FILEPATH
    "Generated FFT pipeline candidate manifest")
set(CUBUTTERFLY_FFT_DISPATCH_SPEC "${CMAKE_CURRENT_SOURCE_DIR}/config/v100_fft_dispatch.json" CACHE FILEPATH
    "Measured FFT mapping dispatch table")
set(CUNTT_HYBRID_DATAFLOW_SPEC "${CMAKE_CURRENT_SOURCE_DIR}/config/v100_hybrid_dataflow.json" CACHE FILEPATH
    "Generated HybridDataflow NTT design points")
set(CUBUTTERFLY_RUNTIME_SELECTOR_SUMMARY "${CMAKE_CURRENT_SOURCE_DIR}/config/legacy/v100_scaling_full_summary.csv" CACHE FILEPATH
    "Measured scaling summary used to generate runtime selector latency anchors")
set(CUBUTTERFLY_FFT14_DIRECT_SELECTOR_SUMMARY "${CMAKE_CURRENT_SOURCE_DIR}/results/fft14_dispatch_revalidation/raw.csv" CACHE FILEPATH
    "Optional focused logN=14 dispatch CSV used to refresh direct-core latency anchors")
set(CUBUTTERFLY_V08_NTT_SELECTOR_SUMMARY "${CMAKE_CURRENT_SOURCE_DIR}/results/v08_comprehensive_full/summary.csv" CACHE FILEPATH
    "Measured v0.8 NTT matrix used to generate numeric/length selector anchors")
set(CUBUTTERFLY_CONFIRMED_SELECTION_SPEC "${CMAKE_CURRENT_SOURCE_DIR}/config/v100_v08_confirmed_selection.json" CACHE FILEPATH
    "Exact V100 resident butterfly selections confirmed with repeated measurements")
set(CUNTT_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CUBUTTERFLY_ARCHITECTURE_MANIFEST}"
    "${CUNTT_DESIGN_SPEC}"
    "${CUBUTTERFLY_FFT_SPACE_SPEC}"
    "${CUBUTTERFLY_ARCHITECTURE_SPACE_SPEC}"
    "${CUBUTTERFLY_FFT_CODEGEN_SPEC}"
    "${CUBUTTERFLY_FFT_PIPELINE_SPEC}"
    "${CUBUTTERFLY_FFT_PIPELINE_CANDIDATES}"
    "${CUBUTTERFLY_FFT_DISPATCH_SPEC}"
    "${CUNTT_HYBRID_DATAFLOW_SPEC}"
    "${CUBUTTERFLY_RUNTIME_SELECTOR_SUMMARY}"
    "${CUBUTTERFLY_FFT14_DIRECT_SELECTOR_SUMMARY}"
    "${CUBUTTERFLY_V08_NTT_SELECTOR_SUMMARY}"
    "${CUBUTTERFLY_CONFIRMED_SELECTION_SPEC}"
    "${CMAKE_CURRENT_SOURCE_DIR}/config/v100_comprehensive_suite.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/config/v100_scaling_space.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/config/v100_scaling_suite.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/scripts/fft_design_space.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/scripts/butterfly_design_space.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_fft_codegen.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_fft_pipeline.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_fft_dispatch.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_runtime_selector.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_scaling_suite.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_design_points.py")
list(APPEND CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_hybrid_dataflow.py")
file(MAKE_DIRECTORY "${CUNTT_GENERATED_DIR}")
execute_process(
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/cufftdx_capabilities.py"
        --mathdx "${CUBUTTERFLY_MATHDX_ROOT}"
        --output "${CUNTT_GENERATED_DIR}/cufftdx_provider_capabilities.hpp"
        --metadata-output "${CUNTT_GENERATED_DIR}/cufftdx_provider_capabilities.json"
    RESULT_VARIABLE CUBUTTERFLY_PROVIDER_RESULT)
if(NOT CUBUTTERFLY_PROVIDER_RESULT EQUAL 0)
    message(FATAL_ERROR "Could not generate cuFFTDx provider capability data")
endif()
execute_process(
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/check_v09_architecture.py"
            --manifest "${CUBUTTERFLY_ARCHITECTURE_MANIFEST}"
            --root "${CMAKE_CURRENT_SOURCE_DIR}"
    RESULT_VARIABLE CUBUTTERFLY_ARCHITECTURE_MANIFEST_RESULT
    ERROR_VARIABLE CUBUTTERFLY_ARCHITECTURE_MANIFEST_ERROR)
if(NOT CUBUTTERFLY_ARCHITECTURE_MANIFEST_RESULT EQUAL 0)
    message(FATAL_ERROR "v0.9 architecture manifest validation failed: ${CUBUTTERFLY_ARCHITECTURE_MANIFEST_ERROR}")
endif()
set(CUNTT_LOCAL_SELECTOR_HEADER "${CUNTT_GENERATED_DIR}/generated_local_selector.hpp")
if(NOT EXISTS "${CUNTT_LOCAL_SELECTOR_HEADER}")
    file(WRITE "${CUNTT_LOCAL_SELECTOR_HEADER}"
"// Initial local-selector stub; install-time calibration may replace this file.\n"
"#pragma once\n"
"#include <array>\n"
"#include <cstddef>\n"
"#include <cstdint>\n"
"namespace cuntt::detail {\n"
"struct LocalSelectorPoint { const char* operator_name; const char* precision; const char* placement; std::size_t log_n; std::size_t batch; std::uint64_t modulus; const char* candidate; double kernel_ms; };\n"
"inline constexpr const char* kLocalSelectorDeviceName = \"none\";\n"
"inline constexpr const char* kLocalSelectorTarget = \"none\";\n"
"inline constexpr int kLocalSelectorComputeMajor = -1;\n"
"inline constexpr int kLocalSelectorComputeMinor = -1;\n"
"inline constexpr std::array<LocalSelectorPoint, 0> kLocalSelectorPoints = {};\n"
"}  // namespace cuntt::detail\n")
endif()
execute_process(
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_design_points.py"
            --spec "${CUNTT_DESIGN_SPEC}" --output-dir "${CUNTT_GENERATED_DIR}"
    RESULT_VARIABLE CUNTT_GENERATOR_RESULT
    ERROR_VARIABLE CUNTT_GENERATOR_ERROR)
if(NOT CUNTT_GENERATOR_RESULT EQUAL 0)
    message(FATAL_ERROR "cuButterfly design-point generation failed: ${CUNTT_GENERATOR_ERROR}")
endif()
execute_process(
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_hybrid_dataflow.py"
            --spec "${CUNTT_HYBRID_DATAFLOW_SPEC}"
            --header "${CUNTT_GENERATED_DIR}/generated_hybrid_dataflow_config.hpp"
    RESULT_VARIABLE CUNTT_HYBRID_DATAFLOW_RESULT
    ERROR_VARIABLE CUNTT_HYBRID_DATAFLOW_ERROR)
if(NOT CUNTT_HYBRID_DATAFLOW_RESULT EQUAL 0)
    message(FATAL_ERROR "HybridDataflow generation failed: ${CUNTT_HYBRID_DATAFLOW_ERROR}")
endif()
execute_process(
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/butterfly_design_space.py"
            --spec "${CUBUTTERFLY_ARCHITECTURE_SPACE_SPEC}"
            --fft-spec "${CUBUTTERFLY_FFT_SPACE_SPEC}" --validate-only
    RESULT_VARIABLE CUBUTTERFLY_ARCHITECTURE_SPACE_RESULT
    ERROR_VARIABLE CUBUTTERFLY_ARCHITECTURE_SPACE_ERROR)
if(NOT CUBUTTERFLY_ARCHITECTURE_SPACE_RESULT EQUAL 0)
    message(FATAL_ERROR "cuButterfly architecture-space validation failed: ${CUBUTTERFLY_ARCHITECTURE_SPACE_ERROR}")
endif()
configure_file("${CUBUTTERFLY_ARCHITECTURE_SPACE_SPEC}"
               "${CUNTT_GENERATED_DIR}/butterfly_architecture_space.json" COPYONLY)
execute_process(
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/fft_design_space.py"
            --spec "${CUBUTTERFLY_FFT_SPACE_SPEC}" --validate-only
    RESULT_VARIABLE CUBUTTERFLY_FFT_SPACE_RESULT
    ERROR_VARIABLE CUBUTTERFLY_FFT_SPACE_ERROR)
if(NOT CUBUTTERFLY_FFT_SPACE_RESULT EQUAL 0)
    message(FATAL_ERROR "cuButterfly FFT architecture-space validation failed: ${CUBUTTERFLY_FFT_SPACE_ERROR}")
endif()
configure_file("${CUBUTTERFLY_FFT_SPACE_SPEC}"
               "${CUNTT_GENERATED_DIR}/fft_architecture_space.json" COPYONLY)
execute_process(
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_fft_codegen.py"
            --space-spec "${CUBUTTERFLY_FFT_SPACE_SPEC}"
            --selection "${CUBUTTERFLY_FFT_CODEGEN_SPEC}"
            --output-dir "${CUNTT_GENERATED_DIR}"
    RESULT_VARIABLE CUBUTTERFLY_FFT_CODEGEN_RESULT
    ERROR_VARIABLE CUBUTTERFLY_FFT_CODEGEN_ERROR)
if(NOT CUBUTTERFLY_FFT_CODEGEN_RESULT EQUAL 0)
    message(FATAL_ERROR "cuButterfly FFT code generation failed: ${CUBUTTERFLY_FFT_CODEGEN_ERROR}")
endif()
execute_process(
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_scaling_suite.py"
            --spec "${CMAKE_CURRENT_SOURCE_DIR}/config/v100_scaling_space.json"
            --check "${CMAKE_CURRENT_SOURCE_DIR}/config/v100_scaling_suite.json"
    RESULT_VARIABLE CUBUTTERFLY_SCALING_SUITE_RESULT
    ERROR_VARIABLE CUBUTTERFLY_SCALING_SUITE_ERROR)
if(NOT CUBUTTERFLY_SCALING_SUITE_RESULT EQUAL 0)
    message(FATAL_ERROR "cuButterfly scaling-suite validation failed: ${CUBUTTERFLY_SCALING_SUITE_ERROR}")
endif()
execute_process(
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_fft_pipeline.py"
            --pipeline "${CUBUTTERFLY_FFT_PIPELINE_SPEC}"
            --architecture "${CUBUTTERFLY_FFT_SPACE_SPEC}"
            --codegen "${CUBUTTERFLY_FFT_CODEGEN_SPEC}"
            --check "${CUBUTTERFLY_FFT_PIPELINE_CANDIDATES}"
    RESULT_VARIABLE CUBUTTERFLY_FFT_PIPELINE_RESULT
    ERROR_VARIABLE CUBUTTERFLY_FFT_PIPELINE_ERROR)
if(NOT CUBUTTERFLY_FFT_PIPELINE_RESULT EQUAL 0)
    message(FATAL_ERROR "cuButterfly FFT pipeline validation failed: ${CUBUTTERFLY_FFT_PIPELINE_ERROR}")
endif()
execute_process(
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_fft_dispatch.py"
            --dispatch "${CUBUTTERFLY_FFT_DISPATCH_SPEC}"
            --header "${CUNTT_GENERATED_DIR}/generated_fft_dispatch.cuh"
    RESULT_VARIABLE CUBUTTERFLY_FFT_DISPATCH_RESULT
    ERROR_VARIABLE CUBUTTERFLY_FFT_DISPATCH_ERROR)
if(NOT CUBUTTERFLY_FFT_DISPATCH_RESULT EQUAL 0)
    message(FATAL_ERROR "cuButterfly FFT dispatch generation failed: ${CUBUTTERFLY_FFT_DISPATCH_ERROR}")
endif()
execute_process(
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_runtime_selector.py"
            --summary "${CUBUTTERFLY_RUNTIME_SELECTOR_SUMMARY}"
            --fft14-direct-summary "${CUBUTTERFLY_FFT14_DIRECT_SELECTOR_SUMMARY}"
            --v08-ntt-summary "${CUBUTTERFLY_V08_NTT_SELECTOR_SUMMARY}"
            --confirmed-selection "${CUBUTTERFLY_CONFIRMED_SELECTION_SPEC}"
            --header "${CUNTT_GENERATED_DIR}/generated_runtime_selector.hpp"
    RESULT_VARIABLE CUBUTTERFLY_RUNTIME_SELECTOR_RESULT
    ERROR_VARIABLE CUBUTTERFLY_RUNTIME_SELECTOR_ERROR)
if(NOT CUBUTTERFLY_RUNTIME_SELECTOR_RESULT EQUAL 0)
    message(FATAL_ERROR "cuButterfly runtime selector generation failed: ${CUBUTTERFLY_RUNTIME_SELECTOR_ERROR}")
endif()
execute_process(
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_application_profiles.py"
            --spec "${CMAKE_CURRENT_SOURCE_DIR}/config/v100_application_profiles.json"
            --header "${CUNTT_GENERATED_DIR}/generated_application_profiles.hpp"
    RESULT_VARIABLE CUBUTTERFLY_APPLICATION_PROFILE_RESULT
    ERROR_VARIABLE CUBUTTERFLY_APPLICATION_PROFILE_ERROR)
if(NOT CUBUTTERFLY_APPLICATION_PROFILE_RESULT EQUAL 0)
    message(FATAL_ERROR "cuButterfly application profile generation failed: ${CUBUTTERFLY_APPLICATION_PROFILE_ERROR}")
endif()
include("${CUNTT_GENERATED_DIR}/design_points.cmake")

set(CUBUTTERFLY_LIBRARY_TYPE STATIC)
if(CUBUTTERFLY_BUILD_SHARED)
    set(CUBUTTERFLY_LIBRARY_TYPE SHARED)
endif()
add_library(cuntt ${CUBUTTERFLY_LIBRARY_TYPE}
    src/real_fft_adapter.cu
    src/bluestein_fft.cu
    src/bluestein_ntt.cu
    src/c_api.cpp
    src/semantic_contract.cpp
    src/direct_fft.cu
    src/direct_real_fft.cu
    src/butterfly.cu
    src/butterfly_reference.cpp
    src/general_kernels.cu
    src/semantic_numeric_kernels.cu
    src/xd_real_fft.cu
    src/xd_resident.cu
    src/mapping_selector.cpp
    src/platform_tuning.cpp
    src/platforms/legacy_selector.cpp
    src/plan_registry.cpp
    src/jit_module.cpp
    src/fft_factor_plan.cpp
    src/mixed_dataflow.cpp
    src/ntt.cu
    src/rns_ntt.cu
    src/rns_tuning.cpp
    src/rns_c_api.cpp
    src/reference.cpp)
target_link_libraries(cuntt PRIVATE $<BUILD_INTERFACE:nlohmann_json::nlohmann_json>)
target_compile_definitions(cuntt PRIVATE
    CUBUTTERFLY_BUNDLED_HARDWARE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/config/hardware"
    CUBUTTERFLY_INSTALLED_RANKER="${CMAKE_INSTALL_FULL_BINDIR}/rank_static_candidates.py"
    CUBUTTERFLY_INSTALLED_HARDWARE_ROOT="${CMAKE_INSTALL_FULL_DATADIR}/cuButterfly/hardware")
target_link_libraries(cuntt PRIVATE ${CMAKE_DL_LIBS})
configure_file(src/jit_config.hpp.in "${CUNTT_GENERATED_DIR}/jit_config.hpp" @ONLY)
file(GLOB_RECURSE CUBUTTERFLY_FINGERPRINT_INPUTS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cuh" "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp")
list(APPEND CUBUTTERFLY_FINGERPRINT_INPUTS "${CMAKE_CURRENT_SOURCE_DIR}/scripts/compile_module.py")
list(APPEND CUBUTTERFLY_FINGERPRINT_INPUTS "${CMAKE_CURRENT_SOURCE_DIR}/scripts/resident_mapping.py")
list(APPEND CUBUTTERFLY_FINGERPRINT_INPUTS "${CMAKE_CURRENT_SOURCE_DIR}/scripts/cufftdx_capabilities.py")
list(APPEND CUBUTTERFLY_FINGERPRINT_INPUTS "${CMAKE_CURRENT_SOURCE_DIR}/scripts/cufftdx_low_module.py")
file(GLOB_RECURSE CUBUTTERFLY_PROVIDER_INPUTS CONFIGURE_DEPENDS
    "${CUBUTTERFLY_MATHDX_ROOT}/include/*.hpp" "${CUBUTTERFLY_MATHDX_ROOT}/include/*.hpp.inc")
list(APPEND CUBUTTERFLY_FINGERPRINT_INPUTS ${CUBUTTERFLY_PROVIDER_INPUTS})
list(APPEND CUBUTTERFLY_FINGERPRINT_INPUTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/runtime_fingerprint.cmake")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CUBUTTERFLY_FINGERPRINT_INPUTS})
set(CUBUTTERFLY_FINGERPRINT_TEXT "${PROJECT_VERSION}|${CMAKE_CXX_COMPILER_ID}|${CMAKE_CXX_COMPILER_VERSION}|${CMAKE_CUDA_COMPILER_VERSION}|${CMAKE_CUDA_ARCHITECTURES}")
foreach(CUBUTTERFLY_BEHAVIOR_SETTING IN ITEMS
        CMAKE_BUILD_TYPE CMAKE_CONFIGURATION_TYPES
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_RELWITHDEBINFO CMAKE_CXX_FLAGS_MINSIZEREL
        CMAKE_CUDA_FLAGS CMAKE_CUDA_FLAGS_DEBUG CMAKE_CUDA_FLAGS_RELEASE
        CMAKE_CUDA_FLAGS_RELWITHDEBINFO CMAKE_CUDA_FLAGS_MINSIZEREL
        CUBUTTERFLY_ENABLE_CUFFTDX CUBUTTERFLY_ENABLE_XD_CUFFTDX
        CUBUTTERFLY_ENABLE_XD_VKFFT
        CUBUTTERFLY_ENABLE_TURBOFFT CUBUTTERFLY_ENABLE_APPT_LAYOUT
        CUBUTTERFLY_APPT_STAGE_CONTRACT CUBUTTERFLY_APPT_PARALLELISM
        CUBUTTERFLY_FFT_CODEGEN_MODE)
    string(APPEND CUBUTTERFLY_FINGERPRINT_TEXT "|${CUBUTTERFLY_BEHAVIOR_SETTING}=${${CUBUTTERFLY_BEHAVIOR_SETTING}}")
endforeach()
foreach(CUBUTTERFLY_FINGERPRINT_INPUT IN LISTS CUBUTTERFLY_FINGERPRINT_INPUTS)
    file(SHA256 "${CUBUTTERFLY_FINGERPRINT_INPUT}" CUBUTTERFLY_FILE_HASH)
    string(APPEND CUBUTTERFLY_FINGERPRINT_TEXT "|${CUBUTTERFLY_FILE_HASH}")
endforeach()
file(SHA256 "${CUBUTTERFLY_FFT_CODEGEN_SPEC}" CUBUTTERFLY_CODEGEN_HASH)
string(SHA256 CUBUTTERFLY_BASE_FINGERPRINT "${CUBUTTERFLY_FINGERPRINT_TEXT}|${CUBUTTERFLY_CODEGEN_HASH}")
file(GENERATE OUTPUT "${CUNTT_GENERATED_DIR}/fingerprint_config.txt" CONTENT "${CUBUTTERFLY_BASE_FINGERPRINT}\n")
# Calibration can replace the local selector after configuration. Keep its
# identity separate so installing selections preserves measured kernel records.
file(GLOB_RECURSE CUBUTTERFLY_GENERATED_FINGERPRINT_INPUTS
    "${CUNTT_GENERATED_DIR}/*.hpp" "${CUNTT_GENERATED_DIR}/*.cuh" "${CUNTT_GENERATED_DIR}/*.cu")
list(REMOVE_ITEM CUBUTTERFLY_GENERATED_FINGERPRINT_INPUTS
    "${CUNTT_GENERATED_DIR}/build_fingerprint.hpp" "${CUNTT_LOCAL_SELECTOR_HEADER}")
add_custom_command(OUTPUT "${CUNTT_GENERATED_DIR}/build_fingerprint.hpp"
    COMMAND "${CMAKE_COMMAND}"
        "-DBASE_FINGERPRINT=${CUBUTTERFLY_BASE_FINGERPRINT}"
        "-DINPUTS=${CUBUTTERFLY_GENERATED_FINGERPRINT_INPUTS}"
        "-DLOCAL_SELECTOR=${CUNTT_LOCAL_SELECTOR_HEADER}"
        "-DTEMPLATE=${CMAKE_CURRENT_SOURCE_DIR}/src/build_fingerprint.hpp.in"
        "-DOUTPUT=${CUNTT_GENERATED_DIR}/build_fingerprint.hpp"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/runtime_fingerprint.cmake"
    DEPENDS ${CUBUTTERFLY_GENERATED_FINGERPRINT_INPUTS} ${CUBUTTERFLY_FINGERPRINT_INPUTS}
        "${CUNTT_LOCAL_SELECTOR_HEADER}"
        "${CUNTT_GENERATED_DIR}/fingerprint_config.txt"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/build_fingerprint.hpp.in"
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/runtime_fingerprint.cmake"
    VERBATIM)
target_sources(cuntt PRIVATE ${CUNTT_GENERATED_SOURCES} "${CUNTT_GENERATED_DIR}/build_fingerprint.hpp")
set_property(SOURCE src/platforms/legacy_selector.cpp APPEND PROPERTY OBJECT_DEPENDS "${CUNTT_LOCAL_SELECTOR_HEADER}")

if(CUBUTTERFLY_ENABLE_CUFFTDX OR CUBUTTERFLY_ENABLE_XD_CUFFTDX)
    include(cmake/CuFFTDxTarget.cmake)
    cubutterfly_cufftdx_target(CUBUTTERFLY_CODELET_ARCH CUBUTTERFLY_CODELET_SM
        ${CMAKE_CUDA_ARCHITECTURES})
    cubutterfly_probe_cufftdx("${CMAKE_CUDA_COMPILER}" "${CUBUTTERFLY_MATHDX_ROOT}"
        "${CUBUTTERFLY_CODELET_ARCH}" "${CUBUTTERFLY_CODELET_SM}"
        "${CMAKE_CURRENT_BINARY_DIR}/cufftdx-target-probe")
endif()

if(CUBUTTERFLY_ENABLE_CUFFTDX)
    if(NOT EXISTS "${CUBUTTERFLY_MATHDX_ROOT}/include/cufftdx.hpp")
        message(FATAL_ERROR
            "cuFFTDx headers were not found under ${CUBUTTERFLY_MATHDX_ROOT}. "
            "Run scripts/install_cufftdx.sh or set CUBUTTERFLY_MATHDX_ROOT.")
    endif()
    target_sources(cuntt PRIVATE src/cufftdx_fft.cu src/cufftdx_fft_fp64.cu src/fft_register_dispatch.cu)
    target_include_directories(cuntt SYSTEM PRIVATE "${CUBUTTERFLY_MATHDX_ROOT}/include")
    target_compile_definitions(cuntt PRIVATE CUBUTTERFLY_HAS_CUFFTDX=1)
    if(CUBUTTERFLY_ENABLE_APPT_LAYOUT AND CUBUTTERFLY_APPT_STAGE_CONTRACT)
        target_compile_definitions(cuntt PRIVATE
            CUBUTTERFLY_APPT_LAYOUT=1
            CUBUTTERFLY_APPT_STAGE_CONTRACT=1
            CUBUTTERFLY_APPT_PARALLELISM=${CUBUTTERFLY_APPT_PARALLELISM})
    endif()
    set(CUBUTTERFLY_CUFFTDX_SM "${CUBUTTERFLY_CODELET_SM}")
    target_compile_definitions(cuntt PRIVATE CUBUTTERFLY_CUFFTDX_SM=${CUBUTTERFLY_CUFFTDX_SM})
else()
    target_sources(cuntt PRIVATE src/cufftdx_fft_stub.cpp src/fft_register_dispatch_stub.cpp)
endif()

# The radix-4 pair is an independent core provider. Its compilation does not
# require MathDx; the plan retains optional vendor fallbacks for other axes.
target_sources(cuntt PRIVATE src/xd_pair_core.cpp src/xd_stockham_pair_fft.cu)
if(CUBUTTERFLY_ENABLE_XD_CUFFTDX)
    if(NOT EXISTS "${CUBUTTERFLY_MATHDX_ROOT}/include/cufftdx.hpp")
        message(FATAL_ERROR "xD cuFFTDx needs headers under CUBUTTERFLY_MATHDX_ROOT; run scripts/install_cufftdx.sh")
    endif()
    set(CUBUTTERFLY_XD_SM "${CUBUTTERFLY_CODELET_SM}")
    target_sources(cuntt PRIVATE src/xd_axis_fft.cu src/xd_pair_fft.cu
        src/xd_register_fft.cu src/xd_cooperative_pair_fft.cu src/xd_local_fft.cu)
    set_source_files_properties(src/xd_axis_fft.cu src/xd_pair_fft.cu
        src/xd_register_fft.cu src/xd_cooperative_pair_fft.cu src/xd_local_fft.cu PROPERTIES
        COMPILE_DEFINITIONS "CUBUTTERFLY_XD_CUFFTDX_SM=${CUBUTTERFLY_XD_SM}")
    target_include_directories(cuntt SYSTEM PRIVATE "${CUBUTTERFLY_MATHDX_ROOT}/include")
else()
    target_sources(cuntt PRIVATE src/xd_axis_fft_stub.cpp src/xd_pair_fft_stub.cpp
        src/xd_register_fft_stub.cpp src/xd_cooperative_pair_fft_stub.cpp src/xd_local_fft_stub.cpp)
endif()

if(CUBUTTERFLY_ENABLE_TURBOFFT)
    foreach(LOG_N RANGE 7 10)
        if(NOT EXISTS "${CUBUTTERFLY_TURBOFFT_ROOT}/include/code_gen/generated/float2/fft_radix_2_logN_${LOG_N}_upload_0.cuh")
            message(FATAL_ERROR
                "TurboFFT logN=${LOG_N} generated header was not found under ${CUBUTTERFLY_TURBOFFT_ROOT}. "
                "Run scripts/install_turbofft.sh or set CUBUTTERFLY_TURBOFFT_ROOT.")
        endif()
    endforeach()
    target_sources(cuntt PRIVATE src/turbofft_fft.cu)
    target_include_directories(cuntt SYSTEM PRIVATE "${CUBUTTERFLY_TURBOFFT_ROOT}/include")
    target_compile_definitions(cuntt PRIVATE CUBUTTERFLY_HAS_TURBOFFT=1)
else()
    target_sources(cuntt PRIVATE src/turbofft_fft_stub.cpp)
endif()

target_include_directories(cuntt
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CUNTT_GENERATED_DIR})

target_compile_features(cuntt PUBLIC cxx_std_17)

target_link_libraries(cuntt PUBLIC CUDA::cudart CUDA::cufft)

target_compile_options(cuntt
    PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)

set_target_properties(cuntt PROPERTIES
    EXPORT_NAME cuButterfly
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON)

option(CUBUTTERFLY_BUILD_EXAMPLES "Build cuButterfly API examples" ON)
if(CUBUTTERFLY_BUILD_EXAMPLES)
    add_executable(cubutterfly_device_api_example examples/device_api.cpp)
    target_link_libraries(cubutterfly_device_api_example PRIVATE cuntt)
    target_compile_options(cubutterfly_device_api_example PRIVATE -Wall -Wextra -Wpedantic)

    add_executable(cubutterfly_basic_fft_example examples/basic_fft.cpp)
    target_link_libraries(cubutterfly_basic_fft_example PRIVATE cuntt)
    target_compile_options(cubutterfly_basic_fft_example PRIVATE -Wall -Wextra -Wpedantic)

    add_executable(cuntt_device_api_example examples/ntt_device_api.cpp)
    target_link_libraries(cuntt_device_api_example PRIVATE cuntt)
    target_compile_options(cuntt_device_api_example PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(cuntt_bench apps/cuntt_bench.cpp)
target_link_libraries(cuntt_bench PRIVATE cuntt)
target_compile_options(cuntt_bench PRIVATE -Wall -Wextra -Wpedantic)

add_executable(cubutterfly_bench apps/cubutterfly_bench.cpp)
target_link_libraries(cubutterfly_bench PRIVATE cuntt)
target_compile_options(cubutterfly_bench PRIVATE -Wall -Wextra -Wpedantic)

add_executable(cubutterfly_graph_bench apps/cubutterfly_graph_bench.cpp)
target_link_libraries(cubutterfly_graph_bench PRIVATE cuntt nlohmann_json::nlohmann_json)
target_compile_options(cubutterfly_graph_bench PRIVATE -Wall -Wextra -Wpedantic)

# Explicit mechanism controls; this is not an AUTO candidate or calibration entry.
add_executable(cubutterfly_boundary_bench apps/cubutterfly_boundary_bench.cu)
target_include_directories(cubutterfly_boundary_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(cubutterfly_boundary_bench PRIVATE cuntt CUDA::cufft)

add_executable(cubutterfly_plan_probe apps/cubutterfly_plan_probe.cpp)
target_link_libraries(cubutterfly_plan_probe PRIVATE cuntt nlohmann_json::nlohmann_json)

add_executable(cubutterfly_stage_microbench apps/cubutterfly_stage_microbench.cpp)
target_link_libraries(cubutterfly_stage_microbench PRIVATE cuntt nlohmann_json::nlohmann_json CUDA::cuda_driver)
target_include_directories(cubutterfly_stage_microbench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_executable(cubutterfly_schedule_microbench apps/cubutterfly_schedule_microbench.cu)
target_link_libraries(cubutterfly_schedule_microbench PRIVATE CUDA::cudart nlohmann_json::nlohmann_json)

add_executable(cubutterfly_pipeline_microbench apps/cubutterfly_pipeline_microbench.cu)
target_link_libraries(cubutterfly_pipeline_microbench PRIVATE CUDA::cudart nlohmann_json::nlohmann_json)
target_include_directories(cubutterfly_pipeline_microbench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

if(CUBUTTERFLY_ENABLE_CUFFTDX)
    add_executable(fft_resident_experiment apps/fft_resident_experiment.cu)
    add_executable(cubutterfly_register_tile_tests tests/test_fft_register_tile.cpp)
    target_link_libraries(cubutterfly_register_tile_tests PRIVATE cuntt)
    add_test(NAME fft_register_tile_correctness COMMAND cubutterfly_register_tile_tests)
    target_link_libraries(fft_resident_experiment PRIVATE cuntt)
    target_include_directories(fft_resident_experiment PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_definitions(fft_resident_experiment PRIVATE CUBUTTERFLY_CUFFTDX_SM=${CUBUTTERFLY_CUFFTDX_SM})
    target_include_directories(fft_resident_experiment SYSTEM PRIVATE ${CUBUTTERFLY_MATHDX_ROOT}/include
        ${CUBUTTERFLY_MATHDX_ROOT}/external/cutlass/include)

    # These diagnostic benchmarks are optional historical artifacts. Some
    # source snapshots retain the documentation but not the benchmark files;
    # the cuFFTDx library adapters must remain buildable without them.
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/apps/cufftdx_role_codelet_bench.cpp")
        add_executable(cufftdx_role_codelet_bench apps/cufftdx_role_codelet_bench.cpp)
        target_link_libraries(cufftdx_role_codelet_bench PRIVATE cuntt)
        target_include_directories(cufftdx_role_codelet_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
        target_compile_options(cufftdx_role_codelet_bench PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/apps/cufftdx_role_pipeline64_bench.cpp")
        add_executable(cufftdx_role_pipeline64_bench apps/cufftdx_role_pipeline64_bench.cpp)
        target_link_libraries(cufftdx_role_pipeline64_bench PRIVATE cuntt)
        target_include_directories(cufftdx_role_pipeline64_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
        target_compile_options(cufftdx_role_pipeline64_bench PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endif()

# Standalone route-B experiment: explicit radix-2 Stockham ping-pong stages
# plus CUDA Graph replay. It is intentionally outside the public selector.
if(CMAKE_CUDA_COMPILER)
    add_executable(stockham_subgraph_bench apps/stockham_subgraph_bench.cu)
    target_include_directories(stockham_subgraph_bench PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(stockham_subgraph_bench PRIVATE cuntt CUDA::cudart CUDA::cufft)
    target_compile_options(stockham_subgraph_bench PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)
endif()

# Compare a real open-source generated kernel under the same resident-subgraph
# and CUDA Graph protocol.  VkFFT remains optional and is never a public
# selector default.
if(CUBUTTERFLY_ENABLE_VKFFT)
    add_executable(open_source_subgraph_bench apps/open_source_subgraph_bench.cu)
    target_link_libraries(open_source_subgraph_bench PRIVATE
        vkfft_xd_bridge CUDA::cudart CUDA::cufft)
    target_compile_options(open_source_subgraph_bench PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)
    add_executable(vkfft_hybrid_subgraph_bench
        apps/vkfft_hybrid_subgraph_bench.cu)
    target_include_directories(vkfft_hybrid_subgraph_bench PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(vkfft_hybrid_subgraph_bench PRIVATE
        cuntt vkfft_xd_bridge CUDA::cudart CUDA::cufft)
    target_compile_options(vkfft_hybrid_subgraph_bench PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)
endif()

add_executable(cubutterfly_plan_bench apps/cubutterfly_plan_bench.cpp)
target_link_libraries(cubutterfly_plan_bench PRIVATE cuntt)
target_compile_options(cubutterfly_plan_bench PRIVATE -Wall -Wextra -Wpedantic)

add_executable(cuntt_tests tests/test_ntt.cpp)
target_link_libraries(cuntt_tests PRIVATE cuntt)
target_compile_options(cuntt_tests PRIVATE -Wall -Wextra -Wpedantic)

add_executable(cubutterfly_tests tests/test_butterfly.cpp)
target_link_libraries(cubutterfly_tests PRIVATE cuntt)
target_compile_options(cubutterfly_tests PRIVATE -Wall -Wextra -Wpedantic)

add_executable(cubutterfly_stage_overlap_tests tests/test_stage_overlap.cpp)
add_executable(cubutterfly_fft_tile_layout_tests tests/test_fft_tile_layout.cpp)
target_include_directories(cubutterfly_fft_tile_layout_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_features(cubutterfly_fft_tile_layout_tests PRIVATE cxx_std_17)
target_link_libraries(cubutterfly_stage_overlap_tests PRIVATE cuntt)
target_compile_options(cubutterfly_stage_overlap_tests PRIVATE -Wall -Wextra -Wpedantic)
if(CUBUTTERFLY_ENABLE_CUFFTDX)
    target_compile_definitions(cubutterfly_stage_overlap_tests PRIVATE CUBUTTERFLY_TEST_CUFFTDX=1)
    add_executable(cubutterfly_factor_pipeline_tests tests/test_factor_pipeline.cpp)
    target_link_libraries(cubutterfly_factor_pipeline_tests PRIVATE cuntt)
    target_compile_options(cubutterfly_factor_pipeline_tests PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_definitions(cubutterfly_factor_pipeline_tests PRIVATE CUBUTTERFLY_TEST_CUFFTDX=1)
    add_executable(fft_stage_overlap_bench apps/fft_stage_overlap_bench.cpp)
    target_link_libraries(fft_stage_overlap_bench PRIVATE cuntt)
endif()

add_executable(cubutterfly_structured_reference_tests tests/test_structured_reference.cpp)
target_link_libraries(cubutterfly_structured_reference_tests PRIVATE cuntt)
target_compile_options(cubutterfly_structured_reference_tests PRIVATE -Wall -Wextra -Wpedantic)
add_executable(cubutterfly_mixed_dataflow_tests tests/test_mixed_dataflow.cpp
    src/mixed_dataflow.cpp src/fft_factor_plan.cpp src/butterfly_reference.cpp src/reference.cpp)
target_include_directories(cubutterfly_mixed_dataflow_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(cubutterfly_mixed_dataflow_tests PRIVATE CUDA::cudart nlohmann_json::nlohmann_json)
target_compile_definitions(cubutterfly_mixed_dataflow_tests PRIVATE
    CUBUTTERFLY_BUNDLED_HARDWARE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/config/hardware")
add_executable(cubutterfly_hardware_model_tests tests/test_hardware_model.cpp)
target_include_directories(cubutterfly_hardware_model_tests PRIVATE src include)
target_link_libraries(cubutterfly_hardware_model_tests PRIVATE nlohmann_json::nlohmann_json)
target_compile_features(cubutterfly_hardware_model_tests PRIVATE cxx_std_17)
target_compile_options(cubutterfly_hardware_model_tests PRIVATE -Wall -Wextra -Wpedantic -UNDEBUG)
add_executable(cubutterfly_static_ranking_tests tests/test_static_candidate_ranking.cpp)
target_include_directories(cubutterfly_static_ranking_tests PRIVATE src)
target_link_libraries(cubutterfly_static_ranking_tests PRIVATE nlohmann_json::nlohmann_json)
target_compile_features(cubutterfly_static_ranking_tests PRIVATE cxx_std_17)
target_compile_options(cubutterfly_static_ranking_tests PRIVATE -Wall -Wextra -Wpedantic -UNDEBUG)
target_compile_options(cubutterfly_mixed_dataflow_tests PRIVATE -Wall -Wextra -Wpedantic)
add_executable(cubutterfly_c_api_tests tests/test_c_api.c)
target_link_libraries(cubutterfly_c_api_tests PRIVATE cuntt m)
target_compile_options(cubutterfly_c_api_tests PRIVATE -Wall -Wextra -Wpedantic)
add_executable(cubutterfly_semantic_contract_tests tests/test_semantic_contract.cpp src/semantic_contract.cpp)
target_include_directories(cubutterfly_semantic_contract_tests PRIVATE src include)
target_link_libraries(cubutterfly_semantic_contract_tests PRIVATE CUDA::cudart nlohmann_json::nlohmann_json)
target_compile_features(cubutterfly_semantic_contract_tests PRIVATE cxx_std_17)
target_compile_options(cubutterfly_semantic_contract_tests PRIVATE -Wall -Wextra -UNDEBUG)
add_executable(cubutterfly_cpp_descriptor_tests tests/test_cpp_descriptor.cpp)
target_link_libraries(cubutterfly_cpp_descriptor_tests PRIVATE cuntt)
target_compile_options(cubutterfly_cpp_descriptor_tests PRIVATE -Wall -Wextra -UNDEBUG)
add_executable(cubutterfly_logical_order_tests tests/test_logical_order.cu)
target_include_directories(cubutterfly_logical_order_tests PRIVATE src)
target_link_libraries(cubutterfly_logical_order_tests PRIVATE cuntt)
if(CUBUTTERFLY_ENABLE_CUFFTDX)
    target_compile_definitions(cubutterfly_tests PRIVATE CUBUTTERFLY_TEST_CUFFTDX=1)
endif()
if(CUBUTTERFLY_ENABLE_TURBOFFT)
    target_compile_definitions(cubutterfly_tests PRIVATE CUBUTTERFLY_TEST_TURBOFFT=1)
endif()

add_executable(cuntt_hardware_microbench apps/hardware_microbench.cu)
target_link_libraries(cuntt_hardware_microbench PRIVATE CUDA::cudart stdc++)
target_compile_options(cuntt_hardware_microbench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)

if(CUBUTTERFLY_ENABLE_VKFFT OR CUBUTTERFLY_ENABLE_XD_VKFFT)
    set(CUBUTTERFLY_VKFFT_INCLUDE "${CUBUTTERFLY_VKFFT_ROOT}/vkFFT")
    if(NOT EXISTS "${CUBUTTERFLY_VKFFT_INCLUDE}/vkFFT.h")
        message(FATAL_ERROR
            "VkFFT headers were not found under ${CUBUTTERFLY_VKFFT_ROOT}. "
            "Run scripts/install_vkfft.sh or set CUBUTTERFLY_VKFFT_ROOT.")
    endif()

endif()
if(CUBUTTERFLY_ENABLE_XD_VKFFT)
    target_sources(cuntt PRIVATE src/xd_vkfft_axis.cpp)
    target_include_directories(cuntt SYSTEM PRIVATE "${CUBUTTERFLY_VKFFT_INCLUDE}")
    set_source_files_properties(src/xd_vkfft_axis.cpp PROPERTIES COMPILE_DEFINITIONS "VKFFT_BACKEND=1")
    target_link_libraries(cuntt PRIVATE CUDA::cudart CUDA::cuda_driver CUDA::nvrtc)
else()
    target_sources(cuntt PRIVATE src/xd_vkfft_axis_stub.cpp)
endif()
if(CUBUTTERFLY_ENABLE_VKFFT)
    add_executable(vkfft_bench apps/vkfft_bench.cu)
    target_include_directories(vkfft_bench SYSTEM PRIVATE "${CUBUTTERFLY_VKFFT_INCLUDE}")
    target_compile_definitions(vkfft_bench PRIVATE VKFFT_BACKEND=1)
    target_link_libraries(vkfft_bench PRIVATE CUDA::cudart CUDA::cuda_driver CUDA::nvrtc CUDA::cufft)
    target_compile_options(vkfft_bench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)
    # The xD benchmark bridge accepts caller-owned buffers and streams so all
    # libraries can be checked and timed under the same CUDA Graph protocol.
    add_library(vkfft_xd_bridge SHARED apps/vkfft_xd_bridge.cu)
    target_include_directories(vkfft_xd_bridge SYSTEM PRIVATE "${CUBUTTERFLY_VKFFT_INCLUDE}")
    target_compile_definitions(vkfft_xd_bridge PRIVATE VKFFT_BACKEND=1)
    target_link_libraries(vkfft_xd_bridge PRIVATE CUDA::cudart CUDA::cuda_driver CUDA::nvrtc)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(cubutterfly_fft_radix4_codelet_tests tests/test_fft_radix4_codelet.cpp)
    add_test(NAME fft_radix4_codelet_cpu COMMAND cubutterfly_fft_radix4_codelet_tests)
    set_tests_properties(fft_radix4_codelet_cpu PROPERTIES LABELS "cpu;fft")
    add_executable(cubutterfly_xd_radix4_pair_capabilities_tests tests/test_xd_radix4_pair_capabilities.cpp)
    target_link_libraries(cubutterfly_xd_radix4_pair_capabilities_tests PRIVATE cuntt CUDA::cudart)
    add_test(NAME xd_radix4_pair_capabilities_cpu COMMAND cubutterfly_xd_radix4_pair_capabilities_tests)
    set_tests_properties(xd_radix4_pair_capabilities_cpu PROPERTIES LABELS "cpu;fft")
    if(CUBUTTERFLY_ENABLE_XD_CUFFTDX)
        add_executable(cubutterfly_xd_axis_resource_tests tests/test_xd_axis_resources.cu)
        target_include_directories(cubutterfly_xd_axis_resource_tests SYSTEM PRIVATE
            "${CUBUTTERFLY_MATHDX_ROOT}/include")
        target_compile_definitions(cubutterfly_xd_axis_resource_tests PRIVATE
            CUBUTTERFLY_XD_CUFFTDX_SM=${CUBUTTERFLY_XD_SM})
        target_link_libraries(cubutterfly_xd_axis_resource_tests PRIVATE CUDA::cudart)
        add_test(NAME xd_axis_resource_descriptors_cpu COMMAND cubutterfly_xd_axis_resource_tests)
        set_tests_properties(xd_axis_resource_descriptors_cpu PROPERTIES LABELS "cpu;xd")
    endif()
    add_executable(cubutterfly_xd_fft_layout_tests tests/test_xd_fft_layout.cu)
    target_link_libraries(cubutterfly_xd_fft_layout_tests PRIVATE cuntt CUDA::cufft CUDA::cudart)
    add_test(NAME xd_fft_layout_correctness COMMAND cubutterfly_xd_fft_layout_tests)
    add_test(NAME xd_fft_fused_axis_correctness COMMAND cubutterfly_xd_fft_layout_tests --fused)
    add_test(NAME xd_fft_wide_axis_correctness COMMAND cubutterfly_xd_fft_layout_tests --wide)
    add_test(NAME xd_fft_pair_correctness COMMAND cubutterfly_xd_fft_layout_tests --pair)
    add_test(NAME xd_fft_register_axis_correctness COMMAND cubutterfly_xd_fft_layout_tests --register-axis)
    add_test(NAME xd_fft_cooperative_pair_correctness COMMAND cubutterfly_xd_fft_layout_tests --cooperative-pair)
    add_test(NAME xd_fft_register_l8_correctness COMMAND cubutterfly_xd_fft_layout_tests --register-axis-l8)
    add_test(NAME xd_fft_register_l16_correctness COMMAND cubutterfly_xd_fft_layout_tests --register-axis-l16)
    add_test(NAME xd_fft_pair_layout1_correctness COMMAND cubutterfly_xd_fft_layout_tests --cooperative-pair-layout1)
    add_test(NAME xd_fft_batch_tile_correctness COMMAND cubutterfly_xd_fft_layout_tests --batch-tile)
    add_test(NAME xd_fft_axis_block_caps_correctness COMMAND cubutterfly_xd_fft_layout_tests --axis-block-caps)
    add_test(NAME xd_fft_axis_exchange_correctness COMMAND cubutterfly_xd_fft_layout_tests --axis-exchange)
    add_test(NAME xd_fft_axis_resources_correctness COMMAND cubutterfly_xd_fft_layout_tests --axis-resources)
    add_test(NAME xd_fft_local_core_correctness COMMAND cubutterfly_xd_fft_layout_tests --local-core)
    set_tests_properties(xd_fft_fused_axis_correctness xd_fft_wide_axis_correctness
        xd_fft_pair_correctness xd_fft_register_axis_correctness
        xd_fft_cooperative_pair_correctness xd_fft_register_l8_correctness
        xd_fft_register_l16_correctness xd_fft_pair_layout1_correctness
        xd_fft_batch_tile_correctness xd_fft_axis_block_caps_correctness
        xd_fft_axis_exchange_correctness xd_fft_axis_resources_correctness xd_fft_local_core_correctness
        PROPERTIES SKIP_RETURN_CODE 77 LABELS "gpu;xd")
    add_executable(cubutterfly_rank3_permute_tests tests/test_rank3_permute.cu)
    target_include_directories(cubutterfly_rank3_permute_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(cubutterfly_rank3_permute_tests PRIVATE cuntt CUDA::cudart)
    add_test(NAME rank3_permute_correctness COMMAND cubutterfly_rank3_permute_tests)
    add_test(NAME semantic_contract_cpu COMMAND cubutterfly_semantic_contract_tests)
    add_test(NAME cpp_descriptor_cpu COMMAND cubutterfly_cpp_descriptor_tests)
    add_test(NAME logical_order_correctness COMMAND cubutterfly_logical_order_tests)
    set_tests_properties(semantic_contract_cpu cpp_descriptor_cpu PROPERTIES LABELS "cpu;semantics")
    set_tests_properties(logical_order_correctness PROPERTIES LABELS "gpu;semantics" SKIP_RETURN_CODE 77)
    set_tests_properties(xd_fft_layout_correctness rank3_permute_correctness PROPERTIES LABELS "gpu;xd")
    add_executable(cubutterfly_ntt_stage_probe_tests tests/test_ntt_stage_probe.cpp)
    target_link_libraries(cubutterfly_ntt_stage_probe_tests PRIVATE cuntt)
    target_include_directories(cubutterfly_ntt_stage_probe_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    add_test(NAME ntt_stage_probe COMMAND cubutterfly_ntt_stage_probe_tests)
    add_test(NAME native_stage_probe COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_stage_probe.py --binary $<TARGET_FILE:cubutterfly_stage_microbench>)
    set_tests_properties(ntt_stage_probe native_stage_probe PROPERTIES LABELS "gpu;stage-calibration")
    add_executable(cubutterfly_mapping_tests tests/test_mapping_records.cpp)
    target_link_libraries(cubutterfly_mapping_tests PRIVATE cuntt)
    add_test(NAME mapping_record_roundtrip COMMAND cubutterfly_mapping_tests)
    set_tests_properties(mapping_record_roundtrip PROPERTIES
        ENVIRONMENT "CUBUTTERFLY_COMPILE_MODE=research")
    add_executable(cubutterfly_ntt_template_candidate_tests tests/test_ntt_template_candidates.cpp)
    target_include_directories(cubutterfly_ntt_template_candidate_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(cubutterfly_ntt_template_candidate_tests PRIVATE cuntt)
    add_test(NAME ntt_template_candidate_contract COMMAND cubutterfly_ntt_template_candidate_tests)
    add_executable(cubutterfly_shared_organization_tests tests/test_shared_organization_candidates.cpp)
    target_include_directories(cubutterfly_shared_organization_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    add_test(NAME shared_organization_contract COMMAND cubutterfly_shared_organization_tests)
    set_tests_properties(shared_organization_contract PROPERTIES LABELS "cpu;organization")
    add_executable(cubutterfly_xd_resident_contract_tests tests/test_xd_resident_contract.cpp)
    target_include_directories(cubutterfly_xd_resident_contract_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(cubutterfly_xd_resident_contract_tests PRIVATE cuntt nlohmann_json::nlohmann_json)
    add_test(NAME xd_resident_rejection_contract COMMAND cubutterfly_xd_resident_contract_tests)
    set_tests_properties(xd_resident_rejection_contract PROPERTIES LABELS "cpu;organization")
    add_executable(cubutterfly_shared_mechanism_tests tests/test_shared_mechanisms.cpp)
    target_link_libraries(cubutterfly_shared_mechanism_tests PRIVATE cuntt)
    add_test(NAME shared_mechanism_correctness COMMAND cubutterfly_shared_mechanism_tests)
    set_tests_properties(shared_mechanism_correctness PROPERTIES LABELS "gpu;organization"
        ENVIRONMENT "CUBUTTERFLY_COMPILE_MODE=precompiled")
    add_test(NAME shared_resident_mechanism_correctness COMMAND cubutterfly_shared_mechanism_tests --resident-only)
    set_tests_properties(shared_resident_mechanism_correctness PROPERTIES LABELS "gpu;organization"
        ENVIRONMENT "CUBUTTERFLY_COMPILE_MODE=research")
    add_test(NAME shared_wide_mechanism_correctness COMMAND cubutterfly_shared_mechanism_tests --wide)
    set_tests_properties(shared_wide_mechanism_correctness PROPERTIES LABELS "gpu;organization"
        ENVIRONMENT "CUBUTTERFLY_COMPILE_MODE=precompiled")
    add_test(NAME shared_graph_mechanism_correctness COMMAND cubutterfly_shared_mechanism_tests --graph)
    set_tests_properties(shared_graph_mechanism_correctness PROPERTIES LABELS "gpu;organization;graph"
        ENVIRONMENT "CUBUTTERFLY_COMPILE_MODE=research")
    add_executable(cubutterfly_public_planner_tests tests/test_public_planner.cpp)
    target_link_libraries(cubutterfly_public_planner_tests PRIVATE cuntt nlohmann_json::nlohmann_json)
    add_test(NAME public_planner_consistency COMMAND cubutterfly_public_planner_tests)
    add_executable(cubutterfly_platform_tuning_tests tests/test_platform_tuning.cpp)
    target_include_directories(cubutterfly_platform_tuning_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(cubutterfly_platform_tuning_tests PRIVATE cuntt nlohmann_json::nlohmann_json)
    add_test(NAME platform_tuning_contract COMMAND cubutterfly_platform_tuning_tests)
    add_test(NAME hardware_registry_contract
        COMMAND "${Python3_EXECUTABLE}" -m pytest -q "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_hardware_registry.py")
    add_test(NAME fft_tile_layout_contract COMMAND cubutterfly_fft_tile_layout_tests)
    add_test(NAME mixed_dataflow COMMAND cubutterfly_mixed_dataflow_tests)
    add_executable(cubutterfly_online_reorder_grouping_tests tests/test_online_reorder_grouping.cu)
    target_link_libraries(cubutterfly_online_reorder_grouping_tests PRIVATE cuntt)
    add_test(NAME online_reorder_grouping COMMAND cubutterfly_online_reorder_grouping_tests)
    add_test(NAME hardware_model_cpu COMMAND cubutterfly_hardware_model_tests)
    add_test(NAME static_ranking_cpu COMMAND "${CMAKE_COMMAND}" -E env
        "CUBUTTERFLY_TEST_PYTHON=${Python3_EXECUTABLE}" $<TARGET_FILE:cubutterfly_static_ranking_tests>
        "${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME cuntt_correctness COMMAND cuntt_tests)
    add_executable(cuntt_rns_tests tests/test_rns_ntt.cu)
    target_link_libraries(cuntt_rns_tests PRIVATE cuntt CUDA::cudart)
    target_compile_features(cuntt_rns_tests PRIVATE cxx_std_17 cuda_std_17)
    add_test(NAME cuntt_rns_correctness COMMAND cuntt_rns_tests)
    set_tests_properties(cuntt_rns_correctness PROPERTIES LABELS "gpu;ntt;rns")
    add_executable(cuntt_rns_tuning_tests tests/test_rns_tuning.cu)
    target_link_libraries(cuntt_rns_tuning_tests PRIVATE cuntt CUDA::cudart nlohmann_json::nlohmann_json)
    target_compile_features(cuntt_rns_tuning_tests PRIVATE cxx_std_17 cuda_std_17)
    add_test(NAME cuntt_rns_tuning_correctness COMMAND cuntt_rns_tuning_tests)
    set_tests_properties(cuntt_rns_tuning_correctness PROPERTIES LABELS "gpu;ntt;rns;cache")
    add_library(cuntt_rns_alloc_failure_inject SHARED tests/rns_allocation_failure_inject.cpp)
    target_link_libraries(cuntt_rns_alloc_failure_inject PRIVATE CUDA::cudart ${CMAKE_DL_LIBS})
    target_compile_features(cuntt_rns_alloc_failure_inject PRIVATE cxx_std_17)
    add_executable(cuntt_rns_mapping_failure_tests tests/test_rns_mapping_failure.cu)
    target_link_libraries(cuntt_rns_mapping_failure_tests PRIVATE cuntt CUDA::cudart ${CMAKE_DL_LIBS})
    target_compile_features(cuntt_rns_mapping_failure_tests PRIVATE cxx_std_17 cuda_std_17)
    set_target_properties(cuntt_rns_mapping_failure_tests PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
    add_test(NAME cuntt_rns_mapping_failure
        COMMAND ${CMAKE_COMMAND} -E env LD_PRELOAD=$<TARGET_FILE:cuntt_rns_alloc_failure_inject>
                $<TARGET_FILE:cuntt_rns_mapping_failure_tests>)
    set_tests_properties(cuntt_rns_mapping_failure PROPERTIES LABELS "gpu;ntt;rns")
    add_executable(cubutterfly_exchange_tests tests/test_butterfly_exchange.cpp)
    target_include_directories(cubutterfly_exchange_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_features(cubutterfly_exchange_tests PRIVATE cxx_std_17)
    add_test(NAME butterfly_exchange_geometry COMMAND cubutterfly_exchange_tests)
    add_test(NAME cubutterfly_correctness COMMAND cubutterfly_tests)
    add_test(NAME fft_stage_overlap_async_correctness COMMAND cubutterfly_stage_overlap_tests)
    if(CUBUTTERFLY_ENABLE_CUFFTDX)
        add_test(NAME fft_factor_pipeline_async_correctness COMMAND cubutterfly_factor_pipeline_tests)
    endif()
    add_test(NAME fft_shared_iterative_correctness
             COMMAND cubutterfly_bench --operator fft --precision fp32
                     --backend shared-iterative --fft-core scalar --compute-unit radix2
                     --local-stages 6 --tile-threads 256 --logN 12 --batch 16
                     --verify --warmup 1 --repeat 2 --csv)
    add_test(NAME cubutterfly_structured_reference COMMAND cubutterfly_structured_reference_tests)
    add_executable(cubutterfly_uint64_zeta_reference_tests tests/test_uint64_zeta_reference.cpp)
    target_link_libraries(cubutterfly_uint64_zeta_reference_tests PRIVATE cuntt)
    add_test(NAME cubutterfly_uint64_zeta_reference COMMAND cubutterfly_uint64_zeta_reference_tests)
    add_test(NAME cubutterfly_c_api COMMAND cubutterfly_c_api_tests)
    if(CUBUTTERFLY_ENABLE_VKFFT)
        add_test(NAME vkfft_in_place_correctness
                 COMMAND vkfft_bench --library vkfft --placement in-place --logN 8
                         --total-points 65536 --warmup 1 --repeat 1 --verify)
        add_test(NAME vkfft_out_of_place_correctness
                 COMMAND vkfft_bench --library vkfft --placement out-of-place --logN 8
                         --total-points 65536 --warmup 1 --repeat 1 --verify)
        add_test(NAME cufft_out_of_place_baseline_correctness
                 COMMAND vkfft_bench --library cufft --placement out-of-place --logN 8
                         --total-points 65536 --warmup 1 --repeat 1 --verify)
    endif()
    add_test(NAME fft_design_space
             COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_fft_design_space.py")
    add_test(NAME butterfly_design_space
             COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_butterfly_design_space.py")
    add_test(NAME design_point_generator
             COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_design_point_generator.py")
    add_test(NAME hybrid_dataflow_generator
             COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_hybrid_dataflow_generator.py")
    add_test(NAME hybrid_dataflow_schedule
             COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_hybrid_dataflow_schedule.py")
    add_test(NAME comprehensive_suite
             COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_comprehensive_suite.py")
    add_test(NAME scaling_suite
             COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_scaling_suite.py")
    add_test(NAME numeric_regime_study
             COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_numeric_regime_study.py")
    add_test(NAME mapping_selector
             COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_mapping_selector.py")
    add_test(NAME external_baseline_suite
             COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_external_baseline_suite.py")
    add_test(NAME scaling_ncu_analysis
             COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_scaling_ncu_analysis.py")
    add_test(NAME packet_streaming_v06_comparison
             COMMAND "${Python3_EXECUTABLE}" -m pytest -q
                     "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_packet_streaming_v06_comparison.py")
    add_test(NAME packet_polling_backoff
             COMMAND "${Python3_EXECUTABLE}" -m pytest -q
                     "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_packet_polling_backoff.py"
                     "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_packet_streaming_ncu.py")
    add_test(NAME packet_readiness_modes
             COMMAND "${Python3_EXECUTABLE}" -m pytest -q
                     "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_packet_readiness_modes.py"
                     "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_packet_readiness_modes_ncu.py")
    add_test(NAME packet_compute_layout
             COMMAND "${Python3_EXECUTABLE}" -m pytest -q
                     "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_packet_compute_layout.py"
                     "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_packet_compute_layout_ncu.py")
    add_test(NAME packet_warp_rows_weights
             COMMAND "${Python3_EXECUTABLE}" -m pytest -q
                     "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_packet_warp_rows_weights.py")
    add_test(NAME packet_folded_barriers
             COMMAND "${Python3_EXECUTABLE}" -m pytest -q
                     "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_packet_folded_barriers.py")
    add_test(NAME three_level_partitions
             COMMAND "${Python3_EXECUTABLE}" -m pytest -q
                     "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_three_level_partitions.py")
    add_test(NAME ntt_partition_space
             COMMAND "${Python3_EXECUTABLE}" -m pytest -q
                     "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_ntt_partition_space.py")
    add_test(NAME coefficient_supply_contract
             COMMAND "${Python3_EXECUTABLE}" -m pytest -q
                     "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_coefficient_supply_contract.py")
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_v08_comprehensive.py")
      add_test(NAME v08_comprehensive_analysis
             COMMAND "${Python3_EXECUTABLE}" -m pytest -q
                     "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_v08_comprehensive.py")
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_v08_selector_holdout.py")
      add_test(NAME v08_selector_holdout_analysis
             COMMAND "${Python3_EXECUTABLE}" -m pytest -q
                     "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_v08_selector_holdout.py")
    endif()
    add_test(NAME stage_count_space
             COMMAND "${Python3_EXECUTABLE}" -m pytest -q
                     "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_stage_count_space.py")
    add_test(NAME resident_execution_groups
             COMMAND "${Python3_EXECUTABLE}" -m pytest -q
                     "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_resident_execution_groups.py")
    add_test(NAME resident_execution_groups_ncu
             COMMAND "${Python3_EXECUTABLE}" -m pytest -q
                     "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_resident_execution_groups_ncu.py")
    add_test(NAME structured_2x2_ncu_analysis
             COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_structured_2x2_ncu.py")
    add_test(NAME structured_coefficient_policy_summary
             COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_structured_coefficient_policies.py")
    add_test(NAME residency_features
             COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_residency_features.py")
    add_test(NAME fft_vectorized_ncu_analysis
             COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_fft_vectorized_ncu.py")
    add_test(NAME fp64_fft_ncu_analysis
             COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_fp64_fft_ncu.py")
    add_test(NAME fp64_fft_robustness_analysis
             COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_fp64_fft_robustness.py")
    add_test(NAME fft_pipeline_generator
             COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_fft_pipeline_generator.py")
    add_test(NAME fft_pipeline_runner
             COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_fft_pipeline_runner.py")
    add_test(NAME fft_pipeline_model
             COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_fft_pipeline_model.py")
    add_test(NAME fft_dispatch_generator
             COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_fft_dispatch_generator.py")
    add_test(NAME installed_package_consumer
             COMMAND "${CMAKE_COMMAND}"
                     -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
                     -DBUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}
                     -DPYTHON_EXECUTABLE=${Python3_EXECUTABLE}
                     -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_install_package.cmake")
    add_test(NAME runtime_selector_generator
             COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_runtime_selector_generator.py")
    add_test(NAME application_profile_generator
             COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_application_profiles.py")
    if(CUBUTTERFLY_ENABLE_CUFFTDX)
        add_test(NAME fft_overlap_fp64_correctness
                 COMMAND cubutterfly_bench --operator fft --precision fp64 --backend online-reorder
                         --fft-core cufftdx-block --stage-partition 8,8 --prefix-threads 256 --prefix-ept 4
                         --suffix-threads 128 --suffix-ept 8 --cross-twiddle recurrence --shared-layout xor-swizzle
                         --logN 16 --batch 5 --stage-overlap --batch-tile-count 2
                         --inverse --placement in-place --warmup 1 --repeat 2 --verify)
        add_test(NAME fft_overlap_prefix_transpose_correctness
                 COMMAND cubutterfly_bench --operator fft --precision fp32 --backend online-reorder
                         --fft-core cufftdx-block --stage-partition 11,9 --prefix-threads 256 --prefix-ept 8
                         --suffix-threads 128 --suffix-ept 16 --direct-boundary prefix-tiled-transpose
                         --cross-twiddle recurrence --logN 20 --batch 5 --stage-overlap --batch-tile-count 2
                         --inverse --placement in-place --warmup 1 --repeat 2 --verify)
        add_test(NAME fft_overlap_suffix_strided_correctness
                 COMMAND cubutterfly_bench --operator fft --precision fp32 --backend online-reorder
                         --fft-core cufftdx-block --stage-partition 8,12 --prefix-threads 256 --prefix-ept 16
                         --suffix-threads 256 --suffix-ept 16 --direct-boundary tiled-transpose
                         --logN 20 --batch 5 --stage-overlap --batch-tile-count 2 --element-stride 2
                         --batch-stride 2097159 --inverse --placement in-place --warmup 1 --repeat 2 --verify)
        add_test(NAME fft_fp64_cufftdx_block_correctness
                 COMMAND cubutterfly_bench --operator fft --precision fp64 --backend temporal-tile
                         --fft-core cufftdx-block --tile-threads 32 --normalization none
                         --logN 8 --batch 8 --warmup 1 --repeat 1 --verify)
        add_test(NAME fft_fp64_cufftdx_online_correctness
                 COMMAND cubutterfly_bench --operator fft --precision fp64 --backend online-reorder
                         --fft-core cufftdx-block --local-stages 8
                         --prefix-threads 256 --prefix-ept 4 --suffix-threads 128 --suffix-ept 8
                         --normalization none --logN 16 --batch 2 --warmup 1 --repeat 1 --verify)
        add_test(NAME fft_fp64_cufftdx_online_inverse_correctness
                 COMMAND cubutterfly_bench --operator fft --precision fp64 --backend online-reorder
                         --fft-core cufftdx-block --local-stages 8
                         --prefix-threads 256 --prefix-ept 4 --suffix-threads 128 --suffix-ept 8
                         --inverse --normalization inverse --logN 16 --batch 2
                         --warmup 1 --repeat 1 --verify)
        add_test(NAME fft_fp64_cufftdx_recurrence_correctness
                 COMMAND cubutterfly_bench --operator fft --precision fp64 --backend online-reorder
                         --fft-core cufftdx-block --local-stages 8
                         --prefix-threads 256 --prefix-ept 4 --suffix-threads 128 --suffix-ept 8
                         --cross-twiddle recurrence --normalization none --logN 16 --batch 2
                         --warmup 1 --repeat 1 --verify)
        add_test(NAME fft_fp64_cufftdx_recurrence_inverse_correctness
                 COMMAND cubutterfly_bench --operator fft --precision fp64 --backend online-reorder
                         --fft-core cufftdx-block --local-stages 8
                         --prefix-threads 256 --prefix-ept 4 --suffix-threads 128 --suffix-ept 8
                         --cross-twiddle recurrence --inverse --normalization inverse
                         --logN 16 --batch 2 --warmup 1 --repeat 1 --verify)
        add_test(NAME fft_fp64_cufftdx_xor_swizzle_correctness
                 COMMAND cubutterfly_bench --operator fft --precision fp64 --backend online-reorder
                         --fft-core cufftdx-block --local-stages 8
                         --prefix-threads 256 --prefix-ept 4 --suffix-threads 128 --suffix-ept 8
                         --cross-twiddle recurrence --shared-layout xor-swizzle
                         --normalization none --logN 16 --batch 2 --warmup 1 --repeat 1 --verify)
        add_test(NAME fft_fp64_cufftdx_xor_swizzle_inverse_correctness
                 COMMAND cubutterfly_bench --operator fft --precision fp64 --backend online-reorder
                         --fft-core cufftdx-block --local-stages 8
                         --prefix-threads 256 --prefix-ept 4 --suffix-threads 128 --suffix-ept 8
                         --cross-twiddle recurrence --shared-layout xor-swizzle --inverse
                         --normalization inverse --logN 16 --batch 2 --warmup 1 --repeat 1 --verify)
        foreach(fp64_shape IN ITEMS "14;7" "15;8" "17;8" "18;9")
            list(GET fp64_shape 0 fp64_log_n)
            list(GET fp64_shape 1 fp64_local_stages)
            add_test(NAME fft_fp64_cufftdx_xor_logN${fp64_log_n}_correctness
                     COMMAND cubutterfly_bench --operator fft --precision fp64 --backend online-reorder
                             --fft-core cufftdx-block --local-stages ${fp64_local_stages}
                             --prefix-threads 256 --prefix-ept 4 --suffix-threads 128 --suffix-ept 8
                             --cross-twiddle recurrence --shared-layout xor-swizzle
                             --normalization none --logN ${fp64_log_n} --batch 2
                             --warmup 1 --repeat 1 --verify)
        endforeach()
        foreach(fp64_shape IN ITEMS "14;7" "18;9")
            list(GET fp64_shape 0 fp64_log_n)
            list(GET fp64_shape 1 fp64_local_stages)
            add_test(NAME fft_fp64_cufftdx_xor_logN${fp64_log_n}_inverse_correctness
                     COMMAND cubutterfly_bench --operator fft --precision fp64 --backend online-reorder
                             --fft-core cufftdx-block --local-stages ${fp64_local_stages}
                             --prefix-threads 256 --prefix-ept 4 --suffix-threads 128 --suffix-ept 8
                             --cross-twiddle recurrence --shared-layout xor-swizzle --inverse
                             --normalization inverse --logN ${fp64_log_n} --batch 2
                             --warmup 1 --repeat 1 --verify)
        endforeach()
        add_test(NAME fft_auto_dispatch_correctness
                 COMMAND cubutterfly_bench --auto-select --operator fft --precision fp32
                         --placement in-place --normalization none --logN 18 --batch 2
                         --warmup 1 --repeat 1 --verify)
        add_test(NAME fft_mixed_direct_correctness
                 COMMAND cubutterfly_bench --operator fft --precision fp32 --backend online-reorder
                         --fft-core cufftdx-block --stage-partition 9,11 --prefix-threads 256 --prefix-ept 8
                         --suffix-threads 256 --suffix-ept 8 --cross-twiddle recurrence
                         --placement in-place --normalization none --logN 20 --batch 1
                         --warmup 1 --repeat 1 --verify)
        add_test(NAME fft_tiled_transpose_correctness
                 COMMAND cubutterfly_bench --operator fft --precision fp32 --backend online-reorder
                         --fft-core cufftdx-block --local-stages 9 --prefix-threads 256 --prefix-ept 8
                         --suffix-threads 256 --suffix-ept 8 --cross-twiddle recurrence
                         --direct-boundary tiled-transpose --placement in-place --normalization none
                         --logN 20 --batch 1 --warmup 1 --repeat 1 --verify)
        add_test(NAME fft_prefix_tiled_transpose_correctness
                 COMMAND cubutterfly_bench --operator fft --precision fp32 --backend online-reorder
                         --fft-core cufftdx-block --stage-partition 11,9 --prefix-threads 256 --prefix-ept 8
                         --suffix-threads 128 --suffix-ept 16 --cross-twiddle recurrence
                         --direct-boundary prefix-tiled-transpose --placement in-place --normalization none
                         --logN 20 --batch 1 --warmup 1 --repeat 1 --verify)
        foreach(prefix_threads IN ITEMS 128 256 512)
            add_test(NAME fft_fp32_xor_prefix_${prefix_threads}_correctness
                     COMMAND cubutterfly_bench --operator fft --precision fp32 --backend online-reorder
                             --fft-core cufftdx-block --stage-partition 8,12
                             --prefix-threads ${prefix_threads} --prefix-ept 16
                             --suffix-threads 256 --suffix-ept 16 --shared-layout xor-swizzle
                             --direct-boundary tiled-transpose --placement in-place
                             --logN 20 --batch 1 --warmup 1 --repeat 1 --verify)
        endforeach()
        add_test(NAME fft_multisegment_d3_correctness
                 COMMAND cubutterfly_bench --operator fft --precision fp32 --backend online-reorder
                         --fft-core cufftdx-block --stage-partition 6,6,8
                         --segment-threads 256,256,256 --segment-ept 8,8,8
                         --boundary-twiddle recurrence,recurrence
                         --placement in-place --normalization none --logN 20 --batch 1
                         --warmup 1 --repeat 1 --verify)
        add_test(NAME fft_multisegment_d4_correctness
                 COMMAND cubutterfly_bench --operator fft --precision fp32 --backend online-reorder
                         --fft-core cufftdx-block --stage-partition 5,5,5,5
                         --segment-threads 256,256,256,256 --segment-ept 8,8,8,8
                         --boundary-twiddle recurrence,table,recurrence
                         --placement in-place --normalization none --logN 20 --batch 1
                         --warmup 1 --repeat 1 --verify)
        add_test(NAME fft_fused_boundary_d3_correctness
                 COMMAND cubutterfly_bench --operator fft --precision fp32 --backend online-reorder
                         --fft-core cufftdx-block --stage-partition 7,3,10
                         --segment-threads 128,128,256 --segment-ept 8,8,16
                         --boundary-twiddle table,recurrence
                         --boundary-residency fused,global-scratch
                         --group-threads 512,512 --group-ept 8,8
                         --placement in-place --normalization none --logN 20 --batch 1
                         --warmup 1 --repeat 1 --verify)
        add_test(NAME fft_fused_boundary_d4_correctness
                 COMMAND cubutterfly_bench --operator fft --precision fp32 --backend online-reorder
                         --fft-core cufftdx-block --stage-partition 5,5,5,5
                         --segment-threads 128,128,128,128 --segment-ept 8,8,8,8
                         --boundary-twiddle table,recurrence,table
                         --boundary-residency fused,global-scratch,fused
                         --group-threads 512,512 --group-ept 8,8
                         --placement in-place --normalization none --logN 20 --batch 1
                         --warmup 1 --repeat 1 --verify)
        add_test(NAME fft_multisegment_inverse_correctness
                 COMMAND cubutterfly_bench --operator fft --precision fp32 --backend online-reorder
                         --fft-core cufftdx-block --stage-partition 6,6,6
                         --segment-threads 256,256,256 --segment-ept 8,8,8
                         --boundary-twiddle table,recurrence --inverse
                         --placement out-of-place --normalization inverse --logN 18 --batch 1
                         --warmup 1 --repeat 1 --verify)
        add_test(NAME fft_multisegment_strided_correctness
                 COMMAND cubutterfly_bench --operator fft --precision fp32 --backend online-reorder
                         --fft-core cufftdx-block --stage-partition 4,4,4
                         --segment-threads 128,128,128 --segment-ept 4,4,4
                         --boundary-twiddle recurrence,table --element-stride 2 --batch-stride 8192
                         --placement out-of-place --normalization none --logN 12 --batch 2
                         --warmup 1 --repeat 1 --verify)
    endif()
endif()

file(RELATIVE_PATH CUBUTTERFLY_INSTALL_BIN_TO_LIB
    "${CMAKE_INSTALL_FULL_BINDIR}" "${CMAKE_INSTALL_FULL_LIBDIR}")
set_target_properties(cuntt PROPERTIES
    INSTALL_RPATH "$ORIGIN"
    INSTALL_RPATH_USE_LINK_PATH TRUE)
set_target_properties(cubutterfly_bench cubutterfly_graph_bench cuntt_bench cuntt_hardware_microbench cubutterfly_plan_probe
    cubutterfly_schedule_microbench cubutterfly_stage_microbench cubutterfly_pipeline_microbench
    PROPERTIES INSTALL_RPATH "$ORIGIN/${CUBUTTERFLY_INSTALL_BIN_TO_LIB}"
               INSTALL_RPATH_USE_LINK_PATH TRUE)

install(TARGETS cuntt
    EXPORT cuButterflyTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
install(DIRECTORY include/cuntt include/cubutterfly DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(DIRECTORY config/platform_tuning DESTINATION "${CMAKE_INSTALL_DATADIR}/cuButterfly")
configure_file(cmake/install_hardware_profiles.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/install_hardware_profiles.cmake" @ONLY)
install(SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/install_hardware_profiles.cmake")
install(PROGRAMS
    scripts/calibrate_hardware.py
    scripts/static_hardware_profile.py
    scripts/rank_static_candidates.py
    scripts/initialize_hardware_profile.py
    scripts/check_hardware_profile.py
    scripts/calibrate_local_hardware.py
    scripts/calibrate_composition.py
    scripts/fit_local_cost_model.py
    scripts/stage_cost_model.py
    scripts/validate_stage_service_profile.py
    scripts/tune_hardware.py
    scripts/hardware_registry.py
    scripts/verify_local_selector.py
    scripts/run_fft_acceptance.py
    scripts/benchmark_stage_overlap.py
    DESTINATION "${CMAKE_INSTALL_BINDIR}")
install(FILES scripts/calibration_seeds.py scripts/calibration_space.py scripts/local_selector_data.py scripts/evolutionary_search.py scripts/schedule_cost_model.py
    scripts/benchmark_semantics.py
    scripts/shared_work_geometry.py
    scripts/calibration_cpu_cache.py
    scripts/resident_mapping.py scripts/cufftdx_capabilities.py scripts/cufftdx_low_module.py
    scripts/research_compile_requests.py scripts/jit_calibration_identity.py scripts/semantic_compile_broker.py
    scripts/stage_service_model.py scripts/stage_service_calibration.py
    scripts/stage_schedule_native.py scripts/stage_schedule_native.cpp
    scripts/stage_acquisition_order.py scripts/stage_checkpoint_import.py
    scripts/stage_composition.py
    scripts/composition_inventory.py
    scripts/pipeline_schedule_model.py
    scripts/run_comprehensive_suite.py DESTINATION "${CMAKE_INSTALL_BINDIR}")
install(FILES benchmark/__init__.py benchmark/runner_registry.py
    DESTINATION "${CMAKE_INSTALL_BINDIR}/benchmark")
install(FILES config/install_search_workloads.json
    DESTINATION "${CMAKE_INSTALL_DATADIR}/cuButterfly/config")
install(DIRECTORY include/ DESTINATION "${CMAKE_INSTALL_DATADIR}/cuButterfly/templates/include")
install(FILES src/fft_register_tile.cuh src/fft_fixed_roots.cuh src/fft_radix4_codelet.cuh src/fft_thread_codelet.cuh src/fft_grouped_suffix.cuh src/fft_native_block.cuh src/fft_factor_streamed.cuh src/fft_factor_native.cuh src/shared_iterative.cuh src/shared_resident.cuh
    src/operator_traits.cuh src/ntt_traits.cuh src/cufftdx_low_block.cuh
    DESTINATION "${CMAKE_INSTALL_DATADIR}/cuButterfly/templates/src")
if(CUBUTTERFLY_ENABLE_CUFFTDX)
    install(DIRECTORY "${CUBUTTERFLY_MATHDX_ROOT}/include/"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/cuButterfly/templates/mathdx/include")
    file(GLOB CUBUTTERFLY_MATHDX_LICENSES "${CUBUTTERFLY_MATHDX_ROOT}/LICENSE*"
        "${CUBUTTERFLY_MATHDX_ROOT}/../../nvidia_mathdx-*.dist-info/LICENSE*")
    install(FILES ${CUBUTTERFLY_MATHDX_LICENSES} DESTINATION "${CMAKE_INSTALL_DATADIR}/cuButterfly/templates/mathdx")
endif()
install(PROGRAMS scripts/compile_module.py
    DESTINATION "${CMAKE_INSTALL_DATADIR}/cuButterfly/templates/scripts")
install(FILES scripts/resident_mapping.py scripts/cufftdx_capabilities.py scripts/cufftdx_low_module.py
    DESTINATION "${CMAKE_INSTALL_DATADIR}/cuButterfly/templates/scripts")
install(TARGETS cubutterfly_bench cubutterfly_graph_bench cuntt_bench cuntt_hardware_microbench cubutterfly_plan_probe cubutterfly_schedule_microbench cubutterfly_stage_microbench cubutterfly_pipeline_microbench
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
install(FILES configs/hardware/gpu_profile_template.json
    DESTINATION "${CMAKE_INSTALL_DATADIR}/cuButterfly/hardware")

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cuButterflyConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cuButterflyConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/cuButterfly")
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/cuButterflyConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion)

install(EXPORT cuButterflyTargets
    FILE cuButterflyTargets.cmake
    NAMESPACE cuButterfly::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/cuButterfly")
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/cuButterflyConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/cuButterflyConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/cuButterfly")
