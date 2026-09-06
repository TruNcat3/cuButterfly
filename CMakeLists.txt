cmake_minimum_required(VERSION 3.20)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 CACHE STRING "CUDA architectures to compile for")
endif()

project(cuButterfly VERSION 0.8.0 LANGUAGES C CXX CUDA)

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

option(CUBUTTERFLY_ENABLE_VKFFT "Build the optional VkFFT comparison benchmark" OFF)
set(CUBUTTERFLY_VKFFT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/external/VkFFT" CACHE PATH
    "Path to a VkFFT source checkout")
option(CUBUTTERFLY_ENABLE_CUFFTDX "Enable the optional cuFFTDx block FFT processing unit" OFF)
set(CUBUTTERFLY_MATHDX_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/external/mathdx/nvidia/mathdx" CACHE PATH
    "Path to an NVIDIA MathDx package")
option(CUBUTTERFLY_ENABLE_TURBOFFT "Enable the optional TurboFFT generated processing unit" OFF)
set(CUBUTTERFLY_TURBOFFT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/external/TurboFFT/TurboFFT" CACHE PATH
    "Path to a TurboFFT source checkout")

set(CUNTT_DESIGN_SPEC "${CMAKE_CURRENT_SOURCE_DIR}/config/v100_design_points.json" CACHE FILEPATH
    "JSON file selecting generated cuButterfly design points")
set(CUBUTTERFLY_FFT_SPACE_SPEC "${CMAKE_CURRENT_SOURCE_DIR}/config/fft_architecture_space.json" CACHE FILEPATH
    "Canonical two-dimensional FFT architecture-space specification")
set(CUBUTTERFLY_ARCHITECTURE_SPACE_SPEC "${CMAKE_CURRENT_SOURCE_DIR}/config/butterfly_architecture_space.json" CACHE FILEPATH
    "Canonical operator-independent butterfly architecture-space specification")
set(CUBUTTERFLY_FFT_CODEGEN_SPEC "${CMAKE_CURRENT_SOURCE_DIR}/config/v100_fft_codegen.json" CACHE FILEPATH
    "Selected FFT template points compiled into this build")
set(CUBUTTERFLY_FFT_PIPELINE_SPEC "${CMAKE_CURRENT_SOURCE_DIR}/config/v100_fft_pipeline.json" CACHE FILEPATH
    "FFT processing-unit and mapping search specification")
set(CUBUTTERFLY_FFT_PIPELINE_CANDIDATES "${CMAKE_CURRENT_SOURCE_DIR}/config/v100_fft_pipeline_candidates.json" CACHE FILEPATH
    "Generated FFT pipeline candidate manifest")
set(CUBUTTERFLY_FFT_DISPATCH_SPEC "${CMAKE_CURRENT_SOURCE_DIR}/config/v100_fft_dispatch.json" CACHE FILEPATH
    "Measured FFT mapping dispatch table")
set(CUNTT_HYBRID_DATAFLOW_SPEC "${CMAKE_CURRENT_SOURCE_DIR}/config/v100_hybrid_dataflow.json" CACHE FILEPATH
    "Generated HybridDataflow NTT design points")
set(CUBUTTERFLY_RUNTIME_SELECTOR_SUMMARY "${CMAKE_CURRENT_SOURCE_DIR}/results/v100_scaling_full_summary.csv" CACHE FILEPATH
    "Measured scaling summary used to generate runtime selector latency anchors")
set(CUBUTTERFLY_FFT14_DIRECT_SELECTOR_SUMMARY "${CMAKE_CURRENT_SOURCE_DIR}/results/fft14_dispatch_revalidation/raw.csv" CACHE FILEPATH
    "Optional focused logN=14 dispatch CSV used to refresh direct-core latency anchors")
set(CUBUTTERFLY_V08_NTT_SELECTOR_SUMMARY "${CMAKE_CURRENT_SOURCE_DIR}/results/v08_comprehensive_full/summary.csv" CACHE FILEPATH
    "Measured v0.8 NTT matrix used to generate numeric/length selector anchors")
set(CUBUTTERFLY_CONFIRMED_SELECTION_SPEC "${CMAKE_CURRENT_SOURCE_DIR}/config/v100_v08_confirmed_selection.json" CACHE FILEPATH
    "Exact V100 resident butterfly selections confirmed with repeated measurements")
set(CUNTT_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
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

add_library(cuntt STATIC
    src/bluestein_fft.cu
    src/bluestein_ntt.cu
    src/c_api.cpp
    src/direct_fft.cu
    src/butterfly.cu
    src/butterfly_reference.cpp
    src/general_kernels.cu
    src/butterfly_performance_model.cpp
    src/mapping_selector.cpp
    src/ntt.cu
    src/reference.cpp)
target_sources(cuntt PRIVATE ${CUNTT_GENERATED_SOURCES})

if(CUBUTTERFLY_ENABLE_CUFFTDX)
    if(NOT EXISTS "${CUBUTTERFLY_MATHDX_ROOT}/include/cufftdx.hpp")
        message(FATAL_ERROR
            "cuFFTDx headers were not found under ${CUBUTTERFLY_MATHDX_ROOT}. "
            "Run scripts/install_cufftdx.sh or set CUBUTTERFLY_MATHDX_ROOT.")
    endif()
    target_sources(cuntt PRIVATE src/cufftdx_fft.cu src/cufftdx_fft_fp64.cu)
    target_include_directories(cuntt SYSTEM PRIVATE "${CUBUTTERFLY_MATHDX_ROOT}/include")
    target_compile_definitions(cuntt PRIVATE CUBUTTERFLY_HAS_CUFFTDX=1)
else()
    target_sources(cuntt PRIVATE src/cufftdx_fft_stub.cpp)
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

if(CUBUTTERFLY_ENABLE_CUFFTDX)
    add_executable(cufftdx_role_codelet_bench apps/cufftdx_role_codelet_bench.cpp)
    target_link_libraries(cufftdx_role_codelet_bench PRIVATE cuntt)
    target_include_directories(cufftdx_role_codelet_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_options(cufftdx_role_codelet_bench PRIVATE -Wall -Wextra -Wpedantic)

    add_executable(cufftdx_role_pipeline64_bench apps/cufftdx_role_pipeline64_bench.cpp)
    target_link_libraries(cufftdx_role_pipeline64_bench PRIVATE cuntt)
    target_include_directories(cufftdx_role_pipeline64_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_options(cufftdx_role_pipeline64_bench PRIVATE -Wall -Wextra -Wpedantic)
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

add_executable(cubutterfly_structured_reference_tests tests/test_structured_reference.cpp)
target_link_libraries(cubutterfly_structured_reference_tests PRIVATE cuntt)
target_compile_options(cubutterfly_structured_reference_tests PRIVATE -Wall -Wextra -Wpedantic)
add_executable(cubutterfly_c_api_tests tests/test_c_api.c)
target_link_libraries(cubutterfly_c_api_tests PRIVATE cuntt)
target_compile_options(cubutterfly_c_api_tests PRIVATE -Wall -Wextra -Wpedantic)
if(CUBUTTERFLY_ENABLE_CUFFTDX)
    target_compile_definitions(cubutterfly_tests PRIVATE CUBUTTERFLY_TEST_CUFFTDX=1)
endif()
if(CUBUTTERFLY_ENABLE_TURBOFFT)
    target_compile_definitions(cubutterfly_tests PRIVATE CUBUTTERFLY_TEST_TURBOFFT=1)
endif()

add_executable(cuntt_hardware_microbench apps/hardware_microbench.cu)
target_link_libraries(cuntt_hardware_microbench PRIVATE CUDA::cudart stdc++)
target_compile_options(cuntt_hardware_microbench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)

if(CUBUTTERFLY_ENABLE_VKFFT)
    set(CUBUTTERFLY_VKFFT_INCLUDE "${CUBUTTERFLY_VKFFT_ROOT}/vkFFT")
    if(NOT EXISTS "${CUBUTTERFLY_VKFFT_INCLUDE}/vkFFT.h")
        message(FATAL_ERROR
            "VkFFT headers were not found under ${CUBUTTERFLY_VKFFT_ROOT}. "
            "Run scripts/install_vkfft.sh or set CUBUTTERFLY_VKFFT_ROOT.")
    endif()

    add_executable(vkfft_bench apps/vkfft_bench.cu)
    target_include_directories(vkfft_bench SYSTEM PRIVATE "${CUBUTTERFLY_VKFFT_INCLUDE}")
    target_compile_definitions(vkfft_bench PRIVATE VKFFT_BACKEND=1)
    target_link_libraries(vkfft_bench PRIVATE CUDA::cudart CUDA::cuda_driver CUDA::nvrtc CUDA::cufft)
    target_compile_options(vkfft_bench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)
endif()

include(CTest)
if(BUILD_TESTING)
    add_test(NAME cuntt_correctness COMMAND cuntt_tests)
    add_test(NAME cubutterfly_correctness COMMAND cubutterfly_tests)
    add_test(NAME cubutterfly_structured_reference COMMAND cubutterfly_structured_reference_tests)
    add_test(NAME cubutterfly_c_api COMMAND cubutterfly_c_api_tests)
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
    add_test(NAME v08_comprehensive_analysis
             COMMAND "${Python3_EXECUTABLE}" -m pytest -q
                     "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_v08_comprehensive.py")
    add_test(NAME v08_selector_holdout_analysis
             COMMAND "${Python3_EXECUTABLE}" -m pytest -q
                     "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_v08_selector_holdout.py")
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
                     -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_install_package.cmake")
    add_test(NAME runtime_selector_generator
             COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_runtime_selector_generator.py")
    add_test(NAME application_profile_generator
             COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_application_profiles.py")
    if(CUBUTTERFLY_ENABLE_CUFFTDX)
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

install(TARGETS cuntt
    EXPORT cuButterflyTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
install(DIRECTORY include/cuntt include/cubutterfly DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(PROGRAMS
    scripts/initialize_hardware_profile.py
    scripts/check_hardware_profile.py
    DESTINATION "${CMAKE_INSTALL_BINDIR}")
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
