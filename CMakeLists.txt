cmake_minimum_required(VERSION 3.20)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 CACHE STRING "CUDA architectures to compile for")
endif()

project(cuButterfly VERSION 0.2.0 LANGUAGES CXX CUDA)

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
set(CUNTT_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CUNTT_DESIGN_SPEC}"
    "${CUBUTTERFLY_FFT_SPACE_SPEC}"
    "${CUBUTTERFLY_ARCHITECTURE_SPACE_SPEC}"
    "${CUBUTTERFLY_FFT_CODEGEN_SPEC}"
    "${CMAKE_CURRENT_SOURCE_DIR}/config/v100_comprehensive_suite.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/config/v100_scaling_space.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/config/v100_scaling_suite.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/scripts/fft_design_space.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/scripts/butterfly_design_space.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_fft_codegen.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_scaling_suite.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_design_points.py")
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
include("${CUNTT_GENERATED_DIR}/design_points.cmake")

add_library(cuntt STATIC
    src/butterfly.cu
    src/butterfly_reference.cpp
    src/ntt.cu
    src/reference.cpp)
target_sources(cuntt PRIVATE ${CUNTT_GENERATED_SOURCES})

if(CUBUTTERFLY_ENABLE_CUFFTDX)
    if(NOT EXISTS "${CUBUTTERFLY_MATHDX_ROOT}/include/cufftdx.hpp")
        message(FATAL_ERROR
            "cuFFTDx headers were not found under ${CUBUTTERFLY_MATHDX_ROOT}. "
            "Run scripts/install_cufftdx.sh or set CUBUTTERFLY_MATHDX_ROOT.")
    endif()
    target_sources(cuntt PRIVATE src/cufftdx_fft.cu)
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
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CUNTT_GENERATED_DIR})

target_link_libraries(cuntt PUBLIC CUDA::cudart CUDA::cufft)

target_compile_options(cuntt
    PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)

set_target_properties(cuntt PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON)

add_executable(cuntt_bench apps/cuntt_bench.cpp)
target_link_libraries(cuntt_bench PRIVATE cuntt)
target_compile_options(cuntt_bench PRIVATE -Wall -Wextra -Wpedantic)

add_executable(cubutterfly_bench apps/cubutterfly_bench.cpp)
target_link_libraries(cubutterfly_bench PRIVATE cuntt)
target_compile_options(cubutterfly_bench PRIVATE -Wall -Wextra -Wpedantic)

add_executable(cuntt_tests tests/test_ntt.cpp)
target_link_libraries(cuntt_tests PRIVATE cuntt)
target_compile_options(cuntt_tests PRIVATE -Wall -Wextra -Wpedantic)

add_executable(cubutterfly_tests tests/test_butterfly.cpp)
target_link_libraries(cubutterfly_tests PRIVATE cuntt)
target_compile_options(cubutterfly_tests PRIVATE -Wall -Wextra -Wpedantic)
if(CUBUTTERFLY_ENABLE_CUFFTDX)
    target_compile_definitions(cubutterfly_tests PRIVATE CUBUTTERFLY_TEST_CUFFTDX=1)
endif()
if(CUBUTTERFLY_ENABLE_TURBOFFT)
    target_compile_definitions(cubutterfly_tests PRIVATE CUBUTTERFLY_TEST_TURBOFFT=1)
endif()

add_executable(cuntt_hardware_microbench apps/hardware_microbench.cu)
target_link_libraries(cuntt_hardware_microbench PRIVATE CUDA::cudart)
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
    add_test(NAME fft_design_space
             COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_fft_design_space.py")
    add_test(NAME butterfly_design_space
             COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_butterfly_design_space.py")
    add_test(NAME comprehensive_suite
             COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_comprehensive_suite.py")
    add_test(NAME scaling_suite
             COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_scaling_suite.py")
endif()
