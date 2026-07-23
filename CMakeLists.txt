cmake_minimum_required(VERSION 3.20)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 CACHE STRING "CUDA architectures to compile for")
endif()

project(cuButterfly VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

find_package(CUDAToolkit REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(CUNTT_DESIGN_SPEC "${CMAKE_CURRENT_SOURCE_DIR}/config/v100_design_points.json" CACHE FILEPATH
    "JSON file selecting generated cuButterfly design points")
set(CUNTT_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CUNTT_DESIGN_SPEC}"
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
include("${CUNTT_GENERATED_DIR}/design_points.cmake")

add_library(cuntt STATIC
    src/butterfly.cu
    src/butterfly_reference.cpp
    src/ntt.cu
    src/reference.cpp)
target_sources(cuntt PRIVATE ${CUNTT_GENERATED_SOURCES})

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

add_executable(cuntt_hardware_microbench apps/hardware_microbench.cu)
target_link_libraries(cuntt_hardware_microbench PRIVATE CUDA::cudart)
target_compile_options(cuntt_hardware_microbench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)

include(CTest)
if(BUILD_TESTING)
    add_test(NAME cuntt_correctness COMMAND cuntt_tests)
    add_test(NAME cubutterfly_correctness COMMAND cubutterfly_tests)
endif()
