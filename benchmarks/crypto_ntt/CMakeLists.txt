cmake_minimum_required(VERSION 3.20)

# This is intentionally an isolated consumer build.  It does not participate
# in the repository's top-level CMake project and only consumes a previously
# built public cuButterfly archive.
project(cuButterflyCryptoNtt LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "CUDA architecture for the isolated benchmark")

set(CUBUTTERFLY_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../.." CACHE PATH
    "cuButterfly source checkout")
set(CUBUTTERFLY_BUILD_DIR "${CUBUTTERFLY_SOURCE_DIR}/build-a100-cufftdx" CACHE PATH
    "Existing cuButterfly build containing libcuntt.a")

find_package(CUDAToolkit REQUIRED)

set(CUBUTTERFLY_ARCHIVE "${CUBUTTERFLY_BUILD_DIR}/libcuntt.a")
set(CUBUTTERFLY_PUBLIC_INCLUDE "${CUBUTTERFLY_SOURCE_DIR}/include")
set(CUBUTTERFLY_GENERATED_INCLUDE "${CUBUTTERFLY_BUILD_DIR}/generated")
set(CUBUTTERFLY_NLOHMANN_INCLUDE "${CUBUTTERFLY_BUILD_DIR}/_deps/nlohmann_json-src/include" CACHE PATH
    "Header-only nlohmann-json include used by the isolated benchmark")
if(NOT EXISTS "${CUBUTTERFLY_ARCHIVE}")
    message(FATAL_ERROR "Missing ${CUBUTTERFLY_ARCHIVE}; build cuButterfly first")
endif()
if(NOT EXISTS "${CUBUTTERFLY_PUBLIC_INCLUDE}/cuntt/ntt.hpp")
    message(FATAL_ERROR "Missing public cuButterfly headers under ${CUBUTTERFLY_PUBLIC_INCLUDE}")
endif()
if(NOT EXISTS "${CUBUTTERFLY_NLOHMANN_INCLUDE}/nlohmann/json.hpp")
    message(FATAL_ERROR "Missing nlohmann/json.hpp under ${CUBUTTERFLY_NLOHMANN_INCLUDE}")
endif()

add_executable(crypto_ntt_bench crypto_ntt_bench.cu)
target_include_directories(crypto_ntt_bench PRIVATE
    "${CUBUTTERFLY_PUBLIC_INCLUDE}"
    "${CUBUTTERFLY_GENERATED_INCLUDE}"
    "${CUBUTTERFLY_NLOHMANN_INCLUDE}")
target_link_libraries(crypto_ntt_bench PRIVATE
    "${CUBUTTERFLY_ARCHIVE}"
    CUDA::cudart
    CUDA::cufft
    ${CMAKE_DL_LIBS})
set_target_properties(crypto_ntt_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
target_compile_options(crypto_ntt_bench PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)
