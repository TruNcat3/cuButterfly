cmake_minimum_required(VERSION 3.24)

project(gpuntt_crypto_bench LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "80" CACHE STRING "CUDA architectures for the benchmark")

set(GPUNTT_SOURCE_DIR "" CACHE PATH
    "Clean, pinned official GPU-NTT checkout")
set(CUBUTTERFLY_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}/../.." CACHE PATH
    "cuButterfly source tree used for the cuNTT root finder")
set(CUBUTTERFLY_BUILD_DIR "${CUBUTTERFLY_ROOT_DIR}/build-a100-cufftdx" CACHE PATH
    "cuButterfly build containing libcuntt.a and nlohmann-json")
set(CUNTT_LIBRARY "" CACHE FILEPATH "Optional explicit path to libcuntt.a")
set(NLOHMANN_INCLUDE_DIR "" CACHE PATH "Optional explicit nlohmann-json include directory")

if(NOT GPUNTT_SOURCE_DIR OR NOT EXISTS "${GPUNTT_SOURCE_DIR}/src/include/gpuntt/ntt_merge/ntt.cuh")
  message(FATAL_ERROR "GPUNTT_SOURCE_DIR must point to the clean official GPU-NTT checkout")
endif()
execute_process(
  COMMAND git -C "${GPUNTT_SOURCE_DIR}" rev-parse HEAD
  OUTPUT_VARIABLE GPUNTT_SOURCE_REVISION
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE GPUNTT_GIT_RESULT)
if(NOT GPUNTT_GIT_RESULT EQUAL 0 OR NOT GPUNTT_SOURCE_REVISION STREQUAL
   "d03c5eaeadaa780d153496afcb3b6a9b79a13a63")
  message(FATAL_ERROR "GPUNTT_SOURCE_DIR must be pinned to official GPU-NTT d03c5eaeadaa780d153496afcb3b6a9b79a13a63")
endif()
execute_process(
  COMMAND git -C "${GPUNTT_SOURCE_DIR}" status --porcelain
  OUTPUT_VARIABLE GPUNTT_GIT_STATUS
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE GPUNTT_STATUS_RESULT)
if(NOT GPUNTT_STATUS_RESULT EQUAL 0 OR GPUNTT_GIT_STATUS)
  message(FATAL_ERROR "GPUNTT_SOURCE_DIR must be a clean checkout")
endif()

if(NOT CUNTT_LIBRARY)
  foreach(candidate
      "${CUBUTTERFLY_BUILD_DIR}/libcuntt.a"
      "${CUBUTTERFLY_BUILD_DIR}/lib/libcuntt.a"
      "${CUBUTTERFLY_BUILD_DIR}/package-consumer-test/install/lib/libcuntt.a")
    if(EXISTS "${candidate}")
      set(CUNTT_LIBRARY "${candidate}")
      break()
    endif()
  endforeach()
endif()
if(NOT CUNTT_LIBRARY OR NOT EXISTS "${CUNTT_LIBRARY}")
  message(FATAL_ERROR "Could not find libcuntt.a; set CUNTT_LIBRARY explicitly")
endif()

if(NOT NLOHMANN_INCLUDE_DIR)
  set(_nlohmann_candidate "${CUBUTTERFLY_BUILD_DIR}/_deps/nlohmann_json-src/include")
  if(EXISTS "${_nlohmann_candidate}/nlohmann/json.hpp")
    set(NLOHMANN_INCLUDE_DIR "${_nlohmann_candidate}")
  endif()
endif()
if(NOT NLOHMANN_INCLUDE_DIR OR NOT EXISTS "${NLOHMANN_INCLUDE_DIR}/nlohmann/json.hpp")
  message(FATAL_ERROR "Could not find nlohmann/json.hpp; set NLOHMANN_INCLUDE_DIR explicitly")
endif()

find_package(CUDAToolkit REQUIRED)

set(GPUNTT_COMMON_SOURCES
    "${GPUNTT_SOURCE_DIR}/src/lib/common/common.cu"
    "${GPUNTT_SOURCE_DIR}/src/lib/common/nttparameters.cu"
    "${GPUNTT_SOURCE_DIR}/src/lib/ntt_merge/ntt.cu")

add_library(gpuntt_crypto_vendor STATIC ${GPUNTT_COMMON_SOURCES})
target_include_directories(gpuntt_crypto_vendor PUBLIC "${GPUNTT_SOURCE_DIR}/src/include")
target_compile_features(gpuntt_crypto_vendor PUBLIC cxx_std_17)
target_link_libraries(gpuntt_crypto_vendor PUBLIC CUDA::cudart)
set_target_properties(gpuntt_crypto_vendor PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_RUNTIME_LIBRARY Static
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

add_executable(gpuntt_crypto_bench gpuntt_crypto_bench.cu)
target_include_directories(gpuntt_crypto_bench PRIVATE
    "${GPUNTT_SOURCE_DIR}/src/include"
    "${CUBUTTERFLY_ROOT_DIR}/include"
    "${NLOHMANN_INCLUDE_DIR}")
target_compile_definitions(gpuntt_crypto_bench PRIVATE
    GPUNTT_SOURCE_REVISION=\"${GPUNTT_SOURCE_REVISION}\")
target_link_libraries(gpuntt_crypto_bench PRIVATE
    gpuntt_crypto_vendor
    "${CUNTT_LIBRARY}"
    CUDA::cudart)
set_target_properties(gpuntt_crypto_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_RUNTIME_LIBRARY Static
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
