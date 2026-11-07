# cuFFTDx instantiates one codelet target per build. No GPU queries or execution.
function(cubutterfly_cufftdx_target output_arch output_sm)
    set(target "")
    foreach(entry IN LISTS ARGN)
        if(NOT entry MATCHES "^([1-9][0-9]*)(-(real|virtual))?$")
            message(FATAL_ERROR "cuFFTDx requires explicit numeric CUDA architectures; got '${entry}'")
        endif()
        set(arch "${CMAKE_MATCH_1}")
        if(NOT target STREQUAL "" AND NOT target STREQUAL arch)
            message(FATAL_ERROR "cuFFTDx codelets require one SM target; mixed architectures '${ARGN}' are unsupported")
        endif()
        set(target "${arch}")
    endforeach()
    if(target STREQUAL "")
        message(FATAL_ERROR "cuFFTDx requires an explicit numeric CUDA architecture")
    endif()
    math(EXPR sm "${target} * 10")
    set(${output_arch} "${target}" PARENT_SCOPE)
    set(${output_sm} "${sm}" PARENT_SCOPE)
endfunction()

function(cubutterfly_probe_cufftdx nvcc mathdx arch sm directory)
    execute_process(COMMAND "${CMAKE_COMMAND}" -E env "CUDA_VISIBLE_DEVICES="
        "${nvcc}" --list-gpu-code RESULT_VARIABLE status
        OUTPUT_VARIABLE codes ERROR_VARIABLE error)
    if(NOT status EQUAL 0 OR NOT codes MATCHES "(^|[ \t\r\n])sm_${arch}([ \t\r\n]|$)")
        message(FATAL_ERROR "nvcc '${nvcc}' does not advertise sm_${arch} via --list-gpu-code. Select a supporting CUDA toolkit. ${error}")
    endif()
    file(MAKE_DIRECTORY "${directory}")
    set(source "${directory}/cufftdx_target_probe.cu")
    file(WRITE "${source}" "#include <cuda_runtime.h>\n#include <cufftdx.hpp>\nusing namespace cufftdx;\nusing FFT = decltype(Size<32>() + Precision<float>() + Type<fft_type::c2c>() + Direction<fft_direction::forward>() + ElementsPerThread<2>() + FFTsPerBlock<1>() + SM<${sm}>() + Block());\n__global__ void probe() { typename FFT::value_type data[FFT::storage_size]; extern __shared__ __align__(16) unsigned char scratch[]; FFT().execute(data, scratch); }\n")
    execute_process(COMMAND "${CMAKE_COMMAND}" -E env "CUDA_VISIBLE_DEVICES="
        "${nvcc}" -std=c++17 "-arch=sm_${arch}" "-I${mathdx}/include"
        -c "${source}" -o "${directory}/cufftdx_target_probe.o"
        RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
    if(NOT status EQUAL 0)
        message(FATAL_ERROR "cuFFTDx target probe failed for sm_${arch} (MathDx SM<${sm}>) with '${nvcc}' and '${mathdx}'. Install compatible CUDA/MathDx packages; the core will not be disabled.\n${output}\n${error}")
    endif()
endfunction()

# Standalone CPU preparation entry; also usable in cmake -P regression tests.
if(CMAKE_SCRIPT_MODE_FILE STREQUAL CMAKE_CURRENT_LIST_FILE)
    cubutterfly_cufftdx_target(arch sm ${CUBUTTERFLY_PROBE_ARCHITECTURES})
    cubutterfly_probe_cufftdx("${CUBUTTERFLY_PROBE_NVCC}" "${CUBUTTERFLY_PROBE_MATHDX}"
        "${arch}" "${sm}" "${CUBUTTERFLY_PROBE_DIRECTORY}")
endif()
