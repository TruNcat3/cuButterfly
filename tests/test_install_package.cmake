set(test_root "${BUILD_DIR}/package-consumer-test")
set(install_prefix "${test_root}/install")
set(consumer_build "${test_root}/build")
file(REMOVE_RECURSE "${test_root}")

execute_process(
    COMMAND "${CMAKE_COMMAND}" --install "${BUILD_DIR}" --prefix "${install_prefix}"
    RESULT_VARIABLE result)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "cuButterfly package installation failed")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env --unset=LD_LIBRARY_PATH
            "${install_prefix}/bin/cubutterfly_bench" --help
    RESULT_VARIABLE result OUTPUT_QUIET)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "installed benchmark could not load shared dependencies without LD_LIBRARY_PATH")
endif()

foreach(script calibrate_hardware.py calibrate_local_hardware.py initialize_hardware_profile.py fit_local_cost_model.py
               stage_cost_model.py stage_service_model.py stage_service_calibration.py validate_stage_service_profile.py
               hardware_registry.py run_fft_acceptance.py)
    execute_process(COMMAND "${PYTHON_EXECUTABLE}" "${install_prefix}/bin/${script}" --help
        RESULT_VARIABLE result OUTPUT_QUIET)
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "installed tool ${script} could not import its dependencies")
    endif()
endforeach()

execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -I "${SOURCE_DIR}/tests/test_install_tool_dependencies.py"
            --bin "${install_prefix}/bin"
            --templates "${install_prefix}/share/cuButterfly/templates"
    WORKING_DIRECTORY "${test_root}"
    RESULT_VARIABLE result)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "installed calibration dependency or shared template check failed")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}"
            -S "${SOURCE_DIR}/tests/package_consumer"
            -B "${consumer_build}"
            -DCMAKE_PREFIX_PATH=${install_prefix}
    RESULT_VARIABLE result)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "installed package consumer configuration failed")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" --build "${consumer_build}"
    RESULT_VARIABLE result)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "installed package consumer build failed")
endif()

execute_process(
    COMMAND "${consumer_build}/package_consumer"
    RESULT_VARIABLE result)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "installed package consumer execution failed")
endif()
