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
