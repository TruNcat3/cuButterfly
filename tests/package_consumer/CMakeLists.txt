cmake_minimum_required(VERSION 3.20)
project(cuButterflyPackageConsumer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(cuButterfly CONFIG REQUIRED)

add_executable(package_consumer main.cpp)
target_link_libraries(package_consumer PRIVATE cuButterfly::cuButterfly)
