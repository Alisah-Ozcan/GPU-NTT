cmake_minimum_required(VERSION 3.12)
project(NTT LANGUAGES C CXX CUDA ASM)

set(CMAKE_CUDA_STANDARD 17)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

set(BINARY_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/bin)

add_subdirectory(src)
add_subdirectory(example)
add_subdirectory(benchmark)