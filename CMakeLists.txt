cmake_minimum_required(VERSION 3.2)

set(CMAKE_CUDA_STANDARD 17)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

project(NTT_Library LANGUAGES C CXX CUDA ASM)

if(NOT DEFINED MODULAR_REDUCTION_TYPE)
    set(MODULAR_REDUCTION_TYPE 0)
endif()

# NTT_Library
add_library(NTT_Library src/common.cu src/nttparameters.cu src/ntt_cpu.cu src/ntt.cu)

if(NOT "${CMAKE_CUDA_BYTE_ORDER}" STREQUAL "LITTLE_ENDIAN")
    message(FATAL_ERROR "Target architecture must be little-endian")
endif()

set_target_properties(NTT_Library PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(test_ntt_cpu test/ntt_cpu_test.cu)
add_executable(test_ntt_gpu test/ntt_gpu_test.cu)
add_executable(bench_ntt_gpu test/ntt_gpu_bench.cu)


if(MODULAR_REDUCTION_TYPE EQUAL 0)
    add_compile_definitions(BARRETT_64)
    target_compile_definitions(test_ntt_cpu PRIVATE BARRETT_64)
    target_compile_definitions(test_ntt_gpu PRIVATE BARRETT_64)
    target_compile_definitions(bench_ntt_gpu PRIVATE BARRETT_64)
elseif(MODULAR_REDUCTION_TYPE EQUAL 1)
    add_compile_definitions(GOLDILOCKS_64)
    target_compile_definitions(test_ntt_cpu PRIVATE GOLDILOCKS_64)
    target_compile_definitions(test_ntt_gpu PRIVATE GOLDILOCKS_64)
    target_compile_definitions(bench_ntt_gpu PRIVATE GOLDILOCKS_64)
elseif(MODULAR_REDUCTION_TYPE EQUAL 2)
    add_compile_definitions(PLANTARD_64)
    target_compile_definitions(test_ntt_cpu PRIVATE PLANTARD_64)
    target_compile_definitions(test_ntt_gpu PRIVATE PLANTARD_64)
    target_compile_definitions(bench_ntt_gpu PRIVATE PLANTARD_64)
else()
    message(FATAL_ERROR "Please use valid MODULAR_REDUCTION_TYPE! (0,1,2)")
endif()


target_link_libraries(test_ntt_cpu PRIVATE NTT_Library)
set_target_properties(test_ntt_cpu PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(test_ntt_gpu PRIVATE NTT_Library)
set_target_properties(test_ntt_gpu PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(bench_ntt_gpu PRIVATE NTT_Library)
set_target_properties(bench_ntt_gpu PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)