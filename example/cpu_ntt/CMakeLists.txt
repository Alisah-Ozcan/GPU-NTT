set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BINARY_LOCATION})

if(NOT DEFINED MODULAR_REDUCTION_TYPE)
    set(MODULAR_REDUCTION_TYPE 0)
endif()

add_executable(cpu_4step_ntt test_cpu_4step_ntt.cu)
add_executable(cpu_merge_ntt test_cpu_merge_ntt.cu)


if(MODULAR_REDUCTION_TYPE EQUAL 0)
    add_compile_definitions(BARRETT_64)
    target_compile_definitions(cpu_4step_ntt PRIVATE BARRETT_64)
    target_compile_definitions(cpu_merge_ntt PRIVATE BARRETT_64)
elseif(MODULAR_REDUCTION_TYPE EQUAL 1)
    add_compile_definitions(GOLDILOCKS_64)
    target_compile_definitions(cpu_4step_ntt PRIVATE GOLDILOCKS_64)
    target_compile_definitions(cpu_merge_ntt PRIVATE GOLDILOCKS_64)
elseif(MODULAR_REDUCTION_TYPE EQUAL 2)
    add_compile_definitions(PLANTARD_64)
    target_compile_definitions(cpu_4step_ntt PRIVATE PLANTARD_64)
    target_compile_definitions(cpu_merge_ntt PRIVATE PLANTARD_64)
else()
    message(FATAL_ERROR "Please use valid MODULAR_REDUCTION_TYPE! (0,1,2)")
endif()


target_link_libraries(cpu_4step_ntt PRIVATE ntt ntt_4step)

set_target_properties(cpu_4step_ntt PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(cpu_4step_ntt PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)


target_link_libraries(cpu_merge_ntt PRIVATE ntt ntt_4step)

set_target_properties(cpu_merge_ntt PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(cpu_merge_ntt PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)




