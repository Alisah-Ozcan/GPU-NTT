set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BINARY_LOCATION})

if(NOT DEFINED MODULAR_REDUCTION_TYPE)
    set(MODULAR_REDUCTION_TYPE 0)
endif()

add_executable(gpu_merge_ntt_examples test_merge_ntt.cu)
add_executable(gpu_merge_intt_examples test_merge_intt.cu)


if(MODULAR_REDUCTION_TYPE EQUAL 0)
    add_compile_definitions(BARRETT_64)
    target_compile_definitions(gpu_merge_ntt_examples PRIVATE BARRETT_64)
    target_compile_definitions(gpu_merge_intt_examples PRIVATE BARRETT_64)
elseif(MODULAR_REDUCTION_TYPE EQUAL 1)
    add_compile_definitions(GOLDILOCKS_64)
    target_compile_definitions(gpu_merge_ntt_examples PRIVATE GOLDILOCKS_64)
    target_compile_definitions(gpu_merge_intt_examples PRIVATE GOLDILOCKS_64)
elseif(MODULAR_REDUCTION_TYPE EQUAL 2)
    add_compile_definitions(PLANTARD_64)
    target_compile_definitions(gpu_merge_ntt_examples PRIVATE PLANTARD_64)
    target_compile_definitions(gpu_merge_intt_examples PRIVATE PLANTARD_64)
else()
    message(FATAL_ERROR "Please use valid MODULAR_REDUCTION_TYPE! (0,1,2)")
endif()

target_link_libraries(gpu_merge_ntt_examples PRIVATE ntt)

set_target_properties(gpu_merge_ntt_examples PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(gpu_merge_ntt_examples PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)


target_link_libraries(gpu_merge_intt_examples PRIVATE ntt)

set_target_properties(gpu_merge_intt_examples PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(gpu_merge_intt_examples PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)