set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BINARY_LOCATION}/example)

set(EXECUTABLES
    gpu_merge_ntt_examples ntt_merge/test_merge_ntt.cu
    gpu_merge_intt_examples ntt_merge/test_merge_intt.cu
    cpu_merge_ntt_examples ntt_merge/test_cpu_merge_ntt.cu
    gpu_device_api_examples ntt_merge/test_device_api.cu
    gpu_device_api_rns_examples ntt_merge/test_device_api_rns.cu
)

function(add_example exe source)
    add_executable(${exe} ${source})
    target_link_libraries(${exe} PRIVATE ntt CUDA::cudart)
    set_target_properties(${exe} PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        POSITION_INDEPENDENT_CODE ON
        CUDA_RUNTIME_LIBRARY Static
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    )
endfunction()

list(LENGTH EXECUTABLES EXECUTABLES_LENGTH)
math(EXPR EXECUTABLES_COUNT "${EXECUTABLES_LENGTH} / 2")
math(EXPR EXECUTABLES_COUNT_LOOP "${EXECUTABLES_COUNT} - 1")

foreach(i RANGE 0 ${EXECUTABLES_COUNT_LOOP})
    math(EXPR index1 "${i} * 2")
    math(EXPR index2 "${i} * 2 + 1")
    list(GET EXECUTABLES ${index1} exe)
    list(GET EXECUTABLES ${index2} source)
    add_example(${exe} ${source})
endforeach()

