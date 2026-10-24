set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BINARY_LOCATION}/test)

set(GPUNTT_TEST_MAX_LOGN 22 CACHE STRING
    "Largest log2 ring size covered by the GPU-NTT tests")

add_executable(merge_ntt_tests test_merge_ntt.cu)
target_link_libraries(merge_ntt_tests PRIVATE ntt CUDA::cudart)
set_target_properties(merge_ntt_tests PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON
    CUDA_RUNTIME_LIBRARY Static
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
)

foreach(bits 32 64)
    foreach(suite single pool_parameters modulus rns signed per_coefficient
                  per_coefficient_rns modulus_ordered poly_ordered)
        add_test(
            NAME merge_ntt_${bits}_${suite}
            COMMAND merge_ntt_tests --bits ${bits} --suite ${suite}
                    --max-logn ${GPUNTT_TEST_MAX_LOGN}
        )
    endforeach()
endforeach()
