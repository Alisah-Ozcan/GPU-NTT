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

add_executable(descriptor_tests test_descriptor.cu)
target_link_libraries(descriptor_tests PRIVATE ntt CUDA::cudart)
set_target_properties(descriptor_tests PROPERTIES
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
)
add_test(NAME descriptor COMMAND descriptor_tests)

add_executable(ntt_device_tests test_ntt_device.cu)
target_link_libraries(ntt_device_tests PRIVATE ntt CUDA::cudart)
set_target_properties(ntt_device_tests PROPERTIES
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
)
add_test(NAME ntt_device COMMAND ntt_device_tests)

foreach(bits 32 64)
    foreach(suite single pool_parameters modulus validation rns signed
                  per_coefficient per_coefficient_rns modulus_ordered
                  poly_ordered large)
        add_test(
            NAME merge_ntt_${bits}_${suite}
            COMMAND merge_ntt_tests --bits ${bits} --suite ${suite}
                    --max-logn ${GPUNTT_TEST_MAX_LOGN}
        )
    endforeach()
    # The large suites need about 3 GB of GPU memory each; run them one at a
    # time.
    set_tests_properties(merge_ntt_${bits}_large PROPERTIES
        RESOURCE_LOCK merge_ntt_large)
endforeach()
