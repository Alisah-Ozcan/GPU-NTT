set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BINARY_LOCATION})

if(NOT DEFINED MODULAR_REDUCTION_TYPE)
    set(MODULAR_REDUCTION_TYPE 0)
endif()

set(EXECUTABLES
    benchmark_merge_ntt bench_merge_ntt.cu
    benchmark_4step_ntt bench_4step_ntt.cu
)

function(add_benchmark exe source)
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
    add_benchmark(${exe} ${source})
endforeach()

if(MODULAR_REDUCTION_TYPE EQUAL 0)
    set(DEFINITION BARRETT_64)
elseif(MODULAR_REDUCTION_TYPE EQUAL 1)
    set(DEFINITION GOLDILOCKS_64)
elseif(MODULAR_REDUCTION_TYPE EQUAL 2)
    set(DEFINITION PLANTARD_64)
else()
    message(FATAL_ERROR "Please use valid MODULAR_REDUCTION_TYPE! (0,1,2)")
endif()

add_compile_definitions(${DEFINITION})

foreach(i RANGE 0 ${num_executables}-1)
    math(EXPR index1 "${i} * 2")
    list(GET EXECUTABLES ${index1} exe)
    target_compile_definitions(${exe} PRIVATE ${DEFINITION})
endforeach()








