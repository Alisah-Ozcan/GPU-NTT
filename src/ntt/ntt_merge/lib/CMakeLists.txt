set(LIB_SOURCE_FILES
    ntt_cpu.cu
    ntt.cu
)

add_library(ntt STATIC ${LIB_SOURCE_FILES})

if(NOT DEFINED MODULAR_REDUCTION_TYPE)
    set(MODULAR_REDUCTION_TYPE 0)
endif()

if(MODULAR_REDUCTION_TYPE EQUAL 0)
    add_compile_definitions(BARRETT_64)
    target_compile_definitions(ntt PRIVATE BARRETT_64)
elseif(MODULAR_REDUCTION_TYPE EQUAL 1)
    add_compile_definitions(GOLDILOCKS_64)
    target_compile_definitions(ntt PRIVATE GOLDILOCKS_64)
elseif(MODULAR_REDUCTION_TYPE EQUAL 2)
    add_compile_definitions(PLANTARD_64)
    target_compile_definitions(ntt PRIVATE PLANTARD_64)
else()
    message(FATAL_ERROR "Please use valid MODULAR_REDUCTION_TYPE! (0,1,2)")
endif()

target_link_libraries(ntt PUBLIC common butterfly_units)

set_target_properties(ntt PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(ntt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)



