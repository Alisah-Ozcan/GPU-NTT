set(COMMON_SOURCE_FILES
    common.cu
    nttparameters.cu
)

if(NOT DEFINED MODULAR_REDUCTION_TYPE)
    set(MODULAR_REDUCTION_TYPE 0)
endif()

add_library(common STATIC ${COMMON_SOURCE_FILES})

target_include_directories(common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)

if(MODULAR_REDUCTION_TYPE EQUAL 0)
    add_compile_definitions(BARRETT_64)
    target_compile_definitions(common PRIVATE BARRETT_64)
elseif(MODULAR_REDUCTION_TYPE EQUAL 1)
    add_compile_definitions(GOLDILOCKS_64)
    target_compile_definitions(common PRIVATE GOLDILOCKS_64)
elseif(MODULAR_REDUCTION_TYPE EQUAL 2)
    add_compile_definitions(PLANTARD_64)
    target_compile_definitions(common PRIVATE PLANTARD_64)
else()
    message(FATAL_ERROR "Please use valid MODULAR_REDUCTION_TYPE! (0,1,2)")
endif()

set_target_properties(common PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)