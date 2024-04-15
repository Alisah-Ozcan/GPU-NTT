set(LIB_SOURCE_FILES
    butterfly.cu
)

add_library(butterfly_units STATIC ${LIB_SOURCE_FILES})

if(NOT DEFINED MODULAR_REDUCTION_TYPE)
    set(MODULAR_REDUCTION_TYPE 0)
endif()

if(MODULAR_REDUCTION_TYPE EQUAL 0)
    add_compile_definitions(BARRETT_64)
    target_compile_definitions(butterfly_units PRIVATE BARRETT_64)
elseif(MODULAR_REDUCTION_TYPE EQUAL 1)
    add_compile_definitions(GOLDILOCKS_64)
    target_compile_definitions(butterfly_units PRIVATE GOLDILOCKS_64)
elseif(MODULAR_REDUCTION_TYPE EQUAL 2)
    add_compile_definitions(PLANTARD_64)
    target_compile_definitions(butterfly_units PRIVATE PLANTARD_64)
else()
    message(FATAL_ERROR "Please use valid MODULAR_REDUCTION_TYPE! (0,1,2)")
endif()

target_link_libraries(butterfly_units PUBLIC common)

set_target_properties(butterfly_units PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(butterfly_units PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)



