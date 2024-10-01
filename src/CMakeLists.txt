file(GLOB_RECURSE LIB_SOURCE_FILES
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    CONFIGURE_DEPENDS
    lib/common/*.cu
    lib/ntt_4step/*.cu
    lib/ntt_merge/*.cu
)

file(GLOB_RECURSE LIB_HEADER_FILES
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    CONFIGURE_DEPENDS
    include/common/*.cuh
    include/ntt_4step/*.cuh
    include/ntt_merge/*.cuh
)

add_library(ntt STATIC ${LIB_SOURCE_FILES})

set_target_properties(ntt PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION 1
  OUTPUT_NAME "ntt-${PROJECT_VERSION}"
)

if(NOT DEFINED MODULAR_REDUCTION_TYPE)
    set(MODULAR_REDUCTION_TYPE 0)
endif()

target_include_directories(
    ntt
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/common>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/ntt_4step>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/ntt_merge>
    $<INSTALL_INTERFACE:include/GPUNTT-${PROJECT_VERSION}>
)

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

set_target_properties(ntt PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_RUNTIME_LIBRARY Static
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
)

install(TARGETS ntt
  EXPORT ${GPUNTT_TARGETS_EXPORT_NAME}
  RUNTIME DESTINATION ${RUNTIME_DESTINATION}
  LIBRARY DESTINATION ${LIBRARY_DESTINATION}
  ARCHIVE DESTINATION ${ARCHIVE_DESTINATION}
)

install(FILES ${LIB_HEADER_FILES} DESTINATION ${INCLUDES_INSTALL_DIR})