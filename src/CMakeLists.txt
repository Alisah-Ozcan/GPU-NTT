set(LIB_SOURCE_FILES
    lib/common/common.cu
    lib/common/nttparameters.cu
    lib/ntt_4step/ntt_4step.cu
    lib/ntt_4step/ntt_4step_cpu.cu
    lib/ntt_merge/ntt.cu
    lib/ntt_merge/ntt_cpu.cu
)

set(LIB_HEADER_FILES
    include/common/common.cuh
    include/common/nttparameters.cuh
    include/common/modular_arith.cuh
    include/ntt_4step/ntt_4step.cuh
    include/ntt_4step/ntt_4step_cpu.cuh
    include/ntt_merge/ntt.cuh
    include/ntt_merge/ntt_cpu.cuh
)

add_library(ntt STATIC ${LIB_SOURCE_FILES})
add_library(GPUNTT::ntt ALIAS ntt)
if(CMAKE_CXX_COMPILE_FEATURES)
    target_compile_features(ntt PUBLIC cxx_std_17)
else()
    message(WARNING
        "CMake does not know C++ compile features for ${CMAKE_CXX_COMPILER_ID} "
        "${CMAKE_CXX_COMPILER_VERSION}; falling back to -std=c++17"
    )
    target_compile_options(ntt PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-std=c++17>)
endif()

if(CMAKE_CUDA_COMPILE_FEATURES)
    target_compile_features(ntt PRIVATE cuda_std_17)
else()
    target_compile_options(ntt PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--std=c++17>)
endif()
target_compile_options(ntt PRIVATE
    $<$<CONFIG:Release>:-O3 -DNDEBUG>
    $<$<CONFIG:RelWithDebInfo>:-O3 -g -DNDEBUG>
    $<$<CONFIG:MinSizeRel>:-Os -DNDEBUG>
    $<$<CONFIG:Debug>:-g>
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:--generate-line-info>
    $<$<AND:$<CONFIG:RelWithDebInfo>,$<COMPILE_LANGUAGE:CUDA>>:--generate-line-info>
)
option(GPUNTT_ENABLE_WARNINGS "Enable extra warnings for GPUNTT" OFF)
if(GPUNTT_ENABLE_WARNINGS)
    target_compile_options(ntt PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>
    )
endif()

target_link_libraries(ntt PUBLIC CUDA::cudart)

set_target_properties(ntt PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION 1
  OUTPUT_NAME "ntt-${PROJECT_VERSION}"
)

target_include_directories(
    ntt
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/common>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/ntt_4step>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/ntt_merge>
    $<INSTALL_INTERFACE:include/GPUNTT-${PROJECT_VERSION}>
)

set_target_properties(ntt PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_RUNTIME_LIBRARY Static
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)

install(TARGETS ntt
  EXPORT ${GPUNTT_TARGETS_EXPORT_NAME}
  RUNTIME DESTINATION ${RUNTIME_DESTINATION}
  LIBRARY DESTINATION ${LIBRARY_DESTINATION}
  ARCHIVE DESTINATION ${ARCHIVE_DESTINATION}
)

install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
    DESTINATION ${INCLUDES_INSTALL_DIR}
)
