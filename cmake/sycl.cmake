add_library(${GPU_TARGET} SHARED ${GPU_TARGET_SOURCE_FILES})

set(PYTHON_SCRIPT "from pathlib import Path;import gemmforge as gf;print(Path(gf.__file__).parents[1], end='')")
execute_process(COMMAND python3 -c "${PYTHON_SCRIPT}"
                OUTPUT_VARIABLE GEMMFORGE_PATH)
set(CMAKE_MODULE_PATH "${GEMMFORGE_PATH}/submodules/Device/cmake"
                      "${CMAKE_ROOT}/Modules")

if (${DEVICE_BACKEND} STREQUAL "opensycl")
    find_package(OpenSYCLSettings REQUIRED)
    find_package(OpenSYCL REQUIRED)
    target_link_libraries(${GPU_TARGET} PUBLIC opensycl-settings::device_flags)
    target_link_libraries(${GPU_TARGET} PUBLIC opensycl-settings::interface)
    add_sycl_to_target(TARGET ${GPU_TARGET}  SOURCES ${DEVICE_SOURCE_FILES})
else()
    find_package(DpcppFlags REQUIRED)
    target_link_libraries(${GPU_TARGET} PRIVATE dpcpp::device_flags)
endif()

target_compile_options(${GPU_TARGET} PRIVATE "-std=c++17" "-O3")
target_compile_definitions(${GPU_TARGET} PRIVATE DEVICE_${DEVICE_BACKEND}_LANG REAL_SIZE=${REAL_SIZE})
target_link_libraries(${GPU_TARGET} PUBLIC stdc++fs)
