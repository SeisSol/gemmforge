cmake_minimum_required(VERSION 3.5)
project(gemm-chain)

#include(../../cmake/setupOptions.cmake)
include(${CMAKE_SOURCE_DIR}/gen_code/config.cmake)

set(TARGET ${CMAKE_PROJECT_NAME})
set(TARGET_SOURCE_FILES common/aux.cpp)

set(TARGET_INCLUDE_DIRS gen_code
                        common
                        common/test_drivers
                        include
                        ../../submodules/Device)

set(GPU_TARGET gpu_part)
set(GPU_TARGET_INCLUDE_DIRS ../../submodules/Device
                            include
                            common
                            common/test_drivers
                            gencode)

if(${DEVICE_BACKEND} STREQUAL "cuda")
    set(GPU_TARGET_SOURCE_FILES common/test_drivers/simple_driver.cpp
                                ${CMAKE_SOURCE_DIR}/gen_code/main.cu
                                gen_code/kernels.cu
                                include/gemmforge_aux.cu)
    set(GEN_COPY_PRODUCTS ${CMAKE_SOURCE_DIR}/gen_code/main.cu
                          ${CMAKE_SOURCE_DIR}/gen_code/kernels.h
                          ${CMAKE_SOURCE_DIR}/gen_code/kernels.h)
elseif(${DEVICE_BACKEND} STREQUAL "hip")
    set(GPU_TARGET_SOURCE_FILES common/test_drivers/simple_driver.cpp
                                gen_code/main.cpp
                                gen_code/kernels.cpp
                                include/gemmforge_aux.cpp)
    set(GEN_COPY_PRODUCTS ${CMAKE_SOURCE_DIR}/gen_code/kernels.cpp ${CMAKE_SOURCE_DIR}/gen_code/kernels.h)
elseif((${DEVICE_BACKEND} STREQUAL "oneapi") OR (${DEVICE_BACKEND} STREQUAL "hipsycl"))
    set(GPU_TARGET_SOURCE_FILES common/test_drivers/simple_driver.cpp
                                gen_code/kernels.cpp
                                gen_code/main.cpp
                                include/gemmforge_aux_sycl.cpp)
    set(GEN_COPY_PRODUCTS ${CMAKE_SOURCE_DIR}/gen_code/kernels.cpp ${CMAKE_SOURCE_DIR}/gen_code/kernels.h)
else()
    message(FATAL_ERROR "unknown backend, given: ${DEVICE_BACKEND}")
endif()


set(GEN_COPY_FILES ${CMAKE_SOURCE_DIR}/params.yaml ${CMAKE_SOURCE_DIR}/config.yaml)

add_subdirectory(../../submodules/Device build)

add_executable(${TARGET} ${TARGET_SOURCE_FILES})
target_compile_options(${TARGET} PUBLIC "-std=c++11" "-fPIC")
target_compile_definitions(${TARGET} PUBLIC "REAL_SIZE=${REAL_SIZE}")
target_include_directories(${TARGET} PUBLIC ${TARGET_INCLUDE_DIRS})

include(../../cmake/setupGpuTarget.cmake)
target_link_libraries(${TARGET} PRIVATE ${GPU_TARGET} device -lpthread)

if (${DEVICE_BACKEND} STREQUAL "hipsycl")
    find_package(OpenMP REQUIRED)
    target_link_libraries(${TARGET} PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(${GPU_TARGET} PUBLIC OpenMP::OpenMP_CXX)
endif()

enable_testing()
add_test(NAME test COMMAND ${CMAKE_PROJECT_NAME})
