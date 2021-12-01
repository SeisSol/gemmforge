cmake_minimum_required(VERSION 3.5)
project(shrmem-bandwidth-test)

set(DEVICE_ARCH "sm_60" CACHE STRING "size of the floating point data type")
set_property(CACHE DEVICE_ARCH PROPERTY STRINGS sm_60 sm_61 sm_70 sm_71
    gfx906 gfx908
    dg1 bdw skl Gen8 Gen9 Gen11 Gen12LP)

set(DEVICE_BACKEND "cuda" CACHE STRING "type of an interface")
set_property(CACHE DEVICE_BACKEND PROPERTY STRINGS "cuda")
set(REAL_SIZE_IN_BYTES 4)


if(${DEVICE_BACKEND} STREQUAL "cuda")
    find_package(CUDA REQUIRED)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
            -std=c++11;
            -arch=${DEVICE_ARCH};
            -res-usage;
            -O3;
            -lineinfo;
            -Xptxas -v;)

    cuda_add_executable(${CMAKE_PROJECT_NAME} ./main.cpp ./cuda_kernel.cu)
    target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE -DREAL_SIZE=${REAL_SIZE_IN_BYTES})
    set_source_files_properties(./main.cpp PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
else()
    message(FATAL_ERROR "unknown backend, given: ${DEVICE_BACKEND}")
endif()

add_subdirectory(../../submodules/Device build)
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ../../submodules/Device)
target_link_libraries(${CMAKE_PROJECT_NAME} device)

