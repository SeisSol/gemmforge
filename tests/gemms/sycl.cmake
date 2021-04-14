set(SOURCE_FILES common/test_drivers/simple_driver_sycl.cpp
               gen_code/kernels.cpp
               include/gemmgen_aux_sycl.cpp)

if (${DEVICE_BACKEND} STREQUAL "HIPSYCL")
    find_package(hipSYCL CONFIG REQUIRED)
    add_sycl_to_target(TARGET device SOURCES ${DEVICE_SOURCE_FILES})
else()
    set(CMAKE_CXX_COMPILER dpcpp)
endif()

add_library(gpu_part SHARED ${SOURCE_FILES})
target_compile_options(gpu_part PRIVATE "-std=c++17" "-O3")
target_compile_definitions(gpu_part PRIVATE DEVICE_${DEVICE_BACKEND}_LANG REAL_SIZE=${REAL_SIZE})
