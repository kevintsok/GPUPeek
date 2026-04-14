cmake_minimum_required(VERSION 3.18)
project(gpupeek_redux_sync LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 90)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall,-Wextra -O3 --expt-relaxed-constexpr")

file(GLOB BENCHMARK_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
list(FILTER BENCHMARK_SOURCES EXCLUDE REGEX "main.cu$")

add_executable(gpupeek_redux_sync
    main.cu
    ${BENCHMARK_SOURCES}
)

get_filename_component(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)

target_include_directories(gpupeek_redux_sync PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_ROOT}/NVIDIA_GPU/common
    ${PROJECT_ROOT}/NVIDIA_GPU/sm_120
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(gpupeek_redux_sync PRIVATE
    ${CUDAToolkit_LIBRARIES}
)
