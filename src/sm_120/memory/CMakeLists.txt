cmake_minimum_required(VERSION 3.18)
project(gpupeek_memory LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 90)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall,-Wextra -O3 --expt-relaxed-constexpr")

set(BENCHMARK_SOURCES
    memory_research_kernel.cu
    memory_research_benchmarks.cu
)

add_executable(gpupeek_memory
    main.cu
    ${BENCHMARK_SOURCES}
)

# Use CMAKE_CURRENT_SOURCE_DIR to reference project files
get_filename_component(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)

target_include_directories(gpupeek_memory PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_ROOT}/src/common
    ${PROJECT_ROOT}/src/sm_120
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(gpupeek_memory PRIVATE
    ${CUDAToolkit_LIBRARIES}
)
