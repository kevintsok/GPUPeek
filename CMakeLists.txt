cmake_minimum_required(VERSION 3.18)
project(GPUPeek LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 90)  # Default, can be overridden

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall,-Wextra -O3 --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --resource-usage")

# Source files
set(COMMON_SOURCES
    NVIDIA_GPU/common/main.cu
    NVIDIA_GPU/common/gpu_info.cu
)

# Generic kernels (included in main.cu)
# Architecture-specific sources (conditionally included based on target)
set(ARCH_SOURCES
    NVIDIA_GPU/sm_120/arch_kernels.cu
    NVIDIA_GPU/sm_120/benchmarks.cu
    NVIDIA_GPU/sm_120/arch.cu
)

# Target
add_executable(gpupeek
    ${COMMON_SOURCES}
    ${ARCH_SOURCES}
)

target_include_directories(gpupeek PRIVATE
    ${CMAKE_SOURCE_DIR}/NVIDIA_GPU/common
    ${CMAKE_SOURCE_DIR}/NVIDIA_GPU/sm_120
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(gpupeek PRIVATE
    ${CUDAToolkit_LIBRARIES}
)

# Print GPU info at build time
add_custom_command(TARGET gpupeek POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Building for CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}"
)
