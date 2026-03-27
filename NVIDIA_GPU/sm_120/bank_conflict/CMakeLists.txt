cmake_minimum_required(VERSION 3.18)
project(gpupeek_bank_conflict LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 90)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/../common)
include_directories(${CMAKE_SOURCE_DIR})

# Executable
add_executable(gpupeek_bank_conflict main.cu)

# CUDA libraries
target_link_libraries(gpupeek_bank_conflict CUDA::cudart)

# Target settings
target_compile_options(gpupeek_bank_conflict PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        -Xcompiler=-Wall
    >
)

# Install
install(TARGETS gpupeek_bank_conflict DESTINATION bin)
