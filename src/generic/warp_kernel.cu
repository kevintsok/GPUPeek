#pragma once
#include <cuda_runtime.h>

// Warp-level and SM-level benchmark kernels

// Warp vote/allreduce operations
__global__ void warpVoteKernel(const int* __restrict__ input,
                                int* __restrict__ output, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    int val = input[idx];
    bool pred = (val > 0);

    // Warp-level vote operations (use sync variants for sm_70+)
    bool any = __any_sync(0xffffffff, pred);
    bool all = __all_sync(0xffffffff, pred);
    unsigned int ballot = __ballot_sync(0xffffffff, pred);

    output[idx] = (any ? 1 : 0) + (all ? 2 : 0);
}

// Warp shuffle operations
__global__ void warpShuffleKernel(const float* __restrict__ input,
                                   float* __restrict__ output, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tid = threadIdx.x;

    float val = input[idx];

    // Various shuffle operations
    float shfl = __shfl_sync(0xffffffff, val, tid & 31);
    float shfl_up = __shfl_up_sync(0xffffffff, val, 2);
    float shfl_down = __shfl_down_sync(0xffffffff, val, 2);
    float shfl_xor = __shfl_xor_sync(0xffffffff, val, 16);

    // Perform some computation to prevent optimization
    output[idx] = shfl + shfl_up + shfl_down + shfl_xor;
}

// Warp reduction
__global__ void warpReductionKernel(const float* __restrict__ input,
                                     float* __restrict__ output, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tid = threadIdx.x;

    float sum = input[idx];

    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Store result from lane 0
    if ((tid & 31) == 0) {
        output[blockIdx.x] = sum;
    }
}

// SM-level synchronization test
template <int BLOCK_SIZE>
__global__ void smSyncKernel(const int* __restrict__ input,
                              int* __restrict__ output, size_t N) {
    __shared__ int shared[BLOCK_SIZE];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    // Load
    shared[tid] = input[idx];
    __syncthreads();

    // Process
    int sum = 0;
    for (int i = 0; i < BLOCK_SIZE; i++) {
        sum += shared[i];
    }

    if (tid == 0) {
        output[blockIdx.x] = sum;
    }
}

// Occupancy test kernel
template <int BLOCK_SIZE, int NUM_REGISTERS>
__global__ void occupancyTestKernel(const float* __restrict__ input,
                                     float* __restrict__ output, size_t N) {
    // Declare registers to control occupancy
    float reg[NUM_REGISTERS];

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize registers
    #pragma unroll
    for (int i = 0; i < NUM_REGISTERS; i++) {
        reg[i] = input[idx] * (i + 1);
    }

    // Compute
    float sum = 0;
    #pragma unroll
    for (int i = 0; i < NUM_REGISTERS; i++) {
        sum += reg[i];
    }

    output[idx] = sum;
}

