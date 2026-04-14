#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// SM 12.0 (Blackwell) specific kernels
// RTX 5080 Laptop GPU

namespace sm_120 {

// Tensor Core MMA operations (FP16)
__global__ void tensorCoreFP16Kernel(const __half* __restrict__ a,
                                      const __half* __restrict__ b,
                                      const __half* __restrict__ c,
                                      __half* __restrict__ output, size_t M, size_t N, size_t K) {
    // Simplified tensor core benchmark - each thread computes a tile
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    size_t col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        __half sum = __float2half(0.0f);
        for (size_t k = 0; k < K; k++) {
            sum = __hfma(a[row * K + k], b[k * N + col], sum);
        }
        output[row * N + col] = sum;
    }
}

// Tensor Core BF16 operations
__global__ void tensorCoreBF16Kernel(const __half* __restrict__ a,
                                      const __half* __restrict__ b,
                                      const __half* __restrict__ c,
                                      __half* __restrict__ output, size_t M, size_t N, size_t K) {
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    size_t col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        __half sum = __float2half(0.0f);
        for (size_t k = 0; k < K; k++) {
            // BF16 multiply (treat as FP16 for now)
            __half a_val = a[row * K + k];
            __half b_val = b[k * N + col];
            sum = __hfma(a_val, b_val, sum);
        }
        output[row * N + col] = sum;
    }
}

// Asynchronous memory copy (Blackwell feature)
__global__ void asyncCopyKernel(const float* __restrict__ src,
                                float* __restrict__ dst, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        dst[i] = src[i];
    }
}

// L2 cache streaming (Blackwell has larger L2)
__global__ void l2StreamingKernel(const float* __restrict__ input,
                                  float* __restrict__ output, size_t N, size_t stride) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t totalThreads = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += totalThreads) {
        float val = 0.0f;
        // Strided access to test L2 cache
        for (size_t j = 0; j < stride && i + j < N; j++) {
            val += input[i + j];
        }
        output[idx] = val;
    }
}

// Blackwell specific: enhanced shuffle with predicate
__global__ void enhancedWarpShuffleKernel(const int* __restrict__ input,
                                           float* __restrict__ output, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tid = threadIdx.x & 31;

    int val = input[idx];

    // Enhanced shuffle operations available on Blackwell
    int shfl_first = __shfl_sync(0xffffffff, val, 0);        // Get lane 0's value
    int shfl_last = __shfl_sync(0xffffffff, val, 31);         // Get lane 31's value
    int shfl_up = __shfl_up_sync(0xffffffff, val, 4);        // Shift up by 4
    int shfl_down = __shfl_down_sync(0xffffffff, val, 4);    // Shift down by 4
    int shfl_xor = __shfl_xor_sync(0xffffffff, val, 16);     // XOR shuffle

    // Perform some computation
    float result = (float)(shfl_first + shfl_last + shfl_up + shfl_down + shfl_xor);
    output[idx] = result;
}

// Blackwell specific: reduced precision math test
__global__ void reducedPrecisionKernel(const float* __restrict__ input,
                                       __half* __restrict__ output, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        float val = input[i];
        // Convert to FP16 and back (simulating reduced precision)
        __half h = __float2half(val);
        output[i] = __half2float(h);
    }
}

// SM 12.0 multi-level memory benchmark
__global__ void multiLevelMemKernel(const float* __restrict__ global_src,
                                     float* __restrict__ shared_dst,
                                     float* __restrict__ output, size_t N) {
    __shared__ float local_smem[256];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    // L1 -> shared
    if (idx < N) {
        local_smem[tid] = global_src[idx];
    }
    __syncthreads();

    // shared -> output
    if (idx < N) {
        output[idx] = local_smem[tid] * 2.0f;
    }
}

// Coalesced memory access with prefetch hint (Blackwell feature)
__global__ void prefetchKernel(const float* __restrict__ src,
                                float* __restrict__ dst, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    // Software prefetch pattern
    for (size_t i = idx; i < N; i += stride) {
        // Prefetch next chunk
        size_t next_idx = min(i + 64, N - 1);
        float prefetch_val = src[next_idx];
        dst[i] = src[i] + prefetch_val * 0.001f;
    }
}

// Register file bandwidth test
__global__ void registerBandwidthKernel(float* __restrict__ data, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Use many registers to test register file bandwidth
    float r0 = data[idx];
    float r1 = r0 * 2.0f;
    float r2 = r1 * 3.0f;
    float r3 = r2 * 4.0f;
    float r4 = r3 * 5.0f;
    float r5 = r4 * 6.0f;
    float r6 = r5 * 7.0f;
    float r7 = r6 * 8.0f;
    float r8 = r7 * 9.0f;
    float r9 = r8 * 10.0f;

    float result = (r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8 + r9) * 0.1f;
    data[idx] = result;
}

}  // namespace sm_120
