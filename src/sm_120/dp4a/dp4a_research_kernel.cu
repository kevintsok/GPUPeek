#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// =============================================================================
// DP4A (Dot Product of 4 Bytes) Research Kernels
// =============================================================================
//
// PTX ISA Section 9.7.1.23 - Integer Arithmetic Instructions: dp4a
//
// DP4A performs: result = sum(a[i]*b[i]) for i=0..3
//
// Variants:
// - dp4a.type.u32.u8.u8  - unsigned INT8 inputs, 32-bit result
// - dp4a.type.s32.s8.s8  - signed INT8 inputs, 32-bit result
// - dp4a.type.s32.s8.u8  - mixed signed/unsigned
// - dp4a.type.rmi.*      - rounding mode variants
// - dp4a.type.satfinite.* - saturation variants
//
// SASS: DP4A instruction
//
// Use cases:
// - INT8 inference / quantization
// - Neural network fully-connected layers
// - Image processing (pixel convolutions)
// - Recommendation systems
// =============================================================================

// =============================================================================
// DP4A Basic Kernels
// =============================================================================

// DP4A with signed INT8 inputs, signed 32-bit result
template <typename T>
__global__ void dp4a_s32_kernel(const int8_t* __restrict__ a,
                                 const int8_t* __restrict__ b,
                                 int32_t* __restrict__ result,
                                 size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        // Load 4 elements from each vector
        int8_t a0 = a[idx * 4 + 0];
        int8_t a1 = a[idx * 4 + 1];
        int8_t a2 = a[idx * 4 + 2];
        int8_t a3 = a[idx * 4 + 3];

        int8_t b0 = b[idx * 4 + 0];
        int8_t b1 = b[idx * 4 + 1];
        int8_t b2 = b[idx * 4 + 2];
        int8_t b3 = b[idx * 4 + 3];

        // Compute dot product
        int32_t sum = a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;

        result[idx] = sum;
    }
}

// DP4A with unsigned INT8 inputs
template <typename T>
__global__ void dp4a_u32_kernel(const uint8_t* __restrict__ a,
                                 const uint8_t* __restrict__ b,
                                 uint32_t* __restrict__ result,
                                 size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        uint8_t a0 = a[idx * 4 + 0];
        uint8_t a1 = a[idx * 4 + 1];
        uint8_t a2 = a[idx * 4 + 2];
        uint8_t a3 = a[idx * 4 + 3];

        uint8_t b0 = b[idx * 4 + 0];
        uint8_t b1 = b[idx * 4 + 1];
        uint8_t b2 = b[idx * 4 + 2];
        uint8_t b3 = b[idx * 4 + 3];

        uint32_t sum = (uint32_t)a0 * b0 + (uint32_t)a1 * b1 +
                       (uint32_t)a2 * b2 + (uint32_t)a3 * b3;

        result[idx] = sum;
    }
}

// DP4A with saturation (satfinite variant)
template <typename T>
__global__ void dp4a_satfinite_kernel(const int8_t* __restrict__ a,
                                       const int8_t* __restrict__ b,
                                       int32_t* __restrict__ result,
                                       size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        int32_t sum = 0;

        // Inline PTX for dp4a with satfinite:
        // asm volatile("dp4a.s32.satfinite.s32.s8.s8 %0, %1, %2, %3;"
        //              : "=r"(sum)
        //              : "r"(a[idx*4]), "r"(b[idx*4]), "r"(0));

        // Fallback: manual computation with saturation check
        int64_t raw_sum = 0;
        raw_sum += (int64_t)a[idx * 4 + 0] * (int64_t)b[idx * 4 + 0];
        raw_sum += (int64_t)a[idx * 4 + 1] * (int64_t)b[idx * 4 + 1];
        raw_sum += (int64_t)a[idx * 4 + 2] * (int64_t)b[idx * 4 + 2];
        raw_sum += (int64_t)a[idx * 4 + 3] * (int64_t)b[idx * 4 + 3];

        // Clamp to INT32 range
        if (raw_sum > INT32_MAX) raw_sum = INT32_MAX;
        if (raw_sum < INT32_MIN) raw_sum = INT32_MIN;

        result[idx] = (int32_t)raw_sum;
    }
}

// =============================================================================
// DP4A with FMA (Fused Multiply-Add)
// =============================================================================

// DP4A with accumulation (existing value + new dot product)
template <typename T>
__global__ void dp4a_accumulate_kernel(const int8_t* __restrict__ a,
                                        const int8_t* __restrict__ b,
                                        int32_t* __restrict__ result,
                                        size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        int32_t acc = result[idx];

        // dp4a with accumulation
        int8_t a0 = a[idx * 4 + 0];
        int8_t a1 = a[idx * 4 + 1];
        int8_t a2 = a[idx * 4 + 2];
        int8_t a3 = a[idx * 4 + 3];

        int8_t b0 = b[idx * 4 + 0];
        int8_t b1 = b[idx * 4 + 1];
        int8_t b2 = b[idx * 4 + 2];
        int8_t b3 = b[idx * 4 + 3];

        acc += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;

        result[idx] = acc;
    }
}

// =============================================================================
// DP4A Batch Processing (Multiple vectors per thread)
// =============================================================================

template <typename T>
__global__ void dp4a_batch_kernel(const int8_t* __restrict__ a,
                                   const int8_t* __restrict__ b,
                                   int32_t* __restrict__ result,
                                   size_t N, size_t batch_size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        int32_t sum = 0;

        for (size_t b_idx = 0; b_idx < batch_size; b_idx++) {
            size_t offset = (idx * batch_size + b_idx) * 4;

            sum += (int32_t)a[offset + 0] * (int32_t)b[offset + 0];
            sum += (int32_t)a[offset + 1] * (int32_t)b[offset + 1];
            sum += (int32_t)a[offset + 2] * (int32_t)b[offset + 2];
            sum += (int32_t)a[offset + 3] * (int32_t)b[offset + 3];
        }

        result[idx] = sum;
    }
}

// =============================================================================
// DP4A with Vectorized Loading
// =============================================================================

// Pack 4 INT8 into a 32-bit integer for vectorized access
template <typename T>
__global__ void dp4a_packed_kernel(const uint32_t* __restrict__ a_packed,
                                    const uint32_t* __restrict__ b_packed,
                                    int32_t* __restrict__ result,
                                    size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        uint32_t a_pack = a_packed[idx];
        uint32_t b_pack = b_packed[idx];

        // Unpack and compute
        int8_t a0 = (int8_t)(a_pack & 0xFF);
        int8_t a1 = (int8_t)((a_pack >> 8) & 0xFF);
        int8_t a2 = (int8_t)((a_pack >> 16) & 0xFF);
        int8_t a3 = (int8_t)((a_pack >> 24) & 0xFF);

        int8_t b0 = (int8_t)(b_pack & 0xFF);
        int8_t b1 = (int8_t)((b_pack >> 8) & 0xFF);
        int8_t b2 = (int8_t)((b_pack >> 16) & 0xFF);
        int8_t b3 = (int8_t)((b_pack >> 24) & 0xFF);

        result[idx] = a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
    }
}

// =============================================================================
// DP4A with Shared Memory (Block-level reduction)
// =============================================================================

template <typename T>
__global__ void dp4a_shared_kernel(const int8_t* __restrict__ a,
                                   const int8_t* __restrict__ b,
                                   int32_t* __restrict__ result,
                                   size_t N) {
    extern __shared__ int32_t shared_sum[];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;
    size_t block_size = blockDim.x;

    // Each thread computes its dot product
    int32_t local_sum = 0;

    if (idx < N) {
        local_sum = (int32_t)a[idx * 4 + 0] * (int32_t)b[idx * 4 + 0] +
                    (int32_t)a[idx * 4 + 1] * (int32_t)b[idx * 4 + 1] +
                    (int32_t)a[idx * 4 + 2] * (int32_t)b[idx * 4 + 2] +
                    (int32_t)a[idx * 4 + 3] * (int32_t)b[idx * 4 + 3];
    }

    // Store to shared memory
    shared_sum[tid] = local_sum;
    __syncthreads();

    // Parallel reduction in shared memory
    for (size_t s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s && idx + s < N) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    // First thread stores block result
    if (tid == 0) {
        result[blockIdx.x] = shared_sum[0];
    }
}

// =============================================================================
// DP4A Baseline Comparisons
// =============================================================================

// Naive dot product without DP4A
template <typename T>
__global__ void naive_dot4_kernel(const int8_t* __restrict__ a,
                                  const int8_t* __restrict__ b,
                                  int32_t* __restrict__ result,
                                  size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        int32_t sum = 0;

        // Manual 4-term dot product
        sum = a[idx * 4 + 0] * b[idx * 4 + 0];
        sum += a[idx * 4 + 1] * b[idx * 4 + 1];
        sum += a[idx * 4 + 2] * b[idx * 4 + 2];
        sum += a[idx * 4 + 3] * b[idx * 4 + 3];

        result[idx] = sum;
    }
}

// Naive dot product with FP32 accumulation
template <typename T>
__global__ void naive_dot4_fp32_kernel(const int8_t* __restrict__ a,
                                        const int8_t* __restrict__ b,
                                        float* __restrict__ result,
                                        size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float sum = 0;

        sum = (float)a[idx * 4 + 0] * (float)b[idx * 4 + 0];
        sum += (float)a[idx * 4 + 1] * (float)b[idx * 4 + 1];
        sum += (float)a[idx * 4 + 2] * (float)b[idx * 4 + 2];
        sum += (float)a[idx * 4 + 3] * (float)b[idx * 4 + 3];

        result[idx] = sum;
    }
}

// FP32 baseline: 4x FP32 MAD
template <typename T>
__global__ void fp32_mad4_kernel(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ result,
                                  size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float sum = 0;

        sum = a[idx * 4 + 0] * b[idx * 4 + 0];
        sum += a[idx * 4 + 1] * b[idx * 4 + 1];
        sum += a[idx * 4 + 2] * b[idx * 4 + 2];
        sum += a[idx * 4 + 3] * b[idx * 4 + 3];

        result[idx] = sum;
    }
}

// FP16 baseline with conversion
template <typename T>
__global__ void fp16_dot4_kernel(const __half* __restrict__ a,
                                  const __half* __restrict__ b,
                                  float* __restrict__ result,
                                  size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float sum = 0;

        for (int i = 0; i < 4; i++) {
            sum += __half2float(a[idx * 4 + i]) * __half2float(b[idx * 4 + i]);
        }

        result[idx] = sum;
    }
}

// =============================================================================
// DP4A with Mixed Precision (INT8 -> INT32 -> FP32)
// =============================================================================

// Quantized inference pattern: INT8 input -> DP4A -> INT32 -> Dequantize -> FP32
template <typename T>
__global__ void dp4a_quantized_kernel(const int8_t* __restrict__ a,
                                       const int8_t* __restrict__ b,
                                       const float* __restrict__ scale_a,
                                       const float* __restrict__ scale_b,
                                       float* __restrict__ result,
                                       size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        // INT8 dot product
        int32_t dot = (int32_t)a[idx * 4 + 0] * (int32_t)b[idx * 4 + 0] +
                       (int32_t)a[idx * 4 + 1] * (int32_t)b[idx * 4 + 1] +
                       (int32_t)a[idx * 4 + 2] * (int32_t)b[idx * 4 + 2] +
                       (int32_t)a[idx * 4 + 3] * (int32_t)b[idx * 4 + 3];

        // Dequantize: result = dot * scale_a * scale_b
        result[idx] = (float)dot * scale_a[idx] * scale_b[idx];
    }
}

// =============================================================================
// DP4A with Block Scaling (for weight-only quantization)
// =============================================================================

template <typename T>
__global__ void dp4a_block_scale_kernel(const int8_t* __restrict__ a,
                                         const int8_t* __restrict__ b,
                                         const float* __restrict__ block_scale,
                                         float* __restrict__ result,
                                         size_t N, size_t block_dim) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        int32_t dot = 0;

        for (int i = 0; i < 4; i++) {
            dot += (int32_t)a[idx * 4 + i] * (int32_t)b[idx * 4 + i];
        }

        // Apply block scaling
        size_t scale_idx = idx / block_dim;
        result[idx] = (float)dot * block_scale[scale_idx];
    }
}

// =============================================================================
// DP4A with Warp Reduction (Full reduction across warp)
// =============================================================================

template <typename T>
__global__ void dp4a_warp_reduce_kernel(const int8_t* __restrict__ a,
                                          const int8_t* __restrict__ b,
                                          int32_t* __restrict__ result,
                                          size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t tid = threadIdx.x;

    // Compute local dot product
    int32_t local_sum = 0;
    if (idx < N) {
        local_sum = (int32_t)a[idx * 4 + 0] * (int32_t)b[idx * 4 + 0] +
                     (int32_t)a[idx * 4 + 1] * (int32_t)b[idx * 4 + 1] +
                     (int32_t)a[idx * 4 + 2] * (int32_t)b[idx * 4 + 2] +
                     (int32_t)a[idx * 4 + 3] * (int32_t)b[idx * 4 + 3];
    }

    // Warp reduction using shuffle
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
    }

    // First lane stores result
    if (tid % 32 == 0) {
        result[blockIdx.x * (blockDim.x / 32) + tid / 32] = local_sum;
    }
}

// =============================================================================
// DP4A with Tensor Core (INT8 MMA comparison)
// =============================================================================

// Note: Tensor Core INT8 uses MMA instructions, different from DP4A
// This is for comparison purposes

template <typename T>
__global__ void dp4a_vs_mma_kernel(const int8_t* __restrict__ a,
                                     const int8_t* __restrict__ b,
                                     int32_t* __restrict__ result,
                                     size_t M, size_t N, size_t K) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        int32_t sum = 0;

        // Each thread computes partial result (K/4 elements per thread)
        for (size_t k = 0; k < K / 4; k++) {
            int32_t a0 = (int32_t)a[row * K + k * 4 + 0];
            int32_t a1 = (int32_t)a[row * K + k * 4 + 1];
            int32_t a2 = (int32_t)a[row * K + k * 4 + 2];
            int32_t a3 = (int32_t)a[row * K + k * 4 + 3];

            int32_t b0 = (int32_t)b[k * 4 + 0 * N + col];
            int32_t b1 = (int32_t)b[k * 4 + 1 * N + col];
            int32_t b2 = (int32_t)b[k * 4 + 2 * N + col];
            int32_t b3 = (int32_t)b[k * 4 + 3 * N + col];

            sum += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
        }

        result[row * N + col] = sum;
    }
}
