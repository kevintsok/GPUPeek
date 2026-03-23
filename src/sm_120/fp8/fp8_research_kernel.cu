#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

// =============================================================================
// FP8 / TCGen05 Block Scaling Research Kernels
// =============================================================================
//
// PTX ISA Section 9.7.16 - TCGen05 (TensorCore 5th Generation)
//
// FP8 Formats:
// - E4M3 (.e4m3): 4-bit exponent, 3-bit mantissa (range: 0-240)
// - E5M2 (.e5m2): 5-bit exponent, 2-bit mantissa (range: 0-57344)
//
// TCGen05 Features:
// - Block scaling for weight-only quantization
// - FP8 support (.e4m3, .e5m2)
// - Scaled MMA operations
//
// TCGen05.mma variants:
// - tcgen05.mma - basic MMA
// - tcgen05.mma.sp - sparse MMA
// - tcgen05.mma.ws - weight-only scaling
// - tcgen05.mma.ws.sp - weight-only + sparse
//
// Block Scaling:
// - Scale factors per block (32 or 16 elements)
// - W8A16: 8-bit weights, 16-bit activations
// - W8A8: 8-bit weights, 8-bit activations
// =============================================================================

using namespace nvcuda::wmma;

// =============================================================================
// FP8 Format Conversion Kernels
// =============================================================================

// Convert FP32 to FP8 E4M3 format
__device__ unsigned char fp32_to_e4m3(float val) {
    // E4M3: 4-bit exponent, 3-bit mantissa
    // Range: 0-240 (normalized), special values for inf/nan

    if (val <= 0.0f) return 0;

    // Find exponent and mantissa
    int exp;
    float mant = frexpf(val, &exp);

    // Adjust for E4M3 bias (7)
    exp += 7;

    if (exp < 0) {
        // Denormalized
        return 0;
    } else if (exp > 15) {
        // Overflow - saturate to max
        return 240;
    }

    // Extract 3-bit mantissa
    mant = mant * 8.0f - 0.5f;
    int m = (int)(mant + 0.5f);
    if (m >= 8) m = 7;

    return (unsigned char)((exp << 3) | m);
}

// Convert FP32 to FP8 E5M2 format
__device__ unsigned char fp32_to_e5m2(float val) {
    // E5M2: 5-bit exponent, 2-bit mantissa
    // Range: 0-57344 (normalized)

    if (val <= 0.0f) return 0;

    // Find exponent and mantissa
    int exp;
    float mant = frexpf(val, &exp);

    // Adjust for E5M2 bias (15)
    exp += 15;

    if (exp < 0) {
        return 0;
    } else if (exp > 31) {
        // Overflow - saturate to max
        return 0x7F;  // Max E5M2 value
    }

    // Extract 2-bit mantissa
    mant = mant * 4.0f - 0.5f;
    int m = (int)(mant + 0.5f);
    if (m >= 4) m = 3;

    return (unsigned char)((exp << 2) | m);
}

// =============================================================================
// FP8 Matrix Conversion Kernels
// =============================================================================

template <typename T>
__global__ void convert_to_fp8_e4m3(const float* __restrict__ src,
                                    unsigned char* __restrict__ dst,
                                    size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        dst[idx] = fp32_to_e4m3(src[idx]);
    }
}

template <typename T>
__global__ void convert_to_fp8_e5m2(const float* __restrict__ src,
                                    unsigned char* __restrict__ dst,
                                    size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        dst[idx] = fp32_to_e5m2(src[idx]);
    }
}

// =============================================================================
// Block Scaling Kernels (Weight-Only Quantization)
// =============================================================================

// Block scaling for W8A16 (8-bit weights, 16-bit activations)
// Scale per block of 32 elements
template <typename T>
__global__ void block_scale_quantize_kernel(const float* __restrict__ src,
                                             int8_t* __restrict__ dst,
                                             float* __restrict__ scales,
                                             size_t N, size_t block_dim) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t block_id = idx / block_dim;
    size_t in_block = idx % block_dim;

    if (idx < N) {
        // Find max absolute value in block
        float block_max = 0.0f;
        size_t block_start = block_id * block_dim;
        size_t block_end = min(block_start + block_dim, N);

        // Cooperative max finding
        for (size_t i = block_start; i < block_end; i++) {
            float abs_val = fabsf(src[i]);
            if (abs_val > block_max) block_max = abs_val;
        }

        // Scale factor: max / 127 (for INT8 range)
        float scale = (block_max > 0.0f) ? (127.0f / block_max) : 1.0f;

        // Store scale
        if (in_block == 0) {
            scales[block_id] = 1.0f / scale;  // Store reciprocal for dequant
        }

        // Quantize
        dst[idx] = (int8_t)(src[idx] * scale + 0.5f);
    }
}

// Block scaling for W8A8 (8-bit weights, 8-bit activations)
template <typename T>
__global__ void block_scale_quantize_w8a8_kernel(const float* __restrict__ src,
                                                   int8_t* __restrict__ dst,
                                                   float* __restrict__ scales,
                                                   size_t N, size_t block_dim) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t block_id = idx / block_dim;

    if (idx < N) {
        // Find max for this block
        float block_max = 0.0f;
        size_t block_start = block_id * block_dim;
        size_t block_end = min(block_start + block_dim, N);

        for (size_t i = block_start; i < block_end; i++) {
            float abs_val = fabsf(src[i]);
            if (abs_val > block_max) block_max = abs_val;
        }

        float scale = (block_max > 0.0f) ? (127.0f / block_max) : 1.0f;

        if (idx % block_dim == 0) {
            scales[block_id] = 1.0f / scale;
        }

        dst[idx] = (int8_t)(src[idx] * scale + 0.5f);
    }
}

// =============================================================================
// Block Scaling MMA (W8A16 Pattern)
// =============================================================================

// W8A16: 8-bit weights, 16-bit activations
// Weights are quantized per block, activations are FP16
template <typename T>
__global__ void w8a16_mma_kernel(const int8_t* __restrict__ W,  // Quantized weights
                                   const __half* __restrict__ A,  // FP16 activations
                                   float* __restrict__ D,         // FP32 output
                                   const float* __restrict__ scales,  // Weight scales
                                   size_t M, size_t N, size_t K) {
    const int M_TILE = 16;
    const int N_TILE = 16;
    const int K_TILE = 16;
    const int BLOCK_DIM = 32;  // Block size for scaling

    extern __shared__ char shared_mem[];
    int8_t* sh_w = (int8_t*)shared_mem;
    __half* sh_a = (__half*)&shared_mem[M_TILE * K_TILE * sizeof(__half)];

    size_t block_row = blockIdx.x;
    size_t block_col = blockIdx.y;

    frag_a mat_a;
    frag_b mat_b;
    frag_c mat_c;

    wmma::fill_fragment(mat_c, 0.0f);

    for (int k = 0; k < K; k += K_TILE) {
        // Load quantized weights
        for (int i = 0; i < M_TILE; i++) {
            for (int j = 0; j < K_TILE; j++) {
                size_t gw = (block_row * M_TILE + i) * K + (k + j);
                size_t sw = i * K_TILE + j;
                sh_w[sw] = W[gw];
            }
        }

        // Load FP16 activations
        for (int i = 0; i < K_TILE; i++) {
            for (int j = 0; j < N_TILE; j++) {
                size_t ga = (k + i) * N + (block_col * N_TILE + j);
                size_t sa = i * N_TILE + j;
                sh_a[sa] = A[ga];
            }
        }
        __syncthreads();

        // MMA with FP16 accumulators
        wmma::load_matrix_sync(mat_a, (__half*)sh_w, K_TILE);
        wmma::load_matrix_sync(mat_b, sh_a, N_TILE);
        wmma::mma_sync(mat_c, mat_a, mat_b, mat_c);

        __syncthreads();
    }

    // Get scale for this block
    float scale = scales[block_row];

    // Store with dequantization
    wmma::store_matrix_sync(D + block_row * M_TILE * N + block_col * N_TILE,
                            mat_c, N, wmma::mem_row_major);

    // Apply scale
    size_t tid = threadIdx.x;
    if (tid < 64) {
        size_t row = block_row * M_TILE + (tid / 8);
        size_t col = block_col * N_TILE + (tid % 8);
        if (row < M && col < N) {
            D[row * N + col] *= scale;
        }
    }
}

// =============================================================================
// FP8 MMA Reference Kernels (Using TCGen05-style)
// =============================================================================

// FP8 E4M3 GEMM - approximate implementation
template <typename T>
__global__ void fp8_gemm_e4m3_kernel(const unsigned char* __restrict__ A,
                                       const unsigned char* __restrict__ B,
                                       float* __restrict__ C,
                                       size_t M, size_t N, size_t K) {
    const int M_TILE = 16;
    const int N_TILE = 16;
    const int K_TILE = 16;

    extern __shared__ char shared_mem[];
    unsigned char* sh_a = (unsigned char*)shared_mem;
    unsigned char* sh_b = (unsigned char*)&shared_mem[M_TILE * K_TILE];

    size_t block_row = blockIdx.x;
    size_t block_col = blockIdx.y;

    // Accumulate in FP32
    float acc = 0.0f;

    for (int k = 0; k < K; k += K_TILE) {
        // Load A tile
        for (int i = 0; i < M_TILE; i++) {
            for (int j = 0; j < K_TILE; j++) {
                size_t ga = (block_row * M_TILE + i) * K + (k + j);
                size_t sa = i * K_TILE + j;
                sh_a[sa] = A[ga];
            }
        }

        // Load B tile
        for (int i = 0; i < K_TILE; i++) {
            for (int j = 0; j < N_TILE; j++) {
                size_t gb = (k + i) * N + (block_col * N_TILE + j);
                size_t sb = i * N_TILE + j;
                sh_b[sb] = B[gb];
            }
        }
        __syncthreads();

        // Compute partial sum
        for (int kk = 0; kk < K_TILE; kk++) {
            for (int nn = 0; nn < N_TILE; nn++) {
                // Convert FP8 E4M3 to FP32 and accumulate
                // Simplified: treat as normalized 0-240 range
                float a_val = (float)sh_a[0 * K_TILE + kk] / 240.0f;
                float b_val = (float)sh_b[kk * N_TILE + nn] / 240.0f;
                acc += a_val * b_val;
            }
        }
        __syncthreads();
    }

    if (block_row * M_TILE < M && block_col * N_TILE < N) {
        C[block_row * M_TILE * N + block_col * N_TILE] = acc;
    }
}

// =============================================================================
// FP8 E5M2 GEMM
template <typename T>
__global__ void fp8_gemm_e5m2_kernel(const unsigned char* __restrict__ A,
                                       const unsigned char* __restrict__ B,
                                       float* __restrict__ C,
                                       size_t M, size_t N, size_t K) {
    const int M_TILE = 16;
    const int N_TILE = 16;
    const int K_TILE = 16;

    extern __shared__ char shared_mem[];
    unsigned char* sh_a = (unsigned char*)shared_mem;
    unsigned char* sh_b = (unsigned char*)&shared_mem[M_TILE * K_TILE];

    size_t block_row = blockIdx.x;
    size_t block_col = blockIdx.y;

    float acc = 0.0f;

    for (int k = 0; k < K; k += K_TILE) {
        for (int i = 0; i < M_TILE; i++) {
            for (int j = 0; j < K_TILE; j++) {
                size_t ga = (block_row * M_TILE + i) * K + (k + j);
                size_t sa = i * K_TILE + j;
                sh_a[sa] = A[ga];
            }
        }
        for (int i = 0; i < K_TILE; i++) {
            for (int j = 0; j < N_TILE; j++) {
                size_t gb = (k + i) * N + (block_col * N_TILE + j);
                size_t sb = i * N_TILE + j;
                sh_b[sb] = B[gb];
            }
        }
        __syncthreads();

        for (int kk = 0; kk < K_TILE; kk++) {
            for (int nn = 0; nn < N_TILE; nn++) {
                // E5M2 range: 0-57344, special handling for inf/nan
                float a_val = (float)sh_a[0 * K_TILE + kk] / 57344.0f;
                float b_val = (float)sh_b[kk * N_TILE + nn] / 57344.0f;
                acc += a_val * b_val;
            }
        }
        __syncthreads();
    }

    if (block_row * M_TILE < M && block_col * N_TILE < N) {
        C[block_row * M_TILE * N + block_col * N_TILE] = acc;
    }
}

// =============================================================================
// Weight-Only Quantization with TCGen05 style
// =============================================================================

// Weight-only quantization: quantize weights to INT8, keep activations in FP16
// Use per-block scaling
template <typename T>
__global__ void weight_only_quant_kernel(const float* __restrict__ weights,
                                          int8_t* __restrict__ quantized,
                                          float* __restrict__ scales,
                                          size_t N, size_t block_size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t block_id = idx / block_size;

    if (idx < N) {
        // Cooperative load for scale computation
        __shared__ float block_max;

        if (threadIdx.x == 0) {
            block_max = 0.0f;
            size_t start = block_id * block_size;
            size_t end = min(start + block_size, N);
            for (size_t i = start; i < end; i++) {
                float abs_val = fabsf(weights[i]);
                if (abs_val > block_max) block_max = abs_val;
            }
        }
        __syncthreads();

        // Scale
        float scale = (block_max > 0.0f) ? (127.0f / block_max) : 1.0f;

        // Store scale
        if (threadIdx.x == 0) {
            scales[block_id] = 1.0f / scale;
        }
        __syncthreads();

        // Quantize
        quantized[idx] = (int8_t)(weights[idx] * scale + 0.5f);
    }
}

// =============================================================================
// TCGen05-style Block Scaled MMA
// =============================================================================

// TCGen05 MMA with block scaling for inference
// Weights: W8A16 (8-bit quantized, per-block scales)
// Activations: FP16
template <typename T>
__global__ void tcgen05_block_scaled_mma_kernel(const int8_t* __restrict__ W,
                                                   const __half* __restrict__ A,
                                                   float* __restrict__ D,
                                                   const float* __restrict__ scales,
                                                   size_t M, size_t N, size_t K) {
    const int M_TILE = 64;  // TCGen05 uses larger tiles
    const int N_TILE = 64;
    const int K_TILE = 32;  // TCGen05 K=32 for W8A16

    extern __shared__ char shared_mem[];
    int8_t* sh_w = (int8_t*)shared_mem;
    __half* sh_a = (__half*)&shared_mem[M_TILE * K_TILE * sizeof(__half)];

    size_t block_row = blockIdx.x;
    size_t block_col = blockIdx.y;

    frag_a mat_a;
    frag_b mat_b;
    frag_c mat_c;

    wmma::fill_fragment(mat_c, 0.0f);

    for (int k = 0; k < K; k += K_TILE) {
        // Load quantized weights
        for (int i = 0; i < M_TILE; i++) {
            for (int j = 0; j < K_TILE; j++) {
                size_t gw = (block_row * M_TILE + i) * K + (k + j);
                size_t sw = i * K_TILE + j;
                sh_w[sw] = W[gw];
            }
        }

        // Load FP16 activations
        for (int i = 0; i < K_TILE; i++) {
            for (int j = 0; j < N_TILE; j++) {
                size_t ga = (k + i) * N + (block_col * N_TILE + j);
                size_t sa = i * N_TILE + j;
                sh_a[sa] = A[ga];
            }
        }
        __syncthreads();

        wmma::load_matrix_sync(mat_a, (__half*)sh_w, K_TILE);
        wmma::load_matrix_sync(mat_b, sh_a, N_TILE);
        wmma::mma_sync(mat_c, mat_a, mat_b, mat_c);

        __syncthreads();
    }

    // Get scale for this weight block
    float scale = scales[block_row * (M / M_TILE) + block_col];

    // Store
    wmma::store_matrix_sync(D + block_row * M_TILE * N + block_col * N_TILE,
                            mat_c, N, wmma::mem_row_major);

    // Apply block scale (approximation - real TCGen05 has built-in scaling)
    for (int i = 0; i < M_TILE; i++) {
        for (int j = 0; j < N_TILE; j++) {
            size_t idx = (block_row * M_TILE + i) * N + (block_col * N_TILE + j);
            if (idx < M * N) {
                D[idx] *= scale;
            }
        }
    }
}

// =============================================================================
// Baseline: FP32 GEMM for comparison
// =============================================================================

template <typename T>
__global__ void fp32_baseline_gemm_kernel(const float* __restrict__ A,
                                            const float* __restrict__ B,
                                            float* __restrict__ C,
                                            size_t M, size_t N, size_t K) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (size_t k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// =============================================================================
// FP16 Baseline GEMM
// =============================================================================

template <typename T>
__global__ void fp16_baseline_gemm_kernel(const __half* __restrict__ A,
                                           const __half* __restrict__ B,
                                           __half* __restrict__ C,
                                           size_t M, size_t N, size_t K) {
    const int BLOCK_SIZE = 16;

    __shared__ float sh_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sh_b[BLOCK_SIZE][BLOCK_SIZE];

    size_t row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    size_t col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int k = 0; k < K; k += BLOCK_SIZE) {
        // Load A
        if (row < M && (k + threadIdx.x) < K) {
            sh_a[threadIdx.y][threadIdx.x] = __half2float(A[row * K + k + threadIdx.x]);
        } else {
            sh_a[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load B
        if ((k + threadIdx.y) < K && col < N) {
            sh_b[threadIdx.y][threadIdx.x] = __half2float(B[(k + threadIdx.y) * N + col]);
        } else {
            sh_b[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute
        for (int kk = 0; kk < BLOCK_SIZE; kk++) {
            sum += sh_a[threadIdx.y][kk] * sh_b[kk][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = __float2half(sum);
    }
}
