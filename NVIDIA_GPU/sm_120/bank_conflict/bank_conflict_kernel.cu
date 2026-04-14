#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// =============================================================================
// Bank Conflict Research Kernels
// =============================================================================
//
// Blackwell Architecture (SM 12.0) Bank Configuration:
// - 32 banks, each 4 bytes (32 bits) wide
// - Shared memory: 128KB per block
// - Word addressing: address % 32 determines bank
// - 8-byte (double) access: (address/2) % 32
//
// Key Concepts:
// - Bank conflict: Multiple threads access same bank in same cycle
// - Broadcast: Same data read by multiple threads (no conflict if same address)
// - Double-pump: Two accesses to same bank in one instruction
// =============================================================================

// -----------------------------------------------------------------------------
// 1. Sequential Access - No Conflict Baseline
// -----------------------------------------------------------------------------

// All threads access sequential addresses - NO bank conflicts
template <typename T>
__global__ void sequentialAccessKernel(const T* __restrict__ src,
                                       T* __restrict__ dst, size_t N) {
    __shared__ T shared_buf[256];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    // Sequential load into shared memory
    if (tid < 256 && idx < N) {
        shared_buf[tid] = src[idx];
    }
    __syncthreads();

    // Sequential store from shared memory
    if (idx < N) {
        T sum = 0;
        for (int i = 0; i < 256; i++) {
            sum += shared_buf[i];
        }
        dst[idx] = sum;
    }
}

// -----------------------------------------------------------------------------
// 2. Strided Access - Bank Conflict Patterns
// -----------------------------------------------------------------------------

// Strided read with FIXED number of accesses per thread
// This ensures all strides do the same amount of work
template <typename T>
__global__ void stridedReadKernel(T* __restrict__ dst, size_t N, int stride) {
    __shared__ T shared_buf[256];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    // Initialize - sequential write (no conflict)
    if (tid < 256) {
        shared_buf[tid] = tid;
    }
    __syncthreads();

    // Strided read - FIXED number of iterations, varying stride
    // This tests how stride affects bank conflict
    T sum = 0;
    if (tid < 256) {
        // Each thread does exactly 8 accesses, regardless of stride
        // For stride=1: 8 sequential accesses
        // For stride=32: 8 accesses all to bank 0 (MAX conflict)
        int start = tid * stride;
        for (int j = 0; j < 8; j++) {
            int addr = (start + j * stride) & 255;  // Wrap around 256 elements
            sum += shared_buf[addr];
        }
    }
    __syncthreads();

    if (idx < N) {
        dst[idx] = sum;
    }
}

// Pure strided write to shared memory - measures write conflict cost
template <typename T>
__global__ void stridedWriteKernel(T* __restrict__ dst, size_t N, int stride) {
    __shared__ T shared_buf[256];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    // Initialize
    if (tid < 256) {
        shared_buf[tid] = 0;
    }
    __syncthreads();

    // Strided write - causes bank conflicts
    // Each thread writes to 8 addresses with fixed stride
    if (tid < 256) {
        int start = tid * stride;
        for (int j = 0; j < 8; j++) {
            int addr = (start + j * stride) & 255;
            shared_buf[addr] = tid;
        }
    }
    __syncthreads();

    if (idx < N) {
        dst[idx] = shared_buf[0];
    }
}

// -----------------------------------------------------------------------------
// 3. Broadcast Access - Single Address Read by All Threads
// -----------------------------------------------------------------------------

// All threads read same address - NO conflict (broadcast)
template <typename T>
__global__ void broadcastReadKernel(T* __restrict__ dst, size_t N) {
    __shared__ T shared_buf[256];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    // All threads write same value to different addresses (no conflict)
    if (tid < 256) {
        shared_buf[tid] = tid;
    }
    __syncthreads();

    // All threads read same address (broadcast, no conflict)
    T val = shared_buf[0];

    if (idx < N) {
        dst[idx] = val;
    }
}

// All threads write to same address - conflict but serialized
template <typename T>
__global__ void broadcastWriteKernel(T* __restrict__ dst, size_t N) {
    __shared__ T shared_buf[256];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    // All threads write to same address - atomic-like behavior
    if (tid < 256) {
        shared_buf[0] = tid;
    }
    __syncthreads();

    if (idx < N) {
        dst[idx] = shared_buf[0];
    }
}

// -----------------------------------------------------------------------------
// 4. Padding Strategy - Mitigate Bank Conflicts
// -----------------------------------------------------------------------------

// Padded strided read with fixed work per thread
// Padding breaks bank mapping to eliminate conflicts
template <typename T>
__global__ void stridedPaddedReadKernel(T* __restrict__ dst, size_t N,
                                        int stride, int padding) {
    // With padding, we need more storage
    // padding=0: 256 elements
    // padding=1: 512 elements (2x)
    // padding=2: 768 elements (3x)
    const int STORAGE = 256 * (padding + 1);
    __shared__ T shared_buf[768];  // Max storage for padding=2

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    // Initialize
    for (int i = tid; i < STORAGE; i += 256) {
        shared_buf[i] = i;
    }
    __syncthreads();

    // Padded strided read - same pattern as stridedRead but with padding
    T sum = 0;
    if (tid < STORAGE) {
        // Each thread does 8 accesses with padding
        int step = stride * (padding + 1);  // Padding increases effective stride
        int start = tid * (padding + 1);
        for (int j = 0; j < 8; j++) {
            int addr = (start + j * step) % STORAGE;
            sum += shared_buf[addr];
        }
    }
    __syncthreads();

    if (idx < N) {
        dst[idx] = sum;
    }
}

// Non-padded strided read (control experiment)
template <typename T>
__global__ void stridedNoPaddingReadKernel(T* __restrict__ dst, size_t N, int stride) {
    __shared__ T shared_buf[256];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    // Initialize
    if (tid < 256) {
        shared_buf[tid] = tid;
    }
    __syncthreads();

    // Strided read with fixed work
    T sum = 0;
    if (tid < 256) {
        int start = tid * stride;
        for (int j = 0; j < 8; j++) {
            int addr = (start + j * stride) & 255;
            sum += shared_buf[addr];
        }
    }
    __syncthreads();

    if (idx < N) {
        dst[idx] = sum;
    }
}

// -----------------------------------------------------------------------------
// 5. Warp-Level Bank Conflict Analysis
// -----------------------------------------------------------------------------

// Single warp access pattern (32 threads)
template <typename T>
__global__ void singleWarpAccessKernel(T* __restrict__ dst, size_t N, int stride) {
    __shared__ T shared_buf[256];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    // Only warp 0 participates
    if (tid < 32) {
        // Initialize
        shared_buf[tid] = tid;
    }
    __syncthreads();

    // Warp-strided access
    T sum = 0;
    if (tid < 32) {
        int start = tid * stride;
        for (int j = 0; j < 8; j++) {
            int addr = (start + j * stride) & 255;
            sum += shared_buf[addr];
        }
    }

    if (idx < N) {
        dst[idx] = sum;
    }
}

// Two warp access pattern (64 threads)
template <typename T>
__global__ void dualWarpAccessKernel(T* __restrict__ dst, size_t N, int stride) {
    __shared__ T shared_buf[256];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    // Only first 2 warps participate
    if (tid < 64) {
        shared_buf[tid] = tid;
    }
    __syncthreads();

    T sum = 0;
    if (tid < 64) {
        int start = tid * stride;
        for (int j = 0; j < 8; j++) {
            int addr = (start + j * stride) & 255;
            sum += shared_buf[addr];
        }
    }

    if (idx < N) {
        dst[idx] = sum;
    }
}

// Full block access (256 threads)
template <typename T>
__global__ void fullBlockAccessKernel(T* __restrict__ dst, size_t N, int stride) {
    __shared__ T shared_buf[256];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    if (tid < 256) {
        shared_buf[tid] = tid;
    }
    __syncthreads();

    T sum = 0;
    if (tid < 256) {
        int start = tid * stride;
        for (int j = 0; j < 8; j++) {
            int addr = (start + j * stride) & 255;
            sum += shared_buf[addr];
        }
    }

    if (idx < N) {
        dst[idx] = sum;
    }
}

// -----------------------------------------------------------------------------
// 6. Bank Conflict with Different Data Types
// -----------------------------------------------------------------------------

// 32-bit access (float, int)
template <typename T>
__global__ void bankConflictFloat(T* __restrict__ dst, size_t N, int stride) {
    __shared__ float shared_buf[256];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    if (tid < 256) {
        shared_buf[tid] = tid;
    }
    __syncthreads();

    float sum = 0;
    if (tid < 256) {
        int start = tid * stride;
        for (int j = 0; j < 8; j++) {
            int addr = (start + j * stride) & 255;
            sum += shared_buf[addr];
        }
    }

    if (idx < N) {
        dst[idx] = sum;
    }
}

// 64-bit access (double) - different bank mapping
template <typename T>
__global__ void bankConflictDouble(T* __restrict__ dst, size_t N, int stride) {
    __shared__ double shared_buf[128];  // 128 elements = 1KB

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    if (tid < 128) {
        shared_buf[tid] = tid;
    }
    __syncthreads();

    double sum = 0;
    if (tid < 256) {
        int start = tid * stride;
        // For double, bank = (addr/2) % 32, so stride of 2 maps to bank stride of 1
        for (int j = 0; j < 8; j++) {
            int addr = (start + j * stride) & 127;  // Wrap in 128
            sum += shared_buf[addr];
        }
    }

    if (idx < N) {
        dst[idx] = sum;
    }
}

// 16-bit access (half) - 2 elements per bank typically
template <typename T>
__global__ void bankConflictHalf(T* __restrict__ dst, size_t N, int stride) {
    __shared__ __half shared_buf[512];  // 512 elements = 1KB

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    if (tid < 512) {
        shared_buf[tid] = tid;
    }
    __syncthreads();

    __half sum = 0;
    if (tid < 256) {
        int start = tid * stride;
        // For half, 2 elements per bank, so stride doubles
        for (int j = 0; j < 8; j++) {
            int addr = (start + j * stride * 2) & 511;
            sum = __hadd(sum, shared_buf[addr]);
        }
    }

    if (idx < N) {
        dst[idx] = __half2float(sum);
    }
}

// -----------------------------------------------------------------------------
// 7. Matrix Transpose with Bank Conflict Analysis
// -----------------------------------------------------------------------------

template <typename T>
__global__ void transposeKernel(const T* __restrict__ src, T* __restrict__ dst,
                                 int rows, int cols) {
    __shared__ T tile[32][33];  // 32x32 with padding=1 to avoid bank conflict

    size_t tid_x = threadIdx.x;
    size_t tid_y = threadIdx.y;
    size_t bx = blockIdx.x;
    size_t by = blockIdx.y;

    // Load into shared memory (coalesced)
    size_t row = by * 32 + tid_y;
    size_t col = bx * 32 + tid_x;
    if (row < rows && col < cols) {
        tile[tid_y][tid_x] = src[row * cols + col];
    }
    __syncthreads();

    // Write to global memory (coalesced after transpose)
    row = bx * 32 + tid_y;
    col = by * 32 + tid_x;
    if (row < cols && col < rows) {
        dst[row * rows + col] = tile[tid_x][tid_y];
    }
}

template <typename T>
__global__ void transposeNoPaddingKernel(const T* __restrict__ src, T* __restrict__ dst,
                                          int rows, int cols) {
    __shared__ T tile[32][32];  // 32x32 WITHOUT padding

    size_t tid_x = threadIdx.x;
    size_t tid_y = threadIdx.y;
    size_t bx = blockIdx.x;
    size_t by = blockIdx.y;

    // Load into shared memory (coalesced)
    size_t row = by * 32 + tid_y;
    size_t col = bx * 32 + tid_x;
    if (row < rows && col < cols) {
        tile[tid_y][tid_x] = src[row * cols + col];
    }
    __syncthreads();

    // Write to global memory (coalesced after transpose)
    row = bx * 32 + tid_y;
    col = by * 32 + tid_x;
    if (row < cols && col < rows) {
        dst[row * rows + col] = tile[tid_x][tid_y];  // Bank conflict on read!
    }
}

// -----------------------------------------------------------------------------
// 8. Reduction with Bank Conflicts
// -----------------------------------------------------------------------------

template <typename T>
__global__ void reductionNoConflictKernel(const T* __restrict__ src,
                                           T* __restrict__ dst, size_t N) {
    __shared__ T shared_buf[256];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    // Sequential load
    if (idx < N) {
        shared_buf[tid] = src[idx];
    }
    __syncthreads();

    // Parallel reduction (no conflict with sequential addressing)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_buf[tid] += shared_buf[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        dst[blockIdx.x] = shared_buf[0];
    }
}

template <typename T>
__global__ void reductionWithConflictKernel(const T* __restrict__ src,
                                            T* __restrict__ dst, size_t N,
                                            int stride) {
    __shared__ T shared_buf[256];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    // Strided load (bank conflict)
    if (idx < N) {
        size_t addr = tid * stride;
        if (addr < 256) {
            shared_buf[addr] = src[idx];
        }
    }
    __syncthreads();

    // Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_buf[tid] += shared_buf[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        dst[blockIdx.x] = shared_buf[0];
    }
}

// -----------------------------------------------------------------------------
// 9. Size Sweep Kernels - Same work, different data size
// -----------------------------------------------------------------------------

// Size sweep: tests bank conflict at different data sizes
// All sizes do the same amount of shared memory work
template <typename T>
__global__ void sizeSweepKernel(T* __restrict__ dst, size_t N, int stride) {
    __shared__ T shared_buf[256];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    // Initialize
    if (tid < 256) {
        shared_buf[tid] = tid;
    }
    __syncthreads();

    // Fixed work strided access
    T sum = 0;
    if (tid < 256) {
        int start = tid * stride;
        for (int j = 0; j < 8; j++) {
            int addr = (start + j * stride) & 255;
            sum += shared_buf[addr];
        }
    }
    __syncthreads();

    if (idx < N) {
        dst[idx] = sum;
    }
}
