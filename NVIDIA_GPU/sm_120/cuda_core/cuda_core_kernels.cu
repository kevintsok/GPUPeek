#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// =============================================================================
// A.1 Data Type Throughput Kernels
// =============================================================================

// FP64 (Double) arithmetic
template <typename T>
__global__ void fp64ArithmeticKernel(const T* __restrict__ a,
                                      const T* __restrict__ b,
                                      T* __restrict__ c, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        T a_val = a[i];
        T b_val = b[i];
        c[i] = a_val * b_val + a_val;
    }
}

// FP32 (Single) arithmetic
template <typename T>
__global__ void fp32ArithmeticKernel(const T* __restrict__ a,
                                      const T* __restrict__ b,
                                      T* __restrict__ c, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        T a_val = a[i];
        T b_val = b[i];
        c[i] = a_val * b_val + a_val;
    }
}

// FP16 (Half) arithmetic
template <typename T>
__global__ void fp16ArithmeticKernel(const T* __restrict__ a,
                                      const T* __restrict__ b,
                                      T* __restrict__ c, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        T a_val = a[i];
        T b_val = b[i];
        c[i] = __hadd(__hmul(a_val, b_val), a_val);
    }
}

// BF16 (BFloat16) arithmetic
template <typename T>
__global__ void bf16ArithmeticKernel(const T* __restrict__ a,
                                      const T* __restrict__ b,
                                      T* __restrict__ c, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        T a_val = a[i];
        T b_val = b[i];
        // BF16 uses __hmul and __hadd
        c[i] = __hmul(a_val, b_val);
    }
}

// INT8 arithmetic
template <typename T>
__global__ void int8ArithmeticKernel(const T* __restrict__ a,
                                      const T* __restrict__ b,
                                      T* __restrict__ c, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        c[i] = a[i] * b[i] + a[i];
    }
}

// INT32 arithmetic
template <typename T>
__global__ void int32ArithmeticKernel(const T* __restrict__ a,
                                       const T* __restrict__ b,
                                       T* __restrict__ c, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        c[i] = a[i] * b[i] + a[i];
    }
}

// =============================================================================
// A.2 Instruction Latency vs Throughput Kernels
// =============================================================================

// Dependent FMA chain (tests latency)
__global__ void dependentFMAChainKernel(float* __restrict__ data, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        float a = data[i];
        float b = data[i + 1];
        float c = data[i + 2];
        // Chain of 32 dependent FMAs - tests latency
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            c = a * b + c;
        }
        data[i] = c;
    }
}

// Independent FMA throughput (tests throughput)
__global__ void independentFMAThroughputKernel(float* __restrict__ data, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        // Bounds check to prevent illegal memory access
        if (i + stride * 2 >= N) continue;
        float a = data[i];
        float b = data[i + stride];
        float c = data[i + stride * 2];
        // Independent operations can be overlapped
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            a = a * b + c;
        }
        data[i] = a;
    }
}

// =============================================================================
// A.3 Vector Instruction Kernels
// =============================================================================

// Float2 vector operations
__global__ void float2Kernel(const float2* __restrict__ a,
                              const float2* __restrict__ b,
                              float2* __restrict__ c, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        float2 av = a[i];
        float2 bv = b[i];
        float2 cv;
        cv.x = av.x * bv.x + av.x;
        cv.y = av.y * bv.y + av.y;
        c[i] = cv;
    }
}

// Float4 vector operations
__global__ void float4Kernel(const float4* __restrict__ a,
                              const float4* __restrict__ b,
                              float4* __restrict__ c, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        float4 av = a[i];
        float4 bv = b[i];
        float4 cv;
        cv.x = av.x * bv.x + av.x;
        cv.y = av.y * bv.y + av.y;
        cv.z = av.z * bv.z + av.z;
        cv.w = av.w * bv.w + av.w;
        c[i] = cv;
    }
}

// Double2 vector operations
__global__ void double2Kernel(const double2* __restrict__ a,
                               const double2* __restrict__ b,
                               double2* __restrict__ c, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        double2 av = a[i];
        double2 bv = b[i];
        double2 cv;
        cv.x = av.x * bv.x + av.x;
        cv.y = av.y * bv.y + av.y;
        c[i] = cv;
    }
}

// =============================================================================
// A.4 Transcendental Functions Kernels
// =============================================================================

// sin/cos latency test
__global__ void sinCosKernel(float* __restrict__ data, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        float val = data[i];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            val = sinf(val) + cosf(val);
        }
        data[i] = val;
    }
}

// exp/log latency test
__global__ void expLogKernel(float* __restrict__ data, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        float val = data[i];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            val = __expf(val) * __logf(val + 1.0f);
        }
        data[i] = val;
    }
}

// =============================================================================
// A.5 Mixed Precision Kernels
// =============================================================================

// FP32 input -> FP16 compute -> FP32 output
__global__ void mixedPrecisionKernel(const float* __restrict__ fma_in,
                                      const float* __restrict__ fmb_in,
                                      float* __restrict__ fp16_out,
                                      float* __restrict__ fp32_out, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        __half a = __float2half(fma_in[i]);
        __half b = __float2half(fmb_in[i]);
        __half c = __hmul(a, b);
        c = __hadd(c, a);
        fp16_out[i] = c;
        fp32_out[i] = __half2float(c);
    }
}

// =============================================================================
// Additional Helper Kernels
// =============================================================================

// FMA without template specialization
__global__ void fmaKernel(const float* __restrict__ a,
                          const float* __restrict__ b,
                          const float* __restrict__ c,
                          float* __restrict__ out, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        out[i] = a[i] * b[i] + c[i];
    }
}

// FMA64 (FP64) kernel
__global__ void fma64Kernel(const double* __restrict__ a,
                             const double* __restrict__ b,
                             const double* __restrict__ c,
                             double* __restrict__ out, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        out[i] = a[i] * b[i] + c[i];
    }
}
