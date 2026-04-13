// Metal Compute Test Kernels
// 研究Apple M系列GPU的计算吞吐量

#include <metal_stdlib>
using namespace metal;

// FP32 矩阵乘法 - 测试浮点计算能力
kernel void matmul_fp32(device const float* a [[buffer(0)]],
                        device const float* b [[buffer(1)]],
                        device float* result [[buffer(2)]],
                        constant uint& M [[buffer(3)]],
                        constant uint& K [[buffer(4)]],
                        constant uint& N [[buffer(5)]],
                        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= N || gid.y >= M) return;

    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        uint a_idx = gid.y * K + k;
        uint b_idx = k * N + gid.x;
        sum += a[a_idx] * b[b_idx];
    }

    result[gid.y * N + gid.x] = sum;
}

// FP16 矩阵乘法 - 测试半精度计算
kernel void matmul_fp16(device const half* a [[buffer(0)]],
                         device const half* b [[buffer(1)]],
                         device half* result [[buffer(2)]],
                         constant uint& M [[buffer(3)]],
                         constant uint& K [[buffer(4)]],
                         constant uint& N [[buffer(5)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= N || gid.y >= M) return;

    half sum = 0.0h;
    for (uint k = 0; k < K; k++) {
        uint a_idx = gid.y * K + k;
        uint b_idx = k * N + gid.x;
        sum += a[a_idx] * b[b_idx];
    }

    result[gid.y * N + gid.x] = sum;
}

// SIMD组操作测试 - 测试线程组内的并行度
kernel void simd_group_test(device const float* input [[buffer(0)]],
                            device float* output [[buffer(1)]],
                            constant uint& size [[buffer(2)]],
                            uint id [[thread_position_in_grid]]) {
    // 使用SIMD组操作
    float val = input[id];

    // SIMD族函数
    val = simd_clamp(val, 0.0f, 1.0f);
    val = simd_sin(val);
    val = simd_cos(val);

    // 使用threadgroup_barrier同步
    threadgroup_barrier(mem_flags::mem_none);

    output[id] = val;
}

// 原子操作测试
kernel void atomic_test(device atomic_uint* counter [[buffer(0)]],
                        constant uint& iterations [[buffer(1)]],
                        uint id [[thread_position_in_grid]]) {
    for (uint i = 0; i < iterations; i++) {
        atomic_fetch_add_explicit(&counter[0], 1, memory_order_relaxed);
    }
}

// 归约操作 - 测试内存访问和计算效率
kernel void reduce_sum(device const float* src [[buffer(0)]],
                       device float* result [[buffer(1)]],
                       constant uint& size [[buffer(2)]],
                       threadgroup float* shared [[threadgroup(0)]],
                       uint id [[thread_position_in_grid]],
                       uint lid [[thread_position_in_threadgroup]]) {
    // 加载数据到线程组内存
    shared[lid] = src[id];

    // 同步
    threadgroup_barrier(mem_flags::mem_none);

    // 归约
    for (uint s = threadgroup_size / 2; s > 0; s >>= 1) {
        if (lid < s) {
            shared[lid] += shared[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_none);
    }

    // 写回结果
    if (lid == 0) {
        atomic_fetch_add_explicit((device atomic_uint*)result, (uint)shared[0], memory_order_relaxed);
    }
}

// 混合精度测试
kernel void mixed_precision_test(device const float* a [[buffer(0)]],
                                  device const float* b [[buffer(1)]],
                                  device float* result [[buffer(2)]],
                                  constant uint& size [[buffer(3)]],
                                  uint id [[thread_position_in_grid]]) {
    // FP32到FP16转换计算
    half a_h = (half)a[id];
    half b_h = (half)b[id];

    // 半精度计算
    half r_h = a_h * b_h + 0.5h;

    // 转换回FP32
    result[id] = (float)r_h;
}

// 三角函数测试
kernel void trig_test(device const float* input [[buffer(0)]],
                      device float* output [[buffer(1)]],
                      constant uint& size [[buffer(2)]],
                      uint id [[thread_position_in_grid]]) {
    float x = input[id];

    // 复杂三角计算
    float s = metal::sin(x);
    float c = metal::cos(x);
    float t = metal::tan(x);

    output[id] = s * c + t * 0.001f;
}

// 指数对数测试
kernel void exp_log_test(device const float* input [[buffer(0)]],
                         device float* output [[buffer(1)]],
                         constant uint& size [[buffer(2)]],
                         uint id [[thread_position_in_grid]]) {
    float x = metal::abs(input[id]) + 0.001f;

    float e = metal::exp(x);
    float l = metal::log(x);
    float p = metal::pow(x, 2.5f);

    output[id] = e * 0.001f + l + p * 0.0001f;
}
