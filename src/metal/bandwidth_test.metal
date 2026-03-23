// Metal Bandwidth Test Kernel
// 研究Apple M系列GPU的内存带宽特性

#include <metal_stdlib>
using namespace metal;

// 简单的内存拷贝内核 - 测试Device to Device带宽
kernel void bandwidth_copy(device const float* src [[buffer(0)]],
                          device float* dst [[buffer(1)]],
                          constant uint& size [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {
    dst[id] = src[id];
}

// 内存设置内核 - 测试内存写入带宽
kernel void bandwidth_set(device float* dst [[buffer(0)]],
                          constant float& value [[buffer(1)]],
                          constant uint& size [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {
    dst[id] = value;
}

// 读取-修改-写入测试 - 测试原子操作和内存带宽
kernel void bandwidth_rmw(device atomic_uint* dst [[buffer(0)]],
                           constant uint& iterations [[buffer(1)]],
                           uint id [[thread_position_in_grid]]) {
    for (uint i = 0; i < iterations; i++) {
        atomic_fetch_add_explicit(&dst[id], 1, memory_order_relaxed);
    }
}

// 向量加法 - 测试计算与内存结合的带宽
kernel void vector_add(device const float* a [[buffer(0)]],
                       device const float* b [[buffer(1)]],
                       device float* result [[buffer(2)]],
                       constant uint& size [[buffer(3)]],
                       uint id [[thread_position_in_grid]]) {
    result[id] = a[id] + b[id];
}

// 矩阵加法 - 测试更大的内存访问模式
kernel void matrix_add(device const float* a [[buffer(0)]],
                       device const float* b [[buffer(1)]],
                       device float* result [[buffer(2)]],
                       constant uint& width [[buffer(3)]],
                       constant uint& height [[buffer(4)]],
                       uint2 gid [[thread_position_in_grid]]) {
    uint id = gid.y * width + gid.x;
    result[id] = a[id] + b[id];
}

// 归约操作 - 测试内存读取带宽
kernel void reduce_max(device const float* src [[buffer(0)]],
                       device float* result [[buffer(1)]],
                       constant uint& size [[buffer(2)]],
                       threadgroup float* shared [[threadgroup(0)]],
                       uint id [[thread_position_in_grid]],
                       uint lid [[thread_position_in_threadgroup]]) {
    // 将数据加载到线程组内存
    shared[lid] = src[id];

    // 同步线程组
    threadgroup_barrier(mem_flags::mem_none);

    // 归约操作
    for (uint s = threadgroup_size / 2; s > 0; s >>= 1) {
        if (lid < s) {
            shared[lid] = fmax(shared[lid], shared[lid + s]);
        }
        threadgroup_barrier(mem_flags::mem_none);
    }

    // 将结果写回
    if (lid == 0) {
        result[0] = fmax(result[0], shared[0]);
    }
}
