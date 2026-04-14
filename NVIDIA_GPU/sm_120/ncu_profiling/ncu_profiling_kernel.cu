#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// =============================================================================
// NCU Profiling 测试内核
// 这些内核专门设计用于 NCU 分析，每个内核测试特定的 GPU 行为
// =============================================================================

/**
 * Memory Bandwidth Kernel - 测试全局内存带宽
 *
 * NCU 分析重点:
 * - sm__throughput.avg.pct_of_peak_sustainedTesla (GPU 利用率)
 * - dram__bytes.sum (内存带宽)
 * - lts__tput.avg.pct_of_peak_sustained (L2 带宽)
 *
 * 预期结果:
 * - 高 GPU 利用率 (~80-100%)
 * - 内存带宽取决于数据大小和访问模式
 */
template <typename T>
__global__ void memoryBandwidthKernel(const T* __restrict__ src,
                                    T* __restrict__ dst, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    // 顺序读写 - 最优内存带宽
    for (size_t i = idx; i < N; i += stride) {
        dst[i] = src[i] * 2.0f;
    }
}

/**
 * Compute Bound Kernel - 计算密集型内核
 *
 * NCU 分析重点:
 * - sm__throughput.avg.pct_of_peak_sustainedTesla
 * - sm__pipe_fp64_cycles_active.pct_of_peak_sustained (FP64 利用率)
 * - sm__pipe_fp32_cycles_active.pct_of_peak_sustained (FP32 利用率)
 *
 * 预期结果:
 * - 高计算利用率
 * - 低内存带宽使用
 */
template <typename T>
__global__ void computeBoundKernel(const T* __restrict__ src,
                                  T* __restrict__ dst, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    // 大量计算来隐藏内存延迟
    for (size_t i = idx; i < N; i += stride) {
        T val = src[i];
        // 16 次 FMA 操作 - 计算密集
        for (int j = 0; j < 16; j++) {
            val = val * 2.0f + 1.0f;
            val = val * 0.5f - 0.5f;
        }
        dst[i] = val;
    }
}

/**
 * Shared Memory Kernel - 测试共享内存带宽和 bank conflict
 *
 * NCU 分析重点:
 * - sm__throughput.avg.pct_of_peak_sustainedTesla
 * - l1tex__average_tilos_elements_per_request_pipe_lsu01 (L1 效率)
 * - sm__pipe_shared_cycles_active.pct_of_peak_sustained (Shared mem 利用率)
 *
 * 预期结果:
 * - Shared memory 带宽约为全局内存的 5-10 倍
 * - 无 bank conflict 时效率最高
 */
template <typename T>
__global__ void sharedMemoryTestKernel(const T* __restrict__ src,
                                     T* __restrict__ dst, size_t N) {
    // 共享内存大小为 256 * sizeof(T)
    __shared__ T shared_buf[256];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    // Phase 1: Global -> Shared
    if (idx < N && tid < 256) {
        shared_buf[tid] = src[idx];
    }
    __syncthreads();

    // Phase 2: Shared -> Global
    if (idx < N && tid < 256) {
        dst[idx] = shared_buf[tid] * 2.0f;
    }
}

/**
 * L2 Cache Streaming Kernel - 测试 L2 缓存效率
 *
 * NCU 分析重点:
 * - lts__tput.avg.pct_of_peak_sustained (L2 带宽利用率)
 * - lts__sectors_cache_lookup.sum (L2 查找)
 * - sm__l1tex__t_sectors_pipe_lsu0_l1_bank0_to_memory_pipeline.misses.sum (L1 miss)
 *
 * 预期结果:
 * - 顺序访问时 L2 命中率高
 * - 跨距访问时 L2 效率下降
 */
template <typename T>
__global__ void l2CacheStreamingKernel(const T* __restrict__ src,
                                     T* __restrict__ dst, size_t N, size_t stride) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t totalThreads = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += totalThreads) {
        T sum = 0;
        // Strided access - 触发 L2
        for (size_t j = 0; j < stride; j++) {
            size_t idx2 = i + j * totalThreads;
            if (idx2 < N) {
                sum += src[idx2];
            }
        }
        if (idx < N) {
            dst[idx] = sum;
        }
    }
}

/**
 * Warp Shuffle Reduction - 测试 warp 级操作效率
 *
 * NCU 分析重点:
 * - sm__pipe_ta_cycles_active.pct_of_peak_sustained (Tensor Engine 利用率)
 * - sm__warp_issue_stalled_by_ barrier.pct_stalled (Barrier stall)
 * - sm__average_active_warps_per_sm_if_sustained (平均活跃 warp 数)
 *
 * 预期结果:
 * - 高 warp 效率
 * - 低 stall 率
 */
template <typename T>
__global__ void warpShuffleReduceKernel(const T* __restrict__ input,
                                       T* __restrict__ output, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    T val = input[idx];

    // Butterfly reduction pattern - 最优 shuffle 效率
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    // 每个 warp 的第一个线程写入结果
    if ((threadIdx.x & 31) == 0) {
        output[blockIdx.x] = val;
    }
}

/**
 * Atomic Operations Kernel - 测试原子操作性能
 *
 * NCU 分析重点:
 * - sm__pipe_exe_cycles_active.pct_of_peak_sustained (执行单元利用率)
 * - sm__average_active_warps_per_sm_if_sustained
 * - sm__warp_issue_stalled_by_ dependency.not_selected.pct_stalled
 *
 * 预期结果:
 * - 原子操作会成为瓶颈
 * - 低 warp 效率因为等待原子操作完成
 */
__global__ void atomicAddTestKernel(const float* __restrict__ src,
                                   float* __restrict__ result, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    // 每个线程计算部分和
    float sum = 0.0f;
    for (size_t i = idx; i < N; i += stride) {
        sum += src[i];
    }

    // 原子加 - 这里会是瓶颈
    atomicAdd(result, sum);
}

/**
 * Branch Divergence Kernel - 测试分支分歧影响
 *
 * NCU 分析重点:
 * - sm__pipe_cpf_cycles_active.pct_of_peak_sustained (CPF 利用率)
 * - sm__warp_divergence_efficiency (Warp 分歧效率)
 * - sm__average_active_warps_per_sm_if_sustained
 *
 * 预期结果:
 * - 高分歧导致低 warp 效率
 * - 分歧 warp 只能串行执行两条路径
 */
template <typename T>
__global__ void branchDivergenceKernel(const int* __restrict__ pred,
                                      T* __restrict__ dst, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    // 高分歧: 每个线程在不同时间取不同分支
    for (size_t i = idx; i < N; i += stride) {
        if (pred[idx & 1]) {  // 相邻线程取不同分支
            dst[i] = pred[i] * 2.0f;
        } else {
            dst[i] = pred[i] * 3.0f;
        }
    }
}

/**
 * Coalesced Memory Access - 测试合并内存访问
 *
 * NCU 分析重点:
 * - dram__bytes.sum (内存带宽)
 * - sm__l1tex__t_sectors_pipe_lsu0_l1_bank0_to_memory_pipeline.misses.sum (L1 miss)
 * - sm__throughput.avg.pct_of_peak_sustainedTesla
 *
 * 预期结果:
 * - 完全合并访问时内存带宽最高
 * - 非合并访问时带宽急剧下降
 */
template <typename T>
__global__ void coalescedAccessKernel(const T* __restrict__ src,
                                     T* __restrict__ dst, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 每个线程处理连续内存 - 完全合并
    for (size_t i = idx; i < N; i += blockDim.x * gridDim.x) {
        dst[i] = src[i] * 2.0f;
    }
}

/**
 * Non-Coalesced Access - 测试非合并内存访问
 *
 * NCU 分析重点:
 * - dram__bytes.sum
 * - l1tex__average_tilos_elements_per_request_pipe_lsu01 (L1 效率低)
 * - sm__throughput.avg.pct_of_peak_sustainedTesla (利用率低)
 *
 * 预期结果:
 * - 带宽显著下降
 * - 更多的内存事务
 */
template <typename T>
__global__ void nonCoalescedAccessKernel(const T* __restrict__ src,
                                        T* __restrict__ dst, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 线程 idx 访问 src[idx*stride] - 非合并
    size_t stride = 17;  // 质数stride制造非合并
    for (size_t i = 0; i < N; i++) {
        size_t src_idx = (idx + i * stride) % N;
        dst[idx * 1024 + i] = src[src_idx] * 2.0f;
    }
}

/**
 * Reduction Tree - 测试树形归约效率
 *
 * NCU 分析重点:
 * - sm__average_active_warps_per_sm_if_sustained
 * - sm__warp_issue_stalled_by_ barrier.pct_stalled
 * - sm__throughput.avg.pct_of_peak_sustainedTesla
 *
 * 预期结果:
 * - 树形归约高效利用 warp
 * - Barrier 使用正确时效率最高
 */
template <typename T>
__global__ void reductionTreeKernel(const T* __restrict__ src,
                                     T* __restrict__ result, size_t N) {
    __shared__ T shared_buf[256];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;
    size_t stride = gridDim.x * blockDim.x;

    // 第一阶段: 计算部分和
    T sum = 0;
    for (size_t i = idx; i < N; i += stride) {
        sum += src[i];
    }
    shared_buf[tid] = sum;
    __syncthreads();

    // 第二阶段: 树形归约
    for (size_t s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            shared_buf[tid] += shared_buf[tid + s];
        }
        __syncthreads();
    }

    // 第三阶段: Warp 内归约 (无需 barrier)
    if (tid < 32) {
        volatile T* smem = shared_buf;
        if (s >= 32) smem[tid] += smem[tid + 32];
        if (s >= 16) smem[tid] += smem[tid + 16];
        if (s >= 8) smem[tid] += smem[tid + 8];
        if (s >= 4) smem[tid] += smem[tid + 4];
        if (s >= 2) smem[tid] += smem[tid + 2];
        if (s >= 1) smem[tid] += smem[tid + 1];
    }

    // 写入结果
    if (tid == 0) {
        result[blockIdx.x] = shared_buf[0];
    }
}

/**
 * Tensor Core Emulation - 模拟 Tensor Core 操作
 *
 * NCU 分析重点:
 * - sm__pipe_tensor_cycles_active.pct_of_peak_sustained (Tensor Engine)
 * - sm__throughput.avg.pct_of_peak_sustainedTesla
 *
 * 注意: 这不是真正的 Tensor Core，只是模拟其工作负载
 */
template <typename T>
__global__ void tensorCoreEmulationKernel(const T* __restrict__ a,
                                           const T* __restrict__ b,
                                           T* __restrict__ c, size_t M, size_t N, size_t K) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        T sum = 0;
        for (size_t k = 0; k < K; k++) {
            // FMA 操作 - 模拟矩阵乘法
            sum += a[row * K + k] * b[k * N + col];
        }
        c[row * N + col] = sum;
    }
}
