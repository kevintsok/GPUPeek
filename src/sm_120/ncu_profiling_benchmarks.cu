#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "../common/timer.h"

// Forward declarations of all NCU profiling kernels
// 这些内核专门设计用于 NCU (Nsight Compute) 分析

// =============================================================================
// 1. Memory Bandwidth Kernels - 测试全局内存带宽
// =============================================================================
// NCU 分析重点:
//   - sm__throughput.avg.pct_of_peak_sustainedTesla (GPU 利用率)
//   - dram__bytes.sum (内存带宽)
//   - lts__tput.avg.pct_of_peak_sustained (L2 带宽)
//
// 教学说明:
//   - dram__bytes.sum 表示从全局内存(DRAM)读取/写入的数据总量
//   - sm__throughput.avg.pct_of_peak_sustainedTesla 是GPU整体利用率
//   - lts__tput.avg.pct_of_peak_sustained 是L2缓存带宽利用率
//   - 理想情况下，我们希望看到高GPU利用率(~80-100%)和高内存带宽

template <typename T>
__global__ void memoryBandwidthKernel(const T* __restrict__ src,
                                    T* __restrict__ dst, size_t N);

// =============================================================================
// 2. Compute Bound Kernels - 计算密集型内核
// =============================================================================
// NCU 分析重点:
//   - sm__throughput.avg.pct_of_peak_sustainedTesla
//   - sm__pipe_fp64_cycles_active.pct_of_peak_sustained (FP64 利用率)
//   - sm__pipe_fp32_cycles_active.pct_of_peak_sustained (FP32 利用率)
//
// 教学说明:
//   - 计算密集型内核应该显示高计算单元利用率
//   - sm__pipe_fp32_cycles_active 表示 FP32 计算单元活跃度
//   - 如果是 FP64 内核，应该看 sm__pipe_fp64_cycles_active
//   - 高计算利用率意味着内核是 compute-bound（计算受限）

template <typename T>
__global__ void computeBoundKernel(const T* __restrict__ src,
                                  T* __restrict__ dst, size_t N);

// =============================================================================
// 3. Shared Memory Kernels - 测试共享内存带宽和 bank conflict
// =============================================================================
// NCU 分析重点:
//   - sm__throughput.avg.pct_of_peak_sustainedTesla
//   - l1tex__average_tilos_elements_per_request_pipe_lsu01 (L1 效率)
//   - sm__pipe_shared_cycles_active.pct_of_peak_sustained (Shared mem 利用率)
//
// 教学说明:
//   - 共享内存是 GPU 上最快的内存，接近 L1 cache
//   - sm__pipe_shared_cycles_active 表示共享内存 pipe 的活跃度
//   - l1tex__average_tilos_elements_per_request 表示 L1 效率
//   - 共享内存带宽约为全局内存的 5-10 倍

template <typename T>
__global__ void sharedMemoryTestKernel(const T* __restrict__ src,
                                     T* __restrict__ dst, size_t N);

// =============================================================================
// 4. L2 Cache Streaming Kernels - 测试 L2 缓存效率
// =============================================================================
// NCU 分析重点:
//   - lts__tput.avg.pct_of_peak_sustained (L2 带宽利用率)
//   - lts__sectors_cache_lookup.sum (L2 查找)
//   - sm__l1tex__t_sectors_pipe_lsu0_l1_bank0_to_memory_pipeline.misses.sum (L1 miss)
//
// 教学说明:
//   - L2 缓存是全局内存和 SM 之间的缓存层
//   - lts__sectors_cache_lookup.sum 显示 L2 查找次数
//   - 如果缓存命中率高，说明数据复用好
//   - 跨距访问(strided access)会导致缓存效率下降

template <typename T>
__global__ void l2CacheStreamingKernel(const T* __restrict__ src,
                                     T* __restrict__ dst, size_t N, size_t stride);

// =============================================================================
// 5. Warp Shuffle Reduction - 测试 warp 级操作效率
// =============================================================================
// NCU 分析重点:
//   - sm__pipe_ta_cycles_active.pct_of_peak_sustained (Tensor Engine 利用率)
//   - sm__warp_issue_stalled_by_ barrier.pct_stalled (Barrier stall)
//   - sm__average_active_warps_per_sm_if_sustained (平均活跃 warp 数)
//
// 教学说明:
//   - Warp shuffle 允许同一 warp 内的线程直接交换数据
//   - 不需要使用共享内存或全局内存
//   - __shfl_down_sync 是 butterfly reduction 模式
//   - 高 warp 效率意味着 warp 分支分歧少

template <typename T>
__global__ void warpShuffleReduceKernel(const T* __restrict__ input,
                                       T* __restrict__ output, size_t N);

// =============================================================================
// 6. Atomic Operations Kernel - 测试原子操作性能
// =============================================================================
// NCU 分析重点:
//   - sm__pipe_exe_cycles_active.pct_of_peak_sustained (执行单元利用率)
//   - sm__average_active_warps_per_sm_if_sustained
//   - sm__warp_issue_stalled_by_ dependency.not_selected.pct_stalled
//
// 教学说明:
//   - 原子操作(atomicAdd)保证数据一致性但会引入延迟
//   - 原子操作会成为瓶颈，因为所有线程需要串行访问
//   - sm__warp_issue_stalled_by_ dependency.not_selected 表示因依赖导致的 stall

__global__ void atomicAddTestKernel(const float* __restrict__ src,
                                   float* __restrict__ result, size_t N);

// =============================================================================
// 7. Branch Divergence Kernel - 测试分支分歧影响
// =============================================================================
// NCU 分析重点:
//   - sm__pipe_cpf_cycles_active.pct_of_peak_sustained (CPF 利用率)
//   - sm__warp_divergence_efficiency (Warp 分歧效率)
//   - sm__average_active_warps_per_sm_if_sustained
//
// 教学说明:
//   - 当 warp 内的线程取不同分支时，会发生分支分歧
//   - 分歧的 warp 只能串行执行两条路径
//   - sm__warp_divergence_efficiency 越接近 100% 越好
//   - pred[idx & 1] 产生相邻线程取不同分支的模式

template <typename T>
__global__ void branchDivergenceKernel(const int* __restrict__ pred,
                                      T* __restrict__ dst, size_t N);

// =============================================================================
// 8. Coalesced Memory Access - 测试合并内存访问
// =============================================================================
// NCU 分析重点:
//   - dram__bytes.sum (内存带宽)
//   - sm__l1tex__t_sectors_pipe_lsu0_l1_bank0_to_memory_pipeline.misses.sum (L1 miss)
//   - sm__throughput.avg.pct_of_peak_sustainedTesla
//
// 教学说明:
//   - 合并访问(Coalesced Access)是 GPU 内存访问的最佳实践
//   - 当连续线程访问连续内存时，GPU 会合并成少数内存事务
//   - 这最大化内存带宽利用率
//   - 非合并访问会大大降低性能

template <typename T>
__global__ void coalescedAccessKernel(const T* __restrict__ src,
                                     T* __restrict__ dst, size_t N);

// =============================================================================
// 9. Non-Coalesced Access - 测试非合并内存访问
// =============================================================================
// NCU 分析重点:
//   - dram__bytes.sum
//   - l1tex__average_tilos_elements_per_request_pipe_lsu01 (L1 效率低)
//   - sm__throughput.avg.pct_of_peak_sustainedTesla (利用率低)
//
// 教学说明:
//   - 非合并访问是性能杀手
//   - stride = 17 (质数) 确保线程访问分散的内存位置
//   - 这会导致更多的内存事务和更低的有效带宽
//   - 在真实应用中应避免这种访问模式

template <typename T>
__global__ void nonCoalescedAccessKernel(const T* __restrict__ src,
                                        T* __restrict__ dst, size_t N);

// =============================================================================
// 10. Reduction Tree - 测试树形归约效率
// =============================================================================
// NCU 分析重点:
//   - sm__average_active_warps_per_sm_if_sustained
//   - sm__warp_issue_stalled_by_ barrier.pct_stalled
//   - sm__throughput.avg.pct_of_peak_sustainedTesla
//
// 教学说明:
//   - 树形归约是并行计算中的经典模式
//   - 分阶段进行，每阶段将数据量减半
//   - 使用 __syncthreads() 同步同 block 内的线程
//   - 最后 32 个线程在 warp 内完成归约(无需 barrier)

template <typename T>
__global__ void reductionTreeKernel(const T* __restrict__ src,
                                     T* __restrict__ result, size_t N);

// =============================================================================
// 11. Tensor Core Emulation - 模拟 Tensor Core 操作
// =============================================================================
// NCU 分析重点:
//   - sm__pipe_tensor_cycles_active.pct_of_peak_sustained (Tensor Engine)
//   - sm__throughput.avg.pct_of_peak_sustainedTesla
//
// 教学说明:
//   - 这是模拟 Tensor Core 工作负载的普通 CUDA 内核
//   - 真正的 Tensor Core 使用 wmma API (computeCapability >= 7.0)
//   - 矩阵乘法是 AI/深度学习中最常见的操作
//   - 真实 Tensor Core 可以达到数百 TFLOPS

template <typename T>
__global__ void tensorCoreEmulationKernel(const T* __restrict__ a,
                                           const T* __restrict__ b,
                                           T* __restrict__ c, size_t M, size_t N, size_t K);


// =============================================================================
// Benchmark Runner Functions - 基准测试运行函数
// =============================================================================

/**
 * 运行所有 NCU Profiling 基准测试
 *
 * 这个函数运行所有为 NCU 分析设计的内核
 * 每个内核都针对特定的 GPU 行为进行测试
 *
 * 使用方法:
 *   ncu --set full ./gpupeek.exe ncu
 *
 * NCU 关键指标解读:
 *   - sm__throughput.avg.pct_of_peak_sustainedTesla: GPU 利用率 (越高越好)
 *   - dram__bytes.sum: 全局内存传输字节数
 *   - lts__tput.avg.pct_of_peak_sustained: L2 带宽利用率
 *   - sm__average_active_warps_per_sm_if_sustained: 平均每 SM 活跃 warp 数
 *   - sm__warp_divergence_efficiency: Warp 分歧效率
 */
void runNCUProfilingBenchmarks(size_t N) {
    printf("\n=== NCU Profiling Benchmarks ===\n");
    printf("这些基准测试专门设计用于 Nsight Compute (NCU) 分析\n");
    printf("每个内核针对特定的 GPU 行为进行测试\n\n");

    const int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535;

    size_t bytes = N * sizeof(float);
    printf("Config: %d blocks x %d threads = %zu threads, %.2f MB\n",
           numBlocks, blockSize, (size_t)numBlocks * blockSize,
           (double)(bytes / (1024.0*1024.0)));

    // 分配内存
    float *d_src, *d_dst, *d_result;
    int *d_pred;
    CHECK_CUDA(cudaMalloc(&d_src, bytes));
    CHECK_CUDA(cudaMalloc(&d_dst, bytes));
    CHECK_CUDA(cudaMalloc(&d_result, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_pred, N * sizeof(int)));

    // 初始化数据
    CHECK_CUDA(cudaMemset(d_src, 1, bytes));
    CHECK_CUDA(cudaMemset(d_dst, 0, bytes));
    CHECK_CUDA(cudaMemset(d_pred, 1, N * sizeof(int)));  // 所有线程走 same branch

    GPUTimer timer;
    const int iterations = 100;

    // =========================================================================
    // Test 1: Memory Bandwidth - 内存带宽测试
    // =========================================================================
    // 教学: 这个测试展示最基础的内存操作 - 顺序读写
    // 期望: 高 GPU 利用率和高内存带宽
    // NCU 指标: dram__bytes.sum 应该等于 2 * bytes * iterations (读+写)
    printf("\n[1/11] Memory Bandwidth Test\n");
    printf("  Purpose: 测试全局内存顺序读写的带宽\n");
    printf("  NCU Metrics: dram__bytes.sum, sm__throughput.avg.pct_of_peak_sustainedTesla\n");
    printf("  Expected: 高内存带宽 (~600-800 GB/s for RTX 5080)\n");

    timer.start();
    for (int i = 0; i < iterations; i++) {
        memoryBandwidthKernel<float><<<numBlocks, blockSize>>>(d_src, d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("  Result: %.2f GB/s (%.3f ms per kernel)\n",
           bytes * iterations / (timer.elapsed_ms() * 1e6),
           timer.elapsed_ms() / iterations);

    // =========================================================================
    // Test 2: Compute Bound - 计算密集型测试
    // =========================================================================
    // 教学: 这个测试展示计算密集型内核的行为
    // 期望: 高计算单元利用率，低内存带宽使用
    // NCU 指标: sm__pipe_fp32_cycles_active.pct_of_peak_sustained 应该高
    printf("\n[2/11] Compute Bound Test\n");
    printf("  Purpose: 测试计算密集型内核的计算单元利用率\n");
    printf("  NCU Metrics: sm__pipe_fp32_cycles_active.pct_of_peak_sustained\n");
    printf("  Expected: 高计算利用率，低内存带宽\n");

    timer.start();
    for (int i = 0; i < iterations; i++) {
        computeBoundKernel<float><<<numBlocks, blockSize>>>(d_src, d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("  Result: %.2f GB/s (%.3f ms per kernel)\n",
           bytes * iterations / (timer.elapsed_ms() * 1e6),
           timer.elapsed_ms() / iterations);

    // =========================================================================
    // Test 3: Shared Memory - 共享内存测试
    // =========================================================================
    // 教学: 共享内存是 GPU 上最快的内存，接近 L1 cache
    // 期望: 共享内存带宽约为全局内存的 5-10 倍
    // NCU 指标: sm__pipe_shared_cycles_active.pct_of_peak_sustained
    printf("\n[3/11] Shared Memory Test\n");
    printf("  Purpose: 测试共享内存(L1 cache)的带宽\n");
    printf("  NCU Metrics: sm__pipe_shared_cycles_active.pct_of_peak_sustained\n");
    printf("  Expected: 共享内存带宽 ~1.5 TB/s (远高于全局内存)\n");

    timer.start();
    for (int i = 0; i < iterations; i++) {
        sharedMemoryTestKernel<float><<<numBlocks, blockSize>>>(d_src, d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("  Result: %.2f GB/s (%.3f ms per kernel)\n",
           bytes * iterations / (timer.elapsed_ms() * 1e6),
           timer.elapsed_ms() / iterations);

    // =========================================================================
    // Test 4: L2 Cache Streaming - L2 缓存流式访问
    // =========================================================================
    // 教学: L2 缓存是全局内存和 SM 之间的缓存层
    // 期望: 顺序访问时 L2 命中率高，带宽接近全局内存
    // NCU 指标: lts__tput.avg.pct_of_peak_sustained, lts__sectors_cache_lookup.sum
    printf("\n[4/11] L2 Cache Streaming Test\n");
    printf("  Purpose: 测试 L2 缓存的流式访问效率\n");
    printf("  NCU Metrics: lts__tput.avg.pct_of_peak_sustained, lts__sectors_cache_lookup.sum\n");
    printf("  Expected: L2 带宽 ~700-800 GB/s\n");

    timer.start();
    for (int i = 0; i < iterations; i++) {
        l2CacheStreamingKernel<float><<<numBlocks, blockSize>>>(d_src, d_dst, N, 1);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("  Result: %.2f GB/s (%.3f ms per kernel)\n",
           bytes * iterations / (timer.elapsed_ms() * 1e6),
           timer.elapsed_ms() / iterations);

    // =========================================================================
    // Test 5: Warp Shuffle Reduction - Warp 级混洗归约
    // =========================================================================
    // 教学: Warp shuffle 允许同一 warp 内的线程直接交换数据，无需共享内存
    // 期望: 高 warp 效率，低 stall 率
    // NCU 指标: sm__average_active_warps_per_sm_if_sustained, warp_divergence_efficiency
    printf("\n[5/11] Warp Shuffle Reduction Test\n");
    printf("  Purpose: 测试 warp 级 shuffle 操作的效率\n");
    printf("  NCU Metrics: sm__average_active_warps_per_sm_if_sustained, warp_divergence_efficiency\n");
    printf("  Expected: 高 warp 效率，低 stall 率\n");

    timer.start();
    for (int i = 0; i < iterations; i++) {
        warpShuffleReduceKernel<float><<<numBlocks, blockSize>>>(d_src, d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("  Result: %.3f ms per kernel\n", timer.elapsed_ms() / iterations);

    // =========================================================================
    // Test 6: Atomic Operations - 原子操作
    // =========================================================================
    // 教学: 原子操作保证数据一致性但会引入延迟，因为需要串行访问
    // 期望: 低 warp 效率，因为线程等待原子操作完成
    // NCU 指标: sm__warp_issue_stalled_by_ dependency.not_selected.pct_stalled
    printf("\n[6/11] Atomic Operations Test\n");
    printf("  Purpose: 测试原子操作对性能的影响\n");
    printf("  NCU Metrics: sm__warp_issue_stalled_by_ dependency.not_selected.pct_stalled\n");
    printf("  Expected: 原子操作成为瓶颈，warp 效率降低\n");

    float h_result = 0.0f;
    CHECK_CUDA(cudaMemcpy(d_result, &h_result, sizeof(float), cudaMemcpyHostToDevice));

    timer.start();
    for (int i = 0; i < iterations; i++) {
        atomicAddTestKernel<<<numBlocks, blockSize>>>(d_src, d_result, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("  Result: %.3f ms per kernel\n", timer.elapsed_ms() / iterations);

    // =========================================================================
    // Test 7: Branch Divergence - 分支分歧
    // =========================================================================
    // 教学: 当 warp 内的线程取不同分支时，会发生分支分歧，需要串行执行
    // 期望: 高分歧导致低 warp 效率
    // NCU 指标: sm__warp_divergence_efficiency, sm__pipe_cpf_cycles_active.pct_of_peak_sustained
    printf("\n[7/11] Branch Divergence Test\n");
    printf("  Purpose: 测试分支分歧对性能的影响\n");
    printf("  NCU Metrics: sm__warp_divergence_efficiency, sm__pipe_cpf_cycles_active.pct_of_peak_sustained\n");
    printf("  Expected: 高分歧导致 warp 效率下降\n");

    timer.start();
    for (int i = 0; i < iterations; i++) {
        branchDivergenceKernel<float><<<numBlocks, blockSize>>>(d_pred, d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("  Result: %.2f GB/s (%.3f ms per kernel)\n",
           bytes * iterations / (timer.elapsed_ms() * 1e6),
           timer.elapsed_ms() / iterations);

    // =========================================================================
    // Test 8: Coalesced Memory Access - 合并内存访问
    // =========================================================================
    // 教学: 合并访问是 GPU 内存访问的最佳实践，当连续线程访问连续内存时
    // 期望: 高内存带宽利用率
    // NCU 指标: dram__bytes.sum, l1tex__average_tilos_elements_per_request_pipe_lsu01
    printf("\n[8/11] Coalesced Memory Access Test\n");
    printf("  Purpose: 测试合并内存访问的效率(最佳实践)\n");
    printf("  NCU Metrics: dram__bytes.sum, l1tex__average_tilos_elements_per_request_pipe_lsu01\n");
    printf("  Expected: 高内存带宽 (~600-800 GB/s)\n");

    timer.start();
    for (int i = 0; i < iterations; i++) {
        coalescedAccessKernel<float><<<numBlocks, blockSize>>>(d_src, d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("  Result: %.2f GB/s (%.3f ms per kernel)\n",
           bytes * iterations / (timer.elapsed_ms() * 1e6),
           timer.elapsed_ms() / iterations);

    // =========================================================================
    // Test 9: Non-Coalesced Access - 非合并内存访问
    // =========================================================================
    // 教学: 非合并访问是性能杀手，应该在真实应用中避免
    // 期望: 带宽显著低于合并访问
    // NCU 指标: dram__bytes.sum (更多内存事务), sm__throughput.avg.pct_of_peak_sustainedTesla (低)
    printf("\n[9/11] Non-Coalesced Memory Access Test\n");
    printf("  Purpose: 测试非合并内存访问的效率(应避免)\n");
    printf("  NCU Metrics: dram__bytes.sum (更多内存事务), sm__throughput (利用率低)\n");
    printf("  Expected: 带宽显著下降 (~100-200 GB/s)\n");

    timer.start();
    for (int i = 0; i < iterations; i++) {
        nonCoalescedAccessKernel<float><<<numBlocks, blockSize>>>(d_src, d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("  Result: %.2f GB/s (%.3f ms per kernel)\n",
           bytes * iterations / (timer.elapsed_ms() * 1e6),
           timer.elapsed_ms() / iterations);

    // =========================================================================
    // Test 10: Reduction Tree - 树形归约
    // =========================================================================
    // 教学: 树形归约是并行计算中的经典模式，分阶段将数据量减半
    // 期望: 树形归约高效利用 warp
    // NCU 指标: sm__average_active_warps_per_sm_if_sustained, barrier stall
    printf("\n[10/11] Reduction Tree Test\n");
    printf("  Purpose: 测试树形归约的并行效率\n");
    printf("  NCU Metrics: sm__average_active_warps_per_sm_if_sustained, barrier stall\n");
    printf("  Expected: 高 warp 利用率，低 barrier stall\n");

    timer.start();
    for (int i = 0; i < iterations; i++) {
        reductionTreeKernel<float><<<numBlocks, blockSize>>>(d_src, d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("  Result: %.3f ms per kernel\n", timer.elapsed_ms() / iterations);

    // =========================================================================
    // Test 11: Tensor Core Emulation - Tensor Core 模拟
    // =========================================================================
    // 教学: 这是模拟 Tensor Core 工作负载的普通 CUDA 内核
    //       真正的 Tensor Core 使用 wmma API，可以达到数百 TFLOPS
    // 期望: 矩阵乘法是 AI/深度学习中最常见的操作
    // NCU 指标: sm__pipe_tensor_cycles_active.pct_of_peak_sustained (如果用真 Tensor Core)
    printf("\n[11/11] Tensor Core Emulation Test\n");
    printf("  Purpose: 模拟矩阵乘法工作负载(AI/深度学习基础)\n");
    printf("  Note: 这是普通 CUDA 内核模拟，不是真正的 Tensor Core\n");
    printf("  Real Tensor Core 使用 wmma API，可达数百 TFLOPS\n");

    // 使用较小的矩阵尺寸以便测试
    size_t M = 256, KK = 256, Ncols = 256;
    float *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc(&d_a, M * KK * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, KK * Ncols * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, M * Ncols * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_a, 1, M * KK * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_b, 2, KK * Ncols * sizeof(float)));

    dim3 blockDim2(16, 16);
    dim3 gridDim2((Ncols + blockDim2.x - 1) / blockDim2.x,
                 (M + blockDim2.y - 1) / blockDim2.y);

    timer.start();
    for (int i = 0; i < iterations; i++) {
        tensorCoreEmulationKernel<float><<<gridDim2, blockDim2>>>(d_a, d_b, d_c, M, Ncols, KK);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("  Result: %.3f ms per kernel (M=%zu, K=%zu, N=%zu)\n",
           timer.elapsed_ms() / iterations, M, KK, Ncols);

    // 清理
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    printf("\n=== NCU Profiling Guide ===\n");
    printf("运行完整 NCU 分析:\n");
    printf("  ncu --set full ./gpupeek.exe ncu\n\n");
    printf("运行特定指标的 NCU 分析:\n");
    printf("  ncu --set full --metrics dram__bytes.sum,sm__throughput.avg.pct_of_peak_sustainedTesla ./gpupeek.exe ncu\n\n");
    printf("导出 NCU 报告:\n");
    printf("  ncu --set full -o ncu_report ./gpupeek.exe ncu\n");
}
