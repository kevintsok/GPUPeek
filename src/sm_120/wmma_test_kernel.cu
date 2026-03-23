#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

// =============================================================================
// WMMA Research Kernels - 基于PTX ISA的WMMA实现
// =============================================================================
//
// 参考: PTX ISA Section 9.7.14 (Warp-Level Matrix Instructions)
//
// WMMA m16n16k16 操作需要:
// - 8个 .b32 寄存器存储结果/输入
// - wmma.load.c.sync 加载累加器矩阵C
// - wmma.mma.sync 执行矩阵乘加
// - wmma.store.d.sync 存储结果矩阵D
//
// C++ API 使用 nvcuda::wmma 命名空间
// =============================================================================

using namespace nvcuda::wmma;

// =============================================================================
// WMMA FP16 Kernel (m16n16k16) - 简化版
// =============================================================================

__global__ void wmma_fp16_test_kernel(const __half* a, const __half* b, float* d,
                                       int M, int N, int K) {
    // WMMA操作要求blockDim.x == 32 (warp size)
    if (threadIdx.x >= 32) return;

    // 每个warp处理一个16x16的输出块
    int warp_id = threadIdx.x;
    int block_row = blockIdx.x;
    int block_col = blockIdx.y;

    // 计算此warp负责的输出块起始位置
    int row_start = block_row * 16;
    int col_start = block_col * 16;

    // 定义fragment类型 - m16n16k16形状
    fragment<matrix_a, 16, 16, 16, __half, row_major> frag_a;
    fragment<matrix_b, 16, 16, 16, __half, col_major> frag_b;
    fragment<accumulator, 16, 16, 16, float> frag_d;

    // 初始化累加器为0
    fill_fragment(frag_d, 0.0f);

    // 遍历K维度
    for (int k = 0; k < K; k += 16) {
        // 加载A矩阵片段到fragment
        load_matrix_sync(frag_a, a + row_start * K + k, K);

        // 加载B矩阵片段到fragment
        load_matrix_sync(frag_b, b + k * N + col_start, N);

        // 执行矩阵乘加
        mma_sync(frag_d, frag_a, frag_b, frag_d);
    }

    // 存储结果
    store_matrix_sync(d + row_start * N + col_start, frag_d, N, mem_row_major);
}
