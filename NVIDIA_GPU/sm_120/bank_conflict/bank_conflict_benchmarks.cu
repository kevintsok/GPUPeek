#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../../common/timer.h"
#include "bank_conflict_kernel.cu"

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// Forward declarations
extern "C" {
void runBankConflictBasicTests();
void runBankConflictSizeSweep();
void runBankConflictPaddingAnalysis();
void runBankConflictDataTypeAnalysis();
void runBankConflictWarpAnalysis();
}

// =============================================================================
// Utility Functions
// =============================================================================

const char* formatBandwidth(double gb_s) {
    static char buf[32];
    if (gb_s >= 1000.0) {
        snprintf(buf, sizeof(buf), "%.2f GB/s", gb_s);
    } else if (gb_s >= 1.0) {
        snprintf(buf, sizeof(buf), "%.1f GB/s", gb_s);
    } else if (gb_s >= 0.001) {
        snprintf(buf, sizeof(buf), "%.2f MB/s", gb_s * 1000.0);
    } else {
        snprintf(buf, sizeof(buf), "%.2f KB/s", gb_s * 1000.0 * 1000.0);
    }
    return buf;
}

const char* formatTime(double ms) {
    static char buf[32];
    if (ms >= 1000.0) {
        snprintf(buf, sizeof(buf), "%.2f s", ms / 1000.0);
    } else if (ms >= 1.0) {
        snprintf(buf, sizeof(buf), "%.2f ms", ms);
    } else if (ms >= 0.001) {
        snprintf(buf, sizeof(buf), "%.3f us", ms * 1000.0);
    } else {
        snprintf(buf, sizeof(buf), "%.3f ns", ms * 1000.0 * 1000.0);
    }
    return buf;
}

const char* formatBandwidthShort(double gb_s) {
    static char buf[32];
    if (gb_s >= 1000.0) {
        snprintf(buf, sizeof(buf), "%.1f GB/s", gb_s);
    } else if (gb_s >= 1.0) {
        snprintf(buf, sizeof(buf), "%.0f MB/s", gb_s * 1000.0);
    } else {
        snprintf(buf, sizeof(buf), "%.0f KB/s", gb_s * 1000.0 * 1000.0);
    }
    return buf;
}

// =============================================================================
// 1. Basic Bank Conflict Tests
// =============================================================================

void runBankConflictBasicTests() {
    printf("\n");
    printf("================================================================================\n");
    printf("1. Basic Bank Conflict Analysis\n");
    printf("================================================================================\n");

    const size_t N = 4 * 1024 * 1024;
    const int iterations = 100;
    const int blockSize = 256;

    int numBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535;

    float *d_dst;
    CHECK_CUDA(cudaMalloc(&d_dst, N * sizeof(float)));

    printf("\n--- Strided Read Bank Conflict Analysis ---\n\n");
    printf("RTX 5080 (Blackwell SM 12.0): 32 banks, 4 bytes/bank\n");
    printf("When stride % 32 == 0, all threads hit same bank = maximum conflict\n\n");

    printf("%-10s %-15s %-12s %-15s\n", "Stride", "Bandwidth", "Rel. Perf.", "Bank Conflict");
    printf("%-10s %-15s %-12s %-15s\n", "------", "----------", "----------", "-------------");

    // Baseline: sequential (stride=1)
    GPUTimer timer;
    timer.start();
    for (int i = 0; i < iterations; i++) {
        stridedReadKernel<float><<<numBlocks, blockSize>>>(d_dst, N, 1);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double baseline = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);
    printf("%-10d %-15s %-12s %-15s\n", 1, formatBandwidth(baseline), "100%", "None");

    // Strides that cause bank conflicts
    int strides[] = {2, 4, 8, 16, 32, 64, 128};
    double results[7];
    results[0] = baseline;

    for (int s = 1; s < 7; s++) {
        int stride = strides[s];
        timer.start();
        for (int i = 0; i < iterations; i++) {
            stridedReadKernel<float><<<numBlocks, blockSize>>>(d_dst, N, stride);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        timer.stop();

        double bw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);
        results[s] = bw;
        double rel_perf = bw / baseline * 100.0;

        const char* conflict_desc;
        if (stride == 32) {
            conflict_desc = "MAXIMUM (all bank 0)";
        } else if (stride >= 16) {
            conflict_desc = "Severe";
        } else if (stride >= 8) {
            conflict_desc = "High";
        } else if (stride >= 4) {
            conflict_desc = "Medium";
        } else {
            conflict_desc = "Low";
        }

        printf("%-10d %-15s %-10.1f%% %-15s\n", stride, formatBandwidth(bw), rel_perf, conflict_desc);
    }

    CHECK_CUDA(cudaFree(d_dst));

    printf("\nKey Findings:\n");
    printf("- Stride = 32: All threads access bank 0 (worst case)\n");
    printf("- Stride = 16: 2 banks involved (bank = tid %% 32, so bank 0 and 16 conflict)\n");
    printf("- Stride = 64: Periodic conflicts due to modulo\n");
}

// =============================================================================
// 2. Write vs Read Conflict Analysis
// =============================================================================

void runBankConflictWriteVsRead() {
    printf("\n");
    printf("================================================================================\n");
    printf("2. Write vs Read Bank Conflict Analysis\n");
    printf("================================================================================\n");

    const size_t N = 4 * 1024 * 1024;
    const int iterations = 100;
    const int blockSize = 256;

    int numBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535;

    float *d_dst;
    CHECK_CUDA(cudaMalloc(&d_dst, N * sizeof(float)));

    printf("\n--- Stride = 32 (Maximum Conflict) ---\n\n");

    printf("%-15s %-15s %-12s\n", "Operation", "Bandwidth", "Relative");
    printf("%-15s %-15s %-12s\n", "---------", "----------", "----------");

    // Write conflict
    GPUTimer timer;
    timer.start();
    for (int i = 0; i < iterations; i++) {
        stridedWriteKernel<float><<<numBlocks, blockSize>>>(d_dst, N, 32);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double write_bw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);

    // Read conflict
    timer.start();
    for (int i = 0; i < iterations; i++) {
        stridedReadKernel<float><<<numBlocks, blockSize>>>(d_dst, N, 32);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double read_bw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);

    printf("%-15s %-15s %-12s\n", "Strided Write", formatBandwidth(write_bw), "100%");
    printf("%-15s %-15s %-10.1f%%\n", "Strided Read", formatBandwidth(read_bw), read_bw / write_bw * 100.0);

    CHECK_CUDA(cudaFree(d_dst));

    printf("\nNote: Write conflicts serialize in commit order; read conflicts may broadcast.\n");
}

// =============================================================================
// 3. Broadcast Efficiency Analysis
// =============================================================================

void runBroadcastAnalysis() {
    printf("\n");
    printf("================================================================================\n");
    printf("3. Broadcast vs Strided Access Analysis\n");
    printf("================================================================================\n");

    const size_t N = 4 * 1024 * 1024;
    const int iterations = 100;
    const int blockSize = 256;

    int numBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535;

    float *d_dst;
    CHECK_CUDA(cudaMalloc(&d_dst, N * sizeof(float)));

    printf("\n--- Broadcast Read (All threads same address) ---\n\n");

    GPUTimer timer;

    // Broadcast read
    timer.start();
    for (int i = 0; i < iterations; i++) {
        broadcastReadKernel<float><<<numBlocks, blockSize>>>(d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double broadcast_bw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);

    // Strided read (conflict)
    timer.start();
    for (int i = 0; i < iterations; i++) {
        stridedReadKernel<float><<<numBlocks, blockSize>>>(d_dst, N, 32);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double strided_bw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);

    printf("%-20s %-15s\n", "Access Pattern", "Bandwidth");
    printf("%-20s %-15s\n", "-------------", "----------");
    printf("%-20s %-15s\n", "Broadcast (same addr)", formatBandwidth(broadcast_bw));
    printf("%-20s %-15s\n", "Strided (stride=32)", formatBandwidth(strided_bw));
    printf("%-20s %-10.1fx\n", "Broadcast Advantage", broadcast_bw / strided_bw);

    CHECK_CUDA(cudaFree(d_dst));
}

// =============================================================================
// 4. Padding Strategy Analysis
// =============================================================================

void runPaddingAnalysis() {
    printf("\n");
    printf("================================================================================\n");
    printf("4. Padding Strategy Analysis\n");
    printf("================================================================================\n");

    const size_t N = 4 * 1024 * 1024;
    const int iterations = 100;
    const int blockSize = 256;

    int numBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535;

    float *d_dst;
    CHECK_CUDA(cudaMalloc(&d_dst, N * sizeof(float)));

    printf("\n--- Padding Effectiveness (Stride=32 conflict case) ---\n\n");

    printf("%-10s %-12s %-15s %-12s %-15s\n",
           "Padding", "Storage", "Bandwidth", "Rel. Perf.", "Conflict");
    printf("%-10s %-12s %-15s %-12s %-15s\n",
           "-------", "-------", "----------", "----------", "---------");

    // No padding (stride=32 causes max conflict)
    GPUTimer timer;
    timer.start();
    for (int i = 0; i < iterations; i++) {
        stridedPaddedReadKernel<float><<<numBlocks, blockSize>>>(d_dst, N, 32, 0);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double no_pad_bw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);
    printf("%-10d %-12s %-15s %-12s %-15s\n",
           0, "1x", formatBandwidth(no_pad_bw), "100%", "Max (bank 0)");

    // Padding = 1 (2x storage, no conflict)
    timer.start();
    for (int i = 0; i < iterations; i++) {
        stridedPaddedReadKernel<float><<<numBlocks, blockSize>>>(d_dst, N, 32, 1);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double pad1_bw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);
    printf("%-10d %-12s %-15s %-10.1f%% %-15s\n",
           1, "2x", formatBandwidth(pad1_bw), pad1_bw / no_pad_bw * 100.0, "None");

    // Padding = 2 (3x storage)
    timer.start();
    for (int i = 0; i < iterations; i++) {
        stridedPaddedReadKernel<float><<<numBlocks, blockSize>>>(d_dst, N, 32, 2);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double pad2_bw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);
    printf("%-10d %-12s %-15s %-10.1f%% %-15s\n",
           2, "3x", formatBandwidth(pad2_bw), pad2_bw / no_pad_bw * 100.0, "None");

    CHECK_CUDA(cudaFree(d_dst));

    printf("\nKey Findings:\n");
    printf("- Padding=1 eliminates bank conflicts by ensuring unique bank access\n");
    printf("- Cost: 2x storage overhead\n");
    printf("- Effective for repeated shared memory access patterns\n");
}

// =============================================================================
// 5. Warp Count Analysis
// =============================================================================

void runWarpCountAnalysis() {
    printf("\n");
    printf("================================================================================\n");
    printf("5. Warp Count vs Bank Conflict Analysis\n");
    printf("================================================================================\n");

    const size_t N = 4 * 1024 * 1024;
    const int iterations = 100;
    const int blockSize = 256;

    int numBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535;

    float *d_dst;
    CHECK_CUDA(cudaMalloc(&d_dst, N * sizeof(float)));

    printf("\n--- Warp Participation vs Stride (Bandwidth in GB/s) ---\n\n");

    printf("%-10s %-12s %-12s %-12s %-12s\n",
           "Stride", "1 Warp", "2 Warps", "4 Warps", "Full Block");
    printf("%-10s %-12s %-12s %-12s %-12s\n",
           "------", "-------", "-------", "-------", "----------");

    int strides[] = {1, 2, 4, 8, 16, 32};
    GPUTimer timer;

    for (int s = 0; s < 6; s++) {
        int stride = strides[s];

        // 1 warp (32 threads)
        timer.start();
        for (int i = 0; i < iterations; i++) {
            singleWarpAccessKernel<float><<<numBlocks * 8, 32>>>(d_dst, N, stride);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        timer.stop();
        double bw1 = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);

        // 2 warps (64 threads)
        timer.start();
        for (int i = 0; i < iterations; i++) {
            dualWarpAccessKernel<float><<<numBlocks * 4, 64>>>(d_dst, N, stride);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        timer.stop();
        double bw2 = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);

        // 4 warps (128 threads)
        timer.start();
        for (int i = 0; i < iterations; i++) {
            fullBlockAccessKernel<float><<<numBlocks * 2, 128>>>(d_dst, N, stride);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        timer.stop();
        double bw4 = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);

        // Full block (256 threads)
        timer.start();
        for (int i = 0; i < iterations; i++) {
            fullBlockAccessKernel<float><<<numBlocks, blockSize>>>(d_dst, N, stride);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        timer.stop();
        double bw256 = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);

        printf("%-10d %-12s %-12s %-12s %-12s\n",
               stride, formatBandwidthShort(bw1), formatBandwidthShort(bw2),
               formatBandwidthShort(bw4), formatBandwidthShort(bw256));
    }

    CHECK_CUDA(cudaFree(d_dst));

    printf("\nKey Findings:\n");
    printf("- More warps = more parallel bank accesses (better utilization)\n");
    printf("- Single warp with stride=32: all 32 threads hit same bank\n");
    printf("- Full block with stride=32: 8 groups of 32 threads, each hitting different bank\n");
}

// =============================================================================
// 6. Data Type Analysis
// =============================================================================

void runDataTypeAnalysis() {
    printf("\n");
    printf("================================================================================\n");
    printf("6. Data Type vs Bank Conflict Analysis\n");
    printf("================================================================================\n");

    const size_t N = 4 * 1024 * 1024;
    const int iterations = 100;
    const int blockSize = 256;

    int numBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535;

    float *d_float;
    double *d_double;
    __half *d_half;
    CHECK_CUDA(cudaMalloc(&d_float, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_double, N * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_half, N * sizeof(__half)));

    printf("\n--- Bank Conflict by Data Type (Stride = 32) ---\n\n");

    printf("%-12s %-15s %-15s %-15s\n",
           "Data Type", "Sequential", "Strided(32)", "Conflict %");
    printf("%-12s %-15s %-15s %-15s\n",
           "---------", "----------", "-----------", "----------");

    GPUTimer timer;

    // Float (4 bytes)
    timer.start();
    for (int i = 0; i < iterations; i++) {
        bankConflictFloat<float><<<numBlocks, blockSize>>>(d_float, N, 1);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    double float_seq = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);

    timer.start();
    for (int i = 0; i < iterations; i++) {
        bankConflictFloat<float><<<numBlocks, blockSize>>>(d_float, N, 32);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    double float_stride = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);

    printf("%-12s %-15s %-15s %-12.1f%%\n",
           "float32", formatBandwidth(float_seq), formatBandwidth(float_stride),
           (1 - float_stride / float_seq) * 100.0);

    // Double (8 bytes) - bank mapping is (addr/2) % 32
    timer.start();
    for (int i = 0; i < iterations; i++) {
        bankConflictDouble<double><<<numBlocks, blockSize>>>(d_double, N, 1);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    double double_seq = N * sizeof(double) * iterations / (timer.elapsed_ms() * 1e6);

    timer.start();
    for (int i = 0; i < iterations; i++) {
        bankConflictDouble<double><<<numBlocks, blockSize>>>(d_double, N, 32);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    double double_stride = N * sizeof(double) * iterations / (timer.elapsed_ms() * 1e6);

    printf("%-12s %-15s %-15s %-12.1f%%\n",
           "float64", formatBandwidth(double_seq), formatBandwidth(double_stride),
           (1 - double_stride / double_seq) * 100.0);

    // Half (2 bytes)
    timer.start();
    for (int i = 0; i < iterations; i++) {
        bankConflictHalf<__half><<<numBlocks, blockSize>>>(d_half, N, 1);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    double half_seq = N * sizeof(__half) * iterations / (timer.elapsed_ms() * 1e6);

    timer.start();
    for (int i = 0; i < iterations; i++) {
        bankConflictHalf<__half><<<numBlocks, blockSize>>>(d_half, N, 32);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    double half_stride = N * sizeof(__half) * iterations / (timer.elapsed_ms() * 1e6);

    printf("%-12s %-15s %-15s %-12.1f%%\n",
           "float16", formatBandwidth(half_seq), formatBandwidth(half_stride),
           (1 - half_stride / half_seq) * 100.0);

    CHECK_CUDA(cudaFree(d_float));
    CHECK_CUDA(cudaFree(d_double));
    CHECK_CUDA(cudaFree(d_half));

    printf("\nKey Findings:\n");
    printf("- float32: stride=32 causes maximum conflict (all bank 0)\n");
    printf("- float64: stride=32 maps to stride=16 in bank space (2 addr per bank)\n");
    printf("- float16: stride=32 maps to stride=64 in bank space (2x4=8 bytes)\n");
}

// =============================================================================
// 7. Matrix Transpose with/without Padding
// =============================================================================

void runTransposeAnalysis() {
    printf("\n");
    printf("================================================================================\n");
    printf("7. Matrix Transpose: Padding vs No Padding\n");
    printf("================================================================================\n");

    const int TILE = 32;
    const int ROWS = 4096;
    const int COLS = 4096;
    const int iterations = 100;

    size_t size = ROWS * COLS * sizeof(float);
    float *d_src, *d_dst;
    CHECK_CUDA(cudaMalloc(&d_src, size));
    CHECK_CUDA(cudaMalloc(&d_dst, size));

    dim3 blockDim(TILE, TILE);
    dim3 gridDim((COLS + TILE - 1) / TILE, (ROWS + TILE - 1) / TILE);

    printf("\n--- %dx%d Transpose with Tile Size %d ---\n\n",
           ROWS, COLS, TILE);

    GPUTimer timer;

    // With padding (33 columns)
    timer.start();
    for (int i = 0; i < iterations; i++) {
        transposeKernel<float><<<gridDim, blockDim>>>(d_src, d_dst, ROWS, COLS);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double padded_bw = (size * iterations) / (timer.elapsed_ms() * 1e6);

    // Without padding (32 columns)
    timer.start();
    for (int i = 0; i < iterations; i++) {
        transposeNoPaddingKernel<float><<<gridDim, blockDim>>>(d_src, d_dst, ROWS, COLS);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double no_pad_bw = (size * iterations) / (timer.elapsed_ms() * 1e6);

    printf("%-20s %-15s\n", "Method", "Bandwidth");
    printf("%-20s %-15s\n", "------", "----------");
    printf("%-20s %-15s\n", "Transpose + Padding", formatBandwidth(padded_bw));
    printf("%-20s %-15s\n", "Transpose No Padding", formatBandwidth(no_pad_bw));
    printf("%-20s %-10.1f%%\n", "Padding Benefit", padded_bw / no_pad_bw * 100.0 - 100.0);

    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));

    printf("\nKey Findings:\n");
    printf("- Without padding: shared_buf[tid_x][tid_y] causes bank conflict\n");
    printf("- With padding (33 cols): 1-word padding breaks bank mapping\n");
    printf("- Classic CUDA optimization: use padding in transpose kernels\n");
}

// =============================================================================
// 8. Size Sweep Analysis
// =============================================================================

void runSizeSweepAnalysis() {
    printf("\n");
    printf("================================================================================\n");
    printf("8. Bank Conflict vs Data Size Sweep\n");
    printf("================================================================================\n");

    const int iterations = 100;
    const int blockSize = 256;

    size_t sizes[] = {
        1 << 10,    // 1KB
        1 << 12,    // 4KB
        1 << 14,    // 16KB
        1 << 16,    // 64KB
        1 << 18,    // 256KB
        1 << 20,    // 1MB
        1 << 22,    // 4MB
        1 << 24     // 16MB
    };

    printf("\n--- Stride Impact at Different Data Sizes ---\n\n");

    printf("%-10s %-15s %-15s %-15s %-15s\n",
           "Size", "Stride=1", "Stride=8", "Stride=32", "Stride=64");
    printf("%-10s %-15s %-15s %-15s %-15s\n",
           "------", "---------", "---------", "---------", "---------");

    for (size_t s = 0; s < 8; s++) {
        size_t N = sizes[s];
        int numBlocks = (N + blockSize - 1) / blockSize;
        if (numBlocks > 65535) numBlocks = 65535;

        float *d_dst;
        CHECK_CUDA(cudaMalloc(&d_dst, N * sizeof(float)));

        GPUTimer timer;

        // Sequential
        timer.start();
        for (int i = 0; i < iterations; i++) {
            stridedReadKernel<float><<<numBlocks, blockSize>>>(d_dst, N, 1);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        timer.stop();
        double bw1 = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);

        // Stride 8
        timer.start();
        for (int i = 0; i < iterations; i++) {
            stridedReadKernel<float><<<numBlocks, blockSize>>>(d_dst, N, 8);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        timer.stop();
        double bw8 = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);

        // Stride 32
        timer.start();
        for (int i = 0; i < iterations; i++) {
            stridedReadKernel<float><<<numBlocks, blockSize>>>(d_dst, N, 32);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        timer.stop();
        double bw32 = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);

        // Stride 64
        timer.start();
        for (int i = 0; i < iterations; i++) {
            stridedReadKernel<float><<<numBlocks, blockSize>>>(d_dst, N, 64);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        timer.stop();
        double bw64 = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);

        const char* size_label;
        if (N >= 1 << 20) {
            static char buf[16];
            snprintf(buf, sizeof(buf), "%dMB", (int)(N / (1 << 20)));
            size_label = buf;
        } else {
            static char buf[16];
            snprintf(buf, sizeof(buf), "%dKB", (int)(N / (1 << 10)));
            size_label = buf;
        }

        printf("%-10s %-15s %-15s %-15s %-15s\n",
               size_label, formatBandwidthShort(bw1), formatBandwidthShort(bw8),
               formatBandwidthShort(bw32), formatBandwidthShort(bw64));

        CHECK_CUDA(cudaFree(d_dst));
    }

    printf("\nNote: Bank conflicts affect all sizes equally since shared memory is per-block.\n");
}

// =============================================================================
// CSV Data Export for Visualization
// =============================================================================

void exportCSVData() {
    printf("\n");
    printf("================================================================================\n");
    printf("Exporting CSV Data for Visualization\n");
    printf("================================================================================\n");

    const int iterations = 100;
    const int blockSize = 256;
    const size_t N = 4 * 1024 * 1024;

    int numBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535;

    float *d_dst;
    CHECK_CUDA(cudaMalloc(&d_dst, N * sizeof(float)));

    FILE *fp = fopen("bank_conflict_stride_data.csv", "w");
    if (fp) {
        fprintf(fp, "stride,bandwidth_gb_s,relative_perf\n");

        GPUTimer timer;

        // Baseline
        timer.start();
        for (int i = 0; i < iterations; i++) {
            stridedReadKernel<float><<<numBlocks, blockSize>>>(d_dst, N, 1);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        timer.stop();
        double baseline = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);
        fprintf(fp, "1,%.2f,100.0\n", baseline);

        int strides[] = {2, 4, 8, 16, 32, 64, 128};
        for (int s = 0; s < 7; s++) {
            int stride = strides[s];
            timer.start();
            for (int i = 0; i < iterations; i++) {
                stridedReadKernel<float><<<numBlocks, blockSize>>>(d_dst, N, stride);
            }
            CHECK_CUDA(cudaDeviceSynchronize());
            timer.stop();
            double bw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);
            fprintf(fp, "%d,%.2f,%.1f\n", stride, bw, bw / baseline * 100.0);
        }

        fclose(fp);
        printf("Exported: bank_conflict_stride_data.csv\n");
    }

    // Padding data
    fp = fopen("bank_conflict_padding_data.csv", "w");
    if (fp) {
        fprintf(fp, "padding,storage_factor,bandwidth_gb_s,relative_perf\n");

        GPUTimer timer;

        timer.start();
        for (int i = 0; i < iterations; i++) {
            stridedPaddedReadKernel<float><<<numBlocks, blockSize>>>(d_dst, N, 32, 0);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        timer.stop();
        double bw0 = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);
        fprintf(fp, "0,1.0,%.2f,100.0\n", bw0);

        timer.start();
        for (int i = 0; i < iterations; i++) {
            stridedPaddedReadKernel<float><<<numBlocks, blockSize>>>(d_dst, N, 32, 1);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        timer.stop();
        double bw1 = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);
        fprintf(fp, "1,2.0,%.2f,%.1f\n", bw1, bw1 / bw0 * 100.0);

        timer.start();
        for (int i = 0; i < iterations; i++) {
            stridedPaddedReadKernel<float><<<numBlocks, blockSize>>>(d_dst, N, 32, 2);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        timer.stop();
        double bw2 = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);
        fprintf(fp, "2,3.0,%.2f,%.1f\n", bw2, bw2 / bw0 * 100.0);

        fclose(fp);
        printf("Exported: bank_conflict_padding_data.csv\n");
    }

    CHECK_CUDA(cudaFree(d_dst));
}

// =============================================================================
// Main Entry Point
// =============================================================================

void runBankConflictResearchBenchmarks() {
    printf("\n");
    printf("################################################################################\n");
    printf("#                                                                              #\n");
    printf("#          RTX 5080 (Blackwell SM 12.0) Bank Conflict Research                 #\n");
    printf("#                                                                              #\n");
    printf("################################################################################\n");

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

    printf("\nGPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("Shared Memory per Block: %zu bytes\n", prop.sharedMemPerBlock);
    printf("Warp Size: %d threads\n", prop.warpSize);
    printf("Banks: 32 (4 bytes each on Blackwell)\n");

    runBankConflictBasicTests();
    runBankConflictWriteVsRead();
    runBroadcastAnalysis();
    runPaddingAnalysis();
    runWarpCountAnalysis();
    runDataTypeAnalysis();
    runTransposeAnalysis();
    runSizeSweepAnalysis();
    exportCSVData();

    printf("\n");
    printf("================================================================================\n");
    printf("Bank Conflict Research Complete!\n");
    printf("================================================================================\n");
}
