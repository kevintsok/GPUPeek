import Foundation
import Metal
import QuartzCore
import simd

// MARK: - Timer

func getTimeNanos() -> UInt64 {
    var info = mach_timebase_info_data_t()
    mach_timebase_info(&info)
    let elapsed = mach_absolute_time()
    return elapsed * UInt64(info.numer) / UInt64(info.denom)
}

func getElapsedSeconds(start: UInt64, end: UInt64) -> Double {
    var info = mach_timebase_info_data_t()
    mach_timebase_info(&info)
    let elapsedTicks = Double(end - start)
    let ticksPerNanosec = Double(info.numer) / Double(info.denom)
    return elapsedTicks * ticksPerNanosec / 1e9
}

// MARK: - Deep Research Shader Library

let deepShaderSource = """
#include <metal_stdlib>
using namespace metal;

// ============================================================
// 1. SHARED MEMORY BANK CONFLICT TEST
// Bank conflict occurs when threads in same SIMD-group access
// addresses that map to same bank
// ============================================================

// No bank conflict: each thread accesses its own row
kernel void shared_bank_none(device float* out [[buffer(0)]],
                           threadgroup float* shared [[threadgroup(0)]],
                           constant uint& size [[buffer(1)]],
                           uint id [[thread_position_in_grid]],
                           uint lid [[thread_position_in_threadgroup]]) {
    constexpr uint THREADGROUP_SIZE = 256;
    shared[lid] = float(lid);  // sequential, no conflict
    threadgroup_barrier(mem_flags::mem_none);
    out[id] = shared[lid];
}

// Bank conflict: all threads access same bank (stride 256)
// stride = THREADGROUP_SIZE causes worst-case conflicts
kernel void shared_bank_conflict(device float* out [[buffer(0)]],
                               threadgroup float* shared [[threadgroup(0)]],
                               constant uint& size [[buffer(1)]],
                               uint id [[thread_position_in_grid]],
                               uint lid [[thread_position_in_threadgroup]]) {
    constexpr uint STRIDE = 256;  // same as threadgroup size
    shared[lid * STRIDE] = float(lid);  // all map to same bank
    threadgroup_barrier(mem_flags::mem_none);
    out[id] = shared[lid * STRIDE];
}

// Moderate conflict: threads in same SIMD-group access consecutive
kernel void shared_bank_moderate(device float* out [[buffer(0)]],
                                threadgroup float* shared [[threadgroup(0)]],
                                constant uint& size [[buffer(1)]],
                                uint id [[thread_position_in_grid]],
                                uint lid [[thread_position_in_threadgroup]]) {
    uint simd_lane = lid & 31;  // within 32-thread SIMD group
    shared[simd_lane * 32 + lid] = float(lid);
    threadgroup_barrier(mem_flags::mem_none);
    out[id] = shared[simd_lane * 32 + lid];
}

// ============================================================
// 2. MEMORY COALESCING TEST
// Coalesced: consecutive threads access consecutive addresses
// Non-coalesced: threads access strided addresses
// ============================================================

kernel void coalesced_read(device const float* in [[buffer(0)]],
                         device float* out [[buffer(1)]],
                         constant uint& size [[buffer(2)]],
                         uint id [[thread_position_in_grid]]) {
    out[id] = in[id] * 2.0f;  // sequential access
}

kernel void noncoalesced_read(device const float* in [[buffer(0)]],
                             device float* out [[buffer(1)]],
                             constant uint& size [[buffer(2)]],
                             uint id [[thread_position_in_grid]]) {
    uint stride = 32;
    uint idx = id * stride % (size / 4);
    out[id] = in[idx] * 2.0f;  // strided access
}

kernel void coalesced_write(device float* out [[buffer(0)]],
                          constant uint& size [[buffer(1)]],
                          uint id [[thread_position_in_grid]]) {
    out[id] = float(id);  // sequential write
}

kernel void noncoalesced_write(device float* out [[buffer(0)]],
                              constant uint& size [[buffer(1)]],
                              uint id [[thread_position_in_grid]]) {
    uint stride = 32;
    uint idx = id * stride % (size / 4);
    out[idx] = float(id);  // strided write
}

// ============================================================
// 3. CONSTANT MEMORY TEST
// Constant memory is cached - test vs device memory
// ============================================================

kernel void constant_read(device const float* dev [[buffer(0)]],
                         constant float* cst [[buffer(1)]],
                         device float* out [[buffer(2)]],
                         constant uint& size [[buffer(3)]],
                         uint id [[thread_position_in_grid]]) {
    // All threads read same value - good for constant cache
    out[id] = cst[0] * dev[id];
}

kernel void constant_scattered(device const float* dev [[buffer(0)]],
                               constant float* cst [[buffer(1)]],
                               device float* out [[buffer(2)]],
                               constant uint& size [[buffer(3)]],
                               uint id [[thread_position_in_grid]]) {
    // Threads read different values - bad for constant cache
    out[id] = cst[id % 1024] * dev[id];
}

// ============================================================
// 3b. CONSTANT MEMORY BROADCAST TEST
// ============================================================

kernel void constant_broadcast(device const float* dev [[buffer(0)]],
                             constant float4& cst [[buffer(1)]],
                             device float* out [[buffer(2)]],
                             constant uint& size [[buffer(3)]],
                             uint id [[thread_position_in_grid]]) {
    // All threads read same float4 value - optimal for constant cache
    out[id] = cst[0] + dev[id];
}

// ============================================================
// 4. OPTIMAL TILE SIZE FOR MATMUL
// ============================================================

kernel void matmul_tiled_8(device const float* a [[buffer(0)]],
                         device const float* b [[buffer(1)]],
                         device float* c [[buffer(2)]],
                         constant uint& M [[buffer(3)]],
                         constant uint& K [[buffer(4)]],
                         constant uint& N [[buffer(5)]],
                         threadgroup float* As [[threadgroup(0)]],
                         threadgroup float* Bs [[threadgroup(1)]],
                         uint2 gid [[thread_position_in_grid]],
                         uint2 tid [[thread_position_in_threadgroup]]) {
    constexpr uint TILE_SIZE = 8;
    uint cRow = gid.y;
    uint cCol = gid.x;
    uint cRowTile = tid.y;
    uint cColTile = tid.x;
    float sum = 0.0f;
    for (uint kTile = 0; kTile < (K + TILE_SIZE - 1) / TILE_SIZE; kTile++) {
        uint aRow = cRow;
        uint aCol = kTile * TILE_SIZE + cColTile;
        if (aRow < M && aCol < K) As[cRowTile * TILE_SIZE + cColTile] = a[aRow * K + aCol];
        else As[cRowTile * TILE_SIZE + cColTile] = 0.0f;
        uint bRow = kTile * TILE_SIZE + cRowTile;
        uint bCol = cCol;
        if (bRow < K && bCol < N) Bs[cRowTile * TILE_SIZE + cColTile] = b[bRow * N + bCol];
        else Bs[cRowTile * TILE_SIZE + cColTile] = 0.0f;
        threadgroup_barrier(mem_flags::mem_none);
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += As[cRowTile * TILE_SIZE + k] * Bs[k * TILE_SIZE + cColTile];
        }
        threadgroup_barrier(mem_flags::mem_none);
    }
    if (cRow < M && cCol < N) c[cRow * N + cCol] = sum;
}

kernel void matmul_tiled_16(device const float* a [[buffer(0)]],
                          device const float* b [[buffer(1)]],
                          device float* c [[buffer(2)]],
                          constant uint& M [[buffer(3)]],
                          constant uint& K [[buffer(4)]],
                          constant uint& N [[buffer(5)]],
                          threadgroup float* As [[threadgroup(0)]],
                          threadgroup float* Bs [[threadgroup(1)]],
                          uint2 gid [[thread_position_in_grid]],
                          uint2 tid [[thread_position_in_threadgroup]]) {
    constexpr uint TILE_SIZE = 16;
    uint cRow = gid.y;
    uint cCol = gid.x;
    uint cRowTile = tid.y;
    uint cColTile = tid.x;
    float sum = 0.0f;
    for (uint kTile = 0; kTile < (K + TILE_SIZE - 1) / TILE_SIZE; kTile++) {
        uint aRow = cRow;
        uint aCol = kTile * TILE_SIZE + cColTile;
        if (aRow < M && aCol < K) As[cRowTile * TILE_SIZE + cColTile] = a[aRow * K + aCol];
        else As[cRowTile * TILE_SIZE + cColTile] = 0.0f;
        uint bRow = kTile * TILE_SIZE + cRowTile;
        uint bCol = cCol;
        if (bRow < K && bCol < N) Bs[cRowTile * TILE_SIZE + cColTile] = b[bRow * N + bCol];
        else Bs[cRowTile * TILE_SIZE + cColTile] = 0.0f;
        threadgroup_barrier(mem_flags::mem_none);
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += As[cRowTile * TILE_SIZE + k] * Bs[k * TILE_SIZE + cColTile];
        }
        threadgroup_barrier(mem_flags::mem_none);
    }
    if (cRow < M && cCol < N) c[cRow * N + cCol] = sum;
}

kernel void matmul_tiled_32(device const float* a [[buffer(0)]],
                          device const float* b [[buffer(1)]],
                          device float* c [[buffer(2)]],
                          constant uint& M [[buffer(3)]],
                          constant uint& K [[buffer(4)]],
                          constant uint& N [[buffer(5)]],
                          threadgroup float* As [[threadgroup(0)]],
                          threadgroup float* Bs [[threadgroup(1)]],
                          uint2 gid [[thread_position_in_grid]],
                          uint2 tid [[thread_position_in_threadgroup]]) {
    constexpr uint TILE_SIZE = 32;
    uint cRow = gid.y;
    uint cCol = gid.x;
    uint cRowTile = tid.y;
    uint cColTile = tid.x;
    float sum = 0.0f;
    for (uint kTile = 0; kTile < (K + TILE_SIZE - 1) / TILE_SIZE; kTile++) {
        uint aRow = cRow;
        uint aCol = kTile * TILE_SIZE + cColTile;
        if (aRow < M && aCol < K) As[cRowTile * TILE_SIZE + cColTile] = a[aRow * K + aCol];
        else As[cRowTile * TILE_SIZE + cColTile] = 0.0f;
        uint bRow = kTile * TILE_SIZE + cRowTile;
        uint bCol = cCol;
        if (bRow < K && bCol < N) Bs[cRowTile * TILE_SIZE + cColTile] = b[bRow * N + bCol];
        else Bs[cRowTile * TILE_SIZE + cColTile] = 0.0f;
        threadgroup_barrier(mem_flags::mem_none);
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += As[cRowTile * TILE_SIZE + k] * Bs[k * TILE_SIZE + cColTile];
        }
        threadgroup_barrier(mem_flags::mem_none);
    }
    if (cRow < M && cCol < N) c[cRow * N + cCol] = sum;
}

// ============================================================
// 5. TEXTURE VS BUFFER PERFORMANCE
// ============================================================

// Note: Textures require special setup - testing buffer operations
// but with different access patterns that simulate texture behavior

kernel void buffer_sequential(device const float* in [[buffer(0)]],
                              device float* out [[buffer(1)]],
                              constant uint& size [[buffer(2)]],
                              uint id [[thread_position_in_grid]]) {
    out[id] = in[id] + 1.0f;
}

// Simulate texture gather-like access (non-sequential but localized)
kernel void buffer_gather(device const float* in [[buffer(0)]],
                         device float* out [[buffer(1)]],
                         constant uint& size [[buffer(2)]],
                         uint id [[thread_position_in_grid]]) {
    uint base = (id / 4) * 4;
    float sum = in[base] + in[base + 1] + in[base + 2] + in[base + 3];
    out[id] = sum * 0.25f;
}

// ============================================================
// 5. SIMD-GROUP (WARP) OPERATIONS
// Apple uses SIMD-groups of 32 threads
// Test shuffle and broadcast primitives
// ============================================================

// Broadcast from lane 0 to all in SIMD-group
kernel void simd_broadcast(device const float* in [[buffer(0)]],
                          device float* out [[buffer(1)]],
                          constant uint& size [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {
    uint simd_id = id & ~31;  // SIMD-group base
    uint lane_0 = simd_id;    // lane 0 in group
    float val = in[lane_0];   // read from lane 0
    // simd_broadcast would broadcast val to all lanes
    out[id] = val;
}

// SIMD-prefix sum (work-efficient)
kernel void simd_prefix_sum(device const float* in [[buffer(0)]],
                           device float* out [[buffer(1)]],
                           constant uint& size [[buffer(2)]],
                           uint id [[thread_position_in_grid]]) {
    uint lid = id & 31;
    float val = in[id];
    // HACK: simplified - real implementation needs stage-by-stage
    out[id] = val;
}

// ============================================================
// 6. OCCUPANCY VS PERFORMANCE
// Test the same kernel with different threadgroup sizes
// ============================================================

kernel void occupancy_test(device const float* in [[buffer(0)]],
                          device float* out [[buffer(1)]],
                          threadgroup float* shared [[threadgroup(0)]],
                          constant uint& size [[buffer(2)]],
                          uint id [[thread_position_in_grid]],
                          uint lid [[thread_position_in_threadgroup]]) {
    uint tg_size = 256;  // will vary
    shared[lid] = in[id];
    threadgroup_barrier(mem_flags::mem_none);
    // do some work
    float sum = 0.0f;
    for (uint i = 0; i < 32; i++) {
        sum += shared[(lid + i) & (tg_size - 1)];
    }
    out[id] = sum;
}

// ============================================================
// 7. MEMORY LATENCY HIDING TEST
// How many concurrent threads needed to hide memory latency?
// ============================================================

kernel void latency_hidden(device const float* in [[buffer(0)]],
                          device float* out [[buffer(1)]],
                          constant uint& size [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {
    // Each thread does multiple independent memory ops
    uint idx = id;
    float sum = 0.0f;
    for (uint i = 0; i < 8; i++) {
        sum += in[idx];
        idx = (idx + 1) % (size / 4);
    }
    out[id] = sum;
}

kernel void latency_exposed(device const float* in [[buffer(0)]],
                            device float* out [[buffer(1)]],
                            constant uint& size [[buffer(2)]],
                            uint id [[thread_position_in_grid]]) {
    // Each thread does single memory op, less latency hiding
    out[id] = in[id];
}

// ============================================================
// 8. REGISTER PRESSURE TEST
// Test performance with different register usage
// ============================================================

kernel void register_low(device const float* in [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        constant uint& size [[buffer(2)]],
                        uint id [[thread_position_in_grid]]) {
    float a = in[id];
    out[id] = a + 1.0f;
}

kernel void register_high(device const float* in [[buffer(0)]],
                         device float* out [[buffer(1)]],
                         constant uint& size [[buffer(2)]],
                         uint id [[thread_position_in_grid]]) {
    // High register usage - forces spill to memory
    float a = in[id];
    float b = a * 2.0f;
    float c = b * 3.0f;
    float d = c / 4.0f;
    float e = d + 5.0f;
    float f = e - 6.0f;
    float g = f * 7.0f;
    float h = g / 8.0f;
    out[id] = h + a + b + c + d + e + f + g + h;
}

// ============================================================
// 9. BRANCH DIVERGENCE TEST
// Test cost of divergent branches within SIMD-group
// ============================================================

kernel void branch_converged(device const float* in [[buffer(0)]],
                            device float* out [[buffer(1)]],
                            constant uint& size [[buffer(2)]],
                            uint id [[thread_position_in_grid]]) {
    // All threads take same path
    out[id] = in[id] > 0.5f ? in[id] : 0.5f;
}

kernel void branch_divergent(device const float* in [[buffer(0)]],
                            device float* out [[buffer(1)]],
                            constant uint& size [[buffer(2)]],
                            uint id [[thread_position_in_grid]]) {
    // Alternating branches - maximum divergence
    uint lane = id & 31;
    if ((lane & 1) == 0) {
        out[id] = in[id] * 2.0f;
    } else {
        out[id] = in[id] + 1.0f;
    }
}

// ============================================================
// 10. ATOMIC OPERATION CONTENTION
// ============================================================

kernel void atomic_low_contention(device atomic_uint* counter [[buffer(0)]],
                                  device float* out [[buffer(1)]],
                                  constant uint& size [[buffer(2)]],
                                  uint id [[thread_position_in_grid]]) {
    // Each thread increments different location
    atomic_fetch_add_explicit(&counter[id % (size / 4)], 1, memory_order_relaxed);
    out[id] = float(id);
}

kernel void atomic_high_contention(device atomic_uint* counter [[buffer(0)]],
                                  device float* out [[buffer(1)]],
                                  constant uint& size [[buffer(2)]],
                                  uint id [[thread_position_in_grid]]) {
    // All threads increment same location
    atomic_fetch_add_explicit(&counter[0], 1, memory_order_relaxed);
    out[id] = float(id);
}

// ============================================================
// 11. COMMAND BUFFER BATCHING TEST
// ============================================================

kernel void batch_kernel_a(device const float* in [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        constant uint& size [[buffer(2)]],
                        uint id [[thread_position_in_grid]]) {
    out[id] = in[id] * 2.0f;
}

kernel void batch_kernel_b(device const float* in [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        constant uint& size [[buffer(2)]],
                        uint id [[thread_position_in_grid]]) {
    out[id] = in[id] + 1.0f;
}

kernel void batch_kernel_c(device const float* in [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        constant uint& size [[buffer(2)]],
                        uint id [[thread_position_in_grid]]) {
    out[id] = in[id] * 3.0f + 2.0f;
}

// ============================================================
// 12. OCCUPANCY ANALYSIS (variable shared memory)
// ============================================================

kernel void occupancy_low(device const float* in [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        threadgroup float* shared [[threadgroup(0)]],
                        constant uint& size [[buffer(2)]],
                        uint id [[thread_position_in_grid]],
                        uint lid [[thread_position_in_threadgroup]]) {
    // Low occupancy: 32 threads, 1KB shared per thread
    shared[lid] = in[id];
    threadgroup_barrier(mem_flags::mem_none);
    float sum = 0.0f;
    for (uint i = 0; i < 16; i++) {
        sum += shared[(lid + i) & 31];
    }
    out[id] = sum;
}

kernel void occupancy_med(device const float* in [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        threadgroup float* shared [[threadgroup(0)]],
                        constant uint& size [[buffer(2)]],
                        uint id [[thread_position_in_grid]],
                        uint lid [[thread_position_in_threadgroup]]) {
    // Medium occupancy: 128 threads, 256B shared per thread
    shared[lid] = in[id];
    threadgroup_barrier(mem_flags::mem_none);
    float sum = 0.0f;
    for (uint i = 0; i < 8; i++) {
        sum += shared[(lid + i) & 127];
    }
    out[id] = sum;
}

kernel void occupancy_high(device const float* in [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        threadgroup float* shared [[threadgroup(0)]],
                        constant uint& size [[buffer(2)]],
                        uint id [[thread_position_in_grid]],
                        uint lid [[thread_position_in_threadgroup]]) {
    // High occupancy: 512 threads, 64B shared per thread
    shared[lid] = in[id];
    threadgroup_barrier(mem_flags::mem_none);
    float sum = 0.0f;
    for (uint i = 0; i < 4; i++) {
        sum += shared[(lid + i) & 511];
    }
    out[id] = sum;
}

// ============================================================
// 11. HIGH PERFORMANCE MEMORY BANDWIDTH
// ============================================================

// Sequential write - baseline
kernel void mem_write_seq(device float* out [[buffer(0)]],
                         constant uint& size [[buffer(1)]],
                         uint id [[thread_position_in_grid]]) {
    out[id] = float(id);
}

// Sequential read - baseline
kernel void mem_read_seq(device const float* in [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        constant uint& size [[buffer(2)]],
                        uint id [[thread_position_in_grid]]) {
    out[id] = in[id];
}

// Read-write (compute) - measures achievable bandwidth with computation
kernel void mem_read_write(device const float* in [[buffer(0)]],
                          device float* out [[buffer(1)]],
                          constant uint& size [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {
    out[id] = in[id] * 2.0f + 1.0f;
}

// Write combining test - multiple writes per thread
kernel void mem_write_combine(device float* out [[buffer(0)]],
                             constant uint& size [[buffer(1)]],
                             uint id [[thread_position_in_grid]]) {
    uint base = id * 4;
    out[base] = float(id);
    out[base + 1] = float(id);
    out[base + 2] = float(id);
    out[base + 3] = float(id);
}

// Sequential with some computation
kernel void mem_read_compute(device const float* in [[buffer(0)]],
                            device float* out [[buffer(1)]],
                            constant uint& size [[buffer(2)]],
                            uint id [[thread_position_in_grid]]) {
    float val = in[id];
    val = val * 2.0f;
    val = val + 1.0f;
    val = val * 3.0f;
    out[id] = val;
}

// Non-temporal (bypass cache) write simulation
kernel void mem_write_nontemporal(device float* out [[buffer(0)]],
                                 constant uint& size [[buffer(1)]],
                                 uint id [[thread_position_in_grid]]) {
    // Write with no reuse - simulates non-temporal
    out[id] = float(id + 1);
}

// Strided read - cache inefficient
kernel void mem_read_stride(device const float* in [[buffer(0)]],
                           device float* out [[buffer(1)]],
                           constant uint& size [[buffer(2)]],
                           uint id [[thread_position_in_grid]]) {
    uint idx = (id * 64) % size;
    out[id] = in[idx];
}

// Strided write
kernel void mem_write_stride(device float* out [[buffer(0)]],
                            constant uint& size [[buffer(1)]],
                            uint id [[thread_position_in_grid]]) {
    uint idx = (id * 64) % size;
    out[idx] = float(id);
}

// Two-pass write: first half, second half (tests memory controller interleaving)
kernel void mem_write_two_pass(device float* out [[buffer(0)]],
                              constant uint& size [[buffer(1)]],
                              uint id [[thread_position_in_grid]]) {
    if (id < size / 2) {
        out[id] = float(id);
    } else {
        out[id] = float(id + size / 2);
    }
}

// Read-modify-write (atomic-like without atomics)
kernel void mem_read_modify_write(device float* in [[buffer(0)]],
                                 constant uint& size [[buffer(1)]],
                                 uint id [[thread_position_in_grid]]) {
    float old = in[id];
    in[id] = old + 1.0f;
}

// Large burst read (float4 vectorized)
kernel void mem_read_burst(device const float4* in [[buffer(0)]],
                          device float4* out [[buffer(1)]],
                          constant uint& size [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {
    out[id] = in[id] * 2.0f;
}

// Double-buffer style: alternating reads from two buffers
kernel void mem_double_buffer_read(device const float* in1 [[buffer(0)]],
                                   device const float* in2 [[buffer(1)]],
                                   device float* out [[buffer(2)]],
                                   constant uint& size [[buffer(3)]],
                                   uint id [[thread_position_in_grid]]) {
    bool use_first = (id / 256) % 2 == 0;
    out[id] = use_first ? in1[id] : in2[id];
}
"""

// MARK: - FP16 Deep Dive Shader Library

let shaderSource = """
#include <metal_stdlib>
using namespace metal;

// FP16 Matrix Multiply - Naive
kernel void matmul_fp16_naive(device const half* a [[buffer(0)]],
                             device const half* b [[buffer(1)]],
                             device half* c [[buffer(2)]],
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
    c[gid.y * N + gid.x] = sum;
}

// FP32 Matrix Multiply - Naive (baseline)
kernel void matmul_fp32_naive(device const float* a [[buffer(0)]],
                             device const float* b [[buffer(1)]],
                             device float* c [[buffer(2)]],
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
    c[gid.y * N + gid.x] = sum;
}

// FP16 Vector Add
kernel void vec_add_fp16(device const half* a [[buffer(0)]],
                       device const half* b [[buffer(1)]],
                       device half* c [[buffer(2)]],
                       constant uint& size [[buffer(3)]],
                       uint id [[thread_position_in_grid]]) {
    c[id] = a[id] + b[id];
}

// FP32 Vector Add (baseline)
kernel void vec_add_fp32(device const float* a [[buffer(0)]],
                        device const float* b [[buffer(1)]],
                        device float* c [[buffer(2)]],
                        constant uint& size [[buffer(3)]],
                        uint id [[thread_position_in_grid]]) {
    c[id] = a[id] + b[id];
}

// FP16 Vector Multiply
kernel void vec_mul_fp16(device const half* a [[buffer(0)]],
                        device const half* b [[buffer(1)]],
                        device half* c [[buffer(2)]],
                        constant uint& size [[buffer(3)]],
                        uint id [[thread_position_in_grid]]) {
    c[id] = a[id] * b[id];
}

// FP32 Vector Multiply (baseline)
kernel void vec_mul_fp32(device const float* a [[buffer(0)]],
                        device const float* b [[buffer(1)]],
                        device float* c [[buffer(2)]],
                        constant uint& size [[buffer(3)]],
                        uint id [[thread_position_in_grid]]) {
    c[id] = a[id] * b[id];
}

// FP16 FMA (Fused Multiply-Add)
kernel void fma_fp16(device const half* a [[buffer(0)]],
                    device const half* b [[buffer(1)]],
                    device half* c [[buffer(2)]],
                    constant uint& size [[buffer(3)]],
                    uint id [[thread_position_in_grid]]) {
    c[id] = fma(a[id], b[id], c[id]);
}

// FP32 FMA (baseline)
kernel void fma_fp32(device const float* a [[buffer(0)]],
                    device const float* b [[buffer(1)]],
                    device float* c [[buffer(2)]],
                    constant uint& size [[buffer(3)]],
                    uint id [[thread_position_in_grid]]) {
    c[id] = fma(a[id], b[id], c[id]);
}

// FP16 Reduction (sum)
kernel void reduce_fp16(device const half* src [[buffer(0)]],
                      device half* dst [[buffer(1)]],
                      threadgroup half* shared [[threadgroup(0)]],
                      constant uint& size [[buffer(2)]],
                      uint id [[thread_position_in_grid]],
                      uint lid [[thread_position_in_threadgroup]]) {
    constexpr uint THREADGROUP_SIZE = 256;
    shared[lid] = src[id];
    threadgroup_barrier(mem_flags::mem_none);

    for (uint s = THREADGROUP_SIZE / 2; s > 0; s >>= 1) {
        if (lid < s) {
            shared[lid] += shared[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_none);
    }

    if (lid == 0) {
        dst[0] = shared[0];
    }
}

// FP32 Reduction (baseline)
kernel void reduce_fp32(device const float* src [[buffer(0)]],
                      device float* dst [[buffer(1)]],
                      threadgroup float* shared [[threadgroup(0)]],
                      constant uint& size [[buffer(2)]],
                      uint id [[thread_position_in_grid]],
                      uint lid [[thread_position_in_threadgroup]]) {
    constexpr uint THREADGROUP_SIZE = 256;
    shared[lid] = src[id];
    threadgroup_barrier(mem_flags::mem_none);

    for (uint s = THREADGROUP_SIZE / 2; s > 0; s >>= 1) {
        if (lid < s) {
            shared[lid] += shared[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_none);
    }

    if (lid == 0) {
        dst[0] = shared[0];
    }
}

// FP16 to FP32 conversion test
kernel void convert_fp16_to_fp32(device const half* src [[buffer(0)]],
                               device float* dst [[buffer(1)]],
                               constant uint& size [[buffer(2)]],
                               uint id [[thread_position_in_grid]]) {
    dst[id] = (float)src[id];
}

// FP32 to FP16 conversion test
kernel void convert_fp32_to_fp16(device const float* src [[buffer(0)]],
                               device half* dst [[buffer(1)]],
                               constant uint& size [[buffer(2)]],
                               uint id [[thread_position_in_grid]]) {
    dst[id] = (half)src[id];
}

// FP16 with shared memory tile for matrix multiply
kernel void matmul_fp16_tiled(device const half* a [[buffer(0)]],
                             device const half* b [[buffer(1)]],
                             device half* c [[buffer(2)]],
                             constant uint& M [[buffer(3)]],
                             constant uint& K [[buffer(4)]],
                             constant uint& N [[buffer(5)]],
                             threadgroup half* As [[threadgroup(0)]],
                             threadgroup half* Bs [[threadgroup(1)]],
                             uint2 gid [[thread_position_in_grid]],
                             uint2 tid [[thread_position_in_threadgroup]]) {
    constexpr uint TILE_SIZE = 16;

    uint cRow = gid.y;
    uint cCol = gid.x;
    uint cRowTile = tid.y;
    uint cColTile = tid.x;

    half sum = 0.0h;

    for (uint kTile = 0; kTile < (K + TILE_SIZE - 1) / TILE_SIZE; kTile++) {
        uint aRow = cRow;
        uint aCol = kTile * TILE_SIZE + cColTile;
        if (aRow < M && aCol < K) {
            As[cRowTile * TILE_SIZE + cColTile] = a[aRow * K + aCol];
        } else {
            As[cRowTile * TILE_SIZE + cColTile] = 0.0h;
        }

        uint bRow = kTile * TILE_SIZE + cRowTile;
        uint bCol = cCol;
        if (bRow < K && bCol < N) {
            Bs[cRowTile * TILE_SIZE + cColTile] = b[bRow * N + bCol];
        } else {
            Bs[cRowTile * TILE_SIZE + cColTile] = 0.0h;
        }

        threadgroup_barrier(mem_flags::mem_none);

        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += As[cRowTile * TILE_SIZE + k] * Bs[k * TILE_SIZE + cColTile];
        }

        threadgroup_barrier(mem_flags::mem_none);
    }

    if (cRow < M && cCol < N) {
        c[cRow * N + cCol] = sum;
    }
}
"""

// MARK: - Device Info

func printDeviceInfo(device: MTLDevice) {
    print("\n=== Apple Metal GPU Info ===")
    print("Device Name: \(device.name)")
    print("Max Threadgroup Memory: \(device.maxThreadgroupMemoryLength / 1024) KB")
    print("Max Threads Per Threadgroup: \(device.maxThreadsPerThreadgroup.width)")
    if device.supportsFamily(.apple7) { print("GPU Family: Apple 7+") }
    if device.supportsFamily(.apple8) { print("GPU Family: Apple 8+") }
    print("")
}

// MARK: - Test: FP16 vs FP32 Matrix Multiply

func testMatmulFP16vsFP32(device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary) throws {
    print("=== FP16 vs FP32 Matrix Multiply ===")

    guard let func_fp16 = library.makeFunction(name: "matmul_fp16_naive"),
          let func_fp32 = library.makeFunction(name: "matmul_fp32_naive"),
          let func_fp16_tiled = library.makeFunction(name: "matmul_fp16_tiled"),
          let pipeline_fp16 = try? device.makeComputePipelineState(function: func_fp16),
          let pipeline_fp32 = try? device.makeComputePipelineState(function: func_fp32),
          let pipeline_fp16_tiled = try? device.makeComputePipelineState(function: func_fp16_tiled) else {
        print("Failed to create pipelines")
        return
    }

    let sizes: [UInt32] = [256, 512, 1024]

    for M in sizes {
        let K = M
        let N = M
        let iterations = 10

        let fp16Size = Int(M * K)
        let fp32Size = Int(M * K)

        guard let aBufferF16 = device.makeBuffer(length: fp16Size * MemoryLayout<UInt16>.size, options: .storageModeShared),
              let bBufferF16 = device.makeBuffer(length: fp16Size * MemoryLayout<UInt16>.size, options: .storageModeShared),
              let cBufferF16 = device.makeBuffer(length: Int(M * N) * MemoryLayout<UInt16>.size, options: .storageModeShared),
              let aBufferF32 = device.makeBuffer(length: fp32Size * MemoryLayout<Float>.size, options: .storageModeShared),
              let bBufferF32 = device.makeBuffer(length: fp32Size * MemoryLayout<Float>.size, options: .storageModeShared),
              let cBufferF32 = device.makeBuffer(length: Int(M * N) * MemoryLayout<Float>.size, options: .storageModeShared) else {
            continue
        }

        // Initialize FP16 data
        let aF16 = aBufferF16.contents().assumingMemoryBound(to: UInt16.self)
        let bF16 = bBufferF16.contents().assumingMemoryBound(to: UInt16.self)
        for i in 0..<fp16Size { aF16[i] = UInt16((i % 256) << 8) }
        for i in 0..<fp16Size { bF16[i] = UInt16(((i + 1) % 256) << 8) }

        // Initialize FP32 data
        let aF32 = aBufferF32.contents().assumingMemoryBound(to: Float.self)
        let bF32 = bBufferF32.contents().assumingMemoryBound(to: Float.self)
        for i in 0..<fp32Size { aF32[i] = Float(i % 256) / 256.0 }
        for i in 0..<fp32Size { bF32[i] = Float((i + 1) % 256) / 256.0 }

        var m = M, k = K, n = N

        // FP16 Naive
        let start_fp16 = getTimeNanos()
        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(pipeline_fp16)
            encoder.setBuffer(aBufferF16, offset: 0, index: 0)
            encoder.setBuffer(bBufferF16, offset: 0, index: 1)
            encoder.setBuffer(cBufferF16, offset: 0, index: 2)
            encoder.setBytes(&m, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.setBytes(&k, length: MemoryLayout<UInt32>.size, index: 4)
            encoder.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 5)
            encoder.dispatchThreads(MTLSize(width: Int(N), height: Int(M), depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        let end_fp16 = getTimeNanos()
        let elapsed_fp16 = getElapsedSeconds(start: start_fp16, end: end_fp16)
        let flops_fp16 = 2.0 * Double(M) * Double(K) * Double(N) * Double(iterations)
        let gflops_fp16 = flops_fp16 / elapsed_fp16 / 1e9

        // FP32 Naive
        let start_fp32 = getTimeNanos()
        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(pipeline_fp32)
            encoder.setBuffer(aBufferF32, offset: 0, index: 0)
            encoder.setBuffer(bBufferF32, offset: 0, index: 1)
            encoder.setBuffer(cBufferF32, offset: 0, index: 2)
            encoder.setBytes(&m, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.setBytes(&k, length: MemoryLayout<UInt32>.size, index: 4)
            encoder.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 5)
            encoder.dispatchThreads(MTLSize(width: Int(N), height: Int(M), depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        let end_fp32 = getTimeNanos()
        let elapsed_fp32 = getElapsedSeconds(start: start_fp32, end: end_fp32)
        let flops_fp32 = 2.0 * Double(M) * Double(K) * Double(N) * Double(iterations)
        let gflops_fp32 = flops_fp32 / elapsed_fp32 / 1e9

        print("Size \(M)x\(K)x\(N): FP16=\(String(format: "%.2f", gflops_fp16)) GFLOPS, FP32=\(String(format: "%.2f", gflops_fp32)) GFLOPS, Ratio=\(String(format: "%.2fx", gflops_fp16 / gflops_fp32))")

        // FP16 Tiled (with shared memory)
        guard let aBufferTiled = device.makeBuffer(length: fp16Size * MemoryLayout<UInt16>.size, options: .storageModeShared),
              let bBufferTiled = device.makeBuffer(length: fp16Size * MemoryLayout<UInt16>.size, options: .storageModeShared),
              let cBufferTiled = device.makeBuffer(length: Int(M * N) * MemoryLayout<UInt16>.size, options: .storageModeShared),
              let sharedA = device.makeBuffer(length: 16 * 16 * MemoryLayout<UInt16>.size, options: .storageModeShared),
              let sharedB = device.makeBuffer(length: 16 * 16 * MemoryLayout<UInt16>.size, options: .storageModeShared) else {
            continue
        }

        let aT = aBufferTiled.contents().assumingMemoryBound(to: UInt16.self)
        let bT = bBufferTiled.contents().assumingMemoryBound(to: UInt16.self)
        for i in 0..<fp16Size { aT[i] = UInt16((i % 256) << 8) }
        for i in 0..<fp16Size { bT[i] = UInt16(((i + 1) % 256) << 8) }

        let start_tiled = getTimeNanos()
        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(pipeline_fp16_tiled)
            encoder.setBuffer(aBufferTiled, offset: 0, index: 0)
            encoder.setBuffer(bBufferTiled, offset: 0, index: 1)
            encoder.setBuffer(cBufferTiled, offset: 0, index: 2)
            encoder.setBytes(&m, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.setBytes(&k, length: MemoryLayout<UInt32>.size, index: 4)
            encoder.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 5)
            encoder.setBuffer(sharedA, offset: 0, index: 6)
            encoder.setBuffer(sharedB, offset: 0, index: 7)
            encoder.dispatchThreads(MTLSize(width: Int(N), height: Int(M), depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        let end_tiled = getTimeNanos()
        let elapsed_tiled = getElapsedSeconds(start: start_tiled, end: end_tiled)
        let flops_tiled = 2.0 * Double(M) * Double(K) * Double(N) * Double(iterations)
        let gflops_tiled = flops_tiled / elapsed_tiled / 1e9

        print("         FP16 Tiled (Shared Memory)=\(String(format: "%.2f", gflops_tiled)) GFLOPS, vs Naive=\(String(format: "%.2fx", gflops_tiled / gflops_fp16))")
    }
    print("")
}

// MARK: - Test: FP16 vs FP32 Vector Operations

func testVectorFP16vsFP32(device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary) throws {
    print("=== FP16 vs FP32 Vector Operations ===")

    let bufferSize = 32 * 1024 * 1024
    let iterations = 50

    let ops = [
        ("vec_add_fp16", "vec_add_fp32", "Vector Add"),
        ("vec_mul_fp16", "vec_mul_fp32", "Vector Multiply"),
        ("fma_fp16", "fma_fp32", "FMA")
    ]

    for (name_fp16, name_fp32, desc) in ops {
        guard let func_fp16 = library.makeFunction(name: name_fp16),
              let func_fp32 = library.makeFunction(name: name_fp32),
              let pipeline_fp16 = try? device.makeComputePipelineState(function: func_fp16),
              let pipeline_fp32 = try? device.makeComputePipelineState(function: func_fp32) else {
            continue
        }

        // FP16
        guard let aBufferF16 = device.makeBuffer(length: bufferSize * MemoryLayout<UInt16>.size, options: .storageModeShared),
              let bBufferF16 = device.makeBuffer(length: bufferSize * MemoryLayout<UInt16>.size, options: .storageModeShared),
              let cBufferF16 = device.makeBuffer(length: bufferSize * MemoryLayout<UInt16>.size, options: .storageModeShared) else {
            continue
        }

        let aF16 = aBufferF16.contents().assumingMemoryBound(to: UInt16.self)
        let bF16 = bBufferF16.contents().assumingMemoryBound(to: UInt16.self)
        for i in 0..<(bufferSize / 2) {
            aF16[i] = UInt16((i % 256) << 8)
            bF16[i] = UInt16(((i + 1) % 256) << 8)
        }

        var size16 = UInt32(bufferSize / 2)

        let start_fp16 = getTimeNanos()
        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(pipeline_fp16)
            encoder.setBuffer(aBufferF16, offset: 0, index: 0)
            encoder.setBuffer(bBufferF16, offset: 0, index: 1)
            encoder.setBuffer(cBufferF16, offset: 0, index: 2)
            encoder.setBytes(&size16, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.dispatchThreads(MTLSize(width: bufferSize / 2, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        let end_fp16 = getTimeNanos()
        let elapsed_fp16 = getElapsedSeconds(start: start_fp16, end: end_fp16)
        let gflops_fp16 = Double(bufferSize / 2 * iterations) / elapsed_fp16 / 1e9

        // FP32
        guard let aBufferF32 = device.makeBuffer(length: bufferSize * MemoryLayout<Float>.size, options: .storageModeShared),
              let bBufferF32 = device.makeBuffer(length: bufferSize * MemoryLayout<Float>.size, options: .storageModeShared),
              let cBufferF32 = device.makeBuffer(length: bufferSize * MemoryLayout<Float>.size, options: .storageModeShared) else {
            continue
        }

        let aF32 = aBufferF32.contents().assumingMemoryBound(to: Float.self)
        let bF32 = bBufferF32.contents().assumingMemoryBound(to: Float.self)
        for i in 0..<bufferSize / 4 {
            aF32[i] = Float(i % 256) / 256.0
            bF32[i] = Float((i + 1) % 256) / 256.0
        }

        var size32 = UInt32(bufferSize / 4)

        let start_fp32 = getTimeNanos()
        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(pipeline_fp32)
            encoder.setBuffer(aBufferF32, offset: 0, index: 0)
            encoder.setBuffer(bBufferF32, offset: 0, index: 1)
            encoder.setBuffer(cBufferF32, offset: 0, index: 2)
            encoder.setBytes(&size32, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.dispatchThreads(MTLSize(width: bufferSize / 4, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        let end_fp32 = getTimeNanos()
        let elapsed_fp32 = getElapsedSeconds(start: start_fp32, end: end_fp32)
        let gflops_fp32 = Double(bufferSize / 4 * iterations) / elapsed_fp32 / 1e9

        print("\(desc): FP16=\(String(format: "%.2f", gflops_fp16)) GOPS, FP32=\(String(format: "%.2f", gflops_fp32)) GOPS, Ratio=\(String(format: "%.2fx", gflops_fp16 / gflops_fp32))")
    }
    print("")
}

// MARK: - Test: FP16/FP32 Conversion Performance

func testConversionPerformance(device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary) throws {
    print("=== FP16/FP32 Conversion Performance ===")

    guard let func_f16_to_f32 = library.makeFunction(name: "convert_fp16_to_fp32"),
          let func_f32_to_f16 = library.makeFunction(name: "convert_fp32_to_fp16"),
          let pipeline_f16_to_f32 = try? device.makeComputePipelineState(function: func_f16_to_f32),
          let pipeline_f32_to_f16 = try? device.makeComputePipelineState(function: func_f32_to_f16) else {
        print("Failed to create conversion pipelines")
        return
    }

    let bufferSize = 32 * 1024 * 1024
    let iterations = 50

    // FP16 -> FP32 conversion
    guard let srcBufferF16 = device.makeBuffer(length: bufferSize * MemoryLayout<UInt16>.size, options: .storageModeShared),
          let dstBufferF32 = device.makeBuffer(length: bufferSize * MemoryLayout<Float>.size, options: .storageModeShared) else {
        return
    }

    let srcF16 = srcBufferF16.contents().assumingMemoryBound(to: UInt16.self)
    for i in 0..<(bufferSize / 2) {
        srcF16[i] = UInt16((i % 256) << 8)
    }

    var size = UInt32(bufferSize / 2)

    let start_f16_to_f32 = getTimeNanos()
    for _ in 0..<iterations {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(pipeline_f16_to_f32)
        encoder.setBuffer(srcBufferF16, offset: 0, index: 0)
        encoder.setBuffer(dstBufferF32, offset: 0, index: 1)
        encoder.setBytes(&size, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.dispatchThreads(MTLSize(width: bufferSize / 2, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let end_f16_to_f32 = getTimeNanos()
    let elapsed_f16_to_f32 = getElapsedSeconds(start: start_f16_to_f32, end: end_f16_to_f32)
    let ops_f16_to_f32 = Double(bufferSize / 2 * iterations)
    let gops_f16_to_f32 = ops_f16_to_f32 / elapsed_f16_to_f32 / 1e9

    // FP32 -> FP16 conversion
    guard let srcBufferF32 = device.makeBuffer(length: bufferSize * MemoryLayout<Float>.size, options: .storageModeShared),
          let dstBufferF16 = device.makeBuffer(length: bufferSize * MemoryLayout<UInt16>.size, options: .storageModeShared) else {
        return
    }

    let srcF32 = srcBufferF32.contents().assumingMemoryBound(to: Float.self)
    for i in 0..<(bufferSize / 4) {
        srcF32[i] = Float(i % 256) / 256.0
    }

    let start_f32_to_f16 = getTimeNanos()
    for _ in 0..<iterations {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(pipeline_f32_to_f16)
        encoder.setBuffer(srcBufferF32, offset: 0, index: 0)
        encoder.setBuffer(dstBufferF16, offset: 0, index: 1)
        encoder.setBytes(&size, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.dispatchThreads(MTLSize(width: bufferSize / 4, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let end_f32_to_f16 = getTimeNanos()
    let elapsed_f32_to_f16 = getElapsedSeconds(start: start_f32_to_f16, end: end_f32_to_f16)
    let ops_f32_to_f16 = Double(bufferSize / 4 * iterations)
    let gops_f32_to_f16 = ops_f32_to_f16 / elapsed_f32_to_f16 / 1e9

    print("FP16 to FP32: \(String(format: "%.2f", gops_f16_to_f32)) GOPS", terminator: "")
    print("  |  FP32 to FP16: \(String(format: "%.2f", gops_f32_to_f16)) GOPS")
    print("")
}

// MARK: - Test: Reduction Performance

func testReductionPerformance(device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary) throws {
    print("=== Threadgroup Reduction Performance ===")

    guard let func_fp16 = library.makeFunction(name: "reduce_fp16"),
          let func_fp32 = library.makeFunction(name: "reduce_fp32"),
          let pipeline_fp16 = try? device.makeComputePipelineState(function: func_fp16),
          let pipeline_fp32 = try? device.makeComputePipelineState(function: func_fp32) else {
        print("Failed to create reduction pipelines")
        return
    }

    let sizes: [UInt32] = [1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024]
    let iterations = 100

    for size in sizes {
        // FP16 Reduction
        guard let srcBufferF16 = device.makeBuffer(length: Int(size) * MemoryLayout<UInt16>.size, options: .storageModeShared),
              let dstBufferF16 = device.makeBuffer(length: 256 * MemoryLayout<UInt16>.size, options: .storageModeShared) else {
            continue
        }

        let srcF16 = srcBufferF16.contents().assumingMemoryBound(to: UInt16.self)
        for i in 0..<Int(size) {
            srcF16[i] = UInt16((i % 256) << 8)
        }

        var sizeVal = size

        let start_fp16 = getTimeNanos()
        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(pipeline_fp16)
            encoder.setBuffer(srcBufferF16, offset: 0, index: 0)
            encoder.setBuffer(dstBufferF16, offset: 0, index: 1)
            encoder.setBytes(&sizeVal, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.dispatchThreads(MTLSize(width: Int(size), height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        let end_fp16 = getTimeNanos()
        let elapsed_fp16 = getElapsedSeconds(start: start_fp16, end: end_fp16)
        let gops_fp16 = Double(size) * Double(iterations) / elapsed_fp16 / 1e9

        // FP32 Reduction
        guard let srcBufferF32 = device.makeBuffer(length: Int(size) * MemoryLayout<Float>.size, options: .storageModeShared),
              let dstBufferF32 = device.makeBuffer(length: 256 * MemoryLayout<Float>.size, options: .storageModeShared) else {
            continue
        }

        let srcF32 = srcBufferF32.contents().assumingMemoryBound(to: Float.self)
        for i in 0..<Int(size) {
            srcF32[i] = Float(i % 256) / 256.0
        }

        let start_fp32 = getTimeNanos()
        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(pipeline_fp32)
            encoder.setBuffer(srcBufferF32, offset: 0, index: 0)
            encoder.setBuffer(dstBufferF32, offset: 0, index: 1)
            encoder.setBytes(&sizeVal, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.dispatchThreads(MTLSize(width: Int(size), height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        let end_fp32 = getTimeNanos()
        let elapsed_fp32 = getElapsedSeconds(start: start_fp32, end: end_fp32)
        let gops_fp32 = Double(size) * Double(iterations) / elapsed_fp32 / 1e9

        print("Size \(size): FP16=\(String(format: "%.3f", gops_fp16)) GOPS, FP32=\(String(format: "%.3f", gops_fp32)) GOPS, Ratio=\(String(format: "%.2fx", gops_fp16 / gops_fp32))")
    }
    print("")
}

// MARK: - Deep GPU Architecture Research

func testDeepGPUResearch(device: MTLDevice, queue: MTLCommandQueue) throws {
    print(String(repeating: "=", count: 60))
    print("=== DEEP GPU ARCHITECTURE RESEARCH ===")
    print(String(repeating: "=", count: 60))

    let deepLibrary: MTLLibrary
    do {
        deepLibrary = try device.makeLibrary(source: deepShaderSource, options: nil)
    } catch {
        print("Failed to compile deep research shaders: \(error)")
        return
    }

    let iterations = 100
    let bufferSize = 8 * 1024 * 1024

    // 1. MEMORY COALESCING TEST
    print("\n--- 1. Memory Coalescing Analysis ---")

    guard let coalescedFunc = deepLibrary.makeFunction(name: "coalesced_read"),
          let noncoalescedFunc = deepLibrary.makeFunction(name: "noncoalesced_read"),
          let coalPipeline = try? device.makeComputePipelineState(function: coalescedFunc),
          let noncoalPipeline = try? device.makeComputePipelineState(function: noncoalescedFunc) else {
        print("Failed to create coalescing pipelines")
        return
    }

    guard let inBuffer = device.makeBuffer(length: bufferSize * MemoryLayout<Float>.size, options: .storageModeShared),
          let outBuffer = device.makeBuffer(length: bufferSize * MemoryLayout<Float>.size, options: .storageModeShared) else {
        return
    }

    let inPtr = inBuffer.contents().assumingMemoryBound(to: Float.self)
    for i in 0..<bufferSize { inPtr[i] = Float(i) }

    var size = UInt32(bufferSize)

    // Coalesced read
    let startCoal = getTimeNanos()
    for _ in 0..<iterations {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(coalPipeline)
        encoder.setBuffer(inBuffer, offset: 0, index: 0)
        encoder.setBuffer(outBuffer, offset: 0, index: 1)
        encoder.setBytes(&size, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.dispatchThreads(MTLSize(width: bufferSize, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let endCoal = getTimeNanos()
    let elapsedCoal = getElapsedSeconds(start: startCoal, end: endCoal)
    let bwCoal = Double(bufferSize) * Double(iterations) * Double(MemoryLayout<Float>.size) / elapsedCoal / 1e9

    // Non-coalesced read
    let startNonCoal = getTimeNanos()
    for _ in 0..<iterations {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(noncoalPipeline)
        encoder.setBuffer(inBuffer, offset: 0, index: 0)
        encoder.setBuffer(outBuffer, offset: 0, index: 1)
        encoder.setBytes(&size, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.dispatchThreads(MTLSize(width: bufferSize, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let endNonCoal = getTimeNanos()
    let elapsedNonCoal = getElapsedSeconds(start: startNonCoal, end: endNonCoal)
    let bwNonCoal = Double(bufferSize) * Double(iterations) * Double(MemoryLayout<Float>.size) / elapsedNonCoal / 1e9

    print("Coalesced Read:  \(String(format: "%.2f", bwCoal)) GB/s")
    print("Non-Coalesced:   \(String(format: "%.2f", bwNonCoal)) GB/s")
    print("Coalescing Gain: \(String(format: "%.1fx", bwCoal / bwNonCoal))")

    // 2. SHARED MEMORY BANK CONFLICT TEST
    print("\n--- 2. Shared Memory Bank Conflict Analysis ---")

    guard let bankNoneFunc = deepLibrary.makeFunction(name: "shared_bank_none"),
          let bankConflictFunc = deepLibrary.makeFunction(name: "shared_bank_conflict"),
          let bankNonePipeline = try? device.makeComputePipelineState(function: bankNoneFunc),
          let bankConflictPipeline = try? device.makeComputePipelineState(function: bankConflictFunc) else {
        print("Failed to create bank conflict pipelines")
        return
    }

    guard let sharedOutBuffer = device.makeBuffer(length: bufferSize * MemoryLayout<Float>.size, options: .storageModeShared) else {
        return
    }

    let threadgroupSize = 256
    let gridSize = bufferSize

    let startBankNone = getTimeNanos()
    for _ in 0..<iterations {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(bankNonePipeline)
        encoder.setBuffer(sharedOutBuffer, offset: 0, index: 0)
        encoder.setBytes(&size, length: MemoryLayout<UInt32>.size, index: 1)
        encoder.dispatchThreads(MTLSize(width: gridSize, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let endBankNone = getTimeNanos()
    let elapsedBankNone = getElapsedSeconds(start: startBankNone, end: endBankNone)

    let startBankConflict = getTimeNanos()
    for _ in 0..<iterations {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(bankConflictPipeline)
        encoder.setBuffer(sharedOutBuffer, offset: 0, index: 0)
        encoder.setBytes(&size, length: MemoryLayout<UInt32>.size, index: 1)
        encoder.dispatchThreads(MTLSize(width: gridSize, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let endBankConflict = getTimeNanos()
    let elapsedBankConflict = getElapsedSeconds(start: startBankConflict, end: endBankConflict)

    let gopsBankNone = Double(iterations) * Double(gridSize) / elapsedBankNone / 1e9
    let gopsBankConflict = Double(iterations) * Double(gridSize) / elapsedBankConflict / 1e9

    print("No Conflict:     \(String(format: "%.2f", gopsBankNone)) GOPS")
    print("With Conflict:   \(String(format: "%.2f", gopsBankConflict)) GOPS")
    print("Conflict Cost:   \(String(format: "%.1fx", elapsedBankConflict / elapsedBankNone))")

    // 3. LATENCY HIDING TEST
    print("\n--- 3. Memory Latency Hiding Analysis ---")

    guard let latencyHiddenFunc = deepLibrary.makeFunction(name: "latency_hidden"),
          let latencyExposedFunc = deepLibrary.makeFunction(name: "latency_exposed"),
          let hiddenPipeline = try? device.makeComputePipelineState(function: latencyHiddenFunc),
          let exposedPipeline = try? device.makeComputePipelineState(function: latencyExposedFunc) else {
        print("Failed to create latency pipelines")
        return
    }

    let hiddenSize = 1024 * 1024

    let startHidden = getTimeNanos()
    for _ in 0..<iterations {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(hiddenPipeline)
        encoder.setBuffer(inBuffer, offset: 0, index: 0)
        encoder.setBuffer(outBuffer, offset: 0, index: 1)
        var sz = UInt32(hiddenSize)
        encoder.setBytes(&sz, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.dispatchThreads(MTLSize(width: hiddenSize, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let endHidden = getTimeNanos()
    let elapsedHidden = getElapsedSeconds(start: startHidden, end: endHidden)
    let opsHidden = Double(hiddenSize) * 8.0 * Double(iterations)

    let startExposed = getTimeNanos()
    for _ in 0..<iterations {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(exposedPipeline)
        encoder.setBuffer(inBuffer, offset: 0, index: 0)
        encoder.setBuffer(outBuffer, offset: 0, index: 1)
        var sz = UInt32(hiddenSize)
        encoder.setBytes(&sz, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.dispatchThreads(MTLSize(width: hiddenSize, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let endExposed = getTimeNanos()
    let elapsedExposed = getElapsedSeconds(start: startExposed, end: endExposed)
    let opsExposed = Double(hiddenSize) * Double(iterations)

    let gopsHidden = opsHidden / elapsedHidden / 1e9
    let gopsExposed = opsExposed / elapsedExposed / 1e9

    print("Hidden (8x ops): \(String(format: "%.2f", gopsHidden)) GOPS")
    print("Exposed (1x op): \(String(format: "%.2f", gopsExposed)) GOPS")
    print("Hiding Factor:   \(String(format: "%.1fx", gopsHidden / gopsExposed))")

    // 4. BRANCH DIVERGENCE TEST
    print("\n--- 4. SIMD-Group Branch Divergence Analysis ---")

    guard let branchConvergedFunc = deepLibrary.makeFunction(name: "branch_converged"),
          let branchDivergentFunc = deepLibrary.makeFunction(name: "branch_divergent"),
          let convergedPipeline = try? device.makeComputePipelineState(function: branchConvergedFunc),
          let divergentPipeline = try? device.makeComputePipelineState(function: branchDivergentFunc) else {
        print("Failed to create branch pipelines")
        return
    }

    let branchSize = 1024 * 1024

    let startConverged = getTimeNanos()
    for _ in 0..<iterations {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(convergedPipeline)
        encoder.setBuffer(inBuffer, offset: 0, index: 0)
        encoder.setBuffer(outBuffer, offset: 0, index: 1)
        var sz = UInt32(branchSize)
        encoder.setBytes(&sz, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.dispatchThreads(MTLSize(width: branchSize, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let endConverged = getTimeNanos()
    let elapsedConverged = getElapsedSeconds(start: startConverged, end: endConverged)
    let gopsConverged = Double(branchSize) * Double(iterations) / elapsedConverged / 1e9

    let startDivergent = getTimeNanos()
    for _ in 0..<iterations {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(divergentPipeline)
        encoder.setBuffer(inBuffer, offset: 0, index: 0)
        encoder.setBuffer(outBuffer, offset: 0, index: 1)
        var sz = UInt32(branchSize)
        encoder.setBytes(&sz, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.dispatchThreads(MTLSize(width: branchSize, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let endDivergent = getTimeNanos()
    let elapsedDivergent = getElapsedSeconds(start: startDivergent, end: endDivergent)
    let gopsDivergent = Double(branchSize) * Double(iterations) / elapsedDivergent / 1e9

    print("Converged:       \(String(format: "%.2f", gopsConverged)) GOPS")
    print("Divergent:      \(String(format: "%.2f", gopsDivergent)) GOPS")
    print("Divergence Cost: \(String(format: "%.1fx", gopsConverged / gopsDivergent))")

    // 5. ATOMIC CONTENTION TEST
    print("\n--- 5. Atomic Operation Contention Analysis ---")

    guard let atomicLowFunc = deepLibrary.makeFunction(name: "atomic_low_contention"),
          let atomicHighFunc = deepLibrary.makeFunction(name: "atomic_high_contention"),
          let atomicLowPipeline = try? device.makeComputePipelineState(function: atomicLowFunc),
          let atomicHighPipeline = try? device.makeComputePipelineState(function: atomicHighFunc) else {
        print("Failed to create atomic pipelines")
        return
    }

    let atomicSize = 256 * 1024

    guard let atomicBuffer = device.makeBuffer(length: atomicSize * MemoryLayout<UInt32>.size, options: .storageModeShared) else {
        return
    }

    let startLowContention = getTimeNanos()
    for _ in 0..<iterations {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(atomicLowPipeline)
        encoder.setBuffer(atomicBuffer, offset: 0, index: 0)
        encoder.setBuffer(outBuffer, offset: 0, index: 1)
        var sz = UInt32(atomicSize)
        encoder.setBytes(&sz, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.dispatchThreads(MTLSize(width: atomicSize, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let endLowContention = getTimeNanos()
    let elapsedLow = getElapsedSeconds(start: startLowContention, end: endLowContention)
    let gopsLow = Double(atomicSize) * Double(iterations) / elapsedLow / 1e9

    let startHighContention = getTimeNanos()
    for _ in 0..<iterations {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(atomicHighPipeline)
        encoder.setBuffer(atomicBuffer, offset: 0, index: 0)
        encoder.setBuffer(outBuffer, offset: 0, index: 1)
        var sz = UInt32(atomicSize)
        encoder.setBytes(&sz, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.dispatchThreads(MTLSize(width: atomicSize, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let endHighContention = getTimeNanos()
    let elapsedHigh = getElapsedSeconds(start: startHighContention, end: endHighContention)
    let gopsHigh = Double(atomicSize) * Double(iterations) / elapsedHigh / 1e9

    print("Low Contention:  \(String(format: "%.3f", gopsLow)) GOPS")
    print("High Contention: \(String(format: "%.3f", gopsHigh)) GOPS")
    print("Contention Cost: \(String(format: "%.1fx", gopsLow / gopsHigh))")

    // 6. REGISTER PRESSURE TEST
    print("\n--- 6. Register Pressure Analysis ---")

    guard let regLowFunc = deepLibrary.makeFunction(name: "register_low"),
          let regHighFunc = deepLibrary.makeFunction(name: "register_high"),
          let regLowPipeline = try? device.makeComputePipelineState(function: regLowFunc),
          let regHighPipeline = try? device.makeComputePipelineState(function: regHighFunc) else {
        print("Failed to create register pipelines")
        return
    }

    let regSize = 1024 * 1024

    let startLowReg = getTimeNanos()
    for _ in 0..<iterations {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(regLowPipeline)
        encoder.setBuffer(inBuffer, offset: 0, index: 0)
        encoder.setBuffer(outBuffer, offset: 0, index: 1)
        var sz = UInt32(regSize)
        encoder.setBytes(&sz, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.dispatchThreads(MTLSize(width: regSize, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let endLowReg = getTimeNanos()
    let elapsedLowReg = getElapsedSeconds(start: startLowReg, end: endLowReg)
    let gopsLowReg = Double(regSize) * Double(iterations) / elapsedLowReg / 1e9

    let startHighReg = getTimeNanos()
    for _ in 0..<iterations {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(regHighPipeline)
        encoder.setBuffer(inBuffer, offset: 0, index: 0)
        encoder.setBuffer(outBuffer, offset: 0, index: 1)
        var sz = UInt32(regSize)
        encoder.setBytes(&sz, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.dispatchThreads(MTLSize(width: regSize, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let endHighReg = getTimeNanos()
    let elapsedHighReg = getElapsedSeconds(start: startHighReg, end: endHighReg)
    let gopsHighReg = Double(regSize) * Double(iterations) / elapsedHighReg / 1e9

    print("Low Register:    \(String(format: "%.2f", gopsLowReg)) GOPS")
    print("High Register:   \(String(format: "%.2f", gopsHighReg)) GOPS")
    print("Register Cost:   \(String(format: "%.1fx", gopsLowReg / gopsHighReg))")

    // 7. CONSTANT MEMORY TEST
    print("\n--- 7. Constant Memory Broadcast Analysis ---")

    guard let constReadFunc = deepLibrary.makeFunction(name: "constant_read"),
          let constScatteredFunc = deepLibrary.makeFunction(name: "constant_scattered"),
          let constBroadcastFunc = deepLibrary.makeFunction(name: "constant_broadcast"),
          let constReadPipeline = try? device.makeComputePipelineState(function: constReadFunc),
          let constScatteredPipeline = try? device.makeComputePipelineState(function: constScatteredFunc),
          let constBroadcastPipeline = try? device.makeComputePipelineState(function: constBroadcastFunc) else {
        print("Failed to create constant memory pipelines")
        return
    }

    let constSize = 1024 * 1024

    guard let devBuffer = device.makeBuffer(length: constSize * MemoryLayout<Float>.size, options: .storageModeShared),
          let cstBuffer = device.makeBuffer(length: 1024 * MemoryLayout<Float>.size, options: .storageModeShared),
          let outBufferC = device.makeBuffer(length: constSize * MemoryLayout<Float>.size, options: .storageModeShared) else {
        return
    }

    let devPtr = devBuffer.contents().assumingMemoryBound(to: Float.self)
    for i in 0..<constSize { devPtr[i] = Float(i % 256) / 256.0 }
    let cstPtr = cstBuffer.contents().assumingMemoryBound(to: Float.self)
    for i in 0..<1024 { cstPtr[i] = 1.5 }

    var constSz = UInt32(constSize)

    // Constant broadcast (all threads same value)
    let startBroadcast = getTimeNanos()
    for _ in 0..<iterations {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(constBroadcastPipeline)
        encoder.setBuffer(devBuffer, offset: 0, index: 0)
        encoder.setBuffer(cstBuffer, offset: 0, index: 1)
        encoder.setBuffer(outBufferC, offset: 0, index: 2)
        encoder.setBytes(&constSz, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.dispatchThreads(MTLSize(width: constSize, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let endBroadcast = getTimeNanos()
    let elapsedBroadcast = getElapsedSeconds(start: startBroadcast, end: endBroadcast)
    let gopsBroadcast = Double(constSize) * Double(iterations) / elapsedBroadcast / 1e9

    // Constant scattered (different values per thread)
    let startScattered = getTimeNanos()
    for _ in 0..<iterations {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(constScatteredPipeline)
        encoder.setBuffer(devBuffer, offset: 0, index: 0)
        encoder.setBuffer(cstBuffer, offset: 0, index: 1)
        encoder.setBuffer(outBufferC, offset: 0, index: 2)
        encoder.setBytes(&constSz, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.dispatchThreads(MTLSize(width: constSize, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let endScattered = getTimeNanos()
    let elapsedScattered = getElapsedSeconds(start: startScattered, end: endScattered)
    let gopsScattered = Double(constSize) * Double(iterations) / elapsedScattered / 1e9

    print("Constant Broadcast (same value): \(String(format: "%.2f", gopsBroadcast)) GOPS")
    print("Constant Scattered (diff values): \(String(format: "%.2f", gopsScattered)) GOPS")
    print("Constant Cache Efficiency: \(String(format: "%.1fx", gopsBroadcast / gopsScattered))")

    // 8. MATMUL TILE SIZE OPTIMIZATION
    print("\n--- 8. MatMul Optimal Tile Size Analysis ---")

    guard let tile8Func = deepLibrary.makeFunction(name: "matmul_tiled_8"),
          let tile16Func = deepLibrary.makeFunction(name: "matmul_tiled_16"),
          let tile32Func = deepLibrary.makeFunction(name: "matmul_tiled_32"),
          let tile8Pipeline = try? device.makeComputePipelineState(function: tile8Func),
          let tile16Pipeline = try? device.makeComputePipelineState(function: tile16Func),
          let tile32Pipeline = try? device.makeComputePipelineState(function: tile32Func) else {
        print("Failed to create tile pipelines")
        return
    }

    let matSize: UInt32 = 512
    let matSizeInt = Int(matSize)
    let matmulIterations = 10

    let aSize = matSizeInt * matSizeInt
    guard let aBuffer = device.makeBuffer(length: aSize * MemoryLayout<Float>.size, options: .storageModeShared),
          let bBuffer = device.makeBuffer(length: aSize * MemoryLayout<Float>.size, options: .storageModeShared),
          let cBuffer = device.makeBuffer(length: aSize * MemoryLayout<Float>.size, options: .storageModeShared),
          let sharedA8 = device.makeBuffer(length: 8 * 8 * MemoryLayout<Float>.size, options: .storageModeShared),
          let sharedB8 = device.makeBuffer(length: 8 * 8 * MemoryLayout<Float>.size, options: .storageModeShared),
          let sharedA16 = device.makeBuffer(length: 16 * 16 * MemoryLayout<Float>.size, options: .storageModeShared),
          let sharedB16 = device.makeBuffer(length: 16 * 16 * MemoryLayout<Float>.size, options: .storageModeShared),
          let sharedA32 = device.makeBuffer(length: 32 * 32 * MemoryLayout<Float>.size, options: .storageModeShared),
          let sharedB32 = device.makeBuffer(length: 32 * 32 * MemoryLayout<Float>.size, options: .storageModeShared) else {
        return
    }

    var M = matSize, K = matSize, N = matSize

    // Tile 8
    let start8 = getTimeNanos()
    for _ in 0..<matmulIterations {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(tile8Pipeline)
        encoder.setBuffer(aBuffer, offset: 0, index: 0)
        encoder.setBuffer(bBuffer, offset: 0, index: 1)
        encoder.setBuffer(cBuffer, offset: 0, index: 2)
        encoder.setBytes(&M, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&K, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&N, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBuffer(sharedA8, offset: 0, index: 6)
        encoder.setBuffer(sharedB8, offset: 0, index: 7)
        encoder.dispatchThreads(MTLSize(width: Int(N), height: Int(M), depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let end8 = getTimeNanos()
    let elapsed8 = getElapsedSeconds(start: start8, end: end8)
    let flops8 = 2.0 * Double(M) * Double(K) * Double(N) * Double(matmulIterations)
    let gflops8 = flops8 / elapsed8 / 1e9

    // Tile 16
    let start16 = getTimeNanos()
    for _ in 0..<matmulIterations {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(tile16Pipeline)
        encoder.setBuffer(aBuffer, offset: 0, index: 0)
        encoder.setBuffer(bBuffer, offset: 0, index: 1)
        encoder.setBuffer(cBuffer, offset: 0, index: 2)
        encoder.setBytes(&M, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&K, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&N, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBuffer(sharedA16, offset: 0, index: 6)
        encoder.setBuffer(sharedB16, offset: 0, index: 7)
        encoder.dispatchThreads(MTLSize(width: Int(N), height: Int(M), depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let end16 = getTimeNanos()
    let elapsed16 = getElapsedSeconds(start: start16, end: end16)
    let flops16 = 2.0 * Double(M) * Double(K) * Double(N) * Double(matmulIterations)
    let gflops16 = flops16 / elapsed16 / 1e9

    // Tile 32
    let start32 = getTimeNanos()
    for _ in 0..<matmulIterations {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(tile32Pipeline)
        encoder.setBuffer(aBuffer, offset: 0, index: 0)
        encoder.setBuffer(bBuffer, offset: 0, index: 1)
        encoder.setBuffer(cBuffer, offset: 0, index: 2)
        encoder.setBytes(&M, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&K, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&N, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBuffer(sharedA32, offset: 0, index: 6)
        encoder.setBuffer(sharedB32, offset: 0, index: 7)
        encoder.dispatchThreads(MTLSize(width: Int(N), height: Int(M), depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let end32 = getTimeNanos()
    let elapsed32 = getElapsedSeconds(start: start32, end: end32)
    let flops32 = 2.0 * Double(M) * Double(K) * Double(N) * Double(matmulIterations)
    let gflops32 = flops32 / elapsed32 / 1e9

    print("Tile Size 8:  \(String(format: "%.2f", gflops8)) GFLOPS")
    print("Tile Size 16: \(String(format: "%.2f", gflops16)) GFLOPS")
    print("Tile Size 32: \(String(format: "%.2f", gflops32)) GFLOPS")
    let bestTile = max(gflops8, max(gflops16, gflops32))
    let bestName = bestTile == gflops8 ? "8" : (bestTile == gflops16 ? "16" : "32")
    print("Best Tile: \(bestName) with \(String(format: "%.2f", bestTile)) GFLOPS")

    // 9. COMMAND BUFFER BATCHING TEST
    print("\n--- 9. Command Buffer Batching Analysis ---")

    guard let batchAFunc = deepLibrary.makeFunction(name: "batch_kernel_a"),
          let batchBFunc = deepLibrary.makeFunction(name: "batch_kernel_b"),
          let batchCFunc = deepLibrary.makeFunction(name: "batch_kernel_c"),
          let batchAPipeline = try? device.makeComputePipelineState(function: batchAFunc),
          let batchBPipeline = try? device.makeComputePipelineState(function: batchBFunc),
          let batchCPipeline = try? device.makeComputePipelineState(function: batchCFunc) else {
        print("Failed to create batch pipelines")
        return
    }

    let batchSize = 1024 * 1024
    let batchIterations = 100
    guard let batchIn = device.makeBuffer(length: batchSize * MemoryLayout<Float>.size, options: .storageModeShared),
          let batchOut = device.makeBuffer(length: batchSize * MemoryLayout<Float>.size, options: .storageModeShared) else {
        return
    }
    var batchSz = UInt32(batchSize)

    // Batched: 3 kernels in single command buffer
    let startBatched = getTimeNanos()
    for _ in 0..<batchIterations {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(batchAPipeline)
        encoder.setBuffer(batchIn, offset: 0, index: 0)
        encoder.setBuffer(batchOut, offset: 0, index: 1)
        encoder.setBytes(&batchSz, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.dispatchThreads(MTLSize(width: batchSize, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

        encoder.setComputePipelineState(batchBPipeline)
        encoder.setBuffer(batchIn, offset: 0, index: 0)
        encoder.setBuffer(batchOut, offset: 0, index: 1)
        encoder.setBytes(&batchSz, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.dispatchThreads(MTLSize(width: batchSize, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

        encoder.setComputePipelineState(batchCPipeline)
        encoder.setBuffer(batchIn, offset: 0, index: 0)
        encoder.setBuffer(batchOut, offset: 0, index: 1)
        encoder.setBytes(&batchSz, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.dispatchThreads(MTLSize(width: batchSize, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let endBatched = getTimeNanos()
    let elapsedBatched = getElapsedSeconds(start: startBatched, end: endBatched)
    let gopsBatched = 3.0 * Double(batchSize) * Double(batchIterations) / elapsedBatched / 1e9

    // Sequential: 3 separate command buffers
    let startSequential = getTimeNanos()
    for _ in 0..<batchIterations {
        guard let cmdA = queue.makeCommandBuffer(),
              let encA = cmdA.makeComputeCommandEncoder() else { continue }
        encA.setComputePipelineState(batchAPipeline)
        encA.setBuffer(batchIn, offset: 0, index: 0)
        encA.setBuffer(batchOut, offset: 0, index: 1)
        encA.setBytes(&batchSz, length: MemoryLayout<UInt32>.size, index: 2)
        encA.dispatchThreads(MTLSize(width: batchSize, height: 1, depth: 1),
                         threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encA.endEncoding()
        cmdA.commit()
        cmdA.waitUntilCompleted()

        guard let cmdB = queue.makeCommandBuffer(),
              let encB = cmdB.makeComputeCommandEncoder() else { continue }
        encB.setComputePipelineState(batchBPipeline)
        encB.setBuffer(batchIn, offset: 0, index: 0)
        encB.setBuffer(batchOut, offset: 0, index: 1)
        encB.setBytes(&batchSz, length: MemoryLayout<UInt32>.size, index: 2)
        encB.dispatchThreads(MTLSize(width: batchSize, height: 1, depth: 1),
                         threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encB.endEncoding()
        cmdB.commit()
        cmdB.waitUntilCompleted()

        guard let cmdC = queue.makeCommandBuffer(),
              let encC = cmdC.makeComputeCommandEncoder() else { continue }
        encC.setComputePipelineState(batchCPipeline)
        encC.setBuffer(batchIn, offset: 0, index: 0)
        encC.setBuffer(batchOut, offset: 0, index: 1)
        encC.setBytes(&batchSz, length: MemoryLayout<UInt32>.size, index: 2)
        encC.dispatchThreads(MTLSize(width: batchSize, height: 1, depth: 1),
                         threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encC.endEncoding()
        cmdC.commit()
        cmdC.waitUntilCompleted()
    }
    let endSequential = getTimeNanos()
    let elapsedSequential = getElapsedSeconds(start: startSequential, end: endSequential)
    let gopsSequential = 3.0 * Double(batchSize) * Double(batchIterations) / elapsedSequential / 1e9

    print("Batched (3 kernels/cmd): \(String(format: "%.2f", gopsBatched)) GOPS")
    print("Sequential (1 kernel/cmd): \(String(format: "%.2f", gopsSequential)) GOPS")
    print("Batch Speedup: \(String(format: "%.2fx", gopsBatched / gopsSequential))")

    // 10. OCCUPANCY ANALYSIS
    print("\n--- 10. Occupancy Analysis (Shared Memory Tradeoff) ---")

    guard let occLowFunc = deepLibrary.makeFunction(name: "occupancy_low"),
          let occMedFunc = deepLibrary.makeFunction(name: "occupancy_med"),
          let occHighFunc = deepLibrary.makeFunction(name: "occupancy_high"),
          let occLowPipeline = try? device.makeComputePipelineState(function: occLowFunc),
          let occMedPipeline = try? device.makeComputePipelineState(function: occMedFunc),
          let occHighPipeline = try? device.makeComputePipelineState(function: occHighFunc) else {
        print("Failed to create occupancy pipelines")
        return
    }

    let occSize = 512 * 1024
    let occIterations = 50
    let occSharedSize = 32 * 1024 // 32KB max shared

    guard let occIn = device.makeBuffer(length: occSize * MemoryLayout<Float>.size, options: .storageModeShared),
          let occOut = device.makeBuffer(length: occSize * MemoryLayout<Float>.size, options: .storageModeShared),
          let occSharedLow = device.makeBuffer(length: 32 * 1024, options: .storageModeShared),
          let occSharedMed = device.makeBuffer(length: 32 * 1024, options: .storageModeShared),
          let occSharedHigh = device.makeBuffer(length: 32 * 1024, options: .storageModeShared) else {
        return
    }
    var occSz = UInt32(occSize)

    // Low occupancy: 32 threads, 1KB shared per thread (32 threads max)
    let startLow = getTimeNanos()
    for _ in 0..<occIterations {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(occLowPipeline)
        encoder.setBuffer(occIn, offset: 0, index: 0)
        encoder.setBuffer(occOut, offset: 0, index: 1)
        encoder.setBuffer(occSharedLow, offset: 0, index: 2)
        encoder.setBytes(&occSz, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.dispatchThreads(MTLSize(width: occSize, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let endLow = getTimeNanos()
    let elapsedLowOcc = getElapsedSeconds(start: startLow, end: endLow)
    let gopsLowOcc = Double(occSize) * Double(occIterations) / elapsedLowOcc / 1e9

    // Medium occupancy: 128 threads, 256B shared per thread (128 threads max)
    let startMed = getTimeNanos()
    for _ in 0..<occIterations {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(occMedPipeline)
        encoder.setBuffer(occIn, offset: 0, index: 0)
        encoder.setBuffer(occOut, offset: 0, index: 1)
        encoder.setBuffer(occSharedMed, offset: 0, index: 2)
        encoder.setBytes(&occSz, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.dispatchThreads(MTLSize(width: occSize, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let endMed = getTimeNanos()
    let elapsedMedOcc = getElapsedSeconds(start: startMed, end: endMed)
    let gopsMedOcc = Double(occSize) * Double(occIterations) / elapsedMedOcc / 1e9

    // High occupancy: 512 threads, 64B shared per thread (512 threads max)
    let startHigh = getTimeNanos()
    for _ in 0..<occIterations {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(occHighPipeline)
        encoder.setBuffer(occIn, offset: 0, index: 0)
        encoder.setBuffer(occOut, offset: 0, index: 1)
        encoder.setBuffer(occSharedHigh, offset: 0, index: 2)
        encoder.setBytes(&occSz, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.dispatchThreads(MTLSize(width: occSize, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 512, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let endHigh = getTimeNanos()
    let elapsedHighOcc = getElapsedSeconds(start: startHigh, end: endHigh)
    let gopsHighOccupancy = Double(occSize) * Double(occIterations) / elapsedHighOcc / 1e9

    print("Low Occupancy (32 thr, 1KB):     \(String(format: "%.2f", gopsLowOcc)) GOPS")
    print("Med Occupancy (128 thr, 256B):   \(String(format: "%.2f", gopsMedOcc)) GOPS")
    print("High Occupancy (512 thr, 64B):   \(String(format: "%.2f", gopsHighOccupancy)) GOPS")
    let bestOcc = max(gopsLowOcc, max(gopsMedOcc, gopsHighOccupancy))
    let bestOccName = bestOcc == gopsLowOcc ? "Low" : (bestOcc == gopsMedOcc ? "Med" : "High")
    print("Best Occupancy: \(bestOccName) with \(String(format: "%.2f", bestOcc)) GOPS")

    print("\n" + String(repeating: "=", count: 60))
    print("Deep GPU Architecture Research Complete")
    print(String(repeating: "=", count: 60))
}

// MARK: - Deep Memory Bandwidth Research

func testDeepMemoryBandwidth(device: MTLDevice, queue: MTLCommandQueue) throws {
    print("\n" + String(repeating: "=", count: 70))
    print("=== DEEP MEMORY BANDWIDTH RESEARCH ===")
    print(String(repeating: "=", count: 70))

    let deepLibrary: MTLLibrary
    do {
        deepLibrary = try device.makeLibrary(source: deepShaderSource, options: nil)
    } catch {
        print("Failed to compile deep memory shaders: \(error)")
        return
    }

    // ============================================================
    // 1. BANDWIDTH VS BUFFER SIZE (Saturation Point)
    // ============================================================
    print("\n--- 1. Bandwidth vs Buffer Size (Saturation Point) ---")

    let sizes: [Int] = [
        64 * 1024,           // 64KB - L1 cache size
        256 * 1024,          // 256KB - L2 cache size
        1024 * 1024,         // 1MB
        8 * 1024 * 1024,     // 8MB
        64 * 1024 * 1024,    // 64MB
        256 * 1024 * 1024    // 256MB
    ]
    let iterations = 50

    // Get kernel functions
    guard let writeSeqFunc = deepLibrary.makeFunction(name: "mem_write_seq"),
          let readSeqFunc = deepLibrary.makeFunction(name: "mem_read_seq"),
          let readWriteFunc = deepLibrary.makeFunction(name: "mem_read_write"),
          let writeCombineFunc = deepLibrary.makeFunction(name: "mem_write_combine"),
          let readComputeFunc = deepLibrary.makeFunction(name: "mem_read_compute") else {
        print("Failed to get memory bandwidth kernels")
        return
    }

    guard let writePipeline = try? device.makeComputePipelineState(function: writeSeqFunc),
          let readPipeline = try? device.makeComputePipelineState(function: readSeqFunc),
          let readWritePipeline = try? device.makeComputePipelineState(function: readWriteFunc),
          let writeCombinePipeline = try? device.makeComputePipelineState(function: writeCombineFunc),
          let readComputePipeline = try? device.makeComputePipelineState(function: readComputeFunc) else {
        print("Failed to create pipelines")
        return
    }

    print("\nBuffer Size   | Write Bandwidth | Read Bandwidth | ReadWrite | WriteCombine | ReadCompute")
    print(String(repeating: "-", count: 85))

    for size in sizes {
        guard let buffer = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared),
              let buffer2 = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared) else {
            continue
        }

        var sizeVal = UInt32(size)

        // Warmup
        for _ in 0..<3 {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(writePipeline)
            encoder.setBuffer(buffer, offset: 0, index: 0)
            encoder.setBytes(&sizeVal, length: MemoryLayout<UInt32>.size, index: 1)
            encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }

        // Write bandwidth
        let startWrite = getTimeNanos()
        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(writePipeline)
            encoder.setBuffer(buffer, offset: 0, index: 0)
            encoder.setBytes(&sizeVal, length: MemoryLayout<UInt32>.size, index: 1)
            encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        let endWrite = getTimeNanos()
        let writeBW = Double(size) * Double(iterations) * Double(MemoryLayout<Float>.size) / getElapsedSeconds(start: startWrite, end: endWrite) / 1e9

        // Read bandwidth
        let startRead = getTimeNanos()
        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(readPipeline)
            encoder.setBuffer(buffer, offset: 0, index: 0)
            encoder.setBuffer(buffer2, offset: 0, index: 1)
            var sz = UInt32(size)
            encoder.setBytes(&sz, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        let endRead = getTimeNanos()
        let readBW = Double(size) * Double(iterations) * Double(MemoryLayout<Float>.size) / getElapsedSeconds(start: startRead, end: endRead) / 1e9

        // Read-write bandwidth
        let startRW = getTimeNanos()
        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(readWritePipeline)
            encoder.setBuffer(buffer, offset: 0, index: 0)
            encoder.setBuffer(buffer2, offset: 0, index: 1)
            var sz = UInt32(size)
            encoder.setBytes(&sz, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        let endRW = getTimeNanos()
        let rwBW = 2.0 * Double(size) * Double(iterations) * Double(MemoryLayout<Float>.size) / getElapsedSeconds(start: startRW, end: endRW) / 1e9

        // Write combine bandwidth
        let startWC = getTimeNanos()
        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(writeCombinePipeline)
            encoder.setBuffer(buffer, offset: 0, index: 0)
            encoder.setBytes(&sizeVal, length: MemoryLayout<UInt32>.size, index: 1)
            encoder.dispatchThreads(MTLSize(width: size / 4, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        let endWC = getTimeNanos()
        let wcBW = Double(size) * Double(iterations) * Double(MemoryLayout<Float>.size) / getElapsedSeconds(start: startWC, end: endWC) / 1e9

        // Read with compute bandwidth
        let startRC = getTimeNanos()
        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(readComputePipeline)
            encoder.setBuffer(buffer, offset: 0, index: 0)
            encoder.setBuffer(buffer2, offset: 0, index: 1)
            var sz = UInt32(size)
            encoder.setBytes(&sz, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        let endRC = getTimeNanos()
        let rcBW = 2.0 * Double(size) * Double(iterations) * Double(MemoryLayout<Float>.size) / getElapsedSeconds(start: startRC, end: endRC) / 1e9

        let sizeStr = size >= 1024 * 1024 ? "\(size / (1024 * 1024))MB" : "\(size / 1024)KB"
        print("\(sizeStr.padStart(11)) | \(String(format: "%.2f", writeBW).padStart(14)) GB/s | \(String(format: "%.2f", readBW).padStart(12)) GB/s | \(String(format: "%.2f", rwBW).padStart(7)) | \(String(format: "%.2f", wcBW).padStart(11)) | \(String(format: "%.2f", rcBW).padStart(10))")
    }

    // ============================================================
    // 2. STRIDE EFFECT ON BANDWIDTH
    // ============================================================
    print("\n--- 2. Memory Stride Effect on Bandwidth ---")

    guard let readStrideFunc = deepLibrary.makeFunction(name: "mem_read_stride"),
          let writeStrideFunc = deepLibrary.makeFunction(name: "mem_write_stride"),
          let strideReadPipeline = try? device.makeComputePipelineState(function: readStrideFunc),
          let strideWritePipeline = try? device.makeComputePipelineState(function: writeStrideFunc) else {
        print("Failed to create stride pipelines")
        return
    }

    let strides: [Int] = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    let strideSize = 1024 * 1024

    print("\nStride | Read BW   | Write BW  | vs Sequential")
    print(String(repeating: "-", count: 50))

    guard let strideBufferR = device.makeBuffer(length: strideSize * 256 * MemoryLayout<Float>.size, options: .storageModeShared),
          let strideBufferW = device.makeBuffer(length: strideSize * 256 * MemoryLayout<Float>.size, options: .storageModeShared),
          let strideOut = device.makeBuffer(length: strideSize * MemoryLayout<Float>.size, options: .storageModeShared) else {
        return
    }

    for stride in strides {
        var sizeVal = UInt32(strideSize)

        // Read with stride
        let startR = getTimeNanos()
        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(strideReadPipeline)
            encoder.setBuffer(strideBufferR, offset: 0, index: 0)
            encoder.setBuffer(strideOut, offset: 0, index: 1)
            encoder.setBytes(&sizeVal, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.dispatchThreads(MTLSize(width: strideSize, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        let endR = getTimeNanos()
        let strideReadBW = Double(strideSize) * Double(iterations) * Double(MemoryLayout<Float>.size) / getElapsedSeconds(start: startR, end: endR) / 1e9

        // Write with stride
        let startW = getTimeNanos()
        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(strideWritePipeline)
            encoder.setBuffer(strideBufferW, offset: 0, index: 0)
            encoder.setBytes(&sizeVal, length: MemoryLayout<UInt32>.size, index: 1)
            encoder.dispatchThreads(MTLSize(width: strideSize, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        let endW = getTimeNanos()
        let strideWriteBW = Double(strideSize) * Double(iterations) * Double(MemoryLayout<Float>.size) / getElapsedSeconds(start: startW, end: endW) / 1e9

        let ratio = stride == 1 ? 1.0 : strideWriteBW / 0.75  // compare to sequential
        print("\(String(format: "%d", stride).padStart(6)) | \(String(format: "%.3f", strideReadBW).padStart(8)) | \(String(format: "%.3f", strideWriteBW).padStart(8)) | \(String(format: "%.2fx", strideWriteBW / 0.75))")
    }

    // ============================================================
    // 3. WRITE COMBINING EFFECT
    // ============================================================
    print("\n--- 3. Write Combining Effect (Multiple Writes per Thread) ---")

    print("Threads | Writes/Thread | Effective BW  | vs Single Write")
    print(String(repeating: "-", count: 60))

    guard let writeSeqPipe = try? device.makeComputePipelineState(function: writeSeqFunc) else {
        return
    }

    let threadCounts: [Int] = [256, 1024, 4096, 16384, 65536]
    let totalElements = 1024 * 1024

    for threads in threadCounts {
        var sizeVal = UInt32(totalElements / threads)

        // Single write per thread
        let start1 = getTimeNanos()
        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(writePipeline)
            encoder.setBuffer(strideBufferW, offset: 0, index: 0)
            encoder.setBytes(&sizeVal, length: MemoryLayout<UInt32>.size, index: 1)
            encoder.dispatchThreads(MTLSize(width: totalElements / threads, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: min(threads, 256), height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        let end1 = getTimeNanos()
        let bw1 = Double(totalElements) * Double(iterations) * Double(MemoryLayout<Float>.size) / getElapsedSeconds(start: start1, end: end1) / 1e9

        // 4 writes per thread
        let start4 = getTimeNanos()
        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(writeCombinePipeline)
            encoder.setBuffer(strideBufferW, offset: 0, index: 0)
            encoder.setBytes(&sizeVal, length: MemoryLayout<UInt32>.size, index: 1)
            encoder.dispatchThreads(MTLSize(width: totalElements / (4 * threads), height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: min(threads / 4, 256), height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        let end4 = getTimeNanos()
        let bw4 = Double(totalElements) * Double(iterations) * Double(MemoryLayout<Float>.size) / getElapsedSeconds(start: start4, end: end4) / 1e9

        print("\(String(format: "%d", threads).padStart(6)) | \(String(format: "1").padStart(13)) | \(String(format: "%.3f", bw1).padStart(13)) GB/s | -")
        print("\(String(format: "%d", threads).padStart(6)) | \(String(format: "4").padStart(13)) | \(String(format: "%.3f", bw4).padStart(13)) GB/s | \(String(format: "%.2fx", bw4 / bw1))")
    }

    // ============================================================
    // 4. READ-WRITE CONCURRENCY
    // ============================================================
    print("\n--- 4. Read vs Write vs ReadWrite Bandwidth Analysis ---")

    let testSize = 8 * 1024 * 1024
    guard let bufA = device.makeBuffer(length: testSize * MemoryLayout<Float>.size, options: .storageModeShared),
          let bufB = device.makeBuffer(length: testSize * MemoryLayout<Float>.size, options: .storageModeShared) else {
        return
    }

    var testSizeVal = UInt32(testSize)

    // Pure read
    let startR = getTimeNanos()
    for _ in 0..<iterations {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(readPipeline)
        encoder.setBuffer(bufA, offset: 0, index: 0)
        encoder.setBuffer(bufB, offset: 0, index: 1)
        encoder.setBytes(&testSizeVal, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.dispatchThreads(MTLSize(width: testSize, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let endR = getTimeNanos()
    let pureReadBW = Double(testSize) * Double(iterations) * Double(MemoryLayout<Float>.size) / getElapsedSeconds(start: startR, end: endR) / 1e9

    // Pure write
    let startW = getTimeNanos()
    for _ in 0..<iterations {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(writePipeline)
        encoder.setBuffer(bufA, offset: 0, index: 0)
        encoder.setBytes(&testSizeVal, length: MemoryLayout<UInt32>.size, index: 1)
        encoder.dispatchThreads(MTLSize(width: testSize, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let endW = getTimeNanos()
    let pureWriteBW = Double(testSize) * Double(iterations) * Double(MemoryLayout<Float>.size) / getElapsedSeconds(start: startW, end: endW) / 1e9

    // Read-write combined (theoretical max would be read + write)
    let startRW = getTimeNanos()
    for _ in 0..<iterations {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(readWritePipeline)
        encoder.setBuffer(bufA, offset: 0, index: 0)
        encoder.setBuffer(bufB, offset: 0, index: 1)
        encoder.setBytes(&testSizeVal, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.dispatchThreads(MTLSize(width: testSize, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let endRW = getTimeNanos()
    let readWriteBW = 2.0 * Double(testSize) * Double(iterations) * Double(MemoryLayout<Float>.size) / getElapsedSeconds(start: startRW, end: endRW) / 1e9

    print("\nPure Read:   \(String(format: "%.2f", pureReadBW)) GB/s")
    print("Pure Write:  \(String(format: "%.2f", pureWriteBW)) GB/s")
    print("ReadWrite:   \(String(format: "%.2f", readWriteBW)) GB/s (2x transfer)")
    print("Theoretical Max (Read + Write): \(String(format: "%.2f", pureReadBW + pureWriteBW)) GB/s")
    print("Efficiency: \(String(format: "%.1f", readWriteBW / (pureReadBW + pureWriteBW) * 100))%")

    // ============================================================
    // 5. BURST READ (Vectorized float4)
    // ============================================================
    print("\n--- 5. Burst Read with Float4 Vectorization ---")

    guard let burstFunc = deepLibrary.makeFunction(name: "mem_read_burst"),
          let burstPipeline = try? device.makeComputePipelineState(function: burstFunc),
          let burstIn = device.makeBuffer(length: testSize * MemoryLayout<Float>.size, options: .storageModeShared),
          let burstOut = device.makeBuffer(length: testSize * MemoryLayout<Float>.size, options: .storageModeShared) else {
        return
    }

    let float4Size = testSize / 4
    var float4SizeVal = UInt32(float4Size)

    let startBurst = getTimeNanos()
    for _ in 0..<iterations {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(burstPipeline)
        encoder.setBuffer(burstIn, offset: 0, index: 0)
        encoder.setBuffer(burstOut, offset: 0, index: 1)
        encoder.setBytes(&float4SizeVal, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.dispatchThreads(MTLSize(width: float4Size, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let endBurst = getTimeNanos()
    let burstBW = 2.0 * Double(testSize) * Double(iterations) * Double(MemoryLayout<Float>.size) / getElapsedSeconds(start: startBurst, end: endBurst) / 1e9

    print("Float4 Burst Read: \(String(format: "%.2f", burstBW)) GB/s")
    print("vs Scalar Read: \(String(format: "%.2f", pureReadBW)) GB/s")
    print("Vectorization Gain: \(String(format: "%.2fx", burstBW / pureReadBW))")

    print("\n" + String(repeating: "=", count: 70))
    print("Deep Memory Bandwidth Research Complete")
    print(String(repeating: "=", count: 70))
}

// MARK: - HIGH PERFORMANCE MEMORY BANDWIDTH TEST

func testHighPerformanceBandwidth(device: MTLDevice, queue: MTLCommandQueue) throws {
    print("\n" + String(repeating: "=", count: 70))
    print("=== HIGH PERFORMANCE MEMORY BANDWIDTH TEST ===")
    print(String(repeating: "=", count: 70))

    // Compile deep shader library for this test
    let deepLibrary: MTLLibrary
    do {
        deepLibrary = try device.makeLibrary(source: deepShaderSource, options: nil)
    } catch {
        print("Failed to compile deep shaders: \(error)")
        return
    }

    let iterations = 100
    let testSize = 64 * 1024 * 1024  // 64MB for meaningful test

    // ============================================================
    // 1. BLIT COMMAND (DEDICATED COPY ENGINE)
    // ============================================================
    print("\n--- 1. Blit Command (Dedicated Copy Engine) ---")

    guard let bufA = device.makeBuffer(length: testSize * MemoryLayout<Float>.size, options: .storageModeShared),
          let bufB = device.makeBuffer(length: testSize * MemoryLayout<Float>.size, options: .storageModeShared) else {
        print("Failed to create buffers")
        return
    }

    // Initialize buffer
    let ptr = bufA.contents().assumingMemoryBound(to: Float.self)
    for i in 0..<testSize { ptr[i] = Float(i) }

    // Warmup
    for _ in 0..<3 {
        guard let cmd = queue.makeCommandBuffer(),
              let blit = cmd.makeBlitCommandEncoder() else { continue }
        blit.copy(from: bufA, sourceOffset: 0, to: bufB, destinationOffset: 0, size: testSize * MemoryLayout<Float>.size)
        blit.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }

    // Blit copy test
    let startBlit = getTimeNanos()
    for _ in 0..<iterations {
        guard let cmd = queue.makeCommandBuffer(),
              let blit = cmd.makeBlitCommandEncoder() else { continue }
        blit.copy(from: bufA, sourceOffset: 0, to: bufB, destinationOffset: 0, size: testSize * MemoryLayout<Float>.size)
        blit.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let endBlit = getTimeNanos()
    let blitBW = Double(testSize) * Double(iterations) * Double(MemoryLayout<Float>.size) / getElapsedSeconds(start: startBlit, end: endBlit) / 1e9

    print("Blit Copy (64MB): \(String(format: "%.2f", blitBW)) GB/s")

    // ============================================================
    // 2. SHARED STORAGE MODE WRITE
    // ============================================================
    print("\n--- 2. Shared Storage Mode Write ---")

    guard let sharedBuf = device.makeBuffer(length: testSize * MemoryLayout<Float>.size, options: .storageModeShared) else {
        print("Failed to create shared buffer")
        return
    }

    let sharedPtr = sharedBuf.contents().assumingMemoryBound(to: Float.self)
    for i in 0..<testSize { sharedPtr[i] = Float(i) }

    // Get kernels from deepLibrary
    guard let writeKernel = deepLibrary.makeFunction(name: "mem_write_seq"),
          let writePipeline = try? device.makeComputePipelineState(function: writeKernel) else {
        return
    }

    var sizeVal = UInt32(testSize)

    let startSharedWrite = getTimeNanos()
    for _ in 0..<iterations {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(writePipeline)
        encoder.setBuffer(sharedBuf, offset: 0, index: 0)
        encoder.setBytes(&sizeVal, length: MemoryLayout<UInt32>.size, index: 1)
        encoder.dispatchThreads(MTLSize(width: testSize, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let endSharedWrite = getTimeNanos()
    let sharedWriteBW = Double(testSize) * Double(iterations) * Double(MemoryLayout<Float>.size) / getElapsedSeconds(start: startSharedWrite, end: endSharedWrite) / 1e9

    print("Shared Mode Write: \(String(format: "%.2f", sharedWriteBW)) GB/s")

    // ============================================================
    // 3. MAXIMUM THREADGROUP SIZE
    // ============================================================
    print("\n--- 3. Maximum Threadgroup Size Effect ---")

    let threadgroupSizes: [Int] = [64, 128, 256, 512, 1024]
    let gridSize = 16 * 1024 * 1024  // Large grid

    print("\nThreadgroup | Grid Size    | Bandwidth")
    print(String(repeating: "-", count: 45))

    for tgSize in threadgroupSizes {
        let startMax = getTimeNanos()
        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(writePipeline)
            encoder.setBuffer(sharedBuf, offset: 0, index: 0)
            encoder.setBytes(&sizeVal, length: MemoryLayout<UInt32>.size, index: 1)
            encoder.dispatchThreads(MTLSize(width: gridSize, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        let endMax = getTimeNanos()
        let maxBW = Double(gridSize) * Double(iterations) * Double(MemoryLayout<Float>.size) / getElapsedSeconds(start: startMax, end: endMax) / 1e9

        print("\(String(format: "%d", tgSize).padStart(12)) | \(String(format: "%d", gridSize).padStart(12)) | \(String(format: "%.2f", maxBW)) GB/s")
    }

    // ============================================================
    // 4. ASYNC EXECUTION WITH TRIPLE BUFFERING
    // ============================================================
    print("\n--- 4. Async Execution (Triple Buffering Simulation) ---")

    guard let buf1 = device.makeBuffer(length: testSize * MemoryLayout<Float>.size, options: .storageModeShared),
          let buf2 = device.makeBuffer(length: testSize * MemoryLayout<Float>.size, options: .storageModeShared),
          let buf3 = device.makeBuffer(length: testSize * MemoryLayout<Float>.size, options: .storageModeShared) else {
        return
    }

    // Triple buffering: start next operation before waiting
    let startAsync = getTimeNanos()

    for _ in 0..<iterations / 3 {
        guard let cmd1 = queue.makeCommandBuffer(),
              let enc1 = cmd1.makeComputeCommandEncoder() else { continue }
        enc1.setComputePipelineState(writePipeline)
        enc1.setBuffer(buf1, offset: 0, index: 0)
        enc1.setBytes(&sizeVal, length: MemoryLayout<UInt32>.size, index: 1)
        enc1.dispatchThreads(MTLSize(width: testSize, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
        enc1.endEncoding()

        guard let cmd2 = queue.makeCommandBuffer(),
              let enc2 = cmd2.makeComputeCommandEncoder() else { continue }
        enc2.setComputePipelineState(writePipeline)
        enc2.setBuffer(buf2, offset: 0, index: 0)
        enc2.setBytes(&sizeVal, length: MemoryLayout<UInt32>.size, index: 1)
        enc2.dispatchThreads(MTLSize(width: testSize, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
        enc2.endEncoding()

        guard let cmd3 = queue.makeCommandBuffer(),
              let enc3 = cmd3.makeComputeCommandEncoder() else { continue }
        enc3.setComputePipelineState(writePipeline)
        enc3.setBuffer(buf3, offset: 0, index: 0)
        enc3.setBytes(&sizeVal, length: MemoryLayout<UInt32>.size, index: 1)
        enc3.dispatchThreads(MTLSize(width: testSize, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
        enc3.endEncoding()

        cmd1.commit()
        cmd2.commit()
        cmd3.commit()

        cmd1.waitUntilCompleted()
        cmd2.waitUntilCompleted()
        cmd3.waitUntilCompleted()
    }

    let endAsync = getTimeNanos()
    let asyncBW = Double(testSize) * Double(iterations) * Double(MemoryLayout<Float>.size) / getElapsedSeconds(start: startAsync, end: endAsync) / 1e9

    print("Async (3x parallel): \(String(format: "%.2f", asyncBW)) GB/s")

    // ============================================================
    // 5. BURST WRITE (ALL THREADS WRITE AT ONCE)
    // ============================================================
    print("\n--- 5. Maximum Parallel Burst Write ---")

    guard let burstKernel = deepLibrary.makeFunction(name: "mem_write_combine"),
          let burstPipeline = try? device.makeComputePipelineState(function: burstKernel) else {
        return
    }

    let elementsPerThread = 16

    let startBurst = getTimeNanos()
    for _ in 0..<iterations {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(burstPipeline)
        encoder.setBuffer(sharedBuf, offset: 0, index: 0)
        encoder.setBytes(&sizeVal, length: MemoryLayout<UInt32>.size, index: 1)
        let threadCount = testSize / elementsPerThread
        encoder.dispatchThreads(MTLSize(width: threadCount, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let endBurst = getTimeNanos()
    let burstBW = Double(testSize) * Double(iterations) * Double(MemoryLayout<Float>.size) / getElapsedSeconds(start: startBurst, end: endBurst) / 1e9

    print("Burst Write (16 elements/thread): \(String(format: "%.2f", burstBW)) GB/s")

    // ============================================================
    // 6. READ WITH MAX THREAD UTILIZATION
    // ============================================================
    print("\n--- 6. Maximum Read Bandwidth Test ---")

    guard let readKernel = deepLibrary.makeFunction(name: "mem_read_seq"),
          let readPipeline = try? device.makeComputePipelineState(function: readKernel) else {
        return
    }

    // Initialize source
    let readPtr = sharedBuf.contents().assumingMemoryBound(to: Float.self)
    for i in 0..<testSize { readPtr[i] = Float(i) }

    guard let readDst = device.makeBuffer(length: testSize * MemoryLayout<Float>.size, options: .storageModeShared) else {
        return
    }

    // Use blit to clear first to ensure fresh read
    for _ in 0..<3 {
        guard let cmd = queue.makeCommandBuffer(),
              let blit = cmd.makeBlitCommandEncoder() else { continue }
        blit.fill(buffer: readDst, range: 0..<(testSize * MemoryLayout<Float>.size), value: 0)
        blit.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }

    let startMaxRead = getTimeNanos()
    for _ in 0..<iterations {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(readPipeline)
        encoder.setBuffer(sharedBuf, offset: 0, index: 0)
        encoder.setBuffer(readDst, offset: 0, index: 1)
        encoder.setBytes(&sizeVal, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.dispatchThreads(MTLSize(width: testSize, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let endMaxRead = getTimeNanos()
    let maxReadBW = Double(testSize) * Double(iterations) * Double(MemoryLayout<Float>.size) / getElapsedSeconds(start: startMaxRead, end: endMaxRead) / 1e9

    print("Max Read (1024 threads): \(String(format: "%.2f", maxReadBW)) GB/s")

    // ============================================================
    // SUMMARY
    // ============================================================
    print("\n" + String(repeating: "=", count: 70))
    print("HIGH PERFORMANCE BANDWIDTH SUMMARY")
    print(String(repeating: "=", count: 70))

    print("\n| Method                    | Bandwidth   | Notes                    |")
    print("|---------------------------|-------------|--------------------------|")
    print("| Blit Copy Engine          | \(String(format: "%.2f", blitBW).padStart(10)) GB/s | Dedicated hardware     |")
    print("| Shared Mode Write         | \(String(format: "%.2f", sharedWriteBW).padStart(10)) GB/s | CPU-GPU unified       |")
    print("| Async Triple Buffering    | \(String(format: "%.2f", asyncBW).padStart(10)) GB/s | 3x parallel ops      |")
    print("| Burst Write (16/thread)   | \(String(format: "%.2f", burstBW).padStart(10)) GB/s | High thread util      |")
    print("| Max Read (1024 threads)   | \(String(format: "%.2f", maxReadBW).padStart(10)) GB/s | Max threadgroup       |")

    print("\n" + String(repeating: "=", count: 70))
    print("High Performance Memory Bandwidth Test Complete")
    print(String(repeating: "=", count: 70))
}

// String padding helper
extension String {
    func padStart(_ length: Int) -> String {
        if self.count >= length { return self }
        return String(repeating: " ", count: length - self.count) + self
    }
}

// MARK: - Main

print("Apple Metal GPU Benchmark - FP16 Deep Dive")
print("======================================")

guard let device = MTLCreateSystemDefaultDevice() else {
    print("Metal is not supported on this device")
    exit(1)
}

printDeviceInfo(device: device)

guard let queue = device.makeCommandQueue() else {
    print("Failed to create command queue")
    exit(1)
}

let library: MTLLibrary
do {
    library = try device.makeLibrary(source: shaderSource, options: nil)
} catch {
    print("Failed to create shader library: \(error)")
    exit(1)
}

print("Shader compilation: SUCCESS\n")

do { try testMatmulFP16vsFP32(device: device, queue: queue, library: library) } catch { print("Error: \(error)") }
do { try testVectorFP16vsFP32(device: device, queue: queue, library: library) } catch { print("Error: \(error)") }
do { try testConversionPerformance(device: device, queue: queue, library: library) } catch { print("Error: \(error)") }
do { try testReductionPerformance(device: device, queue: queue, library: library) } catch { print("Error: \(error)") }
do { try testDeepGPUResearch(device: device, queue: queue) } catch { print("Error: \(error)") }
do { try testDeepMemoryBandwidth(device: device, queue: queue) } catch { print("Error: \(error)") }
do { try testHighPerformanceBandwidth(device: device, queue: queue) } catch { print("Error: \(error)") }

print("FP16 Deep Dive completed.")
