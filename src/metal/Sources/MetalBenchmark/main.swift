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
// 10b. ADVANCED ATOMIC OPERATIONS
// ============================================================

kernel void atomic_fetch_add_op(device atomic_uint* counter [[buffer(0)]],
                               device uint* out [[buffer(1)]],
                               constant uint& size [[buffer(2)]],
                               uint id [[thread_position_in_grid]]) {
    // atomic_fetch_add returns original value before addition
    uint original = atomic_fetch_add_explicit(&counter[id % 256], 1, memory_order_relaxed);
    out[id] = original;
}

kernel void atomic_min_op(device atomic_uint* counter [[buffer(0)]],
                         device uint* out [[buffer(1)]],
                         constant uint& size [[buffer(2)]],
                         uint id [[thread_position_in_grid]]) {
    // atomic_fetch_min returns original value before min
    uint val = id % 1024;
    uint original = atomic_fetch_min_explicit(&counter[0], val, memory_order_relaxed);
    out[id] = original;
}

kernel void atomic_max_op(device atomic_uint* counter [[buffer(0)]],
                         device uint* out [[buffer(1)]],
                         constant uint& size [[buffer(2)]],
                         uint id [[thread_position_in_grid]]) {
    // atomic_fetch_max returns original value before max
    uint val = id % 1024;
    uint original = atomic_fetch_max_explicit(&counter[0], val, memory_order_relaxed);
    out[id] = original;
}

kernel void atomic_compare_exchange_op(device atomic_uint* counter [[buffer(0)]],
                                     device uint* out [[buffer(1)]],
                                     constant uint& size [[buffer(2)]],
                                     uint id [[thread_position_in_grid]]) {
    // CAS loop - try to exchange values
    uint expected = 0;
    uint desired = id + 1;
    bool exchanged = atomic_compare_exchange_weak_explicit(&counter[0], &expected, desired,
                                                          memory_order_relaxed, memory_order_relaxed);
    out[id] = exchanged ? desired : expected;
}

// NOTE: Metal only supports memory_order_relaxed for device address space atomics.
// Other memory orders (acquire, release, seq_cst) are only available for threadgroup
// address space. This is a key architectural difference from CUDA.

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

// ============================================================
// 13. FP64 NOT SUPPORTED IN METAL ON APPLE M2
// Note: Apple M2 does not support double precision in Metal shaders
// ============================================================

// ============================================================
// 14. VECTORIZATION WIDTH COMPARISON (float2/float4)
// Metal supports float2 and float4, but NOT float8
// ============================================================

kernel void vec_float2_op(device const float* in [[buffer(0)]],
                         device float* out [[buffer(1)]],
                         constant uint& size [[buffer(2)]],
                         uint id [[thread_position_in_grid]]) {
    float2 v = float2(in[id*2], in[id*2+1]);
    v = v * 2.0f + 1.0f;
    out[id*2] = v.x;
    out[id*2+1] = v.y;
}

kernel void vec_float4_op(device const float* in [[buffer(0)]],
                         device float* out [[buffer(1)]],
                         constant uint& size [[buffer(2)]],
                         uint id [[thread_position_in_grid]]) {
    float4 v = float4(in[id*4], in[id*4+1], in[id*4+2], in[id*4+3]);
    v = v * 2.0f + 1.0f;
    out[id*4] = v.x;
    out[id*4+1] = v.y;
    out[id*4+2] = v.z;
    out[id*4+3] = v.w;
}

// ============================================================
// 15. HALF-PRECISION VECTORIZATION (half2/half4)
// Metal supports half2 and half4, but NOT half8
// ============================================================

kernel void vec_half2_op(device const half* in [[buffer(0)]],
                        device half* out [[buffer(1)]],
                        constant uint& size [[buffer(2)]],
                        uint id [[thread_position_in_grid]]) {
    half2 v = half2(in[id*2], in[id*2+1]);
    v = v * 2.0h + 1.0h;
    out[id*2] = v[0];
    out[id*2+1] = v[1];
}

kernel void vec_half4_op(device const half* in [[buffer(0)]],
                        device half* out [[buffer(1)]],
                        constant uint& size [[buffer(2)]],
                        uint id [[thread_position_in_grid]]) {
    half4 v = half4(in[id*4], in[id*4+1], in[id*4+2], in[id*4+3]);
    v = v * 2.0h + 1.0h;
    out[id*4] = v[0]; out[id*4+1] = v[1]; out[id*4+2] = v[2]; out[id*4+3] = v[3];
}

// ============================================================
// 16. MEMORY FENCE IMPACT
// ============================================================

kernel void mem_fence_none(device const float* in [[buffer(0)]],
                          device float* out [[buffer(1)]],
                          constant uint& size [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {
    // No fence - aggressive optimization possible
    float val = in[id];
    for (int i = 0; i < 8; i++) {
        val = val * 2.0f + 1.0f;
    }
    out[id] = val;
}

kernel void mem_fence_device(device const float* in [[buffer(0)]],
                             device float* out [[buffer(1)]],
                             constant uint& size [[buffer(2)]],
                             uint id [[thread_position_in_grid]]) {
    // Device memory fence
    float val = in[id];
    for (int i = 0; i < 8; i++) {
        val = val * 2.0f + 1.0f;
    }
    out[id] = val;
    threadgroup_barrier(mem_flags::mem_none);
}

kernel void mem_fence_threadgroup(device const float* in [[buffer(0)]],
                                  device float* out [[buffer(1)]],
                                  constant uint& size [[buffer(2)]],
                                  uint id [[thread_position_in_grid]]) {
    // Threadgroup memory fence
    float val = in[id];
    for (int i = 0; i < 8; i++) {
        val = val * 2.0f + 1.0f;
    }
    out[id] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// ============================================================
// 17. INDIRECT COMPUTE DISPATCH
// ============================================================

kernel void indirect_count(device const float* in [[buffer(0)]],
                          device atomic_uint* counter [[buffer(1)]],
                          device uint* out [[buffer(2)]],
                          constant uint& size [[buffer(3)]],
                          uint id [[thread_position_in_grid]]) {
    if (in[id] > 0.5f) {
        atomic_fetch_add_explicit(&counter[0], 1, memory_order_relaxed);
    }
}

kernel void indirect_execute(device const float* in [[buffer(0)]],
                            device float* out [[buffer(1)]],
                            constant uint& size [[buffer(2)]],
                            uint id [[thread_position_in_grid]]) {
    out[id] = in[id] * 2.0f + 1.0f;
}

// ============================================================
// 17b. INDIRECT ADDRESSING / SCATTER-GATHER
// Tests random index-based memory access patterns
// ============================================================

kernel void gather_addressing(device const float* data [[buffer(0)]],
                           device const uint* indices [[buffer(1)]],
                           device float* out [[buffer(2)]],
                           constant uint& size [[buffer(3)]],
                           uint id [[thread_position_in_grid]]) {
    // Gather: read from data[indices[id]]
    uint idx = indices[id] % size;
    out[id] = data[idx] * 2.0f;
}

kernel void scatter_addressing(device float* data [[buffer(0)]],
                            device const uint* indices [[buffer(1)]],
                            device const float* values [[buffer(2)]],
                            constant uint& size [[buffer(3)]],
                            uint id [[thread_position_in_grid]]) {
    // Scatter: write to data[indices[id]]
    uint idx = indices[id] % size;
    data[idx] = values[id] * 2.0f;
}

kernel void gather_then_process(device const float* data [[buffer(0)]],
                             device const uint* indices [[buffer(1)]],
                             device float* out [[buffer(2)]],
                             constant uint& size [[buffer(3)]],
                             uint id [[thread_position_in_grid]]) {
    // Gather then do multiple operations
    uint idx = indices[id] % size;
    float val = data[idx];
    val = val * 2.0f + 1.0f;
    val = sqrt(val + 0.001f);
    out[id] = val;
}

// ============================================================
// 18. KERNEL FUSION STUDY (multi-op in single kernel)
// ============================================================

kernel void fused_ops(device const float* a [[buffer(0)]],
                    device const float* b [[buffer(1)]],
                    device float* out [[buffer(2)]],
                    constant uint& size [[buffer(3)]],
                    uint id [[thread_position_in_grid]]) {
    // Fused: mul, add, sqrt, div in single kernel
    float v = a[id] * b[id];
    v = v + 1.0f;
    v = sqrt(v);
    v = v / (b[id] + 0.001f);
    out[id] = v;
}

kernel void separate_ops_a(device const float* a [[buffer(0)]],
                           device const float* b [[buffer(1)]],
                           device float* temp [[buffer(2)]],
                           constant uint& size [[buffer(3)]],
                           uint id [[thread_position_in_grid]]) {
    temp[id] = a[id] * b[id];
}

kernel void separate_ops_b(device const float* temp [[buffer(2)]],
                           device const float* b [[buffer(1)]],
                           device float* out [[buffer(3)]],
                           constant uint& size [[buffer(0)]],
                           uint id [[thread_position_in_grid]]) {
    float v = temp[id] + 1.0f;
    v = sqrt(v);
    out[id] = v / (b[id] + 0.001f);
}

// ============================================================
// 19. TEXTURE VS BUFFER MEMORY ACCESS
// Texture has L1/L2 cache, efficient for 2D spatial locality
// ============================================================

kernel void tex_read_linear(device const float* in [[buffer(0)]],
                          device float* out [[buffer(1)]],
                          constant uint& size [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {
    out[id] = in[id] * 2.0f + 1.0f;
}

kernel void tex_read_2d(uint2 gid [[thread_position_in_grid]],
                       texture2d<float, access::read> tex [[texture(0)]],
                       device float* out [[buffer(0)]],
                       constant uint& width [[buffer(1)]]) {
    float4 val = tex.read(gid);
    out[gid.y * width + gid.x] = val.x * 2.0f + 1.0f;
}

kernel void tex_read_2d_half(uint2 gid [[thread_position_in_grid]],
                            texture2d<half, access::read> tex [[texture(0)]],
                            device half* out [[buffer(0)]],
                            constant uint& width [[buffer(1)]]) {
    half4 val = tex.read(gid);
    out[gid.y * width + gid.x] = val.x * 2.0h + 1.0h;
}

kernel void tex_write_linear(device float* out [[buffer(0)]],
                           constant uint& size [[buffer(1)]],
                           uint id [[thread_position_in_grid]]) {
    out[id] = float(id);
}

kernel void tex_write_2d(uint2 gid [[thread_position_in_grid]],
                        texture2d<float, access::write> tex [[texture(0)]],
                        constant float& val [[buffer(0)]]) {
    tex.write(float4(val, val, val, val), gid);
}

// ============================================================
// 20. TEXTURE CACHING BEHAVIOR
// Sequential vs random access patterns with texture
// ============================================================

kernel void tex_seq_read(uint2 gid [[thread_position_in_grid]],
                        texture2d<float, access::read> tex [[texture(0)]],
                        device float* out [[buffer(0)]],
                        constant uint& width [[buffer(1)]]) {
    uint2 coord = uint2(gid.x % width, gid.y % (width / width));
    float4 val = tex.read(coord);
    out[gid.y * width + gid.x] = val.x;
}

kernel void tex_rand_read(uint2 gid [[thread_position_in_grid]],
                         texture2d<float, access::read> tex [[texture(0)]],
                         device float* out [[buffer(0)]],
                         constant uint& width [[buffer(1)]]) {
    uint2 coord = uint2((gid.x * 7919) % width, (gid.y * 6271) % width);
    float4 val = tex.read(coord);
    out[gid.y * width + gid.x] = val.x;
}

// ============================================================
// 21. PIPELINE BUBBLES AND INSTRUCTION LATENCY
// Measure impact of dependent operations
// ============================================================

kernel void dep_chain_add(device const float* in [[buffer(0)]],
                         device float* out [[buffer(1)]],
                         constant uint& size [[buffer(2)]],
                         uint id [[thread_position_in_grid]]) {
    float v = in[id];
    v = v + 1.0f;
    v = v + 1.0f;
    v = v + 1.0f;
    v = v + 1.0f;
    v = v + 1.0f;
    out[id] = v;
}

kernel void dep_chain_mul(device const float* in [[buffer(0)]],
                         device float* out [[buffer(1)]],
                         constant uint& size [[buffer(2)]],
                         uint id [[thread_position_in_grid]]) {
    float v = in[id];
    v = v * 2.0f;
    v = v * 2.0f;
    v = v * 2.0f;
    v = v * 2.0f;
    v = v * 2.0f;
    out[id] = v;
}

kernel void indep_ops(device const float* in [[buffer(0)]],
                     device float* out [[buffer(1)]],
                     constant uint& size [[buffer(2)]],
                     uint id [[thread_position_in_grid]]) {
    float v = in[id];
    float a = v + 1.0f;
    float b = v * 2.0f;
    float c = v + 2.0f;
    float d = v * 3.0f;
    out[id] = a + b + c + d;
}

// ============================================================
// 22. SIMD GROUP OPERATIONS
// Apple GPU uses 32-thread SIMD groups
// ============================================================

kernel void simd_vote_all(device const float* in [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        constant uint& size [[buffer(2)]],
                        uint id [[thread_position_in_grid]]) {
    bool pred = in[id] > 0.5f;
    bool result = simd_all(pred);
    out[id] = result ? 1.0f : 0.0f;
}

kernel void simd_vote_any(device const float* in [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        constant uint& size [[buffer(2)]],
                        uint id [[thread_position_in_grid]]) {
    bool pred = in[id] > 0.5f;
    bool result = simd_any(pred);
    out[id] = result ? 1.0f : 0.0f;
}

kernel void simd_shuffle(device const float* in [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        constant uint& size [[buffer(2)]],
                        uint id [[thread_position_in_grid]]) {
    uint lid = id & 31;
    float val = in[id];
    // Exchange within SIMD group
    float shuffled = simd_shuffle(val, lid ^ 16);  // swap with opposite lane
    out[id] = shuffled;
}

kernel void simd_prefix_sum(device const float* in [[buffer(0)]],
                           device float* out [[buffer(1)]],
                           constant uint& size [[buffer(2)]],
                           uint id [[thread_position_in_grid]]) {
    float sum = in[id];
    sum += simd_shuffle_xor(sum, 16);
    sum += simd_shuffle_xor(sum, 8);
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 1);
    out[id] = sum;
}

// ============================================================
// 23. TREE-BASED PARALLEL REDUCTION
// vs sequential reduction
// ============================================================

kernel void reduce_sequential(device const float* in [[buffer(0)]],
                             device float* out [[buffer(1)]],
                             constant uint& size [[buffer(2)]],
                             uint id [[thread_position_in_grid]]) {
    if (id == 0) {
        float sum = 0.0f;
        for (uint i = 0; i < size; i++) {
            sum += in[i];
        }
        out[0] = sum;
    }
}

kernel void reduce_shared_basic(device const float* in [[buffer(0)]],
                               device float* out [[buffer(1)]],
                               threadgroup float* shared [[threadgroup(0)]],
                               constant uint& size [[buffer(2)]],
                               uint id [[thread_position_in_grid]],
                               uint lid [[thread_position_in_threadgroup]]) {
    // Load into shared memory
    shared[lid] = in[id];
    threadgroup_barrier(mem_flags::mem_none);

    // Sequential within threadgroup first
    for (uint s = 1; s < 256; s *= 2) {
        if ((lid % (s * 2)) == 0) {
            shared[lid] += shared[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_none);
    }

    if (lid == 0) {
        out[0] = shared[0];
    }
}

kernel void reduce_warp_level(device const float* in [[buffer(0)]],
                             device float* out [[buffer(1)]],
                             threadgroup float* shared [[threadgroup(0)]],
                             constant uint& size [[buffer(2)]],
                             uint id [[thread_position_in_grid]],
                             uint lid [[thread_position_in_threadgroup]]) {
    // Load into shared memory
    shared[lid] = in[id];
    threadgroup_barrier(mem_flags::mem_none);

    // Parallel reduction in shared memory
    for (uint s = 128; s > 32; s >>= 1) {
        if (lid < s) {
            shared[lid] += shared[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_none);
    }

    // Warp-level reduction (no barrier needed for lid < 32)
    if (lid < 32) {
        shared[lid] += shared[lid + 16];
        shared[lid] += shared[lid + 8];
        shared[lid] += shared[lid + 4];
        shared[lid] += shared[lid + 2];
        shared[lid] += shared[lid + 1];
    }

    if (lid == 0) {
        out[0] = shared[0];
    }
}

// ============================================================
// 23b. THREADGROUP MEMORY PERFORMANCE
// Tests shared memory bandwidth and access patterns
// 32KB shared memory / 256 threads = 128 floats per thread
// ============================================================

kernel void shared_copy_seq(device const float* in [[buffer(0)]],
                         device float* out [[buffer(1)]],
                         threadgroup float* shared [[threadgroup(0)]],
                         constant uint& size [[buffer(2)]],
                         uint id [[thread_position_in_grid]]) {
    // Sequential shared memory copy - each thread works on its own segment
    // 256 threads x 128 floats = 32KB (full shared memory)
    uint tid = id % 256;
    uint segmentSize = 128;  // floats per thread
    uint offset = tid * segmentSize;

    // Load into shared memory - sequential access per thread
    for (uint i = 0; i < segmentSize; i++) {
        shared[offset + i] = in[offset + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Store from shared memory - sequential access
    for (uint i = 0; i < segmentSize; i++) {
        out[offset + i] = shared[offset + i];
    }
}

kernel void shared_copy_strided(device const float* in [[buffer(0)]],
                              device float* out [[buffer(1)]],
                              threadgroup float* shared [[threadgroup(0)]],
                              constant uint& size [[buffer(2)]],
                              uint id [[thread_position_in_grid]]) {
    // Strided shared memory copy - causes bank conflicts
    // All threads access the same shared memory location simultaneously
    uint tid = id % 256;
    uint segmentSize = 128;
    uint stride = 32;  // Bank conflict stride
    uint offset = tid * segmentSize;

    // Load with stride into shared memory - causes bank conflicts
    for (uint i = 0; i < segmentSize; i++) {
        shared[(offset + i * stride) % 8192] = in[offset + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Store with stride from shared memory
    for (uint i = 0; i < segmentSize; i++) {
        out[offset + i] = shared[(offset + i * stride) % 8192];
    }
}

kernel void shared_fill_and_sum(threadgroup float* shared [[threadgroup(0)]],
                              device float* out [[buffer(0)]],
                              constant uint& size [[buffer(1)]],
                              uint id [[thread_position_in_grid]]) {
    // Fill shared memory then sum all values
    uint tid = id % 256;
    uint segmentSize = 128;
    uint offset = tid * segmentSize;

    // Fill shared memory
    for (uint i = 0; i < segmentSize; i++) {
        shared[offset + i] = float(offset + i);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Sum all values from shared memory
    float sum = 0.0f;
    for (uint i = 0; i < 8192; i++) {
        sum += shared[i];
    }
    out[id] = sum;
}

// ============================================================
// 25. HISTOGRAM COMPUTATION
// Tests atomic operations, memory access patterns, and parallel bins
// ============================================================

kernel void histogram_naive(device const float* data [[buffer(0)]],
                          device atomic_uint* hist [[buffer(1)]],
                          constant uint& size [[buffer(2)]],
                          constant uint& bins [[buffer(3)]],
                          uint id [[thread_position_in_grid]]) {
    // Naive histogram - each thread computes one bin, causes atomic contention
    uint bin = uint(data[id] * float(bins)) % bins;
    atomic_fetch_add_explicit(&hist[bin], 1, memory_order_relaxed);
}

kernel void histogram_local(device const float* data [[buffer(0)]],
                           device atomic_uint* hist [[buffer(1)]],
                           constant uint& size [[buffer(2)]],
                           constant uint& bins [[buffer(3)]],
                           uint id [[thread_position_in_grid]]) {
    // Local histogram with shared memory - reduce atomic contention
    // Each threadgroup has private counts, merged at end
    threadgroup uint local_hist[256];  // Local per threadgroup
    uint lid = id % 256;
    uint gid = id / 256;

    // Initialize local histogram
    if (lid < bins) {
        local_hist[lid] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread increments its bin in local histogram
    uint bin = uint(data[id] * float(bins)) % bins;
    if (bin < bins) {
        local_hist[bin]++;  // Regular increment, not atomic
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Merge local into global (one thread per bin)
    if (lid < bins && gid == 0) {
        atomic_fetch_add_explicit(&hist[lid], local_hist[lid], memory_order_relaxed);
    }
}

kernel void histogram_strided(device const float* data [[buffer(0)]],
                             device atomic_uint* hist [[buffer(1)]],
                             constant uint& size [[buffer(2)]],
                             constant uint& bins [[buffer(3)]],
                             uint id [[thread_position_in_grid]]) {
    // Strided histogram - each thread processes stride elements
    uint stride = 256;
    uint start = id * stride;
    uint end = min(start + stride, size);

    for (uint i = start; i < end; i++) {
        uint bin = uint(data[i] * float(bins)) % bins;
        atomic_fetch_add_explicit(&hist[bin], 1, memory_order_relaxed);
    }
}

kernel void histogram_vectorized(device const float4* data [[buffer(0)]],
                                 device atomic_uint* hist [[buffer(1)]],
                                 constant uint& size [[buffer(2)]],
                                 constant uint& bins [[buffer(3)]],
                                 uint id [[thread_position_in_grid]]) {
    // Vectorized histogram - each thread processes 4 elements
    float4 val = data[id];
    uint bin0 = uint(val.x * float(bins)) % bins;
    uint bin1 = uint(val.y * float(bins)) % bins;
    uint bin2 = uint(val.z * float(bins)) % bins;
    uint bin3 = uint(val.w * float(bins)) % bins;
    atomic_fetch_add_explicit(&hist[bin0], 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&hist[bin1], 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&hist[bin2], 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&hist[bin3], 1, memory_order_relaxed);
}

// ============================================================
// 26. MATRIX TRANSPOSE
// Tests shared memory bank conflict patterns and memory coalescing
// ============================================================

kernel void transpose_naive(device const float* in [[buffer(0)]],
                         device float* out [[buffer(1)]],
                         constant uint& width [[buffer(2)]],
                         constant uint& height [[buffer(3)]],
                         uint2 gid [[thread_position_in_grid]]) {
    // Naive transpose - each thread reads row, writes column
    // Causes non-coalesced writes (bank conflicts)
    uint row = gid.y;
    uint col = gid.x;
    float val = in[row * width + col];
    out[col * height + row] = val;
}

kernel void transpose_shared(device const float* in [[buffer(0)]],
                            device float* out [[buffer(1)]],
                            constant uint& width [[buffer(2)]],
                            constant uint& height [[buffer(3)]],
                            uint2 gid [[thread_position_in_grid]]) {
    // Shared memory transpose - each thread handles a 16x16 tile
    // Calculate tile position from global ID
    uint blockSize = 16;
    uint tileRow = (gid.y / blockSize) * blockSize;
    uint tileCol = (gid.x / blockSize) * blockSize;
    uint localRow = gid.y % blockSize;
    uint localCol = gid.x % blockSize;

    // Each thread loads its element into shared
    threadgroup float tile[256];  // 16x16 tile

    uint inRow = tileRow + localRow;
    uint inCol = tileCol + localCol;
    uint tileIdx = localRow * blockSize + localCol;

    if (inRow < height && inCol < width) {
        tile[tileIdx] = in[inRow * width + inCol];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write transpose: swap row and column within tile
    uint outRow = tileCol + localRow;
    uint outCol = tileRow + localCol;
    uint outIdx = outRow * height + outCol;
    uint readIdx = localCol * blockSize + localRow;

    if (outRow < width && outCol < height) {
        out[outIdx] = tile[readIdx];
    }
}

// ============================================================
// 27. STENCIL COMPUTATION
// Tests shared memory for halo cells and multi-pass stencil operations
// ============================================================

kernel void stencil_naive(device const float* in [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        constant uint& width [[buffer(2)]],
                        constant uint& height [[buffer(3)]],
                        uint2 gid [[thread_position_in_grid]]) {
    // Naive 5-point stencil (up, down, left, right, center)
    // Each thread loads its point and neighbors from global memory
    uint row = gid.y;
    uint col = gid.x;
    uint idx = row * width + col;

    if (row > 0 && row < height - 1 && col > 0 && col < width - 1) {
        float center = in[idx];
        float up = in[idx - width];
        float down = in[idx + width];
        float left = in[idx - 1];
        float right = in[idx + 1];
        out[idx] = (center + up + down + left + right) * 0.2f;
    } else {
        out[idx] = in[idx];  // Boundary copy
    }
}

kernel void stencil_shared(device const float* in [[buffer(0)]],
                         device float* out [[buffer(1)]],
                         constant uint& width [[buffer(2)]],
                         constant uint& height [[buffer(3)]],
                         uint2 gid [[thread_position_in_grid]]) {
    // Shared memory stencil - uses threadgroups for halo cells
    // Tile size 16x16 with 1-cell halo = 18x18 shared memory
    uint tileSize = 16;
    uint halo = 1;

    // Calculate position
    uint tileRow = (gid.y / tileSize) * tileSize;
    uint tileCol = (gid.x / tileSize) * tileSize;
    uint localRow = gid.y % tileSize;
    uint localCol = gid.x % tileSize;
    uint sharedIdx = (localRow + halo) * (tileSize + 2 * halo) + (localCol + halo);

    // Shared memory tile with halo
    threadgroup float tile[324];  // 18x18 = 324 elements

    // Load tile including halo
    uint loadRow = tileRow + localRow;
    uint loadCol = tileCol + localCol;
    if (loadRow < height && loadCol < width) {
        tile[sharedIdx] = in[loadRow * width + loadCol];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute stencil
    if (localRow > 0 && localRow < tileSize - 1 &&
        localCol > 0 && localCol < tileSize - 1 &&
        gid.y > 0 && gid.y < height - 1 &&
        gid.x > 0 && gid.x < width - 1) {
        float center = tile[sharedIdx];
        float up = tile[sharedIdx - (tileSize + 2 * halo)];
        float down = tile[sharedIdx + (tileSize + 2 * halo)];
        float left = tile[sharedIdx - 1];
        float right = tile[sharedIdx + 1];
        out[gid.y * width + gid.x] = (center + up + down + left + right) * 0.2f;
    }
}

kernel void stencil_iterated(device const float* in [[buffer(0)]],
                           device float* out [[buffer(1)]],
                           constant uint& width [[buffer(2)]],
                           constant uint& height [[buffer(3)]],
                           uint2 gid [[thread_position_in_grid]]) {
    // Iterated stencil - 5 iterations of 5-point stencil
    uint row = gid.y;
    uint col = gid.x;
    uint idx = row * width + col;

    if (row > 0 && row < height - 1 && col > 0 && col < width - 1) {
        float val = in[idx];
        for (uint iter = 0; iter < 5; iter++) {
            float up = in[(row - 1) * width + col];
            float down = in[(row + 1) * width + col];
            float left = in[row * width + col - 1];
            float right = in[row * width + col + 1];
            val = (up + down + left + right + val) * 0.2f;
        }
        out[idx] = val;
    } else {
        out[idx] = in[idx];
    }
}

// ============================================================
// 21. STREAM COMPACTION
// Tests parallel filter/compact operations with atomics
// ============================================================

kernel void compact_naive(device const float* in [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        device atomic_uint* count [[buffer(2)]],
                        constant uint& size [[buffer(3)]],
                        uint id [[thread_position_in_grid]]) {
    // Naive compaction - each thread checks predicate and writes to global output
    float val = in[id];
    if (val > 0.5f) {  // Predicate: keep values > 0.5
        uint writeIdx = atomic_fetch_add_explicit(count, 1, memory_order_relaxed);
        out[writeIdx] = val;
    }
}

kernel void compact_tiled(device const float* in [[buffer(0)]],
                       device float* out [[buffer(1)]],
                       device atomic_uint* count [[buffer(2)]],
                       constant uint& size [[buffer(3)]],
                       uint id [[thread_position_in_grid]]) {
    // Tiled compaction - accumulate locally using shared memory, then merge
    uint tileSize = 256;
    uint tileId = id / tileSize;
    uint localId = id % tileSize;

    // Use shared memory for local accumulation (no atomics needed)
    threadgroup uint localCount;
    threadgroup float localOut[128];  // Max 128 elements per tile

    if (localId == 0) {
        localCount = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread checks predicate and writes to local storage
    float val = in[id];
    uint localWriteIdx = 0;
    if (val > 0.5f) {  // Predicate: keep values > 0.5
        localWriteIdx = atomic_fetch_add_explicit(count, 1, memory_order_relaxed);
        if (localWriteIdx < size) {
            out[localWriteIdx] = val;
        }
    }
    // Note: simplified version - just uses global atomics directly
    // Real tiled compaction would use shared memory to reduce atomic contention
}

// ============================================================
// 24. DUAL-BUFFER PIPELINING
// Memory access and computation overlap
// ============================================================

kernel void dual_buffer_compute_a(device const float* inA [[buffer(0)]],
                                  device const float* inB [[buffer(1)]],
                                  device float* out [[buffer(2)]],
                                  device float* temp [[buffer(3)]],
                                  constant uint& size [[buffer(4)]],
                                  uint id [[thread_position_in_grid]]) {
    // Phase 1: Load data
    temp[id] = inA[id];

    // Phase 2: Compute
    float sum = 0.0f;
    for (uint i = 0; i < 16; i++) {
        sum += temp[(id + i) % size] * inB[(id + i) % size];
    }
    out[id] = sum;
}

kernel void dual_buffer_compute_b(device const float* temp [[buffer(3)]],
                                  device const float* inB [[buffer(1)]],
                                  device float* out [[buffer(2)]],
                                  constant uint& size [[buffer(4)]],
                                  uint id [[thread_position_in_grid]]) {
    float sum = 0.0f;
    for (uint i = 0; i < 16; i++) {
        sum += temp[(id + i) % size] * inB[(id + i) % size];
    }
    out[id] = sum;
}

// Single buffer baseline
kernel void single_buffer_compute(device const float* in [[buffer(0)]],
                                 device float* out [[buffer(1)]],
                                 constant uint& size [[buffer(2)]],
                                 uint id [[thread_position_in_grid]]) {
    float val = in[id];
    float sum = 0.0f;
    for (uint i = 0; i < 16; i++) {
        sum += val * in[(id + i) % size];
    }
    out[id] = sum;
}

// ============================================================
// Radix Sort Kernels
// ============================================================

// Radix Sort: Histogram phase (count occurrences of each digit)
kernel void radix_histogram(device const uint* in [[buffer(0)]],
                            device atomic_uint* histogram [[buffer(1)]],
                            constant uint& size [[buffer(2)]],
                            constant uint& pass [[buffer(3)]],
                            uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    uint val = in[id];
    uint digit = (val >> (pass * 4)) & 0xF;  // 4-bit digit (0-15)
    atomic_fetch_add_explicit(&histogram[digit], 1, memory_order_relaxed);
}

// Radix Sort: Prefix sum (exclusive scan on histogram)
kernel void radix_prefix_sum(device const uint* histogram [[buffer(0)]],
                            device uint* prefix [[buffer(1)]]) {
    // 16 elements (radix 16 = 4 bits per pass)
    uint sum = 0;
    for (uint i = 0; i < 16; i++) {
        prefix[i] = sum;
        sum += histogram[i];
    }
}

// Radix Sort: Reorder phase (scatter to buckets using prefix sums)
kernel void radix_reorder(device const uint* in [[buffer(0)]],
                         device uint* out [[buffer(1)]],
                         device const uint* histogram [[buffer(2)]],
                         device const uint* prefix [[buffer(3)]],
                         device atomic_uint* counters [[buffer(4)]],
                         constant uint& size [[buffer(5)]],
                         constant uint& pass [[buffer(6)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    uint val = in[id];
    uint digit = (val >> (pass * 4)) & 0xF;

    uint bucketStart = prefix[digit];
    uint offset = atomic_fetch_add_explicit(&counters[digit], 1, memory_order_relaxed);
    out[bucketStart + offset] = val;
}

// ============================================================
// Sparse Matrix Vector Multiply (SpMV) Kernels
// CSR format: values, column_indices, row_offsets
// ============================================================

// SpMV: Naive CSR - each thread processes one row
kernel void spmv_csr_naive(device const float* values [[buffer(0)]],
                          device const uint* column_indices [[buffer(1)]],
                          device const uint* row_offsets [[buffer(2)]],
                          device const float* x [[buffer(3)]],
                          device float* y [[buffer(4)]],
                          constant uint& num_rows [[buffer(5)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= num_rows) return;

    float sum = 0.0f;
    uint row_start = row_offsets[id];
    uint row_end = row_offsets[id + 1];

    for (uint i = row_start; i < row_end; i++) {
        uint col = column_indices[i];
        sum += values[i] * x[col];
    }
    y[id] = sum;
}

// SpMV: Vectorized - process multiple elements per thread
kernel void spmv_csr_vectorized(device const float4* values [[buffer(0)]],
                                device const uint4* column_indices [[buffer(1)]],
                                device const uint* row_offsets [[buffer(2)]],
                                device const float4* x [[buffer(3)]],
                                device float* y [[buffer(4)]],
                                constant uint& num_rows [[buffer(5)]],
                                uint id [[thread_position_in_grid]]) {
    if (id >= num_rows) return;

    float4 sum = 0.0f;
    uint row_start = row_offsets[id];
    uint row_end = row_offsets[id + 1];

    for (uint i = row_start; i < row_end; i += 4) {
        uint4 cols = column_indices[i / 4];
        float4 vals = values[i / 4];
        sum += vals * x[cols.x];
        if (i + 1 < row_end) sum += vals * x[cols.y];
        if (i + 2 < row_end) sum += vals * x[cols.z];
        if (i + 3 < row_end) sum += vals * x[cols.w];
    }
    y[id] = sum.x + sum.y + sum.z + sum.w;
}

// SpMV: ELLPACK format - fixed width per row
kernel void spmv_ellpack(device const float* values [[buffer(0)]],
                        device const uint* column_indices [[buffer(1)]],
                        device const float* x [[buffer(2)]],
                        device float* y [[buffer(3)]],
                        constant uint& num_rows [[buffer(4)]],
                        constant uint& max_nnz_per_row [[buffer(5)]],
                        uint id [[thread_position_in_grid]]) {
    if (id >= num_rows) return;

    float sum = 0.0f;
    uint base = id * max_nnz_per_row;

    for (uint i = 0; i < max_nnz_per_row; i++) {
        uint col = column_indices[base + i];
        if (col != 0xFFFFFFFF) {  // invalid column marker
            sum += values[base + i] * x[col];
        }
    }
    y[id] = sum;
}

// ============================================================
// Tridiagonal Matrix Solver (Thomas Algorithm)
// Solves Ax = d where A is tridiagonal
// a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = d[i]
// ============================================================

// Thomas Algorithm: Sequential forward-backward sweep
kernel void tridiagonal_thomas(device const float* a [[buffer(0)]],  // sub-diagonal
                               device const float* b [[buffer(1)]],  // main diagonal
                               device const float* c [[buffer(2)]],  // super-diagonal
                               device const float* d [[buffer(3)]],  // RHS
                               device float* x [[buffer(4)]],
                               device float* cp [[buffer(5)]],  // temporary: c'
                               device float* dp [[buffer(6)]],  // temporary: d'
                               constant uint& size [[buffer(7)]],
                               uint id [[thread_position_in_grid]]) {
    if (id >= size) return;

    // Forward sweep (only thread 0 does this in true sequential)
    if (id == 0) {
        // c'[0] = c[0] / b[0]
        // d'[0] = d[0] / b[0]
        cp[0] = c[0] / b[0];
        dp[0] = d[0] / b[0];
    }

    // Process internal elements
    if (id > 0 && id < size - 1) {
        float denom = b[id] - a[id] * cp[id - 1];
        cp[id] = c[id] / denom;
        dp[id] = (d[id] - a[id] * dp[id - 1]) / denom;
    }

    // Backward substitution (only last thread does this)
    if (id == size - 1) {
        x[size - 1] = (d[size - 1] - a[size - 1] * dp[size - 2]) /
                      (b[size - 1] - a[size - 1] * cp[size - 2]);
    }

    // Wait for forward sweep to complete
    threadgroup_barrier(mem_flags::mem_none);

    // Back substitution for internal elements
    if (id < size - 1) {
        x[id] = dp[id] - cp[id] * x[id + 1];
    }
}

// Parallel Tridiagonal: Cyclic Reduction (CR)
// Splits system into even and odd indexed elements for parallelism
kernel void tridiagonal_parallel_cr(device const float* a [[buffer(0)]],
                                    device const float* b [[buffer(1)]],
                                    device const float* c [[buffer(2)]],
                                    device const float* d [[buffer(3)]],
                                    device float* x [[buffer(4)]],
                                    device float* temp_a [[buffer(5)]],
                                    device float* temp_b [[buffer(6)]],
                                    device float* temp_c [[buffer(7)]],
                                    device float* temp_d [[buffer(8)]],
                                    constant uint& size [[buffer(9)]],
                                    uint id [[thread_position_in_grid]]) {
    if (id >= size) return;

    // Odd indices only in first pass
    if (id % 2 == 1 && id < size) {
        uint nhalf = size / 2;
        uint i_odd = id;
        uint i_even = id - 1;

        // Thomas step for pair (even, odd)
        float denom = b[i_odd] - a[i_odd] * c[i_even];
        temp_d[i_odd] = d[i_odd] - a[i_odd] * d[i_even];
        temp_b[i_odd] = b[i_odd];
        temp_a[i_odd] = 0.0f;
        temp_c[i_odd] = c[i_odd];
    }

    // Even indices (except first)
    if (id % 2 == 0 && id > 0 && id < size - 1) {
        uint nhalf = size / 2;
        uint i_even = id;
        uint i_odd = id + 1;

        float denom = b[i_even] - c[i_even - 1] * a[i_even] / b[i_even - 1];
        temp_d[i_even] = d[i_even] - c[i_even - 1] * d[i_even - 1] / b[i_even - 1];
        temp_b[i_even] = denom;
    }

    x[id] = d[id] / b[id];  // Simplified: direct solve
}

// ============================================================
// Prefix Sum / Scan Algorithms
// ============================================================

// Hillis-Steele Scan (Work-efficient, O(n log n) steps)
// Each step doubles the stride
kernel void scan_hillis_steele(device const float* in [[buffer(0)]],
                              device float* out [[buffer(1)]],
                              device float* temp [[buffer(2)]],
                              constant uint& size [[buffer(3)]],
                              uint id [[thread_position_in_grid]]) {
    if (id >= size) return;

    // Load into shared storage
    float val = in[id];

    // Hillis-Steele scan with log n steps
    for (uint stride = 1; stride < size; stride *= 2) {
        temp[id] = val;
        threadgroup_barrier(mem_flags::mem_none);

        if (id >= stride) {
            val += temp[id - stride];
        }
        threadgroup_barrier(mem_flags::mem_none);
    }
    out[id] = val;
}

// Kogge-Stone Scan (O(log n) latency, O(n log n) work)
// Each thread adds value from stride distance
kernel void scan_kogge_stone(device const float* in [[buffer(0)]],
                            device float* out [[buffer(1)]],
                            device float* temp [[buffer(2)]],
                            constant uint& size [[buffer(3)]],
                            uint id [[thread_position_in_grid]]) {
    if (id >= size) return;

    float val = in[id];

    // Kogge-Stone: log n steps, each thread does one add per step
    for (uint stride = 1; stride < size; stride *= 2) {
        temp[id] = val;
        threadgroup_barrier(mem_flags::mem_none);

        uint srcIdx = id + stride;
        if (srcIdx < size) {
            val += temp[srcIdx];
        }
        threadgroup_barrier(mem_flags::mem_none);
    }
    out[id] = val;
}

// Brent-Kung Scan (O(log n) latency, O(n) work optimal)
// Uses two phases: reduce + downsweep
kernel void scan_brent_kung(device const float* in [[buffer(0)]],
                           device float* out [[buffer(1)]],
                           device float* temp [[buffer(2)]],
                           constant uint& size [[buffer(3)]],
                           uint id [[thread_position_in_grid]]) {
    if (id >= size) return;

    // Phase 1: Build reduction tree (bottom-up)
    float val = in[id];
    uint stride = 1;
    while (stride < size) {
        if (id % (stride * 2) == 0 && id + stride < size) {
            val += in[id + stride];
        }
        stride *= 2;
        threadgroup_barrier(mem_flags::mem_none);
    }

    // Store partial result
    temp[id] = val;
    threadgroup_barrier(mem_flags::mem_none);

    // Phase 2: Downsweep (top-down)
    uint fullSize = size;
    while (stride > 1) {
        stride /= 2;
        if (id % (stride * 2) == 0) {
            uint rightIdx = id + stride;
            if (rightIdx < fullSize) {
                float rightVal = temp[rightIdx];
                temp[id] = val;
                val += rightVal;
            }
        }
        threadgroup_barrier(mem_flags::mem_none);
    }

    out[id] = val;
}

// Naive Sequential Scan (baseline)
kernel void scan_naive(device const float* in [[buffer(0)]],
                      device float* out [[buffer(1)]],
                      constant uint& size [[buffer(2)]],
                      uint id [[thread_position_in_grid]]) {
    if (id >= size) return;

    float sum = 0.0f;
    for (uint i = 0; i <= id; i++) {
        sum += in[i];
    }
    out[id] = sum;
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

    // Advanced Atomic Operations
    guard let atomicFetchAddFunc = deepLibrary.makeFunction(name: "atomic_fetch_add_op"),
          let atomicMinFunc = deepLibrary.makeFunction(name: "atomic_min_op"),
          let atomicMaxFunc = deepLibrary.makeFunction(name: "atomic_max_op"),
          let atomicCASFunc = deepLibrary.makeFunction(name: "atomic_compare_exchange_op"),
          let atomicFetchAddPipeline = try? device.makeComputePipelineState(function: atomicFetchAddFunc),
          let atomicMinPipeline = try? device.makeComputePipelineState(function: atomicMinFunc),
          let atomicMaxPipeline = try? device.makeComputePipelineState(function: atomicMaxFunc),
          let atomicCASPipeline = try? device.makeComputePipelineState(function: atomicCASFunc) else {
        print("Failed to create advanced atomic pipelines")
        return
    }

    let atomicAdvSize = 256 * 1024
    guard let atomicAdvBuffer = device.makeBuffer(length: 256 * MemoryLayout<UInt32>.size, options: .storageModeShared),
          let atomicOutBuffer = device.makeBuffer(length: atomicAdvSize * MemoryLayout<UInt32>.size, options: .storageModeShared) else {
        return
    }

    // Initialize counter
    let counterPtr = atomicAdvBuffer.contents().assumingMemoryBound(to: UInt32.self)
    counterPtr[0] = 0

    // Atomic Fetch Add
    var sz = UInt32(atomicAdvSize)
    let startFetchAdd = getTimeNanos()
    for _ in 0..<iterations {
        counterPtr[0] = 0
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(atomicFetchAddPipeline)
        encoder.setBuffer(atomicAdvBuffer, offset: 0, index: 0)
        encoder.setBuffer(atomicOutBuffer, offset: 0, index: 1)
        encoder.setBytes(&sz, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.dispatchThreads(MTLSize(width: atomicAdvSize, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let endFetchAdd = getTimeNanos()
    let gopsFetchAdd = Double(atomicAdvSize) * Double(iterations) / getElapsedSeconds(start: startFetchAdd, end: endFetchAdd) / 1e9

    // Atomic Min
    let startMin = getTimeNanos()
    for _ in 0..<iterations {
        counterPtr[0] = 10000
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(atomicMinPipeline)
        encoder.setBuffer(atomicAdvBuffer, offset: 0, index: 0)
        encoder.setBuffer(atomicOutBuffer, offset: 0, index: 1)
        encoder.setBytes(&sz, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.dispatchThreads(MTLSize(width: atomicAdvSize, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let endMin = getTimeNanos()
    let gopsMin = Double(atomicAdvSize) * Double(iterations) / getElapsedSeconds(start: startMin, end: endMin) / 1e9

    // Atomic Max
    let startMax = getTimeNanos()
    for _ in 0..<iterations {
        counterPtr[0] = 0
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(atomicMaxPipeline)
        encoder.setBuffer(atomicAdvBuffer, offset: 0, index: 0)
        encoder.setBuffer(atomicOutBuffer, offset: 0, index: 1)
        encoder.setBytes(&sz, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.dispatchThreads(MTLSize(width: atomicAdvSize, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let endMax = getTimeNanos()
    let gopsMax = Double(atomicAdvSize) * Double(iterations) / getElapsedSeconds(start: startMax, end: endMax) / 1e9

    // Atomic Compare-And-Swap
    let startCAS = getTimeNanos()
    for _ in 0..<iterations {
        counterPtr[0] = 0
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(atomicCASPipeline)
        encoder.setBuffer(atomicAdvBuffer, offset: 0, index: 0)
        encoder.setBuffer(atomicOutBuffer, offset: 0, index: 1)
        encoder.setBytes(&sz, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.dispatchThreads(MTLSize(width: atomicAdvSize, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let endCAS = getTimeNanos()
    let gopsCAS = Double(atomicAdvSize) * Double(iterations) / getElapsedSeconds(start: startCAS, end: endCAS) / 1e9

    print("\n--- Advanced Atomic Operations ---")
    print("Fetch Add: \(String(format: "%.3f", gopsFetchAdd)) GOPS")
    print("Fetch Min: \(String(format: "%.3f", gopsMin)) GOPS")
    print("Fetch Max: \(String(format: "%.3f", gopsMax)) GOPS")
    print("Compare-And-Swap: \(String(format: "%.3f", gopsCAS)) GOPS")
    print("Note: Metal only supports memory_order_relaxed for device atomics")

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

    // 9. FP64 DOUBLE PRECISION TEST
    print("\n--- 9. FP64 Double Precision Performance ---")

    if let fp64MulFunc = deepLibrary.makeFunction(name: "fp64_vec_mul"),
       let fp64AddFunc = deepLibrary.makeFunction(name: "fp64_vec_add"),
       let fp64MulPipeline = try? device.makeComputePipelineState(function: fp64MulFunc),
       let fp64AddPipeline = try? device.makeComputePipelineState(function: fp64AddFunc) {
        let fp64Size = 1024 * 1024
        let fp64Iter = 50
        guard let fp64In1 = device.makeBuffer(length: fp64Size * MemoryLayout<Double>.size, options: .storageModeShared),
              let fp64In2 = device.makeBuffer(length: fp64Size * MemoryLayout<Double>.size, options: .storageModeShared),
              let fp64Out = device.makeBuffer(length: fp64Size * MemoryLayout<Double>.size, options: .storageModeShared) else {
            print("Failed to create FP64 buffers")
            return
        }

        let fp64In1Ptr = fp64In1.contents().assumingMemoryBound(to: Double.self)
        let fp64In2Ptr = fp64In2.contents().assumingMemoryBound(to: Double.self)
        for i in 0..<fp64Size { fp64In1Ptr[i] = Double(i % 256) / 256.0; fp64In2Ptr[i] = 0.5 }

        var fp64Sz = UInt32(fp64Size)

        let startMul = getTimeNanos()
        for _ in 0..<fp64Iter {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(fp64MulPipeline)
            encoder.setBuffer(fp64In1, offset: 0, index: 0)
            encoder.setBuffer(fp64In2, offset: 0, index: 1)
            encoder.setBuffer(fp64Out, offset: 0, index: 2)
            encoder.setBytes(&fp64Sz, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.dispatchThreads(MTLSize(width: fp64Size, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        let endMul = getTimeNanos()
        let gopsFp64Mul = Double(fp64Size) * Double(fp64Iter) / getElapsedSeconds(start: startMul, end: endMul) / 1e9

        let startAdd = getTimeNanos()
        for _ in 0..<fp64Iter {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(fp64AddPipeline)
            encoder.setBuffer(fp64In1, offset: 0, index: 0)
            encoder.setBuffer(fp64In2, offset: 0, index: 1)
            encoder.setBuffer(fp64Out, offset: 0, index: 2)
            encoder.setBytes(&fp64Sz, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.dispatchThreads(MTLSize(width: fp64Size, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        let endAdd = getTimeNanos()
        let gopsFp64Add = Double(fp64Size) * Double(fp64Iter) / getElapsedSeconds(start: startAdd, end: endAdd) / 1e9

        print("FP64 Vec Multiply: \(String(format: "%.2f", gopsFp64Mul)) GOPS")
        print("FP64 Vec Add:      \(String(format: "%.2f", gopsFp64Add)) GOPS")
    } else {
        print("FP64 not supported or failed to compile")
    }

    // 10. VECTORIZATION WIDTH COMPARISON
    print("\n--- 10. Vectorization Width Comparison (Float) ---")

    guard let float2Func = deepLibrary.makeFunction(name: "vec_float2_op"),
          let float4Func = deepLibrary.makeFunction(name: "vec_float4_op"),
          let float2Pipeline = try? device.makeComputePipelineState(function: float2Func),
          let float4Pipeline = try? device.makeComputePipelineState(function: float4Func) else {
        print("Failed to create vectorization pipelines")
        return
    }

    let vecSize = 8 * 1024 * 1024
    let vecIter = 100
    guard let vecIn = device.makeBuffer(length: vecSize * MemoryLayout<Float>.size, options: .storageModeShared),
          let vecOut = device.makeBuffer(length: vecSize * MemoryLayout<Float>.size, options: .storageModeShared) else {
        return
    }
    var vecSz = UInt32(vecSize)

    // Float2
    let startF2 = getTimeNanos()
    for _ in 0..<vecIter {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(float2Pipeline)
        encoder.setBuffer(vecIn, offset: 0, index: 0)
        encoder.setBuffer(vecOut, offset: 0, index: 1)
        encoder.setBytes(&vecSz, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.dispatchThreads(MTLSize(width: vecSize/2, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let endF2 = getTimeNanos()
    let gopsF2 = Double(vecSize/2) * Double(vecIter) / getElapsedSeconds(start: startF2, end: endF2) / 1e9

    // Float4
    let startF4 = getTimeNanos()
    for _ in 0..<vecIter {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(float4Pipeline)
        encoder.setBuffer(vecIn, offset: 0, index: 0)
        encoder.setBuffer(vecOut, offset: 0, index: 1)
        encoder.setBytes(&vecSz, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.dispatchThreads(MTLSize(width: vecSize/4, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let endF4 = getTimeNanos()
    let gopsF4 = Double(vecSize/4) * Double(vecIter) / getElapsedSeconds(start: startF4, end: endF4) / 1e9

    print("Float2 (width=2):  \(String(format: "%.2f", gopsF2)) GOPS")
    print("Float4 (width=4):  \(String(format: "%.2f", gopsF4)) GOPS")
    let bestVecF = max(gopsF2, gopsF4)
    let bestVecFName = bestVecF == gopsF2 ? "Float2" : "Float4"
    print("Best Float Vector: \(bestVecFName) with \(String(format: "%.2f", bestVecF)) GOPS")

    // 11. HALF-PRECISION VECTORIZATION
    print("\n--- 11. Vectorization Width Comparison (Half) ---")

    guard let half2Func = deepLibrary.makeFunction(name: "vec_half2_op"),
          let half4Func = deepLibrary.makeFunction(name: "vec_half4_op"),
          let half2Pipeline = try? device.makeComputePipelineState(function: half2Func),
          let half4Pipeline = try? device.makeComputePipelineState(function: half4Func) else {
        print("Failed to create half pipelines")
        return
    }

    let halfSize = 16 * 1024 * 1024
    guard let halfIn = device.makeBuffer(length: halfSize * MemoryLayout<UInt16>.size, options: .storageModeShared),
          let halfOut = device.makeBuffer(length: halfSize * MemoryLayout<UInt16>.size, options: .storageModeShared) else {
        return
    }
    var halfSz = UInt32(halfSize)

    // Half2
    let startH2 = getTimeNanos()
    for _ in 0..<vecIter {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(half2Pipeline)
        encoder.setBuffer(halfIn, offset: 0, index: 0)
        encoder.setBuffer(halfOut, offset: 0, index: 1)
        encoder.setBytes(&halfSz, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.dispatchThreads(MTLSize(width: halfSize/2, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let endH2 = getTimeNanos()
    let gopsH2 = Double(halfSize/2) * Double(vecIter) / getElapsedSeconds(start: startH2, end: endH2) / 1e9

    // Half4
    let startH4 = getTimeNanos()
    for _ in 0..<vecIter {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(half4Pipeline)
        encoder.setBuffer(halfIn, offset: 0, index: 0)
        encoder.setBuffer(halfOut, offset: 0, index: 1)
        encoder.setBytes(&halfSz, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.dispatchThreads(MTLSize(width: halfSize/4, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let endH4 = getTimeNanos()
    let gopsH4 = Double(halfSize/4) * Double(vecIter) / getElapsedSeconds(start: startH4, end: endH4) / 1e9

    print("Half2 (width=2):   \(String(format: "%.2f", gopsH2)) GOPS")
    print("Half4 (width=4):   \(String(format: "%.2f", gopsH4)) GOPS")
    let bestVecH = max(gopsH2, gopsH4)
    let bestVecHName = bestVecH == gopsH2 ? "Half2" : "Half4"
    print("Best Half Vector: \(bestVecHName) with \(String(format: "%.2f", bestVecH)) GOPS")

    // 12. MEMORY FENCE IMPACT
    print("\n--- 12. Memory Fence Impact ---")

    guard let fenceNoneFunc = deepLibrary.makeFunction(name: "mem_fence_none"),
          let fenceDeviceFunc = deepLibrary.makeFunction(name: "mem_fence_device"),
          let fenceTGFunc = deepLibrary.makeFunction(name: "mem_fence_threadgroup"),
          let fenceNonePipeline = try? device.makeComputePipelineState(function: fenceNoneFunc),
          let fenceDevicePipeline = try? device.makeComputePipelineState(function: fenceDeviceFunc),
          let fenceTGPipeline = try? device.makeComputePipelineState(function: fenceTGFunc) else {
        print("Failed to create fence pipelines")
        return
    }

    let fenceSize = 4 * 1024 * 1024
    let fenceIter = 100
    guard let fenceIn = device.makeBuffer(length: fenceSize * MemoryLayout<Float>.size, options: .storageModeShared),
          let fenceOut = device.makeBuffer(length: fenceSize * MemoryLayout<Float>.size, options: .storageModeShared) else {
        return
    }
    var fenceSz = UInt32(fenceSize)

    // No fence
    let startNoFence = getTimeNanos()
    for _ in 0..<fenceIter {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(fenceNonePipeline)
        encoder.setBuffer(fenceIn, offset: 0, index: 0)
        encoder.setBuffer(fenceOut, offset: 0, index: 1)
        encoder.setBytes(&fenceSz, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.dispatchThreads(MTLSize(width: fenceSize, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let endNoFence = getTimeNanos()
    let gopsNoFence = Double(fenceSize) * Double(fenceIter) / getElapsedSeconds(start: startNoFence, end: endNoFence) / 1e9

    // Threadgroup fence
    let startTGFence = getTimeNanos()
    for _ in 0..<fenceIter {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(fenceTGPipeline)
        encoder.setBuffer(fenceIn, offset: 0, index: 0)
        encoder.setBuffer(fenceOut, offset: 0, index: 1)
        encoder.setBytes(&fenceSz, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.dispatchThreads(MTLSize(width: fenceSize, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let endTGFence = getTimeNanos()
    let gopsTGFence = Double(fenceSize) * Double(fenceIter) / getElapsedSeconds(start: startTGFence, end: endTGFence) / 1e9

    print("No Fence:           \(String(format: "%.2f", gopsNoFence)) GOPS")
    print("Threadgroup Fence:  \(String(format: "%.2f", gopsTGFence)) GOPS")
    print("Fence Overhead:     \(String(format: "%.1fx", gopsNoFence / gopsTGFence))")

    // 13. KERNEL FUSION STUDY
    print("\n--- 13. Kernel Fusion Analysis ---")

    guard let fusedFunc = deepLibrary.makeFunction(name: "fused_ops"),
          let sepAFunc = deepLibrary.makeFunction(name: "separate_ops_a"),
          let sepBFunc = deepLibrary.makeFunction(name: "separate_ops_b"),
          let fusedPipeline = try? device.makeComputePipelineState(function: fusedFunc),
          let sepAPipeline = try? device.makeComputePipelineState(function: sepAFunc),
          let sepBPipeline = try? device.makeComputePipelineState(function: sepBFunc) else {
        print("Failed to create fusion pipelines")
        return
    }

    let fuseSize = 2 * 1024 * 1024
    let fuseIter = 50
    guard let fuseA = device.makeBuffer(length: fuseSize * MemoryLayout<Float>.size, options: .storageModeShared),
          let fuseB = device.makeBuffer(length: fuseSize * MemoryLayout<Float>.size, options: .storageModeShared),
          let fuseOut = device.makeBuffer(length: fuseSize * MemoryLayout<Float>.size, options: .storageModeShared),
          let fuseTemp = device.makeBuffer(length: fuseSize * MemoryLayout<Float>.size, options: .storageModeShared) else {
        return
    }
    var fuseSz = UInt32(fuseSize)

    // Fused (single kernel)
    let startFused = getTimeNanos()
    for _ in 0..<fuseIter {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(fusedPipeline)
        encoder.setBuffer(fuseA, offset: 0, index: 0)
        encoder.setBuffer(fuseB, offset: 0, index: 1)
        encoder.setBuffer(fuseOut, offset: 0, index: 2)
        encoder.setBytes(&fuseSz, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.dispatchThreads(MTLSize(width: fuseSize, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let endFused = getTimeNanos()
    let gopsFused = Double(fuseSize) * Double(fuseIter) / getElapsedSeconds(start: startFused, end: endFused) / 1e9

    // Separate (2 kernels)
    let startSep = getTimeNanos()
    for _ in 0..<fuseIter {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(sepAPipeline)
        encoder.setBuffer(fuseA, offset: 0, index: 0)
        encoder.setBuffer(fuseB, offset: 0, index: 1)
        encoder.setBuffer(fuseTemp, offset: 0, index: 2)
        encoder.setBytes(&fuseSz, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.dispatchThreads(MTLSize(width: fuseSize, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        guard let cmd2 = queue.makeCommandBuffer(),
              let enc2 = cmd2.makeComputeCommandEncoder() else { continue }
        enc2.setComputePipelineState(sepBPipeline)
        enc2.setBuffer(fuseTemp, offset: 0, index: 2)
        enc2.setBuffer(fuseB, offset: 0, index: 1)
        enc2.setBuffer(fuseOut, offset: 0, index: 3)
        enc2.setBytes(&fuseSz, length: MemoryLayout<UInt32>.size, index: 0)
        enc2.dispatchThreads(MTLSize(width: fuseSize, height: 1, depth: 1),
                         threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc2.endEncoding()
        cmd2.commit()
        cmd2.waitUntilCompleted()
    }
    let endSep = getTimeNanos()
    let gopsSep = Double(fuseSize) * Double(fuseIter) / getElapsedSeconds(start: startSep, end: endSep) / 1e9

    print("Fused (1 kernel):    \(String(format: "%.2f", gopsFused)) GOPS")
    print("Separate (2 kernels): \(String(format: "%.2f", gopsSep)) GOPS")
    print("Fusion Speedup:      \(String(format: "%.2fx", gopsFused / gopsSep))")

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

    // 11. TEXTURE MEMORY ACCESS
    print("\n--- 11. Texture Memory Access Analysis ---")

    guard let texLinearFunc = deepLibrary.makeFunction(name: "tex_read_linear"),
          let tex2DFunc = deepLibrary.makeFunction(name: "tex_read_2d"),
          let tex2DHalfFunc = deepLibrary.makeFunction(name: "tex_read_2d_half"),
          let texLinearPipeline = try? device.makeComputePipelineState(function: texLinearFunc),
          let tex2DPipeline = try? device.makeComputePipelineState(function: tex2DFunc),
          let tex2DHalfPipeline = try? device.makeComputePipelineState(function: tex2DHalfFunc) else {
        print("Failed to create texture pipelines")
        return
    }

    let texSize = 2048
    let texPixels = texSize * texSize
    let texIter = 100

    // Create texture descriptor
    let texDesc = MTLTextureDescriptor()
    texDesc.width = texSize
    texDesc.height = texSize
    texDesc.pixelFormat = .r32Float
    texDesc.usage = [.shaderRead]
    texDesc.storageMode = .shared

    if let tex2D = device.makeTexture(descriptor: texDesc) {
        guard let bufIn = device.makeBuffer(length: texPixels * MemoryLayout<Float>.size, options: .storageModeShared),
              let bufOut = device.makeBuffer(length: texPixels * MemoryLayout<Float>.size, options: .storageModeShared) else {
            print("Failed to create buffers")
            return
        }

        var texSz = UInt32(texSize)

        // Buffer linear read baseline
        let startBuf = getTimeNanos()
        for _ in 0..<texIter {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(texLinearPipeline)
            encoder.setBuffer(bufIn, offset: 0, index: 0)
            encoder.setBuffer(bufOut, offset: 0, index: 1)
            encoder.setBytes(&texSz, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.dispatchThreads(MTLSize(width: texPixels, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        let endBuf = getTimeNanos()
        let gopsBuf = Double(texPixels) * Double(texIter) / getElapsedSeconds(start: startBuf, end: endBuf) / 1e9

        print("Buffer Linear Read: \(String(format: "%.2f", gopsBuf)) GOPS")

        // 2D Texture read
        let startTex = getTimeNanos()
        for _ in 0..<texIter {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(tex2DPipeline)
            encoder.setTexture(tex2D, index: 0)
            encoder.setBuffer(bufOut, offset: 0, index: 0)
            encoder.setBytes(&texSz, length: MemoryLayout<UInt32>.size, index: 1)
            encoder.dispatchThreads(MTLSize(width: texSize, height: texSize, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        let endTex = getTimeNanos()
        let gopsTex = Double(texPixels) * Double(texIter) / getElapsedSeconds(start: startTex, end: endTex) / 1e9
        print("Texture2D Float Read: \(String(format: "%.2f", gopsTex)) GOPS")
    } else {
        print("Failed to create texture")
    }

    // 12. PIPELINE LATENCY (Dependent vs Independent Ops)
    print("\n--- 12. Pipeline Latency Analysis ---")

    guard let depAddFunc = deepLibrary.makeFunction(name: "dep_chain_add"),
          let depMulFunc = deepLibrary.makeFunction(name: "dep_chain_mul"),
          let indepFunc = deepLibrary.makeFunction(name: "indep_ops"),
          let depAddPipeline = try? device.makeComputePipelineState(function: depAddFunc),
          let depMulPipeline = try? device.makeComputePipelineState(function: depMulFunc),
          let indepPipeline = try? device.makeComputePipelineState(function: indepFunc) else {
        print("Failed to create pipeline latency pipelines")
        return
    }

    let pipeSize = 4 * 1024 * 1024
    let pipeIter = 100
    guard let pipeIn = device.makeBuffer(length: pipeSize * MemoryLayout<Float>.size, options: .storageModeShared),
          let pipeOut = device.makeBuffer(length: pipeSize * MemoryLayout<Float>.size, options: .storageModeShared) else {
        print("Failed to create pipeline buffers")
        return
    }
    var pipeSz = UInt32(pipeSize)

    // Dependent chain (add)
    let startDepAdd = getTimeNanos()
    for _ in 0..<pipeIter {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(depAddPipeline)
        encoder.setBuffer(pipeIn, offset: 0, index: 0)
        encoder.setBuffer(pipeOut, offset: 0, index: 1)
        encoder.setBytes(&pipeSz, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.dispatchThreads(MTLSize(width: pipeSize, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let endDepAdd = getTimeNanos()
    let gopsDepAdd = Double(pipeSize) * Double(pipeIter) / getElapsedSeconds(start: startDepAdd, end: endDepAdd) / 1e9

    // Dependent chain (mul)
    let startDepMul = getTimeNanos()
    for _ in 0..<pipeIter {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(depMulPipeline)
        encoder.setBuffer(pipeIn, offset: 0, index: 0)
        encoder.setBuffer(pipeOut, offset: 0, index: 1)
        encoder.setBytes(&pipeSz, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.dispatchThreads(MTLSize(width: pipeSize, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let endDepMul = getTimeNanos()
    let gopsDepMul = Double(pipeSize) * Double(pipeIter) / getElapsedSeconds(start: startDepMul, end: endDepMul) / 1e9

    // Independent ops
    let startIndep = getTimeNanos()
    for _ in 0..<pipeIter {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(indepPipeline)
        encoder.setBuffer(pipeIn, offset: 0, index: 0)
        encoder.setBuffer(pipeOut, offset: 0, index: 1)
        encoder.setBytes(&pipeSz, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.dispatchThreads(MTLSize(width: pipeSize, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let endIndep = getTimeNanos()
    let gopsIndep = Double(pipeSize) * Double(pipeIter) / getElapsedSeconds(start: startIndep, end: endIndep) / 1e9

    print("Dep Chain (add):     \(String(format: "%.2f", gopsDepAdd)) GOPS")
    print("Dep Chain (mul):     \(String(format: "%.2f", gopsDepMul)) GOPS")
    print("Independent Ops:     \(String(format: "%.2f", gopsIndep)) GOPS")
    print("Dep vs Indep Ratio:  \(String(format: "%.2fx", gopsIndep / gopsDepAdd))")

    // 13. SIMD GROUP OPERATIONS
    print("\n--- 13. SIMD Group Operations ---")

    guard let simdVoteAllFunc = deepLibrary.makeFunction(name: "simd_vote_all"),
          let simdVoteAnyFunc = deepLibrary.makeFunction(name: "simd_vote_any"),
          let simdShuffleFunc = deepLibrary.makeFunction(name: "simd_shuffle"),
          let simdPrefixFunc = deepLibrary.makeFunction(name: "simd_prefix_sum"),
          let simdVoteAllPipeline = try? device.makeComputePipelineState(function: simdVoteAllFunc),
          let simdVoteAnyPipeline = try? device.makeComputePipelineState(function: simdVoteAnyFunc),
          let simdShufflePipeline = try? device.makeComputePipelineState(function: simdShuffleFunc),
          let simdPrefixPipeline = try? device.makeComputePipelineState(function: simdPrefixFunc) else {
        print("Failed to create SIMD pipelines")
        return
    }

    let simdSize = 256 * 1024
    let simdIter = 100
    guard let simdIn = device.makeBuffer(length: simdSize * MemoryLayout<Float>.size, options: .storageModeShared),
          let simdOut = device.makeBuffer(length: simdSize * MemoryLayout<Float>.size, options: .storageModeShared) else {
        print("Failed to create SIMD buffers")
        return
    }
    var simdSz = UInt32(simdSize)

    // SIMD Vote All
    let startVoteAll = getTimeNanos()
    for _ in 0..<simdIter {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(simdVoteAllPipeline)
        encoder.setBuffer(simdIn, offset: 0, index: 0)
        encoder.setBuffer(simdOut, offset: 0, index: 1)
        encoder.setBytes(&simdSz, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.dispatchThreads(MTLSize(width: simdSize, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let endVoteAll = getTimeNanos()
    let gopsVoteAll = Double(simdSize) * Double(simdIter) / getElapsedSeconds(start: startVoteAll, end: endVoteAll) / 1e9

    print("SIMD Vote All:   \(String(format: "%.2f", gopsVoteAll)) GOPS")

    // SIMD Shuffle
    let startShuffle = getTimeNanos()
    for _ in 0..<simdIter {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(simdShufflePipeline)
        encoder.setBuffer(simdIn, offset: 0, index: 0)
        encoder.setBuffer(simdOut, offset: 0, index: 1)
        encoder.setBytes(&simdSz, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.dispatchThreads(MTLSize(width: simdSize, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let endShuffle = getTimeNanos()
    let gopsShuffle = Double(simdSize) * Double(simdIter) / getElapsedSeconds(start: startShuffle, end: endShuffle) / 1e9

    print("SIMD Shuffle:    \(String(format: "%.2f", gopsShuffle)) GOPS")

    // SIMD Prefix Sum
    let startPrefix = getTimeNanos()
    for _ in 0..<simdIter {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(simdPrefixPipeline)
        encoder.setBuffer(simdIn, offset: 0, index: 0)
        encoder.setBuffer(simdOut, offset: 0, index: 1)
        encoder.setBytes(&simdSz, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.dispatchThreads(MTLSize(width: simdSize, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let endPrefix = getTimeNanos()
    let gopsPrefix = Double(simdSize) * Double(simdIter) / getElapsedSeconds(start: startPrefix, end: endPrefix) / 1e9

    print("SIMD Prefix Sum: \(String(format: "%.2f", gopsPrefix)) GOPS")

    // 14. TREE-BASED REDUCTION
    print("\n--- 14. Parallel Reduction Algorithms ---")

    guard let reduceSeqFunc = deepLibrary.makeFunction(name: "reduce_sequential"),
          let reduceSharedFunc = deepLibrary.makeFunction(name: "reduce_shared_basic"),
          let reduceWarpFunc = deepLibrary.makeFunction(name: "reduce_warp_level"),
          let reduceSeqPipeline = try? device.makeComputePipelineState(function: reduceSeqFunc),
          let reduceSharedPipeline = try? device.makeComputePipelineState(function: reduceSharedFunc),
          let reduceWarpPipeline = try? device.makeComputePipelineState(function: reduceWarpFunc) else {
        print("Failed to create reduction pipelines")
        return
    }

    let reduceSize = 256 * 1024
    let reduceIter = 10
    guard let reduceIn = device.makeBuffer(length: reduceSize * MemoryLayout<Float>.size, options: .storageModeShared),
          let reduceOut = device.makeBuffer(length: 1024 * MemoryLayout<Float>.size, options: .storageModeShared),
          let reduceShared = device.makeBuffer(length: 1024 * MemoryLayout<Float>.size, options: .storageModeShared) else {
        print("Failed to create reduction buffers")
        return
    }
    var reduceSz = UInt32(reduceSize)

    // Sequential Reduction
    let startSeq = getTimeNanos()
    for _ in 0..<reduceIter {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(reduceSeqPipeline)
        encoder.setBuffer(reduceIn, offset: 0, index: 0)
        encoder.setBuffer(reduceOut, offset: 0, index: 1)
        encoder.setBytes(&reduceSz, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let endSeq = getTimeNanos()
    let gopsSeq = Double(reduceSize) * Double(reduceIter) / getElapsedSeconds(start: startSeq, end: endSeq) / 1e9

    print("Sequential Reduce:  \(String(format: "%.2f", gopsSeq)) GOPS (baseline)")

    // Shared Memory Reduction
    let startShared = getTimeNanos()
    for _ in 0..<reduceIter {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(reduceSharedPipeline)
        encoder.setBuffer(reduceIn, offset: 0, index: 0)
        encoder.setBuffer(reduceOut, offset: 0, index: 1)
        encoder.setBuffer(reduceShared, offset: 0, index: 2)
        encoder.setBytes(&reduceSz, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.dispatchThreads(MTLSize(width: reduceSize, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let endShared = getTimeNanos()
    let gopsShared = Double(reduceSize) * Double(reduceIter) / getElapsedSeconds(start: startShared, end: endShared) / 1e9

    print("Shared Reduce:     \(String(format: "%.2f", gopsShared)) GOPS")

    // Warp-Level Reduction
    let startWarp = getTimeNanos()
    for _ in 0..<reduceIter {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(reduceWarpPipeline)
        encoder.setBuffer(reduceIn, offset: 0, index: 0)
        encoder.setBuffer(reduceOut, offset: 0, index: 1)
        encoder.setBuffer(reduceShared, offset: 0, index: 2)
        encoder.setBytes(&reduceSz, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.dispatchThreads(MTLSize(width: reduceSize, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let endWarp = getTimeNanos()
    let gopsWarp = Double(reduceSize) * Double(reduceIter) / getElapsedSeconds(start: startWarp, end: endWarp) / 1e9

    print("Warp-Level Reduce: \(String(format: "%.2f", gopsWarp)) GOPS")

    // 15. INDIRECT ADDRESSING / SCATTER-GATHER
    print("\n--- 15. Indirect Addressing Analysis ---")

    guard let gatherFunc = deepLibrary.makeFunction(name: "gather_addressing"),
          let scatterFunc = deepLibrary.makeFunction(name: "scatter_addressing"),
          let gatherProcFunc = deepLibrary.makeFunction(name: "gather_then_process"),
          let gatherPipeline = try? device.makeComputePipelineState(function: gatherFunc),
          let scatterPipeline = try? device.makeComputePipelineState(function: scatterFunc),
          let gatherProcPipeline = try? device.makeComputePipelineState(function: gatherProcFunc) else {
        print("Failed to create indirect addressing pipelines")
        return
    }

    let indSize = 256 * 1024
    let indIter = 50
    guard let indData = device.makeBuffer(length: indSize * MemoryLayout<Float>.size, options: .storageModeShared),
          let indIndices = device.makeBuffer(length: indSize * MemoryLayout<UInt32>.size, options: .storageModeShared),
          let indOut = device.makeBuffer(length: indSize * MemoryLayout<Float>.size, options: .storageModeShared) else {
        print("Failed to create indirect buffers")
        return
    }

    // Initialize indices sequentially
    let indicesPtr = indIndices.contents().assumingMemoryBound(to: UInt32.self)
    for i in 0..<indSize {
        indicesPtr[i] = UInt32(i)
    }

    var indSz = UInt32(indSize)

    // Gather (indexed read)
    let startGather = getTimeNanos()
    for _ in 0..<indIter {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(gatherPipeline)
        encoder.setBuffer(indData, offset: 0, index: 0)
        encoder.setBuffer(indIndices, offset: 0, index: 1)
        encoder.setBuffer(indOut, offset: 0, index: 2)
        encoder.setBytes(&indSz, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.dispatchThreads(MTLSize(width: indSize, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let endGather = getTimeNanos()
    let gopsGather = Double(indSize) * Double(indIter) / getElapsedSeconds(start: startGather, end: endGather) / 1e9

    // Scatter (indexed write)
    let startScatter = getTimeNanos()
    for _ in 0..<indIter {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(scatterPipeline)
        encoder.setBuffer(indOut, offset: 0, index: 0)
        encoder.setBuffer(indIndices, offset: 0, index: 1)
        encoder.setBuffer(indData, offset: 0, index: 2)
        encoder.setBytes(&indSz, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.dispatchThreads(MTLSize(width: indSize, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let endScatter = getTimeNanos()
    let gopsScatter = Double(indSize) * Double(indIter) / getElapsedSeconds(start: startScatter, end: endScatter) / 1e9

    // Gather + Process
    let startGatherProc = getTimeNanos()
    for _ in 0..<indIter {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(gatherProcPipeline)
        encoder.setBuffer(indData, offset: 0, index: 0)
        encoder.setBuffer(indIndices, offset: 0, index: 1)
        encoder.setBuffer(indOut, offset: 0, index: 2)
        encoder.setBytes(&indSz, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.dispatchThreads(MTLSize(width: indSize, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let endGatherProc = getTimeNanos()
    let gopsGatherProc = Double(indSize) * Double(indIter) / getElapsedSeconds(start: startGatherProc, end: endGatherProc) / 1e9

    print("Gather (indexed read): \(String(format: "%.3f", gopsGather)) GOPS")
    print("Scatter (indexed write): \(String(format: "%.3f", gopsScatter)) GOPS")
    print("Gather+Process: \(String(format: "%.3f", gopsGatherProc)) GOPS")

    // 16. DOUBLE BUFFERING / PIPELINING
    print("\n--- 16. Double Buffering Analysis ---")

    guard let dualAFunc = deepLibrary.makeFunction(name: "dual_buffer_compute_a"),
          let dualBFunc = deepLibrary.makeFunction(name: "dual_buffer_compute_b"),
          let singleBufFunc = deepLibrary.makeFunction(name: "single_buffer_compute"),
          let dualAPipeline = try? device.makeComputePipelineState(function: dualAFunc),
          let dualBPipeline = try? device.makeComputePipelineState(function: dualBFunc),
          let singleBufPipeline = try? device.makeComputePipelineState(function: singleBufFunc) else {
        print("Failed to create double buffering pipelines")
        return
    }

    let dblSize = 256 * 1024
    let dblIter = 50
    guard let dblInA = device.makeBuffer(length: dblSize * MemoryLayout<Float>.size, options: .storageModeShared),
          let dblInB = device.makeBuffer(length: dblSize * MemoryLayout<Float>.size, options: .storageModeShared),
          let dblOut = device.makeBuffer(length: dblSize * MemoryLayout<Float>.size, options: .storageModeShared),
          let dblTemp = device.makeBuffer(length: dblSize * MemoryLayout<Float>.size, options: .storageModeShared) else {
        print("Failed to create double buffering buffers")
        return
    }

    var dblSz = UInt32(dblSize)

    // Single Buffer Baseline
    let startSingle = getTimeNanos()
    for _ in 0..<dblIter {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(singleBufPipeline)
        encoder.setBuffer(dblInA, offset: 0, index: 0)
        encoder.setBuffer(dblOut, offset: 0, index: 1)
        encoder.setBytes(&dblSz, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.dispatchThreads(MTLSize(width: dblSize, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let endSingle = getTimeNanos()
    let gopsSingle = Double(dblSize) * Double(dblIter) / getElapsedSeconds(start: startSingle, end: endSingle) / 1e9

    // Double Buffer - Phase A
    let startDualA = getTimeNanos()
    for _ in 0..<dblIter {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(dualAPipeline)
        encoder.setBuffer(dblInA, offset: 0, index: 0)
        encoder.setBuffer(dblInB, offset: 0, index: 1)
        encoder.setBuffer(dblOut, offset: 0, index: 2)
        encoder.setBuffer(dblTemp, offset: 0, index: 3)
        encoder.setBytes(&dblSz, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.dispatchThreads(MTLSize(width: dblSize, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let endDualA = getTimeNanos()
    let gopsDualA = Double(dblSize) * Double(dblIter) / getElapsedSeconds(start: startDualA, end: endDualA) / 1e9

    // Double Buffer - Phase B (continuation)
    let startDualB = getTimeNanos()
    for _ in 0..<dblIter {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(dualBPipeline)
        encoder.setBuffer(dblInA, offset: 0, index: 0)
        encoder.setBuffer(dblInB, offset: 0, index: 1)
        encoder.setBuffer(dblOut, offset: 0, index: 2)
        encoder.setBuffer(dblTemp, offset: 0, index: 3)
        encoder.setBytes(&dblSz, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.dispatchThreads(MTLSize(width: dblSize, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let endDualB = getTimeNanos()
    let gopsDualB = Double(dblSize) * Double(dblIter) / getElapsedSeconds(start: startDualB, end: endDualB) / 1e9

    print("Single Buffer: \(String(format: "%.3f", gopsSingle)) GOPS (baseline)")
    print("Double Buffer A: \(String(format: "%.3f", gopsDualA)) GOPS")
    print("Double Buffer B: \(String(format: "%.3f", gopsDualB)) GOPS")

    // 17. THREADGROUP MEMORY PERFORMANCE
    print("\n--- 17. Threadgroup Memory Analysis ---")

    guard let sharedSeqFunc = deepLibrary.makeFunction(name: "shared_copy_seq"),
          let sharedStrideFunc = deepLibrary.makeFunction(name: "shared_copy_strided"),
          let sharedSumFunc = deepLibrary.makeFunction(name: "shared_fill_and_sum"),
          let sharedSeqPipeline = try? device.makeComputePipelineState(function: sharedSeqFunc),
          let sharedStridePipeline = try? device.makeComputePipelineState(function: sharedStrideFunc),
          let sharedSumPipeline = try? device.makeComputePipelineState(function: sharedSumFunc) else {
        print("Failed to create threadgroup pipelines")
        return
    }

    // 256 threads x 128 floats per thread = 32KB (full shared memory)
    // Dispatch 128 threadgroups to fully utilize GPU
    let tgThreads = 256
    let elementsPerThread = 128
    let totalElements = tgThreads * elementsPerThread  // 32KB
    let numThreadgroups = 128  // Fully utilize GPU

    guard let tgIn = device.makeBuffer(length: totalElements * MemoryLayout<Float>.size, options: .storageModeShared),
          let tgOut = device.makeBuffer(length: totalElements * MemoryLayout<Float>.size, options: .storageModeShared) else {
        print("Failed to create threadgroup buffers")
        return
    }

    var tgSz = UInt32(tgThreads)

    // Sequential shared memory copy - each threadgroup (256 threads) fills its 32KB shared memory
    // Each thread: 128 loads + 128 stores = 256 operations
    // Total per dispatch: numThreadgroups * 256 threads * 128 elements * 2 ops
    let totalThreads = numThreadgroups * tgThreads
    let opsPerDispatch = Double(totalThreads) * Double(elementsPerThread) * 2  // load + store

    let tgStartSeq = getTimeNanos()
    guard let cmdSeq = queue.makeCommandBuffer(),
          let encSeq = cmdSeq.makeComputeCommandEncoder() else { return }
    encSeq.setComputePipelineState(sharedSeqPipeline)
    encSeq.setBuffer(tgIn, offset: 0, index: 0)
    encSeq.setBuffer(tgOut, offset: 0, index: 1)
    encSeq.setBytes(&tgSz, length: MemoryLayout<UInt32>.size, index: 2)
    encSeq.dispatchThreads(MTLSize(width: totalElements, height: 1, depth: 1),
                       threadsPerThreadgroup: MTLSize(width: tgThreads, height: 1, depth: 1))
    encSeq.endEncoding()
    cmdSeq.commit()
    cmdSeq.waitUntilCompleted()
    let tgEndSeq = getTimeNanos()
    let tgElapsedSeq = getElapsedSeconds(start: tgStartSeq, end: tgEndSeq)
    let tgGopsSeq = opsPerDispatch / tgElapsedSeq / 1e9

    // Strided shared memory copy - causes bank conflicts
    let tgStartStride = getTimeNanos()
    guard let cmdStride = queue.makeCommandBuffer(),
          let encStride = cmdStride.makeComputeCommandEncoder() else { return }
    encStride.setComputePipelineState(sharedStridePipeline)
    encStride.setBuffer(tgIn, offset: 0, index: 0)
    encStride.setBuffer(tgOut, offset: 0, index: 1)
    encStride.setBytes(&tgSz, length: MemoryLayout<UInt32>.size, index: 2)
    encStride.dispatchThreads(MTLSize(width: totalElements, height: 1, depth: 1),
                       threadsPerThreadgroup: MTLSize(width: tgThreads, height: 1, depth: 1))
    encStride.endEncoding()
    cmdStride.commit()
    cmdStride.waitUntilCompleted()
    let tgEndStride = getTimeNanos()
    let tgGopsStride = opsPerDispatch / getElapsedSeconds(start: tgStartStride, end: tgEndStride) / 1e9

    // Fill and sum in shared memory
    let tgStartSum = getTimeNanos()
    guard let cmdSum = queue.makeCommandBuffer(),
          let encSum = cmdSum.makeComputeCommandEncoder() else { return }
    encSum.setComputePipelineState(sharedSumPipeline)
    encSum.setBuffer(tgOut, offset: 0, index: 0)
    encSum.setBytes(&tgSz, length: MemoryLayout<UInt32>.size, index: 1)
    encSum.dispatchThreads(MTLSize(width: totalElements, height: 1, depth: 1),
                       threadsPerThreadgroup: MTLSize(width: tgThreads, height: 1, depth: 1))
    encSum.endEncoding()
    cmdSum.commit()
    cmdSum.waitUntilCompleted()
    let tgEndSum = getTimeNanos()
    let tgGopsSum = opsPerDispatch / getElapsedSeconds(start: tgStartSum, end: tgEndSum) / 1e9

    print("Shared Seq (no conflict): \(String(format: "%.3f", tgGopsSeq)) GOPS")
    print("Shared Strided (conflict): \(String(format: "%.3f", tgGopsStride)) GOPS")
    print("Shared Fill+Sum: \(String(format: "%.3f", tgGopsSum)) GOPS")

    // ============================================================
    // 18. HISTOGRAM COMPUTATION
    // Tests atomic operations and memory access patterns
    // ============================================================
    print("\n--- 18. Histogram Computation Analysis ---")

    guard let histNaiveFunc = deepLibrary.makeFunction(name: "histogram_naive"),
          let histLocalFunc = deepLibrary.makeFunction(name: "histogram_local"),
          let histStridedFunc = deepLibrary.makeFunction(name: "histogram_strided"),
          let histVecFunc = deepLibrary.makeFunction(name: "histogram_vectorized"),
          let histNaivePipeline = try? device.makeComputePipelineState(function: histNaiveFunc),
          let histLocalPipeline = try? device.makeComputePipelineState(function: histLocalFunc),
          let histStridedPipeline = try? device.makeComputePipelineState(function: histStridedFunc),
          let histVecPipeline = try? device.makeComputePipelineState(function: histVecFunc) else {
        print("Failed to create histogram pipelines")
        return
    }

    let histSize = 4 * 1024 * 1024  // 4M elements
    let histBins: UInt32 = 256
    let histIterations = 10

    // Create data buffer (random values 0.0-1.0)
    guard let histData = device.makeBuffer(length: histSize * MemoryLayout<Float>.size, options: .storageModeShared),
          let histOutput = device.makeBuffer(length: Int(histBins) * MemoryLayout<UInt32>.size, options: .storageModeShared) else {
        print("Failed to create histogram buffers")
        return
    }

    // Initialize data with random values
    let histDataPtr = histData.contents().bindMemory(to: Float.self, capacity: histSize)
    for i in 0..<histSize {
        histDataPtr[i] = Float.random(in: 0.0...1.0)
    }

    var histSz = UInt32(histSize)
    var histBns = histBins

    // Naive histogram - high atomic contention
    let histStartNaive = getTimeNanos()
    for _ in 0..<histIterations {
        // Clear output
        memset(histOutput.contents(), 0, Int(histBins) * MemoryLayout<UInt32>.size)

        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(histNaivePipeline)
        encoder.setBuffer(histData, offset: 0, index: 0)
        encoder.setBuffer(histOutput, offset: 0, index: 1)
        encoder.setBytes(&histSz, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.setBytes(&histBns, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.dispatchThreads(MTLSize(width: histSize, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let histEndNaive = getTimeNanos()
    let histOpsNaive = Double(histSize) * Double(histIterations)
    let histGopsNaive = histOpsNaive / getElapsedSeconds(start: histStartNaive, end: histEndNaive) / 1e9

    // Strided histogram - less contention per thread
    let histStartStride = getTimeNanos()
    for _ in 0..<histIterations {
        memset(histOutput.contents(), 0, Int(histBins) * MemoryLayout<UInt32>.size)

        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(histStridedPipeline)
        encoder.setBuffer(histData, offset: 0, index: 0)
        encoder.setBuffer(histOutput, offset: 0, index: 1)
        encoder.setBytes(&histSz, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.setBytes(&histBns, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.dispatchThreads(MTLSize(width: histSize / 256, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let histEndStride = getTimeNanos()
    let histGopsStride = histOpsNaive / getElapsedSeconds(start: histStartStride, end: histEndStride) / 1e9

    // Vectorized histogram - processes 4 elements per thread
    let histStartVec = getTimeNanos()
    let vecIterations = histIterations * 4  // 4 elements per thread
    for _ in 0..<vecIterations {
        memset(histOutput.contents(), 0, Int(histBins) * MemoryLayout<UInt32>.size)

        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(histVecPipeline)
        encoder.setBuffer(histData, offset: 0, index: 0)
        encoder.setBuffer(histOutput, offset: 0, index: 1)
        encoder.setBytes(&histSz, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.setBytes(&histBns, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.dispatchThreads(MTLSize(width: histSize / 4, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let histEndVec = getTimeNanos()
    let histGopsVec = Double(histSize) * Double(vecIterations) / getElapsedSeconds(start: histStartVec, end: histEndVec) / 1e9

    print("Histogram Naive (contention): \(String(format: "%.3f", histGopsNaive)) GOPS")
    print("Histogram Strided: \(String(format: "%.3f", histGopsStride)) GOPS")
    print("Histogram Vectorized (float4): \(String(format: "%.3f", histGopsVec)) GOPS")

    // ============================================================
    // 19. MATRIX TRANSPOSE
    // Tests shared memory bank conflict patterns
    // ============================================================
    print("\n--- 19. Matrix Transpose Analysis ---")

    guard let transNaiveFunc = deepLibrary.makeFunction(name: "transpose_naive"),
          let transSharedFunc = deepLibrary.makeFunction(name: "transpose_shared"),
          let transNaivePipeline = try? device.makeComputePipelineState(function: transNaiveFunc),
          let transSharedPipeline = try? device.makeComputePipelineState(function: transSharedFunc) else {
        print("Failed to create transpose pipelines")
        return
    }

    // Matrix transpose: 1024x1024 matrix
    let transWidth: UInt32 = 1024
    let transHeight: UInt32 = 1024
    let transSize = Int(transWidth) * Int(transHeight)
    let transIterations = 10

    guard let transIn = device.makeBuffer(length: transSize * MemoryLayout<Float>.size, options: .storageModeShared),
          let transOut = device.makeBuffer(length: transSize * MemoryLayout<Float>.size, options: .storageModeShared) else {
        print("Failed to create transpose buffers")
        return
    }

    var transW = transWidth
    var transH = transHeight

    // Naive transpose
    let transStartNaive = getTimeNanos()
    for _ in 0..<transIterations {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(transNaivePipeline)
        encoder.setBuffer(transIn, offset: 0, index: 0)
        encoder.setBuffer(transOut, offset: 0, index: 1)
        encoder.setBytes(&transW, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.setBytes(&transH, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.dispatchThreads(MTLSize(width: Int(transWidth), height: Int(transHeight), depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let transEndNaive = getTimeNanos()
    let transOpsNaive = Double(transSize) * Double(transIterations)
    let transGopsNaive = transOpsNaive / getElapsedSeconds(start: transStartNaive, end: transEndNaive) / 1e9

    // Shared memory transpose
    let transStartShared = getTimeNanos()
    for _ in 0..<transIterations {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(transSharedPipeline)
        encoder.setBuffer(transIn, offset: 0, index: 0)
        encoder.setBuffer(transOut, offset: 0, index: 1)
        encoder.setBytes(&transW, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.setBytes(&transH, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.dispatchThreads(MTLSize(width: Int(transWidth), height: Int(transHeight), depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let transEndShared = getTimeNanos()
    let transGopsShared = transOpsNaive / getElapsedSeconds(start: transStartShared, end: transEndShared) / 1e9

    print("Transpose Naive: \(String(format: "%.3f", transGopsNaive)) GOPS")
    print("Transpose Shared (tile): \(String(format: "%.3f", transGopsShared)) GOPS")

    // ============================================================
    // 20. STENCIL COMPUTATION
    // Tests shared memory for halo cells and multi-pass stencil operations
    // ============================================================
    print("\n--- 20. Stencil Computation Analysis ---")

    guard let stencilNaiveFunc = deepLibrary.makeFunction(name: "stencil_naive"),
          let stencilSharedFunc = deepLibrary.makeFunction(name: "stencil_shared"),
          let stencilIterFunc = deepLibrary.makeFunction(name: "stencil_iterated"),
          let stencilNaivePipeline = try? device.makeComputePipelineState(function: stencilNaiveFunc),
          let stencilSharedPipeline = try? device.makeComputePipelineState(function: stencilSharedFunc),
          let stencilIterPipeline = try? device.makeComputePipelineState(function: stencilIterFunc) else {
        print("Failed to create stencil pipelines")
        return
    }

    // 1024x1024 grid
    let stencilWidth: UInt32 = 1024
    let stencilHeight: UInt32 = 1024
    let stencilSize = Int(stencilWidth) * Int(stencilHeight)
    let stencilIterations = 10

    guard let stencilIn = device.makeBuffer(length: stencilSize * MemoryLayout<Float>.size, options: .storageModeShared),
          let stencilOut = device.makeBuffer(length: stencilSize * MemoryLayout<Float>.size, options: .storageModeShared) else {
        print("Failed to create stencil buffers")
        return
    }

    var stenW = stencilWidth
    var stenH = stencilHeight

    // Naive stencil - global memory loads for each neighbor
    let stencilStartNaive = getTimeNanos()
    for _ in 0..<stencilIterations {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(stencilNaivePipeline)
        encoder.setBuffer(stencilIn, offset: 0, index: 0)
        encoder.setBuffer(stencilOut, offset: 0, index: 1)
        encoder.setBytes(&stenW, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.setBytes(&stenH, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.dispatchThreads(MTLSize(width: Int(stencilWidth), height: Int(stencilHeight), depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let stencilEndNaive = getTimeNanos()
    let stencilOpsNaive = Double(stencilSize) * Double(stencilIterations)
    let stencilGopsNaive = stencilOpsNaive / getElapsedSeconds(start: stencilStartNaive, end: stencilEndNaive) / 1e9

    // Shared memory stencil
    let stencilStartShared = getTimeNanos()
    for _ in 0..<stencilIterations {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(stencilSharedPipeline)
        encoder.setBuffer(stencilIn, offset: 0, index: 0)
        encoder.setBuffer(stencilOut, offset: 0, index: 1)
        encoder.setBytes(&stenW, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.setBytes(&stenH, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.dispatchThreads(MTLSize(width: Int(stencilWidth), height: Int(stencilHeight), depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let stencilEndShared = getTimeNanos()
    let stencilGopsShared = stencilOpsNaive / getElapsedSeconds(start: stencilStartShared, end: stencilEndShared) / 1e9

    // Iterated stencil (5 iterations)
    let stencilStartIter = getTimeNanos()
    for _ in 0..<stencilIterations {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(stencilIterPipeline)
        encoder.setBuffer(stencilIn, offset: 0, index: 0)
        encoder.setBuffer(stencilOut, offset: 0, index: 1)
        encoder.setBytes(&stenW, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.setBytes(&stenH, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.dispatchThreads(MTLSize(width: Int(stencilWidth), height: Int(stencilHeight), depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let stencilEndIter = getTimeNanos()
    let stencilGopsIter = stencilOpsNaive / getElapsedSeconds(start: stencilStartIter, end: stencilEndIter) / 1e9

    print("Stencil Naive (global mem): \(String(format: "%.3f", stencilGopsNaive)) GOPS")
    print("Stencil Shared (tile+halo): \(String(format: "%.3f", stencilGopsShared)) GOPS")
    print("Stencil Iterated (5x): \(String(format: "%.3f", stencilGopsIter)) GOPS")

    // ============================================================
    // 21. STREAM COMPACTION
    // Tests parallel filter/compact operations with atomics
    // ============================================================
    print("\n--- 21. Stream Compaction Analysis ---")

    guard let compactNaiveFunc = deepLibrary.makeFunction(name: "compact_naive"),
          let compactTiledFunc = deepLibrary.makeFunction(name: "compact_tiled"),
          let compactNaivePipeline = try? device.makeComputePipelineState(function: compactNaiveFunc),
          let compactTiledPipeline = try? device.makeComputePipelineState(function: compactTiledFunc) else {
        print("Failed to create compact pipelines")
        return
    }

    let compactSize = 256 * 1024  // 256K elements
    let compactIterations = 10

    guard let compactIn = device.makeBuffer(length: compactSize * MemoryLayout<Float>.size, options: .storageModeShared),
          let compactOut = device.makeBuffer(length: compactSize * MemoryLayout<Float>.size, options: .storageModeShared),
          let compactCount = device.makeBuffer(length: MemoryLayout<UInt32>.size, options: .storageModeShared) else {
        print("Failed to create compact buffers")
        return
    }

    // Initialize input with random values
    let compactInPtr = compactIn.contents().bindMemory(to: Float.self, capacity: compactSize)
    for i in 0..<compactSize {
        compactInPtr[i] = Float.random(in: 0.0...1.0)
    }

    var compactSizeVar = UInt32(compactSize)

    // Naive compaction
    let compactStartNaive = getTimeNanos()
    for _ in 0..<compactIterations {
        // Reset count
        let countPtr = compactCount.contents().bindMemory(to: UInt32.self, capacity: 1)
        countPtr.pointee = 0

        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(compactNaivePipeline)
        encoder.setBuffer(compactIn, offset: 0, index: 0)
        encoder.setBuffer(compactOut, offset: 0, index: 1)
        encoder.setBuffer(compactCount, offset: 0, index: 2)
        encoder.setBytes(&compactSizeVar, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.dispatchThreads(MTLSize(width: compactSize, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let compactEndNaive = getTimeNanos()
    let compactOpsNaive = Double(compactSize) * Double(compactIterations)
    let compactGopsNaive = compactOpsNaive / getElapsedSeconds(start: compactStartNaive, end: compactEndNaive) / 1e9

    // Tiled compaction
    let compactStartTiled = getTimeNanos()
    for _ in 0..<compactIterations {
        let countPtr = compactCount.contents().bindMemory(to: UInt32.self, capacity: 1)
        countPtr.pointee = 0

        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(compactTiledPipeline)
        encoder.setBuffer(compactIn, offset: 0, index: 0)
        encoder.setBuffer(compactOut, offset: 0, index: 1)
        encoder.setBuffer(compactCount, offset: 0, index: 2)
        encoder.setBytes(&compactSizeVar, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.dispatchThreads(MTLSize(width: compactSize, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let compactEndTiled = getTimeNanos()
    let compactGopsTiled = compactOpsNaive / getElapsedSeconds(start: compactStartTiled, end: compactEndTiled) / 1e9

    print("Compact Naive: \(String(format: "%.3f", compactGopsNaive)) GOPS")
    print("Compact Tiled: \(String(format: "%.3f", compactGopsTiled)) GOPS")

    // ============================================================
    // 22. RADIX SORT
    // Tests parallel sorting with histogram and scatter patterns
    // ============================================================
    print("\n--- 22. Radix Sort Analysis ---")

    guard let radixHistFunc = deepLibrary.makeFunction(name: "radix_histogram"),
          let radixPrefixFunc = deepLibrary.makeFunction(name: "radix_prefix_sum"),
          let radixReorderFunc = deepLibrary.makeFunction(name: "radix_reorder"),
          let radixHistPipeline = try? device.makeComputePipelineState(function: radixHistFunc),
          let radixPrefixPipeline = try? device.makeComputePipelineState(function: radixPrefixFunc),
          let radixReorderPipeline = try? device.makeComputePipelineState(function: radixReorderFunc) else {
        print("Failed to create radix sort pipelines")
        return
    }

    let radixSize = 256 * 1024  // 256K elements
    let radixIterations = 5
    let radixPasses = 4  // 4-bit radix = 16 buckets, 4 passes for 32-bit values

    guard let radixIn = device.makeBuffer(length: radixSize * MemoryLayout<UInt32>.size, options: .storageModeShared),
          let radixOut = device.makeBuffer(length: radixSize * MemoryLayout<UInt32>.size, options: .storageModeShared),
          let radixHistogram = device.makeBuffer(length: 16 * MemoryLayout<UInt32>.size, options: .storageModeShared),
          let radixPrefix = device.makeBuffer(length: 16 * MemoryLayout<UInt32>.size, options: .storageModeShared),
          let radixCounters = device.makeBuffer(length: 16 * MemoryLayout<UInt32>.size, options: .storageModeShared) else {
        print("Failed to create radix sort buffers")
        return
    }

    // Initialize input with random uint values
    let radixInPtr = radixIn.contents().bindMemory(to: UInt32.self, capacity: radixSize)
    for i in 0..<radixSize {
        radixInPtr[i] = UInt32.random(in: 0...UInt32.max)
    }

    var radixSizeVar = UInt32(radixSize)

    // Radix sort: 3-phase approach (histogram -> prefix -> reorder)
    let radixStart = getTimeNanos()
    for _ in 0..<radixIterations {
        for pass in 0..<radixPasses {
            // Phase 1: Histogram
            var passVar = UInt32(pass)

            // Reset histogram
            let histPtr = radixHistogram.contents().bindMemory(to: UInt32.self, capacity: 16)
            for i in 0..<16 { histPtr[i] = 0 }
            let countersPtr = radixCounters.contents().bindMemory(to: UInt32.self, capacity: 16)
            for i in 0..<16 { countersPtr[i] = 0 }

            guard let cmd1 = queue.makeCommandBuffer(),
                  let enc1 = cmd1.makeComputeCommandEncoder() else { continue }
            enc1.setComputePipelineState(radixHistPipeline)
            enc1.setBuffer(radixIn, offset: 0, index: 0)
            enc1.setBuffer(radixHistogram, offset: 0, index: 1)
            enc1.setBytes(&radixSizeVar, length: MemoryLayout<UInt32>.size, index: 2)
            enc1.setBytes(&passVar, length: MemoryLayout<UInt32>.size, index: 3)
            enc1.dispatchThreads(MTLSize(width: radixSize, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc1.endEncoding()
            cmd1.commit()
            cmd1.waitUntilCompleted()

            // Phase 2: Prefix sum (single thread)
            guard let cmd2 = queue.makeCommandBuffer(),
                  let enc2 = cmd2.makeComputeCommandEncoder() else { continue }
            enc2.setComputePipelineState(radixPrefixPipeline)
            enc2.setBuffer(radixHistogram, offset: 0, index: 0)
            enc2.setBuffer(radixPrefix, offset: 0, index: 1)
            enc2.dispatchThreads(MTLSize(width: 16, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 16, height: 1, depth: 1))
            enc2.endEncoding()
            cmd2.commit()
            cmd2.waitUntilCompleted()

            // Phase 3: Reorder
            guard let cmd3 = queue.makeCommandBuffer(),
                  let enc3 = cmd3.makeComputeCommandEncoder() else { continue }
            enc3.setComputePipelineState(radixReorderPipeline)
            enc3.setBuffer(radixIn, offset: 0, index: 0)
            enc3.setBuffer(radixOut, offset: 0, index: 1)
            enc3.setBuffer(radixHistogram, offset: 0, index: 2)
            enc3.setBuffer(radixPrefix, offset: 0, index: 3)
            enc3.setBuffer(radixCounters, offset: 0, index: 4)
            enc3.setBytes(&radixSizeVar, length: MemoryLayout<UInt32>.size, index: 5)
            enc3.setBytes(&passVar, length: MemoryLayout<UInt32>.size, index: 6)
            enc3.dispatchThreads(MTLSize(width: radixSize, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc3.endEncoding()
            cmd3.commit()
            cmd3.waitUntilCompleted()
        }
    }
    let radixEnd = getTimeNanos()
    let radixOps = Int64(radixSize) * Int64(radixIterations) * Int64(radixPasses) * 3  // 3 phases
    let radixGops = Double(radixOps) / getElapsedSeconds(start: radixStart, end: radixEnd) / 1e9

    print("Radix Sort (256K x 4 passes): \(String(format: "%.3f", radixGops)) GOPS")

    // ============================================================
    // 23. SPARSE MATRIX VECTOR MULTIPLY (SpMV)
    // Tests CSR and ELLPACK formats for sparse matrices
    // ============================================================
    print("\n--- 23. Sparse Matrix Vector Multiply (SpMV) Analysis ---")

    guard let spmvCsrNaiveFunc = deepLibrary.makeFunction(name: "spmv_csr_naive"),
          let spmvCsrVecFunc = deepLibrary.makeFunction(name: "spmv_csr_vectorized"),
          let spmvEllpackFunc = deepLibrary.makeFunction(name: "spmv_ellpack"),
          let spmvCsrNaivePipeline = try? device.makeComputePipelineState(function: spmvCsrNaiveFunc),
          let spmvCsrVecPipeline = try? device.makeComputePipelineState(function: spmvCsrVecFunc),
          let spmvEllpackPipeline = try? device.makeComputePipelineState(function: spmvEllpackFunc) else {
        print("Failed to create SpMV pipelines")
        return
    }

    // Sparse matrix: 8192 x 8192 with ~5% density = ~3.3M non-zeros
    let spmvRows = 8192
    let spmvCols = 8192
    let spmvDensity = 0.05  // 5% sparsity
    let spmvNnz = spmvRows * spmvCols / 20  // ~3.3M non-zeros

    let spmvIterations = 10

    // Create CSR format buffers
    guard let spmvValues = device.makeBuffer(length: spmvNnz * MemoryLayout<Float>.size, options: .storageModeShared),
          let spmvColIndices = device.makeBuffer(length: spmvNnz * MemoryLayout<UInt32>.size, options: .storageModeShared),
          let spmvRowOffsets = device.makeBuffer(length: (spmvRows + 1) * MemoryLayout<UInt32>.size, options: .storageModeShared),
          let spmvX = device.makeBuffer(length: spmvCols * MemoryLayout<Float>.size, options: .storageModeShared),
          let spmvY = device.makeBuffer(length: spmvRows * MemoryLayout<Float>.size, options: .storageModeShared) else {
        print("Failed to create SpMV buffers")
        return
    }

    // Initialize sparse matrix in CSR format
    let valuesPtr = spmvValues.contents().bindMemory(to: Float.self, capacity: spmvNnz)
    let colIndicesPtr = spmvColIndices.contents().bindMemory(to: UInt32.self, capacity: spmvNnz)
    let rowOffsetsPtr = spmvRowOffsets.contents().bindMemory(to: UInt32.self, capacity: spmvRows + 1)
    let xPtr = spmvX.contents().bindMemory(to: Float.self, capacity: spmvCols)
    let yPtr = spmvY.contents().bindMemory(to: Float.self, capacity: spmvRows)

    // Fill x vector
    for i in 0..<spmvCols {
        xPtr[i] = Float.random(in: 0.0...1.0)
    }

    // Build CSR matrix (random sparsity pattern per row)
    rowOffsetsPtr[0] = 0
    var nnzIdx = 0
    for row in 0..<spmvRows {
        let rowNnz = spmvNnz / spmvRows  // Average ~20 non-zeros per row
        for j in 0..<rowNnz {
            if nnzIdx < spmvNnz {
                colIndicesPtr[nnzIdx] = UInt32.random(in: 0..<UInt32(spmvCols))
                valuesPtr[nnzIdx] = Float.random(in: -1.0...1.0)
                nnzIdx += 1
            }
        }
        rowOffsetsPtr[row + 1] = UInt32(nnzIdx)
    }

    var spmvRowsVar = UInt32(spmvRows)

    // SpMV CSR Naive
    let spmvStartNaive = getTimeNanos()
    for _ in 0..<spmvIterations {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(spmvCsrNaivePipeline)
        encoder.setBuffer(spmvValues, offset: 0, index: 0)
        encoder.setBuffer(spmvColIndices, offset: 0, index: 1)
        encoder.setBuffer(spmvRowOffsets, offset: 0, index: 2)
        encoder.setBuffer(spmvX, offset: 0, index: 3)
        encoder.setBuffer(spmvY, offset: 0, index: 4)
        encoder.setBytes(&spmvRowsVar, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.dispatchThreads(MTLSize(width: spmvRows, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let spmvEndNaive = getTimeNanos()
    let spmvOpsNaive = Int64(spmvNnz) * Int64(spmvIterations)  // One multiply-add per non-zero
    let spmvGopsNaive = Double(spmvOpsNaive) / getElapsedSeconds(start: spmvStartNaive, end: spmvEndNaive) / 1e9

    print("SpMV CSR Naive (\(spmvRows)x\(spmvCols), \(spmvNnz) nnz): \(String(format: "%.3f", spmvGopsNaive)) GOPS")

    // ============================================================
    // 24. TRIDIAGONAL MATRIX SOLVER
    // Tests Thomas Algorithm for tridiagonal systems
    // ============================================================
    print("\n--- 24. Tridiagonal Matrix Solver Analysis ---")

    guard let tridiagThomasFunc = deepLibrary.makeFunction(name: "tridiagonal_thomas"),
          let tridiagThomasPipeline = try? device.makeComputePipelineState(function: tridiagThomasFunc) else {
        print("Failed to create tridiagonal pipeline")
        return
    }

    let tridiagSize = 1024 * 1024  // 1M elements
    let tridiagIterations = 10

    guard let tridiagA = device.makeBuffer(length: tridiagSize * MemoryLayout<Float>.size, options: .storageModeShared),
          let tridiagB = device.makeBuffer(length: tridiagSize * MemoryLayout<Float>.size, options: .storageModeShared),
          let tridiagC = device.makeBuffer(length: tridiagSize * MemoryLayout<Float>.size, options: .storageModeShared),
          let tridiagD = device.makeBuffer(length: tridiagSize * MemoryLayout<Float>.size, options: .storageModeShared),
          let tridiagX = device.makeBuffer(length: tridiagSize * MemoryLayout<Float>.size, options: .storageModeShared),
          let tridiagCp = device.makeBuffer(length: tridiagSize * MemoryLayout<Float>.size, options: .storageModeShared),
          let tridiagDp = device.makeBuffer(length: tridiagSize * MemoryLayout<Float>.size, options: .storageModeShared) else {
        print("Failed to create tridiagonal buffers")
        return
    }

    // Initialize tridiagonal matrix (diagonally dominant)
    let aPtr = tridiagA.contents().bindMemory(to: Float.self, capacity: tridiagSize)
    let bPtr = tridiagB.contents().bindMemory(to: Float.self, capacity: tridiagSize)
    let cPtr = tridiagC.contents().bindMemory(to: Float.self, capacity: tridiagSize)
    let dPtr = tridiagD.contents().bindMemory(to: Float.self, capacity: tridiagSize)

    for i in 0..<tridiagSize {
        aPtr[i] = (i > 0) ? Float(Float.random(in: -0.1...0.1)) : Float(0.0)
        bPtr[i] = Float(Float.random(in: 0.9...1.0)) + Float(0.2)
        cPtr[i] = (i < tridiagSize - 1) ? Float(Float.random(in: -0.1...0.1)) : Float(0.0)
        dPtr[i] = Float(Float.random(in: 0.0...1.0))
    }

    var tridiagSizeVar = UInt32(tridiagSize)

    // Thomas Algorithm (sequential forward-backward sweep)
    let tridiagStart = getTimeNanos()
    for _ in 0..<tridiagIterations {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(tridiagThomasPipeline)
        encoder.setBuffer(tridiagA, offset: 0, index: 0)
        encoder.setBuffer(tridiagB, offset: 0, index: 1)
        encoder.setBuffer(tridiagC, offset: 0, index: 2)
        encoder.setBuffer(tridiagD, offset: 0, index: 3)
        encoder.setBuffer(tridiagX, offset: 0, index: 4)
        encoder.setBuffer(tridiagCp, offset: 0, index: 5)
        encoder.setBuffer(tridiagDp, offset: 0, index: 6)
        encoder.setBytes(&tridiagSizeVar, length: MemoryLayout<UInt32>.size, index: 7)
        encoder.dispatchThreads(MTLSize(width: tridiagSize, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let tridiagEnd = getTimeNanos()
    let tridiagOps = Int64(tridiagSize) * Int64(tridiagIterations)  // O(n) operations
    let tridiagGops = Double(tridiagOps) / getElapsedSeconds(start: tridiagStart, end: tridiagEnd) / 1e9

    print("Tridiagonal Solver (Thomas, 1M): \(String(format: "%.3f", tridiagGops)) GOPS")

    // ============================================================
    // 25. PREFIX SUM / SCAN ALGORITHMS
    // Tests different parallel scan strategies
    // ============================================================
    print("\n--- 25. Prefix Sum / Scan Analysis ---")

    guard let scanNaiveFunc = deepLibrary.makeFunction(name: "scan_naive"),
          let scanHillisFunc = deepLibrary.makeFunction(name: "scan_hillis_steele"),
          let scanKoggeFunc = deepLibrary.makeFunction(name: "scan_kogge_stone"),
          let scanBrentFunc = deepLibrary.makeFunction(name: "scan_brent_kung"),
          let scanNaivePipeline = try? device.makeComputePipelineState(function: scanNaiveFunc),
          let scanHillisPipeline = try? device.makeComputePipelineState(function: scanHillisFunc),
          let scanKoggePipeline = try? device.makeComputePipelineState(function: scanKoggeFunc),
          let scanBrentPipeline = try? device.makeComputePipelineState(function: scanBrentFunc) else {
        print("Failed to create scan pipelines")
        return
    }

    let scanSize = 256 * 1024  // 256K elements
    let scanIterations = 10

    guard let scanIn = device.makeBuffer(length: scanSize * MemoryLayout<Float>.size, options: .storageModeShared),
          let scanOut = device.makeBuffer(length: scanSize * MemoryLayout<Float>.size, options: .storageModeShared),
          let scanTemp = device.makeBuffer(length: scanSize * MemoryLayout<Float>.size, options: .storageModeShared) else {
        print("Failed to create scan buffers")
        return
    }

    // Initialize input
    let scanInPtr = scanIn.contents().bindMemory(to: Float.self, capacity: scanSize)
    for i in 0..<scanSize {
        scanInPtr[i] = Float(1.0)
    }

    var scanSizeVar = UInt32(scanSize)

    // Naive Scan
    let scanStartNaive = getTimeNanos()
    for _ in 0..<scanIterations {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(scanNaivePipeline)
        encoder.setBuffer(scanIn, offset: 0, index: 0)
        encoder.setBuffer(scanOut, offset: 0, index: 1)
        encoder.setBytes(&scanSizeVar, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.dispatchThreads(MTLSize(width: scanSize, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let scanEndNaive = getTimeNanos()
    let scanOpsNaive = Int64(scanSize) * Int64(scanIterations)  // n operations per iteration
    let scanGopsNaive = Double(scanOpsNaive) / getElapsedSeconds(start: scanStartNaive, end: scanEndNaive) / 1e9

    // Hillis-Steele Scan
    let scanStartHillis = getTimeNanos()
    for _ in 0..<scanIterations {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(scanHillisPipeline)
        encoder.setBuffer(scanIn, offset: 0, index: 0)
        encoder.setBuffer(scanOut, offset: 0, index: 1)
        encoder.setBuffer(scanTemp, offset: 0, index: 2)
        encoder.setBytes(&scanSizeVar, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.dispatchThreads(MTLSize(width: scanSize, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let scanEndHillis = getTimeNanos()
    let scanOpsHillis = Int64(scanSize) * Int64(scanIterations) * Int64(log2(Float(scanSize)))
    let scanGopsHillis = Double(scanOpsHillis) / getElapsedSeconds(start: scanStartHillis, end: scanEndHillis) / 1e9

    // Kogge-Stone Scan
    let scanStartKogge = getTimeNanos()
    for _ in 0..<scanIterations {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(scanKoggePipeline)
        encoder.setBuffer(scanIn, offset: 0, index: 0)
        encoder.setBuffer(scanOut, offset: 0, index: 1)
        encoder.setBuffer(scanTemp, offset: 0, index: 2)
        encoder.setBytes(&scanSizeVar, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.dispatchThreads(MTLSize(width: scanSize, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let scanEndKogge = getTimeNanos()
    let scanOpsKogge = Int64(scanSize) * Int64(scanIterations) * Int64(log2(Float(scanSize)))
    let scanGopsKogge = Double(scanOpsKogge) / getElapsedSeconds(start: scanStartKogge, end: scanEndKogge) / 1e9

    print("Scan Naive (sequential): \(String(format: "%.3f", scanGopsNaive)) GOPS")
    print("Scan Hillis-Steele (work-efficient): \(String(format: "%.3f", scanGopsHillis)) GOPS")
    print("Scan Kogge-Stone (latency-optimal): \(String(format: "%.3f", scanGopsKogge)) GOPS")

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
