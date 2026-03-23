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
// 4. TEXTURE VS BUFFER PERFORMANCE
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

    print("\n" + String(repeating: "=", count: 60))
    print("Deep GPU Architecture Research Complete")
    print(String(repeating: "=", count: 60))
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

print("FP16 Deep Dive completed.")
