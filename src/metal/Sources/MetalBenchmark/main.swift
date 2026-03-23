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

func getTimeInterval(start: UInt64, end: UInt64) -> Double {
    var info = mach_timebase_info_data_t()
    mach_timebase_info(&info)
    let elapsed = end - start
    return Double(elapsed) * Double(info.numer) / Double(info.denom) / 1e9
}

// MARK: - Phase 3: Compute Throughput Shader Library

let shaderSource = """
#include <metal_stdlib>
using namespace metal;

// FP32 Matrix Multiply - Naive
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

// FP32 Matrix Multiply - Tiled with shared memory
kernel void matmul_fp32_tiled(device const float* a [[buffer(0)]],
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
        // Load tile of A
        uint aRow = cRow;
        uint aCol = kTile * TILE_SIZE + cColTile;
        if (aRow < M && aCol < K) {
            As[cRowTile * TILE_SIZE + cColTile] = a[aRow * K + aCol];
        } else {
            As[cRowTile * TILE_SIZE + cColTile] = 0.0f;
        }

        // Load tile of B
        uint bRow = kTile * TILE_SIZE + cRowTile;
        uint bCol = cCol;
        if (bRow < K && bCol < N) {
            Bs[cRowTile * TILE_SIZE + cColTile] = b[bRow * N + bCol];
        } else {
            Bs[cRowTile * TILE_SIZE + cColTile] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_none);

        // Compute tile product
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += As[cRowTile * TILE_SIZE + k] * Bs[k * TILE_SIZE + cColTile];
        }

        threadgroup_barrier(mem_flags::mem_none);
    }

    if (cRow < M && cCol < N) {
        c[cRow * N + cCol] = sum;
    }
}

// Vector dot product
kernel void dot_product(device const float* a [[buffer(0)]],
                      device const float* b [[buffer(1)]],
                      device float* result [[buffer(2)]],
                      constant uint& size [[buffer(3)]],
                      threadgroup float* shared [[threadgroup(0)]],
                      uint id [[thread_position_in_grid]],
                      uint lid [[thread_position_in_threadgroup]]) {
    constexpr uint THREADGROUP_SIZE = 256;
    shared[lid] = a[id] * b[id];
    threadgroup_barrier(mem_flags::mem_none);

    for (uint s = THREADGROUP_SIZE / 2; s > 0; s >>= 1) {
        if (lid < s) {
            shared[lid] += shared[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_none);
    }

    if (lid == 0) {
        result[0] += shared[0];
    }
}

// Fused multiply-add
kernel void fma_test(device const float* a [[buffer(0)]],
                   device const float* b [[buffer(1)]],
                   device float* c [[buffer(2)]],
                   constant uint& size [[buffer(3)]],
                   uint id [[thread_position_in_grid]]) {
    // c = a * b + c (fused)
    c[id] = fma(a[id], b[id], c[id]);
}

// SIMD sin/cos test
kernel void simd_trig(device const float* input [[buffer(0)]],
                     device float* output [[buffer(1)]],
                     constant uint& size [[buffer(2)]],
                     uint id [[thread_position_in_grid]]) {
    float x = input[id];
    float s = metal::sin(x);
    float c = metal::cos(x);
    float t = metal::tan(x);
    output[id] = s + c + t;
}

// Exponential and power
kernel void exp_pow_test(device const float* input [[buffer(0)]],
                        device float* output [[buffer(1)]],
                        constant uint& size [[buffer(2)]],
                        uint id [[thread_position_in_grid]]) {
    float x = input[id];
    float e = metal::exp(x);
    float l = metal::log(x + 1.0f);
    float p = metal::pow(x, 2.0f);
    output[id] = e + l + p;
}

// Integer operations
kernel void int_ops(device const int* a [[buffer(0)]],
                  device const int* b [[buffer(1)]],
                  device int* c [[buffer(2)]],
                  constant uint& size [[buffer(3)]],
                  uint id [[thread_position_in_grid]]) {
    int ia = a[id];
    int ib = b[id];
    c[id] = (ia + ib) * (ia - ib) + (ia ^ ib);
}

// Reduction with warp shuffle
kernel void warp_reduce(device const float* input [[buffer(0)]],
                       device float* result [[buffer(1)]],
                       constant uint& size [[buffer(2)]],
                       uint id [[thread_position_in_grid]]) {
    float val = input[id];

    // Simulate warp shuffle with thread index
    uint tid = id & 31;
    for (uint offset = 16; offset > 0; offset >>= 1) {
        uint other = id + offset;
        if (other < size && tid < offset) {
            val += input[other];
        }
    }

    if (tid == 0) {
        result[id / 32] = val;
    }
}
"""

// MARK: - Device Info

func printDeviceInfo(device: MTLDevice) {
    print("\n=== Apple Metal GPU Info ===")
    print("Device Name: \(device.name)")
    print("Max Threadgroup Memory: \(device.maxThreadgroupMemoryLength / 1024) KB")
    print("Max Threads Per Threadgroup: \(device.maxThreadsPerThreadgroup.width)")

    if device.supportsFamily(.apple7) {
        print("GPU Family: Apple 7+")
    }

    print("\n")
}

// MARK: - Test: FP32 Matrix Multiply

func testMatmulFP32(device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary) throws {
    print("=== FP32 Matrix Multiply ===")

    guard let pipeline_naive = library.makeFunction(name: "matmul_fp32_naive"),
          let pipeline_tiled = library.makeFunction(name: "matmul_fp32_tiled"),
          let compute_naive = try? device.makeComputePipelineState(function: pipeline_naive),
          let compute_tiled = try? device.makeComputePipelineState(function: pipeline_tiled) else {
        print("Failed to create pipeline")
        return
    }

    let sizes: [UInt32] = [256, 512, 1024]

    for M in sizes {
        let K = M
        let N = M

        let aSize = Int(M * K)
        let bSize = Int(K * N)
        let cSize = Int(M * N)
        let iterations = 10

        guard let aBuffer = device.makeBuffer(length: aSize * MemoryLayout<Float>.size, options: .storageModeShared),
              let bBuffer = device.makeBuffer(length: bSize * MemoryLayout<Float>.size, options: .storageModeShared),
              let cBuffer = device.makeBuffer(length: cSize * MemoryLayout<Float>.size, options: .storageModeShared) else {
            continue
        }

        let a = aBuffer.contents().assumingMemoryBound(to: Float.self)
        let b = bBuffer.contents().assumingMemoryBound(to: Float.self)
        for i in 0..<aSize { a[i] = Float(i % 256) / 256.0 }
        for i in 0..<bSize { b[i] = Float(i % 256) / 256.0 }

        var m = M, k = K, n = N

        // Naive
        let start_naive = getTimeNanos()
        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(compute_naive)
            encoder.setBuffer(aBuffer, offset: 0, index: 0)
            encoder.setBuffer(bBuffer, offset: 0, index: 1)
            encoder.setBuffer(cBuffer, offset: 0, index: 2)
            encoder.setBytes(&m, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.setBytes(&k, length: MemoryLayout<UInt32>.size, index: 4)
            encoder.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 5)
            encoder.dispatchThreads(MTLSize(width: Int(N), height: Int(M), depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        let end_naive = getTimeNanos()
        let elapsed_naive = getTimeInterval(start: start_naive, end: end_naive)
        let flops_naive = 2.0 * Double(M) * Double(K) * Double(N) * Double(iterations)
        let gflops_naive = flops_naive / elapsed_naive / 1e9

        // Tiled
        let start_tiled = getTimeNanos()
        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(compute_tiled)
            encoder.setBuffer(aBuffer, offset: 0, index: 0)
            encoder.setBuffer(bBuffer, offset: 0, index: 1)
            encoder.setBuffer(cBuffer, offset: 0, index: 2)
            encoder.setBytes(&m, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.setBytes(&k, length: MemoryLayout<UInt32>.size, index: 4)
            encoder.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 5)
            encoder.dispatchThreadgroups(MTLSize(width: Int(N) / 16, height: Int(M) / 16, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        let end_tiled = getTimeNanos()
        let elapsed_tiled = getTimeInterval(start: start_tiled, end: end_tiled)
        let flops_tiled = 2.0 * Double(M) * Double(K) * Double(N) * Double(iterations)
        let gflops_tiled = flops_tiled / elapsed_tiled / 1e9

        print("Size: \(M)x\(K)x\(N), Naive: \(String(format: "%.2f", gflops_naive)) GFLOPS, Tiled: \(String(format: "%.2f", gflops_tiled)) GFLOPS, Speedup: \(String(format: "%.1fx", gflops_tiled / gflops_naive))")
    }
    print("")
}

// MARK: - Test: FP16 vs FP32

func testFP16vsFP32(device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary) throws {
    print("=== FP16 vs FP32 Matrix Multiply ===")

    guard let pipeline_fp16 = library.makeFunction(name: "matmul_fp16_naive"),
          let pipeline_fp32 = library.makeFunction(name: "matmul_fp32_naive"),
          let compute_fp16 = try? device.makeComputePipelineState(function: pipeline_fp16),
          let compute_fp32 = try? device.makeComputePipelineState(function: pipeline_fp32) else {
        print("Failed to create pipeline")
        return
    }

    let M: UInt32 = 1024
    let K: UInt32 = 1024
    let N: UInt32 = 1024
    let iterations = 10

    let aSize = Int(M * K)
    let bSize = Int(K * N)
    let cSize = Int(M * N)

    guard let aBufferF16 = device.makeBuffer(length: aSize * MemoryLayout<UInt16>.size, options: .storageModeShared),
          let bBufferF16 = device.makeBuffer(length: bSize * MemoryLayout<UInt16>.size, options: .storageModeShared),
          let cBufferF16 = device.makeBuffer(length: cSize * MemoryLayout<UInt16>.size, options: .storageModeShared),
          let aBufferF32 = device.makeBuffer(length: aSize * MemoryLayout<Float>.size, options: .storageModeShared),
          let bBufferF32 = device.makeBuffer(length: bSize * MemoryLayout<Float>.size, options: .storageModeShared),
          let cBufferF32 = device.makeBuffer(length: cSize * MemoryLayout<Float>.size, options: .storageModeShared) else {
        return
    }

    let aF16 = aBufferF16.contents().assumingMemoryBound(to: UInt16.self)
    let bF16 = bBufferF16.contents().assumingMemoryBound(to: UInt16.self)
    for i in 0..<aSize { aF16[i] = UInt16((i % 256) << 8) }
    for i in 0..<bSize { bF16[i] = UInt16((i % 256) << 8) }

    let aF32 = aBufferF32.contents().assumingMemoryBound(to: Float.self)
    let bF32 = bBufferF32.contents().assumingMemoryBound(to: Float.self)
    for i in 0..<aSize { aF32[i] = Float(i % 256) / 256.0 }
    for i in 0..<bSize { bF32[i] = Float(i % 256) / 256.0 }

    var m = M, k = K, n = N

    // FP16
    let start_fp16 = getTimeNanos()
    for _ in 0..<iterations {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(compute_fp16)
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
    let elapsed_fp16 = getTimeInterval(start: start_fp16, end: end_fp16)
    let flops_fp16 = 2.0 * Double(M) * Double(K) * Double(N) * Double(iterations)
    let gflops_fp16 = flops_fp16 / elapsed_fp16 / 1e9

    // FP32
    let start_fp32 = getTimeNanos()
    for _ in 0..<iterations {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(compute_fp32)
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
    let elapsed_fp32 = getTimeInterval(start: start_fp32, end: end_fp32)
    let flops_fp32 = 2.0 * Double(M) * Double(K) * Double(N) * Double(iterations)
    let gflops_fp32 = flops_fp32 / elapsed_fp32 / 1e9

    print("Matrix Size: \(M)x\(K)x\(N)")
    print("FP16: \(String(format: "%.2f", gflops_fp16)) GFLOPS")
    print("FP32: \(String(format: "%.2f", gflops_fp32)) GFLOPS")
    print("FP16/FP32 Ratio: \(String(format: "%.2f", gflops_fp16 / gflops_fp32))x\n")
}

// MARK: - Test: FMA Performance

func testFMAPerformance(device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary) throws {
    print("=== FMA (Fused Multiply-Add) Performance ===")

    guard let pipeline = library.makeFunction(name: "fma_test"),
          let compute = try? device.makeComputePipelineState(function: pipeline) else {
        print("Failed to create pipeline")
        return
    }

    let bufferSize = 32 * 1024 * 1024
    let iterations = 50

    guard let aBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared),
          let bBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared),
          let cBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
        return
    }

    let a = aBuffer.contents().assumingMemoryBound(to: Float.self)
    let b = bBuffer.contents().assumingMemoryBound(to: Float.self)
    let c = cBuffer.contents().assumingMemoryBound(to: Float.self)
    for i in 0..<(bufferSize / MemoryLayout<Float>.size) {
        a[i] = Float(i) * 0.001
        b[i] = Float(i % 256) / 256.0
        c[i] = 0.0
    }

    var size = UInt32(bufferSize / MemoryLayout<Float>.size)

    let start = getTimeNanos()
    for _ in 0..<iterations {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(compute)
        encoder.setBuffer(aBuffer, offset: 0, index: 0)
        encoder.setBuffer(bBuffer, offset: 0, index: 1)
        encoder.setBuffer(cBuffer, offset: 0, index: 2)
        encoder.setBytes(&size, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.dispatchThreads(MTLSize(width: bufferSize / MemoryLayout<Float>.size, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let end = getTimeNanos()
    let elapsed = getTimeInterval(start: start, end: end)

    // 2 reads + 1 write + 1 FMA per element
    let flops = Double(bufferSize / MemoryLayout<Float>.size) * Double(iterations) * 2
    let gflops = flops / elapsed / 1e9

    print("Buffer Size: \(String(format: "%.2f", Double(bufferSize) / 1024 / 1024)) MB")
    print("Elements: \(bufferSize / MemoryLayout<Float>.size / 1024 / 1024) M")
    print("Performance: \(String(format: "%.2f", gflops)) GFLOPS\n")
}

// MARK: - Test: Trigonometric Functions

func testTrigPerformance(device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary) throws {
    print("=== Trigonometric Function Performance ===")

    guard let pipeline = library.makeFunction(name: "simd_trig"),
          let compute = try? device.makeComputePipelineState(function: pipeline) else {
        print("Failed to create pipeline")
        return
    }

    let bufferSize = 32 * 1024 * 1024
    let iterations = 20

    guard let inputBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared),
          let outputBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
        return
    }

    let input = inputBuffer.contents().assumingMemoryBound(to: Float.self)
    for i in 0..<(bufferSize / MemoryLayout<Float>.size) {
        input[i] = Float(i) * 0.001
    }

    var size = UInt32(bufferSize / MemoryLayout<Float>.size)

    let start = getTimeNanos()
    for _ in 0..<iterations {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(compute)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        encoder.setBytes(&size, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.dispatchThreads(MTLSize(width: bufferSize / MemoryLayout<Float>.size, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let end = getTimeNanos()
    let elapsed = getTimeInterval(start: start, end: end)

    // sin + cos + tan per element
    let ops = Double(bufferSize / MemoryLayout<Float>.size) * Double(iterations) * 3
    let gops = ops / elapsed / 1e9

    print("Elements: \(bufferSize / MemoryLayout<Float>.size / 1024 / 1024) M")
    print("Operations: sin + cos + tan per element")
    print("Performance: \(String(format: "%.2f", gops)) GOPS\n")
}

// MARK: - Test: Integer Operations

func testIntegerOperations(device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary) throws {
    print("=== Integer Operations Performance ===")

    guard let pipeline = library.makeFunction(name: "int_ops"),
          let compute = try? device.makeComputePipelineState(function: pipeline) else {
        print("Failed to create pipeline")
        return
    }

    let bufferSize = 32 * 1024 * 1024
    let iterations = 50

    guard let aBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared),
          let bBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared),
          let cBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
        return
    }

    let a = aBuffer.contents().assumingMemoryBound(to: Int32.self)
    let b = bBuffer.contents().assumingMemoryBound(to: Int32.self)
    for i in 0..<(bufferSize / MemoryLayout<Int32>.size) {
        a[i] = Int32(i)
        b[i] = Int32(i % 256)
    }

    var size = UInt32(bufferSize / MemoryLayout<Int32>.size)

    let start = getTimeNanos()
    for _ in 0..<iterations {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(compute)
        encoder.setBuffer(aBuffer, offset: 0, index: 0)
        encoder.setBuffer(bBuffer, offset: 0, index: 1)
        encoder.setBuffer(cBuffer, offset: 0, index: 2)
        encoder.setBytes(&size, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.dispatchThreads(MTLSize(width: bufferSize / MemoryLayout<Int32>.size, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let end = getTimeNanos()
    let elapsed = getTimeInterval(start: start, end: end)

    // 4 integer ops per element (add, sub, mul, xor)
    let ops = Double(bufferSize / MemoryLayout<Int32>.size) * Double(iterations) * 4
    let gops = ops / elapsed / 1e9

    print("Elements: \(bufferSize / MemoryLayout<Int32>.size / 1024 / 1024) M")
    print("Operations: add + sub + mul + xor per element")
    print("Performance: \(String(format: "%.2f", gops)) GOPS\n")
}

// MARK: - Main

print("Apple Metal GPU Benchmark - Phase 3: Compute Throughput")
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

// Matrix Multiply Tests
print("=== Matrix Multiply ===")
do {
    try testMatmulFP32(device: device, queue: queue, library: library)
} catch {
    print("Error: \(error)")
}

print("=== Precision Comparison ===")
do {
    try testFP16vsFP32(device: device, queue: queue, library: library)
} catch {
    print("Error: \(error)")
}

print("=== Arithmetic Operations ===")
do {
    try testFMAPerformance(device: device, queue: queue, library: library)
} catch {
    print("Error: \(error)")
}

print("=== Transcendental Functions ===")
do {
    try testTrigPerformance(device: device, queue: queue, library: library)
} catch {
    print("Error: \(error)")
}

print("=== Integer Operations ===")
do {
    try testIntegerOperations(device: device, queue: queue, library: library)
} catch {
    print("Error: \(error)")
}

print("Phase 3 benchmark completed.")
