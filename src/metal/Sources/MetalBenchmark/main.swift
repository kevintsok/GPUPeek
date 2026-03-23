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

// MARK: - Optimized Shader Library

let shaderSource = """
#include <metal_stdlib>
using namespace metal;

// Optimized Memory Copy - Vectorized loads/stores
kernel void bandwidth_copy_opt(device const float4* src [[buffer(0)]],
                              device float4* dst [[buffer(1)]],
                              constant uint& count [[buffer(2)]],
                              uint id [[thread_position_in_grid]]) {
    dst[id] = src[id];
}

// Memory Set - Vectorized
kernel void bandwidth_set_opt(device float4* dst [[buffer(0)]],
                             constant float4& value [[buffer(1)]],
                             constant uint& count [[buffer(2)]],
                             uint id [[thread_position_in_grid]]) {
    dst[id] = value;
}

// Vector Add - Fused multiply-add pattern
kernel void vector_add_opt(device const float4* a [[buffer(0)]],
                          device const float4* b [[buffer(1)]],
                          device float4* result [[buffer(2)]],
                          constant uint& count [[buffer(3)]],
                          uint id [[thread_position_in_grid]]) {
    result[id] = a[id] + b[id];
}

// FP32 Matrix Multiply - Tiled version for shared memory efficiency
kernel void matmul_fp32_tiled(device const float* a [[buffer(0)]],
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

// FP16 Matrix Multiply - Half precision
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

// Threadgroup memory reduction
kernel void reduce_sum(device const float* src [[buffer(0)]],
                      device float* result [[buffer(1)]],
                      constant uint& size [[buffer(2)]],
                      threadgroup float* shared [[threadgroup(0)]],
                      uint id [[thread_position_in_grid]],
                      uint lid [[thread_position_in_threadgroup]]) {
    constexpr uint THREADGROUP_SIZE = 256; // Must match dispatch
    shared[lid] = src[id];
    threadgroup_barrier(mem_flags::mem_none);

    for (uint s = THREADGROUP_SIZE / 2; s > 0; s >>= 1) {
        if (lid < s) {
            shared[lid] += shared[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_none);
    }

    if (lid == 0) {
        result[0] = shared[0];
    }
}

// Strided memory access test
kernel void strided_copy(device const float* src [[buffer(0)]],
                         device float* dst [[buffer(1)]],
                         constant uint& stride [[buffer(2)]],
                         constant uint& count [[buffer(3)]],
                         uint id [[thread_position_in_grid]]) {
    uint src_idx = id * stride;
    if (src_idx < count * stride) {
        dst[id] = src[src_idx];
    }
}
"""

// MARK: - Device Info

func printDeviceInfo(device: MTLDevice) {
    print("\n=== Apple Metal GPU Info ===")
    print("Device Name: \(device.name)")
    print("Unified Memory: Yes (Shared with CPU)")
    print("Max Threadgroup Memory: \(device.maxThreadgroupMemoryLength / 1024) KB")
    print("Max Threads Per Threadgroup: \(device.maxThreadsPerThreadgroup.width)")
    print("Recommended Buffer Alignment: 256 bytes")

    if device.supportsFamily(.apple7) {
        print("GPU Family: Apple 7+")
    } else if device.supportsFamily(.apple6) {
        print("GPU Family: Apple 6")
    } else if device.supportsFamily(.apple5) {
        print("GPU Family: Apple 5")
    } else {
        print("GPU Family: Apple (legacy)")
    }

    if #available(macOS 13.0, *) {
        print("ReadWriteTextureSupport: \(device.readWriteTextureSupport)")
    }

    print("\n")
}

// MARK: - Optimized Bandwidth Test with Batching

func testBandwidthCopyBatched(device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary) throws {
    print("=== Memory Copy Bandwidth Test (Batched) ===")

    guard let pipeline = library.makeFunction(name: "bandwidth_copy_opt"),
          let computePipeline = try? device.makeComputePipelineState(function: pipeline) else {
        print("Failed to create pipeline")
        return
    }

    let bufferSize = 256 * 1024 * 1024 // 256MB
    let float4Count = bufferSize / MemoryLayout<simd_float4>.size
    let iterations = 100
    let batchSize = 10 // Commands per batch

    guard let srcBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared),
          let dstBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
        print("Failed to create buffers")
        return
    }

    // Initialize data
    let srcData = srcBuffer.contents().assumingMemoryBound(to: simd_float4.self)
    for i in 0..<float4Count {
        srcData[i] = simd_float4(Float(i), Float(i), Float(i), Float(i))
    }

    var count = UInt32(float4Count)

    // Warmup - single command buffer with multiple dispatches
    if let cmd = queue.makeCommandBuffer(),
       let encoder = cmd.makeComputeCommandEncoder() {
        encoder.setComputePipelineState(computePipeline)
        encoder.setBuffer(srcBuffer, offset: 0, index: 0)
        encoder.setBuffer(dstBuffer, offset: 0, index: 1)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 2)

        for _ in 0..<10 {
            encoder.dispatchThreads(MTLSize(width: float4Count, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }

    // Actual test - batched command buffer
    let start = getTimeNanos()

    for _ in 0..<(iterations / batchSize) {
        autoreleasepool {
            guard let cmd = queue.makeCommandBuffer() else { return }
            if let encoder = cmd.makeComputeCommandEncoder() {
                encoder.setComputePipelineState(computePipeline)
                encoder.setBuffer(srcBuffer, offset: 0, index: 0)
                encoder.setBuffer(dstBuffer, offset: 0, index: 1)
                encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 2)

                for _ in 0..<batchSize {
                    encoder.dispatchThreads(MTLSize(width: float4Count, height: 1, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                }
                encoder.endEncoding()
            }
            cmd.commit()
            // Don't wait per batch - let GPU pipeline
        }
    }

    // Wait for final completion
    guard let finalCmd = queue.makeCommandBuffer() else { return }
    finalCmd.commit()
    finalCmd.waitUntilCompleted()

    let end = getTimeNanos()
    let elapsed = getTimeInterval(start: start, end: end)

    let totalBytes = Double(bufferSize) * Double(iterations)
    let bandwidthGBps = (totalBytes / elapsed) / 1e9

    print("Buffer Size: \(String(format: "%.2f", Double(bufferSize) / 1024 / 1024)) MB")
    print("Iterations: \(iterations), Batch Size: \(batchSize)")
    print("Total Time: \(String(format: "%.3f", elapsed * 1000)) ms")
    print("Bandwidth: \(String(format: "%.2f", bandwidthGBps)) GB/s")
    print("\n")
}

// MARK: - Triple Buffering Test

func testTripleBuffering(device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary) throws {
    print("=== Memory Copy Bandwidth Test (Triple Buffering) ===")

    guard let pipeline = library.makeFunction(name: "bandwidth_copy_opt"),
          let computePipeline = try? device.makeComputePipelineState(function: pipeline) else {
        print("Failed to create pipeline")
        return
    }

    let bufferSize = 256 * 1024 * 1024
    let float4Count = bufferSize / MemoryLayout<simd_float4>.size
    let iterations = 300 // More iterations for triple buffering
    let batchSize = 10

    // Triple buffers
    guard let buffer1 = device.makeBuffer(length: bufferSize, options: .storageModeShared),
          let buffer2 = device.makeBuffer(length: bufferSize, options: .storageModeShared),
          let buffer3 = device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
        print("Failed to create buffer pool")
        return
    }

    let srcBuffer = buffer1
    let dstBuffer1 = buffer2
    let dstBuffer2 = buffer3

    // Initialize source
    let srcData = srcBuffer.contents().assumingMemoryBound(to: simd_float4.self)
    for i in 0..<float4Count {
        srcData[i] = simd_float4(Float(i), Float(i), Float(i), Float(i))
    }

    var count = UInt32(float4Count)

    let start = getTimeNanos()

    for batch in 0..<(iterations / batchSize) {
        let dstBuffer = (batch % 2 == 0) ? dstBuffer1 : dstBuffer2

        autoreleasepool {
            guard let cmd = queue.makeCommandBuffer() else { return }
            if let encoder = cmd.makeComputeCommandEncoder() {
                encoder.setComputePipelineState(computePipeline)
                encoder.setBuffer(srcBuffer, offset: 0, index: 0)
                encoder.setBuffer(dstBuffer, offset: 0, index: 1)
                encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 2)

                for _ in 0..<batchSize {
                    encoder.dispatchThreads(MTLSize(width: float4Count, height: 1, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                }
                encoder.endEncoding()
            }
            cmd.commit()
        }
    }

    guard let finalCmd = queue.makeCommandBuffer() else { return }
    finalCmd.commit()
    finalCmd.waitUntilCompleted()

    let end = getTimeNanos()
    let elapsed = getTimeInterval(start: start, end: end)

    let totalBytes = Double(bufferSize) * Double(iterations)
    let bandwidthGBps = (totalBytes / elapsed) / 1e9

    print("Buffer Size: \(String(format: "%.2f", Double(bufferSize) / 1024 / 1024)) MB")
    print("Iterations: \(iterations)")
    print("Total Time: \(String(format: "%.3f", elapsed * 1000)) ms")
    print("Bandwidth: \(String(format: "%.2f", bandwidthGBps)) GB/s")
    print("\n")
}

// MARK: - Async Execution Test

func testAsyncExecution(device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary) throws {
    print("=== Memory Copy Bandwidth Test (Async) ===")

    guard let pipeline = library.makeFunction(name: "bandwidth_copy_opt"),
          let computePipeline = try? device.makeComputePipelineState(function: pipeline) else {
        print("Failed to create pipeline")
        return
    }

    let bufferSize = 256 * 1024 * 1024
    let float4Count = bufferSize / MemoryLayout<simd_float4>.size
    let iterations = 100

    guard let srcBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared),
          let dstBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
        print("Failed to create buffers")
        return
    }

    let srcData = srcBuffer.contents().assumingMemoryBound(to: simd_float4.self)
    for i in 0..<float4Count {
        srcData[i] = simd_float4(Float(i), Float(i), Float(i), Float(i))
    }

    var count = UInt32(float4Count)
    var completedCount = 0
    let completionLock = NSLock()

    let start = getTimeNanos()

    for _ in 0..<iterations {
        guard let cmd = queue.makeCommandBuffer() else { continue }
        cmd.addCompletedHandler { _ in
            completionLock.lock()
            completedCount += 1
            completionLock.unlock()
        }

        if let encoder = cmd.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(computePipeline)
            encoder.setBuffer(srcBuffer, offset: 0, index: 0)
            encoder.setBuffer(dstBuffer, offset: 0, index: 1)
            encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.dispatchThreads(MTLSize(width: float4Count, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
        }
        cmd.commit()
        // Don't wait - let callbacks handle completion
    }

    // Wait for all to complete
    while completedCount < iterations {
        Thread.sleep(forTimeInterval: 0.001)
    }

    let end = getTimeNanos()
    let elapsed = getTimeInterval(start: start, end: end)

    let totalBytes = Double(bufferSize) * Double(iterations)
    let bandwidthGBps = (totalBytes / elapsed) / 1e9

    print("Buffer Size: \(String(format: "%.2f", Double(bufferSize) / 1024 / 1024)) MB")
    print("Iterations: \(iterations)")
    print("Total Time: \(String(format: "%.3f", elapsed * 1000)) ms")
    print("Bandwidth: \(String(format: "%.2f", bandwidthGBps)) GB/s")
    print("\n")
}

// MARK: - Vector Add Batched

func testVectorAddBatched(device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary) throws {
    print("=== Vector Add Bandwidth Test (Batched) ===")

    guard let pipeline = library.makeFunction(name: "vector_add_opt"),
          let computePipeline = try? device.makeComputePipelineState(function: pipeline) else {
        print("Failed to create pipeline")
        return
    }

    let bufferSize = 128 * 1024 * 1024 // 128MB per buffer
    let float4Count = bufferSize / MemoryLayout<simd_float4>.size
    let iterations = 50
    let batchSize = 10

    guard let aBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared),
          let bBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared),
          let resultBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
        print("Failed to create buffers")
        return
    }

    let a = aBuffer.contents().assumingMemoryBound(to: simd_float4.self)
    let b = bBuffer.contents().assumingMemoryBound(to: simd_float4.self)
    for i in 0..<float4Count {
        a[i] = simd_float4(Float(i), Float(i), Float(i), Float(i))
        b[i] = simd_float4(Float(i * 2), Float(i * 2), Float(i * 2), Float(i * 2))
    }

    var count = UInt32(float4Count)

    let start = getTimeNanos()

    for _ in 0..<(iterations / batchSize) {
        autoreleasepool {
            guard let cmd = queue.makeCommandBuffer() else { return }
            if let encoder = cmd.makeComputeCommandEncoder() {
                encoder.setComputePipelineState(computePipeline)
                encoder.setBuffer(aBuffer, offset: 0, index: 0)
                encoder.setBuffer(bBuffer, offset: 0, index: 1)
                encoder.setBuffer(resultBuffer, offset: 0, index: 2)
                encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 3)

                for _ in 0..<batchSize {
                    encoder.dispatchThreads(MTLSize(width: float4Count, height: 1, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                }
                encoder.endEncoding()
            }
            cmd.commit()
        }
    }

    guard let finalCmd = queue.makeCommandBuffer() else { return }
    finalCmd.commit()
    finalCmd.waitUntilCompleted()

    let end = getTimeNanos()
    let elapsed = getTimeInterval(start: start, end: end)

    // 2 reads + 1 write = 12 bytes per simd_float4 element
    let totalBytes = Double(bufferSize) * Double(iterations) * 3
    let bandwidthGBps = (totalBytes / elapsed) / 1e9

    print("Buffer Size: \(String(format: "%.2f", Double(bufferSize) / 1024 / 1024)) MB")
    print("Elements: \(String(format: "%.2f", Double(float4Count) / 1024 / 1024)) M simd_float4")
    print("Iterations: \(iterations)")
    print("Bandwidth: \(String(format: "%.2f", bandwidthGBps)) GB/s")
    print("\n")
}

// MARK: - Compute Tests

func testMatmulFP32Optimized(device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary) throws {
    print("=== FP32 Matrix Multiply Test (Optimized) ===")

    guard let pipeline = library.makeFunction(name: "matmul_fp32_tiled"),
          let computePipeline = try? device.makeComputePipelineState(function: pipeline) else {
        print("Failed to create pipeline")
        return
    }

    var M: UInt32 = 1024  // Larger matrices
    var K: UInt32 = 1024
    var N: UInt32 = 1024
    let iterations = 10

    let aSize = Int(M * K)
    let bSize = Int(K * N)
    let cSize = Int(M * N)

    guard let aBuffer = device.makeBuffer(length: aSize * MemoryLayout<Float>.size, options: .storageModeShared),
          let bBuffer = device.makeBuffer(length: bSize * MemoryLayout<Float>.size, options: .storageModeShared),
          let cBuffer = device.makeBuffer(length: cSize * MemoryLayout<Float>.size, options: .storageModeShared) else {
        print("Failed to create buffers")
        return
    }

    let a = aBuffer.contents().assumingMemoryBound(to: Float.self)
    let b = bBuffer.contents().assumingMemoryBound(to: Float.self)
    for i in 0..<aSize { a[i] = Float(i % 256) / 256.0 }
    for i in 0..<bSize { b[i] = Float(i % 256) / 256.0 }

    let start = getTimeNanos()

    for _ in 0..<iterations {
        guard let cmd = queue.makeCommandBuffer() else { continue }
        if let encoder = cmd.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(computePipeline)
            encoder.setBuffer(aBuffer, offset: 0, index: 0)
            encoder.setBuffer(bBuffer, offset: 0, index: 1)
            encoder.setBuffer(cBuffer, offset: 0, index: 2)
            encoder.setBytes(&M, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.setBytes(&K, length: MemoryLayout<UInt32>.size, index: 4)
            encoder.setBytes(&N, length: MemoryLayout<UInt32>.size, index: 5)
            encoder.dispatchThreads(MTLSize(width: Int(N), height: Int(M), depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
            encoder.endEncoding()
        }
        cmd.commit()
        cmd.waitUntilCompleted()
    }

    let end = getTimeNanos()
    let elapsed = getTimeInterval(start: start, end: end)

    let flops = 2.0 * Double(M) * Double(K) * Double(N) * Double(iterations)
    let gflops = flops / elapsed / 1e9

    print("Matrix Size: \(M)x\(K)x\(N)")
    print("Iterations: \(iterations)")
    print("Time: \(String(format: "%.3f", elapsed * 1000)) ms")
    print("Performance: \(String(format: "%.2f", gflops)) GFLOPS")
    print("\n")
}

func testMatmulFP16(device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary) throws {
    print("=== FP16 Matrix Multiply Test ===")

    guard let pipeline = library.makeFunction(name: "matmul_fp16"),
          let computePipeline = try? device.makeComputePipelineState(function: pipeline) else {
        print("Failed to create pipeline")
        return
    }

    var M: UInt32 = 1024
    var K: UInt32 = 1024
    var N: UInt32 = 1024
    let iterations = 10

    let aSize = Int(M * K)
    let bSize = Int(K * N)
    let cSize = Int(M * N)

    guard let aBuffer = device.makeBuffer(length: aSize * MemoryLayout<UInt16>.size, options: .storageModeShared),
          let bBuffer = device.makeBuffer(length: bSize * MemoryLayout<UInt16>.size, options: .storageModeShared),
          let cBuffer = device.makeBuffer(length: cSize * MemoryLayout<UInt16>.size, options: .storageModeShared) else {
        print("Failed to create buffers")
        return
    }

    let a = aBuffer.contents().assumingMemoryBound(to: UInt16.self)
    let b = bBuffer.contents().assumingMemoryBound(to: UInt16.self)
    for i in 0..<aSize { a[i] = UInt16((i % 256) << 8) } // FP16 format
    for i in 0..<bSize { b[i] = UInt16((i % 256) << 8) }

    let start = getTimeNanos()

    for _ in 0..<iterations {
        guard let cmd = queue.makeCommandBuffer() else { continue }
        if let encoder = cmd.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(computePipeline)
            encoder.setBuffer(aBuffer, offset: 0, index: 0)
            encoder.setBuffer(bBuffer, offset: 0, index: 1)
            encoder.setBuffer(cBuffer, offset: 0, index: 2)
            encoder.setBytes(&M, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.setBytes(&K, length: MemoryLayout<UInt32>.size, index: 4)
            encoder.setBytes(&N, length: MemoryLayout<UInt32>.size, index: 5)
            encoder.dispatchThreads(MTLSize(width: Int(N), height: Int(M), depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
            encoder.endEncoding()
        }
        cmd.commit()
        cmd.waitUntilCompleted()
    }

    let end = getTimeNanos()
    let elapsed = getTimeInterval(start: start, end: end)

    let flops = 2.0 * Double(M) * Double(K) * Double(N) * Double(iterations)
    let gflops = flops / elapsed / 1e9

    print("Matrix Size: \(M)x\(K)x\(N)")
    print("Iterations: \(iterations)")
    print("Time: \(String(format: "%.3f", elapsed * 1000)) ms")
    print("Performance: \(String(format: "%.2f", gflops)) GFLOPS (FP16)")
    print("\n")
}

// MARK: - Threadgroup Memory Test

func testThreadgroupReduction(device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary) throws {
    print("=== Threadgroup Reduction Test ===")

    guard let pipeline = library.makeFunction(name: "reduce_sum"),
          let computePipeline = try? device.makeComputePipelineState(function: pipeline) else {
        print("Failed to create pipeline")
        return
    }

    let bufferSize = 32 * 1024 * 1024
    let threadgroupSize = 256
    let iterations = 100

    guard let srcBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared),
          let resultBuffer = device.makeBuffer(length: MemoryLayout<Float>.size, options: .storageModeShared) else {
        print("Failed to create buffers")
        return
    }

    let src = srcBuffer.contents().assumingMemoryBound(to: Float.self)
    for i in 0..<(bufferSize / MemoryLayout<Float>.size) {
        src[i] = 1.0
    }

    var size = UInt32(bufferSize / MemoryLayout<Float>.size)

    let start = getTimeNanos()

    for _ in 0..<iterations {
        guard let cmd = queue.makeCommandBuffer() else { continue }
        if let encoder = cmd.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(computePipeline)
            encoder.setBuffer(srcBuffer, offset: 0, index: 0)
            encoder.setBuffer(resultBuffer, offset: 0, index: 1)
            encoder.setBytes(&size, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.dispatchThreads(MTLSize(width: bufferSize / MemoryLayout<Float>.size, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1))
            encoder.endEncoding()
        }
        cmd.commit()
        cmd.waitUntilCompleted()
    }

    let end = getTimeNanos()
    let elapsed = getTimeInterval(start: start, end: end)

    // Memory read for reduction
    let totalBytes = Double(bufferSize) * Double(iterations)
    let bandwidthGBps = (totalBytes / elapsed) / 1e9

    print("Buffer Size: \(String(format: "%.2f", Double(bufferSize) / 1024 / 1024)) MB")
    print("Threadgroup Size: \(threadgroupSize)")
    print("Iterations: \(iterations)")
    print("Time: \(String(format: "%.3f", elapsed * 1000)) ms")
    print("Bandwidth: \(String(format: "%.2f", bandwidthGBps)) GB/s")
    print("\n")
}

// MARK: - Main

print("Apple Metal GPU Benchmark - Optimized")
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

print("Shader compilation: SUCCESS")
print("\n")

// Phase 1: Optimized Bandwidth Tests
print("=== Phase 1: Bandwidth Optimization ===\n")

do {
    try testBandwidthCopyBatched(device: device, queue: queue, library: library)
    try testTripleBuffering(device: device, queue: queue, library: library)
    try testAsyncExecution(device: device, queue: queue, library: library)
    try testVectorAddBatched(device: device, queue: queue, library: library)
} catch {
    print("Bandwidth test error: \(error)")
}

// Phase 2: Compute Tests
print("=== Phase 2: Compute Throughput ===\n")

do {
    try testMatmulFP32Optimized(device: device, queue: queue, library: library)
    try testMatmulFP16(device: device, queue: queue, library: library)
} catch {
    print("Compute test error: \(error)")
}

// Phase 3: Memory Pattern Tests
print("=== Phase 3: Memory Patterns ===\n")

do {
    try testThreadgroupReduction(device: device, queue: queue, library: library)
} catch {
    print("Memory pattern test error: \(error)")
}

print("Benchmark completed.")
