import Foundation
import Metal
import QuartzCore

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

// MARK: - Shader Library

let shaderSource = """
#include <metal_stdlib>
using namespace metal;

// Memory Copy Kernel
kernel void bandwidth_copy(device const float* src [[buffer(0)]],
                          device float* dst [[buffer(1)]],
                          constant uint& size [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {
    dst[id] = src[id];
}

// Memory Set Kernel
kernel void bandwidth_set(device float* dst [[buffer(0)]],
                          constant float& value [[buffer(1)]],
                          constant uint& size [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {
    dst[id] = value;
}

// Vector Add Kernel
kernel void vector_add(device const float* a [[buffer(0)]],
                      device const float* b [[buffer(1)]],
                      device float* result [[buffer(2)]],
                      constant uint& size [[buffer(3)]],
                      uint id [[thread_position_in_grid]]) {
    result[id] = a[id] + b[id];
}

// FP32 Matrix Multiply Kernel
kernel void matmul_fp32(device const float* a [[buffer(0)]],
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

// SIMD Group Test
kernel void simd_group_test(device const float* input [[buffer(0)]],
                            device float* output [[buffer(1)]],
                            constant uint& size [[buffer(2)]],
                            uint id [[thread_position_in_grid]]) {
    float val = input[id];
    val = metal::clamp(val, 0.0f, 1.0f);
    val = metal::sin(val);
    val = metal::cos(val);
    threadgroup_barrier(mem_flags::mem_none);
    output[id] = val;
}

// Trig Test
kernel void trig_test(device const float* input [[buffer(0)]],
                     device float* output [[buffer(1)]],
                     constant uint& size [[buffer(2)]],
                     uint id [[thread_position_in_grid]]) {
    float x = input[id];
    float s = metal::sin(x);
    float c = metal::cos(x);
    float t = metal::tan(x);
    output[id] = s * c + t * 0.001f;
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

    // Check GPU family support
    if device.supportsFamily(.apple7) {
        print("GPU Family: Apple 7+")
    } else if device.supportsFamily(.apple6) {
        print("GPU Family: Apple 6")
    } else if device.supportsFamily(.apple5) {
        print("GPU Family: Apple 5")
    } else {
        print("GPU Family: Apple (legacy)")
    }

    print("\n")
}

// MARK: - Bandwidth Tests

func testBandwidthCopy(device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary) throws {
    print("=== Memory Copy Bandwidth Test ===")

    guard let pipeline = library.makeFunction(name: "bandwidth_copy"),
          let computePipeline = try? device.makeComputePipelineState(function: pipeline) else {
        print("Failed to create pipeline")
        return
    }

    let bufferSize = 256 * 1024 * 1024 // 256MB
    let iterations = 100

    guard let srcBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared),
          let dstBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
        print("Failed to create buffers")
        return
    }

    // Initialize data
    let srcData = srcBuffer.contents().assumingMemoryBound(to: Float.self)
    for i in 0..<(bufferSize / MemoryLayout<Float>.size) {
        srcData[i] = Float(i)
    }

    var size = UInt32(bufferSize)

    // Warmup
    if let cmd = queue.makeCommandBuffer(),
       let encoder = cmd.makeComputeCommandEncoder() {
        encoder.setComputePipelineState(computePipeline)
        encoder.setBuffer(srcBuffer, offset: 0, index: 0)
        encoder.setBuffer(dstBuffer, offset: 0, index: 1)
        encoder.setBytes(&size, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.dispatchThreads(MTLSize(width: bufferSize / MemoryLayout<Float>.size, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }

    // Actual test
    let start = getTimeNanos()

    for _ in 0..<iterations {
        if let cmd = queue.makeCommandBuffer(),
           let encoder = cmd.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(computePipeline)
            encoder.setBuffer(srcBuffer, offset: 0, index: 0)
            encoder.setBuffer(dstBuffer, offset: 0, index: 1)
            encoder.setBytes(&size, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.dispatchThreads(MTLSize(width: bufferSize / MemoryLayout<Float>.size, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }
    }

    let end = getTimeNanos()
    let elapsed = getTimeInterval(start: start, end: end)

    let totalBytes = Double(bufferSize) * Double(iterations)
    let bandwidthGBps = (totalBytes / elapsed) / 1e9

    let bufferSizeMB = Double(bufferSize) / 1024.0 / 1024.0
    let elapsedMs = elapsed * 1000

    print("Buffer Size: \(String(format: "%.2f", bufferSizeMB)) MB")
    print("Iterations: \(iterations)")
    print("Total Time: \(String(format: "%.3f", elapsedMs)) ms")
    print("Bandwidth: \(String(format: "%.2f", bandwidthGBps)) GB/s")
    print("\n")
}

func testBandwidthSet(device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary) throws {
    print("=== Memory Set Bandwidth Test ===")

    guard let pipeline = library.makeFunction(name: "bandwidth_set"),
          let computePipeline = try? device.makeComputePipelineState(function: pipeline) else {
        print("Failed to create pipeline")
        return
    }

    let bufferSize = 256 * 1024 * 1024
    let iterations = 100
    let value: Float = 3.14159

    guard let dstBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
        print("Failed to create buffer")
        return
    }

    var size = UInt32(bufferSize)
    var val = value

    let start = getTimeNanos()

    for _ in 0..<iterations {
        if let cmd = queue.makeCommandBuffer(),
           let encoder = cmd.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(computePipeline)
            encoder.setBuffer(dstBuffer, offset: 0, index: 0)
            encoder.setBytes(&val, length: MemoryLayout<Float>.size, index: 1)
            encoder.setBytes(&size, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.dispatchThreads(MTLSize(width: bufferSize / MemoryLayout<Float>.size, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }
    }

    let end = getTimeNanos()
    let elapsed = getTimeInterval(start: start, end: end)

    let totalBytes = Double(bufferSize) * Double(iterations)
    let bandwidthGBps = (totalBytes / elapsed) / 1e9

    print("Bandwidth: \(String(format: "%.2f", bandwidthGBps)) GB/s")
    print("\n")
}

func testVectorAdd(device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary) throws {
    print("=== Vector Add Bandwidth Test ===")

    guard let pipeline = library.makeFunction(name: "vector_add"),
          let computePipeline = try? device.makeComputePipelineState(function: pipeline) else {
        print("Failed to create pipeline")
        return
    }

    let bufferSize = 64 * 1024 * 1024 // 64M elements
    let iterations = 50

    guard let aBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared),
          let bBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared),
          let resultBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
        print("Failed to create buffers")
        return
    }

    let a = aBuffer.contents().assumingMemoryBound(to: Float.self)
    let b = bBuffer.contents().assumingMemoryBound(to: Float.self)
    for i in 0..<(bufferSize / MemoryLayout<Float>.size) {
        a[i] = Float(i)
        b[i] = Float(i * 2)
    }

    var size = UInt32(bufferSize)

    let start = getTimeNanos()

    for _ in 0..<iterations {
        if let cmd = queue.makeCommandBuffer(),
           let encoder = cmd.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(computePipeline)
            encoder.setBuffer(aBuffer, offset: 0, index: 0)
            encoder.setBuffer(bBuffer, offset: 0, index: 1)
            encoder.setBuffer(resultBuffer, offset: 0, index: 2)
            encoder.setBytes(&size, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.dispatchThreads(MTLSize(width: bufferSize / MemoryLayout<Float>.size, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }
    }

    let end = getTimeNanos()
    let elapsed = getTimeInterval(start: start, end: end)

    // 2 reads + 1 write = 3 floats per element
    let totalBytes = Double(bufferSize) * Double(iterations) * 3
    let bandwidthGBps = (totalBytes / elapsed) / 1e9

    let elementsM = bufferSize / MemoryLayout<Float>.size / 1024 / 1024

    print("Elements: \(elementsM) M")
    print("Bandwidth: \(String(format: "%.2f", bandwidthGBps)) GB/s")
    print("\n")
}

// MARK: - Compute Tests

func testMatmulFP32(device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary) throws {
    print("=== FP32 Matrix Multiply Test ===")

    guard let pipeline = library.makeFunction(name: "matmul_fp32"),
          let computePipeline = try? device.makeComputePipelineState(function: pipeline) else {
        print("Failed to create pipeline")
        return
    }

    var M: UInt32 = 512
    var K: UInt32 = 512
    var N: UInt32 = 512
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
        if let cmd = queue.makeCommandBuffer(),
           let encoder = cmd.makeComputeCommandEncoder() {
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
            cmd.commit()
            cmd.waitUntilCompleted()
        }
    }

    let end = getTimeNanos()
    let elapsed = getTimeInterval(start: start, end: end)

    // M*K + K*N reads, M*N writes
    let flops = 2.0 * Double(M) * Double(K) * Double(N) * Double(iterations)
    let gflops = flops / elapsed / 1e9

    let elapsedMs = elapsed * 1000

    print("Matrix Size: \(M)x\(K)x\(N)")
    print("Iterations: \(iterations)")
    print("Time: \(String(format: "%.3f", elapsedMs)) ms")
    print("Performance: \(String(format: "%.2f", gflops)) GFLOPS")
    print("\n")
}

func testTrigFunction(device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary) throws {
    print("=== Trigonometric Function Test ===")

    guard let pipeline = library.makeFunction(name: "trig_test"),
          let computePipeline = try? device.makeComputePipelineState(function: pipeline) else {
        print("Failed to create pipeline")
        return
    }

    let bufferSize = 32 * 1024 * 1024
    let iterations = 20

    guard let inputBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared),
          let outputBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
        print("Failed to create buffers")
        return
    }

    let input = inputBuffer.contents().assumingMemoryBound(to: Float.self)
    for i in 0..<(bufferSize / MemoryLayout<Float>.size) {
        input[i] = Float(i) * 0.001
    }

    var size = UInt32(bufferSize)

    let start = getTimeNanos()

    for _ in 0..<iterations {
        if let cmd = queue.makeCommandBuffer(),
           let encoder = cmd.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(computePipeline)
            encoder.setBuffer(inputBuffer, offset: 0, index: 0)
            encoder.setBuffer(outputBuffer, offset: 0, index: 1)
            encoder.setBytes(&size, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.dispatchThreads(MTLSize(width: bufferSize / MemoryLayout<Float>.size, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }
    }

    let end = getTimeNanos()
    let elapsed = getTimeInterval(start: start, end: end)

    let totalOps = Double(bufferSize / MemoryLayout<Float>.size) * Double(iterations) * 3 // sin + cos + tan
    let gops = totalOps / elapsed / 1e9

    let elementsM = bufferSize / MemoryLayout<Float>.size / 1024 / 1024

    print("Elements: \(elementsM) M")
    print("Operations: sin + cos + tan per element")
    print("Performance: \(String(format: "%.2f", gops)) GOPS")
    print("\n")
}

// MARK: - Main

print("Apple Metal GPU Benchmark")
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

// Run bandwidth tests
do {
    try testBandwidthCopy(device: device, queue: queue, library: library)
    try testBandwidthSet(device: device, queue: queue, library: library)
    try testVectorAdd(device: device, queue: queue, library: library)
} catch {
    print("Bandwidth test error: \(error)")
}

// Run compute tests
do {
    try testMatmulFP32(device: device, queue: queue, library: library)
    try testTrigFunction(device: device, queue: queue, library: library)
} catch {
    print("Compute test error: \(error)")
}

print("Benchmark completed.")
