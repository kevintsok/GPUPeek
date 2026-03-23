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

// MARK: - Phase 2: Memory Subsystem Shader Library

let shaderSource = """
#include <metal_stdlib>
using namespace metal;

// Memory Copy - Reference kernel
kernel void memory_copy(device const float* src [[buffer(0)]],
                      device float* dst [[buffer(1)]],
                      constant uint& size [[buffer(2)]],
                      uint id [[thread_position_in_grid]]) {
    dst[id] = src[id];
}

// Memory Copy - Vectorized (float4)
kernel void memory_copy_v4(device const float4* src [[buffer(0)]],
                         device float4* dst [[buffer(1)]],
                         constant uint& count [[buffer(2)]],
                         uint id [[thread_position_in_grid]]) {
    dst[id] = src[id];
}

// Strided memory access - sequential elements with stride
kernel void strided_access(device const float* src [[buffer(0)]],
                         device float* dst [[buffer(1)]],
                         constant uint& stride [[buffer(2)]],
                         constant uint& size [[buffer(3)]],
                         uint id [[thread_position_in_grid]]) {
    uint src_idx = id * stride;
    if (src_idx < size) {
        dst[id] = src[src_idx];
    }
}

// Random-like access using hash function
kernel void hash_access(device const float* src [[buffer(0)]],
                       device float* dst [[buffer(1)]],
                       constant uint& size [[buffer(2)]],
                       uint id [[thread_position_in_grid]]) {
    // Simple hash-based "random" access pattern
    uint hash = id * 1103515245 + 12345;
    uint idx = hash % size;
    dst[id] = src[idx];
}

// Threadgroup memory copy - using shared memory
kernel void shared_copy(device const float* src [[buffer(0)]],
                      device float* dst [[buffer(1)]],
                      constant uint& size [[buffer(2)]],
                      threadgroup float* shared [[threadgroup(0)]],
                      uint id [[thread_position_in_grid]],
                      uint lid [[thread_position_in_threadgroup]]) {
    constexpr uint THREADGROUP_SIZE = 256;  // Must match dispatch

    // Load into shared memory
    uint elements_per_group = (size + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE;
    uint start = id * elements_per_group;
    uint end = min(start + elements_per_group, size);

    for (uint i = start + lid; i < end; i += THREADGROUP_SIZE) {
        shared[i] = src[i];
    }

    threadgroup_barrier(mem_flags::mem_none);

    // Write from shared memory
    for (uint i = start + lid; i < end; i += THREADGROUP_SIZE) {
        dst[i] = shared[i];
    }
}

// Fill pattern - write same value
kernel void memory_fill(device float* dst [[buffer(0)]],
                      constant float& value [[buffer(1)]],
                      constant uint& size [[buffer(2)]],
                      uint id [[thread_position_in_grid]]) {
    dst[id] = value;
}

// Read-modify-write pattern
kernel void read_modify_write(device const float* src [[buffer(0)]],
                            device float* dst [[buffer(1)]],
                            constant uint& size [[buffer(2)]],
                            uint id [[thread_position_in_grid]]) {
    float val = src[id];
    val = val * 2.0f + 1.0f;  // Simple computation
    dst[id] = val;
}

// Matrix transpose using shared memory
kernel void transpose(device const float* src [[buffer(0)]],
                    device float* dst [[buffer(1)]],
                    constant uint& width [[buffer(2)]],
                    constant uint& height [[buffer(3)]],
                    threadgroup float* shared [[threadgroup(0)]],
                    uint2 gid [[thread_position_in_grid]],
                    uint2 tid [[thread_position_in_threadgroup]]) {
    uint2 pos = gid;
    uint2 tpos = tid;

    constexpr uint TILE_SIZE = 16;
    uint src_idx = pos.y * width + pos.x;
    uint dst_idx = pos.x * height + pos.y;

    // Load tile into shared memory
    shared[tpos.y * TILE_SIZE + tpos.x] = src[src_idx];

    threadgroup_barrier(mem_flags::mem_none);

    // Write transposed tile
    dst[dst_idx] = shared[tpos.x * TILE_SIZE + tpos.y];
}

// Memory bandwidth test - burst access
kernel void burst_access(device const float4* src [[buffer(0)]],
                        device float4* dst [[buffer(1)]],
                        constant uint& count [[buffer(2)]],
                        uint id [[thread_position_in_grid]]) {
    // Coalesced access pattern - best case for GPU
    dst[id] = src[id];
}

// Atomic counter increment
kernel void atomic_inc(device atomic_uint* counters [[buffer(0)]],
                     constant uint& iterations [[buffer(1)]],
                     uint id [[thread_position_in_grid]]) {
    for (uint i = 0; i < iterations; i++) {
        atomic_fetch_add_explicit(&counters[id], 1, memory_order_relaxed);
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

    if device.supportsFamily(.apple7) {
        print("GPU Family: Apple 7+")
    } else if device.supportsFamily(.apple6) {
        print("GPU Family: Apple 6")
    }

    if #available(macOS 13.0, *) {
        print("ReadWriteTextureSupport: \(device.readWriteTextureSupport)")
    }

    print("\n")
}

// MARK: - Test: Sequential vs Strided Access

func testSequentialVsStrided(device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary) throws {
    print("=== Sequential vs Strided Memory Access ===\n")

    guard let pipeline_seq = library.makeFunction(name: "memory_copy_v4"),
          let pipeline_stride = library.makeFunction(name: "strided_access"),
          let compute_seq = try? device.makeComputePipelineState(function: pipeline_seq),
          let compute_stride = try? device.makeComputePipelineState(function: pipeline_stride) else {
        print("Failed to create pipeline")
        return
    }

    let bufferSize = 128 * 1024 * 1024 // 128MB
    let floatCount = bufferSize / MemoryLayout<Float>.size
    let float4Count = bufferSize / MemoryLayout<simd_float4>.size
    let iterations = 50

    guard let srcBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared),
          let dstBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
        return
    }

    // Initialize
    let srcData = srcBuffer.contents().assumingMemoryBound(to: Float.self)
    for i in 0..<floatCount {
        srcData[i] = Float(i)
    }

    // Sequential access (vectorized)
    var count = UInt32(float4Count)
    let start_seq = getTimeNanos()

    for _ in 0..<iterations {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(compute_seq)
        encoder.setBuffer(srcBuffer, offset: 0, index: 0)
        encoder.setBuffer(dstBuffer, offset: 0, index: 1)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.dispatchThreads(MTLSize(width: float4Count, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }

    let end_seq = getTimeNanos()
    let elapsed_seq = getTimeInterval(start: start_seq, end: end_seq)
    let bw_seq = (Double(bufferSize) * Double(iterations) / elapsed_seq) / 1e9

    // Strided access
    var stride: UInt32 = 4  // 4-element stride
    var size = UInt32(floatCount)

    let start_stride = getTimeNanos()

    for _ in 0..<iterations {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(compute_stride)
        encoder.setBuffer(srcBuffer, offset: 0, index: 0)
        encoder.setBuffer(dstBuffer, offset: 0, index: 1)
        encoder.setBytes(&stride, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.setBytes(&size, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.dispatchThreads(MTLSize(width: floatCount / Int(stride), height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }

    let end_stride = getTimeNanos()
    let elapsed_stride = getTimeInterval(start: start_stride, end: end_stride)
    let elements_accessed = Double(floatCount / Int(stride))
    let bytes_per_element = Double(MemoryLayout<Float>.size)
    let total_bytes_stride = elements_accessed * bytes_per_element * Double(iterations)
    let bw_stride = (total_bytes_stride / elapsed_stride) / 1e9

    print("Sequential (vectorized): \(String(format: "%.2f", bw_seq)) GB/s")
    print("Strided (stride=4): \(String(format: "%.2f", bw_stride)) GB/s")
    print("Strided overhead: \(String(format: "%.1f", bw_seq / bw_stride))x slower\n")
}

// MARK: - Test: Threadgroup Memory Utilization

func testThreadgroupMemory(device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary) throws {
    print("=== Threadgroup Memory Utilization ===\n")

    guard let pipeline = library.makeFunction(name: "shared_copy"),
          let compute = try? device.makeComputePipelineState(function: pipeline) else {
        print("Failed to create pipeline")
        return
    }

    let bufferSize = 64 * 1024 * 1024 // 64MB
    let floatCount = bufferSize / MemoryLayout<Float>.size
    let iterations = 30
    let threadgroupSizes = [64, 128, 256, 512, 1024]

    guard let srcBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared),
          let dstBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
        return
    }

    let srcData = srcBuffer.contents().assumingMemoryBound(to: Float.self)
    for i in 0..<floatCount {
        srcData[i] = Float(i)
    }

    var size = UInt32(floatCount)

    for tgSize in threadgroupSizes {
        let start = getTimeNanos()

        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(compute)
            encoder.setBuffer(srcBuffer, offset: 0, index: 0)
            encoder.setBuffer(dstBuffer, offset: 0, index: 1)
            encoder.setBytes(&size, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.dispatchThreadgroups(MTLSize(width: floatCount / tgSize, height: 1, depth: 1),
                                       threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }

        let end = getTimeNanos()
        let elapsed = getTimeInterval(start: start, end: end)
        let bw = (Double(bufferSize) * Double(iterations) / elapsed) / 1e9

        let sharedMemKB = device.maxThreadgroupMemoryLength / 1024
        print("Threadgroup Size: \(tgSize), Bandwidth: \(String(format: "%.2f", bw)) GB/s (Max shared: \(sharedMemKB) KB)")
    }
    print("")
}

// MARK: - Test: Memory Access Patterns

func testMemoryPatterns(device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary) throws {
    print("=== Memory Access Patterns ===\n")

    guard let pipeline_burst = library.makeFunction(name: "burst_access"),
          let pipeline_rmw = library.makeFunction(name: "read_modify_write"),
          let pipeline_fill = library.makeFunction(name: "memory_fill"),
          let compute_burst = try? device.makeComputePipelineState(function: pipeline_burst),
          let compute_rmw = try? device.makeComputePipelineState(function: pipeline_rmw),
          let compute_fill = try? device.makeComputePipelineState(function: pipeline_fill) else {
        return
    }

    let bufferSize = 128 * 1024 * 1024 // 128MB
    let float4Count = bufferSize / MemoryLayout<simd_float4>.size
    let floatCount = bufferSize / MemoryLayout<Float>.size
    let iterations = 50

    guard let srcBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared),
          let dstBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
        return
    }

    let srcData = srcBuffer.contents().assumingMemoryBound(to: Float.self)
    for i in 0..<floatCount {
        srcData[i] = Float(i)
    }

    var count4 = UInt32(float4Count)
    var count = UInt32(floatCount)
    var value: Float = 3.14159

    // Burst (coalesced) access
    let start_burst = getTimeNanos()
    for _ in 0..<iterations {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(compute_burst)
        encoder.setBuffer(srcBuffer, offset: 0, index: 0)
        encoder.setBuffer(dstBuffer, offset: 0, index: 1)
        encoder.setBytes(&count4, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.dispatchThreads(MTLSize(width: float4Count, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let end_burst = getTimeNanos()
    let bw_burst = (Double(bufferSize) * Double(iterations) / getTimeInterval(start: start_burst, end: end_burst)) / 1e9

    // Read-Modify-Write
    let start_rmw = getTimeNanos()
    for _ in 0..<iterations {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(compute_rmw)
        encoder.setBuffer(srcBuffer, offset: 0, index: 0)
        encoder.setBuffer(dstBuffer, offset: 0, index: 1)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.dispatchThreads(MTLSize(width: floatCount, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let end_rmw = getTimeNanos()
    let bw_rmw = (Double(bufferSize) * Double(iterations) / getTimeInterval(start: start_rmw, end: end_rmw)) / 1e9

    // Fill (write-only)
    let start_fill = getTimeNanos()
    for _ in 0..<iterations {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(compute_fill)
        encoder.setBuffer(dstBuffer, offset: 0, index: 0)
        encoder.setBytes(&value, length: MemoryLayout<Float>.size, index: 1)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.dispatchThreads(MTLSize(width: floatCount, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let end_fill = getTimeNanos()
    let bw_fill = (Double(bufferSize) * Double(iterations) / getTimeInterval(start: start_fill, end: end_fill)) / 1e9

    print("Burst (coalesced read): \(String(format: "%.2f", bw_burst)) GB/s")
    print("Read-Modify-Write: \(String(format: "%.2f", bw_rmw)) GB/s")
    print("Fill (write-only): \(String(format: "%.2f", bw_fill)) GB/s")
    print("\n")
}

// MARK: - Test: Matrix Transpose (Cache-sensitive)

func testMatrixTranspose(device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary) throws {
    print("=== Matrix Transpose (Cache-Efficient) ===\n")

    guard let pipeline = library.makeFunction(name: "transpose"),
          let compute = try? device.makeComputePipelineState(function: pipeline) else {
        print("Failed to create pipeline")
        return
    }

    var width: UInt32 = 4096
    var height: UInt32 = 4096
    let iterations = 20

    let srcSize = Int(width * height)
    let bufferSize = srcSize * MemoryLayout<Float>.size

    guard let srcBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared),
          let dstBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
        return
    }

    let srcData = srcBuffer.contents().assumingMemoryBound(to: Float.self)
    for i in 0..<srcSize {
        srcData[i] = Float(i)
    }

    let start = getTimeNanos()

    for _ in 0..<iterations {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(compute)
        encoder.setBuffer(srcBuffer, offset: 0, index: 0)
        encoder.setBuffer(dstBuffer, offset: 0, index: 1)
        encoder.setBytes(&width, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.setBytes(&height, length: MemoryLayout<UInt32>.size, index: 3)

        let tileSize = 16
        encoder.dispatchThreadgroups(MTLSize(width: Int(width) / tileSize, height: Int(height) / tileSize, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: tileSize, height: tileSize, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }

    let end = getTimeNanos()
    let elapsed = getTimeInterval(start: start, end: end)

    // Transpose reads and writes: 2 * width * height * sizeof(float)
    let bytes_per_float = Double(MemoryLayout<Float>.size)
    let totalBytes = Double(srcSize) * 2.0 * bytes_per_float * Double(iterations)
    let bw = (totalBytes / elapsed) / 1e9
    let time_ms = elapsed * 1000

    print("Matrix Size: \(width)x\(height)")
    print("Iterations: \(iterations)")
    print("Time: \(String(format: "%.2f", time_ms)) ms")
    print("Bandwidth: \(String(format: "%.2f", bw)) GB/s")
    print("\n")
}

// MARK: - Test: Atomic Operations

func testAtomicOperations(device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary) throws {
    print("=== Atomic Operations Performance ===\n")

    guard let pipeline = library.makeFunction(name: "atomic_inc"),
          let compute = try? device.makeComputePipelineState(function: pipeline) else {
        print("Failed to create pipeline")
        return
    }

    let numCounters = 1024
    var iterations: UInt32 = 1000
    let dispatchSize = 256

    guard let buffer = device.makeBuffer(length: numCounters * MemoryLayout<UInt32>.size, options: .storageModeShared) else {
        return
    }

    let bufferPtr = buffer.contents().assumingMemoryBound(to: UInt32.self)
    for i in 0..<numCounters {
        bufferPtr[i] = 0
    }

    let start = getTimeNanos()

    guard let cmd = queue.makeCommandBuffer(),
          let encoder = cmd.makeComputeCommandEncoder() else { return }
    encoder.setComputePipelineState(compute)
    encoder.setBuffer(buffer, offset: 0, index: 0)
    encoder.setBytes(&iterations, length: MemoryLayout<UInt32>.size, index: 1)
    encoder.dispatchThreads(MTLSize(width: numCounters, height: 1, depth: 1),
                          threadsPerThreadgroup: MTLSize(width: dispatchSize, height: 1, depth: 1))
    encoder.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()

    let end = getTimeNanos()
    let elapsed = getTimeInterval(start: start, end: end)

    let totalOps = Double(numCounters) * Double(iterations)
    let ops_per_sec = totalOps / elapsed
    let gops = ops_per_sec / 1e9

    // Verify
    var sum: UInt32 = 0
    for i in 0..<numCounters {
        sum += bufferPtr[i]
    }
    let expected = UInt32(numCounters) * iterations
    let verified = sum == expected ? "PASS" : "FAIL"

    print("Counters: \(numCounters), Iterations: \(iterations)")
    print("Time: \(String(format: "%.3f", elapsed * 1000)) ms")
    print("Throughput: \(String(format: "%.3f", gops)) GOPS")
    print("Verification: \(verified) (sum=\(sum), expected=\(expected))")
    print("\n")
}

// MARK: - Main

print("Apple Metal GPU Benchmark - Phase 2: Memory Subsystem")
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

// Memory Access Patterns
print("=== Memory Access Patterns ===")
do {
    try testSequentialVsStrided(device: device, queue: queue, library: library)
} catch {
    print("Error: \(error)")
}

// Threadgroup Memory
print("=== Threadgroup Memory ===")
do {
    try testThreadgroupMemory(device: device, queue: queue, library: library)
} catch {
    print("Error: \(error)")
}

// Memory Operations
print("=== Memory Operations ===")
do {
    try testMemoryPatterns(device: device, queue: queue, library: library)
} catch {
    print("Error: \(error)")
}

// Matrix Transpose
print("=== Matrix Operations ===")
do {
    try testMatrixTranspose(device: device, queue: queue, library: library)
} catch {
    print("Error: \(error)")
}

// Atomic Operations
print("=== Atomic Operations ===")
do {
    try testAtomicOperations(device: device, queue: queue, library: library)
} catch {
    print("Error: \(error)")
}

print("Phase 2 benchmark completed.")
