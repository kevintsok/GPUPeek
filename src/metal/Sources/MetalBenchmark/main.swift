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

// MARK: - Phase 5: Architecture Deep Dive Shader Library

let shaderSource = """
#include <metal_stdlib>
using namespace metal;

// Memory compression test - repeated pattern detection
kernel void compression_pattern_test(device const float* input [[buffer(0)]],
                                    device float* output [[buffer(1)]],
                                    constant uint& size [[buffer(2)]],
                                    uint id [[thread_position_in_grid]]) {
    // Write same value repeatedly - tests write compression
    output[id] = 1.0f;
}

kernel void compression_random_test(device const float* input [[buffer(0)]],
                                   device float* output [[buffer(1)]],
                                   constant uint& size [[buffer(2)]],
                                   uint id [[thread_position_in_grid]]) {
    // Write unique values - no compression possible
    output[id] = float(id);
}

// Texture tile test - simulates TBDR behavior
kernel void texture_tile_write(device float* output [[buffer(0)]],
                               constant uint2& size [[buffer(1)]],
                               constant uint& tile_size [[buffer(2)]],
                               uint2 gid [[thread_position_in_grid]]) {
    // Write in tile patterns - tests tile-based rendering
    uint tile_x = gid.x / tile_size;
    uint tile_y = gid.y / tile_size;
    uint tile_idx = tile_y * (size.x / tile_size) + tile_x;

    if (gid.x < size.x && gid.y < size.y) {
        output[gid.y * size.x + gid.x] = float(tile_idx);
    }
}

// Sequential vs random access comparison
kernel void sequential_access(device const float* src [[buffer(0)]],
                             device float* dst [[buffer(1)]],
                             constant uint& size [[buffer(2)]],
                             uint id [[thread_position_in_grid]]) {
    // Perfectly sequential access pattern
    dst[id] = src[id] * 2.0f;
}

kernel void random_access(device const float* src [[buffer(0)]],
                         device float* dst [[buffer(1)]],
                         constant uint& size [[buffer(2)]],
                         constant uint& seed [[buffer(3)]],
                         uint id [[thread_position_in_grid]]) {
    // Pseudo-random access pattern based on thread ID
    uint idx = (id * 1103515245 + seed) % size;
    dst[id] = src[idx] * 2.0f;
}

// Read-modify-write with dependency chain
kernel void read_modify_write(device float* data [[buffer(0)]],
                             constant uint& iterations [[buffer(1)]],
                             uint id [[thread_position_in_grid]]) {
    float val = data[id];
    for (uint i = 0; i < iterations; i++) {
        val = val * 1.001f + 0.0001f;
    }
    data[id] = val;
}

// Large stride access pattern
kernel void stride_access(device const float* src [[buffer(0)]],
                         device float* dst [[buffer(1)]],
                         constant uint& size [[buffer(2)]],
                         constant uint& stride [[buffer(3)]],
                         uint id [[thread_position_in_grid]]) {
    uint idx = id * stride;
    if (idx < size) {
        dst[id] = src[idx];
    }
}

// Fill with different patterns
kernel void fill_zeros(device float* output [[buffer(0)]],
                      constant uint& size [[buffer(1)]],
                      uint id [[thread_position_in_grid]]) {
    if (id < size) {
        output[id] = 0.0f;
    }
}

kernel void fill_ones(device float* output [[buffer(0)]],
                     constant uint& size [[buffer(1)]],
                     uint id [[thread_position_in_grid]]) {
    if (id < size) {
        output[id] = 1.0f;
    }
}

kernel void fill_pattern(device float* output [[buffer(0)]],
                        constant uint& size [[buffer(1)]],
                        uint id [[thread_position_in_grid]]) {
    if (id < size) {
        // Alternating pattern
        output[id] = (id % 2 == 0) ? 1.0f : -1.0f;
    }
}

// Memory bandwidth stress test - all threads active
kernel void bandwidth_stress_read(device const float* src [[buffer(0)]],
                                  device float* dst [[buffer(1)]],
                                  constant uint& size [[buffer(2)]],
                                  uint id [[thread_position_in_grid]]) {
    float val = src[id];
    dst[id] = val;
}

kernel void bandwidth_stress_write(device float* dst [[buffer(0)]],
                                  constant uint& size [[buffer(1)]],
                                  uint id [[thread_position_in_grid]]) {
    dst[id] = float(id) * 0.001f;
}

// Double-buffer style ping-pong test
kernel void ping_pong_read(device const float* src [[buffer(0)]],
                          device float* dst [[buffer(1)]],
                          constant uint& size [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {
    dst[id] = src[id];
}

// Memory latency test - single thread per group accessing different locations
kernel void latency_test_read(device const float* src [[buffer(0)]],
                             device float* dst [[buffer(1)]],
                             threadgroup float* shared [[threadgroup(0)]],
                             constant uint& iterations [[buffer(2)]],
                             uint id [[thread_position_in_grid]],
                             uint lid [[thread_position_in_threadgroup]]) {
    // Load single element per thread group
    shared[lid] = src[lid];
    threadgroup_barrier(mem_flags::mem_none);

    // Process
    float val = shared[0];
    for (uint i = 0; i < iterations; i++) {
        val = val * 1.0001f + 0.00001f;
    }

    // Store single element
    shared[0] = val;
    threadgroup_barrier(mem_flags::mem_none);
    dst[lid] = shared[0];
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
    if device.supportsFamily(.apple8) {
        print("GPU Family: Apple 8+")
    }

    // Check for texture support
    print("ReadWriteTextureSupport: \(device.readWriteTextureSupport)")
    print("Texture Support Tier: \(device.readWriteTextureSupport == .tier2 ? "Tier2" : "Tier1")")

    print("\n")
}

// MARK: - Test: Memory Compression Effects

func testMemoryCompression(device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary) throws {
    print("=== Memory Compression Effects ===")

    let bufferSizes = [1 * 1024 * 1024, 8 * 1024 * 1024, 64 * 1024 * 1024]
    let iterations = 50

    let kernels = [
        ("compression_pattern_test", "Repeated Pattern"),
        ("compression_random_test", "Random Values")
    ]

    for size in bufferSizes {
        print("Buffer Size: \(String(format: "%.1f", Double(size) / 1024 / 1024)) MB")

        for (name, desc) in kernels {
            guard let function = library.makeFunction(name: name),
                  let pipeline = try? device.makeComputePipelineState(function: function) else {
                continue
            }

            guard let inputBuffer = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared),
                  let outputBuffer = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared) else {
                continue
            }

            // Initialize input
            let input = inputBuffer.contents().assumingMemoryBound(to: Float.self)
            for i in 0..<size {
                input[i] = Float(i) * 0.001
            }

            var sz = UInt32(size)

            let start = getTimeNanos()
            for _ in 0..<iterations {
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(pipeline)
                encoder.setBuffer(inputBuffer, offset: 0, index: 0)
                encoder.setBuffer(outputBuffer, offset: 0, index: 1)
                encoder.setBytes(&sz, length: MemoryLayout<UInt32>.size, index: 2)
                encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                encoder.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }
            let end = getTimeNanos()
            let elapsed = getTimeInterval(start: start, end: end)

            let bandwidth = Double(size) * Double(iterations) * Double(MemoryLayout<Float>.size) / elapsed / 1e9
            print("  \(desc): \(String(format: "%.2f", bandwidth)) GB/s")
        }
    }
    print("")
}

// MARK: - Test: Sequential vs Random Access

func testAccessPatterns(device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary) throws {
    print("=== Access Pattern Comparison ===")

    let size = 32 * 1024 * 1024
    let iterations = 30
    let seeds: [UInt32] = [12345, 67890, 11111]

    // Sequential baseline
    guard let func_seq = library.makeFunction(name: "sequential_access"),
          let pipeline_seq = try? device.makeComputePipelineState(function: func_seq),
          let srcBuffer = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared),
          let dstBuffer = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared) else {
        return
    }

    let src = srcBuffer.contents().assumingMemoryBound(to: Float.self)
    for i in 0..<size {
        src[i] = Float(i) * 0.001
    }

    var sz = UInt32(size)

    let start_seq = getTimeNanos()
    for _ in 0..<iterations {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(pipeline_seq)
        encoder.setBuffer(srcBuffer, offset: 0, index: 0)
        encoder.setBuffer(dstBuffer, offset: 0, index: 1)
        encoder.setBytes(&sz, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let end_seq = getTimeNanos()
    let elapsed_seq = getTimeInterval(start: start_seq, end: end_seq)
    let bw_seq = Double(size) * Double(iterations) * Double(MemoryLayout<Float>.size) / elapsed_seq / 1e9
    print("Sequential Access: \(String(format: "%.2f", bw_seq)) GB/s (baseline)")

    // Random access
    for seed in seeds {
        guard let func_rand = library.makeFunction(name: "random_access"),
              let pipeline_rand = try? device.makeComputePipelineState(function: func_rand) else {
            continue
        }

        var seedVal = seed

        let start_rand = getTimeNanos()
        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(pipeline_rand)
            encoder.setBuffer(srcBuffer, offset: 0, index: 0)
            encoder.setBuffer(dstBuffer, offset: 0, index: 1)
            encoder.setBytes(&sz, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.setBytes(&seedVal, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        let end_rand = getTimeNanos()
        let elapsed_rand = getTimeInterval(start: start_rand, end: end_rand)
        let bw_rand = Double(size) * Double(iterations) * Double(MemoryLayout<Float>.size) / elapsed_rand / 1e9

        let slowdown = bw_seq / bw_rand
        print("Random Access (seed=\(seed)): \(String(format: "%.2f", bw_rand)) GB/s (\(String(format: "%.1fx", slowdown)) slower)")
    }
    print("")
}

// MARK: - Test: Stride Access

func testStrideAccess(device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary) throws {
    print("=== Stride Access Patterns ===")

    let strides: [UInt32] = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    let size = 8 * 1024 * 1024
    let iterations = 30
    let effectiveSize = size

    guard let func_stride = library.makeFunction(name: "stride_access"),
          let pipeline_stride = try? device.makeComputePipelineState(function: func_stride),
          let srcBuffer = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared),
          let dstBuffer = device.makeBuffer(length: (size / 256) * MemoryLayout<Float>.size, options: .storageModeShared) else {
        return
    }

    let src = srcBuffer.contents().assumingMemoryBound(to: Float.self)
    for i in 0..<size {
        src[i] = Float(i) * 0.001
    }

    let outputSize = size / 256
    var sz = UInt32(size)
    var stride: UInt32 = 1

    let start = getTimeNanos()
    for _ in 0..<iterations {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(pipeline_stride)
        encoder.setBuffer(srcBuffer, offset: 0, index: 0)
        encoder.setBuffer(dstBuffer, offset: 0, index: 1)
        encoder.setBytes(&sz, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.setBytes(&stride, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.dispatchThreads(MTLSize(width: outputSize, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let end = getTimeNanos()
    let elapsed = getTimeInterval(start: start, end: end)

    // Calculate actual data read
    let actualReadSize = Double(outputSize * Int(stride))
    let bandwidth = actualReadSize * Double(iterations) * Double(MemoryLayout<Float>.size) / elapsed / 1e9
    let utilization = (1.0 / Double(stride)) * 100

    print("Stride \(stride): \(String(format: "%.2f", bandwidth)) GB/s read, utilization: \(String(format: "%.1f", utilization))%")
    print("")
}

// MARK: - Test: Fill Patterns

func testFillPatterns(device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary) throws {
    print("=== Fill Pattern Performance ===")

    let size = 32 * 1024 * 1024
    let iterations = 50

    let kernels = [
        ("fill_zeros", "Zeros"),
        ("fill_ones", "Ones"),
        ("fill_pattern", "Alternating")
    ]

    for (name, desc) in kernels {
        guard let function = library.makeFunction(name: name),
              let pipeline = try? device.makeComputePipelineState(function: function),
              let outputBuffer = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared) else {
            continue
        }

        var sz = UInt32(size)

        let start = getTimeNanos()
        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(outputBuffer, offset: 0, index: 0)
            encoder.setBytes(&sz, length: MemoryLayout<UInt32>.size, index: 1)
            encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        let end = getTimeNanos()
        let elapsed = getTimeInterval(start: start, end: end)

        let bandwidth = Double(size) * Double(iterations) * Double(MemoryLayout<Float>.size) / elapsed / 1e9
        print("\(desc): \(String(format: "%.2f", bandwidth)) GB/s")
    }
    print("")
}

// MARK: - Test: Bandwidth Stress

func testBandwidthStress(device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary) throws {
    print("=== Bandwidth Stress Test ===")

    let sizes = [1, 4, 16, 64] // MB
    let iterations = 20

    guard let func_read = library.makeFunction(name: "bandwidth_stress_read"),
          let pipeline_read = try? device.makeComputePipelineState(function: func_read),
          let func_write = library.makeFunction(name: "bandwidth_stress_write"),
          let pipeline_write = try? device.makeComputePipelineState(function: func_write) else {
        return
    }

    for sizeMB in sizes {
        let size = sizeMB * 1024 * 1024

        guard let srcBuffer = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared),
              let dstBuffer = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared) else {
            continue
        }

        var sz = UInt32(size)

        // Read bandwidth
        let start_read = getTimeNanos()
        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(pipeline_read)
            encoder.setBuffer(srcBuffer, offset: 0, index: 0)
            encoder.setBuffer(dstBuffer, offset: 0, index: 1)
            encoder.setBytes(&sz, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        let end_read = getTimeNanos()
        let elapsed_read = getTimeInterval(start: start_read, end: end_read)
        let bw_read = Double(size) * Double(iterations) * Double(MemoryLayout<Float>.size) / elapsed_read / 1e9

        // Write bandwidth
        let start_write = getTimeNanos()
        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(pipeline_write)
            encoder.setBuffer(dstBuffer, offset: 0, index: 0)
            encoder.setBytes(&sz, length: MemoryLayout<UInt32>.size, index: 1)
            encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        let end_write = getTimeNanos()
        let elapsed_write = getTimeInterval(start: start_write, end: end_write)
        let bw_write = Double(size) * Double(iterations) * Double(MemoryLayout<Float>.size) / elapsed_write / 1e9

        print("Size: \(sizeMB) MB - Read: \(String(format: "%.2f", bw_read)) GB/s, Write: \(String(format: "%.2f", bw_write)) GB/s")
    }
    print("")
}

// MARK: - Test: Read-Modify-Write

func testReadModifyWrite(device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary) throws {
    print("=== Read-Modify-Write Performance ===")

    let size = 8 * 1024 * 1024
    let iterations = [1, 10, 50, 100]

    guard let function = library.makeFunction(name: "read_modify_write"),
          let pipeline = try? device.makeComputePipelineState(function: function),
          let buffer = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared) else {
        return
    }

    let data = buffer.contents().assumingMemoryBound(to: Float.self)
    for i in 0..<size {
        data[i] = Float(i) * 0.001
    }

    for iter in iterations {
        var iters = UInt32(iter)

        // Reset data
        for i in 0..<size {
            data[i] = Float(i) * 0.001
        }

        let start = getTimeNanos()
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(buffer, offset: 0, index: 0)
        encoder.setBytes(&iters, length: MemoryLayout<UInt32>.size, index: 1)
        encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        let end = getTimeNanos()
        let elapsed = getTimeInterval(start: start, end: end)

        let ops = Double(size) * Double(iter)
        let gflops = ops / elapsed / 1e9
        print("\(iter) iterations: \(String(format: "%.2f", gflops)) GFLOPS")
    }
    print("")
}

// MARK: - Test: Memory Latency

func testMemoryLatency(device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary) throws {
    print("=== Memory Latency Test ===")

    let groupSize = 256
    let iterations: [UInt32] = [1, 10, 50, 100]
    let numGroups = 1024

    guard let function = library.makeFunction(name: "latency_test_read"),
          let pipeline = try? device.makeComputePipelineState(function: function),
          let srcBuffer = device.makeBuffer(length: groupSize * MemoryLayout<Float>.size, options: .storageModeShared),
          let dstBuffer = device.makeBuffer(length: groupSize * MemoryLayout<Float>.size, options: .storageModeShared) else {
        return
    }

    let src = srcBuffer.contents().assumingMemoryBound(to: Float.self)
    for i in 0..<groupSize {
        src[i] = Float(i) * 0.001
    }

    for iter in iterations {
        var iters = UInt32(iter)

        let start = getTimeNanos()
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(srcBuffer, offset: 0, index: 0)
        encoder.setBuffer(dstBuffer, offset: 0, index: 1)
        encoder.setBytes(&iters, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.dispatchThreadgroups(MTLSize(width: numGroups, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: groupSize, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        let end = getTimeNanos()
        let elapsed = getTimeInterval(start: start, end: end)

        let totalOps = UInt64(numGroups * groupSize) * UInt64(iter)
        let latencyPerOp = elapsed * 1e9 / Double(totalOps)
        print("\(iter) compute iterations: \(String(format: "%.2f", latencyPerOp)) ns/op")
    }
    print("")
}

// MARK: - Main

print("Apple Metal GPU Benchmark - Phase 5: Architecture Deep Dive")
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

// Memory Compression
print("--- Memory Compression Effects ---")
do {
    try testMemoryCompression(device: device, queue: queue, library: library)
} catch {
    print("Error: \(error)")
}

// Access Patterns
print("--- Access Patterns ---")
do {
    try testAccessPatterns(device: device, queue: queue, library: library)
} catch {
    print("Error: \(error)")
}

// Stride Access
print("--- Stride Access ---")
do {
    try testStrideAccess(device: device, queue: queue, library: library)
} catch {
    print("Error: \(error)")
}

// Fill Patterns
print("--- Fill Patterns ---")
do {
    try testFillPatterns(device: device, queue: queue, library: library)
} catch {
    print("Error: \(error)")
}

// Bandwidth Stress
print("--- Bandwidth Stress ---")
do {
    try testBandwidthStress(device: device, queue: queue, library: library)
} catch {
    print("Error: \(error)")
}

// Read-Modify-Write
print("--- Read-Modify-Write ---")
do {
    try testReadModifyWrite(device: device, queue: queue, library: library)
} catch {
    print("Error: \(error)")
}

// Memory Latency
print("--- Memory Latency ---")
do {
    try testMemoryLatency(device: device, queue: queue, library: library)
} catch {
    print("Error: \(error)")
}

print("Phase 5 benchmark completed.")
