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

// MARK: - Thread Occupancy Shader Library

let shaderSource = """
#include <metal_stdlib>
using namespace metal;

// Occupancy test - compute intensive kernel with known register usage
kernel void occupancy_test(device const float4* src [[buffer(0)]],
                         device float4* dst [[buffer(1)]],
                         constant uint& size [[buffer(2)]],
                         uint id [[thread_position_in_grid]]) {
    float4 val = src[id];

    // Force register pressure with multiple variables
    float a = val.x;
    float b = val.y;
    float c = val.z;
    float d = val.w;

    // Compute intensive - use all variables
    for (int i = 0; i < 10; i++) {
        a = a * 1.001f + 0.0001f;
        b = b * 1.001f + 0.0001f;
        c = c * 1.001f + 0.0001f;
        d = d * 1.001f + 0.0001f;
    }

    dst[id] = float4(a, b, c, d);
}

// Low register pressure kernel
kernel void low_reg_pressure(device const float4* src [[buffer(0)]],
                            device float4* dst [[buffer(1)]],
                            constant uint& size [[buffer(2)]],
                            uint id [[thread_position_in_grid]]) {
    float4 val = src[id];
    val = val * 1.001f;
    dst[id] = val;
}

// High register pressure - many live variables
kernel void high_reg_pressure(device const float4* src [[buffer(0)]],
                             device float4* dst [[buffer(1)]],
                             constant uint& size [[buffer(2)]],
                             uint id [[thread_position_in_grid]]) {
    float4 v0 = src[id];
    float4 v1 = v0 * 1.1f;
    float4 v2 = v1 * 1.2f;
    float4 v3 = v2 * 1.3f;
    float4 v4 = v3 * 1.4f;
    float4 v5 = v4 * 1.5f;
    float4 v6 = v5 * 1.6f;
    float4 v7 = v6 * 1.7f;
    float4 v8 = v7 * 1.8f;
    float4 v9 = v8 * 1.9f;

    // Mix them together
    float4 result = (v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8 + v9) * 0.1f;
    dst[id] = result;
}

// Occupancy test with shared memory
kernel void occupancy_shared_test(device const float4* src [[buffer(0)]],
                                 device float4* dst [[buffer(1)]],
                                 threadgroup float4* shared [[threadgroup(0)]],
                                 constant uint& size [[buffer(2)]],
                                 uint id [[thread_position_in_grid]],
                                 uint lid [[thread_position_in_threadgroup]]) {
    // Load into shared
    shared[lid] = src[id];
    threadgroup_barrier(mem_flags::mem_none);

    // Process
    float4 val = shared[lid];
    for (int i = 0; i < 5; i++) {
        val = val * 1.001f;
    }

    threadgroup_barrier(mem_flags::mem_none);
    shared[lid] = val;

    threadgroup_barrier(mem_flags::mem_none);
    dst[id] = shared[lid];
}

// Memory intensive kernel - fewer registers
kernel void memory_intensive(device const float4* src [[buffer(0)]],
                           device float4* dst [[buffer(1)]],
                           constant uint& size [[buffer(2)]],
                           uint id [[thread_position_in_grid]]) {
    // Simple memory operation - should achieve high occupancy
    dst[id] = src[id] * 2.0f;
}

// Compute intensive kernel - fewer threads may be better
kernel void compute_intensive(device const float4* src [[buffer(0)]],
                             device float4* dst [[buffer(1)]],
                             constant uint& size [[buffer(2)]],
                             uint id [[thread_position_in_grid]]) {
    float4 val = src[id];

    // Heavy computation
    for (int i = 0; i < 100; i++) {
        val.x = metal::sin(val.x) * metal::cos(val.x);
        val.y = metal::sin(val.y) * metal::cos(val.y);
        val.z = metal::sin(val.z) * metal::cos(val.z);
        val.w = metal::sin(val.w) * metal::cos(val.w);
    }

    dst[id] = val;
}

// Different thread group sizes for same workload
kernel void tg_64(device const float4* src [[buffer(0)]],
                  device float4* dst [[buffer(1)]],
                  constant uint& size [[buffer(2)]],
                  uint id [[thread_position_in_grid]]) {
    // All work done by small thread groups
    dst[id] = src[id] * 2.0f;
}

kernel void tg_128(device const float4* src [[buffer(0)]],
                   device float4* dst [[buffer(1)]],
                   constant uint& size [[buffer(2)]],
                   uint id [[thread_position_in_grid]]) {
    dst[id] = src[id] * 2.0f;
}

kernel void tg_256(device const float4* src [[buffer(0)]],
                   device float4* dst [[buffer(1)]],
                   constant uint& size [[buffer(2)]],
                   uint id [[thread_position_in_grid]]) {
    dst[id] = src[id] * 2.0f;
}

kernel void tg_512(device const float4* src [[buffer(0)]],
                   device float4* dst [[buffer(1)]],
                   constant uint& size [[buffer(2)]],
                   uint id [[thread_position_in_grid]]) {
    dst[id] = src[id] * 2.0f;
}

kernel void tg_1024(device const float4* src [[buffer(0)]],
                    device float4* dst [[buffer(1)]],
                    constant uint& size [[buffer(2)]],
                    uint id [[thread_position_in_grid]]) {
    dst[id] = src[id] * 2.0f;
}
"""

// MARK: - Device Info

func printDeviceInfo(device: MTLDevice) {
    print("\n=== Apple Metal GPU Info ===")
    print("Device Name: \(device.name)")
    print("Max Threadgroup Memory: \(device.maxThreadgroupMemoryLength / 1024) KB")
    print("Max Threads Per Threadgroup: \(device.maxThreadsPerThreadgroup.width)")
    if device.supportsFamily(.apple7) { print("GPU Family: Apple 7+") }
    print("")
}

// MARK: - Test: Thread Group Size Scaling

func testThreadGroupScaling(device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary) throws {
    print("=== Thread Group Size Scaling ===")

    let sizes = [64, 128, 256, 512, 1024]
    let iterations = 50

    let functions = [
        "tg_64", "tg_128", "tg_256", "tg_512", "tg_1024"
    ]

    let bufferSize = 8 * 1024 * 1024 * MemoryLayout<SIMD4<Float>>.size
    let elementCount = 8 * 1024 * 1024

    guard let srcBuf = device.makeBuffer(length: bufferSize, options: .storageModeShared),
          let dstBuf = device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
        return
    }

    print("Same workload with different thread group sizes:")

    for (i, funcName) in functions.enumerated() {
        guard let function = library.makeFunction(name: funcName),
              let pipeline = try? device.makeComputePipelineState(function: function) else {
            continue
        }

        var size = UInt32(elementCount)

        let start = getTimeNanos()
        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(srcBuf, offset: 0, index: 0)
            encoder.setBuffer(dstBuf, offset: 0, index: 1)
            encoder.setBytes(&size, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.dispatchThreads(MTLSize(width: elementCount, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: sizes[i], height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        let end = getTimeNanos()
        let elapsed = getElapsedSeconds(start: start, end: end)
        let bytes = Double(bufferSize) * Double(iterations)
        let bw = bytes / elapsed / 1e9

        print("  TG-\(String(format: "%4d", sizes[i])): \(String(format: "%.2f", bw)) GB/s")
    }
}

// MARK: - Test: Register Pressure

func testRegisterPressure(device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary) throws {
    print("\n=== Register Pressure ===")

    let iterations = 30
    let bufferSize = 4 * 1024 * 1024 * MemoryLayout<SIMD4<Float>>.size
    let elementCount = 4 * 1024 * 1024

    let kernels = [
        ("low_reg_pressure", "Low Pressure"),
        ("occupancy_test", "Medium Pressure"),
        ("high_reg_pressure", "High Pressure")
    ]

    for (name, desc) in kernels {
        guard let function = library.makeFunction(name: name),
              let pipeline = try? device.makeComputePipelineState(function: function) else {
            continue
        }

        guard let srcBuf = device.makeBuffer(length: bufferSize, options: .storageModeShared),
              let dstBuf = device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
            continue
        }

        var size = UInt32(elementCount)

        let start = getTimeNanos()
        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(srcBuf, offset: 0, index: 0)
            encoder.setBuffer(dstBuf, offset: 0, index: 1)
            encoder.setBytes(&size, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.dispatchThreads(MTLSize(width: elementCount, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        let end = getTimeNanos()
        let elapsed = getElapsedSeconds(start: start, end: end)
        let bytes = Double(bufferSize) * Double(iterations)
        let bw = bytes / elapsed / 1e9

        print("  \(desc): \(String(format: "%.2f", bw)) GB/s")
    }
}

// MARK: - Test: Occupancy vs Performance

func testOccupancyVsPerformance(device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary) throws {
    print("\n=== Occupancy vs Performance ===")

    let iterations = 30
    let bufferSize = 4 * 1024 * 1024 * MemoryLayout<SIMD4<Float>>.size
    let elementCount = 4 * 1024 * 1024

    let kernels = [
        ("memory_intensive", "Memory Intensive", 256),
        ("compute_intensive", "Compute Intensive", 256)
    ]

    for (name, desc, tgSize) in kernels {
        guard let function = library.makeFunction(name: name),
              let pipeline = try? device.makeComputePipelineState(function: function) else {
            continue
        }

        guard let srcBuf = device.makeBuffer(length: bufferSize, options: .storageModeShared),
              let dstBuf = device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
            continue
        }

        var size = UInt32(elementCount)

        let start = getTimeNanos()
        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(srcBuf, offset: 0, index: 0)
            encoder.setBuffer(dstBuf, offset: 0, index: 1)
            encoder.setBytes(&size, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.dispatchThreads(MTLSize(width: elementCount, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        let end = getTimeNanos()
        let elapsed = getElapsedSeconds(start: start, end: end)
        let bytes = Double(bufferSize) * Double(iterations)
        let bw = bytes / elapsed / 1e9

        print("  \(desc): \(String(format: "%.2f", bw)) GB/s")
    }
}

// MARK: - Test: Shared Memory Impact

func testSharedMemoryImpact(device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary) throws {
    print("\n=== Shared Memory Impact ===")

    let iterations = 30
    let bufferSize = 4 * 1024 * 1024 * MemoryLayout<SIMD4<Float>>.size
    let elementCount = 4 * 1024 * 1024

    guard let function_no_shared = library.makeFunction(name: "occupancy_test"),
          let function_shared = library.makeFunction(name: "occupancy_shared_test"),
          let pipeline_no_shared = try? device.makeComputePipelineState(function: function_no_shared),
          let pipeline_shared = try? device.makeComputePipelineState(function: function_shared) else {
        return
    }

    guard let srcBuf = device.makeBuffer(length: bufferSize, options: .storageModeShared),
          let dstBuf = device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
        return
    }

    var size = UInt32(elementCount)

    // Without shared memory
    let start_no_shared = getTimeNanos()
    for _ in 0..<iterations {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(pipeline_no_shared)
        encoder.setBuffer(srcBuf, offset: 0, index: 0)
        encoder.setBuffer(dstBuf, offset: 0, index: 1)
        encoder.setBytes(&size, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.dispatchThreads(MTLSize(width: elementCount, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let end_no_shared = getTimeNanos()
    let elapsed_no_shared = getElapsedSeconds(start: start_no_shared, end: end_no_shared)
    let bytes = Double(bufferSize) * Double(iterations)
    let bw_no_shared = bytes / elapsed_no_shared / 1e9

    // With shared memory
    let start_shared = getTimeNanos()
    for _ in 0..<iterations {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(pipeline_shared)
        encoder.setBuffer(srcBuf, offset: 0, index: 0)
        encoder.setBuffer(dstBuf, offset: 0, index: 1)
        encoder.setBytes(&size, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.dispatchThreadgroups(MTLSize(width: elementCount / 256, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let end_shared = getTimeNanos()
    let elapsed_shared = getElapsedSeconds(start: start_shared, end: end_shared)
    let bw_shared = bytes / elapsed_shared / 1e9

    print("  Without shared: \(String(format: "%.2f", bw_no_shared)) GB/s")
    print("  With shared: \(String(format: "%.2f", bw_shared)) GB/s")
    print("  Ratio: \(String(format: "%.2fx", bw_shared / bw_no_shared))")
}

// MARK: - Main

print("Apple Metal GPU Benchmark - Thread Occupancy Deep Dive")
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

do { try testThreadGroupScaling(device: device, queue: queue, library: library) } catch { print("Error: \(error)") }
do { try testRegisterPressure(device: device, queue: queue, library: library) } catch { print("Error: \(error)") }
do { try testOccupancyVsPerformance(device: device, queue: queue, library: library) } catch { print("Error: \(error)") }
do { try testSharedMemoryImpact(device: device, queue: queue, library: library) } catch { print("Error: \(error)") }

print("\nThread Occupancy Deep Dive completed.")
