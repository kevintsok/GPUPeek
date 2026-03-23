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

// MARK: - Texture & Cache Deep Dive Shader Library

let shaderSource = """
#include <metal_stdlib>
using namespace metal;

// Sequential 2D spatial access (for comparison with texture)
kernel void buffer_2d_spatial(device const float4* src [[buffer(0)]],
                             device float4* dst [[buffer(1)]],
                             constant uint& width [[buffer(2)]],
                             constant uint& height [[buffer(3)]],
                             uint2 gid [[thread_position_in_grid]]) {
    if (gid.x < width && gid.y < height) {
        uint idx = gid.y * width + gid.x;
        dst[idx] = src[idx] * 2.0f;
    }
}

// Cache line access test - access every 64 bytes (typical cache line)
kernel void cache_line_stride(device const float4* src [[buffer(0)]],
                             device float4* dst [[buffer(1)]],
                             constant uint& size [[buffer(2)]],
                             uint id [[thread_position_in_grid]]) {
    // 16 floats = 64 bytes = 1 cache line
    uint idx = id * 16;
    if (idx < size) {
        dst[id] = src[idx];
    }
}

// Half-cache-line stride (32 bytes)
kernel void half_cache_line_stride(device const float4* src [[buffer(0)]],
                                  device float4* dst [[buffer(1)]],
                                  constant uint& size [[buffer(2)]],
                                  uint id [[thread_position_in_grid]]) {
    // 8 floats = 32 bytes = half cache line
    uint idx = id * 8;
    if (idx < size) {
        dst[id] = src[idx];
    }
}

// Quarter-cache-line stride (16 bytes)
kernel void quarter_cache_line_stride(device const float4* src [[buffer(0)]],
                                     device float4* dst [[buffer(1)]],
                                     constant uint& size [[buffer(2)]],
                                     uint id [[thread_position_in_grid]]) {
    // 4 floats = 16 bytes = quarter cache line
    uint idx = id * 4;
    if (idx < size) {
        dst[id] = src[idx];
    }
}

// Sequential baseline - no stride
kernel void sequential_baseline(device const float4* src [[buffer(0)]],
                                device float4* dst [[buffer(1)]],
                                constant uint& size [[buffer(2)]],
                                uint id [[thread_position_in_grid]]) {
    dst[id] = src[id];
}

// Write combining test - sequential vs staggered writes
kernel void write_combine_seq(device float* dst [[buffer(0)]],
                            constant uint& size [[buffer(1)]],
                            uint id [[thread_position_in_grid]]) {
    dst[id] = float(id);
}

kernel void write_combine_stagger(device float* dst [[buffer(0)]],
                                 constant uint& size [[buffer(1)]],
                                 uint id [[thread_position_in_grid]]) {
    // Staggered: write to same cache line from all threads
    uint line_base = (id / 16) * 16;
    uint offset = id % 16;
    if (line_base + offset < size) {
        dst[line_base + offset] = float(id);
    }
}

// Streamed write - all threads write to sequential locations (write combining optimal)
kernel void write_stream(device float* dst [[buffer(0)]],
                        constant uint& size [[buffer(1)]],
                        uint id [[thread_position_in_grid]]) {
    dst[id] = float(id) * 0.001f;
}

// Random write within cache line
kernel void write_random_in_line(device float* dst [[buffer(0)]],
                                 constant uint& size [[buffer(1)]],
                                 constant uint& seed [[buffer(2)]],
                                 uint id [[thread_position_in_grid]]) {
    uint line_base = (id / 16) * 16;
    uint offset = (id * 1103515245 + seed) % 16;
    if (line_base + offset < size) {
        dst[line_base + offset] = float(id);
    }
}

// Double-buffer style: alternating reads from two buffers
kernel void double_buffer_read(device const float4* srcA [[buffer(0)]],
                              device const float4* srcB [[buffer(1)]],
                              device float4* dst [[buffer(2)]],
                              constant uint& size [[buffer(3)]],
                              constant uint& flip [[buffer(4)]],
                              uint id [[thread_position_in_grid]]) {
    if (flip == 0) {
        dst[id] = srcA[id];
    } else {
        dst[id] = srcB[id];
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
    print("ReadWriteTextureSupport: \(device.readWriteTextureSupport == .tier2 ? "Tier2" : "Tier1")")
    print("")
}

// MARK: - Test: Cache Line Behavior

func testCacheLineBehavior(device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary) throws {
    print("=== Cache Line Behavior ===")

    let sizes = [64, 256, 1024, 4096, 16384] // KB
    let iterations = 30

    guard let func_seq = library.makeFunction(name: "sequential_baseline"),
          let func_full = library.makeFunction(name: "cache_line_stride"),
          let func_half = library.makeFunction(name: "half_cache_line_stride"),
          let func_quarter = library.makeFunction(name: "quarter_cache_line_stride"),
          let pipeline_seq = try? device.makeComputePipelineState(function: func_seq),
          let pipeline_full = try? device.makeComputePipelineState(function: func_full),
          let pipeline_half = try? device.makeComputePipelineState(function: func_half),
          let pipeline_quarter = try? device.makeComputePipelineState(function: func_quarter) else {
        return
    }

    print("\nCache Line Stride Test (64B = 1 cache line):")
    for sizeKB in sizes {
        let floatCount = (sizeKB * 1024) / MemoryLayout<Float>.size
        let size = floatCount / 4 // 4 floats per float4
        let bufferSize = size * MemoryLayout<SIMD4<Float>>.size

        guard let srcBuf = device.makeBuffer(length: bufferSize, options: .storageModeShared),
              let dstBuf = device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
            continue
        }

        var sz = UInt32(size)

        // Sequential baseline
        let start_seq = getTimeNanos()
        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(pipeline_seq)
            encoder.setBuffer(srcBuf, offset: 0, index: 0)
            encoder.setBuffer(dstBuf, offset: 0, index: 1)
            encoder.setBytes(&sz, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        let end_seq = getTimeNanos()
        let elapsed_sec = getElapsedSeconds(start: start_seq, end: end_seq)
        let bytes_seq = Double(bufferSize) * Double(iterations)
        let bw_seq = bytes_seq / elapsed_sec / 1e9

        // Full cache line stride (16 floats = 64B)
        let start_full = getTimeNanos()
        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(pipeline_full)
            encoder.setBuffer(srcBuf, offset: 0, index: 0)
            encoder.setBuffer(dstBuf, offset: 0, index: 1)
            encoder.setBytes(&sz, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.dispatchThreads(MTLSize(width: size / 16, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        let end_full = getTimeNanos()
        let elapsed_full = getElapsedSeconds(start: start_full, end: end_full)
        let bytes_full = Double(size * MemoryLayout<SIMD4<Float>>.size / 16) * Double(iterations)
        let bw_full = bytes_full / elapsed_full / 1e9

        print("  \(sizeKB) KB: Sequential \(String(format: "%.2f", bw_seq)) GB/s, Stride-64B \(String(format: "%.2f", bw_full)) GB/s (ratio: \(String(format: "%.1fx", bw_seq / bw_full)))")
    }
}

// MARK: - Test: Write Combining

func testWriteCombining(device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary) throws {
    print("\n=== Write Combining Test ===")

    let size = 8 * 1024 * 1024
    let iterations = 30

    guard let func_seq = library.makeFunction(name: "write_combine_seq"),
          let func_stagger = library.makeFunction(name: "write_combine_stagger"),
          let func_stream = library.makeFunction(name: "write_stream"),
          let func_random = library.makeFunction(name: "write_random_in_line"),
          let pipeline_seq = try? device.makeComputePipelineState(function: func_seq),
          let pipeline_stagger = try? device.makeComputePipelineState(function: func_stagger),
          let pipeline_stream = try? device.makeComputePipelineState(function: func_stream),
          let pipeline_random = try? device.makeComputePipelineState(function: func_random) else {
        return
    }

    guard let buf_seq = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared),
          let buf_stagger = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared),
          let buf_stream = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared),
          let buf_random = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared) else {
        return
    }

    var sz = UInt32(size)
    var seed: UInt32 = 12345

    // Sequential writes
    let start_seq = getTimeNanos()
    for _ in 0..<iterations {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(pipeline_seq)
        encoder.setBuffer(buf_seq, offset: 0, index: 0)
        encoder.setBytes(&sz, length: MemoryLayout<UInt32>.size, index: 1)
        encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let end_seq = getTimeNanos()
    let elapsed_seq = getElapsedSeconds(start: start_seq, end: end_seq)
    let bytes_seq = Double(size) * Double(iterations) * Double(MemoryLayout<Float>.size)
    let bw_seq = bytes_seq / elapsed_seq / 1e9

    // Staggered writes (all threads write to same cache line)
    let start_stagger = getTimeNanos()
    for _ in 0..<iterations {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(pipeline_stagger)
        encoder.setBuffer(buf_stagger, offset: 0, index: 0)
        encoder.setBytes(&sz, length: MemoryLayout<UInt32>.size, index: 1)
        encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let end_stagger = getTimeNanos()
    let elapsed_stagger = getElapsedSeconds(start: start_stagger, end: end_stagger)
    let bytes_stagger = Double(size) * Double(iterations) * Double(MemoryLayout<Float>.size)
    let bw_stagger = bytes_stagger / elapsed_stagger / 1e9

    // Streamed writes
    let start_stream = getTimeNanos()
    for _ in 0..<iterations {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(pipeline_stream)
        encoder.setBuffer(buf_stream, offset: 0, index: 0)
        encoder.setBytes(&sz, length: MemoryLayout<UInt32>.size, index: 1)
        encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let end_stream = getTimeNanos()
    let elapsed_stream = getElapsedSeconds(start: start_stream, end: end_stream)
    let bytes_stream = Double(size) * Double(iterations) * Double(MemoryLayout<Float>.size)
    let bw_stream = bytes_stream / elapsed_stream / 1e9

    // Random within cache line
    let start_random = getTimeNanos()
    for _ in 0..<iterations {
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(pipeline_random)
        encoder.setBuffer(buf_random, offset: 0, index: 0)
        encoder.setBytes(&sz, length: MemoryLayout<UInt32>.size, index: 1)
        encoder.setBytes(&seed, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let end_random = getTimeNanos()
    let elapsed_random = getElapsedSeconds(start: start_random, end: end_random)
    let bytes_random = Double(size) * Double(iterations) * Double(MemoryLayout<Float>.size)
    let bw_random = bytes_random / elapsed_random / 1e9

    print("  Sequential writes: \(String(format: "%.2f", bw_seq)) GB/s")
    print("  Staggered writes: \(String(format: "%.2f", bw_stagger)) GB/s")
    print("  Streamed writes: \(String(format: "%.2f", bw_stream)) GB/s")
    print("  Random in line: \(String(format: "%.2f", bw_random)) GB/s")
    print("  Staggered/Sequential: \(String(format: "%.2fx", bw_stagger / bw_seq))")
}

// MARK: - Main

print("Apple Metal GPU Benchmark - Cache & Memory Deep Dive")
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

// Run tests
do { try testCacheLineBehavior(device: device, queue: queue, library: library) } catch { print("Error: \(error)") }
do { try testWriteCombining(device: device, queue: queue, library: library) } catch { print("Error: \(error)") }

print("\nCache & Memory Deep Dive completed.")
