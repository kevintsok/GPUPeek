import Foundation
import Metal

// MARK: - Stream Compaction Benchmark

let streamCompactionShaders = """
#include <metal_stdlib>
using namespace metal;

// Naive stream compaction - each thread atomically claims position
kernel void compact_naive(device const float* in [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        device atomic_uint* count [[buffer(2)]],
                        constant uint& size [[buffer(3)]],
                        uint id [[thread_position_in_grid]]) {
    if (id >= size) return;

    float val = in[id];
    if (val > 0.5f) {  // Predicate: keep values > 0.5
        uint writeIdx = atomic_fetch_add_explicit(count, 1, memory_order_relaxed, memory_scope_device);
        out[writeIdx] = val;
    }
}

// Tiled compaction - local accumulation then global merge
kernel void compact_tiled(device const float* in [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        device atomic_uint* count [[buffer(2)]],
                        constant uint& size [[buffer(3)]],
                        uint id [[thread_position_in_grid]]) {
    if (id >= size) return;

    constexpr uint TILE_SIZE = 256;
    uint tileId = id / TILE_SIZE;
    uint localId = id % TILE_SIZE;

    threadgroup uint localCount;
    threadgroup float localOut[128];  // Max 128 elements per tile

    if (localId == 0) {
        localCount = 0;
    }
    threadgroup_barrier(flags::mem_threadgroup);

    // Each thread checks predicate and writes to local storage
    float val = in[id];
    uint localWriteIdx = 0;
    if (val > 0.5f) {
        // Simple local index calculation
        localWriteIdx = atomic_fetch_add_explicit(&localCount, 1, memory_order_relaxed, memory_scope_threadgroup);
        if (localWriteIdx < 128) {
            localOut[localWriteIdx] = val;
        }
    }
    threadgroup_barrier(flags::mem_threadgroup);

    // Tile leader writes to global
    if (localId == 0 && localCount > 0) {
        uint globalOffset = atomic_fetch_add_explicit(count, localCount, memory_order_relaxed, memory_scope_device);
        for (uint i = 0; i < localCount; i++) {
            out[globalOffset + i] = localOut[i];
        }
    }
}

// Compact indices - create array of indices where predicate is true
kernel void compact_indices(device const uchar* predicate [[buffer(0)]],
                         device uint* indices [[buffer(1)]],
                         device atomic_uint* count [[buffer(2)]],
                         constant uint& size [[buffer(3)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= size) return;

    if (predicate[id] == 1) {
        uint writeIdx = atomic_fetch_add_explicit(count, 1, memory_order_relaxed, memory_scope_device);
        indices[writeIdx] = id;
    }
}

// Predicate generation kernel
kernel void generate_predicate(device const float* in [[buffer(0)]],
                             device uchar* predicate [[buffer(1)]],
                             constant uint& size [[buffer(2)]],
                             uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    predicate[id] = (in[id] > 0.5f) ? 1 : 0;
}

// Multi-predicate compaction
kernel void compact_multi_predicate(device const float* in [[buffer(0)]],
                                   device float* out [[buffer(1)]],
                                   device atomic_uint* count [[buffer(2)]],
                                   constant uint& size [[buffer(3)]],
                                   constant float& threshold [[buffer(4)]],
                                   uint id [[thread_position_in_grid]]) {
    if (id >= size) return;

    float val = in[id];
    if (val > threshold) {
        uint writeIdx = atomic_fetch_add_explicit(count, 1, memory_order_relaxed, memory_scope_device);
        out[writeIdx] = val;
    }
}

// Radix-based compaction for integer data
kernel void compact_radix(device const uint* in [[buffer(0)]],
                         device uint* out [[buffer(1)]],
                         device atomic_uint* count [[buffer(2)]],
                         constant uint& size [[buffer(3)]],
                         constant uint& bit [[buffer(4)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= size) return;

    uint val = in[id];
    if ((val & (1u << bit)) != 0) {  // Keep elements with bit set
        uint writeIdx = atomic_fetch_add_explicit(count, 1, memory_order_relaxed, memory_scope_device);
        out[writeIdx] = val;
    }
}

// Branchless compaction using select
kernel void compact_branchless(device const float* in [[buffer(0)]],
                              device float* out [[buffer(1)]],
                              device uint* valid [[buffer(2)]],
                              constant uint& size [[buffer(3)]],
                              uint id [[thread_position_in_grid]]) {
    if (id >= size) return;

    float val = in[id];
    uint pred = (val > 0.5f) ? 1 : 0;

    // Branchless: use pred to select index
    uint idx = atomic_fetch_add_explicit(&valid[0], pred, memory_order_relaxed, memory_scope_device);
    // Select with pred=0 means don't write
    out[idx * pred] = val;  // idx*0 = 0, but this still writes... need better approach
}
"""

public struct StreamCompactionBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    public init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
    }

    public func run() throws {
        print(String(repeating: "=", count: 70))
        print("Stream Compaction Benchmark")
        print(String(repeating: "=", count: 70))

        guard let library = try? device.makeLibrary(source: streamCompactionShaders, options: nil) else {
            print("Failed to compile stream compaction shaders")
            return
        }

        let sizes = [65536, 262144, 1048576]
        let keepRatio: Float = 0.3  // 30% of elements will be kept

        for size in sizes {
            print("\n--- Array Size: \(size) (~\(Int(Float(size) * keepRatio)) elements kept) ---")

            // Calculate expected output size based on data pattern
            guard let inputBuf = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared),
                  let outputBuf = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared),
                  let countBuf = device.makeBuffer(length: MemoryLayout<UInt32>.size, options: .storageModeShared) else {
                continue
            }

            // Initialize input with controlled distribution
            let inputPtr = inputBuf.contents().bindMemory(to: Float.self, capacity: size)
            for i in 0..<size {
                // Create pattern where ~30% pass predicate
                inputPtr[i] = Float(i % 10) < 3 ? Float(0.6) : Float(0.4)
            }

            var sizeValue = UInt32(size)
            let countPtr = countBuf.contents().bindMemory(to: UInt32.self, capacity: 1)

            // Test naive compaction
            if let compactFunc = library.makeFunction(name: "compact_naive"),
               let compactPipeline = try? device.makeComputePipelineState(function: compactFunc) {
                let iterations = 10
                let start = getTimeNanos()
                for _ in 0..<iterations {
                    countPtr.pointee = 0

                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }
                    encoder.setComputePipelineState(compactPipeline)
                    encoder.setBuffer(inputBuf, offset: 0, index: 0)
                    encoder.setBuffer(outputBuf, offset: 0, index: 1)
                    encoder.setBuffer(countBuf, offset: 0, index: 2)
                    encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 3)
                    encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    encoder.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()
                }
                let end = getTimeNanos()
                let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
                let throughput = Double(size) / elapsed / 1e9
                let compacted = countPtr.pointee
                print("Naive Compact: \(String(format: "%.2f", throughput)) GE/s, \(compacted) elements (\(String(format: "%.4f", elapsed * 1000)) ms)")
            }

            // Test tiled compaction
            if let tiledFunc = library.makeFunction(name: "compact_tiled"),
               let tiledPipeline = try? device.makeComputePipelineState(function: tiledFunc) {
                let iterations = 10
                let start = getTimeNanos()
                for _ in 0..<iterations {
                    countPtr.pointee = 0

                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }
                    encoder.setComputePipelineState(tiledPipeline)
                    encoder.setBuffer(inputBuf, offset: 0, index: 0)
                    encoder.setBuffer(outputBuf, offset: 0, index: 1)
                    encoder.setBuffer(countBuf, offset: 0, index: 2)
                    encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 3)
                    encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    encoder.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()
                }
                let end = getTimeNanos()
                let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
                let throughput = Double(size) / elapsed / 1e9
                let compacted = countPtr.pointee
                print("Tiled Compact: \(String(format: "%.2f", throughput)) GE/s, \(compacted) elements (\(String(format: "%.4f", elapsed * 1000)) ms)")
            }

            // Test predicate generation
            if let predFunc = library.makeFunction(name: "generate_predicate"),
               let predPipeline = try? device.makeComputePipelineState(function: predFunc) {
                let iterations = 10
                let start = getTimeNanos()
                for _ in 0..<iterations {
                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }
                    encoder.setComputePipelineState(predPipeline)
                    encoder.setBuffer(inputBuf, offset: 0, index: 0)
                    encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 2)
                    encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    encoder.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()
                }
                let end = getTimeNanos()
                let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
                let throughput = Double(size) / elapsed / 1e9
                print("Predicate Gen: \(String(format: "%.2f", throughput)) GE/s (\(String(format: "%.4f", elapsed * 1000)) ms)")
            }

            // Test multi-predicate with different threshold
            if let multiFunc = library.makeFunction(name: "compact_multi_predicate"),
               let multiPipeline = try? device.makeComputePipelineState(function: multiFunc) {
                let iterations = 10
                let start = getTimeNanos()
                var threshold = Float(0.3)
                for _ in 0..<iterations {
                    countPtr.pointee = 0

                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }
                    encoder.setComputePipelineState(multiPipeline)
                    encoder.setBuffer(inputBuf, offset: 0, index: 0)
                    encoder.setBuffer(outputBuf, offset: 0, index: 1)
                    encoder.setBuffer(countBuf, offset: 0, index: 2)
                    encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 3)
                    encoder.setBytes(&threshold, length: MemoryLayout<Float>.size, index: 4)
                    encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    encoder.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()
                }
                let end = getTimeNanos()
                let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
                let throughput = Double(size) / elapsed / 1e9
                let compacted = countPtr.pointee
                print("Multi-Predicate: \(String(format: "%.2f", throughput)) GE/s, \(compacted) elements (\(String(format: "%.4f", elapsed * 1000)) ms)")
            }
        }

        print("\n--- Key Findings ---")
        print("1. Naive compaction uses atomics for every element - high contention")
        print("2. Tiled compaction reduces atomic contention by local accumulation")
        print("3. Stream compaction is essential for filtering/selection operations")
        print("4. Predicate generation and compaction often pipelined")
        print("5. Choose compaction algorithm based on keep ratio and data size")
    }
}
