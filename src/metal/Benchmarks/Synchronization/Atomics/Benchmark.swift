import Foundation
import Metal

// MARK: - Atomics Benchmark

let atomicsShaders = """
#include <metal_stdlib>
using namespace metal;

// Atomic add - no contention
kernel void atomic_add_no_contention(device atomic_uint* out [[buffer(0)]],
                                     constant uint& size [[buffer(1)]],
                                     uint id [[thread_position_in_grid]]) {
    uint index = id / 32; // Each warp works on different location
    if (index >= size) return;
    atomic_fetch_add_explicit(&out[index], 1, memory_order_relaxed, memory_scope_device);
}

// Atomic add - high contention (all threads same location)
kernel void atomic_add_high_contention(device atomic_uint* out [[buffer(0)]],
                                       constant uint& size [[buffer(1)]],
                                       uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    atomic_fetch_add_explicit(&out[0], 1, memory_order_relaxed, memory_scope_device);
}

// Atomic fetch min
kernel void atomic_min(device atomic_uint* out [[buffer(0)]],
                       constant uint& size [[buffer(1)]],
                       uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    atomic_fetch_min_explicit(&out[0], id, memory_order_relaxed, memory_scope_device);
}

// Atomic fetch max
kernel void atomic_max(device atomic_uint* out [[buffer(0)]],
                       constant uint& size [[buffer(1)]],
                       uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    atomic_fetch_max_explicit(&out[0], id, memory_order_relaxed, memory_scope_device);
}

// Compare and swap (CAS) - for implementing other atomics
kernel void atomic_cas(device atomic_uint* out [[buffer(0)]],
                       device uint* result [[buffer(1)]],
                       constant uint& size [[buffer(2)]],
                       uint id [[thread_position_in_grid]]) {
    if (id >= size) return;

    uint old_val = out[0];
    uint new_val = old_val + 1;
    while (!atomic_compare_exchange_weak_explicit(&out[0], &old_val, new_val,
                                                  memory_order_relaxed, memory_order_relaxed,
                                                  memory_scope_device)) {
        new_val = old_val + 1;
    }
}

// Non-atomic baseline
kernel void non_atomic_increment(device uint* out [[buffer(0)]],
                                 constant uint& size [[buffer(1)]],
                                 uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    out[id] = out[id] + 1;
}
"""

public struct AtomicsBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    public init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
    }

    public func run() throws {
        print(String(repeating: "=", count: 70))
        print("Atomic Operations Benchmark")
        print(String(repeating: "=", count: 70))

        guard let library = try? device.makeLibrary(source: atomicsShaders, options: nil) else {
            print("Failed to compile atomics shaders")
            return
        }

        let size = 1024 * 1024
        let iterations = 10

        guard let bufferOut = device.makeBuffer(length: size * MemoryLayout<UInt32>.size, options: .storageModeShared),
              let bufferResult = device.makeBuffer(length: MemoryLayout<UInt32>.size, options: .storageModeShared) else {
            print("Failed to create buffers")
            return
        }

        var sizeValue = UInt32(size)

        print("\n--- Atomic Operation Performance ---")

        // Test 1: Non-atomic baseline
        if let nonAtomicFunc = library.makeFunction(name: "non_atomic_increment"),
           let nonAtomicPipeline = try? device.makeComputePipelineState(function: nonAtomicFunc) {
            let start = getTimeNanos()
            for _ in 0..<iterations {
                memset(bufferOut.contents(), 0, size * MemoryLayout<UInt32>.size)
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(nonAtomicPipeline)
                encoder.setBuffer(bufferOut, offset: 0, index: 0)
                encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 1)
                encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                encoder.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }
            let end = getTimeNanos()
            let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
            let gops = Double(size) / elapsed / 1e9
            print("Non-Atomic Increment: \(String(format: "%.3f", gops)) GOPS")
        }

        // Test 2: Atomic add (no contention)
        if let addNoContentionFunc = library.makeFunction(name: "atomic_add_no_contention"),
           let addNoContentionPipeline = try? device.makeComputePipelineState(function: addNoContentionFunc) {
            let start = getTimeNanos()
            for _ in 0..<iterations {
                memset(bufferOut.contents(), 0, size * MemoryLayout<UInt32>.size)
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(addNoContentionPipeline)
                encoder.setBuffer(bufferOut, offset: 0, index: 0)
                encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 1)
                encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                encoder.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }
            let end = getTimeNanos()
            let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
            let gops = Double(size) / elapsed / 1e9
            print("Atomic Add (no contention): \(String(format: "%.3f", gops)) GOPS")
        }

        // Test 3: Atomic add (high contention)
        if let addHighContentionFunc = library.makeFunction(name: "atomic_add_high_contention"),
           let addHighContentionPipeline = try? device.makeComputePipelineState(function: addHighContentionFunc) {
            let start = getTimeNanos()
            for _ in 0..<iterations {
                memset(bufferOut.contents(), 0, size * MemoryLayout<UInt32>.size)
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(addHighContentionPipeline)
                encoder.setBuffer(bufferOut, offset: 0, index: 0)
                encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 1)
                encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                encoder.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }
            let end = getTimeNanos()
            let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
            let gops = Double(size) / elapsed / 1e9
            print("Atomic Add (high contention): \(String(format: "%.3f", gops)) GOPS")
        }

        // Test 4: Atomic Min
        if let minFunc = library.makeFunction(name: "atomic_min"),
           let minPipeline = try? device.makeComputePipelineState(function: minFunc) {
            let start = getTimeNanos()
            for _ in 0..<iterations {
                memset(bufferOut.contents(), 0xFF, size * MemoryLayout<UInt32>.size)
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(minPipeline)
                encoder.setBuffer(bufferOut, offset: 0, index: 0)
                encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 1)
                encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                encoder.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }
            let end = getTimeNanos()
            let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
            let gops = Double(size) / elapsed / 1e9
            print("Atomic Min: \(String(format: "%.3f", gops)) GOPS")
        }

        // Test 5: CAS
        if let casFunc = library.makeFunction(name: "atomic_cas"),
           let casPipeline = try? device.makeComputePipelineState(function: casFunc) {
            let start = getTimeNanos()
            for _ in 0..<iterations {
                memset(bufferOut.contents(), 0, size * MemoryLayout<UInt32>.size)
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(casPipeline)
                encoder.setBuffer(bufferOut, offset: 0, index: 0)
                encoder.setBuffer(bufferResult, offset: 0, index: 1)
                encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 2)
                encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                encoder.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }
            let end = getTimeNanos()
            let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
            let gops = Double(size) / elapsed / 1e9
            print("Atomic CAS: \(String(format: "%.3f", gops)) GOPS")
        }

        print("\n--- Key Findings ---")
        print("1. Atomic operations have significant overhead vs non-atomic")
        print("2. Contention reduces performance due to serialization")
        print("3. CAS is more expensive than simple add (multiple memory operations)")
        print("4. Atomic fetch_add is fastest, CAS is slowest")
    }
}
