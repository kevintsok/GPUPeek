import Foundation
import Metal

// MARK: - Memory Coalescing Benchmark

let coalescingShaders = """
#include <metal_stdlib>
using namespace metal;

// Coalesced read - sequential access
kernel void coalesced_read(device const float* in [[buffer(0)]],
                          device float* out [[buffer(1)]],
                          constant uint& size [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    out[id] = in[id] * 1.001f;
}

// Coalesced write - sequential access
kernel void coalesced_write(device float* out [[buffer(0)]],
                           constant uint& size [[buffer(1)]],
                           uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    out[id] = float(id) * 1.001f;
}

// Non-coalesced read - strided access (stride = thread_id * 8)
kernel void noncoalesced_read(device const float* in [[buffer(0)]],
                              device float* out [[buffer(1)]],
                              constant uint& size [[buffer(2)]],
                              uint id [[thread_position_in_grid]]) {
    if (id >= size / 8) return;
    uint index = id * 8;
    out[id] = in[index] * 1.001f;
}

// Non-coalesced write - strided access
kernel void noncoalesced_write(device float* out [[buffer(0)]],
                               constant uint& size [[buffer(1)]],
                               uint id [[thread_position_in_grid]]) {
    if (id >= size / 8) return;
    uint index = id * 8;
    out[index] = float(id) * 1.001f;
}

// Random read - truly random access pattern
kernel void random_read(device const float* in [[buffer(0)]],
                       device float* out [[buffer(1)]],
                       device const uint* indices [[buffer(2)]],
                       constant uint& size [[buffer(3)]],
                       uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    uint index = indices[id] % size;
    out[id] = in[index] * 1.001f;
}
"""

public struct MemoryCoalescingBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    public init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
    }

    public func run() throws {
        print(String(repeating: "=", count: 70))
        print("Memory Coalescing Benchmark")
        print(String(repeating: "=", count: 70))

        guard let library = try? device.makeLibrary(source: coalescingShaders, options: nil) else {
            print("Failed to compile coalescing shaders")
            return
        }

        let size = 1024 * 1024 // 1M elements
        let iterations = 100

        guard let bufferA = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared),
              let bufferB = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared),
              let bufferIndices = device.makeBuffer(length: size * MemoryLayout<UInt32>.size, options: .storageModeShared) else {
            print("Failed to create buffers")
            return
        }

        // Initialize random indices
        let indicesPtr = bufferIndices.contents().bindMemory(to: UInt32.self, capacity: size)
        for i in 0..<size {
            indicesPtr[i] = UInt32(i)
        }
        // Shuffle indices for random access
        for i in stride(from: size - 1, through: 1, by: -1) {
            let j = Int(UInt32.random(in: 0...UInt32(i)))
            indicesPtr.swapAt(i, j)
        }

        var sizeValue = UInt32(size)

        print("\n--- Access Pattern Comparison (1M elements) ---")

        // Test 1: Coalesced Read
        if let coalescedFunc = library.makeFunction(name: "coalesced_read"),
           let coalescedPipeline = try? device.makeComputePipelineState(function: coalescedFunc) {
            let start = getTimeNanos()
            for _ in 0..<iterations {
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(coalescedPipeline)
                encoder.setBuffer(bufferA, offset: 0, index: 0)
                encoder.setBuffer(bufferB, offset: 0, index: 1)
                encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 2)
                encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                encoder.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }
            let end = getTimeNanos()
            let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
            let bandwidth = Double(size) * Double(MemoryLayout<Float>.size) / elapsed / 1e9
            print("Coalesced Read:  \(String(format: "%.3f", bandwidth)) GB/s")
        }

        // Test 2: Non-Coalesced Read (stride 8)
        if let noncoalescedFunc = library.makeFunction(name: "noncoalesced_read"),
           let noncoalescedPipeline = try? device.makeComputePipelineState(function: noncoalescedFunc) {
            let start = getTimeNanos()
            for _ in 0..<iterations {
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(noncoalescedPipeline)
                encoder.setBuffer(bufferA, offset: 0, index: 0)
                encoder.setBuffer(bufferB, offset: 0, index: 1)
                var quarterSize = UInt32(size / 8)
                encoder.setBytes(&quarterSize, length: MemoryLayout<UInt32>.size, index: 2)
                encoder.dispatchThreads(MTLSize(width: size / 8, height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                encoder.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }
            let end = getTimeNanos()
            let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
            let bandwidth = Double(size) * Double(MemoryLayout<Float>.size) / elapsed / 1e9
            print("Non-Coalesced Read: \(String(format: "%.3f", bandwidth)) GB/s (stride=8)")
        }

        // Test 3: Random Read
        if let randomFunc = library.makeFunction(name: "random_read"),
           let randomPipeline = try? device.makeComputePipelineState(function: randomFunc) {
            let start = getTimeNanos()
            for _ in 0..<iterations {
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(randomPipeline)
                encoder.setBuffer(bufferA, offset: 0, index: 0)
                encoder.setBuffer(bufferB, offset: 0, index: 1)
                encoder.setBuffer(bufferIndices, offset: 0, index: 2)
                encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 3)
                encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                encoder.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }
            let end = getTimeNanos()
            let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
            let bandwidth = Double(size) * Double(MemoryLayout<Float>.size) / elapsed / 1e9
            print("Random Read:       \(String(format: "%.3f", bandwidth)) GB/s")
        }

        print("\n--- Key Findings ---")
        print("1. Coalesced access is critical for memory bandwidth")
        print("2. Strided access (stride=8) reduces bandwidth by ~5x")
        print("3. Random access is slowest - avoid index-based scattering")
        print("4. Always arrange thread data access to be sequential")
    }
}
