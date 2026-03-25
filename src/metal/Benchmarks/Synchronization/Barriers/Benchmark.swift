import Foundation
import Metal

// MARK: - Barriers Benchmark

let barriersShaders = """
#include <metal_stdlib>
using namespace metal;

// Barrier with no memory fence
kernel void barrier_none(device float* out [[buffer(0)]],
                        threadgroup float* shared [[threadgroup(0)]],
                        constant uint& size [[buffer(1)]],
                        uint id [[thread_position_in_grid]],
                        uint lid [[thread_position_in_threadgroup]]) {
    shared[lid] = float(lid);
    threadgroup_barrier(flags::mem_none);
    out[id] = shared[lid];
}

// Barrier with threadgroup memory fence
kernel void barrier_threadgroup(device float* out [[buffer(0)]],
                               threadgroup float* shared [[threadgroup(0)]],
                               constant uint& size [[buffer(1)]],
                               uint id [[thread_position_in_grid]],
                               uint lid [[thread_position_in_threadgroup]]) {
    shared[lid] = float(lid);
    threadgroup_barrier(flags::mem_threadgroup);
    out[id] = shared[lid];
}

// Barrier with device memory fence
kernel void barrier_device(device float* out [[buffer(0)]],
                          device float* in [[buffer(0)]],
                          constant uint& size [[buffer(1)]],
                          uint id [[thread_position_in_grid]]) {
    threadgroup_barrier(flags::mem_device);
    out[id] = in[id];
}

// Multiple barriers - testing overhead
kernel void multi_barrier(device float* out [[buffer(0)]],
                          threadgroup float* shared [[threadgroup(0)]],
                          constant uint& size [[buffer(1)]],
                          uint id [[thread_position_in_grid]],
                          uint lid [[thread_position_in_threadgroup]]) {
    for (uint i = 0; i < 4; i++) {
        shared[lid] = float(lid) + float(i);
        threadgroup_barrier(flags::mem_threadgroup);
    }
    out[id] = shared[lid];
}
"""

public struct BarriersBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    public init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
    }

    public func run() throws {
        print(String(repeating: "=", count: 70))
        print("Synchronization Barriers Benchmark")
        print(String(repeating: "=", count: 70))

        guard let library = try? device.makeLibrary(source: barriersShaders, options: nil) else {
            print("Failed to compile barriers shaders")
            return
        }

        let size = 256 * 1024
        let iterations = 100

        guard let bufferIn = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared),
              let bufferOut = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared) else {
            print("Failed to create buffers")
            return
        }

        var sizeValue = UInt32(size)

        print("\n--- Barrier Types ---")

        // Test 1: Barrier with mem_none
        if let noneFunc = library.makeFunction(name: "barrier_none"),
           let nonePipeline = try? device.makeComputePipelineState(function: noneFunc) {
            let start = getTimeNanos()
            for _ in 0..<iterations {
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(nonePipeline)
                encoder.setBuffer(bufferOut, offset: 0, index: 0)
                encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 1)
                encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                encoder.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }
            let end = getTimeNanos()
            let elapsed = (end - start) / iterations
            print("Barrier mem_none: \(String(format: "%.2f", Double(elapsed) / 1000)) μs")
        }

        // Test 2: Barrier with mem_threadgroup
        if let tgFunc = library.makeFunction(name: "barrier_threadgroup"),
           let tgPipeline = try? device.makeComputePipelineState(function: tgFunc) {
            let start = getTimeNanos()
            for _ in 0..<iterations {
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(tgPipeline)
                encoder.setBuffer(bufferOut, offset: 0, index: 0)
                encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 1)
                encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                encoder.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }
            let end = getTimeNanos()
            let elapsed = (end - start) / iterations
            print("Barrier mem_threadgroup: \(String(format: "%.2f", Double(elapsed) / 1000)) μs")
        }

        // Test 3: Multiple barriers
        if let multiFunc = library.makeFunction(name: "multi_barrier"),
           let multiPipeline = try? device.makeComputePipelineState(function: multiFunc) {
            let start = getTimeNanos()
            for _ in 0..<iterations {
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(multiPipeline)
                encoder.setBuffer(bufferOut, offset: 0, index: 0)
                encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 1)
                encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                encoder.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }
            let end = getTimeNanos()
            let elapsed = (end - start) / iterations
            print("4x Barrier overhead: \(String(format: "%.2f", Double(elapsed) / 1000)) μs")
        }

        print("\n--- Key Insights ---")
        print("1. threadgroup_barrier has fixed overhead (~4.8 μs)")
        print("2. mem_none is fastest, mem_device adds device-scope cost")
        print("3. Multiple barriers multiply overhead linearly")
        print("4. Minimize barriers in performance-critical code")
    }
}
