import Foundation
import Metal

// MARK: - Async Operations Benchmark

let asyncShaders = """
#include <metal_stdlib>
using namespace metal;

kernel void async_add(device float* out [[buffer(0)]],
                     device const float* in [[buffer(1)]],
                     uint id [[thread_position_in_grid]]) {
    out[id] = in[id] + 1.0f;
}
"""

public struct AsyncOperationsBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    public init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
    }

    public func run() throws {
        print(String(repeating: "=", count: 70))
        print("Asynchronous Operations and Command Buffer Overlap")
        print("Async GPU operations and CPU/GPU work overlap")
        print(String(repeating: "=", count: 70))

        guard let library = try? device.makeLibrary(source: asyncShaders, options: nil),
              let asyncFunc = library.makeFunction(name: "async_add"),
              let asyncPipeline = try? device.makeComputePipelineState(function: asyncFunc) else {
            print("Failed to create async pipeline")
            return
        }

        let workSize = 1024 * 1024
        let iterations = 10

        guard let bufferA = device.makeBuffer(length: workSize * MemoryLayout<Float>.size, options: .storageModeShared),
              let bufferB = device.makeBuffer(length: workSize * MemoryLayout<Float>.size, options: .storageModeShared) else {
            return
        }

        let ptr = bufferA.contents().bindMemory(to: Float.self, capacity: workSize)
        for i in 0..<workSize {
            ptr[i] = Float(i)
        }

        print("\n--- Synchronous Execution (Baseline) ---")
        let syncStart = getTimeNanos()
        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(asyncPipeline)
            encoder.setBuffer(bufferB, offset: 0, index: 0)
            encoder.setBuffer(bufferA, offset: 0, index: 1)
            encoder.dispatchThreads(MTLSize(width: workSize, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        let syncEnd = getTimeNanos()
        let syncTime = getElapsedSeconds(start: syncStart, end: syncEnd)
        print("Synchronous total time: \(String(format: "%.2f", syncTime * 1000)) ms")
        print("Per-kernel time: \(String(format: "%.2f", syncTime / Double(iterations) * 1000)) ms")

        print("\n--- Asynchronous Execution (Non-blocking) ---")
        let asyncStart = getTimeNanos()
        var cmdBuffers: [MTLCommandBuffer] = []
        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(asyncPipeline)
            encoder.setBuffer(bufferB, offset: 0, index: 0)
            encoder.setBuffer(bufferA, offset: 0, index: 1)
            encoder.dispatchThreads(MTLSize(width: workSize, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            cmdBuffers.append(cmd)
        }

        for cmd in cmdBuffers {
            cmd.commit()
        }

        for cmd in cmdBuffers {
            cmd.waitUntilCompleted()
        }
        let asyncEnd = getTimeNanos()
        let asyncTime = getElapsedSeconds(start: asyncStart, end: asyncEnd)
        print("Asynchronous total time: \(String(format: "%.2f", asyncTime * 1000)) ms")
        print("Per-kernel time (committed together): \(String(format: "%.2f", asyncTime / Double(iterations) * 1000)) ms")

        print("\n--- Overlap Analysis ---")
        print("| Metric | Value |")
        print("|--------|-------|")
        print("| Synchronous total | \(String(format: "%.2f", syncTime * 1000)) ms |")
        print("| Asynchronous total | \(String(format: "%.2f", asyncTime * 1000)) ms |")
        let overlapRatio = syncTime / asyncTime
        print("| Overlap ratio | \(String(format: "%.2fx", overlapRatio)) |")

        print("\n--- Key Insights ---")
        print("1. Asynchronous commit allows multiple kernels to be queued")
        print("2. GPU can pipeline multiple operations efficiently")
        print("3. CPU doesn't block waiting for each kernel to complete")
        print("4. For batch operations, async can provide significant speedup")
        print("5. Use MTLCommandBuffer completion handlers for notification")
        print("6. Command buffer can be scheduled on multiple queues for parallelism")
    }
}
