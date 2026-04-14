import Foundation
import Metal

// MARK: - Shared Event Synchronization Benchmark

let sharedEventShaders = """
#include <metal_stdlib>
using namespace metal;

kernel void event_add(device float* out [[buffer(0)]],
                  device const float* in [[buffer(1)]],
                  uint id [[thread_position_in_grid]]) {
    out[id] = in[id] + 1.0f;
}
"""

public struct SharedEventBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    public init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
    }

    public func run() throws {
        print(String(repeating: "=", count: 70))
        print("Shared Event Synchronization")
        print("GPU-to-CPU and GPU-to-GPU synchronization")
        print(String(repeating: "=", count: 70))

        guard let sharedEvent = device.makeSharedEvent() else {
            print("Failed to create shared event")
            return
        }

        guard let library = try? device.makeLibrary(source: sharedEventShaders, options: nil),
              let eventFunc = library.makeFunction(name: "event_add"),
              let eventPipeline = try? device.makeComputePipelineState(function: eventFunc) else {
            print("Failed to create event pipeline")
            return
        }

        let workSize = 256 * 1024

        guard let bufferA = device.makeBuffer(length: workSize * MemoryLayout<Float>.size, options: .storageModeShared),
              let bufferB = device.makeBuffer(length: workSize * MemoryLayout<Float>.size, options: .storageModeShared) else {
            return
        }

        let ptr = bufferA.contents().bindMemory(to: Float.self, capacity: workSize)
        for i in 0..<workSize {
            ptr[i] = Float(i)
        }

        print("\n--- GPU Signal and CPU Wait ---")
        guard let cmd1 = queue.makeCommandBuffer(),
              let enc1 = cmd1.makeComputeCommandEncoder() else { return }
        enc1.setComputePipelineState(eventPipeline)
        enc1.setBuffer(bufferB, offset: 0, index: 0)
        enc1.setBuffer(bufferA, offset: 0, index: 1)
        enc1.dispatchThreads(MTLSize(width: workSize, height: 1, depth: 1),
                          threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc1.endEncoding()

        cmd1.encodeSignalEvent(sharedEvent, value: 1)
        cmd1.commit()

        let cpuWaitStart = getTimeNanos()
        let signaled = sharedEvent.wait(untilSignaledValue: 1, timeoutMS: 10_000)
        let cpuWaitEnd = getTimeNanos()
        let cpuWaitTime = Double(cpuWaitEnd - cpuWaitStart) / 1e6

        print("CPU waited for GPU signal: \(String(format: "%.2f", cpuWaitTime)) ms (signaled: \(signaled))")

        print("\n--- Shared Event Features ---")
        print("| Feature | Description |")
        print("|---------|-------------|")
        print("| makeSharedEvent | Creates event visible to CPU and GPU |")
        print("| encodeSignalEvent | GPU encodes signal into command buffer |")
        print("| waitUntilSignaled | CPU blocks until event is signaled |")
        print("| getCompletedValue | Query event state without blocking |")
        print("| GPU signals, CPU waits | Pattern for batch processing notification |")

        print("\n--- Key Insights ---")
        print("1. MTLSharedEvent enables GPU-to-CPU synchronization")
        print("2. GPU signals when command buffer completes via encodeSignalEvent")
        print("3. CPU waits for GPU completion using waitUntilSignaled")
        print("4. getCompletedValue allows polling without blocking")
        print("5. Shared events work across multiple GPU command queues")
        print("6. Use for: batch completion notification, pipeline synchronization")
    }
}
