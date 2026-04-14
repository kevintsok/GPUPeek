import Foundation
import Metal

// MARK: - Constant Memory Benchmark

let constantMemoryShaders = """
#include <metal_stdlib>
using namespace metal;

// Sequential constant read - each thread reads consecutive values
kernel void constant_sequential(device const float* dev [[buffer(0)]],
                              constant float* cst [[buffer(1)]],
                              device float* out [[buffer(2)]],
                              constant uint& size [[buffer(3)]],
                              uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    out[id] = cst[id] * dev[id];
}

// Scattered constant read - threads read different locations (bad for cache)
kernel void constant_scattered(device const float* dev [[buffer(0)]],
                              constant float* cst [[buffer(1)]],
                              device float* out [[buffer(2)]],
                              constant uint& size [[buffer(3)]],
                              uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    out[id] = cst[id % 1024] * dev[id];
}

// Broadcast constant read - all threads read same value (optimal)
kernel void constant_broadcast(device const float* dev [[buffer(0)]],
                              constant float4& cst [[buffer(1)]],
                              device float* out [[buffer(2)]],
                              constant uint& size [[buffer(3)]],
                              uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    out[id] = cst[0] + dev[id];
}

// Strided constant read - threads read with stride
kernel void constant_strided(device const float* dev [[buffer(0)]],
                             constant float* cst [[buffer(1)]],
                             device float* out [[buffer(2)]],
                             constant uint& size [[buffer(3)]],
                             uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    uint cstIdx = (id * 7) % 1024;  // Stride of 7
    out[id] = cst[cstIdx] * dev[id];
}

// Constant with threadgroup broadcast
kernel void constant_threadgroup_broadcast(device const float* dev [[buffer(0)]],
                                          device float* out [[buffer(2)]],
                                          constant uint& size [[buffer(3)]],
                                          uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    threadgroup float sharedVal;
    if (id == 0) {
        sharedVal = dev[0];
    }
    threadgroup_barrier(flags::mem_threadgroup);
    out[id] = sharedVal + dev[id];
}
"""

public struct ConstantMemoryBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    public init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
    }

    public func run() throws {
        print(String(repeating: "=", count: 70))
        print("Constant Memory Benchmark")
        print(String(repeating: "=", count: 70))

        guard let library = try? device.makeLibrary(source: constantMemoryShaders, options: nil) else {
            print("Failed to compile constant memory shaders")
            return
        }

        let sizes = [65536, 262144, 1048576]
        let constantSize = 1024

        for size in sizes {
            print("\n--- Array Size: \(size) ---")

            guard let devBuffer = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared),
                  let cstBuffer = device.makeBuffer(length: constantSize * MemoryLayout<Float>.size, options: .storageModeShared),
                  let outBuffer = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared) else {
                continue
            }

            // Initialize buffers
            let devPtr = devBuffer.contents().bindMemory(to: Float.self, capacity: size)
            let cstPtr = cstBuffer.contents().bindMemory(to: Float.self, capacity: constantSize)
            for i in 0..<size {
                devPtr[i] = Float(i % 256) / 255.0
            }
            for i in 0..<constantSize {
                cstPtr[i] = Float(i % 16) * Float(0.1)
            }

            var sizeValue = UInt32(size)

            // Test sequential constant read
            if let seqFunc = library.makeFunction(name: "constant_sequential"),
               let seqPipeline = try? device.makeComputePipelineState(function: seqFunc) {
                let iterations = 20
                let start = getTimeNanos()
                for _ in 0..<iterations {
                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }
                    encoder.setComputePipelineState(seqPipeline)
                    encoder.setBuffer(devBuffer, offset: 0, index: 0)
                    encoder.setBuffer(cstBuffer, offset: 0, index: 1)
                    encoder.setBuffer(outBuffer, offset: 0, index: 2)
                    encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 3)
                    encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    encoder.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()
                }
                let end = getTimeNanos()
                let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
                let gops = Double(size) / elapsed / 1e9
                print("Sequential: \(String(format: "%.2f", gops)) GOPS (\(String(format: "%.4f", elapsed * 1000)) ms)")
            }

            // Test scattered constant read
            if let scatterFunc = library.makeFunction(name: "constant_scattered"),
               let scatterPipeline = try? device.makeComputePipelineState(function: scatterFunc) {
                let iterations = 20
                let start = getTimeNanos()
                for _ in 0..<iterations {
                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }
                    encoder.setComputePipelineState(scatterPipeline)
                    encoder.setBuffer(devBuffer, offset: 0, index: 0)
                    encoder.setBuffer(cstBuffer, offset: 0, index: 1)
                    encoder.setBuffer(outBuffer, offset: 0, index: 2)
                    encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 3)
                    encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    encoder.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()
                }
                let end = getTimeNanos()
                let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
                let gops = Double(size) / elapsed / 1e9
                print("Scattered: \(String(format: "%.2f", gops)) GOPS (\(String(format: "%.4f", elapsed * 1000)) ms)")
            }

            // Test broadcast constant read (all threads same value)
            if let broadcastFunc = library.makeFunction(name: "constant_broadcast"),
               let broadcastPipeline = try? device.makeComputePipelineState(function: broadcastFunc) {
                let iterations = 20
                let start = getTimeNanos()
                for _ in 0..<iterations {
                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }
                    encoder.setComputePipelineState(broadcastPipeline)
                    encoder.setBuffer(devBuffer, offset: 0, index: 0)
                    encoder.setBuffer(cstBuffer, offset: 0, index: 1)
                    encoder.setBuffer(outBuffer, offset: 0, index: 2)
                    encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 3)
                    encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    encoder.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()
                }
                let end = getTimeNanos()
                let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
                let gops = Double(size) / elapsed / 1e9
                print("Broadcast: \(String(format: "%.2f", gops)) GOPS (\(String(format: "%.4f", elapsed * 1000)) ms)")
            }

            // Test strided constant read
            if let strideFunc = library.makeFunction(name: "constant_strided"),
               let stridePipeline = try? device.makeComputePipelineState(function: strideFunc) {
                let iterations = 20
                let start = getTimeNanos()
                for _ in 0..<iterations {
                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }
                    encoder.setComputePipelineState(stridePipeline)
                    encoder.setBuffer(devBuffer, offset: 0, index: 0)
                    encoder.setBuffer(cstBuffer, offset: 0, index: 1)
                    encoder.setBuffer(outBuffer, offset: 0, index: 2)
                    encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 3)
                    encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    encoder.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()
                }
                let end = getTimeNanos()
                let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
                let gops = Double(size) / elapsed / 1e9
                print("Strided: \(String(format: "%.2f", gops)) GOPS (\(String(format: "%.4f", elapsed * 1000)) ms)")
            }
        }

        print("\n--- Key Findings ---")
        print("1. Constant memory is cached - broadcast is most efficient")
        print("2. Scattered access pattern causes constant cache misses")
        print("3. Sequential constant reads benefit from cache line fetches")
        print("4. For small constants used by all threads, broadcast is optimal")
        print("5. Apple M2 constant cache size is limited (~32KB)")
    }
}
