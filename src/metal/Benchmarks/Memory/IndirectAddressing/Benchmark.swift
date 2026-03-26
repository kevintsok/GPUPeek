import Foundation
import Metal

// MARK: - Indirect Addressing (Scatter-Gather) Benchmark

let indirectAddressingShaders = """
#include <metal_stdlib>
using namespace metal;

// Gather: read from scattered indices
kernel void gather_addressing(device const float* data [[buffer(0)]],
                         device const uint* indices [[buffer(1)]],
                         device float* out [[buffer(2)]],
                         constant uint& size [[buffer(3)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    uint idx = indices[id] % size;
    out[id] = data[idx] * 2.0f;
}

// Scatter: write to scattered indices
kernel void scatter_addressing(device float* data [[buffer(0)]],
                            device const uint* indices [[buffer(1)]],
                            device const float* values [[buffer(2)]],
                            constant uint& size [[buffer(3)]],
                            uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    uint idx = indices[id] % size;
    data[idx] = values[id] * 2.0f;
}

// Gather + Process: read then compute
kernel void gather_then_process(device const float* data [[buffer(0)]],
                            device const uint* indices [[buffer(1)]],
                            device float* out [[buffer(2)]],
                            constant uint& size [[buffer(3)]],
                            uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    uint idx = indices[id] % size;
    float val = data[idx];
    val = val * 2.0f + 1.0f;
    val = sqrt(val + 0.001f);
    out[id] = val;
}

// Sequential baseline: sequential access for comparison
kernel void sequential_access(device const float* data [[buffer(0)]],
                         device float* out [[buffer(1)]],
                         constant uint& size [[buffer(2)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    out[id] = data[id] * 2.0f;
}

// Strided access: access with fixed stride
kernel void strided_access(device const float* data [[buffer(0)]],
                       device float* out [[buffer(1)]],
                       constant uint& size [[buffer(2)]],
                       constant uint& stride [[buffer(3)]],
                       uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    uint idx = (id * stride) % size;
    out[id] = data[idx] * 2.0f;
}
"""

public struct IndirectAddressingBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    public init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
    }

    public func run() throws {
        print(String(repeating: "=", count: 70))
        print("Indirect Addressing (Scatter-Gather) Benchmark")
        print(String(repeating: "=", count: 70))

        guard let library = try? device.makeLibrary(source: indirectAddressingShaders, options: nil) else {
            print("Failed to compile indirect addressing shaders")
            return
        }

        let sizes = [4096, 16384, 65536]
        let patterns = ["Random", "Sequential", "Strided"]

        for size in sizes {
            print("\n--- Array Size: \(size) ---")

            guard let dataBuffer = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared),
                  let indicesBuffer = device.makeBuffer(length: size * MemoryLayout<UInt32>.size, options: .storageModeShared),
                  let valuesBuffer = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared),
                  let outBuffer = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared) else {
                continue
            }

            // Initialize data array
            let dataPtr = dataBuffer.contents().bindMemory(to: Float.self, capacity: size)
            for i in 0..<size {
                dataPtr[i] = Float(i % 256) / Float(256.0)
            }

            // Initialize indices with different patterns
            let indicesPtr = indicesBuffer.contents().bindMemory(to: UInt32.self, capacity: size)

            // Random pattern (pseudo-random for testing)
            for i in 0..<size {
                indicesPtr[i] = UInt32((i * 12345 + 67890) % size)
            }

            // Initialize values
            let valuesPtr = valuesBuffer.contents().bindMemory(to: Float.self, capacity: size)
            for i in 0..<size {
                valuesPtr[i] = Float(i % 128) / Float(128.0)
            }

            var sizeValue = UInt32(size)

            // Test gather addressing
            if let gatherFunc = library.makeFunction(name: "gather_addressing"),
               let gatherPipeline = try? device.makeComputePipelineState(function: gatherFunc) {
                let iterations = 50
                let start = getTimeNanos()
                for _ in 0..<iterations {
                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }
                    encoder.setComputePipelineState(gatherPipeline)
                    encoder.setBuffer(dataBuffer, offset: 0, index: 0)
                    encoder.setBuffer(indicesBuffer, offset: 0, index: 1)
                    encoder.setBuffer(outBuffer, offset: 0, index: 2)
                    encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 3)
                    encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
                    encoder.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()
                }
                let end = getTimeNanos()
                let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
                let bandwidth = Double(size * MemoryLayout<Float>.size) / elapsed / 1e9
                print("Gather (Random Index): \(String(format: "%.2f", bandwidth)) GB/s (\(String(format: "%.4f", elapsed * 1000)) ms)")
            }

            // Test sequential baseline
            if let seqFunc = library.makeFunction(name: "sequential_access"),
               let seqPipeline = try? device.makeComputePipelineState(function: seqFunc) {
                let iterations = 50
                let start = getTimeNanos()
                for _ in 0..<iterations {
                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }
                    encoder.setComputePipelineState(seqPipeline)
                    encoder.setBuffer(dataBuffer, offset: 0, index: 0)
                    encoder.setBuffer(outBuffer, offset: 0, index: 1)
                    encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 2)
                    encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
                    encoder.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()
                }
                let end = getTimeNanos()
                let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
                let bandwidth = Double(size * MemoryLayout<Float>.size) / elapsed / 1e9
                print("Sequential (Baseline):   \(String(format: "%.2f", bandwidth)) GB/s (\(String(format: "%.4f", elapsed * 1000)) ms)")
            }

            // Test strided access
            if let strideFunc = library.makeFunction(name: "strided_access"),
               let stridePipeline = try? device.makeComputePipelineState(function: strideFunc) {
                var strideValue: UInt32 = 7  // Prime stride to maximize scattering
                let iterations = 50
                let start = getTimeNanos()
                for _ in 0..<iterations {
                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }
                    encoder.setComputePipelineState(stridePipeline)
                    encoder.setBuffer(dataBuffer, offset: 0, index: 0)
                    encoder.setBuffer(outBuffer, offset: 0, index: 1)
                    encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 2)
                    encoder.setBytes(&strideValue, length: MemoryLayout<UInt32>.size, index: 3)
                    encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
                    encoder.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()
                }
                let end = getTimeNanos()
                let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
                let bandwidth = Double(size * MemoryLayout<Float>.size) / elapsed / 1e9
                print("Strided (stride=7):      \(String(format: "%.2f", bandwidth)) GB/s (\(String(format: "%.4f", elapsed * 1000)) ms)")
            }

            // Test gather then process
            if let processFunc = library.makeFunction(name: "gather_then_process"),
               let processPipeline = try? device.makeComputePipelineState(function: processFunc) {
                let iterations = 50
                let start = getTimeNanos()
                for _ in 0..<iterations {
                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }
                    encoder.setComputePipelineState(processPipeline)
                    encoder.setBuffer(dataBuffer, offset: 0, index: 0)
                    encoder.setBuffer(indicesBuffer, offset: 0, index: 1)
                    encoder.setBuffer(outBuffer, offset: 0, index: 2)
                    encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 3)
                    encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
                    encoder.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()
                }
                let end = getTimeNanos()
                let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
                let bandwidth = Double(size * MemoryLayout<Float>.size) / elapsed / 1e9
                print("Gather+Process:           \(String(format: "%.2f", bandwidth)) GB/s (\(String(format: "%.4f", elapsed * 1000)) ms)")
            }
        }

        // Scatter test (requires separate buffer to avoid write-after-read hazards)
        print("\n--- Scatter Write Test ---")
        let scatterSize = 16384
        guard let scatterDataBuffer = device.makeBuffer(length: scatterSize * MemoryLayout<Float>.size, options: .storageModeShared),
              let scatterIndicesBuffer = device.makeBuffer(length: scatterSize * MemoryLayout<UInt32>.size, options: .storageModeShared),
              let scatterValuesBuffer = device.makeBuffer(length: scatterSize * MemoryLayout<Float>.size, options: .storageModeShared) else {
            return
        }

        // Initialize scatter indices
        let scatterIndicesPtr = scatterIndicesBuffer.contents().bindMemory(to: UInt32.self, capacity: scatterSize)
        for i in 0..<scatterSize {
            scatterIndicesPtr[i] = UInt32((i * 12345 + 67890) % scatterSize)
        }

        let scatterValuesPtr = scatterValuesBuffer.contents().bindMemory(to: Float.self, capacity: scatterSize)
        for i in 0..<scatterSize {
            scatterValuesPtr[i] = Float(i % 128) / 128.0f
        }

        var scatterSizeValue = UInt32(scatterSize)

        if let scatterFunc = library.makeFunction(name: "scatter_addressing"),
           let scatterPipeline = try? device.makeComputePipelineState(function: scatterFunc) {
            let iterations = 50
            let start = getTimeNanos()
            for _ in 0..<iterations {
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(scatterPipeline)
                encoder.setBuffer(scatterDataBuffer, offset: 0, index: 0)
                encoder.setBuffer(scatterIndicesBuffer, offset: 0, index: 1)
                encoder.setBuffer(scatterValuesBuffer, offset: 0, index: 2)
                encoder.setBytes(&scatterSizeValue, length: MemoryLayout<UInt32>.size, index: 3)
                encoder.dispatchThreads(MTLSize(width: scatterSize, height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
                encoder.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }
            let end = getTimeNanos()
            let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
            let bandwidth = Double(scatterSize * MemoryLayout<Float>.size) / elapsed / 1e9
            print("Scatter (Random Index):  \(String(format: "%.2f", bandwidth)) GB/s (\(String(format: "%.4f", elapsed * 1000)) ms)")
        }

        print("\n--- Key Findings ---")
        print("1. Random indexing (gather/scatter) destroys memory coalescing")
        print("2. Sequential access is baseline - all random patterns should be slower")
        print("3. Strided access with large stride has similar cost to random")
        print("4. GPU caches help with locality but cannot fully compensate")
        print("5. For scatter-gather heavy workloads, consider sorting by index first")
    }
}
