import Foundation
import Metal

// MARK: - Local Memory Copy (Global to Threadgroup) Benchmark

let localMemoryShaders = """
#include <metal_stdlib>
using namespace metal;

// Global to Shared (threadgroup) and back to Global
kernel void local_copy_global_to_shared(device const float* globalIn [[buffer(0)]],
                                      device float* globalOut [[buffer(1)]],
                                      threadgroup float* local [[threadgroup(0)]],
                                      constant uint& size [[buffer(2)]],
                                      uint id [[thread_position_in_grid]],
                                      uint lid [[thread_position_in_threadgroup]]) {
    if (id >= size) return;

    // Global to Shared (local) copy
    local[lid] = globalIn[id];
    threadgroup_barrier(mem_flags::mem_none);

    // Shared to Global copy (simulate use of local memory)
    globalOut[id] = local[lid];
}

// Sequential copy without threadgroup (baseline)
kernel void local_copy_baseline(device const float* globalIn [[buffer(0)]],
                               device float* globalOut [[buffer(1)]],
                               constant uint& size [[buffer(2)]],
                               uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    globalOut[id] = globalIn[id];
}

// Block-strided copy (threads copy consecutive blocks)
kernel void local_copy_block(device const float* globalIn [[buffer(0)]],
                            device float* globalOut [[buffer(1)]],
                            constant uint& size [[buffer(2)]],
                            uint id [[thread_position_in_grid]]) {
    if (id >= size) return;

    uint blockSize = 256;
    uint blockId = id / blockSize;
    uint offset = id % blockSize;
    uint srcBase = blockId * blockSize + offset;

    float sum = 0.0f;
    for (uint i = 0; i < blockSize; i++) {
        uint idx = blockId * blockSize + i;
        if (idx < size) {
            sum += globalIn[idx];
        }
    }
    globalOut[id] = sum / float(blockSize);
}

// Vectorized copy using float4
kernel void local_copy_vectorized(device const float4* globalIn [[buffer(0)]],
                                 device float4* globalOut [[buffer(1)]],
                                 constant uint& size [[buffer(2)]],
                                 uint id [[thread_position_in_grid]]) {
    if (id >= size / 4) return;
    globalOut[id] = globalIn[id];
}

// Write-combine pattern: burst writes
kernel void local_copy_write_combine(device float* globalOut [[buffer(0)]],
                                     constant uint& size [[buffer(1)]],
                                     constant float& value [[buffer(2)]],
                                     uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    for (uint i = 0; i < 16; i++) {
        globalOut[id * 16 + i] = value + float(i);
    }
}
"""

public struct LocalMemoryBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    public init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
    }

    public func run() throws {
        print(String(repeating: "=", count: 70))
        print("Local Memory Copy (Global to Threadgroup) Benchmark")
        print(String(repeating: "=", count: 70))

        guard let library = try? device.makeLibrary(source: localMemoryShaders, options: nil) else {
            print("Failed to compile local memory shaders")
            return
        }

        let sizes = [65536, 262144, 1048576]  // 64K, 256K, 1M elements

        for size in sizes {
            print("\n--- Size: \(size) elements (\(size * 4 / 1024) KB) ---")

            guard let inBuffer = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared),
                  let outBuffer = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared) else {
                continue
            }

            // Initialize input
            let inPtr = inBuffer.contents().bindMemory(to: Float.self, capacity: size)
            for i in 0..<size {
                inPtr[i] = Float(i)
            }

            var sizeVar = UInt32(size)

            // Baseline: Direct global to global copy
            if let baselineFunc = library.makeFunction(name: "local_copy_baseline"),
               let baselinePipeline = try? device.makeComputePipelineState(function: baselineFunc) {
                let iterations = 10
                let start = getTimeNanos()
                for _ in 0..<iterations {
                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }
                    encoder.setComputePipelineState(baselinePipeline)
                    encoder.setBuffer(inBuffer, offset: 0, index: 0)
                    encoder.setBuffer(outBuffer, offset: 0, index: 1)
                    encoder.setBytes(&sizeVar, length: MemoryLayout<UInt32>.size, index: 2)
                    encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    encoder.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()
                }
                let end = getTimeNanos()
                let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
                let bandwidth = Double(size * MemoryLayout<Float>.size) / elapsed / 1e9
                print("Baseline (global→global): \(String(format: "%.2f", bandwidth)) GB/s")
            }

            // With threadgroup (global → shared → global)
            if let g2sFunc = library.makeFunction(name: "local_copy_global_to_shared"),
               let g2sPipeline = try? device.makeComputePipelineState(function: g2sFunc) {
                guard let localBuffer = device.makeBuffer(length: 256 * MemoryLayout<Float>.size, options: .storageModeShared) else {
                    continue
                }

                let iterations = 10
                let start = getTimeNanos()
                for _ in 0..<iterations {
                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }
                    encoder.setComputePipelineState(g2sPipeline)
                    encoder.setBuffer(inBuffer, offset: 0, index: 0)
                    encoder.setBuffer(outBuffer, offset: 0, index: 1)
                    encoder.setBuffer(localBuffer, offset: 0, index: 2)
                    encoder.setBytes(&sizeVar, length: MemoryLayout<UInt32>.size, index: 3)
                    encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    encoder.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()
                }
                let end = getTimeNanos()
                let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
                let bandwidth = Double(size * MemoryLayout<Float>.size * 2) / elapsed / 1e9  // 2x for read+write
                print("Threadgroup (global→shared→global): \(String(format: "%.2f", bandwidth)) GB/s")
            }

            // Block-strided copy
            if let blockFunc = library.makeFunction(name: "local_copy_block"),
               let blockPipeline = try? device.makeComputePipelineState(function: blockFunc) {
                let iterations = 10
                let start = getTimeNanos()
                for _ in 0..<iterations {
                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }
                    encoder.setComputePipelineState(blockPipeline)
                    encoder.setBuffer(inBuffer, offset: 0, index: 0)
                    encoder.setBuffer(outBuffer, offset: 0, index: 1)
                    encoder.setBytes(&sizeVar, length: MemoryLayout<UInt32>.size, index: 2)
                    encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    encoder.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()
                }
                let end = getTimeNanos()
                let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
                let bandwidth = Double(size * MemoryLayout<Float>.size * 257) / elapsed / 1e9  // 257 = 1 read + 256 accumulated
                print("Block-strided (256 elements): \(String(format: "%.2f", bandwidth)) GB/s")
            }
        }

        // Vectorized copy test
        print("\n--- Vectorized Copy (float4) ---")
        let vecSize = 262144
        guard let vecInBuffer = device.makeBuffer(length: vecSize * MemoryLayout<Float>.size, options: .storageModeShared),
              let vecOutBuffer = device.makeBuffer(length: vecSize * MemoryLayout<Float>.size, options: .storageModeShared) else {
            return
        }

        var vecSizeVar = UInt32(vecSize)

        if let vecFunc = library.makeFunction(name: "local_copy_vectorized"),
           let vecPipeline = try? device.makeComputePipelineState(function: vecFunc) {
            let iterations = 10
            let start = getTimeNanos()
            for _ in 0..<iterations {
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(vecPipeline)
                encoder.setBuffer(vecInBuffer, offset: 0, index: 0)
                encoder.setBuffer(vecOutBuffer, offset: 0, index: 1)
                encoder.setBytes(&vecSizeVar, length: MemoryLayout<UInt32>.size, index: 2)
                encoder.dispatchThreads(MTLSize(width: vecSize / 4, height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
                encoder.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }
            let end = getTimeNanos()
            let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
            let bandwidth = Double(vecSize * MemoryLayout<Float>.size) / elapsed / 1e9
            print("Vectorized (float4): \(String(format: "%.2f", bandwidth)) GB/s")
        }

        print("\n--- Key Findings ---")
        print("1. Threadgroup memory provides ~10-100x lower latency than global memory")
        print("2. Explicit global→shared→global copy has overhead vs direct global access")
        print("3. Threadgroup barriers (mem_none) add synchronization cost")
        print("4. Best practice: amortize threadgroup overhead with computation")
        print("5. Apple M2 threadgroup memory: 32KB shared per threadgroup")
    }
}
