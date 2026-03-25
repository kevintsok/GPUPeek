import Foundation
import Metal

// MARK: - Cache Analysis Benchmark

let cacheShaders = """
#include <metal_stdlib>
using namespace metal;

// Sequential access - best cache behavior
kernel void seq_access(device const float* in [[buffer(0)]],
                      device float* out [[buffer(1)]],
                      constant uint& size [[buffer(2)]],
                      uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    out[id] = in[id] * 1.001f;
}

// Strided access - stride 2
kernel void stride2_access(device const float* in [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        constant uint& size [[buffer(2)]],
                        uint id [[thread_position_in_grid]]) {
    if (id >= size / 2) return;
    out[id] = in[id * 2] * 1.001f;
}

// Strided access - stride 8
kernel void stride8_access(device const float* in [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        constant uint& size [[buffer(2)]],
                        uint id [[thread_position_in_grid]]) {
    if (id >= size / 8) return;
    out[id] = in[id * 8] * 1.001f;
}

// Random access - worst cache behavior
kernel void random_access(device const float* in [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        device const uint* indices [[buffer(2)]],
                        constant uint& size [[buffer(3)]],
                        uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    uint idx = indices[id] % size;
    out[id] = in[idx] * 1.001f;
}

// Working set test - various sizes
kernel void working_set(device const float* in [[buffer(0)]],
                       device float* out [[buffer(1)]],
                       constant uint& size [[buffer(2)]],
                       uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    float sum = 0.0f;
    for (uint i = 0; i < 16; i++) {
        uint idx = (id + i) % size;
        sum += in[idx];
    }
    out[id] = sum * 0.0625f;
}

// Write-only test - bypass cache
kernel void write_only(device float* out [[buffer(0)]],
                     constant uint& size [[buffer(1)]],
                     uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    out[id] = float(id) * 1.001f;
}
"""

public struct CacheBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    public init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
    }

    public func run() throws {
        print(String(repeating: "=", count: 70))
        print("Cache Behavior Analysis Benchmark")
        print(String(repeating: "=", count: 70))

        guard let library = try? device.makeLibrary(source: cacheShaders, options: nil) else {
            print("Failed to compile cache shaders")
            return
        }

        let testSizes = [32 * 1024, 256 * 1024, 1024 * 1024, 8 * 1024 * 1024]
        let iterations = 50

        print("\n--- Cache Line Effects (Sequential vs Strided) ---")

        for size in testSizes {
            guard let bufferIn = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared),
                  let bufferOut = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared) else {
                continue
            }

            var sizeValue = UInt32(size)

            // Sequential access
            if let seqFunc = library.makeFunction(name: "seq_access"),
               let seqPipeline = try? device.makeComputePipelineState(function: seqFunc) {
                let start = getTimeNanos()
                for _ in 0..<iterations {
                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }
                    encoder.setComputePipelineState(seqPipeline)
                    encoder.setBuffer(bufferIn, offset: 0, index: 0)
                    encoder.setBuffer(bufferOut, offset: 0, index: 1)
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
                let sizeLabel = size >= 1024 * 1024 ? "\(size / (1024 * 1024))MB" : "\(size / 1024)KB"
                print("Sequential (\(sizeLabel)): \(String(format: "%.2f", bandwidth)) GB/s")
            }

            // Stride 8 access
            if let strideFunc = library.makeFunction(name: "stride8_access"),
               let stridePipeline = try? device.makeComputePipelineState(function: strideFunc) {
                let start = getTimeNanos()
                for _ in 0..<iterations {
                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }
                    encoder.setComputePipelineState(stridePipeline)
                    encoder.setBuffer(bufferIn, offset: 0, index: 0)
                    encoder.setBuffer(bufferOut, offset: 0, index: 1)
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
                let sizeLabel = size >= 1024 * 1024 ? "\(size / (1024 * 1024))MB" : "\(size / 1024)KB"
                print("Stride-8 (\(sizeLabel)): \(String(format: "%.2f", bandwidth)) GB/s")
            }
        }

        print("\n--- Working Set Size Effects ---")

        for size in testSizes {
            guard let bufferIn = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared),
                  let bufferOut = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared) else {
                continue
            }

            var sizeValue = UInt32(size)

            if let wsFunc = library.makeFunction(name: "working_set"),
               let wsPipeline = try? device.makeComputePipelineState(function: wsFunc) {
                let start = getTimeNanos()
                for _ in 0..<iterations {
                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }
                    encoder.setComputePipelineState(wsPipeline)
                    encoder.setBuffer(bufferIn, offset: 0, index: 0)
                    encoder.setBuffer(bufferOut, offset: 0, index: 1)
                    encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 2)
                    encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    encoder.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()
                }
                let end = getTimeNanos()
                let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
                let gops = Double(size) * 16.0 / elapsed / 1e9
                let sizeLabel = size >= 1024 * 1024 ? "\(size / (1024 * 1024))MB" : "\(size / 1024)KB"
                print("Working Set (\(sizeLabel)): \(String(format: "%.3f", gops)) GOPS")
            }
        }

        print("\n--- Key Insights ---")
        print("1. Sequential access best: exploits spatial locality")
        print("2. Strided access wastes cache: skips most of each line")
        print("3. L1 cache ~32KB, L2 cache ~4MB on Apple M2")
        print("4. Working set > L2 causes significant slowdown")
    }
}
