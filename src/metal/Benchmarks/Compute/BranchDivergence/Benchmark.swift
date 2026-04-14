import Foundation
import Metal

// MARK: - Branch Divergence Benchmark

let branchDivergenceShaders = """
#include <metal_stdlib>
using namespace metal;

// Converged branch: all threads take same path
kernel void branch_converged(device const float* in [[buffer(0)]],
                            device float* out [[buffer(1)]],
                            constant uint& size [[buffer(2)]],
                            uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    // All threads take same branch - no divergence
    float val = in[id];
    if (val > 0.5f) {
        out[id] = val * 2.0f;
    } else {
        out[id] = val * 0.5f;
    }
}

// Divergent branch: threads in same SIMD-group take different paths
kernel void branch_divergent_even_odd(device const float* in [[buffer(0)]],
                                     device float* out [[buffer(1)]],
                                     constant uint& size [[buffer(2)]],
                                     uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    uint lane = id & 31;  // SIMD group lane (0-31)
    float val = in[id];
    // Even lanes take one branch, odd lanes take another
    if ((lane & 1) == 0) {
        out[id] = val * 2.0f;
    } else {
        out[id] = val * 0.5f;
    }
}

// Divergent branch: quarter-warp divergence (8 threads per group)
kernel void branch_divergent_quarter(device const float* in [[buffer(0)]],
                                     device float* out [[buffer(1)]],
                                     constant uint& size [[buffer(2)]],
                                     uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    uint lane = id & 31;
    float val = in[id];
    // First 8 lanes take one branch
    if (lane < 8) {
        out[id] = val * 3.0f;
    } else {
        out[id] = val * 0.33f;
    }
}

// Divergent branch: alternating pattern
kernel void branch_divergent_alternating(device const float* in [[buffer(0)]],
                                        device float* out [[buffer(1)]],
                                        constant uint& size [[buffer(2)]],
                                        uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    uint lane = id & 31;
    float val = in[id];
    // Every 4 lanes switch branches
    if ((lane & 3) == 0) {
        out[id] = val * 4.0f;
    } else {
        out[id] = val * 0.25f;
    }
}

// No branch: using select/ternary operator
kernel void branchless_select(device const float* in [[buffer(0)]],
                              device float* out [[buffer(1)]],
                              constant uint& size [[buffer(2)]],
                              uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    float val = in[id];
    // Branchless using select - no divergence
    float2 result = select(float2(val * 0.5f), float2(val * 2.0f), float2(val > 0.5f));
    out[id] = result[0];
}

// Converged with loop (all threads execute same iterations)
kernel void loop_converged(device const float* in [[buffer(0)]],
                          device float* out [[buffer(1)]],
                          constant uint& size [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    float val = in[id];
    float sum = 0.0f;
    for (uint i = 0; i < 16; i++) {
        sum += val;
    }
    out[id] = sum;
}

// Divergent loop (different iterations per lane)
kernel void loop_divergent(device const float* in [[buffer(0)]],
                          device float* out [[buffer(1)]],
                          constant uint& size [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    uint lane = id & 31;
    float val = in[id];
    float sum = 0.0f;
    // Different loop counts per lane
    uint iterations = 4 + (lane & 15);
    for (uint i = 0; i < iterations; i++) {
        sum += val;
    }
    out[id] = sum;
}
"""

public struct BranchDivergenceBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    public init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
    }

    public func run() throws {
        print(String(repeating: "=", count: 70))
        print("Branch Divergence Benchmark")
        print(String(repeating: "=", count: 70))

        guard let library = try? device.makeLibrary(source: branchDivergenceShaders, options: nil) else {
            print("Failed to compile branch divergence shaders")
            return
        }

        let sizes = [65536, 262144]
        let iterations = 20

        for size in sizes {
            print("\n--- Array Size: \(size) ---")

            guard let bufferIn = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared),
                  let bufferOut = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared) else {
                continue
            }

            // Initialize with random-like data
            let inPtr = bufferIn.contents().bindMemory(to: Float.self, capacity: size)
            for i in 0..<size {
                inPtr[i] = Float(i % 256) / 255.0
            }

            var sizeValue = UInt32(size)

            // Test converged branch
            if let func_ = library.makeFunction(name: "branch_converged"),
               let pipeline = try? device.makeComputePipelineState(function: func_) {
                let start = getTimeNanos()
                for _ in 0..<iterations {
                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }
                    encoder.setComputePipelineState(pipeline)
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
                let gops = Double(size) / elapsed / 1e9
                print("Converged Branch: \(String(format: "%.2f", gops)) GOPS (\(String(format: "%.4f", elapsed * 1000)) ms)")
            }

            // Test even-odd divergent branch
            if let func_ = library.makeFunction(name: "branch_divergent_even_odd"),
               let pipeline = try? device.makeComputePipelineState(function: func_) {
                let start = getTimeNanos()
                for _ in 0..<iterations {
                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }
                    encoder.setComputePipelineState(pipeline)
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
                let gops = Double(size) / elapsed / 1e9
                print("Divergent (Even/Odd): \(String(format: "%.2f", gops)) GOPS (\(String(format: "%.4f", elapsed * 1000)) ms)")
            }

            // Test quarter-warp divergent branch
            if let func_ = library.makeFunction(name: "branch_divergent_quarter"),
               let pipeline = try? device.makeComputePipelineState(function: func_) {
                let start = getTimeNanos()
                for _ in 0..<iterations {
                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }
                    encoder.setComputePipelineState(pipeline)
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
                let gops = Double(size) / elapsed / 1e9
                print("Divergent (Quarter): \(String(format: "%.2f", gops)) GOPS (\(String(format: "%.4f", elapsed * 1000)) ms)")
            }

            // Test branchless select
            if let func_ = library.makeFunction(name: "branchless_select"),
               let pipeline = try? device.makeComputePipelineState(function: func_) {
                let start = getTimeNanos()
                for _ in 0..<iterations {
                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }
                    encoder.setComputePipelineState(pipeline)
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
                let gops = Double(size) / elapsed / 1e9
                print("Branchless Select: \(String(format: "%.2f", gops)) GOPS (\(String(format: "%.4f", elapsed * 1000)) ms)")
            }

            // Test converged loop
            if let func_ = library.makeFunction(name: "loop_converged"),
               let pipeline = try? device.makeComputePipelineState(function: func_) {
                let start = getTimeNanos()
                for _ in 0..<iterations {
                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }
                    encoder.setComputePipelineState(pipeline)
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
                let gops = Double(size * 16) / elapsed / 1e9
                print("Converged Loop: \(String(format: "%.2f", gops)) GOPS (\(String(format: "%.4f", elapsed * 1000)) ms)")
            }

            // Test divergent loop
            if let func_ = library.makeFunction(name: "loop_divergent"),
               let pipeline = try? device.makeComputePipelineState(function: func_) {
                let start = getTimeNanos()
                for _ in 0..<iterations {
                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }
                    encoder.setComputePipelineState(pipeline)
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
                let gops = Double(size * 12) / elapsed / 1e9  // Average 12 iterations
                print("Divergent Loop: \(String(format: "%.2f", gops)) GOPS (\(String(format: "%.4f", elapsed * 1000)) ms)")
            }
        }

        print("\n--- Key Findings ---")
        print("1. Converged branches have no divergence penalty")
        print("2. Even/Odd divergence splits SIMD-group in half")
        print("3. Quarter-warp (8 threads) is common divergence pattern")
        print("4. Branchless using select() avoids divergence")
        print("5. Divergent loops waste execution resources")
    }
}
