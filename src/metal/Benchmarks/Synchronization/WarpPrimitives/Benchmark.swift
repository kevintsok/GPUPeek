import Foundation
import Metal

// MARK: - Warp/SIMD Primitives Benchmark

let warpShaders = """
#include <metal_stdlib>
using namespace metal;

// SIMD vote any - returns true if any thread in warp has condition
kernel void simd_vote_any(device const uint* in [[buffer(0)]],
                         device uint* out [[buffer(1)]],
                         constant uint& size [[buffer(2)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    bool predicate = in[id] > 128;
    uint result = simd_any(predicate);
    out[id] = result;
}

// SIMD vote all - returns true if all threads in warp have condition
kernel void simd_vote_all(device const uint* in [[buffer(0)]],
                         device uint* out [[buffer(1)]],
                         constant uint& size [[buffer(2)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    bool predicate = in[id] > 128;
    uint result = simd_all(predicate);
    out[id] = result;
}

// SIMD shuffle - exchange data between lanes
kernel void simd_shuffle(device const float* in [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        constant uint& size [[buffer(2)]],
                        uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    float val = in[id];
    // Shuffle down by 1 lane
    float shuffled = simd_shuffle_down(val, 1.0f);
    out[id] = val + shuffled;
}

// SIMD shuffle xor - for butterfly patterns
kernel void simd_shuffle_xor(device const float* in [[buffer(0)]],
                              device float* out [[buffer(1)]],
                              constant uint& size [[buffer(2)]],
                              uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    float val = in[id];
    // XOR shuffle with mask 16
    float shuffled = simd_shuffle_xor(val, 16);
    out[id] = val + shuffled;
}

// SIMD broadcast - broadcast value from one lane to all
kernel void simd_broadcast(device const float* in [[buffer(0)]],
                          device float* out [[buffer(1)]],
                          constant uint& size [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    float val = in[id];
    // Broadcast from lane 0
    float broadcasted = simd_broadcast(val, 0);
    out[id] = broadcasted;
}

// SIMD prefix sum (Hill Steele)
kernel void simd_prefix_sum(device const float* in [[buffer(0)]],
                           device float* out [[buffer(1)]],
                           constant uint& size [[buffer(2)]],
                           uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    float val = in[id];
    
    float sum = val;
    sum += simd_shuffle_down(val, 1);
    sum += simd_shuffle_down(sum, 2);
    sum += simd_shuffle_down(sum, 4);
    sum += simd_shuffle_down(sum, 8);
    sum += simd_shuffle_down(sum, 16);
    
    out[id] = sum;
}
"""

public struct WarpPrimitivesBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    public init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
    }

    public func run() throws {
        print(String(repeating: "=", count: 70))
        print("SIMD Group (Warp) Primitives Benchmark")
        print(String(repeating: "=", count: 70))

        guard let library = try? device.makeLibrary(source: warpShaders, options: nil) else {
            print("Failed to compile warp shaders")
            return
        }

        let size = 256 * 1024
        let iterations = 100

        guard let bufferIn = device.makeBuffer(length: size * MemoryLayout<UInt32>.size, options: .storageModeShared),
              let bufferOut = device.makeBuffer(length: size * MemoryLayout<UInt32>.size, options: .storageModeShared) else {
            print("Failed to create buffers")
            return
        }

        // Initialize input
        let inPtr = bufferIn.contents().bindMemory(to: UInt32.self, capacity: size)
        for i in 0..<size {
            inPtr[i] = UInt32(i % 256)
        }

        var sizeValue = UInt32(size)

        print("\n--- SIMD Vote Operations ---")

        // Test 1: SIMD vote any
        if let anyFunc = library.makeFunction(name: "simd_vote_any"),
           let anyPipeline = try? device.makeComputePipelineState(function: anyFunc) {
            let start = getTimeNanos()
            for _ in 0..<iterations {
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(anyPipeline)
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
            print("SIMD Vote Any: \(String(format: "%.3f", gops)) GOPS")
        }

        // Test 2: SIMD vote all
        if let allFunc = library.makeFunction(name: "simd_vote_all"),
           let allPipeline = try? device.makeComputePipelineState(function: allFunc) {
            let start = getTimeNanos()
            for _ in 0..<iterations {
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(allPipeline)
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
            print("SIMD Vote All: \(String(format: "%.3f", gops)) GOPS")
        }

        print("\n--- SIMD Shuffle Operations ---")

        // Test 3: SIMD shuffle
        if let shuffleFunc = library.makeFunction(name: "simd_shuffle"),
           let shufflePipeline = try? device.makeComputePipelineState(function: shuffleFunc) {
            let start = getTimeNanos()
            for _ in 0..<iterations {
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(shufflePipeline)
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
            print("SIMD Shuffle: \(String(format: "%.3f", gops)) GOPS")
        }

        // Test 4: SIMD shuffle xor
        if let xorFunc = library.makeFunction(name: "simd_shuffle_xor"),
           let xorPipeline = try? device.makeComputePipelineState(function: xorFunc) {
            let start = getTimeNanos()
            for _ in 0..<iterations {
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(xorPipeline)
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
            print("SIMD Shuffle XOR: \(String(format: "%.3f", gops)) GOPS")
        }

        // Test 5: SIMD broadcast
        if let broadcastFunc = library.makeFunction(name: "simd_broadcast"),
           let broadcastPipeline = try? device.makeComputePipelineState(function: broadcastFunc) {
            let start = getTimeNanos()
            for _ in 0..<iterations {
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(broadcastPipeline)
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
            print("SIMD Broadcast: \(String(format: "%.3f", gops)) GOPS")
        }

        // Test 6: SIMD prefix sum
        if let prefixFunc = library.makeFunction(name: "simd_prefix_sum"),
           let prefixPipeline = try? device.makeComputePipelineState(function: prefixFunc) {
            let start = getTimeNanos()
            for _ in 0..<iterations {
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(prefixPipeline)
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
            print("SIMD Prefix Sum: \(String(format: "%.3f", gops)) GOPS")
        }

        print("\n--- Key Insights ---")
        print("1. SIMD vote operations are hardware-native and very fast")
        print("2. SIMD shuffle enables efficient warp-level communication")
        print("3. XOR shuffle is optimal for reduction patterns")
        print("4. SIMD prefix sum is O(log n) with hardware support")
        print("5. Broadcast is most efficient for sharing single value")
    }
}
