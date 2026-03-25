import Foundation
import Metal

// MARK: - Double Buffer Benchmark

let doubleBufferShaders = """
#include <metal_stdlib>
using namespace metal;

// Single buffer - no overlap
kernel void single_buffer(device const float* in [[buffer(0)]],
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

// Double buffer - read from buffer A, write to buffer B
kernel void double_buffer_read_a(device const float* inA [[buffer(0)]],
                              device float* outB [[buffer(1)]],
                              constant uint& size [[buffer(2)]],
                              uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    float sum = 0.0f;
    for (uint i = 0; i < 16; i++) {
        uint idx = (id + i) % size;
        sum += inA[idx];
    }
    outB[id] = sum * 0.0625f;
}

// Double buffer - read from buffer B, write to buffer A
kernel void double_buffer_read_b(device const float* inB [[buffer(0)]],
                              device float* outA [[buffer(1)]],
                              constant uint& size [[buffer(2)]],
                              uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    float sum = 0.0f;
    for (uint i = 0; i < 16; i++) {
        uint idx = (id + i) % size;
        sum += inB[idx];
    }
    outA[id] = sum * 0.0625f;
}

// Triple buffer - max overlap potential
kernel void triple_buffer_1(device const float* in [[buffer(0)]],
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

// Ping-pong with explicit sync - alternating buffers
kernel void pingpong_phase_a(device const float* inA [[buffer(0)]],
                           device float* outA [[buffer(1)]],
                           constant uint& size [[buffer(2)]],
                           uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    float sum = 0.0f;
    for (uint i = 0; i < 16; i++) {
        uint idx = (id + i) % size;
        sum += inA[idx];
    }
    outA[id] = sum * 0.0625f;
}
"""

public struct DoubleBufferBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    public init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
    }

    public func run() throws {
        print(String(repeating: "=", count: 70))
        print("Double Buffering Benchmark")
        print(String(repeating: "=", count: 70))

        guard let library = try? device.makeLibrary(source: doubleBufferShaders, options: nil) else {
            print("Failed to compile double buffer shaders")
            return
        }

        let size = 256 * 1024
        let iterations = 10
        let passes = 100 // Number of ping-pong passes

        guard let bufferA = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared),
              let bufferB = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared) else {
            print("Failed to create buffers")
            return
        }

        var sizeValue = UInt32(size)

        print("\n--- Single Buffer Baseline ---")

        if let singleFunc = library.makeFunction(name: "single_buffer"),
           let singlePipeline = try? device.makeComputePipelineState(function: singleFunc) {
            let start = getTimeNanos()
            for _ in 0..<iterations {
                for _ in 0..<passes {
                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }
                    encoder.setComputePipelineState(singlePipeline)
                    encoder.setBuffer(bufferA, offset: 0, index: 0)
                    encoder.setBuffer(bufferB, offset: 0, index: 1)
                    encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 2)
                    encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    encoder.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()
                }
            }
            let end = getTimeNanos()
            let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations * passes)
            let gops = Double(size) / elapsed / 1e9
            print("Single Buffer: \(String(format: "%.3f", gops)) GOPS")
        }

        print("\n--- Double Buffer (Ping-Pong) ---")

        if let readAFunc = library.makeFunction(name: "double_buffer_read_a"),
           let readBPipeline = try? device.makeComputePipelineState(function: readAFunc),
           let readBFunc = library.makeFunction(name: "double_buffer_read_b"),
           let readAPipeline = try? device.makeComputePipelineState(function: readBFunc) {
            
            let start = getTimeNanos()
            for _ in 0..<iterations {
                for pass in 0..<passes {
                    if pass % 2 == 0 {
                        guard let cmd = queue.makeCommandBuffer(),
                              let encoder = cmd.makeComputeCommandEncoder() else { continue }
                        encoder.setComputePipelineState(readBPipeline)
                        encoder.setBuffer(bufferA, offset: 0, index: 0)
                        encoder.setBuffer(bufferB, offset: 0, index: 1)
                        encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 2)
                        encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                              threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                        encoder.endEncoding()
                        cmd.commit()
                        cmd.waitUntilCompleted()
                    } else {
                        guard let cmd = queue.makeCommandBuffer(),
                              let encoder = cmd.makeComputeCommandEncoder() else { continue }
                        encoder.setComputePipelineState(readAPipeline)
                        encoder.setBuffer(bufferB, offset: 0, index: 0)
                        encoder.setBuffer(bufferA, offset: 0, index: 1)
                        encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 2)
                        encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                              threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                        encoder.endEncoding()
                        cmd.commit()
                        cmd.waitUntilCompleted()
                    }
                }
            }
            let end = getTimeNanos()
            let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations * passes)
            let gops = Double(size) / elapsed / 1e9
            print("Double Buffer: \(String(format: "%.3f", gops)) GOPS")
        }

        print("\n--- Key Insights ---")
        print("1. Double buffering enables read-write overlap")
        print("2. Triple buffering adds extra buffer for max pipeline depth")
        print("3. Benefits appear with async execution, not synchronous wait")
        print("4. Use MTLCommandBuffer completion handlers for true overlap")
    }
}
