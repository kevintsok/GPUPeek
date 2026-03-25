import Foundation
import Metal

// MARK: - Register Pressure Benchmark

let registerPressureShaders = """
#include <metal_stdlib>
using namespace metal;

// Low register pressure: minimal variables
kernel void register_low(device const float* in [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        constant uint& size [[buffer(2)]],
                        uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    float a = in[id];
    out[id] = a + 1.0f;
}

// Medium register pressure: 4 variables
kernel void register_medium(device const float* in [[buffer(0)]],
                           device float* out [[buffer(1)]],
                           constant uint& size [[buffer(2)]],
                           uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    float a = in[id];
    float b = a * 2.0f;
    float c = b + a;
    float d = c * 0.5f;
    out[id] = d + 1.0f;
}

// High register pressure: 8 variables
kernel void register_high(device const float* in [[buffer(0)]],
                          device float* out [[buffer(1)]],
                          constant uint& size [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    float a = in[id];
    float b = a * 2.0f;
    float c = b * 3.0f;
    float d = c / 4.0f;
    float e = d + 5.0f;
    float f = e - 6.0f;
    float g = f * 7.0f;
    float h = g / 8.0f;
    out[id] = h + a + b + c + d + e + f + g + h;
}

// Very high register pressure: 16 variables
kernel void register_very_high(device const float* in [[buffer(0)]],
                               device float* out [[buffer(1)]],
                               constant uint& size [[buffer(2)]],
                               uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    float a0 = in[id];
    float a1 = a0 * 2.0f;
    float a2 = a1 * 3.0f;
    float a3 = a2 / 4.0f;
    float a4 = a3 + 5.0f;
    float a5 = a4 - 6.0f;
    float a6 = a5 * 7.0f;
    float a7 = a6 / 8.0f;
    float a8 = a7 + 9.0f;
    float a9 = a8 - 10.0f;
    float a10 = a9 * 11.0f;
    float a11 = a10 / 12.0f;
    float a12 = a11 + 13.0f;
    float a13 = a12 - 14.0f;
    float a14 = a13 * 15.0f;
    float a15 = a14 / 16.0f;
    out[id] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15;
}

// Loop-based register test: constant registers, loop variable reused
kernel void register_loop_low(device const float* in [[buffer(0)]],
                              device float* out [[buffer(1)]],
                              constant uint& size [[buffer(2)]],
                              uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    float sum = 0.0f;
    for (uint i = 0; i < 32; i++) {
        float val = in[(id + i) % size];
        sum += val;
    }
    out[id] = sum * 0.03125f;  // /32
}

// Loop with high register pressure inside loop body
kernel void register_loop_high(device const float* in [[buffer(0)]],
                               device float* out [[buffer(1)]],
                               constant uint& size [[buffer(2)]],
                               uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    float sum = 0.0f;
    for (uint i = 0; i < 32; i++) {
        float v0 = in[(id + i) % size];
        float v1 = v0 * 2.0f;
        float v2 = v1 + v0;
        float v3 = v2 * 0.5f;
        sum += v3;
    }
    out[id] = sum * 0.03125f;
}

// Shared memory bound (tests register vs shared trade-off)
kernel void shared_tradeoff_low(device const float* in [[buffer(0)]],
                                device float* out [[buffer(1)]],
                                threadgroup float* shared [[threadgroup(0)]],
                                constant uint& size [[buffer(2)]],
                                uint id [[thread_position_in_grid]],
                                uint lid [[thread_position_in_threadgroup]]) {
    if (id >= size) return;
    // Low register usage - rely on shared memory
    shared[lid] = in[id];
    threadgroup_barrier(mem_flags::mem_none);
    float sum = shared[lid];
    out[id] = sum;
}

kernel void shared_tradeoff_high(device const float* in [[buffer(0)]],
                                 device float* out [[buffer(1)]],
                                 threadgroup float* shared [[threadgroup(0)]],
                                 constant uint& size [[buffer(2)]],
                                 uint id [[thread_position_in_grid]],
                                 uint lid [[thread_position_in_threadgroup]]) {
    if (id >= size) return;
    // High register usage - compute in registers
    float v0 = in[id];
    float v1 = v0 * 2.0f;
    float v2 = v1 + v0;
    float v3 = v2 * 0.5f;
    float v4 = v3 + 1.0f;
    float v5 = v4 - 0.5f;
    float v6 = v5 * 1.5f;
    float v7 = v6 / 1.5f;
    shared[lid] = v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7;
    threadgroup_barrier(mem_flags::mem_none);
    out[id] = shared[lid];
}
"""

public struct RegisterPressureBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    public init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
    }

    public func run() throws {
        print(String(repeating: "=", count: 70))
        print("Register Pressure Benchmark")
        print(String(repeating: "=", count: 70))

        guard let library = try? device.makeLibrary(source: registerPressureShaders, options: nil) else {
            print("Failed to compile register pressure shaders")
            return
        }

        let sizes = [65536, 262144, 1048576]
        let iterations = 50

        for size in sizes {
            print("\n--- Array Size: \(size) ---")

            guard let bufferIn = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared),
                  let bufferOut = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared) else {
                continue
            }

            var sizeValue = UInt32(size)

            // Initialize input
            let inPtr = bufferIn.contents().bindMemory(to: Float.self, capacity: size)
            for i in 0..<size {
                inPtr[i] = Float(i % 256) / 255.0
            }

            // Low register
            if let func_ = library.makeFunction(name: "register_low"),
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
                print("Low Reg (1 var): \(String(format: "%.2f", gops)) GOPS (\(String(format: "%.4f", elapsed * 1000)) ms)")
            }

            // Medium register
            if let func_ = library.makeFunction(name: "register_medium"),
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
                print("Medium Reg (4 var): \(String(format: "%.2f", gops)) GOPS (\(String(format: "%.4f", elapsed * 1000)) ms)")
            }

            // High register
            if let func_ = library.makeFunction(name: "register_high"),
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
                print("High Reg (8 var): \(String(format: "%.2f", gops)) GOPS (\(String(format: "%.4f", elapsed * 1000)) ms)")
            }

            // Very high register
            if let func_ = library.makeFunction(name: "register_very_high"),
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
                print("Very High Reg (16 var): \(String(format: "%.2f", gops)) GOPS (\(String(format: "%.4f", elapsed * 1000)) ms)")
            }

            // Loop low
            if let func_ = library.makeFunction(name: "register_loop_low"),
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
                let gops = Double(size * 32) / elapsed / 1e9
                print("Loop Low Reg: \(String(format: "%.2f", gops)) GOPS (\(String(format: "%.4f", elapsed * 1000)) ms)")
            }

            // Loop high
            if let func_ = library.makeFunction(name: "register_loop_high"),
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
                let gops = Double(size * 32) / elapsed / 1e9
                print("Loop High Reg: \(String(format: "%.2f", gops)) GOPS (\(String(format: "%.4f", elapsed * 1000)) ms)")
            }
        }

        print("\n--- Key Findings ---")
        print("1. Low register pressure: minimal variables, potentially more occupancy")
        print("2. High register pressure: more computation in-flight but lower occupancy")
        print("3. Very high registers (16+): may cause register spilling to memory")
        print("4. Loop body register usage affects kernel performance significantly")
        print("5. Trade-off: registers vs shared memory vs occupancy")
    }
}
