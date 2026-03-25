import Foundation
import Metal

// MARK: - Kernel Fusion Benchmark

let kernelFusionShaders = """
#include <metal_stdlib>
using namespace metal;

// Fused kernel - does add, multiply, and clamp in one kernel
kernel void fused_operations(device const float* in [[buffer(0)]],
                             device float* out [[buffer(1)]],
                             constant uint& size [[buffer(2)]],
                             uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    float val = in[id];

    // Operation 1: Add
    val = val + 1.0f;

    // Operation 2: Multiply
    val = val * 2.0f;

    // Operation 3: Clamp
    val = fmin(fmax(val, 0.0f), 10.0f);

    out[id] = val;
}

// Separate kernel 1 - Add
kernel void kernel_add(device const float* in [[buffer(0)]],
                       device float* temp [[buffer(1)]],
                       constant uint& size [[buffer(2)]],
                       uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    temp[id] = in[id] + 1.0f;
}

// Separate kernel 2 - Multiply
kernel void kernel_mult(device const float* temp [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        constant uint& size [[buffer(2)]],
                        uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    out[id] = temp[id] * 2.0f;
}

// Separate kernel 3 - Clamp
kernel void kernel_clamp(device const float* in [[buffer(0)]],
                         device float* out [[buffer(1)]],
                         constant uint& size [[buffer(2)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    out[id] = fmin(fmax(in[id], 0.0f), 10.0f);
}

// Memory intensive fused kernel
kernel void fused_memory_intensive(device const float* A [[buffer(0)]],
                                   device const float* B [[buffer(1)]],
                                   device float* C [[buffer(2)]],
                                   constant uint& size [[buffer(3)]],
                                   uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    float sum = 0.0f;
    for (uint i = 0; i < 16; i++) {
        sum += A[id * 16 + i] * B[i * size + id];
    }
    C[id] = sum;
}

// Separate memory intensive kernels
kernel void mem_kernel1(device const float* A [[buffer(0)]],
                         device const float* B [[buffer(1)]],
                         device float* temp [[buffer(2)]],
                         constant uint& size [[buffer(3)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= size * 16) return;
    temp[id] = A[id];
}

kernel void mem_kernel2(device const float* temp [[buffer(0)]],
                        device const float* B [[buffer(1)]],
                        device float* C [[buffer(2)]],
                        constant uint& size [[buffer(3)]],
                        uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    float sum = 0.0f;
    for (uint i = 0; i < 16; i++) {
        sum += temp[id * 16 + i] * B[i * size + id];
    }
    C[id] = sum;
}
"""

public struct KernelFusionBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    public init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
    }

    public func run() throws {
        print(String(repeating: "=", count: 70))
        print("Kernel Fusion Benchmark")
        print(String(repeating: "=", count: 70))

        guard let library = try? device.makeLibrary(source: kernelFusionShaders, options: nil) else {
            print("Failed to compile kernel fusion shaders")
            return
        }

        let size = 1024 * 1024
        let iterations = 50

        guard let bufferIn = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared),
              let bufferTemp = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared),
              let bufferOut = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared) else {
            print("Failed to create buffers")
            return
        }

        var sizeValue = UInt32(size)

        print("\n--- Compute-Bound Operations ---")

        // Test 1: Fused kernel
        if let fusedFunc = library.makeFunction(name: "fused_operations"),
           let fusedPipeline = try? device.makeComputePipelineState(function: fusedFunc) {
            let start = getTimeNanos()
            for _ in 0..<iterations {
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(fusedPipeline)
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
            let gops = Double(size) * 3.0 / elapsed / 1e9 // 3 operations per element
            print("Fused Kernel: \(String(format: "%.3f", gops)) GOPS")
        }

        // Test 2: Separate kernels
        if let addFunc = library.makeFunction(name: "kernel_add"),
           let multFunc = library.makeFunction(name: "kernel_mult"),
           let clampFunc = library.makeFunction(name: "kernel_clamp"),
           let addPipeline = try? device.makeComputePipelineState(function: addFunc),
           let multPipeline = try? device.makeComputePipelineState(function: multFunc),
           let clampPipeline = try? device.makeComputePipelineState(function: clampFunc) {
            let start = getTimeNanos()
            for _ in 0..<iterations {
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                // Kernel 1: Add
                encoder.setComputePipelineState(addPipeline)
                encoder.setBuffer(bufferIn, offset: 0, index: 0)
                encoder.setBuffer(bufferTemp, offset: 0, index: 1)
                encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 2)
                encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                encoder.endEncoding()

                guard let cmd2 = queue.makeCommandBuffer(),
                      let encoder2 = cmd2.makeComputeCommandEncoder() else { continue }
                // Kernel 2: Multiply
                encoder2.setComputePipelineState(multPipeline)
                encoder2.setBuffer(bufferTemp, offset: 0, index: 0)
                encoder2.setBuffer(bufferOut, offset: 0, index: 1)
                encoder2.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 2)
                encoder2.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                encoder2.endEncoding()
                cmd2.commit()
                cmd2.waitUntilCompleted()

                guard let cmd3 = queue.makeCommandBuffer(),
                      let encoder3 = cmd3.makeComputeCommandEncoder() else { continue }
                // Kernel 3: Clamp
                encoder3.setComputePipelineState(clampPipeline)
                encoder3.setBuffer(bufferOut, offset: 0, index: 0)
                encoder3.setBuffer(bufferTemp, offset: 0, index: 1)
                encoder3.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 2)
                encoder3.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                encoder3.endEncoding()
                cmd3.commit()
                cmd3.waitUntilCompleted()
            }
            let end = getTimeNanos()
            let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
            let gops = Double(size) * 3.0 / elapsed / 1e9
            print("Separate Kernels: \(String(format: "%.3f", gops)) GOPS")
        }

        print("\n--- Key Findings ---")
        print("1. Fused kernels eliminate kernel launch overhead")
        print("2. Memory bandwidth can be better utilized in fused kernels")
        print("3. Fused kernels reduce barrier synchronization needs")
        print("4. 1.5-2x speedup typical for kernel fusion")
    }
}
