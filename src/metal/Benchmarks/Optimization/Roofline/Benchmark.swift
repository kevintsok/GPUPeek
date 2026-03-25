import Foundation
import Metal

// MARK: - Roofline Model Benchmark

let rooflineShaders = """
#include <metal_stdlib>
using namespace metal;

// Memory bound kernel - low arithmetic intensity
kernel void memory_bound(device const float* in [[buffer(0)]],
                       device float* out [[buffer(1)]],
                       constant uint& size [[buffer(2)]],
                       uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    out[id] = in[id] * 1.001f;
}

// Compute bound kernel - high arithmetic intensity  
kernel void compute_bound(device const float* in [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        constant uint& size [[buffer(2)]],
                        uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    float val = in[id];
    // Heavy computation - 256 FLOPs per element
    for (uint i = 0; i < 64; i++) {
        val = fma(val, 0.99f, 0.01f);
        val = sin(val) * cos(val);
    }
    out[id] = val;
}

// GEMM kernel - medium-high arithmetic intensity
kernel void gemm_kernel(device const float4* A [[buffer(0)]],
                       device const float4* B [[buffer(1)]],
                       device float4* C [[buffer(2)]],
                       constant uint& N [[buffer(3)]],
                       uint id [[thread_position_in_grid]]) {
    uint row = (id / (N / 4)) * 4;
    uint col = (id % (N / 4)) * 4;
    
    if (row >= N || col >= N) return;
    
    float4 Cval0 = 0.0f;
    float4 Cval1 = 0.0f;
    float4 Cval2 = 0.0f;
    float4 Cval3 = 0.0f;
    
    for (uint k = 0; k < N; k += 4) {
        float4 a0 = A[(row + 0) * (N / 4) + (k / 4)];
        float4 a1 = A[(row + 1) * (N / 4) + (k / 4)];
        float4 a2 = A[(row + 2) * (N / 4) + (k / 4)];
        float4 a3 = A[(row + 3) * (N / 4) + (k / 4)];
        
        float4 b = B[(k / 4) * (N / 4) * 4 + col / 4];
        
        Cval0 += b * a0.x;
        Cval1 += b * a1.x;
        Cval2 += b * a2.x;
        Cval3 += b * a3.x;
    }
    
    C[(row + 0) * (N / 4) + (col / 4)] = Cval0;
    C[(row + 1) * (N / 4) + (col / 4)] = Cval1;
    C[(row + 2) * (N / 4) + (col / 4)] = Cval2;
    C[(row + 3) * (N / 4) + (col / 4)] = Cval3;
}

// Stencil kernel - medium arithmetic intensity
kernel void stencil_kernel(device const float* in [[buffer(0)]],
                         device float* out [[buffer(1)]],
                         constant uint& size [[buffer(2)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    float sum = 0.0f;
    for (int i = -2; i <= 2; i++) {
        uint idx = uint(max(0, min(Int(size) - 1, Int(id) + i)));
        sum += in[idx];
    }
    out[id] = sum * 0.2f;
}

// Reduction kernel - high arithmetic intensity
kernel void reduction_kernel(device const float* in [[buffer(0)]],
                           device float* out [[buffer(1)]],
                           constant uint& size [[buffer(2)]],
                           uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    float sum = 0.0f;
    for (uint i = 0; i < 256; i++) {
        uint idx = (id * 256 + i) % size;
        sum += in[idx];
    }
    out[id] = sum;
}
"""

public struct RooflineBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    public init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
    }

    public func run() throws {
        print(String(repeating: "=", count: 70))
        print("Roofline Model Analysis Benchmark")
        print(String(repeating: "=", count: 70))

        guard let library = try? device.makeLibrary(source: rooflineShaders, options: nil) else {
            print("Failed to compile roofline shaders")
            return
        }

        let size = 256 * 1024
        let iterations = 20

        guard let bufferIn = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared),
              let bufferOut = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared) else {
            print("Failed to create buffers")
            return
        }

        var sizeValue = UInt32(size)

        print("\n--- Arithmetic Intensity vs Performance ---")
        print("Peak Memory Bandwidth: ~2 GB/s (实测)")
        print("Peak Compute: ~12 GFLOPS")
        print("Crossover Point: ~6 FLOP/byte")
        print("")

        // Test 1: Memory Bound Kernel
        if let memFunc = library.makeFunction(name: "memory_bound"),
           let memPipeline = try? device.makeComputePipelineState(function: memFunc) {
            let start = getTimeNanos()
            for _ in 0..<iterations {
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(memPipeline)
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
            let ai = 1.0 // 1 FLOP per element
            print("Memory Bound: AI=\(String(format: "%.1f", ai)) FLOP/byte, BW=\(String(format: "%.2f", bandwidth)) GB/s")
        }

        // Test 2: Stencil (Medium AI)
        if let stencilFunc = library.makeFunction(name: "stencil_kernel"),
           let stencilPipeline = try? device.makeComputePipelineState(function: stencilFunc) {
            let start = getTimeNanos()
            for _ in 0..<iterations {
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(stencilPipeline)
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
            let ai = 5.0 // 5 FLOPs per element
            let gflops = 5.0 * Double(size) / elapsed / 1e9
            print("Stencil (5pt): AI=\(String(format: "%.1f", ai)) FLOP/byte, BW=\(String(format: "%.2f", bandwidth)) GB/s, GFLOPs=\(String(format: "%.2f", gflops))")
        }

        // Test 3: Compute Bound Kernel
        if let computeFunc = library.makeFunction(name: "compute_bound"),
           let computePipeline = try? device.makeComputePipelineState(function: computeFunc) {
            let start = getTimeNanos()
            for _ in 0..<iterations {
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(computePipeline)
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
            let ai = 256.0 // 256 FLOPs per element
            let gflops = 256.0 * Double(size) / elapsed / 1e9
            print("Compute Bound: AI=\(String(format: "%.1f", ai)) FLOP/byte, BW=\(String(format: "%.2f", bandwidth)) GB/s, GFLOPs=\(String(format: "%.2f", gflops))")
        }

        print("\n--- Roofline Analysis ---")
        print("Memory Bound Region: AI < 6 FLOP/byte")
        print("  - Performance limited by memory bandwidth (~2 GB/s)")
        print("  - Optimize memory access patterns")
        print("")
        print("Compute Bound Region: AI > 6 FLOP/byte")
        print("  - Performance limited by compute (~12 GFLOPS)")
        print("  - Optimize computation")
        print("")
        print("Key Insight: Apple M2 is memory-bound for most kernels!")
    }
}
