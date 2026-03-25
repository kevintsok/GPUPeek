import Foundation
import Metal

// MARK: - Occupancy Benchmark

let occupancyShaders = """
#include <metal_stdlib>
using namespace metal;

// Kernel with low occupancy - few threads per group, high shared memory
kernel void low_occupancy(device const float* in [[buffer(0)]],
                         device float* out [[buffer(1)]],
                         threadgroup float* shared [[threadgroup(0)]],
                         constant uint& size [[buffer(2)]],
                         uint id [[thread_position_in_grid]]) {
    uint tid = id;
    if (tid >= size) return;
    
    // Use 1KB shared per threadgroup (32 threads max)
    uint local_id = tid % 32;
    shared[local_id * 32] = in[tid];
    threadgroup_barrier(flags::mem_threadgroup);
    
    float sum = 0.0f;
    for (uint i = 0; i < 32; i++) {
        sum += shared[i * 32];
    }
    
    out[tid] = sum;
}

// Kernel with medium occupancy
kernel void med_occupancy(device const float* in [[buffer(0)]],
                          device float* out [[buffer(1)]],
                          threadgroup float* shared [[threadgroup(0)]],
                          constant uint& size [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {
    uint tid = id;
    if (tid >= size) return;
    
    // Use 256B shared per threadgroup (128 threads)
    uint local_id = tid % 128;
    shared[local_id * 2] = in[tid];
    threadgroup_barrier(flags::mem_threadgroup);
    
    float sum = 0.0f;
    for (uint i = 0; i < 128; i++) {
        sum += shared[i * 2];
    }
    
    out[tid] = sum;
}

// Kernel with high occupancy - minimal shared memory
kernel void high_occupancy(device const float* in [[buffer(0)]],
                           device float* out [[buffer(1)]],
                           threadgroup float* shared [[threadgroup(0)]],
                           constant uint& size [[buffer(2)]],
                           uint id [[thread_position_in_grid]]) {
    uint tid = id;
    if (tid >= size) return;
    
    // Use only 64B shared per threadgroup (512 threads)
    uint local_id = tid % 512;
    shared[local_id] = in[tid];
    threadgroup_barrier(flags::mem_threadgroup);
    
    float sum = 0.0f;
    for (uint i = 0; i < 512; i++) {
        sum += shared[i];
    }
    
    out[tid] = sum;
}

// Register heavy kernel - compute bound
kernel void compute_bound(device const float* in [[buffer(0)]],
                         device float* out [[buffer(1)]],
                         constant uint& size [[buffer(2)]],
                         uint id [[thread_position_in_grid]]) {
    uint tid = id;
    if (tid >= size) return;
    
    float val = in[tid];
    // Heavy computation - reduces register pressure impact
    for (uint i = 0; i < 64; i++) {
        val = fma(val, 0.99f, 0.01f);
        val = sin(val) * cos(val);
    }
    
    out[tid] = val;
}

// Memory bound kernel - low compute
kernel void memory_bound(device const float* in [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        constant uint& size [[buffer(2)]],
                        uint id [[thread_position_in_grid]]) {
    uint tid = id;
    if (tid >= size) return;
    
    out[tid] = in[tid] * 1.001f;
}
"""

public struct OccupancyBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    public init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
    }

    public func run() throws {
        print(String(repeating: "=", count: 70))
        print("Occupancy Analysis Benchmark")
        print(String(repeating: "=", count: 70))

        guard let library = try? device.makeLibrary(source: occupancyShaders, options: nil) else {
            print("Failed to compile occupancy shaders")
            return
        }

        let size = 256 * 1024
        let iterations = 50

        guard let bufferIn = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared),
              let bufferOut = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared) else {
            print("Failed to create buffers")
            return
        }

        var sizeValue = UInt32(size)

        print("\n--- Occupancy Levels ---")

        // Test 1: Low Occupancy
        if let lowFunc = library.makeFunction(name: "low_occupancy"),
           let lowPipeline = try? device.makeComputePipelineState(function: lowFunc) {
            print("Max threads per threadgroup: \(lowPipeline.maxTotalThreadsPerThreadgroup)")
            print("Threadgroup memory: \(lowPipeline.threadgroupMemoryLength) bytes")
            
            let start = getTimeNanos()
            for _ in 0..<iterations {
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(lowPipeline)
                encoder.setBuffer(bufferIn, offset: 0, index: 0)
                encoder.setBuffer(bufferOut, offset: 0, index: 1)
                encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 2)
                encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                encoder.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }
            let end = getTimeNanos()
            let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
            let gops = Double(size) / elapsed / 1e9
            print("Low Occupancy (32 thr): \(String(format: "%.3f", gops)) GOPS")
        }

        // Test 2: Medium Occupancy
        if let medFunc = library.makeFunction(name: "med_occupancy"),
           let medPipeline = try? device.makeComputePipelineState(function: medFunc) {
            print("\nMax threads per threadgroup: \(medPipeline.maxTotalThreadsPerThreadgroup)")
            print("Threadgroup memory: \(medPipeline.threadgroupMemoryLength) bytes")
            
            let start = getTimeNanos()
            for _ in 0..<iterations {
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(medPipeline)
                encoder.setBuffer(bufferIn, offset: 0, index: 0)
                encoder.setBuffer(bufferOut, offset: 0, index: 1)
                encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 2)
                encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
                encoder.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }
            let end = getTimeNanos()
            let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
            let gops = Double(size) / elapsed / 1e9
            print("Medium Occupancy (128 thr): \(String(format: "%.3f", gops)) GOPS")
        }

        // Test 3: High Occupancy
        if let highFunc = library.makeFunction(name: "high_occupancy"),
           let highPipeline = try? device.makeComputePipelineState(function: highFunc) {
            print("\nMax threads per threadgroup: \(highPipeline.maxTotalThreadsPerThreadgroup)")
            print("Threadgroup memory: \(highPipeline.threadgroupMemoryLength) bytes")
            
            let start = getTimeNanos()
            for _ in 0..<iterations {
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(highPipeline)
                encoder.setBuffer(bufferIn, offset: 0, index: 0)
                encoder.setBuffer(bufferOut, offset: 0, index: 1)
                encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 2)
                encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 512, height: 1, depth: 1))
                encoder.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }
            let end = getTimeNanos()
            let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
            let gops = Double(size) / elapsed / 1e9
            print("High Occupancy (512 thr): \(String(format: "%.3f", gops)) GOPS")
        }

        print("\n--- Kernel Type vs Occupancy ---")

        // Test compute bound
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
            let gops = Double(size) / elapsed / 1e9
            print("Compute Bound: \(String(format: "%.3f", gops)) GOPS")
        }

        // Test memory bound
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
            let gops = Double(size) / elapsed / 1e9
            print("Memory Bound: \(String(format: "%.3f", gops)) GOPS")
        }

        print("\n--- Key Insights ---")
        print("1. Occupancy affects latency hiding ability")
        print("2. Memory-bound kernels benefit more from high occupancy")
        print("3. Compute-bound kernels can tolerate lower occupancy")
        print("4. Apple M2: 32KB max shared memory per threadgroup")
    }
}
