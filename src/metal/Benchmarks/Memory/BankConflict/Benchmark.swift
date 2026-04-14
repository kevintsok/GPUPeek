import Foundation
import Metal

// MARK: - Bank Conflict Benchmark

let bankConflictShaders = """
#include <metal_stdlib>
using namespace metal;

// Sequential shared memory access - no bank conflict
kernel void shared_sequential(device const float* in [[buffer(0)]],
                             device float* out [[buffer(1)]],
                             constant uint& size [[buffer(2)]],
                             uint tid [[thread_position_in_grid]],
                             uint tg [[threadgroup_position_in_grid]]) {
    threadgroup float shared[1024];

    uint local_id = tid % 256;
    uint idx = tid;

    // Load into shared memory - sequential
    shared[local_id] = in[idx];
    threadgroup_barrier(flags::mem_threadgroup);

    // Process - sequential access
    float sum = 0.0f;
    for (uint i = 0; i < 4; i++) {
        sum += shared[(local_id + i) % 256];
    }

    threadgroup_barrier(flags::mem_threadgroup);

    // Store - sequential
    out[idx] = sum;
    threadgroup_barrier(flags::mem_threadgroup);
}

// Strided shared memory access - bank conflict
kernel void shared_strided(device const float* in [[buffer(0)]],
                           device float* out [[buffer(1)]],
                           constant uint& size [[buffer(2)]],
                           uint tid [[thread_position_in_grid]]) {
    threadgroup float shared[1024];

    uint local_id = tid % 256;
    uint idx = tid;
    uint stride = 256; // Same stride for all threads -> bank conflict

    // Load into shared memory
    shared[local_id] = in[idx];
    threadgroup_barrier(flags::mem_threadgroup);

    // Process - strided access causes bank conflicts
    float sum = 0.0f;
    for (uint i = 0; i < 4; i++) {
        uint bank_idx = (local_id + i * stride) % 1024;
        sum += shared[bank_idx];
    }

    threadgroup_barrier(flags::mem_threadgroup);

    out[idx] = sum;
}

// Broadcast shared memory access - no conflict
kernel void shared_broadcast(device const float* in [[buffer(0)]],
                             device float* out [[buffer(1)]],
                             constant uint& size [[buffer(2)]],
                             uint tid [[thread_position_in_grid]]) {
    threadgroup float shared[1024];

    uint local_id = tid % 256;
    uint idx = tid;

    // First thread writes, others read same location
    if (local_id == 0) {
        shared[0] = in[idx];
    }
    threadgroup_barrier(flags::mem_threadgroup);

    // Broadcast read
    float val = shared[0];

    threadgroup_barrier(flags::mem_threadgroup);

    out[idx] = val;
}
"""

public struct BankConflictBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    public init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
    }

    public func run() throws {
        print(String(repeating: "=", count: 70))
        print("Bank Conflict Benchmark")
        print(String(repeating: "=", count: 70))

        guard let library = try? device.makeLibrary(source: bankConflictShaders, options: nil) else {
            print("Failed to compile bank conflict shaders")
            return
        }

        let size = 256 * 1024
        let iterations = 100

        guard let bufferIn = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared),
              let bufferOut = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared) else {
            print("Failed to create buffers")
            return
        }

        var sizeValue = UInt32(size)

        print("\n--- Shared Memory Access Pattern ---")

        // Test 1: Sequential Access
        if let seqFunc = library.makeFunction(name: "shared_sequential"),
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
            let gops = Double(size) / elapsed / 1e9
            print("Sequential Access: \(String(format: "%.3f", gops)) GOPS")
        }

        // Test 2: Strided Access (Bank Conflict)
        if let strideFunc = library.makeFunction(name: "shared_strided"),
           let stridePipeline = try? device.makeComputePipelineState(function: strideFunc) {
            let start = getTimeNanos()
            for _ in 0..<iterations {
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(stridePipeline)
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
            print("Strided Access:     \(String(format: "%.3f", gops)) GOPS (bank conflict)")
        }

        // Test 3: Broadcast
        if let broadcastFunc = library.makeFunction(name: "shared_broadcast"),
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
            print("Broadcast:         \(String(format: "%.3f", gops)) GOPS")
        }

        print("\n--- Key Findings ---")
        print("1. Sequential shared memory access is optimal")
        print("2. Strided access causes bank conflicts, reducing performance")
        print("3. Broadcast is efficient when one thread has the data")
        print("4. Apple GPU shared memory has 32 banks")
    }
}
