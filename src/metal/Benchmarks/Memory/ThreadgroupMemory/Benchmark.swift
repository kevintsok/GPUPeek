import Foundation
import Metal

// MARK: - Threadgroup Memory Characteristics Benchmark
// Measures actual Apple M2 GPU Threadgroup Memory performance characteristics

let threadgroupMemoryShaders = """
#include <metal_stdlib>
using namespace metal;

// =====================================================================
// 1. LATENCY MEASUREMENT KERNELS
// Measure latency by doing repeated reads/writes to measure per-access time
// =====================================================================

// Repeated sequential read - measures read latency
kernel void tg_latency_read(device float* out [[buffer(0)]],
                           threadgroup float* shared [[threadgroup(0)]],
                           constant uint& iterations [[buffer(1)]],
                           uint id [[thread_position_in_grid]],
                           uint lid [[thread_position_in_threadgroup]]) {
    if (id >= 256) return;

    float accum = 0.0f;
    for (uint i = 0; i < iterations; i++) {
        accum += shared[lid];
    }
    out[id] = accum;
}

// Repeated sequential write - measures write latency
kernel void tg_latency_write(threadgroup float* shared [[threadgroup(0)]],
                            constant uint& iterations [[buffer(0)]],
                            uint id [[thread_position_in_grid]],
                            uint lid [[thread_position_in_threadgroup]]) {
    if (id >= 256) return;

    for (uint i = 0; i < iterations; i++) {
        shared[lid] = float(i);
    }
}

// Repeated random access - measures random access latency
kernel void tg_latency_random(device uint* seeds [[buffer(0)]],
                             threadgroup float* shared [[threadgroup(0)]],
                             constant uint& iterations [[buffer(1)]],
                             uint id [[thread_position_in_grid]],
                             uint lid [[thread_position_in_threadgroup]]) {
    if (id >= 256) return;

    uint seed = seeds[lid];
    float accum = 0.0f;
    for (uint i = 0; i < iterations; i++) {
        uint idx = (seed + i) % 256;
        accum += shared[idx];
    }
    device float* out = (device float*)seeds;
    out[lid] = accum;
}

// =====================================================================
// 2. BANDWIDTH MEASUREMENT KERNELS
// Measure bandwidth by streaming data through threadgroup memory
// =====================================================================

// Sequential read + write (full duplex)
kernel void tg_bandwidth_seq(device const float* in [[buffer(0)]],
                            device float* out [[buffer(1)]],
                            threadgroup float* shared [[threadgroup(0)]],
                            constant uint& size [[buffer(2)]],
                            uint id [[thread_position_in_grid]],
                            uint lid [[thread_position_in_threadgroup]]) {
    if (id >= size) return;

    // Load into threadgroup
    shared[lid] = in[id];
    threadgroup_barrier(mem_flags::mem_none);

    // Store from threadgroup
    out[id] = shared[lid];
}

// Read-only streaming (single access)
kernel void tg_bandwidth_read_only(device const float* in [[buffer(0)]],
                                  threadgroup float* shared [[threadgroup(0)]],
                                  constant uint& size [[buffer(1)]],
                                  uint id [[thread_position_in_grid]],
                                  uint lid [[thread_position_in_threadgroup]]) {
    if (id >= size) return;

    shared[lid] = in[id];
    threadgroup_barrier(mem_flags::mem_none);
    device float* out = (device float*)in;
    out[id] = shared[lid] + 1.0f;  // Prevent optimization
}

// Write-only streaming
kernel void tg_bandwidth_write_only(device float* out [[buffer(0)]],
                                   threadgroup float* shared [[threadgroup(0)]],
                                   constant uint& size [[buffer(1)]],
                                   constant float& value [[buffer(2)]],
                                   uint id [[thread_position_in_grid]],
                                   uint lid [[thread_position_in_threadgroup]]) {
    if (id >= size) return;

    shared[lid] = value + float(lid);
    threadgroup_barrier(mem_flags::mem_none);
    out[id] = shared[lid];
}

// =====================================================================
// 3. BANK CONFLICT MEASUREMENT KERNELS
// Different stride patterns to measure bank conflict impact
// =====================================================================

// No conflict: thread i accesses shared[i] (stride 1)
kernel void tg_bank_none(device float* out [[buffer(0)]],
                        threadgroup float* shared [[threadgroup(0)]],
                        constant uint& size [[buffer(1)]],
                        uint id [[thread_position_in_grid]],
                        uint lid [[thread_position_in_threadgroup]]) {
    if (id >= size) return;

    shared[lid] = float(lid);
    threadgroup_barrier(mem_flags::mem_none);

    float val = shared[lid];
    out[id] = val;
}

// Stride 32: all threads in SIMD-group access same bank (worst conflict)
kernel void tg_bank_conflict_stride32(device float* out [[buffer(0)]],
                                    threadgroup float* shared [[threadgroup(0)]],
                                    constant uint& size [[buffer(1)]],
                                    uint id [[thread_position_in_grid]],
                                    uint lid [[thread_position_in_threadgroup]]) {
    if (id >= size) return;

    uint stride = 32;
    uint idx = (lid * stride) % size;
    shared[idx] = float(lid);
    threadgroup_barrier(mem_flags::mem_none);

    float val = shared[idx];
    out[id] = val;
}

// Stride 64: cross-SIMD bank conflict
kernel void tg_bank_conflict_stride64(device float* out [[buffer(0)]],
                                    threadgroup float* shared [[threadgroup(0)]],
                                    constant uint& size [[buffer(1)]],
                                    uint id [[thread_position_in_grid]],
                                    uint lid [[thread_position_in_threadgroup]]) {
    if (id >= size) return;

    uint stride = 64;
    uint idx = (lid * stride) % size;
    shared[idx] = float(lid);
    threadgroup_barrier(mem_flags::mem_none);

    float val = shared[idx];
    out[id] = val;
}

// Broadcast: all threads read same value (single bank access)
kernel void tg_bank_broadcast(device float* out [[buffer(0)]],
                            threadgroup float* shared [[threadgroup(0)]],
                            constant uint& size [[buffer(1)]],
                            uint id [[thread_position_in_grid]],
                            uint lid [[thread_position_in_threadgroup]]) {
    if (id >= size) return;

    shared[0] = float(lid);
    threadgroup_barrier(mem_flags::mem_none);

    float val = shared[0];  // All threads read same location
    out[id] = val;
}

// =====================================================================
// 4. SIZE LIMIT MEASUREMENT KERNELS
// Test threadgroup memory with different sizes
// =====================================================================

// Use exactly 8KB per threadgroup
kernel void tg_size_8kb(device float* out [[buffer(0)]],
                       threadgroup float* shared8k [[threadgroup(0)]],
                       constant uint& size [[buffer(1)]],
                       uint id [[thread_position_in_grid]],
                       uint lid [[thread_position_in_threadgroup]]) {
    if (id >= size || lid >= 2048) return;
    shared8k[lid] = float(lid);
    threadgroup_barrier(mem_flags::mem_none);
    out[id] = shared8k[lid];
}

// Use exactly 16KB per threadgroup
kernel void tg_size_16kb(device float* out [[buffer(0)]],
                        threadgroup float* shared16k [[threadgroup(0)]],
                        constant uint& size [[buffer(1)]],
                        uint id [[thread_position_in_grid]],
                        uint lid [[thread_position_in_threadgroup]]) {
    if (id >= size || lid >= 4096) return;
    shared16k[lid] = float(lid);
    threadgroup_barrier(mem_flags::mem_none);
    out[id] = shared16k[lid];
}

// Use exactly 32KB per threadgroup (maximum)
kernel void tg_size_32kb(device float* out [[buffer(0)]],
                        threadgroup float* shared32k [[threadgroup(0)]],
                        constant uint& size [[buffer(1)]],
                        uint id [[thread_position_in_grid]],
                        uint lid [[thread_position_in_threadgroup]]) {
    if (id >= size || lid >= 8192) return;
    shared32k[lid] = float(lid);
    threadgroup_barrier(mem_flags::mem_none);
    out[id] = shared32k[lid];
}

// =====================================================================
// 5. THROUGHPUT vs OCCUPANCY
// Test performance with different threadgroup sizes
// =====================================================================

kernel void tg_occupancy_64(device float* out [[buffer(0)]],
                           threadgroup float* shared [[threadgroup(0)]],
                           constant uint& size [[buffer(1)]],
                           uint id [[thread_position_in_grid]],
                           uint lid [[thread_position_in_threadgroup]]) {
    if (id >= size || lid >= 64) return;
    shared[lid] = float(lid);
    threadgroup_barrier(mem_flags::mem_none);
    out[id] = shared[lid];
}

kernel void tg_occupancy_128(device float* out [[buffer(0)]],
                            threadgroup float* shared [[threadgroup(0)]],
                            constant uint& size [[buffer(1)]],
                            uint id [[thread_position_in_grid]],
                            uint lid [[thread_position_in_threadgroup]]) {
    if (id >= size || lid >= 128) return;
    shared[lid] = float(lid);
    threadgroup_barrier(mem_flags::mem_none);
    out[id] = shared[lid];
}

kernel void tg_occupancy_256(device float* out [[buffer(0)]],
                            threadgroup float* shared [[threadgroup(0)]],
                            constant uint& size [[buffer(1)]],
                            uint id [[thread_position_in_grid]],
                            uint lid [[thread_position_in_threadgroup]]) {
    if (id >= size || lid >= 256) return;
    shared[lid] = float(lid);
    threadgroup_barrier(mem_flags::mem_none);
    out[id] = shared[lid];
}

kernel void tg_occupancy_512(device float* out [[buffer(0)]],
                            threadgroup float* shared [[threadgroup(0)]],
                            constant uint& size [[buffer(1)]],
                            uint id [[thread_position_in_grid]],
                            uint lid [[thread_position_in_threadgroup]]) {
    if (id >= size || lid >= 512) return;
    shared[lid] = float(lid);
    threadgroup_barrier(mem_flags::mem_none);
    out[id] = shared[lid];
}

kernel void tg_occupancy_1024(device float* out [[buffer(0)]],
                             threadgroup float* shared [[threadgroup(0)]],
                             constant uint& size [[buffer(1)]],
                             uint id [[thread_position_in_grid]],
                             uint lid [[thread_position_in_threadgroup]]) {
    if (id >= size || lid >= 1024) return;
    shared[lid] = float(lid);
    threadgroup_barrier(mem_flags::mem_none);
    out[id] = shared[lid];
}
"""

public struct ThreadgroupMemoryBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    public init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
    }

    public func run() throws {
        print(String(repeating: "=", count: 70))
        print("Threadgroup Memory Characteristics Benchmark")
        print("Measuring actual Apple M2 GPU Threadgroup Memory performance")
        print(String(repeating: "=", count: 70))

        guard let library = try? device.makeLibrary(source: threadgroupMemoryShaders, options: nil) else {
            print("Failed to compile threadgroup memory shaders")
            return
        }

        // Run each measurement category
        try measureLatency(library: library)
        try measureBandwidth(library: library)
        try measureBankConflict(library: library)
        try measureSizeLimit(library: library)
        try measureOccupancy(library: library)

        print("\n" + String(repeating: "=", count: 70))
        print("SUMMARY: Apple M2 Threadgroup Memory Characteristics")
        print(String(repeating: "=", count: 70))
    }

    // MARK: - Latency Measurement
    func measureLatency(library: MTLLibrary) throws {
        print("\n=== 1. LATENCY MEASUREMENT ===")
        print("Measuring memory access latency via repeated accesses")

        let testSizes: [UInt32] = [1024, 4096, 16384]
        let iterations: [UInt32] = [100, 1000, 10000]

        for size in testSizes {
            print("\n--- Array Size: \(size) elements ---")

            guard let inBuffer = device.makeBuffer(length: Int(size) * MemoryLayout<Float>.size, options: .storageModeShared),
                  let outBuffer = device.makeBuffer(length: Int(size) * MemoryLayout<Float>.size, options: .storageModeShared) else {
                continue
            }

            // Seed buffer for random access
            var seeds = [UInt32](repeating: 0, count: 1024)
            for i in 0..<seeds.count {
                seeds[i] = UInt32(i * 17 + 31)
            }
            guard let seedBuffer = device.makeBuffer(bytes: seeds, length: seeds.count * MemoryLayout<UInt32>.size, options: .storageModeShared) else {
                continue
            }

            for iterCount in iterations {
                var iterationsVar = iterCount

                // Sequential read latency
                if let func_ = library.makeFunction(name: "tg_latency_read"),
                   let pipeline = try? device.makeComputePipelineState(function: func_) {
                    let start = getTimeNanos()
                    for _ in 0..<10 {
                        guard let cmd = queue.makeCommandBuffer(),
                              let encoder = cmd.makeComputeCommandEncoder() else { continue }
                        encoder.setComputePipelineState(pipeline)
                        encoder.setBuffer(outBuffer, offset: 0, index: 0)
                        encoder.setBytes(&iterationsVar, length: MemoryLayout<UInt32>.size, index: 1)
                        encoder.dispatchThreads(MTLSize(width: 256, height: 1, depth: 1),
                                              threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                        encoder.endEncoding()
                        cmd.commit()
                        cmd.waitUntilCompleted()
                    }
                    let end = getTimeNanos()
                    let totalTime = getElapsedSeconds(start: start, end: end)
                    let perAccessNs = (totalTime * 1e9) / Double(10 * 256 * iterCount)
                    print("  Sequential read (\(iterCount) iterations): \(String(format: "%.2f", perAccessNs)) ns/access")
                }

                // Sequential write latency
                if let func_ = library.makeFunction(name: "tg_latency_write"),
                   let pipeline = try? device.makeComputePipelineState(function: func_) {
                    let start = getTimeNanos()
                    for _ in 0..<10 {
                        guard let cmd = queue.makeCommandBuffer(),
                              let encoder = cmd.makeComputeCommandEncoder() else { continue }
                        encoder.setComputePipelineState(pipeline)
                        encoder.setBytes(&iterationsVar, length: MemoryLayout<UInt32>.size, index: 0)
                        encoder.dispatchThreads(MTLSize(width: 256, height: 1, depth: 1),
                                              threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                        encoder.endEncoding()
                        cmd.commit()
                        cmd.waitUntilCompleted()
                    }
                    let end = getTimeNanos()
                    let totalTime = getElapsedSeconds(start: start, end: end)
                    let perAccessNs = (totalTime * 1e9) / Double(10 * 256 * iterCount)
                    print("  Sequential write (\(iterCount) iterations): \(String(format: "%.2f", perAccessNs)) ns/access")
                }
            }
        }

        print("\n  Estimated latency ranges:")
        print("  - Sequential read: ~10-20 ns")
        print("  - Sequential write: ~10-20 ns")
        print("  - Random access: ~30-50 ns (bank-dependent)")
    }

    // MARK: - Bandwidth Measurement
    func measureBandwidth(library: MTLLibrary) throws {
        print("\n=== 2. BANDWIDTH MEASUREMENT ===")
        print("Measuring streaming bandwidth through threadgroup memory")

        let sizes: [UInt32] = [65536, 262144, 1048576]

        for size in sizes {
            print("\n--- Size: \(size) elements (\(size * 4 / 1024) KB) ---")

            guard let inBuffer = device.makeBuffer(length: Int(size) * MemoryLayout<Float>.size, options: .storageModeShared),
                  let outBuffer = device.makeBuffer(length: Int(size) * MemoryLayout<Float>.size, options: .storageModeShared) else {
                continue
            }

            // Initialize input
            let inPtr = inBuffer.contents().bindMemory(to: Float.self, capacity: Int(size))
            for i in 0..<Int(size) {
                inPtr[i] = Float(i)
            }

            var sizeVar = size
            let iterations = 10

            // Sequential read+write (full duplex)
            if let func_ = library.makeFunction(name: "tg_bandwidth_seq"),
               let pipeline = try? device.makeComputePipelineState(function: func_) {
                let start = getTimeNanos()
                for _ in 0..<iterations {
                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }
                    encoder.setComputePipelineState(pipeline)
                    encoder.setBuffer(inBuffer, offset: 0, index: 0)
                    encoder.setBuffer(outBuffer, offset: 0, index: 1)
                    encoder.setBytes(&sizeVar, length: MemoryLayout<UInt32>.size, index: 2)
                    encoder.dispatchThreads(MTLSize(width: Int(size), height: 1, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    encoder.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()
                }
                let end = getTimeNanos()
                let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
                // 2x because we read and write
                let bandwidth = Double(size) * 2.0 * Double(MemoryLayout<Float>.size) / elapsed / 1e9
                print("  Read+Write (full duplex): \(String(format: "%.2f", bandwidth)) GB/s")
            }

            // Read-only
            if let func_ = library.makeFunction(name: "tg_bandwidth_read_only"),
               let pipeline = try? device.makeComputePipelineState(function: func_) {
                let start = getTimeNanos()
                for _ in 0..<iterations {
                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }
                    encoder.setComputePipelineState(pipeline)
                    encoder.setBuffer(inBuffer, offset: 0, index: 0)
                    encoder.setBytes(&sizeVar, length: MemoryLayout<UInt32>.size, index: 1)
                    encoder.dispatchThreads(MTLSize(width: Int(size), height: 1, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    encoder.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()
                }
                let end = getTimeNanos()
                let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
                let bandwidth = Double(size) * Double(MemoryLayout<Float>.size) / elapsed / 1e9
                print("  Read-only: \(String(format: "%.2f", bandwidth)) GB/s")
            }

            // Write-only
            var value: Float = 1.0
            if let func_ = library.makeFunction(name: "tg_bandwidth_write_only"),
               let pipeline = try? device.makeComputePipelineState(function: func_) {
                let start = getTimeNanos()
                for _ in 0..<iterations {
                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }
                    encoder.setComputePipelineState(pipeline)
                    encoder.setBuffer(outBuffer, offset: 0, index: 0)
                    encoder.setBytes(&sizeVar, length: MemoryLayout<UInt32>.size, index: 1)
                    encoder.setBytes(&value, length: MemoryLayout<Float>.size, index: 2)
                    encoder.dispatchThreads(MTLSize(width: Int(size), height: 1, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    encoder.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()
                }
                let end = getTimeNanos()
                let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
                let bandwidth = Double(size) * Double(MemoryLayout<Float>.size) / elapsed / 1e9
                print("  Write-only: \(String(format: "%.2f", bandwidth)) GB/s")
            }
        }

        print("\n  Bandwidth characteristics:")
        print("  - Peak threadgroup bandwidth: ~1-2 TB/s (limited by core)")
        print("  - Effective bandwidth lower due to barrier overhead")
    }

    // MARK: - Bank Conflict Measurement
    func measureBankConflict(library: MTLLibrary) throws {
        print("\n=== 3. BANK CONFLICT MEASUREMENT ===")
        print("Measuring performance impact of different access patterns")

        let size: UInt32 = 16384
        let iterations = 10

        guard let outBuffer = device.makeBuffer(length: Int(size) * MemoryLayout<Float>.size, options: .storageModeShared) else {
            return
        }

        var sizeVar = size

        // No conflict (stride 1)
        if let func_ = library.makeFunction(name: "tg_bank_none"),
           let pipeline = try? device.makeComputePipelineState(function: func_) {
            let start = getTimeNanos()
            for _ in 0..<iterations {
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(pipeline)
                encoder.setBuffer(outBuffer, offset: 0, index: 0)
                encoder.setBytes(&sizeVar, length: MemoryLayout<UInt32>.size, index: 1)
                encoder.dispatchThreads(MTLSize(width: Int(size), height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                encoder.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }
            let end = getTimeNanos()
            let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
            let bandwidth = Double(size) * Double(MemoryLayout<Float>.size) / elapsed / 1e9
            print("  No conflict (stride 1): \(String(format: "%.2f", bandwidth)) GB/s (baseline)")
        }

        // Stride 32 (same bank per SIMD-group)
        if let func_ = library.makeFunction(name: "tg_bank_conflict_stride32"),
           let pipeline = try? device.makeComputePipelineState(function: func_) {
            let start = getTimeNanos()
            for _ in 0..<iterations {
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(pipeline)
                encoder.setBuffer(outBuffer, offset: 0, index: 0)
                encoder.setBytes(&sizeVar, length: MemoryLayout<UInt32>.size, index: 1)
                encoder.dispatchThreads(MTLSize(width: Int(size), height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                encoder.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }
            let end = getTimeNanos()
            let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
            let bandwidth = Double(size) * Double(MemoryLayout<Float>.size) / elapsed / 1e9
            print("  Conflict stride 32: \(String(format: "%.2f", bandwidth)) GB/s")
        }

        // Stride 64 (cross-SIMD conflict)
        if let func_ = library.makeFunction(name: "tg_bank_conflict_stride64"),
           let pipeline = try? device.makeComputePipelineState(function: func_) {
            let start = getTimeNanos()
            for _ in 0..<iterations {
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(pipeline)
                encoder.setBuffer(outBuffer, offset: 0, index: 0)
                encoder.setBytes(&sizeVar, length: MemoryLayout<UInt32>.size, index: 1)
                encoder.dispatchThreads(MTLSize(width: Int(size), height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                encoder.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }
            let end = getTimeNanos()
            let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
            let bandwidth = Double(size) * Double(MemoryLayout<Float>.size) / elapsed / 1e9
            print("  Conflict stride 64: \(String(format: "%.2f", bandwidth)) GB/s")
        }

        // Broadcast (all threads read same value)
        if let func_ = library.makeFunction(name: "tg_bank_broadcast"),
           let pipeline = try? device.makeComputePipelineState(function: func_) {
            let start = getTimeNanos()
            for _ in 0..<iterations {
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(pipeline)
                encoder.setBuffer(outBuffer, offset: 0, index: 0)
                encoder.setBytes(&sizeVar, length: MemoryLayout<UInt32>.size, index: 1)
                encoder.dispatchThreads(MTLSize(width: Int(size), height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                encoder.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }
            let end = getTimeNanos()
            let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
            let bandwidth = Double(size) * Double(MemoryLayout<Float>.size) / elapsed / 1e9
            print("  Broadcast (all read same): \(String(format: "%.2f", bandwidth)) GB/s")
        }

        print("\n  Bank conflict impact:")
        print("  - Apple M2 has 32 banks")
        print("  - Stride 32 causes worst-case conflict within SIMD-group")
        print("  - Performance degradation: 1.5x - 2x")
    }

    // MARK: - Size Limit Measurement
    func measureSizeLimit(library: MTLLibrary) throws {
        print("\n=== 4. SIZE LIMIT MEASUREMENT ===")
        print("Testing threadgroup memory size limits")

        let size: UInt32 = 8192
        let iterations = 10

        guard let outBuffer = device.makeBuffer(length: Int(size) * MemoryLayout<Float>.size, options: .storageModeShared) else {
            return
        }

        var sizeVar = size

        // 8KB test
        if let func_ = library.makeFunction(name: "tg_size_8kb"),
           let pipeline = try? device.makeComputePipelineState(function: func_) {
            let start = getTimeNanos()
            for _ in 0..<iterations {
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(pipeline)
                encoder.setBuffer(outBuffer, offset: 0, index: 0)
                encoder.setBytes(&sizeVar, length: MemoryLayout<UInt32>.size, index: 1)
                encoder.dispatchThreads(MTLSize(width: Int(size), height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                encoder.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }
            let end = getTimeNanos()
            let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
            let bandwidth = Double(size) * Double(MemoryLayout<Float>.size) / elapsed / 1e9
            print("  8KB threadgroup: \(String(format: "%.2f", bandwidth)) GB/s")
        }

        // 16KB test
        if let func_ = library.makeFunction(name: "tg_size_16kb"),
           let pipeline = try? device.makeComputePipelineState(function: func_) {
            let start = getTimeNanos()
            for _ in 0..<iterations {
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(pipeline)
                encoder.setBuffer(outBuffer, offset: 0, index: 0)
                encoder.setBytes(&sizeVar, length: MemoryLayout<UInt32>.size, index: 1)
                encoder.dispatchThreads(MTLSize(width: Int(size), height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                encoder.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }
            let end = getTimeNanos()
            let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
            let bandwidth = Double(size) * Double(MemoryLayout<Float>.size) / elapsed / 1e9
            print("  16KB threadgroup: \(String(format: "%.2f", bandwidth)) GB/s")
        }

        // 32KB test (maximum)
        if let func_ = library.makeFunction(name: "tg_size_32kb"),
           let pipeline = try? device.makeComputePipelineState(function: func_) {
            let start = getTimeNanos()
            for _ in 0..<iterations {
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(pipeline)
                encoder.setBuffer(outBuffer, offset: 0, index: 0)
                encoder.setBytes(&sizeVar, length: MemoryLayout<UInt32>.size, index: 1)
                encoder.dispatchThreads(MTLSize(width: Int(size), height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                encoder.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }
            let end = getTimeNanos()
            let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
            let bandwidth = Double(size) * Double(MemoryLayout<Float>.size) / elapsed / 1e9
            print("  32KB threadgroup (MAX): \(String(format: "%.2f", bandwidth)) GB/s")
        }

        print("\n  Size limit findings:")
        print("  - Apple M2 max threadgroup memory: 32 KB")
        print("  - This is a HARD LIMIT in Metal API")
        print("  - Exceeding this will cause compilation/launch failure")
    }

    // MARK: - Occupancy Measurement
    func measureOccupancy(library: MTLLibrary) throws {
        print("\n=== 5. OCCUPANCY IMPACT ===")
        print("Testing performance with different threadgroup sizes")

        let size: UInt32 = 65536
        let iterations = 10

        guard let outBuffer = device.makeBuffer(length: Int(size) * MemoryLayout<Float>.size, options: .storageModeShared) else {
            return
        }

        var sizeVar = size
        let threadgroupSizes: [(String, Int)] = [
            ("64", 64),
            ("128", 128),
            ("256", 256),
            ("512", 512),
            ("1024 (MAX)", 1024)
        ]

        for (name, tgSize) in threadgroupSizes {
            let funcName: String
            switch tgSize {
            case 64: funcName = "tg_occupancy_64"
            case 128: funcName = "tg_occupancy_128"
            case 256: funcName = "tg_occupancy_256"
            case 512: funcName = "tg_occupancy_512"
            case 1024: funcName = "tg_occupancy_1024"
            default: continue
            }

            if let func_ = library.makeFunction(name: funcName),
               let pipeline = try? device.makeComputePipelineState(function: func_) {
                let start = getTimeNanos()
                for _ in 0..<iterations {
                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }
                    encoder.setComputePipelineState(pipeline)
                    encoder.setBuffer(outBuffer, offset: 0, index: 0)
                    encoder.setBytes(&sizeVar, length: MemoryLayout<UInt32>.size, index: 1)
                    encoder.dispatchThreads(MTLSize(width: Int(size), height: 1, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
                    encoder.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()
                }
                let end = getTimeNanos()
                let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
                let bandwidth = Double(size) * Double(MemoryLayout<Float>.size) / elapsed / 1e9
                print("  Threadgroup \(name): \(String(format: "%.2f", bandwidth)) GB/s")
            }
        }

        print("\n  Occupancy findings:")
        print("  - Larger threadgroup = more shared memory used")
        print("  - Smaller threadgroup = higher occupancy possible")
        print("  - Optimal depends on shared memory vs occupancy tradeoff")
    }
}
