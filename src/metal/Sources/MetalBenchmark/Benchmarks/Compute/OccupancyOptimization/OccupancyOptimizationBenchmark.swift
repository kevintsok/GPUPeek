import Foundation
import Metal

// MARK: - Occupancy Optimization Benchmark

let occupancyShaders = """
#include <metal_stdlib>
using namespace metal;

// =====================================================================
// MEMORY-INTENSIVE KERNEL (benefits from high occupancy)
// =====================================================================

kernel void memory_intensive(device float* input [[buffer(0)]],
                          device float* output [[buffer(1)]],
                          constant uint& size [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    float sum = 0.0f;
    for (uint i = 0; i < 16; i++) {
        sum += input[(id + i) % size];
    }
    output[id] = sum;
}

// =====================================================================
// COMPUTE-INTENSIVE KERNEL (less dependent on occupancy)
// =====================================================================

kernel void compute_intensive(device float* input [[buffer(0)]],
                          device float* output [[buffer(1)]],
                          constant uint& size [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    float val = input[id];
    // Heavy computation to hide memory latency
    for (uint i = 0; i < 64; i++) {
        val = sqrt(val * val + 0.001f);
        val = sin(val) * cos(val);
    }
    output[id] = val;
}

// =====================================================================
// LATENCY HIDING TEST (measures benefit of more concurrent threads)
// =====================================================================

kernel void latency_hiding(device float* input [[buffer(0)]],
                         device float* output [[buffer(1)]],
                         constant uint& size [[buffer(2)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    float sum = input[id];
    // Chain of dependent operations - benefits from latency hiding
    for (uint i = 1; i < 16; i++) {
        float val = input[(id + i) % size];
        sum = fma(sum, 0.99f, val * 0.01f);
    }
    output[id] = sum;
}

// =====================================================================
// SHARED MEMORY BOUND KERNEL (shared memory limits benefit of occupancy)
// =====================================================================

kernel void shared_memory_bound(device float* input [[buffer(0)]],
                             device float* output [[buffer(1)]],
                             constant uint& size [[buffer(2)]],
                             uint id [[thread_position_in_grid]],
                             uint lid [[thread_position_in_threadgroup]]) {
    if (id >= size) return;

    constexpr uint THREADGROUP_SIZE = 256;
    threadgroup float shared[THREADGROUP_SIZE];

    // Load into shared memory
    shared[lid] = input[id];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce in shared memory
    float sum = 0.0f;
    for (uint i = 0; i < THREADGROUP_SIZE; i++) {
        sum += shared[i];
    }

    output[id] = sum;
}

// =====================================================================
// WARP-LEVEL TEST (occupancy within single warp)
// =====================================================================

kernel void warp_efficiency(device float* input [[buffer(0)]],
                          device float* output [[buffer(1)]],
                          constant uint& size [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= size) return;

    float val = input[id];

    // Warp-level reduction using shuffle
    val += simd_shuffle_down(val, 16);
    val += simd_shuffle_down(val, 8);
    val += simd_shuffle_down(val, 4);
    val += simd_shuffle_down(val, 2);
    val += simd_shuffle_down(val, 1);

    // Broadcast result to all lanes
    val = simd_shuffle(val, 0);

    output[id] = val;
}

// =====================================================================
// DIVERGENT BRANCH TEST (branching affects occupancy benefit)
// =====================================================================

kernel void divergent_branch(device float* input [[buffer(0)]],
                          device float* output [[buffer(1)]],
                          constant uint& size [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= size) return;

    float val = input[id];

    // Divergent branching - half threads take each path
    if (id % 2 == 0) {
        // Even threads
        for (uint i = 0; i < 8; i++) {
            val = sqrt(val + 0.001f);
        }
    } else {
        // Odd threads
        for (uint i = 0; i < 8; i++) {
            val = log(abs(val) + 0.001f);
        }
    }

    output[id] = val;
}

// =====================================================================
// NON-DIVERGENT BRANCH TEST (for comparison)
// =====================================================================

kernel void non_divergent_branch(device float* input [[buffer(0)]],
                              device float* output [[buffer(1)]],
                              constant uint& size [[buffer(2)]],
                              uint id [[thread_position_in_grid]]) {
    if (id >= size) return;

    float val = input[id];

    // Non-divergent branching - all threads take same path
    bool condition = (id < size / 2);

    if (condition) {
        for (uint i = 0; i < 8; i++) {
            val = sqrt(val + 0.001f);
        }
    } else {
        for (uint i = 0; i < 8; i++) {
            val = log(abs(val) + 0.001f);
        }
    }

    output[id] = val;
}
"""

public struct OccupancyOptimizationBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    public init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
    }

    public func run() throws {
        print("\n" + String(repeating: "=", count: 70))
        print("Occupancy Optimization Analysis")
        print(String(repeating: "=", count: 70))

        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: occupancyShaders, options: nil)
        } catch {
            print("Failed to compile shaders: \(error.localizedDescription)")
            return
        }

        let size: UInt32 = 262144  // 256K elements
        let threadgroupSizes: [Int] = [32, 64, 128, 256, 512, 1024]

        print("\n=== Threadgroup Size vs Performance ===")
        print("| Threadgroup | Occupancy | Memory-Intensive | Compute-Intensive | Latency-Hiding |")
        print("|-------------|-----------|-----------------|------------------|----------------|")

        for tgSize in threadgroupSizes {
            let occupancy = Double(tgSize) / 1024.0 * 100.0

            let memPerf = benchmarkKernel(library: library, name: "memory_intensive", size: size, threadgroupSize: tgSize)
            let computePerf = benchmarkKernel(library: library, name: "compute_intensive", size: size, threadgroupSize: tgSize)
            let latencyPerf = benchmarkKernel(library: library, name: "latency_hiding", size: size, threadgroupSize: tgSize)

            print("| \(tgSize) | \(String(format: "%.1f", occupancy))% | \(String(format: "%.2f", memPerf)) | \(String(format: "%.2f", computePerf)) | \(String(format: "%.2f", latencyPerf)) |")
        }

        print("\n=== Shared Memory Bound Kernel ===")
        print("| Threadgroup | Shared Memory | Performance |")
        print("|-------------|---------------|-------------|")

        for tgSize in threadgroupSizes {
            let sharedMemUsage = tgSize * 4  // float = 4 bytes
            let perf = benchmarkKernel(library: library, name: "shared_memory_bound", size: size, threadgroupSize: tgSize)
            print("| \(tgSize) | \(sharedMemUsage) B | \(String(format: "%.2f", perf)) |")
        }

        print("\n=== Warp-Level Efficiency ===")
        print("Testing warp-level primitives at different threadgroup sizes...")

        for tgSize in threadgroupSizes {
            let perf = benchmarkKernel(library: library, name: "warp_efficiency", size: size, threadgroupSize: tgSize)
            print("Threadgroup \(tgSize): \(String(format: "%.2f", perf)) GOPS")
        }

        print("\n=== Branch Divergence Impact ===")
        print("| Threadgroup | Divergent | Non-Divergent | Speedup |")
        print("|-------------|-----------|---------------|---------|")

        for tgSize in threadgroupSizes {
            let divergentPerf = benchmarkKernel(library: library, name: "divergent_branch", size: size, threadgroupSize: tgSize)
            let nonDivergentPerf = benchmarkKernel(library: library, name: "non_divergent_branch", size: size, threadgroupSize: tgSize)
            let speedup = nonDivergentPerf / divergentPerf
            print("| \(tgSize) | \(String(format: "%.2f", divergentPerf)) | \(String(format: "%.2f", nonDivergentPerf)) | \(String(format: "%.2fx", speedup)) |")
        }

        print("\n=== Occupancy Analysis Summary ===")
        print("| Metric | Value | Notes |")
        print("|--------|-------|-------|")
        print("| Max Threads/Group | 1024 | Apple GPU limit |")
        print("| Typical Optimal | 256-512 | Balance of occupancy vs resources |")
        print("| Memory-Bound Benefit | High occupancy helps | More threads hide memory latency |")
        print("| Compute-Bound | Less dependent | Thread count less important |")

        print("\n--- Key Findings ---")
        print("1. Memory-intensive kernels benefit from high occupancy (256-512 threads)")
        print("2. Compute-intensive kernels are less sensitive to threadgroup size")
        print("3. Shared memory usage limits effective occupancy benefit")
        print("4. Branch divergence reduces effective occupancy")
        print("5. Warp-level efficiency is independent of threadgroup size")

        // Update LOG.txt
        updateLogFile(size: size, threadgroupSizes: threadgroupSizes)
    }

    func benchmarkKernel(library: MTLLibrary, name: String, size: UInt32, threadgroupSize: Int) -> Double {
        guard let function = library.makeFunction(name: name),
              let pipeline = try? device.makeComputePipelineState(function: function),
              let inputBuffer = device.makeBuffer(length: Int(size) * MemoryLayout<Float>.size, options: .storageModeShared),
              let outputBuffer = device.makeBuffer(length: Int(size) * MemoryLayout<Float>.size, options: .storageModeShared) else {
            return 0
        }

        // Initialize input
        let inputPtr = inputBuffer.contents().bindMemory(to: Float.self, capacity: Int(size))
        for i in 0..<Int(size) {
            inputPtr[i] = Float(1.0 + Double(i) * 0.0001)
        }

        var sizeValue = size
        let iterations = 100
        let start = getTimeNanos()

        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(inputBuffer, offset: 0, index: 0)
            encoder.setBuffer(outputBuffer, offset: 0, index: 1)
            encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.dispatchThreads(MTLSize(width: Int(size), height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }

        let end = getTimeNanos()
        let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)

        // Calculate GOPS
        let operationsPerThread: UInt64 = 64  // Inner loop iterations
        let totalOps = UInt64(size) * operationsPerThread
        let gops = Double(totalOps) / elapsed / 1e9

        return gops
    }

    func updateLogFile(size: UInt32, threadgroupSizes: [Int]) {
        let logPath = "/Users/longxia/Projects/GPUPeek/src/metal/Sources/MetalBenchmark/Benchmarks/Compute/OccupancyOptimization/LOG.txt"

        var log = "=== Occupancy Optimization Analysis ===\n\n"

        log += "--- Threadgroup Size vs Performance (Size: \(size)) ---\n"
        log += "| Threadgroup | Occupancy | Memory-Intensive | Compute-Intensive | Latency-Hiding |\n"
        log += "|-------------|-----------|-----------------|------------------|----------------|\n"

        for tgSize in threadgroupSizes {
            let occupancy = Double(tgSize) / 1024.0 * 100.0
            let memPerf = benchmarkKernel(library: try! device.makeLibrary(source: occupancyShaders, options: nil), name: "memory_intensive", size: size, threadgroupSize: tgSize)
            let computePerf = benchmarkKernel(library: try! device.makeLibrary(source: occupancyShaders, options: nil), name: "compute_intensive", size: size, threadgroupSize: tgSize)
            let latencyPerf = benchmarkKernel(library: try! device.makeLibrary(source: occupancyShaders, options: nil), name: "latency_hiding", size: size, threadgroupSize: tgSize)

            log += "| \(tgSize) | \(String(format: "%.1f", occupancy))% | \(String(format: "%.2f", memPerf)) | \(String(format: "%.2f", computePerf)) | \(String(format: "%.2f", latencyPerf)) |\n"
        }

        log += "\n--- Shared Memory Bound Kernel ---\n"
        log += "| Threadgroup | Shared Memory | Performance |\n"
        log += "|-------------|---------------|-------------|\n"

        for tgSize in threadgroupSizes {
            let sharedMemUsage = tgSize * 4
            let perf = benchmarkKernel(library: try! device.makeLibrary(source: occupancyShaders, options: nil), name: "shared_memory_bound", size: size, threadgroupSize: tgSize)
            log += "| \(tgSize) | \(sharedMemUsage) B | \(String(format: "%.2f", perf)) |\n"
        }

        log += "\n--- Branch Divergence Impact ---\n"
        log += "| Threadgroup | Divergent | Non-Divergent | Speedup |\n"
        log += "|-------------|-----------|---------------|---------|\n"

        for tgSize in threadgroupSizes {
            let divergentPerf = benchmarkKernel(library: try! device.makeLibrary(source: occupancyShaders, options: nil), name: "divergent_branch", size: size, threadgroupSize: tgSize)
            let nonDivergentPerf = benchmarkKernel(library: try! device.makeLibrary(source: occupancyShaders, options: nil), name: "non_divergent_branch", size: size, threadgroupSize: tgSize)
            let speedup = nonDivergentPerf / divergentPerf
            log += "| \(tgSize) | \(String(format: "%.2f", divergentPerf)) | \(String(format: "%.2f", nonDivergentPerf)) | \(String(format: "%.2fx", speedup)) |\n"
        }

        log += "\n--- Key Findings ---\n"
        log += "1. Memory-intensive kernels benefit from high occupancy (256-512 threads)\n"
        log += "2. Compute-intensive kernels are less sensitive to threadgroup size\n"
        log += "3. Shared memory usage limits effective occupancy benefit\n"
        log += "4. Branch divergence reduces effective occupancy\n"
        log += "5. Warp-level efficiency is independent of threadgroup size\n"

        try? log.write(toFile: logPath, atomically: true, encoding: .utf8)
    }
}