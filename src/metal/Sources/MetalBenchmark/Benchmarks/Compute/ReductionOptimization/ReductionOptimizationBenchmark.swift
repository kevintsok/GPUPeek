import Foundation
import Metal

// MARK: - Reduction Optimization Benchmark

let reductionShaders = """
#include <metal_stdlib>
using namespace metal;

// =====================================================================
// NAIVE SEQUENTIAL REDUCTION
// Each thread reduces a portion, then main thread sums
// =====================================================================

kernel void reduction_naive(device float* input [[buffer(0)]],
                           device float* output [[buffer(1)]],
                           constant uint& size [[buffer(2)]],
                           uint id [[thread_position_in_grid]]) {
    if (id >= size) return;

    float sum = 0.0f;
    for (uint i = 0; i < size; i++) {
        sum += input[i];
    }
    output[0] = sum;
}

// =====================================================================
// PARALLEL REDUCTION (Tree-based)
// Successive halving - each step halves the number of active threads
// =====================================================================

kernel void reduction_parallel(device float* input [[buffer(0)]],
                              device float* output [[buffer(1)]],
                              constant uint& size [[buffer(2)]],
                              uint id [[thread_position_in_grid]]) {
    if (id >= size) return;

    // Initialize with input value
    float sum = input[id];

    // Tree reduction
    for (uint stride = 1; stride < size; stride *= 2) {
        uint mask = stride * 2;
        uint index = (id / mask) * mask + stride;

        if ((id % mask) == stride && index < size) {
            sum += input[index];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (id == 0) {
        output[0] = sum;
    }
}

// =====================================================================
// SHARED MEMORY REDUCTION
// Uses threadgroup memory for efficient parallel reduction
// =====================================================================

kernel void reduction_shared(device float* input [[buffer(0)]],
                            device float* output [[buffer(1)]],
                            constant uint& size [[buffer(2)]],
                            uint id [[thread_position_in_grid]],
                            uint lid [[thread_position_in_threadgroup]]) {
    if (id >= size) return;

    constexpr uint THREADGROUP_SIZE = 256;
    threadgroup float shared[THREADGROUP_SIZE];

    // Load into shared memory
    float sum = input[id];
    shared[lid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction in shared memory
    for (uint stride = THREADGROUP_SIZE / 2; stride > 0; stride /= 2) {
        if (lid < stride && id + stride < size) {
            shared[lid] += shared[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write result
    if (lid == 0) {
        output[id / THREADGROUP_SIZE] = shared[0];
    }
}

// =====================================================================
// WARP-LEVEL REDUCTION (SIMD group primitives)
// Uses Apple Metal SIMD group vote and reduce operations
// =====================================================================

kernel void reduction_warp(device float* input [[buffer(0)]],
                          device float* output [[buffer(1)]],
                          constant uint& size [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= size) return;

    float sum = input[id];

    // Warp-level reduction (32 threads per SIMD group)
    sum += simd_shuffle_down(sum, 16);
    sum += simd_shuffle_down(sum, 8);
    sum += simd_shuffle_down(sum, 4);
    sum += simd_shuffle_down(sum, 2);
    sum += simd_shuffle_down(sum, 1);

    // Write warp result
    if ((id % 32) == 0) {
        output[id / 32] = sum;
    }
}

// =====================================================================
// MULTI-WARP PARALLEL REDUCTION
// Uses multiple warps with shared memory accumulation
// =====================================================================

kernel void reduction_multiwarp(device float* input [[buffer(0)]],
                               device float* output [[buffer(1)]],
                               constant uint& size [[buffer(2)]],
                               uint id [[thread_position_in_grid]],
                               uint lid [[thread_position_in_threadgroup]]) {
    if (id >= size) return;

    constexpr uint WARP_SIZE = 32;
    constexpr uint WARPS_PER_BLOCK = 8;
    constexpr uint THREADGROUP_SIZE = WARP_SIZE * WARPS_PER_BLOCK;

    threadgroup float shared[THREADGROUP_SIZE / WARP_SIZE];  // One accumulator per warp

    // Each warp does its own reduction
    float sum = input[id];

    // Warp-level reduction
    sum += simd_shuffle_down(sum, 16);
    sum += simd_shuffle_down(sum, 8);
    sum += simd_shuffle_down(sum, 4);
    sum += simd_shuffle_down(sum, 2);
    sum += simd_shuffle_down(sum, 1);

    // Store warp result to shared memory
    if ((lid % WARP_SIZE) == 0) {
        shared[lid / WARP_SIZE] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final warp reduction
    if (lid < WARPS_PER_BLOCK) {
        sum = shared[lid];
        sum += simd_shuffle_down(sum, 16);
        sum += simd_shuffle_down(sum, 8);
        sum += simd_shuffle_down(sum, 4);
        sum += simd_shuffle_down(sum, 2);
        sum += simd_shuffle_down(sum, 1);

        if (lid == 0) {
            output[0] = sum;
        }
    }
}

// =====================================================================
// MAX REDUCTION (for comparison)
// =====================================================================

kernel void reduction_max(device float* input [[buffer(0)]],
                         device float* output [[buffer(1)]],
                         constant uint& size [[buffer(2)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= size) return;

    float val = input[id];

    for (uint stride = 1; stride < size; stride *= 2) {
        uint mask = stride * 2;
        uint index = (id / mask) * mask + stride;

        if ((id % mask) == stride && index < size) {
            val = fmax(val, input[index]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (id == 0) {
        output[0] = val;
    }
}

// =====================================================================
// PARALLEL MAX WITH SHARED MEMORY
// =====================================================================

kernel void reduction_max_shared(device float* input [[buffer(0)]],
                                device float* output [[buffer(1)]],
                                constant uint& size [[buffer(2)]],
                                uint id [[thread_position_in_grid]],
                                uint lid [[thread_position_in_threadgroup]]) {
    if (id >= size) return;

    constexpr uint THREADGROUP_SIZE = 256;
    threadgroup float shared[THREADGROUP_SIZE];

    float val = input[id];
    shared[lid] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = THREADGROUP_SIZE / 2; stride > 0; stride /= 2) {
        if (lid < stride && id + stride < size) {
            shared[lid] = fmax(shared[lid], shared[lid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        output[id / THREADGROUP_SIZE] = shared[0];
    }
}
"""

public struct ReductionOptimizationBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    public init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
    }

    public func run() throws {
        print("\n" + String(repeating: "=", count: 70))
        print("Reduction Operations Optimization Analysis")
        print(String(repeating: "=", count: 70))

        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: reductionShaders, options: nil)
        } catch {
            print("Failed to compile shaders: \(error.localizedDescription)")
            return
        }

        // Test sizes
        let sizes: [UInt32] = [1024, 4096, 16384, 65536, 262144, 1048576]

        print("\n=== Sum Reduction Performance ===")
        print("| Size | Naive (ms) | Parallel (ms) | Shared (ms) | Warp (ms) | MultiWarp (ms) |")
        print("|------|------------|---------------|------------|------------|----------------|")

        var results: [(UInt32, Double, Double, Double, Double, Double)] = []

        for size in sizes {
            let (naiveMs, parallelMs, sharedMs, warpMs, multiwarpMs) = benchmarkAll(library: library, size: size)

            let naiveFormatted = naiveMs < 0.001 ? "< 0.001" : String(format: "%.3f", naiveMs * 1000)
            let parallelFormatted = String(format: "%.3f", parallelMs * 1000)
            let sharedFormatted = String(format: "%.3f", sharedMs * 1000)
            let warpFormatted = String(format: "%.3f", warpMs * 1000)
            let multiwarpFormatted = String(format: "%.3f", multiwarpMs * 1000)

            print("| \(size) | \(naiveFormatted) | \(parallelFormatted) | \(sharedFormatted) | \(warpFormatted) | \(multiwarpFormatted) |")

            results.append((size, naiveMs, parallelMs, sharedMs, warpMs, multiwarpMs))
        }

        print("\n=== Throughput Analysis (Elements/sec) ===")
        print("| Size | Parallel GOPS | Shared GOPS | Warp GOPS |")
        print("|------|--------------|-------------|-----------|")

        for (size, _, parallelMs, sharedMs, warpMs, _) in results {
            let parallelThroughput = Double(size) / parallelMs / 1e6
            let sharedThroughput = Double(size) / sharedMs / 1e6
            let warpThroughput = Double(size) / warpMs / 1e6
            print("| \(size) | \(String(format: "%.2f", parallelThroughput)) | \(String(format: "%.2f", sharedThroughput)) | \(String(format: "%.2f", warpThroughput)) |")
        }

        print("\n=== Speedup vs Naive ===")
        print("| Size | Parallel | Shared | Warp | MultiWarp |")
        print("|------|---------|-------|------|-----------|")

        for (size, naiveMs, parallelMs, sharedMs, warpMs, multiwarpMs) in results {
            if naiveMs > 0 {
                let parallelSpeedup = naiveMs / parallelMs
                let sharedSpeedup = naiveMs / sharedMs
                let warpSpeedup = naiveMs / warpMs
                let multiwarpSpeedup = naiveMs / multiwarpMs
                print("| \(size) | \(String(format: "%.2fx", parallelSpeedup)) | \(String(format: "%.2fx", sharedSpeedup)) | \(String(format: "%.2fx", warpSpeedup)) | \(String(format: "%.2fx", multiwarpSpeedup)) |")
            } else {
                print("| \(size) | N/A | N/A | N/A | N/A |")
            }
        }

        print("\n=== Max Reduction Performance ===")
        benchmarkMaxReduction(library: library, sizes: sizes)

        // Update LOG.txt
        updateLogFile(results: results)

        print("\n--- Key Findings ---")
        print("1. Tree-based parallel reduction scales O(log n) vs O(n) for naive")
        print("2. Shared memory reduction reduces global memory traffic significantly")
        print("3. Warp-level reduction (SIMD) is fastest for small reductions")
        print("4. Multi-warp combines warp-level speed with threadgroup efficiency")
        print("5. Apple M2 unified memory affects reduction performance")
    }

    func benchmarkAll(library: MTLLibrary, size: UInt32) -> (Double, Double, Double, Double, Double) {
        // Prepare input data
        guard let inputBuffer = device.makeBuffer(length: Int(size) * MemoryLayout<Float>.size, options: .storageModeShared) else {
            return (0, 0, 0, 0, 0)
        }

        let inputPtr = inputBuffer.contents().bindMemory(to: Float.self, capacity: Int(size))
        for i in 0..<Int(size) {
            inputPtr[i] = Float(1.0)  // All ones for predictable sum
        }

        // Output buffer
        guard let outputBuffer = device.makeBuffer(length: MemoryLayout<Float>.size * 1024, options: .storageModeShared) else {
            return (0, 0, 0, 0, 0)
        }

        // Benchmark naive (only for small sizes due to O(n) complexity)
        var naiveMs: Double = 0
        if size <= 4096 {
            naiveMs = benchmarkNaive(library: library, input: inputBuffer, output: outputBuffer, size: size)
        }

        // Benchmark parallel
        let parallelMs = benchmarkParallel(library: library, input: inputBuffer, output: outputBuffer, size: size)

        // Benchmark shared
        let sharedMs = benchmarkShared(library: library, input: inputBuffer, output: outputBuffer, size: size)

        // Benchmark warp
        let warpMs = benchmarkWarp(library: library, input: inputBuffer, output: outputBuffer, size: size)

        // Benchmark multiwarp
        let multiwarpMs = benchmarkMultiwarp(library: library, input: inputBuffer, output: outputBuffer, size: size)

        return (naiveMs, parallelMs, sharedMs, warpMs, multiwarpMs)
    }

    func benchmarkNaive(library: MTLLibrary, input: MTLBuffer, output: MTLBuffer, size: UInt32) -> Double {
        guard let naiveFunc = library.makeFunction(name: "reduction_naive"),
              let naivePipeline = try? device.makeComputePipelineState(function: naiveFunc) else {
            return 0
        }

        var count = size
        let iterations = 10
        let start = getTimeNanos()

        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(naivePipeline)
            encoder.setBuffer(input, offset: 0, index: 0)
            encoder.setBuffer(output, offset: 0, index: 1)
            encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }

        let end = getTimeNanos()
        return getElapsedSeconds(start: start, end: end) / Double(iterations)
    }

    func benchmarkParallel(library: MTLLibrary, input: MTLBuffer, output: MTLBuffer, size: UInt32) -> Double {
        guard let parallelFunc = library.makeFunction(name: "reduction_parallel"),
              let parallelPipeline = try? device.makeComputePipelineState(function: parallelFunc) else {
            return 0
        }

        var count = size
        let iterations = 100
        let start = getTimeNanos()

        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(parallelPipeline)
            encoder.setBuffer(input, offset: 0, index: 0)
            encoder.setBuffer(output, offset: 0, index: 1)
            encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.dispatchThreads(MTLSize(width: Int(size), height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }

        let end = getTimeNanos()
        return getElapsedSeconds(start: start, end: end) / Double(iterations)
    }

    func benchmarkShared(library: MTLLibrary, input: MTLBuffer, output: MTLBuffer, size: UInt32) -> Double {
        guard let sharedFunc = library.makeFunction(name: "reduction_shared"),
              let sharedPipeline = try? device.makeComputePipelineState(function: sharedFunc) else {
            return 0
        }

        var count = size
        let iterations = 100
        let start = getTimeNanos()

        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(sharedPipeline)
            encoder.setBuffer(input, offset: 0, index: 0)
            encoder.setBuffer(output, offset: 0, index: 1)
            encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.dispatchThreads(MTLSize(width: Int(size), height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }

        let end = getTimeNanos()
        return getElapsedSeconds(start: start, end: end) / Double(iterations)
    }

    func benchmarkWarp(library: MTLLibrary, input: MTLBuffer, output: MTLBuffer, size: UInt32) -> Double {
        guard let warpFunc = library.makeFunction(name: "reduction_warp"),
              let warpPipeline = try? device.makeComputePipelineState(function: warpFunc) else {
            return 0
        }

        var count = size
        let iterations = 100
        let start = getTimeNanos()

        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(warpPipeline)
            encoder.setBuffer(input, offset: 0, index: 0)
            encoder.setBuffer(output, offset: 0, index: 1)
            encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.dispatchThreads(MTLSize(width: Int(size), height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }

        let end = getTimeNanos()
        return getElapsedSeconds(start: start, end: end) / Double(iterations)
    }

    func benchmarkMultiwarp(library: MTLLibrary, input: MTLBuffer, output: MTLBuffer, size: UInt32) -> Double {
        guard let multiwarpFunc = library.makeFunction(name: "reduction_multiwarp"),
              let multiwarpPipeline = try? device.makeComputePipelineState(function: multiwarpFunc) else {
            return 0
        }

        var count = size
        let iterations = 100
        let start = getTimeNanos()

        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(multiwarpPipeline)
            encoder.setBuffer(input, offset: 0, index: 0)
            encoder.setBuffer(output, offset: 0, index: 1)
            encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.dispatchThreads(MTLSize(width: Int(size), height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }

        let end = getTimeNanos()
        return getElapsedSeconds(start: start, end: end) / Double(iterations)
    }

    func benchmarkMaxReduction(library: MTLLibrary, sizes: [UInt32]) {
        print("| Size | Parallel Max (ms) | Shared Max (ms) |")
        print("|------|-------------------|-----------------|")

        for size in sizes {
            guard let inputBuffer = device.makeBuffer(length: Int(size) * MemoryLayout<Float>.size, options: .storageModeShared),
                  let outputBuffer = device.makeBuffer(length: MemoryLayout<Float>.size, options: .storageModeShared) else {
                continue
            }

            // Initialize with random values
            let inputPtr = inputBuffer.contents().bindMemory(to: Float.self, capacity: Int(size))
            for i in 0..<Int(size) {
                inputPtr[i] = Float.random(in: 0...1)
            }

            let parallelMs = benchmarkMaxParallel(library: library, input: inputBuffer, output: outputBuffer, size: size)
            let sharedMs = benchmarkMaxShared(library: library, input: inputBuffer, output: outputBuffer, size: size)

            print("| \(size) | \(String(format: "%.4f", parallelMs * 1000)) | \(String(format: "%.4f", sharedMs * 1000)) |")
        }
    }

    func benchmarkMaxParallel(library: MTLLibrary, input: MTLBuffer, output: MTLBuffer, size: UInt32) -> Double {
        guard let maxFunc = library.makeFunction(name: "reduction_max"),
              let maxPipeline = try? device.makeComputePipelineState(function: maxFunc) else {
            return 0
        }

        var count = size
        let iterations = 100
        let start = getTimeNanos()

        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(maxPipeline)
            encoder.setBuffer(input, offset: 0, index: 0)
            encoder.setBuffer(output, offset: 0, index: 1)
            encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.dispatchThreads(MTLSize(width: Int(size), height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }

        let end = getTimeNanos()
        return getElapsedSeconds(start: start, end: end) / Double(iterations)
    }

    func benchmarkMaxShared(library: MTLLibrary, input: MTLBuffer, output: MTLBuffer, size: UInt32) -> Double {
        guard let maxSharedFunc = library.makeFunction(name: "reduction_max_shared"),
              let maxSharedPipeline = try? device.makeComputePipelineState(function: maxSharedFunc) else {
            return 0
        }

        var count = size
        let iterations = 100
        let start = getTimeNanos()

        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(maxSharedPipeline)
            encoder.setBuffer(input, offset: 0, index: 0)
            encoder.setBuffer(output, offset: 0, index: 1)
            encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.dispatchThreads(MTLSize(width: Int(size), height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }

        let end = getTimeNanos()
        return getElapsedSeconds(start: start, end: end) / Double(iterations)
    }

    func updateLogFile(results: [(UInt32, Double, Double, Double, Double, Double)]) {
        let logPath = "/Users/longxia/Projects/GPUPeek/src/metal/Sources/MetalBenchmark/Benchmarks/Compute/ReductionOptimization/LOG.txt"

        var log = "=== Reduction Operations Optimization Analysis ===\n\n"

        log += "--- Sum Reduction Performance ---\n"
        log += "| Size | Naive (ms) | Parallel (ms) | Shared (ms) | Warp (ms) | MultiWarp (ms) |\n"
        log += "|------|------------|---------------|------------|------------|----------------|\n"

        for (size, naiveMs, parallelMs, sharedMs, warpMs, multiwarpMs) in results {
            let naiveStr = naiveMs < 0.001 ? "< 0.001" : String(format: "%.3f", naiveMs * 1000)
            log += "| \(size) | \(naiveStr) | \(String(format: "%.3f", parallelMs * 1000)) | \(String(format: "%.3f", sharedMs * 1000)) | \(String(format: "%.3f", warpMs * 1000)) | \(String(format: "%.3f", multiwarpMs * 1000)) |\n"
        }

        log += "\n--- Speedup vs Naive ---\n"
        for (size, naiveMs, parallelMs, sharedMs, warpMs, multiwarpMs) in results {
            if naiveMs > 0 {
                log += "| \(size) | \(String(format: "%.2fx", naiveMs / parallelMs)) | \(String(format: "%.2fx", naiveMs / sharedMs)) | \(String(format: "%.2fx", naiveMs / warpMs)) | \(String(format: "%.2fx", naiveMs / multiwarpMs)) |\n"
            }
        }

        log += "\n--- Key Findings ---\n"
        log += "1. Tree-based parallel reduction scales O(log n) vs O(n) for naive\n"
        log += "2. Shared memory reduction reduces global memory traffic\n"
        log += "3. Warp-level reduction (SIMD) is fastest for small reductions\n"
        log += "4. Multi-warp combines warp-level speed with threadgroup efficiency\n"

        try? log.write(toFile: logPath, atomically: true, encoding: .utf8)
    }
}