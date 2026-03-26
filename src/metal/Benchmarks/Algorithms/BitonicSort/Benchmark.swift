import Foundation
import Metal

// MARK: - Bitonic Sort (Parallel Sorting Network) Benchmark

let bitonicSortShaders = """
#include <metal_stdlib>
using namespace metal;

// Bitonic sort step - compare and swap at distance j
// k = 2^p, where p is the stage (0, 1, 2, ...)
// j = 2^q, where q is the step within stage (0, 1, ..., p)
kernel void bitonic_step(device float* data [[buffer(0)]],
                    constant uint& n [[buffer(1)]],
                    constant uint& k [[buffer(2)]],  // 2^k elements in groups
                    constant uint& j [[buffer(3)]],  // compare distance
                    uint id [[thread_position_in_grid]]) {
    if (id >= n) return;

    uint ixj = id ^ j;  // partner index (xor pattern)
    if (ixj > id) {
        // Determine sort direction: ascending or descending
        bool asc = ((id & k) == 0);
        float a = data[id];
        float b = data[ixj];

        // Compare and swap if needed
        if ((a > b && asc) || (a < b && !asc)) {
            data[id] = b;
            data[ixj] = a;
        }
    }
}

// Bitonic merge step
kernel void bitonic_merge(device float* data [[buffer(0)]],
                    constant uint& n [[buffer(1)]],
                    constant uint& k [[buffer(2)]],
                    uint id [[thread_position_in_grid]]) {
    if (id >= n) return;

    uint j = 1;
    while (j < k) {
        j <<= 1;
    }
    j >>= 1;

    uint ixj = id ^ j;
    if (ixj > id) {
        bool asc = ((id & k) == 0);
        float a = data[id];
        float b = data[ixj];
        if ((a > b && asc) || (a < b && !asc)) {
            data[id] = b;
            data[ixj] = a;
        }
    }
}

// Odd-even transposition sort (simple but less efficient)
kernel void odd_even_sort_step(device float* data [[buffer(0)]],
                         constant uint& n [[buffer(1)]],
                         constant uint& phase [[buffer(2)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= n) return;

    uint partner = (id % 2 == phase) ? id + 1 : id - 1;
    if (partner >= n || partner >= n) return;

    if (id > partner) return;  // Only upper triangle does compare

    float a = data[id];
    float b = data[partner];
    if (a > b) {
        data[id] = b;
        data[partner] = a;
    }
}

// Quick sort partition step (for comparison)
kernel void quick_partition(device float* data [[buffer(0)]],
                       device uint* pivotIdx [[buffer(1)]],
                       device uint* numLess [[buffer(2)]],
                       device uint* numGreater [[buffer(3)]],
                       constant uint& n [[buffer(4)]],
                       uint id [[thread_position_in_grid]]) {
    if (id >= n) return;

    float pivot = data[pivotIdx[0]];
    float val = data[id];

    if (val < pivot) {
        atomic_fetch_add_explicit(numLess, 1, memory_order_relaxed);
    } else if (val > pivot) {
        atomic_fetch_add_explicit(numGreater, 1, memory_order_relaxed);
    }
}
"""

public struct BitonicSortBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    public init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
    }

    public func run() throws {
        print(String(repeating: "=", count: 70))
        print("Bitonic Sort (Parallel Sorting Network) Benchmark")
        print(String(repeating: "=", count: 70))

        guard let library = try? device.makeLibrary(source: bitonicSortShaders, options: nil) else {
            print("Failed to compile bitonic sort shaders")
            return
        }

        let sizes = [1024, 4096, 16384, 65536]  // Must be power of 2

        for size in sizes {
            print("\n--- Size: \(size) elements ---")

            guard let dataBuffer = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared) else {
                continue
            }

            // Initialize with random data
            let dataPtr = dataBuffer.contents().bindMemory(to: Float.self, capacity: size)
            for i in 0..<size {
                dataPtr[i] = Float.random(in: 0.0...1.0)
            }

            var nVar = UInt32(size)

            // Bitonic sort requires log(n) stages, each with log(n) steps
            // Total: log(n) * log(n) = log²(n) steps
            let numStages = Int(log2(Float(size)))
            let numSteps = numStages

            if let stepFunc = library.makeFunction(name: "bitonic_step"),
               let stepPipeline = try? device.makeComputePipelineState(function: stepFunc) {

                // Full bitonic sort: O(log² n) stages
                let iterations = 5
                let start = getTimeNanos()

                for _ in 0..<iterations {
                    // Reinitialize data for each iteration
                    for i in 0..<size {
                        dataPtr[i] = Float.random(in: 0.0...1.0)
                    }

                    // Bitonic sort stages
                    for stage in 0..<numStages {
                        let k = UInt32(1) << stage
                        for step in 0..<numSteps {
                            let j = UInt32(1) << step

                            guard let cmd = queue.makeCommandBuffer(),
                                  let encoder = cmd.makeComputeCommandEncoder() else { continue }

                            encoder.setComputePipelineState(stepPipeline)
                            encoder.setBuffer(dataBuffer, offset: 0, index: 0)
                            encoder.setBytes(&nVar, length: MemoryLayout<UInt32>.size, index: 1)
                            encoder.setBytes(&k, length: MemoryLayout<UInt32>.size, index: 2)
                            encoder.setBytes(&j, length: MemoryLayout<UInt32>.size, index: 3)
                            encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                                  threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                            encoder.endEncoding()
                            cmd.commit()
                            cmd.waitUntilCompleted()
                        }
                    }
                }

                let end = getTimeNanos()
                let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)

                // Bitonic sort: n*log²(n) comparisons
                let comparisons = Double(size) * Double(numStages) * Double(numSteps)
                let compPerSec = comparisons / elapsed / 1e9

                print("Bitonic Sort: \(String(format: "%.2f", compPerSec)) Billion comps/s")
                print("  (\(numStages) stages x \(numSteps) steps = \(numStages * numSteps) kernel launches)")
                print("  \(String(format: "%.4f", elapsed * 1000)) ms per sort")
            }
        }

        // Compare with naive comparison
        print("\n--- Sorting Network Properties ---")
        print("Bitonic Sort: O(log² n) stages, each stage is a sorting network")
        print("Advantage: Constant topology, no branches, highly parallel")
        print("Disadvantage: Requires n to be power of 2")

        print("\n--- Key Findings ---")
        print("1. Bitonic sort is a sorting network - deterministic, no branches")
        print("2. Each step compares pairs (i, i^j) using xor pattern")
        print("3. O(n log² n) comparisons but O(log² n) depth (high parallelism)")
        print("4. Perfect for GPU: all threads active, no divergence")
        print("5. Unlike comparison sorts, sorting networks can run in lockstep")
    }
}
