import Foundation
import Metal

// MARK: - Database Operations Benchmark

let databaseShaders = """
#include <metal_stdlib>
using namespace metal;

// Parallel filter (WHERE clause)
kernel void db_filter(device uint* keys [[buffer(0)]],
             device uint* values [[buffer(1)]],
             device uint* output [[buffer(2)]],
             device atomic_uint* count [[buffer(3)]],
             constant uint& size [[buffer(4)]],
             constant uint& threshold [[buffer(5)]],
             uint id [[thread_position_in_grid]]) {
    if (id >= size) return;

    if (keys[id] > threshold) {
        uint idx = atomic_fetch_add_explicit(count, 1, memory_order_relaxed);
        output[idx] = values[id];
    }
}

// Parallel aggregation (SUM/COUNT)
kernel void db_aggregate(device uint* keys [[buffer(0)]],
                device uint* values [[buffer(1)]],
                device atomic_uint* buckets [[buffer(2)]],
                constant uint& size [[buffer(3)]],
                constant uint& num_buckets [[buffer(4)]],
                uint id [[thread_position_in_grid]]) {
    if (id >= size) return;

    uint bucket = keys[id] % num_buckets;
    atomic_fetch_add_explicit(&buckets[bucket], values[id], memory_order_relaxed);
}

// Parallel prefix sum for ranking
kernel void db_rank(device uint* keys [[buffer(0)]],
            device uint* ranks [[buffer(1)]],
            device uint* temp [[buffer(2)]],
            constant uint& size [[buffer(3)]],
            uint id [[thread_position_in_grid]]) {
    if (id >= size) return;

    uint key = keys[id];
    uint rank = 0;
    for (uint i = 0; i < size; i++) {
        if (keys[i] < key) rank++;
    }
    ranks[id] = rank;
}

// Histogram for GROUP BY
kernel void db_group_by(device uint* keys [[buffer(0)]],
                device atomic_uint* histogram [[buffer(1)]],
                constant uint& size [[buffer(2)]],
                constant uint& num_groups [[buffer(3)]],
                uint id [[thread_position_in_grid]]) {
    if (id >= size) return;

    uint group = keys[id] % num_groups;
    atomic_fetch_add_explicit(&histogram[group], 1, memory_order_relaxed);
}

// Top-K selection (simplified)
kernel void db_topk(device uint* keys [[buffer(0)]],
            device uint* output [[buffer(1)]],
            constant uint& size [[buffer(2)]],
            constant uint& k [[buffer(3)]],
            uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    // Simplified: just copy top k elements
    if (id < k) {
        output[id] = keys[size - k + id];
    }
}
"""

public struct DatabaseOpsBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    public init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
    }

    public func run() throws {
        print(String(repeating: "=", count: 70))
        print("Database Operations and Parallel Aggregation")
        print("Parallel filtering, aggregation, and join operations")
        print(String(repeating: "=", count: 70))

        guard let library = try? device.makeLibrary(source: databaseShaders, options: nil) else {
            print("Failed to compile database shaders")
            return
        }

        guard let filterFunc = library.makeFunction(name: "db_filter"),
              let aggregateFunc = library.makeFunction(name: "db_aggregate"),
              let groupByFunc = library.makeFunction(name: "db_group_by"),
              let filterPipeline = try? device.makeComputePipelineState(function: filterFunc),
              let aggregatePipeline = try? device.makeComputePipelineState(function: aggregateFunc),
              let groupByPipeline = try? device.makeComputePipelineState(function: groupByFunc) else {
            print("Failed to create database pipelines")
            return
        }

        let sizes = [256, 1024, 4096, 16384]
        let iterations = 100

        print("\n--- Database Filter Performance (WHERE clause) ---")
        print("| Size | Throughput |")
        print("|------|------------|")

        for size in sizes {
            guard let keysBuffer = device.makeBuffer(length: size * MemoryLayout<UInt32>.size, options: .storageModeShared),
                  let valuesBuffer = device.makeBuffer(length: size * MemoryLayout<UInt32>.size, options: .storageModeShared),
                  let outputBuffer = device.makeBuffer(length: size * MemoryLayout<UInt32>.size, options: .storageModeShared),
                  let countBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.size, options: .storageModeShared) else {
                continue
            }

            let keysPtr = keysBuffer.contents().assumingMemoryBound(to: UInt32.self)
            let valuesPtr = valuesBuffer.contents().assumingMemoryBound(to: UInt32.self)
            for i in 0..<size {
                keysPtr[i] = UInt32(i)
                valuesPtr[i] = UInt32(i * 2)
            }

            let threshold = UInt32(size / 2)

            let startFilter = getTimeNanos()
            for _ in 0..<iterations {
                let countPtr = countBuffer.contents().assumingMemoryBound(to: UInt32.self)
                countPtr[0] = 0

                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(filterPipeline)
                encoder.setBuffer(keysBuffer, offset: 0, index: 0)
                encoder.setBuffer(valuesBuffer, offset: 0, index: 1)
                encoder.setBuffer(outputBuffer, offset: 0, index: 2)
                encoder.setBuffer(countBuffer, offset: 0, index: 3)
                var sizeUInt = UInt32(size)
                var thresholdUInt = UInt32(threshold)
                encoder.setBytes(&sizeUInt, length: MemoryLayout<UInt32>.size, index: 4)
                encoder.setBytes(&thresholdUInt, length: MemoryLayout<UInt32>.size, index: 5)
                encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: min(size, 256), height: 1, depth: 1))
                encoder.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }
            let endFilter = getTimeNanos()
            let filterTime = getElapsedSeconds(start: startFilter, end: endFilter)
            let filterThroughput = Double(size) * Double(iterations) / filterTime / 1e6

            print("| \(size) | \(String(format: "%.2f", filterThroughput)) M/s |")
        }

        print("\n--- Database Aggregation Performance (GROUP BY) ---")
        print("| Size | Groups | Throughput |")
        print("|------|--------|------------|")

        let numGroups = 64

        for size in sizes {
            guard let keysBuffer = device.makeBuffer(length: size * MemoryLayout<UInt32>.size, options: .storageModeShared),
                  let valuesBuffer = device.makeBuffer(length: size * MemoryLayout<UInt32>.size, options: .storageModeShared),
                  let bucketsBuffer = device.makeBuffer(length: numGroups * MemoryLayout<UInt32>.size, options: .storageModeShared) else {
                continue
            }

            let keysPtr = keysBuffer.contents().assumingMemoryBound(to: UInt32.self)
            let valuesPtr = valuesBuffer.contents().assumingMemoryBound(to: UInt32.self)
            for i in 0..<size {
                keysPtr[i] = UInt32(i) % UInt32(numGroups)
                valuesPtr[i] = 1
            }

            let startAgg = getTimeNanos()
            for _ in 0..<iterations {
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(aggregatePipeline)
                encoder.setBuffer(keysBuffer, offset: 0, index: 0)
                encoder.setBuffer(valuesBuffer, offset: 0, index: 1)
                encoder.setBuffer(bucketsBuffer, offset: 0, index: 2)
                var sizeUInt = UInt32(size)
                var groupsUInt = UInt32(numGroups)
                encoder.setBytes(&sizeUInt, length: MemoryLayout<UInt32>.size, index: 3)
                encoder.setBytes(&groupsUInt, length: MemoryLayout<UInt32>.size, index: 4)
                encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: min(size, 256), height: 1, depth: 1))
                encoder.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }
            let endAgg = getTimeNanos()
            let aggTime = getElapsedSeconds(start: startAgg, end: endAgg)
            let aggThroughput = Double(size) * Double(iterations) / aggTime / 1e6

            print("| \(size) | \(numGroups) | \(String(format: "%.2f", aggThroughput)) M/s |")
        }

        print("\n--- Key Insights ---")
        print("1. Parallel filter: WHERE clause maps to predicate-based selection")
        print("2. Aggregation: GROUP BY uses atomic operations for parallel reduction")
        print("3. Ranking: O(n²) on GPU but parallelizes over elements")
        print("4. Top-K: specialized algorithms exist for better GPU performance")
        print("5. Applications: data analytics, ML feature engineering, ETL pipelines")
    }
}
