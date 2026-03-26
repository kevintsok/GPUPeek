import Foundation
import Metal

// MARK: - Bucket Sort / Hash-Based Distribution Benchmark

let bucketSortShaders = """
#include <metal_stdlib>
using namespace metal;

// Bucket Sort: Phase 1 - Hash elements to buckets
kernel void bucket_hash(device const float* in [[buffer(0)]],
                  device atomic_uint* bucket_counts [[buffer(1)]],
                  device uint* bucket_ids [[buffer(2)]],
                  constant uint& size [[buffer(3)]],
                  constant uint& num_buckets [[buffer(4)]],
                  uint id [[thread_position_in_grid]]) {
    if (id >= size) return;

    float val = in[id];
    uint bucket = uint(val * float(num_buckets));
    if (bucket >= num_buckets) bucket = num_buckets - 1;

    bucket_ids[id] = bucket;
    atomic_fetch_add_explicit(&bucket_counts[bucket], 1, memory_order_relaxed);
}

// Bucket Sort: Phase 2 - Scan bucket counts for offsets
kernel void bucket_scan_counts(device const atomic_uint* counts [[buffer(0)]],
                          device uint* offsets [[buffer(1)]],
                          constant uint& num_buckets [[buffer(2)]]) {
    uint sum = 0;
    for (uint i = 0; i < num_buckets; i++) {
        uint cnt = atomic_load_explicit(&counts[i], memory_order_relaxed);
        offsets[i] = sum;
        sum += cnt;
    }
}

// Bucket Sort: Phase 3 - Distribute elements to buckets
kernel void bucket_distribute(device const float* in [[buffer(0)]],
                         device float* out [[buffer(1)]],
                         device const uint* bucket_ids [[buffer(2)]],
                         device const uint* offsets [[buffer(3)]],
                         device atomic_uint* bucket_pos [[buffer(4)]],
                         constant uint& size [[buffer(5)]],
                         constant uint& num_buckets [[buffer(6)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= size) return;

    float val = in[id];
    uint bucket = bucket_ids[id];
    uint myOffset = atomic_fetch_add_explicit(&bucket_pos[bucket], 1, memory_order_relaxed);
    out[offsets[bucket] + myOffset] = val;
}

// Bucket Sort: Phase 4 - Sort within each bucket (simple insertion sort)
kernel void bucket_local_sort(device float* bucket_data [[buffer(0)]],
                         device const uint* bucket_offsets [[buffer(1)]],
                         device const uint* bucket_counts [[buffer(2)]],
                         device float* temp [[buffer(3)]],
                         constant uint& num_buckets [[buffer(4)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= num_buckets) return;

    uint start = bucket_offsets[id];
    uint count = bucket_counts[id];

    if (count <= 1) return;

    // Copy to temp
    for (uint i = 0; i < count; i++) {
        temp[i] = bucket_data[start + i];
    }

    // Simple insertion sort
    for (uint i = 1; i < count; i++) {
        float key = temp[i];
        uint j = i;
        while (j > 0 && temp[j - 1] > key) {
            temp[j] = temp[j - 1];
            j--;
        }
        temp[j] = key;
    }

    // Copy back
    for (uint i = 0; i < count; i++) {
        bucket_data[start + i] = temp[i];
    }
}

// Histogram-based counting (for integer keys)
kernel void bucket_histogram(device const uint* keys [[buffer(0)]],
                        device atomic_uint* histogram [[buffer(1)]],
                        constant uint& size [[buffer(2)]],
                        constant uint& num_bins [[buffer(3)]],
                        uint id [[thread_position_in_grid]]) {
    if (id >= size) return;

    uint key = keys[id];
    uint bin = key % num_bins;
    atomic_fetch_add_explicit(&histogram[bin], 1, memory_order_relaxed);
}

// Parallel bucket assign with atomic counters
kernel void bucket_assign(device const uint* keys [[buffer(0)]],
                     device uint* positions [[buffer(1)]],
                     device const uint* histogram [[buffer(2)]],
                     device const uint* prefix_sum [[buffer(3)]],
                     constant uint& size [[buffer(4)]],
                     constant uint& num_bins [[buffer(5)]],
                     uint id [[thread_position_in_grid]]) {
    if (id >= size) return;

    uint key = keys[id];
    uint bin = key % num_bins;
    uint rank_in_bin = id;  // Simplified: actual implementation needs per-bin counting
    positions[id] = prefix_sum[bin] + rank_in_bin;
}
"""

public struct BucketSortBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    public init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
    }

    public func run() throws {
        print(String(repeating: "=", count: 70))
        print("Bucket Sort / Hash-Based Distribution Benchmark")
        print(String(repeating: "=", count: 70))

        guard let library = try? device.makeLibrary(source: bucketSortShaders, options: nil) else {
            print("Failed to compile bucket sort shaders")
            return
        }

        let sizes = [16384, 65536, 262144, 1048576]  // 16K to 1M
        let numBuckets: UInt32 = 256

        for size in sizes {
            print("\n--- Size: \(size) elements ---")

            guard let inBuffer = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared),
                  let outBuffer = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared),
                  let countsBuffer = device.makeBuffer(length: Int(numBuckets) * MemoryLayout<UInt32>.size, options: .storageModeShared),
                  let idsBuffer = device.makeBuffer(length: size * MemoryLayout<UInt32>.size, options: .storageModeShared),
                  let offsetsBuffer = device.makeBuffer(length: Int(numBuckets) * MemoryLayout<UInt32>.size, options: .storageModeShared),
                  let posBuffer = device.makeBuffer(length: Int(numBuckets) * MemoryLayout<UInt32>.size, options: .storageModeShared),
                  let tempBuffer = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared) else {
                continue
            }

            // Initialize input with random values [0, 1)
            let inPtr = inBuffer.contents().bindMemory(to: Float.self, capacity: size)
            for i in 0..<size {
                inPtr[i] = Float.random(in: 0.0..<1.0)
            }

            var sizeVar = UInt32(size)
            var numBucketsVar = numBuckets

            // Get pipeline states
            guard let hashFunc = library.makeFunction(name: "bucket_hash"),
                  let scanFunc = library.makeFunction(name: "bucket_scan_counts"),
                  let distFunc = library.makeFunction(name: "bucket_distribute"),
                  let sortFunc = library.makeFunction(name: "bucket_local_sort"),
                  let hashPipeline = try? device.makeComputePipelineState(function: hashFunc),
                  let scanPipeline = try? device.makeComputePipelineState(function: scanFunc),
                  let distPipeline = try? device.makeComputePipelineState(function: distFunc),
                  let sortPipeline = try? device.makeComputePipelineState(function: sortFunc) else {
                continue
            }

            let iterations = 10
            let start = getTimeNanos()

            for _ in 0..<iterations {
                // Reset counts and positions
                let countsPtr = countsBuffer.contents().bindMemory(to: UInt32.self, capacity: Int(numBuckets))
                let posPtr = posBuffer.contents().bindMemory(to: UInt32.self, capacity: Int(numBuckets))
                for i in 0..<Int(numBuckets) {
                    countsPtr[i] = 0
                    posPtr[i] = 0
                }

                // Phase 1: Hash to buckets
                guard let cmd1 = queue.makeCommandBuffer(),
                      let enc1 = cmd1.makeComputeCommandEncoder() else { continue }
                enc1.setComputePipelineState(hashPipeline)
                enc1.setBuffer(inBuffer, offset: 0, index: 0)
                enc1.setBuffer(countsBuffer, offset: 0, index: 1)
                enc1.setBuffer(idsBuffer, offset: 0, index: 2)
                enc1.setBytes(&sizeVar, length: MemoryLayout<UInt32>.size, index: 3)
                enc1.setBytes(&numBucketsVar, length: MemoryLayout<UInt32>.size, index: 4)
                enc1.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                enc1.endEncoding()
                cmd1.commit()
                cmd1.waitUntilCompleted()

                // Phase 2: Scan bucket counts
                guard let cmd2 = queue.makeCommandBuffer(),
                      let enc2 = cmd2.makeComputeCommandEncoder() else { continue }
                enc2.setComputePipelineState(scanPipeline)
                enc2.setBuffer(countsBuffer, offset: 0, index: 0)
                enc2.setBuffer(offsetsBuffer, offset: 0, index: 1)
                enc2.setBytes(&numBucketsVar, length: MemoryLayout<UInt32>.size, index: 2)
                enc2.dispatchThreads(MTLSize(width: Int(numBuckets), height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
                enc2.endEncoding()
                cmd2.commit()
                cmd2.waitUntilCompleted()

                // Phase 3: Distribute elements
                guard let cmd3 = queue.makeCommandBuffer(),
                      let enc3 = cmd3.makeComputeCommandEncoder() else { continue }
                enc3.setComputePipelineState(distPipeline)
                enc3.setBuffer(inBuffer, offset: 0, index: 0)
                enc3.setBuffer(outBuffer, offset: 0, index: 1)
                enc3.setBuffer(idsBuffer, offset: 0, index: 2)
                enc3.setBuffer(offsetsBuffer, offset: 0, index: 3)
                enc3.setBuffer(posBuffer, offset: 0, index: 4)
                enc3.setBytes(&sizeVar, length: MemoryLayout<UInt32>.size, index: 5)
                enc3.setBytes(&numBucketsVar, length: MemoryLayout<UInt32>.size, index: 6)
                enc3.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                enc3.endEncoding()
                cmd3.commit()
                cmd3.waitUntilCompleted()

                // Phase 4: Local sort within buckets
                guard let cmd4 = queue.makeCommandBuffer(),
                      let enc4 = cmd4.makeComputeCommandEncoder() else { continue }
                enc4.setComputePipelineState(sortPipeline)
                enc4.setBuffer(outBuffer, offset: 0, index: 0)
                enc4.setBuffer(offsetsBuffer, offset: 0, index: 1)
                enc4.setBuffer(countsBuffer, offset: 0, index: 2)
                enc4.setBuffer(tempBuffer, offset: 0, index: 3)
                enc4.setBytes(&numBucketsVar, length: MemoryLayout<UInt32>.size, index: 4)
                enc4.dispatchThreads(MTLSize(width: Int(numBuckets), height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
                enc4.endEncoding()
                cmd4.commit()
                cmd4.waitUntilCompleted()
            }

            let end = getTimeNanos()
            let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
            let elementsPerSec = Double(size) / elapsed / 1e6

            print("Bucket Sort (\(numBuckets) buckets): \(String(format: "%.2f", elementsPerSec)) M elements/s")
            print("  \(String(format: "%.4f", elapsed * 1000)) ms per sort")
        }

        print("\n--- Key Findings ---")
        print("1. Bucket sort is a distribution sort - O(n)平均复杂度")
        print("2. Four phases: Hash → Scan → Distribute → Sort")
        print("3. Atomic operations needed for parallel bucket assignment")
        print("4.适合均匀分布的数据，分布不均会退化")
        print("5. 与Radix Sort结合是大数据排序的高效方案")
    }
}
