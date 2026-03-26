import Foundation
import Metal

// MARK: - Priority Queue Benchmark

let priorityQueueShaders = """
#include <metal_stdlib>
using namespace metal;

// Binary heap: push operation (insert element)
// Heap property: parent > children (max-heap)
kernel void heap_push(device uint* heap [[buffer(0)]],
                 device atomic_uint* heap_size [[buffer(1)]],
                 constant uint& value [[buffer(2)]],
                 uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;  // Only first thread does work

    uint size = atomic_fetch_add_explicit(heap_size, 1, memory_order_relaxed, memory_scope_device);
    heap[size] = value;

    // Bubble up
    uint idx = size;
    while (idx > 0) {
        uint parent = (idx - 1) / 2;
        if (heap[parent] >= heap[idx]) break;
        uint temp = heap[parent];
        heap[parent] = heap[idx];
        heap[idx] = temp;
        idx = parent;
    }
}

// Binary heap: pop operation (remove max)
kernel void heap_pop(device uint* heap [[buffer(0)]],
                 device atomic_uint* heap_size [[buffer(1)]],
                 device uint* output [[buffer(2)]],
                 uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;

    uint size = atomic_load_explicit(heap_size, memory_order_relaxed, memory_scope_device);
    if (size == 0) {
        output[0] = 0xFFFFFFFF;  // sentinel for empty
        return;
    }

    uint max = heap[0];
    output[0] = max;

    // Move last element to root
    uint last = heap[size - 1];
    atomic_fetch_sub_explicit(heap_size, 1, memory_order_relaxed, memory_scope_device);
    heap[0] = last;

    // Bubble down
    uint idx = 0;
    while (true) {
        uint left = 2 * idx + 1;
        uint right = 2 * idx + 2;
        uint largest = idx;

        if (left < size && heap[left] > heap[largest])
            largest = left;
        if (right < size && heap[right] > heap[largest])
            largest = right;

        if (largest == idx) break;

        uint temp = heap[idx];
        heap[idx] = heap[largest];
        heap[largest] = temp;
        idx = largest;
    }
}

// Parallel bucket sort (batch priority queue operations)
kernel void bucket_sort_batch(device const uint* in [[buffer(0)]],
                          device uint* out [[buffer(1)]],
                          device atomic_uint* counts [[buffer(2)]],
                          constant uint& size [[buffer(3)]],
                          constant uint& num_buckets [[buffer(4)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= size) return;

    uint val = in[id];
    uint bucket = val * num_buckets / 256;  // Simple hash

    uint offset = atomic_fetch_add_explicit(&counts[bucket], 1,
                                          memory_order_relaxed, memory_scope_device);
    out[bucket * size + offset] = val;
}

// Radix sort for priority queue preparation
kernel void radix_count(device const uint* in [[buffer(0)]],
                       device atomic_uint* counts [[buffer(1)]],
                       constant uint& size [[buffer(2)]],
                       constant uint& bit [[buffer(3)]],
                       uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    uint val = in[id];
    uint bit_val = (val >> bit) & 1u;
    atomic_fetch_add_explicit(&counts[bit_val], 1,
                            memory_order_relaxed, memory_scope_device);
}

kernel void radix_reorder(device const uint* in [[buffer(0)]],
                        device uint* out [[buffer(1)]],
                        device const uint* counts [[buffer(2)]],
                        constant uint& size [[buffer(3)]],
                        constant uint& bit [[buffer(4)]],
                        uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    uint val = in[id];
    uint bit_val = (val >> bit) & 1u;

    uint offset = 0;
    for (uint i = 0; i < bit_val; i++) {
        offset += counts[i];
    }

    if (bit_val == 1) {
        offset += counts[0];  // Add zeros count for ones
    }

    out[offset] = val;
}

// Top-K selection using partial sort
kernel void topk_select(device const uint* in [[buffer(0)]],
                       device uint* out [[buffer(1)]],
                       constant uint& size [[buffer(2)]],
                       constant uint& k [[buffer(3)]],
                       uint id [[thread_position_in_grid]]) {
    if (id >= size) return;

    uint val = in[id];
    uint count = 0;

    for (uint i = 0; i < size; i++) {
        if (in[i] > val) count++;
        if (count >= k) break;
    }

    if (count < k) {
        out[count] = val;
    }
}
"""

public struct PriorityQueueBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    public init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
    }

    public func run() throws {
        print(String(repeating: "=", count: 70))
        print("Priority Queue Benchmark")
        print(String(repeating: "=", count: 70))

        guard let library = try? device.makeLibrary(source: priorityQueueShaders, options: nil) else {
            print("Failed to compile priority queue shaders")
            return
        }

        let sizes = [1024, 4096, 16384]

        for size in sizes {
            print("\n--- Heap Size: \(size) ---")

            let heapSize = size * 2  // Extra space for heap
            guard let heapBuf = device.makeBuffer(length: heapSize * MemoryLayout<UInt32>.size, options: .storageModeShared),
                  let sizeBuf = device.makeBuffer(length: MemoryLayout<UInt32>.size, options: .storageModeShared),
                  let outputBuf = device.makeBuffer(length: size * MemoryLayout<UInt32>.size, options: .storageModeShared) else {
                continue
            }

            // Initialize heap size to 0
            let sizePtr = sizeBuf.contents().bindMemory(to: UInt32.self, capacity: 1)
            sizePtr.pointee = 0

            var sizeValue = UInt32(size)

            // Test heap push (sequential by design)
            if let pushFunc = library.makeFunction(name: "heap_push"),
               let pushPipeline = try? device.makeComputePipelineState(function: pushFunc) {
                let iterations = 100
                let start = getTimeNanos()
                for i in 0..<iterations {
                    sizePtr.pointee = 0  // Reset

                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }

                    var val = UInt32(i % 256)

                    encoder.setComputePipelineState(pushPipeline)
                    encoder.setBuffer(heapBuf, offset: 0, index: 0)
                    encoder.setBuffer(sizeBuf, offset: 0, index: 1)
                    encoder.setBytes(&val, length: MemoryLayout<UInt32>.size, index: 2)
                    encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
                    encoder.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()
                }
                let end = getTimeNanos()
                let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
                let ops = 1.0 / elapsed
                print("Heap Push: \(String(format: "%.0f", ops)) ops/s (\(String(format: "%.4f", elapsed * 1000)) ms)")
            }

            // Test heap pop (sequential by design)
            if let popFunc = library.makeFunction(name: "heap_pop"),
               let popPipeline = try? device.makeComputePipelineState(function: popFunc) {
                // First populate heap
                let populateCmd = queue.makeCommandBuffer()
                if let enc = populateCmd?.makeComputeCommandEncoder() {
                    enc.setBuffer(heapBuf, offset: 0, index: 0)
                    enc.setBuffer(sizeBuf, offset: 0, index: 1)
                    // Add some initial elements
                }
                populateCmd?.commit()
                populateCmd?.waitUntilCompleted()

                let iterations = 100
                let start = getTimeNanos()
                for _ in 0..<iterations {
                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }

                    encoder.setComputePipelineState(popPipeline)
                    encoder.setBuffer(heapBuf, offset: 0, index: 0)
                    encoder.setBuffer(sizeBuf, offset: 0, index: 1)
                    encoder.setBuffer(outputBuf, offset: 0, index: 2)
                    encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
                    encoder.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()
                }
                let end = getTimeNanos()
                let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
                let ops = 1.0 / elapsed
                print("Heap Pop: \(String(format: "%.0f", ops)) ops/s (\(String(format: "%.4f", elapsed * 1000)) ms)")
            }
        }

        // Test bucket sort (parallel batch operation)
        print("\n--- Batch Operations ---")

        for size in [16384, 65536] {
            let numBuckets: UInt32 = 256

            guard let inBuf = device.makeBuffer(length: size * MemoryLayout<UInt32>.size, options: .storageModeShared),
                  let outBuf = device.makeBuffer(length: size * MemoryLayout<UInt32>.size, options: .storageModeShared),
                  let countBuf = device.makeBuffer(length: Int(numBuckets) * MemoryLayout<UInt32>.size, options: .storageModeShared) else {
                continue
            }

            // Initialize input
            let inPtr = inBuf.contents().bindMemory(to: UInt32.self, capacity: size)
            for i in 0..<size {
                inPtr[i] = UInt32(i % 256)  // Values 0-255
            }

            // Initialize counts
            let countPtr = countBuf.contents().bindMemory(to: UInt32.self, capacity: Int(numBuckets))
            for i in 0..<Int(numBuckets) {
                countPtr[i] = 0
            }

            var sizeValue = UInt32(size)

            if let bucketFunc = library.makeFunction(name: "bucket_sort_batch"),
               let bucketPipeline = try? device.makeComputePipelineState(function: bucketFunc) {
                let iterations = 10
                let start = getTimeNanos()
                for _ in 0..<iterations {
                    // Reset counts
                    for i in 0..<Int(numBuckets) {
                        countPtr[i] = 0
                    }

                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }

                    encoder.setComputePipelineState(bucketPipeline)
                    encoder.setBuffer(inBuf, offset: 0, index: 0)
                    encoder.setBuffer(outBuf, offset: 0, index: 1)
                    encoder.setBuffer(countBuf, offset: 0, index: 2)
                    encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 3)
                    encoder.setBytes(&numBuckets, length: MemoryLayout<UInt32>.size, index: 4)
                    encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    encoder.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()
                }
                let end = getTimeNanos()
                let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
                let throughput = Double(size) / elapsed / 1e6
                print("Bucket Sort (\(size)): \(String(format: "%.2f", throughput)) M elements/s (\(String(format: "%.2f", elapsed * 1000)) ms)")
            }
        }

        print("\n--- Key Findings ---")
        print("1. Binary heap push/pop are sequential - limited parallelism")
        print("2. GPU priority queues use batch operations for parallelism")
        print("3. Bucket sort as alternative for batch priority operations")
        print("4. Top-K selection uses partial sorting")
        print("5. Priority queues on GPU often use specialized data structures")
    }
}
