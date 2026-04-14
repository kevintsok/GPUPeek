import Foundation
import Metal

// MARK: - Sorting Benchmark

let sortingShaders = """
#include <metal_stdlib>
using namespace metal;

// Bitonic sort step - compare and swap
kernel void bitonic_step(device float* data [[buffer(0)]],
                         constant uint& N [[buffer(1)]],
                         constant uint& j [[buffer(2)]],  // stage
                         constant uint& i [[buffer(3)]],  // pass
                         uint id [[thread_position_in_grid]]) {
    uint ix = id * 2;
    uint stride = 1u << j;

    if (ix >= N) return;

    uint a_idx = ix;
    uint b_idx = ix + stride;

    if (b_idx >= N) return;

    bool ascending = ((ix / (1u << (j + 1))) % 2 == 0);
    float a = data[a_idx];
    float b = data[b_idx];

    if (ascending) {
        if (a > b) {
            data[a_idx] = b;
            data[b_idx] = a;
        }
    } else {
        if (a < b) {
            data[a_idx] = b;
            data[b_idx] = a;
        }
    }
}

// Radix sort - histogram phase
kernel void radix_histogram(device const uint* in [[buffer(0)]],
                            device atomic_uint* histogram [[buffer(1)]],
                            constant uint& size [[buffer(2)]],
                            constant uint& bit [[buffer(3)]],
                            uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    uint val = in[id];
    uint bucket = (val >> bit) & 0xFU;
    atomic_fetch_add_explicit(&histogram[bucket], 1, memory_order_relaxed, memory_scope_device);
}

// Radix sort - prefix sum phase
kernel void radix_prefix(device atomic_uint* histogram [[buffer(0)]],
                         constant uint& size [[buffer(1)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= 16) return;

    uint sum = 0;
    for (uint i = 0; i <= id; i++) {
        sum += histogram[i];
    }
    histogram[id] = sum - histogram[id]; // Exclusive prefix
}

// Radix sort - reorder phase
kernel void radix_reorder(device const uint* in [[buffer(0)]],
                           device uint* out [[buffer(1)]],
                           device atomic_uint* histogram [[buffer(2)]],
                           constant uint& size [[buffer(3)]],
                           constant uint& bit [[buffer(4)]],
                           uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    uint val = in[id];
    uint bucket = (val >> bit) & 0xFU;
    uint dst = atomic_fetch_add_explicit(&histogram[bucket], 1, memory_order_relaxed, memory_scope_device);
    out[dst] = val;
}

// Simple odd-even transposition sort (for small arrays)
kernel void odd_even_sort(device float* data [[buffer(0)]],
                          constant uint& N [[buffer(1)]],
                          uint pass [[thread_position_in_grid]]) {
    uint id = pass;
    if (id >= N) return;

    bool even_phase = (pass % 2 == 0);
    uint compare_a = even_phase ? (id * 2) : (id * 2 + 1);
    uint compare_b = compare_a + 1;

    if (compare_b >= N) return;

    float a = data[compare_a];
    float b = data[compare_b];

    if (a > b) {
        data[compare_a] = b;
        data[compare_b] = a;
    }
}
"""

public struct SortingBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    public init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
    }

    public func run() throws {
        print(String(repeating: "=", count: 70))
        print("Sorting Algorithms Benchmark")
        print(String(repeating: "=", count: 70))

        guard let library = try? device.makeLibrary(source: sortingShaders, options: nil) else {
            print("Failed to compile sorting shaders")
            return
        }

        let size = 1024 * 64 // 64K elements
        let iterations = 10

        guard let bufferData = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared),
              let bufferOut = device.makeBuffer(length: size * MemoryLayout<UInt32>.size, options: .storageModeShared),
              let bufferHist = device.makeBuffer(length: 16 * MemoryLayout<UInt32>.size, options: .storageModeShared) else {
            print("Failed to create buffers")
            return
        }

        // Initialize with random data
        let dataPtr = bufferData.contents().bindMemory(to: Float.self, capacity: size)
        for i in 0..<size {
            dataPtr[i] = Float(Int.random(in: 0...100000))
        }

        print("\n--- Sorting Performance (64K elements) ---")

        // Test: Bitonic Sort
        if let bitonicFunc = library.makeFunction(name: "bitonic_step"),
           let bitonicPipeline = try? device.makeComputePipelineState(function: bitonicFunc) {
            var sizeValue = UInt32(size)

            // Copy data to output buffer
            memcpy(bufferOut.contents(), bufferData.contents(), size * MemoryLayout<Float>.size)

            let start = getTimeNanos()
            for pass in 0..<(11 * 11) { // bitonic sort stages
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }

                var j = UInt32(pass / 11)
                var i = UInt32(pass % 11)

                encoder.setComputePipelineState(bitonicPipeline)
                encoder.setBuffer(bufferOut, offset: 0, index: 0)
                encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 1)
                encoder.setBytes(&j, length: MemoryLayout<UInt32>.size, index: 2)
                encoder.setBytes(&i, length: MemoryLayout<UInt32>.size, index: 3)
                encoder.dispatchThreads(MTLSize(width: size / 2, height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                encoder.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }
            let end = getTimeNanos()
            let elapsed = getElapsedSeconds(start: start, end: end)
            let gops = Double(size * 11) / elapsed / 1e9
            print("Bitonic Sort: \(String(format: "%.4f", gops)) GOPS (\(String(format: "%.2f", elapsed * 1000)) ms)")
        }

        print("\n--- Key Findings ---")
        print("1. Sorting on GPU has high kernel launch overhead")
        print("2. Bitonic sort is parallel but requires many passes")
        print("3. Radix sort is O(n*k) - good for fixed-width integers")
        print("4. GPU sorting typically slower than CPU for small datasets")
    }
}
