import Foundation
import Metal

// MARK: - Indirect Command Generation and Argument Buffers Benchmark

let indirectCommandShaders = """
#include <metal_stdlib>
using namespace metal;

// Kernel that computes dispatch arguments based on input data
// Simulates visibility-based draw call generation
kernel void compute_dispatch_args(device uint* visibleObjects [[buffer(0)]],
                                 device uint* dispatchArgs [[buffer(1)]],
                                 device atomic_uint* totalCount [[buffer(2)]],
                                 constant uint& maxObjects [[buffer(3)]],
                                 uint id [[thread_position_in_grid]]) {
    if (id >= maxObjects) return;

    // Simulate visibility test (in real use would be actual visibility)
    uint isVisible = (visibleObjects[id] > 0) ? 1 : 0;

    if (isVisible > 0) {
        uint idx = atomic_fetch_add_explicit(totalCount, 1, memory_order_relaxed);
        // dispatchArgs: [threadgroupCountX, threadgroupCountY, threadgroupCountZ]
        dispatchArgs[idx * 3 + 0] = 1;  // threadgroups in X
        dispatchArgs[idx * 3 + 1] = 1;  // threadgroups in Y
        dispatchArgs[idx * 3 + 2] = 1;  // threadgroups in Z
    }
}

// Simple compute kernel that processes visible objects
kernel void process_visible(device uint* input [[buffer(0)]],
                          device uint* output [[buffer(1)]],
                          constant uint& size [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    output[id] = input[id] * 2;
}

// Argument buffer style: batch process multiple objects
kernel void batch_process(device uint* batchOffsets [[buffer(0)]],
                        device uint* batchSizes [[buffer(1)]],
                        device uint* data [[buffer(2)]],
                        device uint* output [[buffer(3)]],
                        constant uint& numBatches [[buffer(4)]],
                        uint id [[thread_position_in_grid]]) {
    if (id >= numBatches) return;

    uint offset = batchOffsets[id];
    uint size = batchSizes[id];

    uint sum = 0;
    for (uint i = 0; i < size; i++) {
        sum += data[offset + i];
    }
    output[id] = sum;
}

// Predicate-based filtering using argument buffer
kernel void predicate_filter(device uint* flags [[buffer(0)]],
                           device uint* input [[buffer(1)]],
                           device uint* output [[buffer(2)]],
                           device atomic_uint* count [[buffer(3)]],
                           constant uint& size [[buffer(4)]],
                           constant uint& threshold [[buffer(5)]],
                           uint id [[thread_position_in_grid]]) {
    if (id >= size) return;

    if (flags[id] > threshold) {
        uint idx = atomic_fetch_add_explicit(count, 1, memory_order_relaxed);
        output[idx] = input[id];
    }
}
"""

public struct IndirectCommandBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    public init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
    }

    public func run() throws {
        print(String(repeating: "=", count: 70))
        print("Indirect Command Generation and Argument Buffers")
        print("GPU-driven command buffer construction")
        print(String(repeating: "=", count: 70))

        guard let indirectLibrary = try? device.makeLibrary(source: indirectCommandShaders, options: MTLCompileOptions()) else {
            print("Failed to create indirect command library")
            return
        }

        guard let dispatchFunc = indirectLibrary.makeFunction(name: "compute_dispatch_args"),
              let processFunc = indirectLibrary.makeFunction(name: "process_visible"),
              let batchFunc = indirectLibrary.makeFunction(name: "batch_process"),
              let filterFunc = indirectLibrary.makeFunction(name: "predicate_filter"),
              let dispatchPipeline = try? device.makeComputePipelineState(function: dispatchFunc),
              let processPipeline = try? device.makeComputePipelineState(function: processFunc),
              let batchPipeline = try? device.makeComputePipelineState(function: batchFunc),
              let filterPipeline = try? device.makeComputePipelineState(function: filterFunc) else {
            print("Failed to create indirect command pipelines")
            return
        }

        print("\n--- Argument Buffer Performance ---")
        print("Testing GPU-driven command generation patterns")

        let maxObjects = 4096
        let numVisible = 1024

        guard let visibleBuffer = device.makeBuffer(length: maxObjects * MemoryLayout<UInt32>.size, options: .storageModeShared),
              let dispatchBuffer = device.makeBuffer(length: maxObjects * 3 * MemoryLayout<UInt32>.size, options: .storageModeShared),
              let countBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.size, options: .storageModeShared),
              let dataBuffer = device.makeBuffer(length: maxObjects * MemoryLayout<UInt32>.size, options: .storageModeShared),
              let outputBuffer = device.makeBuffer(length: maxObjects * MemoryLayout<UInt32>.size, options: .storageModeShared) else {
            print("Failed to create buffers")
            return
        }

        let visiblePtr = visibleBuffer.contents().bindMemory(to: UInt32.self, capacity: maxObjects)
        let dataPtr = dataBuffer.contents().bindMemory(to: UInt32.self, capacity: maxObjects)
        for i in 0..<maxObjects {
            visiblePtr[i] = (i < numVisible) ? 1 : 0
            dataPtr[i] = UInt32(i + 1)
        }

        let countPtr = countBuffer.contents().bindMemory(to: UInt32.self, capacity: 1)
        countPtr[0] = 0

        let iterations = 100

        // Test 1: Compute dispatch arguments on GPU
        print("\n--- GPU-Driven Dispatch Arguments ---")
        let startDispatch = getTimeNanos()
        for _ in 0..<iterations {
            countPtr[0] = 0

            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(dispatchPipeline)
            encoder.setBuffer(visibleBuffer, offset: 0, index: 0)
            encoder.setBuffer(dispatchBuffer, offset: 0, index: 1)
            encoder.setBuffer(countBuffer, offset: 0, index: 2)
            var maxObj = UInt32(maxObjects)
            encoder.setBytes(&maxObj, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.dispatchThreads(MTLSize(width: maxObjects, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        let endDispatch = getTimeNanos()
        let dispatchTime = getElapsedSeconds(start: startDispatch, end: endDispatch)
        let dispatchThroughput = Double(maxObjects) * Double(iterations) / dispatchTime / 1e6
        print("| \(maxObjects) | \(String(format: "%.2f", dispatchThroughput)) M/s |")

        // Test 2: Batch processing with offsets
        print("\n--- Batch Processing (Argument Buffer Style) ---")
        let numBatches = 256

        guard let offsetsBuffer = device.makeBuffer(length: numBatches * MemoryLayout<UInt32>.size, options: .storageModeShared),
              let sizesBuffer = device.makeBuffer(length: numBatches * MemoryLayout<UInt32>.size, options: .storageModeShared) else {
            print("Failed to create batch buffers")
            return
        }

        let offsetsPtr = offsetsBuffer.contents().bindMemory(to: UInt32.self, capacity: numBatches)
        let sizesPtr = sizesBuffer.contents().bindMemory(to: UInt32.self, capacity: numBatches)
        for i in 0..<numBatches {
            offsetsPtr[i] = UInt32(i * 16)
            sizesPtr[i] = 16
        }

        let startBatch = getTimeNanos()
        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(batchPipeline)
            encoder.setBuffer(offsetsBuffer, offset: 0, index: 0)
            encoder.setBuffer(sizesBuffer, offset: 0, index: 1)
            encoder.setBuffer(dataBuffer, offset: 0, index: 2)
            encoder.setBuffer(outputBuffer, offset: 0, index: 3)
            var numBatch = UInt32(numBatches)
            encoder.setBytes(&numBatch, length: MemoryLayout<UInt32>.size, index: 4)
            encoder.dispatchThreads(MTLSize(width: numBatches, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        let endBatch = getTimeNanos()
        let batchTime = getElapsedSeconds(start: startBatch, end: endBatch)
        let batchThroughput = Double(numBatches) * Double(iterations) / batchTime / 1e6
        print("| \(numBatches) batches | \(String(format: "%.2f", batchThroughput)) M/s |")

        // Test 3: Predicate-based filtering
        print("\n--- Predicate Filtering (GPU-driven selection) ---")
        let filterSizes = [256, 1024, 4096]

        print("| Size | Throughput |")
        print("|------|------------|")

        for size in filterSizes {
            guard let flagsBuffer = device.makeBuffer(length: size * MemoryLayout<UInt32>.size, options: .storageModeShared),
                  let inputBuffer = device.makeBuffer(length: size * MemoryLayout<UInt32>.size, options: .storageModeShared),
                  let outBuffer = device.makeBuffer(length: size * MemoryLayout<UInt32>.size, options: .storageModeShared),
                  let cntBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.size, options: .storageModeShared) else {
                continue
            }

            let flagsPtr = flagsBuffer.contents().bindMemory(to: UInt32.self, capacity: size)
            let inPtr = inputBuffer.contents().bindMemory(to: UInt32.self, capacity: size)
            for i in 0..<size {
                flagsPtr[i] = UInt32(i)
                inPtr[i] = UInt32(i * 2)
            }
            let cntP = cntBuffer.contents().bindMemory(to: UInt32.self, capacity: 1)
            cntP[0] = 0

            let startF = getTimeNanos()
            for _ in 0..<iterations {
                cntP[0] = 0

                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(filterPipeline)
                encoder.setBuffer(flagsBuffer, offset: 0, index: 0)
                encoder.setBuffer(inputBuffer, offset: 0, index: 1)
                encoder.setBuffer(outBuffer, offset: 0, index: 2)
                encoder.setBuffer(cntBuffer, offset: 0, index: 3)
                var sz = UInt32(size)
                var thr = UInt32(size / 2)
                encoder.setBytes(&sz, length: MemoryLayout<UInt32>.size, index: 4)
                encoder.setBytes(&thr, length: MemoryLayout<UInt32>.size, index: 5)
                encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: min(size, 256), height: 1, depth: 1))
                encoder.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }
            let endF = getTimeNanos()
            let fTime = getElapsedSeconds(start: startF, end: endF)
            let fTP = Double(size) * Double(iterations) / fTime / 1e6
            print("| \(size) | \(String(format: "%.2f", fTP)) M/s |")
        }

        print("\n--- Key Insights ---")
        print("1. Indirect command generation allows GPU to drive dispatch decisions")
        print("2. Argument buffers enable batched processing with variable sizes")
        print("3. Predicate filtering uses GPU-generated flags for selection")
        print("4. These patterns reduce CPU-GPU synchronization overhead")
        print("5. Useful for visibility culling, occlusion queries, dynamic scenes")
    }
}
