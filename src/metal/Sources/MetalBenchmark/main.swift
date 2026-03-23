import Foundation
import Metal
import QuartzCore
import simd

// MARK: - Timer

func getTimeNanos() -> UInt64 {
    var info = mach_timebase_info_data_t()
    mach_timebase_info(&info)
    let elapsed = mach_absolute_time()
    return elapsed * UInt64(info.numer) / UInt64(info.denom)
}

func getTimeInterval(start: UInt64, end: UInt64) -> Double {
    var info = mach_timebase_info_data_t()
    mach_timebase_info(&info)
    let elapsed = end - start
    return Double(elapsed) * Double(info.numer) / Double(info.denom) / 1e9
}

// MARK: - Phase 4: Parallel Computing Shader Library

let shaderSource = """
#include <metal_stdlib>
using namespace metal;

// Threadgroup size scaling test - different sizes
kernel void threadgroup_scale_small(device const float* input [[buffer(0)]],
                                   device float* output [[buffer(1)]],
                                   threadgroup float* shared [[threadgroup(0)]],
                                   constant uint& size [[buffer(2)]],
                                   uint id [[thread_position_in_grid]],
                                   uint lid [[thread_position_in_threadgroup]]) {
    constexpr uint LOCAL_SIZE = 64;
    shared[lid] = input[id];
    threadgroup_barrier(mem_flags::mem_none);
    output[id] = shared[lid % LOCAL_SIZE];
}

kernel void threadgroup_scale_medium(device const float* input [[buffer(0)]],
                                    device float* output [[buffer(1)]],
                                    threadgroup float* shared [[threadgroup(0)]],
                                    constant uint& size [[buffer(2)]],
                                    uint id [[thread_position_in_grid]],
                                    uint lid [[thread_position_in_threadgroup]]) {
    constexpr uint LOCAL_SIZE = 256;
    shared[lid] = input[id];
    threadgroup_barrier(mem_flags::mem_none);
    output[id] = shared[lid % LOCAL_SIZE];
}

kernel void threadgroup_scale_large(device const float* input [[buffer(0)]],
                                   device float* output [[buffer(1)]],
                                   threadgroup float* shared [[threadgroup(0)]],
                                   constant uint& size [[buffer(2)]],
                                   uint id [[thread_position_in_grid]],
                                   uint lid [[thread_position_in_threadgroup]]) {
    constexpr uint LOCAL_SIZE = 512;
    shared[lid] = input[id];
    threadgroup_barrier(mem_flags::mem_none);
    output[id] = shared[lid % LOCAL_SIZE];
}

kernel void threadgroup_scale_xlarge(device const float* input [[buffer(0)]],
                                    device float* output [[buffer(1)]],
                                    threadgroup float* shared [[threadgroup(0)]],
                                    constant uint& size [[buffer(2)]],
                                    uint id [[thread_position_in_grid]],
                                    uint lid [[thread_position_in_threadgroup]]) {
    constexpr uint LOCAL_SIZE = 1024;
    shared[lid] = input[id];
    threadgroup_barrier(mem_flags::mem_none);
    output[id] = shared[lid % LOCAL_SIZE];
}

// SIMD vector operations test
kernel void simd_vector_add(device const float4* a [[buffer(0)]],
                            device const float4* b [[buffer(1)]],
                            device float4* c [[buffer(2)]],
                            constant uint& size [[buffer(3)]],
                            uint id [[thread_position_in_grid]]) {
    float4 va = a[id];
    float4 vb = b[id];
    float4 vc = va + vb;
    c[id] = vc;
}

kernel void simd_vector_mul(device const float4* a [[buffer(0)]],
                            device const float4* b [[buffer(1)]],
                            device float4* c [[buffer(2)]],
                            constant uint& size [[buffer(3)]],
                            uint id [[thread_position_in_grid]]) {
    float4 va = a[id];
    float4 vb = b[id];
    float4 vc = va * vb;
    c[id] = vc;
}

kernel void simd_dot_product(device const float4* a [[buffer(0)]],
                            device const float4* b [[buffer(1)]],
                            device float* result [[buffer(2)]],
                            constant uint& size [[buffer(3)]],
                            uint id [[thread_position_in_grid]]) {
    float4 va = a[id];
    float4 vb = b[id];
    float4 vr = va * vb;
    // Horizontal add using simd
    float sum = vr[0] + vr[1] + vr[2] + vr[3];
    result[id] = sum;
}

// Atomic counter operations
kernel void atomic_add_counter(device atomic_uint* counters [[buffer(0)]],
                               constant uint& num_counters [[buffer(1)]],
                               constant uint& iterations [[buffer(2)]],
                               uint id [[thread_position_in_grid]]) {
    uint counter_idx = id % num_counters;
    for (uint i = 0; i < iterations; i++) {
        atomic_fetch_add_explicit(&counters[counter_idx], 1, memory_order_relaxed);
    }
}

kernel void atomic_add_single(device atomic_uint* result [[buffer(0)]],
                             constant uint& iterations [[buffer(1)]],
                             uint id [[thread_position_in_grid]]) {
    if (id == 0) {
        for (uint i = 0; i < iterations; i++) {
            atomic_fetch_add_explicit(result, 1, memory_order_relaxed);
        }
    }
}

// Thread divergence test - conditional branches
kernel void thread_divergence_test(device const float* input [[buffer(0)]],
                                   device float* output [[buffer(1)]],
                                   constant uint& threshold [[buffer(2)]],
                                   uint id [[thread_position_in_grid]]) {
    float val = input[id];
    float result = 0.0f;
    if (val > float(threshold)) {
        // Path 1: compute-intensive
        for (int i = 0; i < 10; i++) {
            result += metal::sqrt(val) * metal::sin(val);
        }
    } else {
        // Path 2: memory-intensive
        for (int i = 0; i < 10; i++) {
            result += val * 2.0f;
        }
    }
    output[id] = result;
}

// Warp-level reduction simulation
kernel void warp_reduction(device const float* input [[buffer(0)]],
                           device float* output [[buffer(1)]],
                           constant uint& size [[buffer(2)]],
                           uint id [[thread_position_in_grid]]) {
    float val = input[id];
    uint tid = id & 31;

    // Simulate warp shuffle reduction
    for (uint offset = 16; offset > 0; offset >>= 1) {
        uint other = id + offset;
        if (other < size && tid < offset) {
            val += input[other];
        }
    }

    if (tid == 0) {
        output[id / 32] = val;
    }
}

// SIMD shuffle test
kernel void simd_shuffle_test(device const float4* input [[buffer(0)]],
                              device float4* output [[buffer(1)]],
                              uint id [[thread_position_in_grid]]) {
    float4 v = input[id];

    // SIMD shuffle operations
    float4 s0 = v.xyzw;
    float4 s1 = v.zwxy;
    float4 s2 = v.wxyz;

    output[id] = (s0 + s1 + s2) / 3.0f;
}

// Reduction with shared memory
kernel void shared_reduction(device const float* input [[buffer(0)]],
                             device float* result [[buffer(1)]],
                             constant uint& size [[buffer(2)]],
                             threadgroup float* shared [[threadgroup(0)]],
                             uint id [[thread_position_in_grid]],
                             uint lid [[thread_position_in_threadgroup]]) {
    constexpr uint THREADGROUP_SIZE = 256;
    shared[lid] = input[id];
    threadgroup_barrier(mem_flags::mem_none);

    for (uint s = THREADGROUP_SIZE / 2; s > 0; s >>= 1) {
        if (lid < s) {
            shared[lid] += shared[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_none);
    }

    if (lid == 0) {
        result[0] = shared[0];
    }
}

// Barrier overhead test
kernel void barrier_overhead_test(device const float* input [[buffer(0)]],
                                  device float* output [[buffer(1)]],
                                  threadgroup float* shared [[threadgroup(0)]],
                                  constant uint& iterations [[buffer(2)]],
                                  uint id [[thread_position_in_grid]],
                                  uint lid [[thread_position_in_threadgroup]]) {
    float val = input[id];
    for (uint i = 0; i < iterations; i++) {
        shared[lid] = val;
        threadgroup_barrier(mem_flags::mem_none);
        if (lid == 0) {
            val = shared[0];
        }
        threadgroup_barrier(mem_flags::mem_none);
    }
    output[id] = val;
}
"""

// MARK: - Device Info

func printDeviceInfo(device: MTLDevice) {
    print("\n=== Apple Metal GPU Info ===")
    print("Device Name: \(device.name)")
    print("Max Threadgroup Memory: \(device.maxThreadgroupMemoryLength / 1024) KB")
    print("Max Threads Per Threadgroup: \(device.maxThreadsPerThreadgroup.width)")

    if device.supportsFamily(.apple7) {
        print("GPU Family: Apple 7+")
    }

    print("\n")
}

// MARK: - Test: Threadgroup Size Scaling

func testThreadgroupScaling(device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary) throws {
    print("=== Threadgroup Size Scaling ===")

    let kernels = [
        ("threadgroup_scale_small", 64),
        ("threadgroup_scale_medium", 256),
        ("threadgroup_scale_large", 512),
        ("threadgroup_scale_xlarge", 1024)
    ]

    let bufferSize = 32 * 1024 * 1024
    let iterations = 20

    for (name, threadgroupSize) in kernels {
        guard let function = library.makeFunction(name: name),
              let pipeline = try? device.makeComputePipelineState(function: function) else {
            print("Failed to create pipeline: \(name)")
            continue
        }

        guard let inputBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared),
              let outputBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
            continue
        }

        let input = inputBuffer.contents().assumingMemoryBound(to: Float.self)
        for i in 0..<(bufferSize / MemoryLayout<Float>.size) {
            input[i] = Float(i) * 0.001
        }

        var size = UInt32(bufferSize / MemoryLayout<Float>.size)
        let threadsPerThreadgroup = MTLSize(width: threadgroupSize, height: 1, depth: 1)
        let numThreadgroups = (bufferSize / MemoryLayout<Float>.size) / threadgroupSize

        let start = getTimeNanos()
        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(inputBuffer, offset: 0, index: 0)
            encoder.setBuffer(outputBuffer, offset: 0, index: 1)
            encoder.setBytes(&size, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.dispatchThreadgroups(MTLSize(width: numThreadgroups, height: 1, depth: 1),
                                        threadsPerThreadgroup: threadsPerThreadgroup)
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        let end = getTimeNanos()
        let elapsed = getTimeInterval(start: start, end: end)

        let bandwidth = Double(bufferSize) * Double(iterations) / elapsed / 1e9
        print("Threadgroup \(threadgroupSize): \(String(format: "%.2f", bandwidth)) GB/s")
    }
    print("")
}

// MARK: - Test: SIMD Operations

func testSIMDOperations(device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary) throws {
    print("=== SIMD Vector Operations ===")

    let kernels = ["simd_vector_add", "simd_vector_mul", "simd_dot_product", "simd_shuffle_test"]
    let names = ["Vector Add (float4)", "Vector Mul (float4)", "Dot Product (float4)", "SIMD Shuffle"]

    let bufferSize = 32 * 1024 * 1024
    let iterations = 50
    let numElements = bufferSize / MemoryLayout<SIMD4<Float>>.size

    for (i, name) in kernels.enumerated() {
        guard let function = library.makeFunction(name: name),
              let pipeline = try? device.makeComputePipelineState(function: function) else {
            print("Failed to create pipeline: \(name)")
            continue
        }

        guard let aBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared),
              let bBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared),
              let cBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
            continue
        }

        let a = aBuffer.contents().assumingMemoryBound(to: SIMD4<Float>.self)
        let b = bBuffer.contents().assumingMemoryBound(to: SIMD4<Float>.self)
        for j in 0..<numElements {
            a[j] = SIMD4<Float>(Float(j), Float(j + 1), Float(j + 2), Float(j + 3))
            b[j] = SIMD4<Float>(0.001, 0.002, 0.003, 0.004)
        }

        var size = UInt32(numElements)

        let start = getTimeNanos()
        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(aBuffer, offset: 0, index: 0)
            encoder.setBuffer(bBuffer, offset: 0, index: 1)
            encoder.setBuffer(cBuffer, offset: 0, index: 2)
            if name != "simd_shuffle_test" {
                encoder.setBytes(&size, length: MemoryLayout<UInt32>.size, index: 3)
            }
            encoder.dispatchThreads(MTLSize(width: numElements, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        let end = getTimeNanos()
        let elapsed = getTimeInterval(start: start, end: end)

        // float4 = 16 bytes, operations per element vary
        let flops: Double
        if name == "simd_vector_add" {
            flops = Double(numElements) * Double(iterations) * 1 // 1 add per element
        } else if name == "simd_vector_mul" {
            flops = Double(numElements) * Double(iterations) * 1 // 1 mul per element
        } else if name == "simd_dot_product" {
            flops = Double(numElements) * Double(iterations) * 4 // 4 muls + 3 adds
        } else {
            flops = Double(numElements) * Double(iterations) * 3 // 3 adds
        }
        let gflops = flops / elapsed / 1e9
        print("\(names[i]): \(String(format: "%.2f", gflops)) GFLOPS")
    }
    print("")
}

// MARK: - Test: Atomic Operations

func testAtomicOperations(device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary) throws {
    print("=== Atomic Operations ===")

    let numCounters = [1, 16, 64, 256, 1024]
    var iterations: UInt32 = 1000
    let totalThreads = 8192

    // Test: Contention scaling
    print("Contention Scaling (atomic_fetch_add):")
    for numCounter in numCounters {
        guard let function = library.makeFunction(name: "atomic_add_counter"),
              let pipeline = try? device.makeComputePipelineState(function: function) else {
            continue
        }

        let counterSize = numCounter * MemoryLayout<UInt32>.size
        guard let counterBuffer = device.makeBuffer(length: counterSize, options: .storageModeShared) else {
            continue
        }

        // Initialize counters to 0 (Metal atomics will handle atomic semantics)
        let counters = counterBuffer.contents().assumingMemoryBound(to: UInt32.self)
        for i in 0..<numCounter {
            counters[i] = 0
        }

        var nc = UInt32(numCounter)
        var iters = iterations

        let start = getTimeNanos()
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(counterBuffer, offset: 0, index: 0)
        encoder.setBytes(&nc, length: MemoryLayout<UInt32>.size, index: 1)
        encoder.setBytes(&iters, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.dispatchThreads(MTLSize(width: totalThreads, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        let end = getTimeNanos()
        let elapsed = getTimeInterval(start: start, end: end)

        let totalOps = UInt64(totalThreads) * UInt64(iterations)
        let gops = Double(totalOps) / elapsed / 1e9

        // Verify
        var sum: UInt32 = 0
        for i in 0..<numCounter {
            sum += counters[i]
        }
        let verified = (sum == totalOps) ? "PASS" : "FAIL"

        print("  \(numCounter) counters: \(String(format: "%.3f", gops)) GOPS, verified: \(verified)")
    }

    // Single atomic bottleneck test
    print("\nSingle Atomic Bottleneck:")
    guard let function = library.makeFunction(name: "atomic_add_single"),
          let pipeline = try? device.makeComputePipelineState(function: function) else {
        return
    }

    guard let counterBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.size, options: .storageModeShared) else {
        return
    }
    let counter = counterBuffer.contents().assumingMemoryBound(to: UInt32.self)
    counter[0] = 0

    let start = getTimeNanos()
    guard let cmd = queue.makeCommandBuffer(),
          let encoder = cmd.makeComputeCommandEncoder() else { return }
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(counterBuffer, offset: 0, index: 0)
    encoder.setBytes(&iterations, length: MemoryLayout<UInt32>.size, index: 1)
    encoder.dispatchThreads(MTLSize(width: totalThreads, height: 1, depth: 1),
                         threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    encoder.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    let end = getTimeNanos()
    let elapsed = getTimeInterval(start: start, end: end)

    let totalOps = UInt64(totalThreads) * UInt64(iterations)
    let gops = Double(totalOps) / elapsed / 1e9
    print("  All threads to single counter: \(String(format: "%.3f", gops)) GOPS")

    print("")
}

// MARK: - Test: Thread Divergence

func testThreadDivergence(device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary) throws {
    print("=== Thread Divergence Test ===")

    guard let function = library.makeFunction(name: "thread_divergence_test"),
          let pipeline = try? device.makeComputePipelineState(function: function) else {
        print("Failed to create pipeline")
        return
    }

    let bufferSize = 8 * 1024 * 1024
    let iterations = 10

    guard let inputBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared),
          let outputBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
        return
    }

    let input = inputBuffer.contents().assumingMemoryBound(to: Float.self)
    for i in 0..<(bufferSize / MemoryLayout<Float>.size) {
        input[i] = Float(i % 256) / 256.0
    }

    let thresholds: [UInt32] = [64, 128, 192] // 25%, 50%, 75% branches take different paths
    for threshold in thresholds {
        var thresh = threshold

        let start = getTimeNanos()
        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(inputBuffer, offset: 0, index: 0)
            encoder.setBuffer(outputBuffer, offset: 0, index: 1)
            encoder.setBytes(&thresh, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.dispatchThreads(MTLSize(width: bufferSize / MemoryLayout<Float>.size, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        let end = getTimeNanos()
        let elapsed = getTimeInterval(start: start, end: end)

        let ops = Double(bufferSize / MemoryLayout<Float>.size) * Double(iterations) * 10 // 10 iterations in kernel
        let gflops = ops / elapsed / 1e9
        let branchPercent = Double(threshold) / 256.0 * 100
        print("Threshold \(branchPercent)%: \(String(format: "%.2f", gflops)) GFLOPS")
    }
    print("")
}

// MARK: - Test: Barrier Overhead

func testBarrierOverhead(device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary) throws {
    print("=== Threadgroup Barrier Overhead ===")

    guard let function = library.makeFunction(name: "barrier_overhead_test"),
          let pipeline = try? device.makeComputePipelineState(function: function) else {
        print("Failed to create pipeline")
        return
    }

    let bufferSize = 4 * 1024 * 1024
    let iterations = [1, 10, 100, 500]

    guard let inputBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared),
          let outputBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
        return
    }

    let input = inputBuffer.contents().assumingMemoryBound(to: Float.self)
    for i in 0..<(bufferSize / MemoryLayout<Float>.size) {
        input[i] = Float(i) * 0.001
    }

    for iter in iterations {
        var iters = UInt32(iter)

        let start = getTimeNanos()
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else { continue }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        encoder.setBytes(&iters, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.dispatchThreadgroups(MTLSize(width: bufferSize / MemoryLayout<Float>.size / 256, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        let end = getTimeNanos()
        let elapsed = getTimeInterval(start: start, end: end)

        let totalBarriers = (bufferSize / MemoryLayout<Float>.size / 256) * iter
        let barrierNs = elapsed * 1e9 / Double(totalBarriers)
        print("\(iter) barriers per thread: \(String(format: "%.2f", barrierNs)) ns/barrier")
    }
    print("")
}

// MARK: - Test: Shared Memory Reduction

func testSharedMemoryReduction(device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary) throws {
    print("=== Shared Memory Reduction ===")

    guard let function = library.makeFunction(name: "shared_reduction"),
          let pipeline = try? device.makeComputePipelineState(function: function) else {
        print("Failed to create pipeline")
        return
    }

    let sizes = [65536, 262144, 1048576] // Various reduction sizes
    let iterations = 100

    for size in sizes {
        let bufferSize = size * MemoryLayout<Float>.size

        guard let inputBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared),
              let outputBuffer = device.makeBuffer(length: MemoryLayout<Float>.size, options: .storageModeShared) else {
            continue
        }

        let input = inputBuffer.contents().assumingMemoryBound(to: Float.self)
        for i in 0..<size {
            input[i] = Float(i) * 0.001
        }

        var sz = UInt32(size)

        let start = getTimeNanos()
        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(inputBuffer, offset: 0, index: 0)
            encoder.setBuffer(outputBuffer, offset: 0, index: 1)
            encoder.setBytes(&sz, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.dispatchThreadgroups(MTLSize(width: size / 256, height: 1, depth: 1),
                                        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        let end = getTimeNanos()
        let elapsed = getTimeInterval(start: start, end: end)

        // Reduction: (n-1) adds per element in tree
        let flops = Double(size - 1) * Double(iterations)
        let gflops = flops / elapsed / 1e9

        print("Size \(size): \(String(format: "%.2f", gflops)) GFLOPS")
    }
    print("")
}

// MARK: - Main

print("Apple Metal GPU Benchmark - Phase 4: Parallel Computing Characteristics")
print("======================================")

guard let device = MTLCreateSystemDefaultDevice() else {
    print("Metal is not supported on this device")
    exit(1)
}

printDeviceInfo(device: device)

guard let queue = device.makeCommandQueue() else {
    print("Failed to create command queue")
    exit(1)
}

let library: MTLLibrary
do {
    library = try device.makeLibrary(source: shaderSource, options: nil)
} catch {
    print("Failed to create shader library: \(error)")
    exit(1)
}

print("Shader compilation: SUCCESS\n")

// Threadgroup Scaling
print("--- Threadgroup Performance ---")
do {
    try testThreadgroupScaling(device: device, queue: queue, library: library)
} catch {
    print("Error: \(error)")
}

// SIMD Operations
print("--- SIMD Operations ---")
do {
    try testSIMDOperations(device: device, queue: queue, library: library)
} catch {
    print("Error: \(error)")
}

// Atomic Operations
print("--- Atomic Operations ---")
do {
    try testAtomicOperations(device: device, queue: queue, library: library)
} catch {
    print("Error: \(error)")
}

// Thread Divergence
print("--- Thread Divergence ---")
do {
    try testThreadDivergence(device: device, queue: queue, library: library)
} catch {
    print("Error: \(error)")
}

// Barrier Overhead
print("--- Barrier Overhead ---")
do {
    try testBarrierOverhead(device: device, queue: queue, library: library)
} catch {
    print("Error: \(error)")
}

// Shared Memory Reduction
print("--- Shared Memory Reduction ---")
do {
    try testSharedMemoryReduction(device: device, queue: queue, library: library)
} catch {
    print("Error: \(error)")
}

print("Phase 4 benchmark completed.")
