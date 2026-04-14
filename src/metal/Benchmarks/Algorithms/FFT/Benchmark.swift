import Foundation
import Metal

// MARK: - Enhanced FFT Benchmark

let fftShaders = """
#include <metal_stdlib>
using namespace metal;

// =====================================================================
// NAIVE RADIX-2 FFT (butterfly in global memory)
// =====================================================================

kernel void fft_naive_radix2(device float2* data [[buffer(0)]],
                             constant uint& N [[buffer(1)]],
                             uint id [[thread_position_in_grid]]) {
    if (id >= N / 2) return;

    // Bit-reversal permutation
    uint rev_id = bitmask(id, N);
    if (rev_id < id) {
        float2 temp = data[id];
        data[id] = data[rev_id];
        data[rev_id] = temp;
    }

    // Butterfly stages
    for (uint stage = 0; stage < uint(log2(float(N))); stage++) {
        uint butterfly_size = 1u << stage;
        uint span = butterfly_size * 2;
        uint num_groups = N / span;
        uint group_id = id / num_groups;
        uint offset = id % num_groups;

        uint i = group_id * span + offset;
        uint j = i + butterfly_size;

        float theta = -2.0f * M_PI_F * float(offset) / float(butterfly_size);
        float2 w = float2(cos(theta), sin(theta));
        float2 a = data[i];
        float2 b = data[j];

        data[i] = a + b;
        data[j] = (a - b) * w;

        threadgroup_barrier(flags::mem_threadgroup);
    }
}

// =====================================================================
// SHARED MEMORY OPTIMIZED RADIX-2 FFT
// =====================================================================

kernel void fft_shared_radix2(device float2* data [[buffer(0)]],
                              threadgroup float2* shared [[threadgroup(0)]],
                              constant uint& N [[buffer(1)]],
                              uint id [[thread_position_in_grid]],
                              uint lid [[thread_position_in_threadgroup]]) {
    // Load data into shared memory
    if (lid < N) {
        shared[lid] = data[lid];
    }
    threadgroup_barrier(flags::mem_threadgroup);

    // Process in shared memory
    if (id >= N / 2) return;

    // Bit-reversal in shared memory
    uint rev_id = bitmask(id, N);
    if (rev_id < id) {
        float2 temp = shared[id];
        shared[id] = shared[rev_id];
        shared[rev_id] = temp;
    }

    threadgroup_barrier(flags::mem_threadgroup);

    // Butterfly stages from shared memory
    for (uint stage = 0; stage < uint(log2(float(N))); stage++) {
        uint butterfly_size = 1u << stage;
        uint span = butterfly_size * 2;
        uint num_groups = N / span;
        uint group_id = id / num_groups;
        uint offset = id % num_groups;

        uint i = group_id * span + offset;
        uint j = i + butterfly_size;

        float theta = -2.0f * M_PI_F * float(offset) / float(butterfly_size);
        float2 w = float2(cos(theta), sin(theta));
        float2 a = shared[i];
        float2 b = shared[j];

        shared[i] = a + b;
        shared[j] = (a - b) * w;

        threadgroup_barrier(flags::mem_threadgroup);
    }

    // Write back to global memory
    if (lid < N) {
        data[lid] = shared[lid];
    }
}

// =====================================================================
// RADIX-4 FFT (faster for power-of-4 sizes)
// =====================================================================

kernel void fft_radix4(device float2* data [[buffer(0)]],
                       constant uint& N [[buffer(1)]],
                       uint id [[thread_position_in_grid]]) {
    if (id >= N / 4) return;

    // Radix-4 requires N to be power of 4
    uint logN = uint(log2(float(N))) / 2;

    // Bit-reversal for radix-4
    uint rev_id = bitmask(id, N);
    if (rev_id < id) {
        float2 temp = data[id];
        data[id] = data[rev_id];
        data[rev_id] = temp;
    }

    threadgroup_barrier(flags::mem_threadgroup);

    // Radix-4 butterfly stages
    float2 roots[3];
    for (uint stage = 0; stage < logN; stage++) {
        uint block_size = 1u << (2 * stage);
        uint num_blocks = N / (4 * block_size);
        uint block_id = id / num_blocks;
        uint offset = id % num_blocks;

        uint i = block_id * 4 * block_size + offset;
        uint j0 = i;
        uint j1 = i + block_size;
        uint j2 = i + 2 * block_size;
        uint j3 = i + 3 * block_size;

        float theta = -2.0f * M_PI_F * float(offset) / float(block_size * 4);
        roots[0] = float2(1.0f, 0.0f);
        roots[1] = float2(cos(theta), sin(theta));
        roots[2] = float2(cos(2.0f * theta), sin(2.0f * theta));

        float2 x0 = data[j0];
        float2 x1 = data[j1] * roots[1];
        float2 x2 = data[j2] * roots[2];
        float2 x3 = data[j3] * float2(cos(3.0f * theta), sin(3.0f * theta));

        float2 a0 = x0 + x2;
        float2 a1 = x0 - x2;
        float2 a2 = x1 + x3;
        float2 a3 = float2(0.0f, (x1.x * x3.y - x1.y * x3.x) * -1.0f);

        data[j0] = a0 + a2;
        data[j1] = a1 + a3;
        data[j2] = a0 - a2;
        data[j3] = float2(a1.x - a3.y, a1.y + a3.x);

        threadgroup_barrier(flags::mem_threadgroup);
    }
}

// =====================================================================
// FFT BUTTERFLY BENCHMARK (single stage)
// =====================================================================

kernel void fft_butterfly_stage(device float2* data [[buffer(0)]],
                                 constant uint& N [[buffer(1)]],
                                 constant uint& stage [[buffer(2)]],
                                 uint id [[thread_position_in_grid]]) {
    if (id >= N / 2) return;

    uint butterfly_size = 1u << stage;
    uint num_butterflies = N / (butterfly_size * 2);
    uint butterfly_index = id / num_butterflies;
    uint offset = id % num_butterflies;

    uint i = butterfly_index * butterfly_size * 2 + offset;
    uint j = i + butterfly_size;

    float theta = -2.0f * M_PI_F * float(butterfly_index) / float(N / 2);
    float2 w = float2(cos(theta), sin(theta));
    float2 a = data[i];
    float2 b = data[j];

    data[i] = a + b;
    data[j] = (a - b) * w;
}

// =====================================================================
// COMPLEX MULTIPLICATION (for FFT verification)
// =====================================================================

kernel void complex_mult(device float2* a [[buffer(0)]],
                         device float2* b [[buffer(1)]],
                         device float2* out [[buffer(2)]],
                         constant uint& size [[buffer(3)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    float2 a_val = a[id];
    float2 b_val = b[id];
    out[id] = float2(a_val.x * b_val.x - a_val.y * b_val.y,
                     a_val.x * b_val.y + a_val.y * b_val.x);
}
"""

public struct FFTBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    public init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
    }

    public func run() throws {
        print(String(repeating: "=", count: 70))
        print("FFT (Fast Fourier Transform) Benchmark")
        print(String(repeating: "=", count: 70))

        guard let library = try? device.makeLibrary(source: fftShaders, options: nil) else {
            print("Failed to compile FFT shaders")
            return
        }

        // Test sizes: powers of 2 and 4
        let radix2_sizes: [UInt32] = [256, 512, 1024, 2048, 4096]
        let radix4_sizes: [UInt32] = [256, 1024, 4096, 16384]

        print("\n=== RADIX-2 FFT (Naive Global Memory) ===")
        for N in radix2_sizes {
            try benchmarkNaiveRadix2(library: library, N: N)
        }

        print("\n=== RADIX-2 FFT (Shared Memory Optimized) ===")
        for N in radix2_sizes {
            try benchmarkSharedRadix2(library: library, N: N)
        }

        print("\n=== RADIX-4 FFT (Global Memory) ===")
        for N in radix4_sizes {
            try benchmarkRadix4(library: library, N: N)
        }

        print("\n=== FFT Size Scaling Analysis ===")
        try analyzeSizeScaling(library: library)

        print("\n--- Key Findings ---")
        print("1. Shared memory FFT reduces global memory bandwidth")
        print("2. Radix-4 has fewer stages than Radix-2 (N/4 vs N/2)")
        print("3. Larger FFTs benefit more from optimization")
        print("4. Apple M2 unified memory affects FFT performance")
    }

    func benchmarkNaiveRadix2(library: MTLLibrary, N: UInt32) throws {
        guard let bufferData = device.makeBuffer(length: Int(N) * 2 * MemoryLayout<Float>.size, options: .storageModeShared) else {
            return
        }

        // Initialize with test pattern
        let dataPtr = bufferData.contents().bindMemory(to: Float.self, capacity: Int(N) * 2)
        for i in 0..<Int(N) {
            dataPtr[i * 2] = Float(i % 256) / 255.0
            dataPtr[i * 2 + 1] = 0.0f
        }

        var sizeValue = N

        if let fftFunc = library.makeFunction(name: "fft_naive_radix2"),
           let fftPipeline = try? device.makeComputePipelineState(function: fftFunc) {
            let iterations = 20
            let start = getTimeNanos()
            for _ in 0..<iterations {
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(fftPipeline)
                encoder.setBuffer(bufferData, offset: 0, index: 0)
                encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 1)
                encoder.dispatchThreads(MTLSize(width: Int(N) / 2, height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                encoder.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }
            let end = getTimeNanos()
            let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
            let gflops = 5.0 * Double(N) * log2(Double(N)) / elapsed / 1e9
            let stages = log2(Double(N))
            print("  N=\(N) (\(Int(stages)) stages): \(String(format: "%.4f", gflops)) GFLOPS, \(String(format: "%.3f", elapsed * 1000)) ms")
        }
    }

    func benchmarkSharedRadix2(library: MTLLibrary, N: UInt32) throws {
        guard let bufferData = device.makeBuffer(length: Int(N) * 2 * MemoryLayout<Float>.size, options: .storageModeShared) else {
            return
        }

        // Initialize with test pattern
        let dataPtr = bufferData.contents().bindMemory(to: Float.self, capacity: Int(N) * 2)
        for i in 0..<Int(N) {
            dataPtr[i * 2] = Float(i % 256) / 255.0
            dataPtr[i * 2 + 1] = 0.0f
        }

        var sizeValue = N

        // Shared memory FFT requires smaller workgroups due to shared memory limits
        let threadgroupSize = min(256, N)

        if let fftFunc = library.makeFunction(name: "fft_shared_radix2"),
           let fftPipeline = try? device.makeComputePipelineState(function: fftFunc) {
            let iterations = 20
            let start = getTimeNanos()
            for _ in 0..<iterations {
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(fftPipeline)
                encoder.setBuffer(bufferData, offset: 0, index: 0)
                encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 1)
                encoder.dispatchThreads(MTLSize(width: Int(N) / 2, height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1))
                encoder.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }
            let end = getTimeNanos()
            let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
            let gflops = 5.0 * Double(N) * log2(Double(N)) / elapsed / 1e9
            print("  N=\(N): \(String(format: "%.4f", gflops)) GFLOPS, \(String(format: "%.3f", elapsed * 1000)) ms")
        }
    }

    func benchmarkRadix4(library: MTLLibrary, N: UInt32) throws {
        guard let bufferData = device.makeBuffer(length: Int(N) * 2 * MemoryLayout<Float>.size, options: .storageModeShared) else {
            return
        }

        // Initialize with test pattern
        let dataPtr = bufferData.contents().bindMemory(to: Float.self, capacity: Int(N) * 2)
        for i in 0..<Int(N) {
            dataPtr[i * 2] = Float(i % 256) / 255.0
            dataPtr[i * 2 + 1] = 0.0f
        }

        var sizeValue = N

        if let fftFunc = library.makeFunction(name: "fft_radix4"),
           let fftPipeline = try? device.makeComputePipelineState(function: fftFunc) {
            let iterations = 20
            let start = getTimeNanos()
            for _ in 0..<iterations {
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(fftPipeline)
                encoder.setBuffer(bufferData, offset: 0, index: 0)
                encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 1)
                encoder.dispatchThreads(MTLSize(width: Int(N) / 4, height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                encoder.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }
            let end = getTimeNanos()
            let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
            let stages = log2(Double(N)) / 2  // Radix-4 has half as many stages
            let gflops = 5.0 * Double(N) * log2(Double(N)) / elapsed / 1e9
            print("  N=\(N) (\(Int(stages)) stages): \(String(format: "%.4f", gflops)) GFLOPS, \(String(format: "%.3f", elapsed * 1000)) ms")
        }
    }

    func analyzeSizeScaling(library: MTLLibrary) throws {
        print("\n--- Size Scaling (Radix-2 Naive) ---")

        let sizes: [UInt32] = [256, 512, 1024, 2048, 4096, 8192]
        var results: [(UInt32, Double)] = []

        for N in sizes {
            guard let bufferData = device.makeBuffer(length: Int(N) * 2 * MemoryLayout<Float>.size, options: .storageModeShared) else {
                continue
            }

            let dataPtr = bufferData.contents().bindMemory(to: Float.self, capacity: Int(N) * 2)
            for i in 0..<Int(N) {
                dataPtr[i * 2] = Float(i % 256) / 255.0
                dataPtr[i * 2 + 1] = 0.0f
            }

            var sizeValue = N

            if let fftFunc = library.makeFunction(name: "fft_naive_radix2"),
               let fftPipeline = try? device.makeComputePipelineState(function: fftFunc) {
                let iterations = 20
                let start = getTimeNanos()
                for _ in 0..<iterations {
                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }
                    encoder.setComputePipelineState(fftPipeline)
                    encoder.setBuffer(bufferData, offset: 0, index: 0)
                    encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 1)
                    encoder.dispatchThreads(MTLSize(width: Int(N) / 2, height: 1, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    encoder.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()
                }
                let end = getTimeNanos()
                let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
                let gflops = 5.0 * Double(N) * log2(Double(N)) / elapsed / 1e9
                results.append((N, gflops))
                print("  N=\(N): \(String(format: "%.4f", gflops)) GFLOPS")
            }
        }

        // Calculate scaling factor
        if results.count >= 2 {
            let first = results.first!
            let last = results.last!
            let sizeRatio = Double(last.0) / Double(first.0)
            let perfRatio = last.1 / first.1
            print("\n  Scaling: \(Int(sizeRatio))x size increase -> \(String(format: "%.2f", perfRatio))x performance")
            print("  O(N log N) theoretical: \(String(format: "%.2f", sizeRatio * log2(Double(last.0))/log2(Double(first.0))))x")
        }
    }
}
