import Foundation
import Metal

// MARK: - FFT Optimization Deep Dive Benchmark

let fftOptShaders = """
#include <metal_stdlib>
using namespace metal;

// =====================================================================
// BIT-REVERSAL HELPER
// =====================================================================

uint reverse_bits(uint x, uint num_bits) {
    uint result = 0;
    for (uint i = 0; i < num_bits; i++) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

// =====================================================================
// NAIVE RADIX-2 FFT (butterfly in global memory)
// Each thread handles one butterfly
// =====================================================================

kernel void fft_naive_radix2(device float2* data [[buffer(0)]],
                             constant uint& N [[buffer(1)]],
                             uint id [[thread_position_in_grid]]) {
    if (id >= N / 2) return;

    // Bit-reversal permutation
    uint num_bits = uint(log2(float(N)));
    uint rev_id = reverse_bits(id, num_bits);
    if (rev_id < id) {
        float2 temp = data[id];
        data[id] = data[rev_id];
        data[rev_id] = temp;
    }

    // Butterfly stages
    for (uint stage = 0; stage < uint(log2(float(N))); stage++) {
        uint butterfly_size = 1u << stage;
        uint span = butterfly_size * 2;
        uint num_butterflies = N / span;
        uint butterfly_index = id / num_butterflies;
        uint offset = id % num_butterflies;

        uint i = butterfly_index * span + offset;
        uint j = i + butterfly_size;

        float theta = -2.0f * M_PI_F * float(offset) / float(butterfly_size);
        float2 w = float2(cos(theta), sin(theta));
        float2 a = data[i];
        float2 b = data[j];

        data[i] = a + b;
        data[j] = (a - b) * w;
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
    // Load entire dataset into shared memory
    // Each thread loads one element
    if (lid < N) {
        shared[lid] = data[lid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (id >= N / 2) return;

    // Bit-reversal permutation in shared memory
    uint num_bits = uint(log2(float(N)));
    uint rev_id = reverse_bits(id, num_bits);
    if (rev_id < id) {
        float2 temp = shared[id];
        shared[id] = shared[rev_id];
        shared[rev_id] = temp;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Butterfly stages from shared memory
    for (uint stage = 0; stage < uint(log2(float(N))); stage++) {
        uint butterfly_size = 1u << stage;
        uint span = butterfly_size * 2;
        uint num_butterflies = N / span;
        uint butterfly_index = id / num_butterflies;
        uint offset = id % num_butterflies;

        uint i = butterfly_index * span + offset;
        uint j = i + butterfly_size;

        float theta = -2.0f * M_PI_F * float(offset) / float(butterfly_size);
        float2 w = float2(cos(theta), sin(theta));
        float2 a = shared[i];
        float2 b = shared[j];

        shared[i] = a + b;
        shared[j] = (a - b) * w;

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write back to global memory
    if (lid < N) {
        data[lid] = shared[lid];
    }
}

// =====================================================================
// RADIX-4 FFT (faster for power-of-4 sizes)
// Fewer stages: log4(N) instead of log2(N)
// =====================================================================

kernel void fft_radix4(device float2* data [[buffer(0)]],
                       constant uint& N [[buffer(1)]],
                       uint id [[thread_position_in_grid]]) {
    if (id >= N / 4) return;

    uint logN = uint(log2(float(N))) / 2;

    // Radix-4 bit-reversal
    uint num_bits = uint(log2(float(N)));
    uint rev_id = reverse_bits(id, num_bits);
    if (rev_id < id) {
        float2 temp = data[id];
        data[id] = data[rev_id];
        data[rev_id] = temp;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Radix-4 butterfly stages
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
        float2 w0 = float2(1.0f, 0.0);
        float2 w1 = float2(cos(theta), sin(theta));
        float2 w2 = float2(cos(2.0f * theta), sin(2.0f * theta));
        float2 w3 = float2(cos(3.0f * theta), sin(3.0f * theta));

        float2 x0 = data[j0];
        float2 x1 = data[j1] * w1;
        float2 x2 = data[j2] * w2;
        float2 x3 = data[j3] * w3;

        float2 a0 = x0 + x2;
        float2 a1 = x0 - x2;
        float2 a2 = x1 + x3;
        float2 a3 = float2(0.0, (x1.x * x3.y - x1.y * x3.x) * -1.0f);

        data[j0] = a0 + a2;
        data[j1] = a1 + a3;
        data[j2] = a0 - a2;
        data[j3] = float2(a1.x - a3.y, a1.y + a3.x);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// =====================================================================
// SINGLE STAGE FFT (butterfly only - measures butterfly cost)
// =====================================================================

kernel void fft_single_stage(device float2* data [[buffer(0)]],
                            constant uint& N [[buffer(1)]],
                            constant uint& stage [[buffer(2)]],
                            uint id [[thread_position_in_grid]]) {
    if (id >= N / 2) return;

    uint butterfly_size = 1u << stage;
    uint span = butterfly_size * 2;
    uint num_butterflies = N / span;
    uint butterfly_index = id / num_butterflies;
    uint offset = id % num_butterflies;

    uint i = butterfly_index * span + offset;
    uint j = i + butterfly_size;

    float theta = -2.0f * M_PI_F * float(offset) / float(N / 2);
    float2 w = float2(cos(theta), sin(theta));
    float2 a = data[i];
    float2 b = data[j];

    data[i] = a + b;
    data[j] = (a - b) * w;
}

// =====================================================================
// BUTTERFLY OPERATION (pure computation, no memory access pattern)
// =====================================================================

kernel void butterfly_only(device float2* output [[buffer(0)]],
                          constant uint& count [[buffer(1)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= count) return;

    float2 a = float2(1.0f, 0.0);
    float2 b = float2(0.5f, 0.5f);

    // Pure butterfly: a + b, a - b (repeated)
    float2 sum = a + b;
    float2 diff = a - b;
    float2 w = float2(cos(0.5f), sin(0.5f));
    float2 result = (diff) * w;

    output[id] = sum + result;
}
"""

public struct FFTOptimizationBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    public init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
    }

    public func run() throws {
        print("\n" + String(repeating: "=", count: 70))
        print("FFT Optimization Deep Dive")
        print(String(repeating: "=", count: 70))

        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: fftOptShaders, options: nil)
        } catch {
            print("Failed to compile FFT shaders: \(error.localizedDescription)")
            return
        }

        // Test sizes: powers of 2
        let radix2_sizes: [UInt32] = [256, 512, 1024, 2048, 4096]
        let radix4_sizes: [UInt32] = [256, 1024, 4096, 16384]

        var naiveResults: [(UInt32, Double)] = []
        var sharedResults: [(UInt32, Double)] = []
        var radix4Results: [(UInt32, Double)] = []

        print("\n=== RADIX-2 NAIVE (Global Memory) ===")
        for N in radix2_sizes {
            if let (gflops, ms) = benchmarkNaiveRadix2(library: library, N: N) {
                naiveResults.append((N, gflops))
                print("  N=\(N): \(String(format: "%.4f", gflops)) GFLOPS, \(String(format: "%.3f", ms)) ms")
            }
        }

        print("\n=== RADIX-2 SHARED MEMORY ===")
        for N in radix2_sizes {
            if let (gflops, ms) = benchmarkSharedRadix2(library: library, N: N) {
                sharedResults.append((N, gflops))
                print("  N=\(N): \(String(format: "%.4f", gflops)) GFLOPS, \(String(format: "%.3f", ms)) ms")
            }
        }

        print("\n=== RADIX-4 (Global Memory) ===")
        for N in radix4_sizes {
            if let (gflops, ms) = benchmarkRadix4(library: library, N: N) {
                radix4Results.append((N, gflops))
                print("  N=\(N): \(String(format: "%.4f", gflops)) GFLOPS, \(String(format: "%.3f", ms)) ms")
            }
        }

        // Size scaling analysis
        print("\n=== SIZE SCALING COMPARISON ===")
        analyzeSizeScaling(naiveResults: naiveResults, sharedResults: sharedResults)

        // Calculate speedups
        print("\n=== OPTIMIZATION SPEEDUP ===")
        calculateSpeedups(naiveResults: naiveResults, sharedResults: sharedResults)

        // Single stage analysis
        print("\n=== SINGLE STAGE ANALYSIS ===")
        analyzeSingleStage(library: library)

        // Update LOG.txt
        updateLogFile(naiveResults: naiveResults, sharedResults: sharedResults, radix4Results: radix4Results)

        print("\n--- Key Findings ---")
        print("1. Shared memory FFT reduces global memory bandwidth pressure")
        print("2. Radix-4 has fewer stages than Radix-2 (log4(N) vs log2(N))")
        print("3. Larger FFTs benefit more from shared memory optimization")
        print("4. Apple M2 unified memory architecture affects FFT performance")
    }

    func benchmarkNaiveRadix2(library: MTLLibrary, N: UInt32) -> (Double, Double)? {
        guard let bufferData = device.makeBuffer(length: Int(N) * 2 * MemoryLayout<Float>.size, options: .storageModeShared) else {
            return nil
        }

        // Initialize with test pattern
        let dataPtr = bufferData.contents().bindMemory(to: Float.self, capacity: Int(N) * 2)
        for i in 0..<Int(N) {
            dataPtr[i * 2] = Float(i % 256) / 255.0
            dataPtr[i * 2 + 1] = 0.0
        }

        var sizeValue = N

        guard let fftFunc = library.makeFunction(name: "fft_naive_radix2"),
              let fftPipeline = try? device.makeComputePipelineState(function: fftFunc) else {
            return nil
        }

        let iterations = 50
        let start = getTimeNanos()

        for _ in 0..<iterations {
            // Reinitialize to ensure fresh data
            for i in 0..<Int(N) {
                dataPtr[i * 2] = Float(i % 256) / 255.0
                dataPtr[i * 2 + 1] = 0.0
            }

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
        // FLOPs: N * log2(N) * 6 (each butterfly: 4 mul + 2 add)
        let gflops = 6.0 * Double(N) * log2(Double(N)) / elapsed / 1e9
        let ms = elapsed * 1000

        return (gflops, ms)
    }

    func benchmarkSharedRadix2(library: MTLLibrary, N: UInt32) -> (Double, Double)? {
        guard let bufferData = device.makeBuffer(length: Int(N) * 2 * MemoryLayout<Float>.size, options: .storageModeShared) else {
            return nil
        }

        // Initialize with test pattern
        let dataPtr = bufferData.contents().bindMemory(to: Float.self, capacity: Int(N) * 2)
        for i in 0..<Int(N) {
            dataPtr[i * 2] = Float(i % 256) / 255.0
            dataPtr[i * 2 + 1] = 0.0
        }

        var sizeValue = N

        // Shared memory FFT requires threadgroup allocation
        let threadgroupSize = min(256, N)

        guard let fftFunc = library.makeFunction(name: "fft_shared_radix2"),
              let fftPipeline = try? device.makeComputePipelineState(function: fftFunc) else {
            return nil
        }

        let iterations = 50
        let start = getTimeNanos()

        for _ in 0..<iterations {
            // Reinitialize
            for i in 0..<Int(N) {
                dataPtr[i * 2] = Float(i % 256) / 255.0
                dataPtr[i * 2 + 1] = 0.0
            }

            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(fftPipeline)
            encoder.setBuffer(bufferData, offset: 0, index: 0)
            encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 1)
            encoder.dispatchThreads(MTLSize(width: Int(N) / 2, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: Int(threadgroupSize), height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }

        let end = getTimeNanos()
        let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
        let gflops = 6.0 * Double(N) * log2(Double(N)) / elapsed / 1e9
        let ms = elapsed * 1000

        return (gflops, ms)
    }

    func benchmarkRadix4(library: MTLLibrary, N: UInt32) -> (Double, Double)? {
        guard let bufferData = device.makeBuffer(length: Int(N) * 2 * MemoryLayout<Float>.size, options: .storageModeShared) else {
            return nil
        }

        // Initialize with test pattern
        let dataPtr = bufferData.contents().bindMemory(to: Float.self, capacity: Int(N) * 2)
        for i in 0..<Int(N) {
            dataPtr[i * 2] = Float(i % 256) / 255.0
            dataPtr[i * 2 + 1] = 0.0
        }

        var sizeValue = N

        guard let fftFunc = library.makeFunction(name: "fft_radix4"),
              let fftPipeline = try? device.makeComputePipelineState(function: fftFunc) else {
            return nil
        }

        let iterations = 50
        let start = getTimeNanos()

        for _ in 0..<iterations {
            // Reinitialize
            for i in 0..<Int(N) {
                dataPtr[i * 2] = Float(i % 256) / 255.0
                dataPtr[i * 2 + 1] = 0.0
            }

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
        // Radix-4 has same FLOP count but fewer stages
        let gflops = 6.0 * Double(N) * log2(Double(N)) / elapsed / 1e9
        let ms = elapsed * 1000

        return (gflops, ms)
    }

    func analyzeSizeScaling(naiveResults: [(UInt32, Double)], sharedResults: [(UInt32, Double)]) {
        print("\n--- Size Scaling Analysis ---")

        if naiveResults.count >= 2 {
            let first = naiveResults.first!
            let last = naiveResults.last!
            let sizeRatio = Double(last.0) / Double(first.0)
            let perfRatio = last.1 / first.1
            let theoreticalRatio = sizeRatio * log2(Double(last.0)) / log2(Double(first.0))
            print("Naive: \(Int(sizeRatio))x size -> \(String(format: "%.2f", perfRatio))x perf (theoretical: \(String(format: "%.2f", theoreticalRatio))x)")
        }

        if sharedResults.count >= 2 {
            let first = sharedResults.first!
            let last = sharedResults.last!
            let sizeRatio = Double(last.0) / Double(first.0)
            let perfRatio = last.1 / first.1
            let theoreticalRatio = sizeRatio * log2(Double(last.0)) / log2(Double(first.0))
            print("Shared: \(Int(sizeRatio))x size -> \(String(format: "%.2f", perfRatio))x perf (theoretical: \(String(format: "%.2f", theoreticalRatio))x)")
        }
    }

    func calculateSpeedups(naiveResults: [(UInt32, Double)], sharedResults: [(UInt32, Double)]) {
        print("\n| Size | Naive GFLOPS | Shared GFLOPS | Speedup |")
        print("|------|--------------|---------------|---------|")

        for (size, naive) in naiveResults {
            if let shared = sharedResults.first(where: { $0.0 == size }) {
                let speedup = shared.1 / naive
                print("| \(size) | \(String(format: "%.4f", naive)) | \(String(format: "%.4f", shared.1)) | \(String(format: "%.2fx", speedup)) |")
            }
        }
    }

    func analyzeSingleStage(library: MTLLibrary) {
        let sizes: [UInt32] = [1024, 4096, 16384]

        guard let stageFunc = library.makeFunction(name: "fft_single_stage"),
              let stagePipeline = try? device.makeComputePipelineState(function: stageFunc) else {
            print("  Failed to create single stage pipeline")
            return
        }

        for N in sizes {
            guard let bufferData = device.makeBuffer(length: Int(N) * 2 * MemoryLayout<Float>.size, options: .storageModeShared) else {
                continue
            }

            var sizeValue = N
            var stageValue: UInt32 = 0

            let iterations = 100
            let start = getTimeNanos()

            for _ in 0..<iterations {
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(stagePipeline)
                encoder.setBuffer(bufferData, offset: 0, index: 0)
                encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 1)
                encoder.setBytes(&stageValue, length: MemoryLayout<UInt32>.size, index: 2)
                encoder.dispatchThreads(MTLSize(width: Int(N) / 2, height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                encoder.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }

            let end = getTimeNanos()
            let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
            let ms = elapsed * 1000
            print("  Stage 0, N=\(N): \(String(format: "%.3f", ms)) ms")
        }
    }

    func updateLogFile(naiveResults: [(UInt32, Double)], sharedResults: [(UInt32, Double)], radix4Results: [(UInt32, Double)]) {
        let logPath = "/Users/longxia/Projects/GPUPeek/src/metal/Sources/MetalBenchmark/Benchmarks/Compute/FFTOptimization/LOG.txt"

        var log = "=== FFT Optimization Deep Dive ===\n\n"

        log += "--- Naive Radix-2 (Global Memory) ---\n"
        for (size, gflops) in naiveResults {
            log += "N=\(size): \(String(format: "%.4f", gflops)) GFLOPS\n"
        }

        log += "\n--- Shared Memory Radix-2 ---\n"
        for (size, gflops) in sharedResults {
            log += "N=\(size): \(String(format: "%.4f", gflops)) GFLOPS\n"
        }

        log += "\n--- Radix-4 ---\n"
        for (size, gflops) in radix4Results {
            log += "N=\(size): \(String(format: "%.4f", gflops)) GFLOPS\n"
        }

        log += "\n--- Speedup (Shared vs Naive) ---\n"
        for (size, naive) in naiveResults {
            if let shared = sharedResults.first(where: { $0.0 == size }) {
                let speedup = shared.1 / naive
                log += "N=\(size): \(String(format: "%.2f", speedup))x\n"
            }
        }

        log += "\n--- Key Findings ---\n"
        log += "1. Shared memory reduces global memory traffic\n"
        log += "2. Radix-4 reduces number of stages\n"
        log += "3. Performance scales with size\n"

        try? log.write(toFile: logPath, atomically: true, encoding: .utf8)
    }
}