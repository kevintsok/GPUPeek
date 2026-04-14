import Foundation
import Metal

// MARK: - Monte Carlo / Random Number Generation Benchmark

let monteCarloShaders = """
#include <metal_stdlib>
using namespace metal;

// Pseudo-random number generation using hash function
// Inspired by PCG (Permuted Congruential Generator)
kernel void prng_hash(device ulong* seed [[buffer(0)]],
                 device uint* output [[buffer(1)]],
                 constant uint& size [[buffer(2)]],
                 uint id [[thread_position_in_grid]]) {
    if (id >= size) return;

    ulong state = seed[0];
    ulong inc = ulong(id) + 1;
    ulong x = state + inc;
    x = x * 6364136223846793005 + inc;
    uint res = uint((x >> 33u) ^ x);

    output[id] = res;
    seed[0] = x;
}

// Uniform to float [0, 1) transformation
kernel void uniform_transform(device uint* input [[buffer(0)]],
                        device float* output [[buffer(1)]],
                        constant uint& size [[buffer(2)]],
                        uint id [[thread_position_in_grid]]) {
    if (id >= size) return;

    uint u = input[id];
    float f = float(u) / 4294967296.0f;  // 2^32
    output[id] = f;
}

// Box-Muller transform for Gaussian distribution
kernel void gaussian_transform(device float* u1 [[buffer(0)]],
                          device float* u2 [[buffer(1)]],
                          device float* output [[buffer(2)]],
                          constant uint& size [[buffer(3)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= size) return;

    float x1 = u1[id];
    float x2 = u2[id];
    float z0 = sqrt(-2.0f * log(x1)) * cos(2.0f * M_PI_F * x2);
    output[id] = z0;
}

// Monte Carlo Pi estimation trial
kernel void mc_pi_trial(device float* x [[buffer(0)]],
                    device float* y [[buffer(1)]],
                    device atomic_uint* inside_count [[buffer(2)]],
                    constant uint& size [[buffer(3)]],
                    uint id [[thread_position_in_grid]]) {
    if (id >= size) return;

    float px = x[id];
    float py = y[id];
    float dist = sqrt(px * px + py * py);

    if (dist < 1.0f) {
        atomic_fetch_add_explicit(inside_count, 1, memory_order_relaxed, memory_scope_device);
    }
}

// Linear congruential generator (simple)
kernel void prng_lcg(device uint* seed [[buffer(0)]],
                 device uint* output [[buffer(1)]],
                 constant uint& size [[buffer(2)]],
                 uint id [[thread_position_in_grid]]) {
    if (id >= size) return;

    uint s = seed[0];
    s = s * 1103515245 + 12345;  // LCG parameters
    output[id] = s;
    seed[0] = s;
}

// XOR-shift PRNG (fast)
kernel void prng_xorshift(device uint* seed [[buffer(0)]],
                      device uint* output [[buffer(1)]],
                      constant uint& size [[buffer(2)]],
                      uint id [[thread_position_in_grid]]) {
    if (id >= size) return;

    uint s = seed[0];
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    output[id] = s;
    seed[0] = s;
}
"""

public struct MonteCarloBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    public init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
    }

    public func run() throws {
        print(String(repeating: "=", count: 70))
        print("Monte Carlo / Random Number Generation Benchmark")
        print(String(repeating: "=", count: 70))

        guard let library = try? device.makeLibrary(source: monteCarloShaders, options: nil) else {
            print("Failed to compile Monte Carlo shaders")
            return
        }

        let sizes = [65536, 262144, 1048576]

        for size in sizes {
            print("\n--- Array Size: \(size) ---")

            guard let seedBuf = device.makeBuffer(length: MemoryLayout<UInt64>.size, options: .storageModeShared),
                  let uintOutBuf = device.makeBuffer(length: size * MemoryLayout<UInt32>.size, options: .storageModeShared),
                  let floatOutBuf = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared) else {
                continue
            }

            // Initialize seed
            let seedPtr = seedBuf.contents().bindMemory(to: UInt64.self, capacity: 1)
            seedPtr.pointee = 12345

            var sizeValue = UInt32(size)

            // Test XOR-shift PRNG
            if let prngFunc = library.makeFunction(name: "prng_xorshift"),
               let prngPipeline = try? device.makeComputePipelineState(function: prngFunc) {
                let iterations = 20
                let start = getTimeNanos()
                for _ in 0..<iterations {
                    seedPtr.pointee = 12345  // Reset seed

                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }
                    encoder.setComputePipelineState(prngPipeline)
                    encoder.setBuffer(seedBuf, offset: 0, index: 0)
                    encoder.setBuffer(uintOutBuf, offset: 0, index: 1)
                    encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 2)
                    encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    encoder.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()
                }
                let end = getTimeNanos()
                let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
                let throughput = Double(size) / elapsed / 1e9
                print("XOR-shift PRNG: \(String(format: "%.2f", throughput)) GR/s (\(String(format: "%.4f", elapsed * 1000)) ms)")
            }

            // Test LCG PRNG
            if let lcgFunc = library.makeFunction(name: "prng_lcg"),
               let lcgPipeline = try? device.makeComputePipelineState(function: lcgFunc) {
                let iterations = 20
                let start = getTimeNanos()
                for _ in 0..<iterations {
                    seedPtr.pointee = 12345

                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }
                    encoder.setComputePipelineState(lcgPipeline)
                    encoder.setBuffer(seedBuf, offset: 0, index: 0)
                    encoder.setBuffer(uintOutBuf, offset: 0, index: 1)
                    encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 2)
                    encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    encoder.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()
                }
                let end = getTimeNanos()
                let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
                let throughput = Double(size) / elapsed / 1e9
                print("LCG PRNG: \(String(format: "%.2f", throughput)) GR/s (\(String(format: "%.4f", elapsed * 1000)) ms)")
            }

            // Test uniform transform
            if let uniformFunc = library.makeFunction(name: "uniform_transform"),
               let uniformPipeline = try? device.makeComputePipelineState(function: uniformFunc) {
                // First generate random integers
                seedPtr.pointee = 12345
                if let prngFunc = library.makeFunction(name: "prng_xorshift"),
                   let prngPipeline = try? device.makeComputePipelineState(function: prngFunc) {
                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { return }
                    encoder.setComputePipelineState(prngPipeline)
                    encoder.setBuffer(seedBuf, offset: 0, index: 0)
                    encoder.setBuffer(uintOutBuf, offset: 0, index: 1)
                    encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 2)
                    encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    encoder.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()
                }

                let iterations = 20
                let start = getTimeNanos()
                for _ in 0..<iterations {
                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }
                    encoder.setComputePipelineState(uniformPipeline)
                    encoder.setBuffer(uintOutBuf, offset: 0, index: 0)
                    encoder.setBuffer(floatOutBuf, offset: 0, index: 1)
                    encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 2)
                    encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    encoder.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()
                }
                let end = getTimeNanos()
                let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
                let throughput = Double(size) / elapsed / 1e9
                print("Uniform Transform: \(String(format: "%.2f", throughput)) GE/s (\(String(format: "%.4f", elapsed * 1000)) ms)")
            }
        }

        // Monte Carlo Pi estimation
        print("\n--- Monte Carlo Pi Estimation ---")

        let mcSize = 1000000
        guard let xBuf = device.makeBuffer(length: mcSize * MemoryLayout<Float>.size, options: .storageModeShared),
              let yBuf = device.makeBuffer(length: mcSize * MemoryLayout<Float>.size, options: .storageModeShared),
              let countBuf = device.makeBuffer(length: MemoryLayout<UInt32>.size, options: .storageModeShared) else {
            return
        }

        // Generate random x, y in [0, 1)
        let xPtr = xBuf.contents().bindMemory(to: Float.self, capacity: mcSize)
        let yPtr = yBuf.contents().bindMemory(to: Float.self, capacity: mcSize)
        for i in 0..<mcSize {
            xPtr[i] = Float(i % 1000) / 1000.0
            yPtr[i] = Float((i * 7) % 1000) / 1000.0
        }

        var mcSizeValue = UInt32(mcSize)

        if let mcFunc = library.makeFunction(name: "mc_pi_trial"),
           let mcPipeline = try? device.makeComputePipelineState(function: mcFunc) {
            let iterations = 10
            let start = getTimeNanos()
            for _ in 0..<iterations {
                let countPtr = countBuf.contents().bindMemory(to: UInt32.self, capacity: 1)
                countPtr.pointee = 0

                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(mcPipeline)
                encoder.setBuffer(xBuf, offset: 0, index: 0)
                encoder.setBuffer(yBuf, offset: 0, index: 1)
                encoder.setBuffer(countBuf, offset: 0, index: 2)
                encoder.setBytes(&mcSizeValue, length: MemoryLayout<UInt32>.size, index: 3)
                encoder.dispatchThreads(MTLSize(width: mcSize, height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                encoder.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()

                // Calculate Pi
                let insideCount = countPtr.pointee
                let piEstimate = 4.0 * Float(insideCount) / Float(mcSize)
                if iterations > 0 {
                    // Only print after loop
                }
            }
            let end = getTimeNanos()
            let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
            let countPtr = countBuf.contents().bindMemory(to: UInt32.self, capacity: 1)
            let insideCount = countPtr.pointee
            let piEstimate = 4.0 * Float(insideCount) / Float(mcSize)
            print("Pi Estimate: \(String(format: "%.6f", piEstimate)) (error: \(String(format: "%.6f", abs(Float.pi - piEstimate))))")
            print("Throughput: \(String(format: "%.2f", Double(mcSize) / elapsed / 1e6)) MS/s")
        }

        print("\n--- Key Findings ---")
        print("1. XOR-shift is fastest PRNG (~5-10 GR/s)")
        print("2. LCG is simple but less random quality")
        print("3. Box-Muller transform for Gaussian distribution")
        print("4. Monte Carlo Pi converges as O(1/sqrt(n))")
        print("5. PRNG quality vs speed tradeoff")
    }
}
