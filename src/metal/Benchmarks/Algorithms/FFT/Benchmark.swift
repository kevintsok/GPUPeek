import Foundation
import Metal

// MARK: - FFT Benchmark

let fftShaders = """
#include <metal_stdlib>
using namespace metal;

// Cooley-Tukey FFT butterfly operation
kernel void fft_butterfly(device float2* data [[buffer(0)]],
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

// In-place FFT with bit-reversal
kernel void fft_radix2(device float2* data [[buffer(0)]],
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

    threadgroup_barrier(flags::mem_threadgroup);

    // Butterfly operations for each stage
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

// Complex multiplication for FFT
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

        let sizes = [1024, 4096, 16384]

        for N in sizes {
            print("\n--- FFT Size: \(N) ---")

            guard let bufferData = device.makeBuffer(length: N * 2 * MemoryLayout<Float>.size, options: .storageModeShared) else {
                continue
            }

            // Initialize with complex values
            let dataPtr = bufferData.contents().bindMemory(to: Float.self, capacity: N * 2)
            for i in 0..<N {
                dataPtr[i * 2] = Float(i % 256) / 255.0
                dataPtr[i * 2 + 1] = 0.0f
            }

            var sizeValue = UInt32(N)

            // Test FFT Radix-2
            if let fftFunc = library.makeFunction(name: "fft_radix2"),
               let fftPipeline = try? device.makeComputePipelineState(function: fftFunc) {
                let iterations = 10
                let start = getTimeNanos()
                for _ in 0..<iterations {
                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }
                    encoder.setComputePipelineState(fftPipeline)
                    encoder.setBuffer(bufferData, offset: 0, index: 0)
                    encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 1)
                    encoder.dispatchThreads(MTLSize(width: N / 2, height: 1, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    encoder.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()
                }
                let end = getTimeNanos()
                let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
                let gops = 5.0 * Double(N) * log2(Double(N)) / elapsed / 1e9
                print("FFT Radix-2: \(String(format: "%.4f", gops)) GOPS (\(String(format: "%.2f", elapsed * 1000)) ms)")
            }
        }

        print("\n--- Key Findings ---")
        print("1. FFT on GPU has O(n log n) complexity")
        print("2. Butterfly operations are memory-intensive")
        print("3. GPU FFT benefits large datasets (>16K elements)")
        print("4. Memory bandwidth is often the bottleneck")
    }
}
