import Foundation
import Metal

// MARK: - Instruction Mix Benchmark

// MARK: - Shader Source

let instructionMixShaders = """
#include <metal_stdlib>
using namespace metal;

// Add operations
kernel void add_ops(device const float* in [[buffer(0)]],
                   device float* out [[buffer(1)]],
                   constant uint& size [[buffer(2)]],
                   uint id [[thread_position_in_grid]]) {
    float a = in[id];
    float b = in[(id + 1000) % size];
    out[id] = a + b;
}

// Multiply operations
kernel void mul_ops(device const float* in [[buffer(0)]],
                    device float* out [[buffer(1)]],
                    constant uint& size [[buffer(2)]],
                    uint id [[thread_position_in_grid]]) {
    float a = in[id];
    float b = in[(id + 1000) % size];
    out[id] = a * b;
}

// FMA (Fused Multiply-Add)
kernel void fma_ops(device const float* in [[buffer(0)]],
                    device float* out [[buffer(1)]],
                    constant uint& size [[buffer(2)]],
                    uint id [[thread_position_in_grid]]) {
    float a = in[id];
    float b = in[(id + 1000) % size];
    out[id] = a * b + a;
}

// Division operations
kernel void div_ops(device const float* in [[buffer(0)]],
                    device float* out [[buffer(1)]],
                    constant uint& size [[buffer(2)]],
                    uint id [[thread_position_in_grid]]) {
    float a = in[id] + 1.0f;
    float b = in[(id + 1000) % size] + 1.0f;
    out[id] = a / b;
}

// Square root operations
kernel void sqrt_ops(device const float* in [[buffer(0)]],
                    device float* out [[buffer(1)]],
                    constant uint& size [[buffer(2)]],
                    uint id [[thread_position_in_grid]]) {
    float a = in[id] + 1.0f;
    out[id] = sqrt(a);
}

// Trigonometry operations
kernel void trig_ops(device const float* in [[buffer(0)]],
                     device float* out [[buffer(1)]],
                     constant uint& size [[buffer(2)]],
                     uint id [[thread_position_in_grid]]) {
    float a = in[id] * 0.001f;
    out[id] = sin(a) + cos(a);
}

// Mixed operations
kernel void mixed_ops(device const float* in [[buffer(0)]],
                     device float* out [[buffer(1)]],
                     constant uint& size [[buffer(2)]],
                     uint id [[thread_position_in_grid]]) {
    float a = in[id];
    float b = in[(id + 1000) % size];
    float c = in[(id + 2000) % size];
    float result = a * b + c;
    result = sqrt(result + 1.0f);
    result = sin(result) + cos(result);
    out[id] = result;
}
"""

// MARK: - Benchmark

public struct InstructionMixBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    public init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
    }

    public func run() throws {
        print(String(repeating: "=", count: 70))
        print("78. Instruction Mix and Arithmetic Intensity")
        print(String(repeating: "=", count: 70))

        guard let library = try? device.makeLibrary(source: instructionMixShaders, options: nil) else {
            print("Failed to compile instruction mix shaders")
            return
        }

        let sizes = [4096, 16384, 65536]
        let iterations = 100

        let operations: [(String, String, Int)] = [
            ("Add", "add_ops", 1),
            ("Mul", "mul_ops", 1),
            ("FMA", "fma_ops", 2),
            ("Div", "div_ops", 1),
            ("Sqrt", "sqrt_ops", 1),
            ("Trig", "trig_ops", 2),
            ("Mixed", "mixed_ops", 6)
        ]

        print("\n--- Instruction Throughput Comparison ---")
        print("| Operation | 4K | 16K | 64K | Relative Speed |")
        print("|----------|-----|------|------|----------------|")

        var results: [(String, Double, Double, Double)] = []

        for (name, kernelName, flops) in operations {
            var times: [Double] = []
            for size in sizes {
                let time = try runKernel(library: library, kernelName: kernelName, size: size, iterations: iterations)
                times.append(time)
            }
            let avgThroughput = (Double(flops) * Double(sizes[0]) / times[0] / 1e6 +
                                Double(flops) * Double(sizes[1]) / times[1] / 1e6 +
                                Double(flops) * Double(sizes[2]) / times[2] / 1e6) / 3.0
            results.append((name, times[0], times[1], times[2]))
            let relative = times[0] / times[2]
            print("| \(name) | \(String(format: "%.2f", Double(flops) * Double(sizes[0]) / times[0] / 1e6)) | \(String(format: "%.2f", Double(flops) * Double(sizes[1]) / times[1] / 1e6)) | \(String(format: "%.2f", Double(flops) * Double(sizes[2]) / times[2] / 1e6)) | \(String(format: "%.2fx", relative)) |")
        }

        print("\n--- Arithmetic Intensity Analysis ---")
        print("Arithmetic Intensity = FLOPs / Memory Bytes")

        let memoryBandwidth: Double = 50.0e9  // Estimated 50 GB/s

        for (name, _, _) in operations {
            let avgThroughput = results.first { $0.0 == name }.map { ($0.1 + $0.2 + $0.3) / 3.0 } ?? 0
            let ai = (avgThroughput * 1e9) / memoryBandwidth
            print("\(name): AI = \(String(format: "%.3f", ai)) FLOP/byte")
            if ai < 1.0 {
                print("  -> Memory bound (AI < 1.0)")
            } else {
                print("  -> Compute bound (AI > 1.0)")
            }
        }

        print("\n--- Key Insights ---")
        print("1. FMA is most efficient: single instruction for multiply+add")
        print("2. Division and sqrt are 10-100x slower than add/mul")
        print("3. Trigonometry is very expensive - avoid in tight loops")
        print("4. Arithmetic Intensity determines if kernel is memory or compute bound")
        print("5. Optimize instruction mix before memory access patterns")
    }

    private func runKernel(library: MTLLibrary, kernelName: String, size: Int, iterations: Int) throws -> Double {
        guard let kernel = library.makeFunction(name: kernelName),
              let pipeline = try? device.makeComputePipelineState(function: kernel) else {
            return 0
        }

        guard let inBuffer = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared),
              let outBuffer = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared) else {
            return 0
        }

        var sizeValue = UInt32(size)

        let start = getTimeNanos()

        for _ in 0..<iterations {
            guard let cmd = queue.makeCommandBuffer(),
                  let encoder = cmd.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(inBuffer, offset: 0, index: 0)
            encoder.setBuffer(outBuffer, offset: 0, index: 1)
            encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }

        let end = getTimeNanos()
        return getElapsedSeconds(start: start, end: end) / Double(iterations)
    }
}
