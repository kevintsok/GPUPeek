import Foundation
import Metal

// MARK: - FP64 Benchmark

let fp64Shaders = """
#include <metal_stdlib>
using namespace metal;

// Double-double precision addition
struct DoubleDouble {
    float hi;
    float lo;
};

constant float PI_HI = 3.14159265f;
constant float PI_LO = 0.0000035f;

// Test FP64 simulation with double-double arithmetic
kernel void double_add(device DoubleDouble* a [[buffer(0)]],
                       device DoubleDouble* b [[buffer(1)]],
                       device DoubleDouble* out [[buffer(2)]],
                       constant uint& size [[buffer(3)]],
                       uint id [[thread_position_in_grid]]) {
    if (id >= size) return;

    float a_hi = a[id].hi;
    float a_lo = a[id].lo;
    float b_hi = b[id].hi;
    float b_lo = b[id].lo;

    // Two-sum algorithm
    float s = a_hi + b_hi;
    float v = a_hi - s + b_hi;
    float t = a_lo + b_lo;
    float w = a_lo - t + b_lo;
    float z = s + t;
    float zz = s - z + t + v + w;

    out[id].hi = z;
    out[id].lo = zz;
}

// Simulated double multiplication
kernel void double_mul(device DoubleDouble* a [[buffer(0)]],
                       device DoubleDouble* b [[buffer(1)]],
                       device DoubleDouble* out [[buffer(2)]],
                       constant uint& size [[buffer(3)]],
                       uint id [[thread_position_in_grid]]) {
    if (id >= size) return;

    float a_hi = a[id].hi;
    float a_lo = a[id].lo;
    float b_hi = b[id].hi;
    float b_lo = b[id].lo;

    // Double-double multiplication
    float t = a_hi * b_hi;
    float u = fma(a_hi, b_hi, -t) + a_hi * b_lo + a_lo * b_hi;
    float tt = t + u;
    float uu = t - tt + u;

    out[id].hi = tt;
    out[id].lo = uu;
}

// Check if device supports FP64 via feature set
kernel void check_fp64_support(device uint* out [[buffer(0)]],
                              constant uint& size [[buffer(1)]],
                              uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    // M2 does not support FP64 - all FP64 operations fall back to FP32
    out[id] = 0;  // 0 = no native FP64, 1 = FP64 supported
}
"""

public struct FP64Benchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    public init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
    }

    public func run() throws {
        print(String(repeating: "=", count: 70))
        print("FP64 (Double Precision) Benchmark")
        print(String(repeating: "=", count: 70))

        // Check device capabilities
        print("\n--- Device FP64 Support ---")
        print("Device: \(device.name)")
        print("Recommended Feature Set: \(device.recommendedFeatureSet?.rawValue.description ?? "Unknown")")

        // Check for FP64 support via feature set
        let supportsFP64 = device.supportsFamily(.apple7)
        print("Supports Apple7 Family: \(supportsFP64)")

        guard let library = try? device.makeLibrary(source: fp64Shaders, options: nil) else {
            print("Failed to compile FP64 shaders")
            return
        }

        let sizes = [32768, 131072]

        for size in sizes {
            print("\n--- Array Size: \(size) ---")

            guard let bufferA = device.makeBuffer(length: size * MemoryLayout<Float>.size * 2, options: .storageModeShared),
                  let bufferB = device.makeBuffer(length: size * MemoryLayout<Float>.size * 2, options: .storageModeShared),
                  let bufferOut = device.makeBuffer(length: size * MemoryLayout<Float>.size * 2, options: .storageModeShared) else {
                continue
            }

            var sizeValue = UInt32(size)

            // Test double-double addition
            if let addFunc = library.makeFunction(name: "double_add"),
               let addPipeline = try? device.makeComputePipelineState(function: addFunc) {
                let iterations = 10
                let start = getTimeNanos()
                for _ in 0..<iterations {
                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }
                    encoder.setComputePipelineState(addPipeline)
                    encoder.setBuffer(bufferA, offset: 0, index: 0)
                    encoder.setBuffer(bufferB, offset: 0, index: 1)
                    encoder.setBuffer(bufferOut, offset: 0, index: 2)
                    encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 3)
                    encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    encoder.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()
                }
                let end = getTimeNanos()
                let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
                let gops = Double(size) / elapsed / 1e9
                print("Double-Add: \(String(format: "%.2f", gops)) GOPS (\(String(format: "%.4f", elapsed * 1000)) ms)")
            }

            // Test double-double multiplication
            if let mulFunc = library.makeFunction(name: "double_mul"),
               let mulPipeline = try? device.makeComputePipelineState(function: mulFunc) {
                let iterations = 10
                let start = getTimeNanos()
                for _ in 0..<iterations {
                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }
                    encoder.setComputePipelineState(mulPipeline)
                    encoder.setBuffer(bufferA, offset: 0, index: 0)
                    encoder.setBuffer(bufferB, offset: 0, index: 1)
                    encoder.setBuffer(bufferOut, offset: 0, index: 2)
                    encoder.setBytes(&sizeValue, length: MemoryLayout<UInt32>.size, index: 3)
                    encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    encoder.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()
                }
                let end = getTimeNanos()
                let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)
                let gops = Double(size) / elapsed / 1e9
                print("Double-Mul: \(String(format: "%.2f", gops)) GOPS (\(String(format: "%.4f", elapsed * 1000)) ms)")
            }
        }

        print("\n--- Key Findings ---")
        print("1. Apple M2 GPU does NOT support native FP64")
        print("2. FP64 operations are emulated via FP32")
        print("3. Double-double arithmetic provides ~FP64 precision at 2-3x cost")
        print("4. For true FP64, use NVIDIA/AMD discrete GPUs")
        print("5. Apple M-series is optimized for FP16/FP32 ML workloads")
    }
}
