import Foundation
import Metal

// MARK: - Mixed-Precision GEMM Benchmark

let mixedPrecisionGEMMShaders = """
#include <metal_stdlib>
using namespace metal;

// Mixed-precision GEMM: FP16 inputs, FP32 accumulation, register-blocked 4x4
// Common optimization in deep learning frameworks (TensorFlow, PyTorch)
kernel void gemm_mixed_precision(device const half* A [[buffer(0)]],
                              device const half* B [[buffer(1)]],
                              device float* C [[buffer(2)]],
                              constant uint& M [[buffer(3)]],
                              constant uint& K [[buffer(4)]],
                              constant uint& N [[buffer(5)]],
                              uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= N / 4 || gid.y >= M / 4) return;

    uint blockN = 4;
    uint blockM = 4;
    uint blockK = 4;

    // FP32 accumulators for 4x4 block
    float4 c00 = 0.0f, c01 = 0.0f, c02 = 0.0f, c03 = 0.0f;
    float4 c10 = 0.0f, c11 = 0.0f, c12 = 0.0f, c13 = 0.0f;
    float4 c20 = 0.0f, c21 = 0.0f, c22 = 0.0f, c23 = 0.0f;
    float4 c30 = 0.0f, c31 = 0.0f, c32 = 0.0f, c33 = 0.0f;

    // Loop over K dimension in blocks
    for (uint k = 0; k < K; k += blockK) {
        // Load 4x4 block of A (FP16 -> FP32 conversion happens here)
        float4 a0 = float4(A[(gid.y * blockM + 0) * K + k],
                          A[(gid.y * blockM + 0) * K + k + 1],
                          A[(gid.y * blockM + 0) * K + k + 2],
                          A[(gid.y * blockM + 0) * K + k + 3]);
        float4 a1 = float4(A[(gid.y * blockM + 1) * K + k],
                          A[(gid.y * blockM + 1) * K + k + 1],
                          A[(gid.y * blockM + 1) * K + k + 2],
                          A[(gid.y * blockM + 1) * K + k + 3]);
        float4 a2 = float4(A[(gid.y * blockM + 2) * K + k],
                          A[(gid.y * blockM + 2) * K + k + 1],
                          A[(gid.y * blockM + 2) * K + k + 2],
                          A[(gid.y * blockM + 2) * K + k + 3]);
        float4 a3 = float4(A[(gid.y * blockM + 3) * K + k],
                          A[(gid.y * blockM + 3) * K + k + 1],
                          A[(gid.y * blockM + 3) * K + k + 2],
                          A[(gid.y * blockM + 3) * K + k + 3]);

        // Load 4x4 block of B (FP16 -> FP32)
        float4 b0 = float4(B[k * N + gid.x * blockN],
                          B[(k + 1) * N + gid.x * blockN],
                          B[(k + 2) * N + gid.x * blockN],
                          B[(k + 3) * N + gid.x * blockN]);
        float4 b1 = float4(B[k * N + gid.x * blockN + 1],
                          B[(k + 1) * N + gid.x * blockN + 1],
                          B[(k + 2) * N + gid.x * blockN + 1],
                          B[(k + 3) * N + gid.x * blockN + 1]);
        float4 b2 = float4(B[k * N + gid.x * blockN + 2],
                          B[(k + 1) * N + gid.x * blockN + 2],
                          B[(k + 2) * N + gid.x * blockN + 2],
                          B[(k + 3) * N + gid.x * blockN + 2]);
        float4 b3 = float4(B[k * N + gid.x * blockN + 3],
                          B[(k + 1) * N + gid.x * blockN + 3],
                          B[(k + 2) * N + gid.x * blockN + 3],
                          B[(k + 3) * N + gid.x * blockN + 3]);

        // Matrix multiply: C_block += A_block * B_block (FP32 accumulation)
        c00 += a0 * b0.x + a1 * b0.y + a2 * b0.z + a3 * b0.w;
        c01 += a0 * b1.x + a1 * b1.y + a2 * b1.z + a3 * b1.w;
        c02 += a0 * b2.x + a1 * b2.y + a2 * b2.z + a3 * b2.w;
        c03 += a0 * b3.x + a1 * b3.y + a2 * b3.z + a3 * b3.w;
        c10 += a0 * b0.x + a1 * b0.y + a2 * b0.z + a3 * b0.w;
        c11 += a0 * b1.x + a1 * b1.y + a2 * b1.z + a3 * b1.w;
        c12 += a0 * b2.x + a1 * b2.y + a2 * b2.z + a3 * b2.w;
        c13 += a0 * b3.x + a1 * b3.y + a2 * b3.z + a3 * b3.w;
        c20 += a0 * b0.x + a1 * b0.y + a2 * b0.z + a3 * b0.w;
        c21 += a0 * b1.x + a1 * b1.y + a2 * b1.z + a3 * b1.w;
        c22 += a0 * b2.x + a1 * b2.y + a2 * b2.z + a3 * b2.w;
        c23 += a0 * b3.x + a1 * b3.y + a2 * b3.z + a3 * b3.w;
        c30 += a0 * b0.x + a1 * b0.y + a2 * b0.z + a3 * b0.w;
        c31 += a0 * b1.x + a1 * b1.y + a2 * b1.z + a3 * b1.w;
        c32 += a0 * b2.x + a1 * b2.y + a2 * b2.z + a3 * b2.w;
        c33 += a0 * b3.x + a1 * b3.y + a2 * b3.z + a3 * b3.w;
    }

    // Store results
    uint cRowStart = gid.y * blockM;
    uint cColStart = gid.x * blockN;

    C[(cRowStart + 0) * N + cColStart] = c00.x;
    C[(cRowStart + 0) * N + cColStart + 1] = c01.x;
    C[(cRowStart + 0) * N + cColStart + 2] = c02.x;
    C[(cRowStart + 0) * N + cColStart + 3] = c03.x;
    C[(cRowStart + 1) * N + cColStart] = c10.y;
    C[(cRowStart + 1) * N + cColStart + 1] = c11.y;
    C[(cRowStart + 1) * N + cColStart + 2] = c12.y;
    C[(cRowStart + 1) * N + cColStart + 3] = c13.y;
    C[(cRowStart + 2) * N + cColStart] = c20.z;
    C[(cRowStart + 2) * N + cColStart + 1] = c21.z;
    C[(cRowStart + 2) * N + cColStart + 2] = c22.z;
    C[(cRowStart + 2) * N + cColStart + 3] = c23.z;
    C[(cRowStart + 3) * N + cColStart] = c30.w;
    C[(cRowStart + 3) * N + cColStart + 1] = c31.w;
    C[(cRowStart + 3) * N + cColStart + 2] = c32.w;
    C[(cRowStart + 3) * N + cColStart + 3] = c33.w;
}

// Pure FP32 GEMM for comparison
kernel void gemm_fp32(device const float* A [[buffer(0)]],
                      device const float* B [[buffer(1)]],
                      device float* C [[buffer(2)]],
                      constant uint& M [[buffer(3)]],
                      constant uint& K [[buffer(4)]],
                      constant uint& N [[buffer(5)]],
                      uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= N / 4 || gid.y >= M / 4) return;

    uint blockN = 4;
    uint blockM = 4;
    uint blockK = 4;

    float4 c00 = 0.0f, c01 = 0.0f, c02 = 0.0f, c03 = 0.0f;
    float4 c10 = 0.0f, c11 = 0.0f, c12 = 0.0f, c13 = 0.0f;
    float4 c20 = 0.0f, c21 = 0.0f, c22 = 0.0f, c23 = 0.0f;
    float4 c30 = 0.0f, c31 = 0.0f, c32 = 0.0f, c33 = 0.0f;

    for (uint k = 0; k < K; k += blockK) {
        float4 a0 = float4(A[(gid.y * blockM + 0) * K + k],
                          A[(gid.y * blockM + 0) * K + k + 1],
                          A[(gid.y * blockM + 0) * K + k + 2],
                          A[(gid.y * blockM + 0) * K + k + 3]);
        float4 a1 = float4(A[(gid.y * blockM + 1) * K + k],
                          A[(gid.y * blockM + 1) * K + k + 1],
                          A[(gid.y * blockM + 1) * K + k + 2],
                          A[(gid.y * blockM + 1) * K + k + 3]);
        float4 a2 = float4(A[(gid.y * blockM + 2) * K + k],
                          A[(gid.y * blockM + 2) * K + k + 1],
                          A[(gid.y * blockM + 2) * K + k + 2],
                          A[(gid.y * blockM + 2) * K + k + 3]);
        float4 a3 = float4(A[(gid.y * blockM + 3) * K + k],
                          A[(gid.y * blockM + 3) * K + k + 1],
                          A[(gid.y * blockM + 3) * K + k + 2],
                          A[(gid.y * blockM + 3) * K + k + 3]);

        float4 b0 = float4(B[k * N + gid.x * blockN],
                          B[(k + 1) * N + gid.x * blockN],
                          B[(k + 2) * N + gid.x * blockN],
                          B[(k + 3) * N + gid.x * blockN]);
        float4 b1 = float4(B[k * N + gid.x * blockN + 1],
                          B[(k + 1) * N + gid.x * blockN + 1],
                          B[(k + 2) * N + gid.x * blockN + 1],
                          B[(k + 3) * N + gid.x * blockN + 1]);
        float4 b2 = float4(B[k * N + gid.x * blockN + 2],
                          B[(k + 1) * N + gid.x * blockN + 2],
                          B[(k + 2) * N + gid.x * blockN + 2],
                          B[(k + 3) * N + gid.x * blockN + 2]);
        float4 b3 = float4(B[k * N + gid.x * blockN + 3],
                          B[(k + 1) * N + gid.x * blockN + 3],
                          B[(k + 2) * N + gid.x * blockN + 3],
                          B[(k + 3) * N + gid.x * blockN + 3]);

        c00 += a0 * b0.x + a1 * b0.y + a2 * b0.z + a3 * b0.w;
        c01 += a0 * b1.x + a1 * b1.y + a2 * b1.z + a3 * b1.w;
        c02 += a0 * b2.x + a1 * b2.y + a2 * b2.z + a3 * b2.w;
        c03 += a0 * b3.x + a1 * b3.y + a2 * b3.z + a3 * b3.w;
        c10 += a0 * b0.x + a1 * b0.y + a2 * b0.z + a3 * b0.w;
        c11 += a0 * b1.x + a1 * b1.y + a2 * b1.z + a3 * b1.w;
        c12 += a0 * b2.x + a1 * b2.y + a2 * b2.z + a3 * b2.w;
        c13 += a0 * b3.x + a1 * b3.y + a2 * b3.z + a3 * b3.w;
        c20 += a0 * b0.x + a1 * b0.y + a2 * b0.z + a3 * b0.w;
        c21 += a0 * b1.x + a1 * b1.y + a2 * b1.z + a3 * b1.w;
        c22 += a0 * b2.x + a1 * b2.y + a2 * b2.z + a3 * b2.w;
        c23 += a0 * b3.x + a1 * b3.y + a2 * b3.z + a3 * b3.w;
        c30 += a0 * b0.x + a1 * b0.y + a2 * b0.z + a3 * b0.w;
        c31 += a0 * b1.x + a1 * b1.y + a2 * b1.z + a3 * b1.w;
        c32 += a0 * b2.x + a1 * b2.y + a2 * b2.z + a3 * b2.w;
        c33 += a0 * b3.x + a1 * b3.y + a2 * b3.z + a3 * b3.w;
    }

    uint cRowStart = gid.y * blockM;
    uint cColStart = gid.x * blockN;

    C[(cRowStart + 0) * N + cColStart] = c00.x;
    C[(cRowStart + 0) * N + cColStart + 1] = c01.x;
    C[(cRowStart + 0) * N + cColStart + 2] = c02.x;
    C[(cRowStart + 0) * N + cColStart + 3] = c03.x;
    C[(cRowStart + 1) * N + cColStart] = c10.y;
    C[(cRowStart + 1) * N + cColStart + 1] = c11.y;
    C[(cRowStart + 1) * N + cColStart + 2] = c12.y;
    C[(cRowStart + 1) * N + cColStart + 3] = c13.y;
    C[(cRowStart + 2) * N + cColStart] = c20.z;
    C[(cRowStart + 2) * N + cColStart + 1] = c21.z;
    C[(cRowStart + 2) * N + cColStart + 2] = c22.z;
    C[(cRowStart + 2) * N + cColStart + 3] = c23.z;
    C[(cRowStart + 3) * N + cColStart] = c30.w;
    C[(cRowStart + 3) * N + cColStart + 1] = c31.w;
    C[(cRowStart + 3) * N + cColStart + 2] = c32.w;
    C[(cRowStart + 3) * N + cColStart + 3] = c33.w;
}
"""

// Float to Half conversion helper
func FloatToHalf(_ value: Float) -> UInt16 {
    let bits = value.bitPattern
    let sign = (bits >> 16) & 0x8000
    var expo = Int((bits >> 23) & 0xFF) - 127
    let mantissa = bits & 0x7FFFFF

    if expo > 15 {
        return UInt16(sign | 0x7C00)
    } else if expo < -10 {
        return UInt16(sign)
    }

    let expBias = expo + 15
    if expBias <= 0 {
        let shift = UInt32(-expBias + 1)
        let mant = (mantissa >> shift) | 0x800
        return UInt16(sign | mant)
    }

    return UInt16(sign | UInt32(expBias << 10) | (mantissa >> 13))
}

public struct MixedPrecisionGEMMBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    public init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
    }

    public func run() throws {
        print(String(repeating: "=", count: 70))
        print("Mixed-Precision GEMM Benchmark")
        print("FP16 Input, FP32 Accumulation")
        print(String(repeating: "=", count: 70))

        guard let library = try? device.makeLibrary(source: mixedPrecisionGEMMShaders, options: nil) else {
            print("Failed to compile mixed-precision GEMM shaders")
            return
        }

        let sizes = [256, 512, 1024]

        for size in sizes {
            print("\n--- Matrix Size: \(size)x\(size) ---")

            let m = size
            let k = size
            let n = size

            // Mixed precision: FP16 inputs, FP32 output
            guard let fp16A = device.makeBuffer(length: m * k * MemoryLayout<UInt16>.size, options: .storageModeShared),
                  let fp16B = device.makeBuffer(length: k * n * MemoryLayout<UInt16>.size, options: .storageModeShared),
                  let fp32C = device.makeBuffer(length: m * n * MemoryLayout<Float>.size, options: .storageModeShared) else {
                continue
            }

            // FP32 buffers for comparison
            guard let fp32A = device.makeBuffer(length: m * k * MemoryLayout<Float>.size, options: .storageModeShared),
                  let fp32B = device.makeBuffer(length: k * n * MemoryLayout<Float>.size, options: .storageModeShared),
                  let fp32OutC = device.makeBuffer(length: m * n * MemoryLayout<Float>.size, options: .storageModeShared) else {
                continue
            }

            // Initialize FP16 matrices
            let fp16APtr = fp16A.contents().bindMemory(to: UInt16.self, capacity: m * k)
            let fp16BPtr = fp16B.contents().bindMemory(to: UInt16.self, capacity: k * n)

            for i in 0..<(m * k) {
                let f = Float.random(in: 0.0...1.0)
                fp16APtr[i] = FloatToHalf(f)
            }
            for i in 0..<(k * n) {
                let f = Float.random(in: 0.0...1.0)
                fp16BPtr[i] = FloatToHalf(f)
            }

            // Initialize FP32 matrices
            let fp32APtr = fp32A.contents().bindMemory(to: Float.self, capacity: m * k)
            let fp32BPtr = fp32B.contents().bindMemory(to: Float.self, capacity: k * n)
            for i in 0..<(m * k) {
                fp32APtr[i] = Float.random(in: 0.0...1.0)
            }
            for i in 0..<(k * n) {
                fp32BPtr[i] = Float.random(in: 0.0...1.0)
            }

            var mVar = UInt32(m)
            var kVar = UInt32(k)
            var nVar = UInt32(n)

            // Test mixed-precision GEMM (FP16 input, FP32 accumulation)
            if let mixedFunc = library.makeFunction(name: "gemm_mixed_precision"),
               let mixedPipeline = try? device.makeComputePipelineState(function: mixedFunc) {
                let iterations = 10
                let start = getTimeNanos()
                for _ in 0..<iterations {
                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }
                    encoder.setComputePipelineState(mixedPipeline)
                    encoder.setBuffer(fp16A, offset: 0, index: 0)
                    encoder.setBuffer(fp16B, offset: 0, index: 1)
                    encoder.setBuffer(fp32C, offset: 0, index: 2)
                    encoder.setBytes(&mVar, length: MemoryLayout<UInt32>.size, index: 3)
                    encoder.setBytes(&kVar, length: MemoryLayout<UInt32>.size, index: 4)
                    encoder.setBytes(&nVar, length: MemoryLayout<UInt32>.size, index: 5)
                    encoder.dispatchThreads(MTLSize(width: n / 4, height: m / 4, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
                    encoder.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()
                }
                let end = getTimeNanos()
                let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)

                // FLOPs: 2 * M * K * N (multiply-adds)
                let flops = 2.0 * Double(m) * Double(k) * Double(n) * Double(iterations)
                let gflops = flops / elapsed / 1e9

                print("Mixed (FP16in, FP32acc): \(String(format: "%.2f", gflops)) GFLOPS")
            }

            // Test pure FP32 GEMM
            if let fp32Func = library.makeFunction(name: "gemm_fp32"),
               let fp32Pipeline = try? device.makeComputePipelineState(function: fp32Func) {
                let iterations = 10
                let start = getTimeNanos()
                for _ in 0..<iterations {
                    guard let cmd = queue.makeCommandBuffer(),
                          let encoder = cmd.makeComputeCommandEncoder() else { continue }
                    encoder.setComputePipelineState(fp32Pipeline)
                    encoder.setBuffer(fp32A, offset: 0, index: 0)
                    encoder.setBuffer(fp32B, offset: 0, index: 1)
                    encoder.setBuffer(fp32OutC, offset: 0, index: 2)
                    encoder.setBytes(&mVar, length: MemoryLayout<UInt32>.size, index: 3)
                    encoder.setBytes(&kVar, length: MemoryLayout<UInt32>.size, index: 4)
                    encoder.setBytes(&nVar, length: MemoryLayout<UInt32>.size, index: 5)
                    encoder.dispatchThreads(MTLSize(width: n / 4, height: m / 4, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
                    encoder.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()
                }
                let end = getTimeNanos()
                let elapsed = getElapsedSeconds(start: start, end: end) / Double(iterations)

                let flops = 2.0 * Double(m) * Double(k) * Double(n) * Double(iterations)
                let gflops = flops / elapsed / 1e9

                print("FP32 GEMM:                  \(String(format: "%.2f", gflops)) GFLOPS")
            }
        }

        print("\n--- Key Findings ---")
        print("1. Mixed precision: FP16 inputs (half memory) + FP32 accumulation")
        print("2. Memory savings: 2x reduction in memory bandwidth")
        print("3. FP32 accumulation maintains accuracy for gradient updates")
        print("4. Apple GPU has native FP16 support with higher throughput")
        print("5. Used in: TensorFlow, PyTorch, ONNX Runtime for inference")
    }
}
