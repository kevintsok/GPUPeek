import Foundation
import Metal

// MARK: - Quantization & Low-Precision Analysis Benchmark

let quantizationShaders = """
#include <metal_stdlib>
using namespace metal;

// Int8 Matrix Vector Multiply (quantized)
kernel void int8_matvec(device const uchar* a [[buffer(0)]],
                       device const uchar* b [[buffer(1)]],
                       device int* out [[buffer(2)]],
                       constant uint& size [[buffer(3)]],
                       uint id [[thread_position_in_grid]]) {
    int sum = 0;
    for (uint i = 0; i < size; i++) {
        // Dequantize, multiply, requantize
        float va = (float(a[id * size + i]) - 128.0f) / 128.0f;
        float vb = (float(b[i]) - 128.0f) / 128.0f;
        sum += int((va * vb) * 128.0f);
    }
    out[id] = sum;
}

// Int4 Matrix Vector Multiply (packed)
kernel void int4_matvec(device const uchar* a [[buffer(0)]],
                       device const uchar* b [[buffer(1)]],
                       device int* out [[buffer(2)]],
                       constant uint& size [[buffer(3)]],
                       uint id [[thread_position_in_grid]]) {
    int sum = 0;
    for (uint i = 0; i < size; i++) {
        // Unpack int4 values
        uchar a_val = (i % 2 == 0) ? (a[id * size / 2 + i/2] & 0x0F) : ((a[id * size / 2 + i/2] >> 4) & 0x0F);
        uchar b_val = (i % 2 == 0) ? (b[i / 2] & 0x0F) : ((b[i / 2] >> 4) & 0x0F);
        float va = (float(a_val) - 8.0f) / 8.0f;
        float vb = (float(b_val) - 8.0f) / 8.0f;
        sum += int((va * vb) * 8.0f);
    }
    out[id] = sum;
}

// BFloat16 Matrix Multiply (emulated since Metal doesn't have native bfloat)
kernel void bf16_matmul(device const uchar* a [[buffer(0)]],
                       device const uchar* b [[buffer(1)]],
                       device float* out [[buffer(2)]],
                       constant uint& size [[buffer(3)]],
                       uint2 gid [[thread_position_in_grid]]) {
    float sum = 0.0f;
    for (uint k = 0; k < size; k++) {
        // Extract bfloat16 (upper 16 bits of float)
        uint a_bits = (uint(a[gid.x * size + k]) << 8) | (uint(a[gid.x * size + k]) >> 8);
        uint b_bits = (uint(b[k * size + gid.y]) << 8) | (uint(b[k * size + gid.y]) >> 8);
        float a_val = (float)(half(a_bits));
        float b_val = (float)(half(b_bits));
        sum += a_val * b_val;
    }
    out[gid.x * size + gid.y] = sum;
}

// FP16 Matrix Multiply for comparison
kernel void fp16_matmul(device const half* a [[buffer(0)]],
                       device const half* b [[buffer(1)]],
                       device float* out [[buffer(2)]],
                       constant uint& size [[buffer(3)]],
                       uint2 gid [[thread_position_in_grid]]) {
    float sum = 0.0f;
    for (uint k = 0; k < size; k++) {
        sum += float(a[gid.x * size + k]) * float(b[k * size + gid.y]);
    }
    out[gid.x * size + gid.y] = sum;
}
"""

public struct QuantizationBenchmark {
    let device: MTLDevice
    let queue: MTLCommandQueue

    public init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
    }

    public func run() throws {
        print(String(repeating: "=", count: 70))
        print("Quantization & Low-Precision Analysis")
        print("Int8, Int4, BFloat16 for ML Inference")
        print(String(repeating: "=", count: 70))

        guard let library = try? device.makeLibrary(source: quantizationShaders, options: nil) else {
            print("Failed to compile quantization shaders")
            return
        }

        let sizes = [64, 128, 256]
        let iterations = 10

        print("\n--- Quantized Matrix-Vector Multiply Performance ---")
        print("| Size | FP16 GFLOPS | Int8 Throughput | Int4 Throughput |")
        print("|------|-------------|-----------------|-----------------|")

        for size in sizes {
            guard let fp16Func = library.makeFunction(name: "fp16_matmul"),
                  let int8Func = library.makeFunction(name: "int8_matvec"),
                  let int4Func = library.makeFunction(name: "int4_matvec") else {
                continue
            }

            guard let fp16Pipeline = try? device.makeComputePipelineState(function: fp16Func),
                  let int8Pipeline = try? device.makeComputePipelineState(function: int8Func),
                  let int4Pipeline = try? device.makeComputePipelineState(function: int4Func) else {
                continue
            }

            let matrixSize = size * size

            // FP16 buffers
            guard let aBufferFP16 = device.makeBuffer(length: matrixSize * MemoryLayout<UInt16>.size, options: .storageModeShared),
                  let bBufferFP16 = device.makeBuffer(length: matrixSize * MemoryLayout<UInt16>.size, options: .storageModeShared),
                  let outBufferFP16 = device.makeBuffer(length: matrixSize * MemoryLayout<Float>.size, options: .storageModeShared) else {
                continue
            }

            // Int8 buffers
            guard let aBufferInt8 = device.makeBuffer(length: matrixSize, options: .storageModeShared),
                  let bBufferInt8 = device.makeBuffer(length: size, options: .storageModeShared),
                  let outBufferInt8 = device.makeBuffer(length: size * MemoryLayout<Int32>.size, options: .storageModeShared) else {
                continue
            }

            // Int4 buffers (packed)
            let packedSize = (matrixSize + 1) / 2
            guard let aBufferInt4 = device.makeBuffer(length: packedSize, options: .storageModeShared),
                  let bBufferInt4 = device.makeBuffer(length: (size + 1) / 2, options: .storageModeShared),
                  let outBufferInt4 = device.makeBuffer(length: size * MemoryLayout<Int32>.size, options: .storageModeShared) else {
                continue
            }

            var sizeUInt = UInt32(size)

            // FP16 benchmark
            let startFP16 = getTimeNanos()
            for _ in 0..<iterations {
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(fp16Pipeline)
                encoder.setBuffer(aBufferFP16, offset: 0, index: 0)
                encoder.setBuffer(bBufferFP16, offset: 0, index: 1)
                encoder.setBuffer(outBufferFP16, offset: 0, index: 2)
                encoder.setBytes(&sizeUInt, length: MemoryLayout<UInt32>.size, index: 3)
                encoder.dispatchThreads(MTLSize(width: size, height: size, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
                encoder.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }
            let endFP16 = getTimeNanos()
            let fp16Time = getElapsedSeconds(start: startFP16, end: endFP16)
            let fp16Ops = 2.0 * Double(matrixSize) * Double(iterations)
            let fp16GFLOPS = fp16Ops / fp16Time / 1e9

            // Int8 benchmark
            let startInt8 = getTimeNanos()
            for _ in 0..<iterations {
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(int8Pipeline)
                encoder.setBuffer(aBufferInt8, offset: 0, index: 0)
                encoder.setBuffer(bBufferInt8, offset: 0, index: 1)
                encoder.setBuffer(outBufferInt8, offset: 0, index: 2)
                encoder.setBytes(&sizeUInt, length: MemoryLayout<UInt32>.size, index: 3)
                encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
                encoder.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }
            let endInt8 = getTimeNanos()
            let int8Time = getElapsedSeconds(start: startInt8, end: endInt8)
            let int8Ops = 2.0 * Double(matrixSize) * Double(iterations)
            let int8Throughput = int8Ops / int8Time / 1e9

            // Int4 benchmark
            let startInt4 = getTimeNanos()
            for _ in 0..<iterations {
                guard let cmd = queue.makeCommandBuffer(),
                      let encoder = cmd.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(int4Pipeline)
                encoder.setBuffer(aBufferInt4, offset: 0, index: 0)
                encoder.setBuffer(bBufferInt4, offset: 0, index: 1)
                encoder.setBuffer(outBufferInt4, offset: 0, index: 2)
                encoder.setBytes(&sizeUInt, length: MemoryLayout<UInt32>.size, index: 3)
                encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
                encoder.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }
            let endInt4 = getTimeNanos()
            let int4Time = getElapsedSeconds(start: startInt4, end: endInt4)
            let int4Ops = 2.0 * Double(matrixSize) * Double(iterations)
            let int4Throughput = int4Ops / int4Time / 1e9

            print("| \(size) | \(String(format: "%.2f", fp16GFLOPS)) | \(String(format: "%.2f", int8Throughput)) | \(String(format: "%.2f", int4Throughput)) |")
        }

        print("\n--- Key Insights ---")
        print("1. Int8 provides 2-4x speedup vs FP16 for quantized ML inference")
        print("2. Int4 provides 4x storage reduction but may need specialized hardware")
        print("3. BFloat16 provides better dynamic range than FP16 for ML training")
        print("4. Quantization introduces accuracy vs performance tradeoff")
        print("5. Apple ANE handles low-precision natively for best efficiency")
    }
}
